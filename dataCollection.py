import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from time import time
import os


# Define the output folder path
outputFolderPath = 'Dataset/DataCollect'

# Check if the directory exists, and if not, create it
if not os.path.exists(outputFolderPath):
    os.makedirs(outputFolderPath)

##################################################
classID = 0  # 0 is fake and 1 is real
outputFolderPath = 'Dataset/DataCollect'
confidence = 0.8
save = True
blurThreshold = 30  # Larger is more focus

debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 720, 720
floatingPoint = 6

##################################################


cap = cv2.VideoCapture(0)
cap.set(3,camWidth)
cap.set(4,camHeight)

detector = FaceDetector()

while True:
     success, img = cap.read()
     imgOut = img.copy()
     img, bboxs = detector.findFaces(img, draw=False)

     listBlur = []  # True False values indicating if the faces are blur or not
     listInfo = []  # The normalized values and the class name for the label txt file

     if bboxs:
         # bboxInfo - "id","bbox","score","center"
         for bbox in bboxs:
             x, y, w, h = bbox["bbox"]
             score = bbox["score"][0]
             # print(x, y, w, h)

             # ------  Check the score --------
             if score > confidence:

                # ------------Adding an offset to the face Detected---------
                offsetW = (offsetPercentageW/100)*w
                x = int(x - offsetW)
                w = int(w + offsetW*2)

                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH*3)
                h = int(h + offsetH * 3.5)

                # ------------To avoid values below 0 ---------
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # ------  Find Blurriness --------
                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue>blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # ------  Normalize Values  --------
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2

                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)
                # print(xcn, ycn, wn, hn)

                # ------  To avoid values above 1 --------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # ------  Drawing --------
                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 0),scale=2, thickness=3)

                # ------  debugging --------
                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 0), scale=2,thickness=3)
     if save:
         if all(listBlur) and listBlur != []:
             timeNow = str(int(time() * 1000))  # Generating a unique filename
             image_path = os.path.join(outputFolderPath, f"{timeNow}.jpg")
             print("Saving image to:", image_path)
             cv2.imwrite(image_path, img)

             #----------save label text file-----------
             label_file_path = os.path.join(outputFolderPath, f"{timeNow}.txt")
             with open(label_file_path, 'w') as label_file:
                 for info in listInfo:
                     label_file.write(info)






     cv2.imshow("Image", imgOut)
     cv2.waitKey(1)


