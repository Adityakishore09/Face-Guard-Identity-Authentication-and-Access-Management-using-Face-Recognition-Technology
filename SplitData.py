import os
import random
import shutil
from itertools import islice

# Use backslashes for the path separator in Windows
outputFolderPath = "Testing_Scripts\\Dataset\\SplitData"
inputFolderPath = "Testing_Scripts\\Dataset\\all"
splitRatio = {"train": 0.6, "val": 0.2, "test": 0.2}
classes = ["Proxy_FAKE_Attendance","Real_ACTUAL_Attendance"]

if os.path.exists(outputFolderPath):
    try:
        shutil.rmtree(outputFolderPath)
        print("Removed Directory")
    except OSError as e:
        print(f"Error while removing directory: {e}")

# Create the necessary subdirectories
os.makedirs(f"{outputFolderPath}\\train\\images", exist_ok=True)
os.makedirs(f"{outputFolderPath}\\train\\labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}\\val\\images", exist_ok=True)
os.makedirs(f"{outputFolderPath}\\val\\labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}\\test\\images", exist_ok=True)
os.makedirs(f"{outputFolderPath}\\test\\labels", exist_ok=True)

# -----------Get the name----------
listNames = os.listdir(inputFolderPath)
# print(listNames)
# print(len(listNames))
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))
# print(len(uniqueNames))





# -----------shuffle----------
random.shuffle(uniqueNames)
# print(uniqueNames)


# -----------Find the number of images----------
lenData = len(uniqueNames)
# print(f"Total Images:{lenData}")
lenTrain = int(lenData*splitRatio['train'])
lenVal = int(lenData*splitRatio['val'])
lenTest = int(lenData*splitRatio['test'])
print(f"Total Images:{lenData} \nSplit: {lenTrain} {lenVal} {lenTest}")

# -----------Put Remaining images in Training----------
if lenData != lenTrain + lenTest + lenVal:
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining



# -----------Split the list----------
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images:{lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')


# -----------Copy the files----------
sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for fileName in out:
        source_image_path = os.path.join(inputFolderPath, f"{fileName}.jpg")
        source_label_path = os.path.join(inputFolderPath, f"{fileName}.txt")
        target_image_path = os.path.join(outputFolderPath, sequence[i], "images", f"{fileName}.jpg")
        target_label_path = os.path.join(outputFolderPath, sequence[i], "labels", f"{fileName}.txt")

        print(f"Copying image from {source_image_path} to {target_image_path}")
        print(f"Copying label from {source_label_path} to {target_label_path}")

        shutil.copy(source_image_path, target_image_path)
        shutil.copy(source_label_path, target_label_path)

    directory_contents = os.listdir(inputFolderPath)
    print("Contents of input directory:")
    for item in directory_contents:
      print(item)


print("Split Process Completed...")

# -----------data.yaml----------
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'


f = open(f"{outputFolderPath}/data.yaml", 'a')
f.write(dataYaml)
f.close()

print("Data.yaml file Created...")