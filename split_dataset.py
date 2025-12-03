import os
import shutil
import random

SOURCE_TRAIN = "/home/kruthika/Downloads/archive/chest_xray/chest_xray/train"
SOURCE_TEST = "/home/kruthika/Downloads/archive/chest_xray/chest_xray/test"

DEST1 = "hospital_1"
DEST2 = "hospital_2"

CLASSES = ["NORMAL", "PNEUMONIA"]

for split in ["train", "test"]:
    for cls in CLASSES:
        os.makedirs(f"{DEST1}/{split}/{cls}", exist_ok=True)
        os.makedirs(f"{DEST2}/{split}/{cls}", exist_ok=True)

        src = os.path.join(SOURCE_TRAIN if split=="train" else SOURCE_TEST, cls)
        imgs = os.listdir(src)
        random.shuffle(imgs)

        half = len(imgs) // 2
        imgs1 = imgs[:half]
        imgs2 = imgs[half:]

        for img in imgs1:
            shutil.copy(os.path.join(src, img), f"{DEST1}/{split}/{cls}")

        for img in imgs2:
            shutil.copy(os.path.join(src, img), f"{DEST2}/{split}/{cls}")

print("Hospital 1 and 2 dataset created.")

