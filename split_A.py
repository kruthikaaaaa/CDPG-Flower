import os
import shutil
import random

SOURCE = "/home/kruthika/Downloads/archive/chest_xray"
DEST1 = "hospital_1"
DEST2 = "hospital_2"

def create_dirs():
    for d in [DEST1, DEST2]:
        os.makedirs(f"{d}/train", exist_ok=True)
        os.makedirs(f"{d}/test", exist_ok=True)

def split_class(class_path, out1, out2):
    images = os.listdir(class_path)
    random.shuffle(images)

    half = len(images) // 2
    part1 = images[:half]
    part2 = images[half:]

    for img in part1:
        shutil.copy(os.path.join(class_path, img), out1)

    for img in part2:
        shutil.copy(os.path.join(class_path, img), out2)

def split_dataset():
    create_dirs()

    # TRAIN SPLIT
    train_path = os.path.join(SOURCE, "train")
    for cls in ["NORMAL", "PNEUMONIA"]:
        split_class(
            os.path.join(train_path, cls),
            f"{DEST1}/train/{cls}",
            f"{DEST2}/train/{cls}",
        )

    # TEST (same test for both clients)
    test_path = os.path.join(SOURCE, "test")
    for cls in ["NORMAL", "PNEUMONIA"]:
        for d in [DEST1, DEST2]:
            os.makedirs(f"{d}/test/{cls}", exist_ok=True)

        for img in os.listdir(os.path.join(test_path, cls)):
            img_path = os.path.join(test_path, cls, img)
            shutil.copy(img_path, f"{DEST1}/test/{cls}")
            shutil.copy(img_path, f"{DEST2}/test/{cls}")

    print("✅ Client 1 & Client 2 dataset created successfully!")

if __name__ == "__main__":
    split_dataset()
