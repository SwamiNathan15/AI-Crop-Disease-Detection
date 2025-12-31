import os, shutil, random

SRC = "Tomato_dataset"
TRAIN = "data/train"
VAL = "data/val"

os.makedirs(TRAIN, exist_ok=True)
os.makedirs(VAL, exist_ok=True)

for cls in os.listdir(SRC):
    images = os.listdir(os.path.join(SRC, cls))
    random.shuffle(images)

    split = int(0.8 * len(images))
    train_imgs = images[:split]
    val_imgs = images[split:]

    os.makedirs(os.path.join(TRAIN, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL, cls), exist_ok=True)

    print(f"Processing class: {cls}")

    for i, img in enumerate(train_imgs):
        shutil.copy2(
            os.path.join(SRC, cls, img),
            os.path.join(TRAIN, cls, img)
        )
        if i % 200 == 0:
            print(f"  Copied {i} train images")

    for i, img in enumerate(val_imgs):
        shutil.copy2(
            os.path.join(SRC, cls, img),
            os.path.join(VAL, cls, img)
        )
        if i % 100 == 0:
            print(f"  Copied {i} val images")

print("Train–Validation split completed")
