import os

for folder in ["data/train", "data/val"]:
    print("\n", folder)
    for cls in os.listdir(folder):
        print(cls, ":", len(os.listdir(os.path.join(folder, cls))))
