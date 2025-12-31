import os

DATASET_PATH = "Tomato_dataset"

print("Classes found:\n")
for cls in sorted(os.listdir(DATASET_PATH)):
    cls_path = os.path.join(DATASET_PATH, cls)
    if os.path.isdir(cls_path):
        print(cls, "->", len(os.listdir(cls_path)), "images")
