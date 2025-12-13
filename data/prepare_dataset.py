# data/prepare_dataset.py

# pip install datasets pillow pandas tqdm
import os
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ---------------- CONFIG ---------------- #

SELECTED_CLASSES = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5,
    'bus': 6, 'truck': 8, 'traffic light': 10, 'stop sign': 13,
    'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19,
    'cow': 21, 'elephant': 22, 'bottle': 44, 'cup': 47, 'bowl': 51,
    'pizza': 59, 'cake': 61, 'chair': 62, 'couch': 63, 'bed': 65,
    'potted plant': 64
}

IMAGES_PER_CLASS = 100
BASE_DIR = "dataset/smartvision_dataset"

# ---------------------------------------- #

def load_coco_stream():
    print("📥 Loading COCO dataset (streaming mode)")
    return load_dataset("detection-datasets/coco", split="train", streaming=True)


def create_folders():
    print("📁 Creating dataset folders...")
    os.makedirs(BASE_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        for cls in SELECTED_CLASSES:
            os.makedirs(f"{BASE_DIR}/classification/{split}/{cls}", exist_ok=True)

    os.makedirs(f"{BASE_DIR}/detection/images", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/detection/labels", exist_ok=True)


def main():
    dataset = load_coco_stream()
    create_folders()

    # ===== Image collection =====
    class_images = {c: [] for c in SELECTED_CLASSES}
    counts = {c: 0 for c in SELECTED_CLASSES}

    for idx, item in enumerate(dataset):
        for cat in item["objects"]["category"]:
            for cls, cid in SELECTED_CLASSES.items():
                if cat == cid and counts[cls] < IMAGES_PER_CLASS:
                    class_images[cls].append(item)
                    counts[cls] += 1
                    break

        if all(v >= IMAGES_PER_CLASS for v in counts.values()):
            break

    print("✅ Image collection complete")

    # ===== Train / Val / Test split =====
    splits = {"train": {}, "val": {}, "test": {}}

    for cls, items in class_images.items():
        n = len(items)
        splits["train"][cls] = items[:int(0.7*n)]
        splits["val"][cls] = items[int(0.7*n):int(0.85*n)]
        splits["test"][cls] = items[int(0.85*n):]

    # ===== Save classification crops =====
    for split, data in splits.items():
        for cls, items in data.items():
            cls_id = SELECTED_CLASSES[cls]
            for i, item in enumerate(items):
                for bbox, cid in zip(item["objects"]["bbox"], item["objects"]["category"]):
                    if cid == cls_id:
                        x,y,w,h = bbox
                        img = item["image"].crop((x,y,x+w,y+h)).resize((224,224))
                        img.save(f"{BASE_DIR}/classification/{split}/{cls}/{cls}_{i}.jpg")
                        break

    print("🎉 Dataset preparation finished!")


if __name__ == "__main__":
    main()
