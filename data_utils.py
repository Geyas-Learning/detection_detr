"""
=========================================================
DATA UTILITIES — DETR SPACECRAFT DATA PIPELINE (FINAL)
=========================================================
"""

import os
import cv2
import json
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from util.misc import nested_tensor_from_tensor_list 
from typing import List, Tuple
import torchvision.transforms as T


# ============================================================
# CONSTANTS
# ============================================================

DATA_ROOT = os.path.join(".", "data")   # Project root-level directory

CLASS_NAMES = [
    "VenusExpress", "Cheops", "LisaPathfinder", "ObservationSat1",
    "Proba2", "Proba3", "Proba3ocs", "Smart1", "Soho", "XMM Newton"
]

NUM_CLASSES = len(CLASS_NAMES)
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ============================================================
# DETR TRANSFORM PIPELINE
# ============================================================

def get_detr_transforms():
    """Official DETR-style preprocessing."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])


# ============================================================
# COLLATE FUNCTION FOR DETR
# ============================================================

def detr_collate_fn(batch: List[Tuple[torch.Tensor, dict]]):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]

    nested = nested_tensor_from_tensor_list(images)
    return nested, targets


# ============================================================
# TRAINING / VALIDATION DATASET
# ============================================================

class SpacecraftDataset(Dataset):
    """
    DETR-ready dataset using processed CSV:

    CSV schema:
        image_path, xmin, ymin, xmax, ymax, class_id
    """
    def __init__(self, csv_file, root_dir=DATA_ROOT, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV not found: {csv_file}")

        self.df = pd.read_csv(csv_file)
        self.root = root_dir
        self.transform = transform if transform is not None else get_detr_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_rel = row["image_path"]        # ex: images/Class/split/img.jpg
        img_path = os.path.join(self.root, img_rel)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        xmin = row["xmin"]; ymin = row["ymin"]
        xmax = row["xmax"]; ymax = row["ymax"]
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        class_id = row["class_id"]

        # Convert to DETR format (relative cx,cy,w,h)
        cx = (xmin + xmax) / 2.0 / w
        cy = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        boxes = torch.tensor([[cx, cy, bw, bh]], dtype=torch.float32)
        labels = torch.tensor([class_id], dtype=torch.long)

        img_tensor = self.transform(img)

        target = {
            "boxes": boxes,
            "labels": labels,
            "orig_size": torch.tensor([h, w]),
            "size": torch.tensor([img_tensor.shape[1], img_tensor.shape[2]]),
        }

        return img_tensor, target


# ============================================================
# TEST DATASET
# ============================================================

class TestSpacecraftDataset(Dataset):
    """Loads only images for inference."""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.names = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.transform = transform if transform else get_detr_transforms()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        path = os.path.join(self.img_dir, name)

        img = Image.open(path).convert("RGB")
        w, h = img.size

        tensor_img = self.transform(img)

        return tensor_img, name, h, w


# ============================================================
# CSV PREPROCESSOR
# ============================================================

def preprocess_original_csv(original_csv_name, output_csv, split):
    """
    Convert original CSV → processed CSV.
    """

    full_in = os.path.join(DATA_ROOT, original_csv_name)
    print(f"[INFO] Preprocessing CSV: {full_in} → {output_csv}")

    df = pd.read_csv(full_in)
    rows = []

    for _, row in df.iterrows():
        name = row["Image name"]
        cls = row["Class"]
        bbox = row["Bounding box"]

        if cls not in CLASS_MAP:
            continue

        try:
            xmin, ymin, xmax, ymax = [int(v.strip()) for v in bbox.strip("()").split(",")]
        except:
            print(f"[WARN] Bad bbox: {bbox}")
            continue

        image_path = os.path.join("images", cls, split, name)

        rows.append({
            "image_path": image_path,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "class_id": CLASS_MAP[cls],
        })

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"[INFO] Saved processed CSV: {output_csv} ({len(rows)} rows)")


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image_paths(csv_path):
    print(f"[CHECK] Validating: {csv_path}")

    df = pd.read_csv(csv_path)
    valid, missing = 0, 0

    for _, row in df.iterrows():
        path = os.path.join(DATA_ROOT, row["image_path"])
        if os.path.exists(path):
            valid += 1
        else:
            missing += 1
            print(f"[MISSING] {path}")

    print(f"[CHECK] Valid: {valid} | Missing: {missing}")
    return valid, missing


# ============================================================
# COCO JSON CONVERTER (VALID COCO FORMAT)
# ============================================================

def convert_to_coco_json(processed_csv, json_out):
    """
    Converts processed CSV into valid COCO-format JSON.
    Makes file_name relative to data/images/ (no leading "images/").
    """

    df = pd.read_csv(processed_csv)
    images_list = []
    ann_list = []

    ann_id = 1

    for idx, row in df.iterrows():
        rel_path = row["image_path"]  # "images/Class/val/img.jpg"
        abs_path = os.path.join(DATA_ROOT, rel_path)

        if not os.path.exists(abs_path):
            continue

        # REMOVE "images/" prefix → make relative to data/images
        clean_path = rel_path.replace("images/", "", 1)

        img = Image.open(abs_path)
        w, h = img.size

        images_list.append({
            "id": idx + 1,
            "file_name": clean_path,   # <-- FIXED HERE
            "width": w,
            "height": h
        })

        xmin = row["xmin"]
        ymin = row["ymin"]
        xmax = row["xmax"]
        ymax = row["ymax"]
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)

        bbox_w = xmax - xmin
        bbox_h = ymax - ymin

        ann_list.append({
            "id": ann_id,
            "image_id": idx + 1,
            "bbox": [xmin, ymin, bbox_w, bbox_h],
            "area": bbox_w * bbox_h,
            "category_id": row["class_id"] + 1,
            "iscrowd": 0
        })
        ann_id += 1

    coco_dict = {
        "info": {
            "description": "Spacecraft Dataset",
            "version": "1.0"
        },
        "licenses": [],
        "images": images_list,
        "annotations": ann_list,
        "categories": [
            {"id": cid + 1, "name": CLASS_NAMES[cid]}
            for cid in range(NUM_CLASSES)
        ]
    }


    with open(json_out, "w") as f:
        json.dump(coco_dict, f, indent=4)

    print(f"[COCO] Saved JSON → {json_out}")
