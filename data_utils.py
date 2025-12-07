"""
=========================================================
DATA UTILITIES — for DETR spacecraft model (w/ Augmentation)
=========================================================
"""
import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from os.path import join, isfile
from torch import nn, Tensor
from typing import List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ✅ ESSENTIAL DETR IMPORT: Now imports from the util/misc.py file
from util.misc import nested_tensor_from_tensor_list
from util.box_ops import box_cxcywh_to_xyxy

# ✅ Base data directory
DATA_ROOT = os.path.join(".", "data") # assuming script runs from cvia/

CLASS_NAMES = [
    "VenusExpress", "Cheops", "LisaPathfinder", "ObservationSat1",
    "Proba2", "Proba3", "Proba3ocs", "Smart1", "Soho", "XMM Newton"
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

# ============================================================
# COLLATE FUNCTION (UNCHANGED)
# ============================================================
def detr_collate_fn(batch: List[Tuple[Tensor, dict]]):
    """
    Collate function for DETR DataLoader.
    Batches images into a NestedTensor using the official utility 
    and keeps targets as a list of dicts.
    """
    # Separate images and targets
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Use the utility function to create the NestedTensor (from util.misc)
    nested_tensor = nested_tensor_from_tensor_list(images) 

    return nested_tensor, targets

# ============================================================
# 1️⃣ DATASET CLASS (MODIFIED w/ AUGMENTATION)
# ============================================================
class SpacecraftDataset(Dataset):
    """
    Loads labeled spacecraft images for training/validation w/ augmentation.
    Returns: (image_tensor, target_dict)
    target_dict = {'boxes': tensor(N, 4) in cxcywh normalized, 'labels': tensor(N)}
    """
    def __init__(self, csv_file, root_dir=DATA_ROOT, training=False):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.training = training  # ✅ Training mode flag
        
        # ✅ Train transforms (augmentation + normalization)
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.3),
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.7),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            ], p=0.4),
            A.Resize(224, 224),  # Fixed size after aug
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='coco',  # [x_center, y_center, width, height] normalized [0,1]
            label_fields=['category_ids']
        ))
        
        # ✅ Val/Test transforms (normalization only)
        self.test_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['category_ids']
        ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_path"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Missing image: {img_path}")

        full_h, full_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Handle no-object case
        if pd.isna(row["xmin"]) or row["xmin"] == -1:
            bboxes = []
            labels = []
        else:
            # Convert pixel bbox to normalized COCO format [x_center, y_center, w, h]
            xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
            x_center = (xmin + xmax) / 2 / full_w
            y_center = (ymin + ymax) / 2 / full_h
            width = (xmax - xmin) / full_w
            height = (ymax - ymin) / full_h
            bboxes = [[x_center, y_center, width, height]]
            labels = [int(row["class_id"])]

        # ✅ Apply augmentation (train) or normalization (val/test)
        if self.training and bboxes:  # Aug only if objects present
            transformed = self.train_transform(image=img, bboxes=bboxes, category_ids=labels)
        else:
            transformed = self.test_transform(image=img, bboxes=bboxes, category_ids=labels)
        
        img_tensor = transformed['image']
        bboxes = transformed['bboxes']
        labels = transformed['category_ids']

        # ✅ Convert back to DETR format (cxcywh normalized)
        if bboxes:
            # Albumentations returns COCO format, DETR expects cxcywh
            bboxes_cxcywh = torch.as_tensor(bboxes, dtype=torch.float32)
            target = {
                'boxes': bboxes_cxcywh,  # Already cxcywh normalized
                'labels': torch.as_tensor(labels, dtype=torch.int64),
                'orig_size': torch.as_tensor([full_h, full_w], dtype=torch.int64),
                'size': torch.as_tensor([224, 224], dtype=torch.int64)
            }
        else:
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'orig_size': torch.as_tensor([full_h, full_w], dtype=torch.int64),
                'size': torch.as_tensor([224, 224], dtype=torch.int64)
            }

        return img_tensor, target

# ============================================================
# 1️⃣b TEST DATASET CLASS (MODIFIED w/ Albumentations)
# ============================================================
class TestSpacecraftDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.image_names = sorted([
            f for f in os.listdir(img_dir)
            if isfile(join(img_dir, f)) and f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        # ✅ Test transform (normalization only)
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Could not read image: {img_path}")
            return torch.empty((3, 224, 224), dtype=torch.float32), img_name, 0, 0

        full_h, full_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply normalization
        transformed = self.transform(image=img)
        img_tensor = transformed['image']
        
        return img_tensor, img_name, full_h, full_w

# ============================================================
# 2️⃣ CSV PREPROCESSING (UNCHANGED)
# ============================================================
def preprocess_original_csv(csv_path, output_csv, split="train"):
    """
    Converts original annotation CSV into model-ready format.
    Input: [Image name, Class, Bounding box]
    Output: [image_path, xmin, ymin, xmax, ymax, class_id]
    """
    print(f"[INFO] Preprocessing {csv_path} → {output_csv}")
    df = pd.read_csv(os.path.join(DATA_ROOT, csv_path))
    new_rows = []

    for _, row in df.iterrows():
        image_name = row["Image name"]
        cls_name = row["Class"]
        bbox_str = row["Bounding box"]
        if cls_name not in CLASS_MAP:
            continue
        try:
            xmin, ymin, xmax, ymax = [int(x.strip()) for x in bbox_str.strip("()").split(",")]
        except Exception:
            print(f"[WARN] Skipping malformed bbox: {bbox_str}")
            continue
        image_path = os.path.join("images", cls_name, split, image_name)
        new_rows.append({
            "image_path": image_path,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "class_id": CLASS_MAP[cls_name]
        })

    pd.DataFrame(new_rows).to_csv(output_csv, index=False)
    print(f"[INFO] ✅ Saved processed CSV → {output_csv} ({len(new_rows)} rows)")

# ============================================================
# 3️⃣ VALIDATION (UNCHANGED)
# ============================================================
def validate_image_paths(csv_path):
    print(f"[CHECK] Validating image paths from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    valid, missing = 0, 0
    for _, row in df.iterrows():
        path = os.path.join(DATA_ROOT, row["image_path"])
        if os.path.exists(path):
            valid += 1
        else:
            missing += 1
    print(f"[CHECK] ✅ Valid: {valid} | ⚠️ Missing: {missing}")
    return valid, missing
