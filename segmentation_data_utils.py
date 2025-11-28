"""
=========================================================
 SEGMENTATION DATA UTILITIES — DETR SPACECRAFT SEGMENTATION
=========================================================
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.ops import box_convert   # <-- reliable conversion

# ---------------------------
# SAME CLASS NAMES AS DETECTION
# ---------------------------

CLASS_NAMES = [
    "VenusExpress", "Cheops", "LisaPathfinder", "ObservationSat1",
    "Proba2", "Proba3", "Proba3ocs", "Smart1", "Soho", "XMM Newton"
]

CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

DATA_ROOT = "data"

# ---------------------------
# TRANSFORMS
# ---------------------------

def get_segmentation_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])


# ============================================================
# TRAIN/VAL SEGMENTATION DATASET (CSV-BASED, WITH GT)
# ============================================================

class SpacecraftSegmentationDataset(Dataset):
    """
    Loads images and masks/bboxes from CSV for Training and Validation.
    """

    def __init__(self, csv_file, root_dir=DATA_ROOT, split="train", transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV not found: {csv_file}")

        self.df = pd.read_csv(csv_file)
        self.root = root_dir
        self.split = split
        self.transform = transform if transform else get_segmentation_transforms()

    def __len__(self):
        return len(self.df)

    def _load_mask(self, mask_path):
        """Loads mask and converts to binary (0,1)."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            return np.zeros((1, 1), dtype=np.uint8)

        return (mask > 128).astype(np.uint8)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---------------------------
        # IMAGE
        # ---------------------------
        cls = row["Class"]
        img_name = row["Image name"]
        class_dir = os.path.join(self.root, "images", cls, self.split)

        img_path = os.path.join(class_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        img_tensor = self.transform(img)

        # ---------------------------
        # MASK
        # ---------------------------
        mask_name = img_name.replace("_img.jpg", "_layer.jpg")
        mask_dir = os.path.join(self.root, "mask", cls, self.split)
        mask_path = os.path.join(mask_dir, mask_name)

        mask = self._load_mask(mask_path)
        mask_tensor = torch.tensor(mask, dtype=torch.uint8)

        # ---------------------------
        # BOUNDING BOX FROM MASK
        # ---------------------------
        ys, xs = np.where(mask == 1)

        if len(xs) == 0 or len(ys) == 0:
            # fallback 1x1 box
            cx, cy = w // 2, h // 2
            xmin, ymin, xmax, ymax = cx, cy, cx + 1, cy + 1
        else:
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()

        # Format: xyxy
        xyxy = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

        # Convert to cxcywh (DETR format)
        cxcywh = box_convert(xyxy, in_fmt="xyxy", out_fmt="cxcywh")

        # Normalize by image size
        cxcywh = cxcywh / torch.tensor([w, h, w, h], dtype=torch.float32)

        # ---------------------------
        # TARGET
        # ---------------------------
        target = {
            "labels": torch.tensor([CLASS_MAP[cls]], dtype=torch.long),
            "masks": mask_tensor.unsqueeze(0),  # (1, H, W)
            "boxes": cxcywh,                    # normalized DETR format
            "orig_size": torch.as_tensor([h, w]),
            "size": torch.as_tensor([h, w]),
        }

        return img_tensor, target


# ============================================================
# TEST SEGMENTATION DATASET
# ============================================================

class TestSegmentationDataset(Dataset):
    """
    Loads images from segmentation_test/images for inference.
    """

    def __init__(self, transform=None):
        self.image_dir = os.path.join("segmentation_test", "images")

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Missing folder: {self.image_dir}")

        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        self.transform = transform if transform else get_segmentation_transforms()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        img_tensor = self.transform(img)

        return img_tensor, {
            "orig_size": torch.as_tensor([h, w]),
            "size": torch.as_tensor([h, w]),
            "file_identifier": img_name
        }
