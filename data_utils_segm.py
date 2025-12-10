import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple
from util.misc import nested_tensor_from_tensor_list

DATA_ROOT = os.path.join(".", "data")

# BODY_PANEL_CLASS_NAMES / SEG_NUM_CLASSES are kept in case you extend later.
BODY_PANEL_CLASS_NAMES = ["Body", "Panel"]
# DETR segmentation requires two foreground classes (Body, Panel) + one background class (No-object)
SEG_NUM_CLASSES = 3
# This is mainly for human readability; DETR uses indices 0 and 1 internally
SEG_CLASS_MAP = {name: i for i, name in enumerate(BODY_PANEL_CLASS_NAMES)}


def detr_collate_fn_segm(batch: List[Tuple[torch.Tensor, dict]]):
    """
    Collate function specialized for DETR (Detection Transformer).
    It converts a batch of images into a NestedTensor (padding) and keeps targets as a list of dicts.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    nested = nested_tensor_from_tensor_list(images)
    return nested, targets


class SpacecraftSegmDataset(Dataset):
    """
    DETR Instance Segmentation Dataset:
    - Loads RGB image and multi-channel ground truth mask.
    - Extracts binary masks for 'Body' (Class 0) and 'Panel' (Class 1) based on color channels.
    - Creates DETR targets: normalized bounding boxes, class labels, and binary masks for each instance.
    """

    def __init__(self, csv_file: str, split: str = "train", root_dir: str = DATA_ROOT):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV not found: {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.root = root_dir
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Original challenge columns
        image_name = row["Image name"]      # e.g. image_00007_img.jpg
        cls_name = row["Class"]            # e.g. Cheops

        # --- Image Loading and Preprocessing ---
        img_path = os.path.join(
            self.root, "images", cls_name, self.split, image_name
        )

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image missing: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Downsample image (standard practice for DETR efficiency)
        target_size = 512
        img_resized = cv2.resize(
            img, (target_size, target_size), interpolation=cv2.INTER_LINEAR
        )
        img_tensor = (
            torch.tensor(img_resized, dtype=torch.float32)
            .permute(2, 0, 1) / 255.0
        )

        # --- Mask Loading and Ground Truth Generation ---
        mask_name = image_name.replace("_img.jpg", "_layer.jpg")
        mask_path = os.path.join(
            self.root, "mask", cls_name, self.split, mask_name
        )

        # Resize mask using INTER_NEAREST to preserve sharp edges/pixel values
        mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR) 
        if mask_bgr is None:
            raise FileNotFoundError(f"Mask missing: {mask_path}")

        # Resize mask using INTER_NEAREST to preserve sharp edges/pixel values    
        mask_bgr = cv2.resize(
            mask_bgr, (target_size, target_size), interpolation=cv2.INTER_NEAREST
        )

        # Spacecraft body (Class 0) is Red-labeled (BGR index 2)
        body_bin = (mask_bgr[:, :, 2] > 0).astype(np.uint8)

        # Solar panels (Class 1) are Blue-labeled (BGR index 0)
        panel_bin = (mask_bgr[:, :, 0] > 0).astype(np.uint8)

        instances = []
        if body_bin.sum() > 0:
            # Body instance found
            instances.append(("body", body_bin))

        # The 5% size rule is handled by the *evaluation script*, not the training data loader.
        if panel_bin.sum() > 0:
            # Panel instance found 
            instances.append(("panel", panel_bin))

        if len(instances) == 0:
            # Handle images with no valid objects for DETR targets
            masks_tensor = torch.zeros((0, target_size, target_size), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.long)
            box = torch.zeros((0, 4), dtype=torch.float32)
        else:
            masks_list = []
            labels_list = []
            boxes_list = []

            for name, mb in instances:
                masks_list.append(torch.tensor(mb, dtype=torch.uint8))

                if name == "body":
                    labels_list.append(0)  # 0 = Body
                else:
                    labels_list.append(1)  # 1 = Panel

                # Bounding Box generation from mask (min/max coordinates)
                ys, xs = np.where(mb == 1)
                if len(xs) == 0 or len(ys) == 0:
                    xmin, ymin, xmax, ymax = 0, 0, target_size, target_size
                else:
                    xmin, xmax = xs.min(), xs.max()
                    ymin, ymax = ys.min(), ys.max()

                boxes_list.append([
                    xmin / target_size,
                    ymin / target_size,
                    xmax / target_size,
                    ymax / target_size,
                ])

            masks_tensor = torch.stack(masks_list, dim=0)             # [N, H, W]
            labels = torch.tensor(labels_list, dtype=torch.long)      # [N]
            box = torch.tensor(boxes_list, dtype=torch.float32)       # [N, 4]
        
        # Final DETR Target Dictionary (Ground Truth)
        target = {
            "labels": labels,
            "masks": masks_tensor,
            "boxes": box,
            "orig_size": torch.as_tensor([orig_h, orig_w]),
            "size": torch.as_tensor([target_size, target_size]),
        }
        return img_tensor, target

def preprocess_original_csv_segm(raw_csv_name: str, output_csv: str, split: str):
    """
    Utility function to process the raw, potentially multi-object CSV into a simplified format
    suitable for loading. It simplifies multi-class entries (e.g., 'Cheops' and 'Soho')
    into a single spacecraft name for directory lookup (e.g., 'Cheops').
    """
    raw_path = os.path.join(DATA_ROOT, raw_csv_name)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw CSV not found: {raw_path}")

    print(f"[SEGM] Preprocessing {raw_path} → {output_csv} for split='{split}'")
    df = pd.read_csv(raw_path)

    # Clean Class column: if it looks like "['Cheops', 'Soho']", keep first element
    def clean_class(c):
        if isinstance(c, str) and c.startswith("[") and c.endswith("]"):
            inner = c[1:-1]
            parts = [p.strip().strip("'").strip('"') for p in inner.split(",")]
            for p in parts:
                if p:
                    return p
            return inner
        return c

    df["Class"] = df["Class"].apply(clean_class)

    # keep only needed columns
    df_out = df[["Image name", "Class"]].copy()
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"[SEGM] Saved segmentation CSV → {output_csv} ({len(df_out)} rows)")
