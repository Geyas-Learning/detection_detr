import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple
from util.misc import nested_tensor_from_tensor_list

DATA_ROOT = os.path.join(".", "data")

# You currently use a single foreground class in the dataset (class_id = 0).
# BODY_PANEL_CLASS_NAMES / SEG_NUM_CLASSES are kept in case you extend later.
BODY_PANEL_CLASS_NAMES = ["Body", "Panel"]
SEG_NUM_CLASSES = 2
SEG_CLASS_MAP = {name: i for i, name in enumerate(BODY_PANEL_CLASS_NAMES)}


def detr_collate_fn_segm(batch: List[Tuple[torch.Tensor, dict]]):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    nested = nested_tensor_from_tensor_list(images)
    return nested, targets


class SpacecraftSegmDataset(Dataset):
    """
    Uses segmentation CSVs with columns:
      - 'Image name'
      - 'Class'

    Assumes images at: data/images/<Class>/<split>/<Image name>
    Assumes masks  at: data/mask/<Class>/<split>/<Image name with _layer.jpg>
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

        # Image path: data/images/Class/split/image_XXXX_img.jpg
        img_path = os.path.join(
            self.root, "images", cls_name, self.split, image_name
        )

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image missing: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # downsample
        target_size = 512
        img_resized = cv2.resize(img, (target_size, target_size),
                                 interpolation=cv2.INTER_LINEAR)
        img_tensor = (
            torch.tensor(img_resized, dtype=torch.float32)
            .permute(2, 0, 1) / 255.0
        )

        # Mask path: data/mask/Class/split/image_XXXX_layer.jpg
        mask_name = image_name.replace("_img.jpg", "_layer.jpg")
        mask_path = os.path.join(
            self.root, "mask", cls_name, self.split, mask_name
        )
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask missing: {mask_path}")
        mask = cv2.resize(mask, (target_size, target_size),
                          interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask > 128).astype(np.uint8)
        mask_tensor = torch.tensor(mask_bin, dtype=torch.uint8)

        ys, xs = np.where(mask_bin == 1)
        if len(xs) == 0 or len(ys) == 0:
            xmin, ymin, xmax, ymax = 0, 0, target_size, target_size
        else:
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
        box = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

        labels = torch.tensor([0], dtype=torch.long)

        target = {
            "labels": labels,
            "masks": mask_tensor.unsqueeze(0),
            "boxes": box,
            "orig_size": torch.as_tensor([orig_h, orig_w]),
            "size": torch.as_tensor([target_size, target_size]),
        }
        return img_tensor, target



def preprocess_original_csv_segm(raw_csv_name: str, output_csv: str, split: str):
    """
    Input CSV columns:
      - 'Image name'
      - 'Mask name'   (ignored here)
      - 'Class'       (can be a single name or a list string)
      - 'Bounding box' (ignored)
    Output CSV columns:
      - 'Image name'
      - 'Class' (single spacecraft name)
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
