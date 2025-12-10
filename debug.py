import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple

# --- Configuration (MUST MATCH YOUR SETUP) ---
DATA_ROOT = os.path.join(".", "data") 
# Replace with the actual path to your training CSV
TRAIN_CSV_FILE = os.path.join(DATA_ROOT, "train_segm_processed.csv") 
# ---------------------------------------------

def check_raw_mask_values(csv_file: str, root_dir: str = DATA_ROOT, num_samples: int = 10):
    """
    Loads raw mask files based on the CSV and prints the unique pixel values.
    This helps verify how the Body and Panel classes are encoded (e.g., 0, 10, 200).
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"[ERROR] CSV not found: {csv_file}. Please check the path.")
        return

    print(f"--- Checking Raw Mask Values (First {min(num_samples, len(df))} Samples) ---")
    
    # Iterate over a limited number of rows for debugging
    for idx in range(min(num_samples, len(df))):
        row = df.iloc[idx]
        image_name = row["Image name"]
        cls_name = row["Class"]
        split = "train" # Assumes the CSV is for the 'train' split

        # Mask path: data/mask/Class/split/image_XXXX_layer.jpg
        mask_name = image_name.replace("_img.jpg", "_layer.jpg")
        mask_path = os.path.join(
            root_dir, "mask", cls_name, split, mask_name
        )
        
        # Read mask as grayscale (single channel)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"[WARN] Mask missing: {mask_path}")
            continue

        # Find and print unique pixel values
        unique_values = np.unique(mask)
        
        print(f"\n[{idx+1}/{min(num_samples, len(df))}] Mask: {mask_name}")
        print(f"  → Class: {cls_name}")
        print(f"  → Unique Pixel Values: {unique_values.tolist()}")

        # Based on your data loader:
        # If Body is (mask > 0) and Panel is (mask >= 55), we expect:
        # - Background: 0
        # - Body-only: values < 55 (e.g., 1 to 54)
        # - Panel area: values >= 55 (e.g., 55 to 255)
        
        # Sanity Check for 2-class encoding
        if len(unique_values) == 2 and 0 in unique_values:
            print("  → **Encoding appears to be single-class (Foreground/Background only).**")
        elif len(unique_values) >= 3 and 0 in unique_values:
            panel_pixels = unique_values[unique_values >= 55].tolist()
            body_pixels = unique_values[(unique_values > 0) & (unique_values < 55)].tolist()
            print(f"  → **Expected Multi-Class Check (PANEL_THRESHOLD=55):**")
            print(f"    - Panel Pixels (>= 55): {panel_pixels}")
            print(f"    - Body Pixels (0 < value < 55): {body_pixels}")
        
    print("\n--- Debug Check Complete ---")


if __name__ == "__main__":
    check_raw_mask_values(TRAIN_CSV_FILE)