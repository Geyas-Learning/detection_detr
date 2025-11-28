"""
=========================================================
MAIN DETR SEGMENTATION PIPELINE — TRAIN • VAL • TEST
=========================================================
"""

import os
import argparse 
import warnings
import torch.distributed as dist 
import builtins 

# Suppress harmless torchvision warnings globally
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')

# Assuming these imports are correct for your project structure
from segmentation_data_utils import CLASS_NAMES, CLASS_MAP
from detr_segmentation_pipeline import (
    train_model,           # <-- RENAMED
    test_model,            # <-- RENAMED
    visualize_predictions  # <-- RENAMED
)

DATA_ROOT = os.path.join(".", "data")


# ============================================================
# CSV ENSURE FUNCTION (UNCHANGED)
# ============================================================

def ensure_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")


# ============================================================
# MAIN SEGMENTATION PIPELINE
# ============================================================

def main():
    
    # DDP AWARENESS (UNCHANGED)
    is_main_process = True
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
        if rank != 0:
            is_main_process = False
            builtins.print = lambda *args, **kwargs: None 
    
    
    print("\n🚀 Starting DETR Segmentation Pipeline")

    # STEP 1 — Prepare CSV paths
    train_csv = os.path.join(DATA_ROOT, "train.csv") # Updated path based on common DETR dataset naming
    val_csv = os.path.join(DATA_ROOT, "val.csv")    # Updated path based on common DETR dataset naming

    # Using 'train.csv' and 'val.csv' from the original context, assuming the dataset name changed
    if not os.path.exists(train_csv):
         train_csv = os.path.join(DATA_ROOT, "train.csv")
    if not os.path.exists(val_csv):
         val_csv = os.path.join(DATA_ROOT, "val.csv")

    ensure_csv(train_csv)
    ensure_csv(val_csv)

    # STEP 2 — Project directory
    project_dir = "runs_segmentation/train_detrSeg"
    os.makedirs(project_dir, exist_ok=True)

    # STEP 3 — Define Default Segmentation Config
    config = {
        "run_name": "detr_segmentation",
        "project_dir": project_dir,

        "train_csv": train_csv,
        "val_csv": val_csv,

        "epochs": 30,
        "batch": 4,     
        "num_classes": len(CLASS_NAMES),
    }

    # STEP 4 — Parse Command-Line Arguments and Override Config
    parser = argparse.ArgumentParser(description="DETR Segmentation Training")
    parser.add_argument('--epochs', type=int, default=config['epochs'], help='Number of training epochs.')
    parser.add_argument('--batch', type=int, default=config['batch'], help='Batch size for DataLoaders.')
    args = parser.parse_args()
    
    config['epochs'] = args.epochs
    config['batch'] = args.batch
    
    print(f"Effective Config: Epochs={config['epochs']}, Batch={config['batch']}")


    print("\n==============================================")
    print("  [SEG PIPELINE] Running Segmentation...")
    print("==============================================")

    # ------------------------ TRAIN ------------------------
    print("\n[STEP 1/3] 🧠 Training segmentation model...")
    train_model(config) # <-- RENAMED

    # ------------------------ INFERENCE --------------------
    # This step now runs on the 'segmentation_test/images' directory
    print("\n[STEP 2/3] 🔍 Running segmentation inference on DEDICATED TEST set...")
    test_model(config) # <-- RENAMED

    # ------------------------ VISUALIZATION ---------------
    print("\n[STEP 3/3] 🎨 Visualizing results for TEST set...")
    visualize_predictions(config) # <-- RENAMED

    print("\n==============================================")
    print("  ✅ Segmentation Pipeline Completed Successfully!")
    print("==============================================\n")


if __name__ == "__main__":
    main()