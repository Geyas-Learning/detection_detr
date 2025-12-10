# =========================================================
# main_segm.py
# =========================================================
"""
=========================================================
MAIN PIPELINE — DETR for Spacecraft SEMANTIC SEGMENTATION
=========================================================
"""
import importlib.util
import os
from data_utils_segm import preprocess_original_csv_segm, DATA_ROOT


MODEL_NAME = "detr_pipeline_functions_segm"
MODEL_FILE_PATH = f"detection_detr/{MODEL_NAME}.py"


def ensure_csv(csv_path, raw_csv_name, split):
    """Ensure processed segmentation CSV exists; rebuild if missing."""
    if not os.path.exists(csv_path):
        print(f"[WARN] Missing processed CSV → rebuilding: {csv_path}")
        preprocess_original_csv_segm(raw_csv_name, csv_path, split)
        return

def main():
    print(f"\n🚀 Starting Spacecraft SEGMENTATION Pipeline using model: {MODEL_NAME}")

    # --- Prepare CSVs ---
    train_csv = os.path.join(DATA_ROOT, "train_segm_processed.csv")
    val_csv = os.path.join(DATA_ROOT, "val_segm_processed.csv")
    
    # Use the segmentation-specific processor
    ensure_csv(train_csv, "train.csv", "train")
    ensure_csv(val_csv, "val.csv", "val")
    
    # --- Import Pipeline Functions ---
    spec = importlib.util.spec_from_file_location(MODEL_NAME, MODEL_FILE_PATH)
    if spec is None:
        print(f"❌ ERROR: Cannot find pipeline file at {MODEL_FILE_PATH}")
        return
        
    model_lib = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(model_lib)
    except Exception as e:
        print(f"❌ ERROR: Failed to import model module '{MODEL_NAME}'.\nDetails: {e}")
        return

    # --- Config ---
    config = {
        "epochs": 15,
        "batch": 4, 
        # segmentation output will generate in this folder
        "project_dir": "./runs_segmentation/train_detr", 
        "train_csv": train_csv,
        "val_csv": val_csv,
        # tesing files: "segmentation_test"
        "test_images_full_path": "./segmentation_test/images", 
        "run_name": MODEL_NAME
    }
    
    os.makedirs(config["project_dir"], exist_ok=True) 

    # --- Run pipeline ---
    print("\n[PIPELINE] 🚦 Beginning model operations...")

    if hasattr(model_lib, "train_model"):
        print("[STEP 1/3] 🧠 Training segmentation model...")
        model_lib.train_model(config)
    else:
        print("❌ train_model not found.")

    if hasattr(model_lib, "test_model"):
        print("\n[STEP 2/3] 🔍 Running inference (mask generation)...")
        model_lib.test_model(config)
    else:
        print("❌ test_model not found.")

    if hasattr(model_lib, "visualize_predictions"):
        print("\n[STEP 3/3] 🖼️ Skipping visualization. Check predicted_masks folder.")
        model_lib.visualize_predictions(config)
    else:
        print("❌ visualize_predictions not found.")


if __name__ == "__main__":
    main()