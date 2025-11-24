"""
=========================================================
MAIN DETR PIPELINE — TRAIN • VAL • mAP • TEST • VISUALIZE
=========================================================
"""

import os
import importlib.util
import argparse # <--- NEW IMPORT: Enables command-line arguments

from data_utils import (
    preprocess_original_csv,
    validate_image_paths,
    convert_to_coco_json,
    DATA_ROOT
)


# ============================================================
# CHANGE MODEL FILE HERE
# ============================================================

MODEL_NAME = "detr_pipeline_functions_tensor"
MODEL_FILE_PATH = f"detection_detr/{MODEL_NAME}.py"

# ============================================================
# CSV ENSURE FUNCTION
# ============================================================

def ensure_csv(csv_path, original_csv, split):
    """
    Ensures processed CSV exists and is valid.
    If missing or corrupted → rebuild automatically.
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] Missing processed CSV → rebuilding {csv_path}")
        preprocess_original_csv(original_csv, csv_path, split)
        return

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline()
            if "," not in header:
                raise ValueError("Corrupt CSV header")
    except Exception:
        print(f"[WARN] Corrupted CSV → rebuilding {csv_path}")
        preprocess_original_csv(original_csv, csv_path, split)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    
    # ------------------------------------------------------------
    # STEP 0 — ARGUMENT PARSING <--- MODIFIED SECTION
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser(description="DETR Training and Evaluation Pipeline.")
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=25, 
        help='Number of training epochs (default: 25).'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=8, 
        help='Training batch size (default: 8).'
    )
    args = parser.parse_args()

    print(f"\n🚀 Starting DETR Spacecraft Pipeline (Model: {MODEL_NAME})")
    print(f"[INFO] Using Arguments: Epochs={args.epochs}, Batch Size={args.batch_size}")

    # ------------------------------------------------------------
    # STEP 1 — PREPARE TRAIN + VAL CSV
    # ------------------------------------------------------------

    train_csv = os.path.join(DATA_ROOT, "train_processed_tensor.csv")
    val_csv   = os.path.join(DATA_ROOT, "val_processed_tensor.csv")

    ensure_csv(train_csv, "train.csv", "train")
    ensure_csv(val_csv,   "val.csv",   "val")

    validate_image_paths(train_csv)
    validate_image_paths(val_csv)

    # ------------------------------------------------------------
    # STEP 2 — BUILD COCO JSONs FOR mAP EVALUATION
    # ------------------------------------------------------------

    print("\n📦 Creating COCO JSON (train & val)...")

    coco_train_json = os.path.join(DATA_ROOT, "coco_train.json")
    coco_val_json   = os.path.join(DATA_ROOT, "coco_val.json")

    convert_to_coco_json(train_csv, coco_train_json)
    convert_to_coco_json(val_csv,   coco_val_json)

    # ------------------------------------------------------------
    # STEP 3 — DYNAMICALLY LOAD MODEL MODULE
    # ------------------------------------------------------------

    print(f"\n[INFO] Importing model code: {MODEL_FILE_PATH}")

    spec = importlib.util.spec_from_file_location("model_lib", MODEL_FILE_PATH)

    if spec is None:
        print(f"❌ ERROR: Could not open model file → {MODEL_FILE_PATH}")
        return

    model_lib = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(model_lib)
    except Exception as e:
        print(f"❌ ERROR loading model module:\n{e}")
        return

    # ------------------------------------------------------------
    # STEP 4 — TRAIN CONFIGURATION <--- MODIFIED SECTION
    # ------------------------------------------------------------

    project_dir = os.path.join("..", "runs_tensor", "train_detr34")
    os.makedirs(project_dir, exist_ok=True)

    config = {
        # Model identity
        "run_name": MODEL_NAME,
        "project_dir": project_dir,

        # CSVs
        "train_csv": train_csv,
        "val_csv":   val_csv,

        # COCO JSONs
        "coco_train_json": coco_train_json,
        "coco_val_json":   coco_val_json,

        # Training settings (Now uses command-line arguments)
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": 224,
        "num_classes": 10,

        # Test images (OUTSIDE /data/)
        "test_images_full_path": os.path.join("..", "test", "images")
    }

    # ------------------------------------------------------------
    # STEP 5 — RUN PIPELINE STAGES
    # ------------------------------------------------------------

    print("\n==============================================")
    print("  [PIPELINE] Running Model Operations...")
    print("==============================================")

    # ------------------------ TRAIN ------------------------
    if hasattr(model_lib, "train_model"):
        print("\n[STEP 1/5] 🧠 Training model...")
        model_lib.train_model(config)
    else:
        print("❌ ERROR: train_model() not found in model file.")

    # ------------------------ EVALUATE mAP ------------------------
    if hasattr(model_lib, "evaluate_map"):
        print("\n[STEP 2/5] 📊 Evaluating COCO mAP...")
        model_lib.evaluate_map(config)
    else:
        print("❌ ERROR: evaluate_map() not found in model file.")
    
    # ------------------------ CLASSIFICATION METRICS (NEW STEP) ------------------------
    if hasattr(model_lib, "generate_classification_metrics"):
        print("\n[STEP 3/5] 🔢 Generating Classification Report & Confusion Matrix...")
        model_lib.generate_classification_metrics(config)
    else:
        print("❌ ERROR: generate_classification_metrics() missing in model file.")


    # ------------------------ INFERENCE ------------------------
    if hasattr(model_lib, "test_model"):
        print("\n[STEP 4/5] 🔍 Running inference on test images...")
        model_lib.test_model(config)
    else:
        print("❌ ERROR: test_model() missing in model file.")

    # ------------------------ VISUALIZATION ------------------------
    if hasattr(model_lib, "visualize_predictions"):
        print("\n[STEP 5/5] 🖼️ Visualizing predictions...")
        model_lib.visualize_predictions(config)
    else:
        print("❌ ERROR: visualize_predictions() missing in model file.")

    print("\n==============================================")
    print("  ✅ Pipeline Completed Successfully! (5 Steps)")
    print("==============================================\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()