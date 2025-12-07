"""
DETR Model for spacecraft detection
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import SpacecraftDataset, TestSpacecraftDataset, DATA_ROOT, NUM_CLASSES, CLASS_NAMES, detr_collate_fn
import pandas as pd
import numpy as np
import cv2

# Import necessary DETR components
from detr_model.detr import SetCriterion, PostProcess
from detr_model.__init__ import build_model
from detr_model.matcher import HungarianMatcher

# ✅ ESSENTIAL DETR IMPORTS: Now imports from the util/ folder
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import nested_tensor_from_tensor_list

import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # optional strict determinism:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

set_seed(42)


# ============================================================
# ARGUMENTS/CONFIG UTILITY (for DETR's build_model)
# ============================================================

class Args:
    """Simple class to hold required DETR configuration arguments."""
    def __init__(self, config):
        
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        
        # FIX #2: Required by detr.py's build function (dataset_file, frozen_weights)
        self.dataset_file = 'coco' 
        self.frozen_weights = None
        
        # FIX #3: Ensure num_classes is ALWAYS set (10 classes + 1 "No Object" class = 11)
        self.num_classes = config.get("num_classes", 11) 
        
        # --- Model Architecture ---
        self.backbone = 'resnet50' # Standard DETR backbone
        self.dilation = False
        self.position_embedding = 'sine'
        self.masks = config.get("masks", False) # Ensures this is set correctly
        self.num_queries = 100 # Max number of objects DETR will predict per image
        self.aux_loss = True
        self.lr = config.get("lr", 5e-5)
        self.lr_backbone = config.get("lr_backbone", 1e-5)
        
        # --- Transformer parameters ---
        self.hidden_dim = 256
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 2048
        self.enc_layers = 8
        self.dec_layers = 8
        self.pre_norm = False
        self.clip_max_norm = 0.1 # DETR often uses gradient clipping

        # --- Loss coefficients ---
        self.set_cost_class = 1
        self.set_cost_bbox = 5
        self.set_cost_giou = 2
        self.bbox_loss_coef = 6
        self.giou_loss_coef = 3
        self.eos_coef = 0.1 # Relative weight of the "no object" class

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(config):
    print("[DETR] 🚀 Training started...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Build Model and Criterion
    detr_args = Args(config)
    
    # DETR num_classes is typically the number of actual classes (10) + 1 (for 'no object')
    detr_args.num_classes = NUM_CLASSES
    model, criterion, postprocessors = build_model(detr_args)
    model.to(device)
    criterion.to(device)
    
    # 2. Multi-GPU support (using DataParallel for simplicity)
    if torch.cuda.device_count() > 1:
        print(f"⚡ Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # 3. Load datasets
    train_data = SpacecraftDataset(config["train_csv"], DATA_ROOT, training=True)   # Aug ON
    val_data = SpacecraftDataset(config["val_csv"], DATA_ROOT, training=False)      # Aug OFF

    # 4. DataLoader with custom collate_fn
    train_loader = DataLoader(train_data, batch_size=config["batch"], shuffle=True,
                              num_workers=8, pin_memory=True, collate_fn=detr_collate_fn)
    val_loader = DataLoader(val_data, batch_size=config["batch"], shuffle=False,
                            num_workers=8, pin_memory=True, collate_fn=detr_collate_fn)

    # 5. Optimizer
    # Different learning rates for the backbone and transformer/head are standard
    param_dicts = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad],
        "lr": detr_args.lr_backbone},
    ]
    opt = optim.AdamW(param_dicts, lr=detr_args.lr, weight_decay=1e-4)

    best_loss = float("inf")

    print(f"[DETR] Training for {config['epochs']} epochs...")
    for epoch in range(config["epochs"]):
        # --- Training ---
        model.train()
        total_loss = 0
        for samples, targets in train_loader:
            # samples is NestedTensor, targets is list of dicts
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            opt.zero_grad()
            outputs = model(samples)
            
            # Compute DETR loss
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            losses.backward()
            if detr_args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), detr_args.clip_max_norm)
                
            opt.step()
            total_loss += losses.item()

        avg_train_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for samples, targets in val_loader:
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                outputs = model(samples)
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                val_total_loss += losses.item()

        avg_val_loss = val_total_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_dir = os.path.join(config["project_dir"], "weights")
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{config['run_name']}_best.pt")
            
            # Save the raw model state dict
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), model_path)
            print(f"✅ Saved best model (Val Loss: {best_loss:.4f}) → {model_path}")

    print("[DETR] ✅ Training complete.")


# ============================================================
# TESTING FUNCTION
# ============================================================
def test_model(config):
    """
    Loads trained DETR model, performs inference on test directory,
    and saves predictions (class + denormalized bounding box).
    """
    print(f"[DETR] 🔍 Starting inference on: {config['test_images_full_path']}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model and Post-processor
    detr_args = Args(config)
    detr_args.num_classes = NUM_CLASSES
    model, _, postprocessors = build_model(detr_args)
    model.to(device)
    
    # Load weights
    model_path = os.path.join(config["project_dir"], "weights", f"{config['run_name']}_best.pt")
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model weights not found: {model_path}")
        return

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Loaded weights from: {model_path}")

    # 2. Load test data
    test_dir = config["test_images_full_path"]
    test_data = TestSpacecraftDataset(test_dir)
    test_loader = DataLoader(test_data, batch_size=config["batch"], shuffle=False,
                             num_workers=8, pin_memory=True)

    results = []

    # 3. Inference loop
    with torch.no_grad():
        for imgs_tensor, img_names, H_tensor, W_tensor in test_loader:
            
            # Convert image batch to NestedTensor for DETR model input
            samples = [img.to(device) for img in imgs_tensor] 
            samples = nested_tensor_from_tensor_list(samples).to(device)
            
            outputs = model(samples)
            
            # Prepare original target sizes for post-processing/denormalization
            orig_target_sizes = torch.stack([
                torch.tensor([h, w], dtype=torch.float32, device=device) 
                for h, w in zip(H_tensor.tolist(), W_tensor.tolist())
            ])
            
            # PostProcess scales normalized boxes to pixel values (xyxy format)
            results_post = postprocessors['bbox'](outputs, orig_target_sizes)

            for i, res in enumerate(results_post):
                img_name = img_names[i]
                H, W = H_tensor[i].item(), W_tensor[i].item()
                
                # Get the top prediction
                scores, labels, boxes = res['scores'], res['labels'], res['boxes']
                
                if scores.numel() == 0:
                    class_name = "N/A"
                    bbox_str = "(0, 0, 0, 0)"
                else:
                    # Take the highest scoring detection
                    best_idx = torch.argmax(scores)
                    
                    # Boxes are already scaled to original image size (W, H) in xyxy format
                    xmin, ymin, xmax, ymax = boxes[best_idx].round().long().tolist()
                    class_id = labels[best_idx].item()
                    
                    # Clamp coordinates to bounds
                    xmin, ymin = max(0, xmin), max(0, ymin)
                    xmax, ymax = min(W, xmax), min(H, ymax)

                    # Labels in DETR can be 0-9. Class 0 corresponds to CLASS_NAMES[0]
                    class_name = CLASS_NAMES[class_id]
                    bbox_str = f"({xmin}, {ymin}, {xmax}, {ymax})"

                results.append({
                    "Image name": img_name,
                    "Class": class_name,
                    "Bounding box": bbox_str
                })

    # 4. Save predictions
    output_path = os.path.join(config["project_dir"], f"{config['run_name']}_test_predictions.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"[DETR] ✅ Inference complete. Results saved to: {output_path}")


# ============================================================
# VISUALIZATION (Kept from original)
# ============================================================
def visualize_predictions(config):
    print("[Visualizer] 🖼️ Starting prediction visualization...")

    output_dir = os.path.join(config["project_dir"], "predictions_visualized")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(config["project_dir"], f"{config['run_name']}_test_predictions.csv")
    test_img_dir = config["test_images_full_path"]

    if not os.path.exists(csv_path):
        print(f"[ERROR] Prediction CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        img_name = row["Image name"]
        class_name = row["Class"]
        bbox_str = row["Bounding box"]

        try:
            xmin, ymin, xmax, ymax = [int(x.strip()) for x in bbox_str.strip("()").split(",")]
        except Exception:
            print(f"[WARN] Skipping malformed bbox: {bbox_str}")
            continue

        img_path = os.path.join(test_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not load image: {img_path}")
            continue

        # Draw bbox + label
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
        label = f"{class_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(img, (xmin, max(0, ymin - text_height - 4)), (xmin + text_width, ymin), color, -1)
        cv2.putText(img, label, (xmin, max(10, ymin - 2)), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, img)

    print(f"[Visualizer] ✅ Visualization complete. Annotated images saved to: {output_dir}")