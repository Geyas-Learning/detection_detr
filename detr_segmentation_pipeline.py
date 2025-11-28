"""
=========================================================
DETR SEGMENTATION PIPELINE — TRAIN • VAL • INFER • VISUALIZE
(Updated: Renamed functions, Implemented Test Set Inference)
=========================================================
"""

import os
import time
import csv
import torch
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# === NEW DDP & FP16 IMPORTS ===
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast 
# ==============================

# ---------------------------
# INTERNAL IMPORTS
# ---------------------------
from segmentation_data_utils import (
    SpacecraftSegmentationDataset, 
    TestSegmentationDataset, # <-- NEW IMPORT
    CLASS_NAMES, 
    CLASS_MAP
)
from detr_model.__init__ import build_segmentation_model
from util.misc import nested_tensor_from_tensor_list


# ============================================================
# CUSTOM COLATE FUNCTION (UNCHANGED)
# ============================================================
def collate_fn(batch: List[Tuple[Any, Any]]) -> Tuple[List[Any], List[Any]]:
    """Separates a batch of (image, target) tuples into a list of images and a list of targets."""
    return tuple(zip(*batch))


# ============================================================
# ARGUMENT WRAPPER (UNCHANGED)
# ============================================================

class SegArgs:
    def __init__(self, config):
        # ... (UNCHANGED ARGUMENTS)
        self.device = "cuda" 
        self.dataset_file = "custom"
        self.num_classes = config["num_classes"]
        self.backbone = "resnet50"
        self.dilation = False
        self.position_embedding = "sine"
        self.masks = True
        self.lr_backbone = 1e-5
        self.num_queries = 100
        self.hidden_dim = 256
        self.nheads = 8
        self.dropout = 0.1
        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.pre_norm = False
        self.set_cost_class = 1
        self.set_cost_bbox = 5
        self.set_cost_giou = 2
        self.aux_loss = True
        self.eos_coef = 0.1
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 2
        self.mask_loss_coef = 1
        self.dice_loss_coef = 1
        self.frozen_weights = None
        self.clip_max_norm = 0.1


# ============================================================
# TRAINING LOOP (RENAMED to train_model)
# ============================================================

def train_model(config): # Renamed from train_segmentation
    
    # DDP SETUP & INITIALIZATION (UNCHANGED)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = f"cuda:{rank}"
        is_main_process = (rank == 0)
    else:
        rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_main_process = True
        
    print(f"\n[DETR-SEG] 🧠 Starting Segmentation Training on rank {rank}/{world_size}...")

    # Build model & Data Loaders (UNCHANGED)
    args = SegArgs(config)
    model, criterion, postprocessors = build_segmentation_model(args)
    model.to(device)
    criterion.to(device)

    train_set = SpacecraftSegmentationDataset(csv_file=config["train_csv"], root_dir="data", split="train")
    val_set = SpacecraftSegmentationDataset(csv_file=config["val_csv"], root_dir="data", split="val")

    # ... (DataLoader setup, Optimizer, Scaler, Accumulation steps - UNCHANGED)
    num_workers = 12
    if world_size > 1:
        sampler_train = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        sampler_val = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_set, batch_size=config["batch"], sampler=sampler_train, num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=config["batch"], sampler=sampler_val, num_workers=num_workers, collate_fn=collate_fn)
        model = DDP(model, device_ids=[rank])
    else:
        train_loader = DataLoader(train_set, batch_size=config["batch"], shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=config["batch"], shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": args.lr_backbone},
    ]

    optimizer = optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler()
    ACCUMULATION_STEPS = 4

    # ... (Logging, CSV setup, Epoch loop - UNCHANGED)
    if is_main_process:
        # ... (writer, csv_path, csv_header setup)
        writer = SummaryWriter(os.path.join(config["project_dir"], "tensorboard_seg"))
        csv_path = os.path.join(config["project_dir"], "segmentation_results.csv")
        csv_header = [
            "epoch","time",
            "train/loss_ce","train/loss_mask","train/loss_dice",
            "val/loss_total",
            "metrics/mIoU","metrics/Dice","metrics/PixelAcc",
            "metrics/IoU_fg","metrics/IoU_bg",
            "metrics/Dice_fg","metrics/Dice_bg",
            "lr/backbone","lr/head"
        ]
        with open(csv_path, "w", newline="") as f: csv.writer(f).writerow(csv_header)
        best_loss = float("inf")
        print(f"[ACCUMULATION] Using ACCUMULATION_STEPS = {ACCUMULATION_STEPS}. Please ensure your physical batch size (--batch) is small enough (e.g., 2).")

    print(f"[DETR-SEG] Training for {config['epochs']} epochs...\n")

    # -------------------------------
    # 🔁 EPOCH LOOP
    # -------------------------------
    for epoch in range(config["epochs"]):
        if world_size > 1:
            # Essential for correct shuffling in DDP
            sampler_train.set_epoch(epoch) 

        epoch_start = time.time()
        model.train()
        
        train_loss_ce = 0
        train_loss_mask = 0
        train_loss_dice = 0
        total_batches = 0

        # -------------------------------
        # TRAIN
        # -------------------------------
        for i, (samples, targets) in enumerate(train_loader): # Added 'i' for batch index
            
            samples = nested_tensor_from_tensor_list(samples)
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # optimizer.zero_grad() # REMOVED for accumulation. Zeroing happens after step.

            # FP16: Wrap forward pass in autocast
            with autocast():
                outputs = model(samples)
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict

                total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
                # Normalize loss to account for accumulation
                total_loss = total_loss / ACCUMULATION_STEPS 

            # FP16: Use scaler for backward pass (Always call backward to accumulate gradients)
            scaler.scale(total_loss).backward()

            # Perform optimization step only after ACCUMULATION_STEPS batches
            if (i + 1) % ACCUMULATION_STEPS == 0:
                if args.clip_max_norm > 0:
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)

                # FP16: Use scaler for optimizer step
                scaler.step(optimizer)
                scaler.update() 
                
                # Zero gradients for the next accumulation cycle
                optimizer.zero_grad()

            # Track losses (using the un-divided loss_dict items)
            train_loss_ce += loss_dict["loss_ce"].item()
            train_loss_mask += loss_dict["loss_mask"].item()
            train_loss_dice += loss_dict["loss_dice"].item()
            total_batches += 1
            
        # Optional: One final step for residual gradients (if dataset size % ACCUMULATION_STEPS != 0)
        if (len(train_loader) % ACCUMULATION_STEPS != 0) and (i + 1) % ACCUMULATION_STEPS != 0:
             if args.clip_max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)

             scaler.step(optimizer)
             scaler.update()
             optimizer.zero_grad() # Zero gradients to clear for the next epoch


        avg_loss_ce = train_loss_ce / total_batches
        avg_loss_mask = train_loss_mask / total_batches
        avg_loss_dice = train_loss_dice / total_batches
        
        if is_main_process:
            writer.add_scalar("train/loss_ce", avg_loss_ce, epoch)
            writer.add_scalar("train/loss_mask", avg_loss_mask, epoch)
            writer.add_scalar("train/loss_dice", avg_loss_dice, epoch)

        # -------------------------------
        # VALIDATION
        # -------------------------------
        model.eval()
        val_loss_total = 0
        iou_list = []
        dice_list = []
        acc_list = []
        fg_iou_list = []
        bg_iou_list = []
        fg_dice_list = []
        bg_dice_list = []


        with torch.no_grad():
            for samples, targets in val_loader:
                
                samples = nested_tensor_from_tensor_list(samples)
                samples = samples.to(device)
                targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # FP16: Wrap forward pass in autocast
                with autocast():
                    outputs = model(samples)
                    loss_dict = criterion(outputs, targets_gpu)
                
                weight_dict = criterion.weight_dict
                val_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
                val_loss_total += val_loss.item()
                
                # Postprocessing for metrics
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets_gpu], dim=0).to(device)
                max_h, max_w = samples.tensors.shape[-2:]
                max_target_sizes = torch.as_tensor([[max_h, max_w]] * len(targets_gpu), device=device)
                
                # FIX: Use keyword arguments
                pred_list = postprocessors["segm"]([outputs], target_sizes=orig_target_sizes, max_target_sizes=max_target_sizes)
                pred = pred_list[0]
                
                # Metric calculation using CPU targets
                pred_mask = (pred["masks"][0].cpu().numpy() > 0.5).astype(np.uint8)
                gt_mask = targets[0]["masks"][0].cpu().numpy().astype(np.uint8)

                # Ensure masks are the same size (PostProcessor resizes pred_mask to orig_size)
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Flatten for Confusion Matrix
                y_true = gt_mask.flatten()
                y_pred = pred_mask.flatten()
                
                # Confusion Matrix: [[TN, FP], [FN, TP]] where background=0, foreground=1
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

                # Calculate metrics
                EPS = 1e-6
                iou_fg = TP / (TP + FP + FN + EPS)
                iou_bg = TN / (TN + FP + FN + EPS)
                dice_fg = (2 * TP) / (2 * TP + FP + FN + EPS)
                dice_bg = (2 * TN) / (2 * TN + FP + FN + EPS)
                pixel_acc = (TP + TN) / (TP + TN + FP + FN + EPS)
                
                # Append to lists
                iou_list.append((iou_fg + iou_bg) / 2)
                dice_list.append((dice_fg + dice_bg) / 2)
                acc_list.append(pixel_acc)
                fg_iou_list.append(iou_fg)
                bg_iou_list.append(iou_bg)
                fg_dice_list.append(dice_fg)
                bg_dice_list.append(dice_bg)


        avg_val_loss = val_loss_total / len(val_loader)
        mIoU = np.mean(iou_list)
        mDice = np.mean(dice_list)
        pAcc = np.mean(acc_list)
        IoU_fg = np.mean(fg_iou_list)
        IoU_bg = np.mean(bg_iou_list)
        Dice_fg = np.mean(fg_dice_list)
        Dice_bg = np.mean(bg_dice_list)


        # -------------------------------
        # LOGGING AND SAVING (Only on Main Process)
        # -------------------------------
        if is_main_process:
            writer.add_scalar("val/mIoU", mIoU, epoch)
            writer.add_scalar("val/Dice", mDice, epoch)
            
            # PRINT EPOCH SUMMARY
            print(
                f"Epoch [{epoch+1}/{config['epochs']}] "
                f"Loss_ce: {avg_loss_ce:.4f} | "
                f"Loss_mask: {avg_loss_mask:.4f} | "
                f"Loss_dice: {avg_loss_dice:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"mIoU: {mIoU:.4f} | Dice: {mDice:.4f}"
            )

            # SAVE BEST MODEL
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss

                save_dir = os.path.join(config["project_dir"], "weights_seg")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{config['run_name']}_best_seg.pt")

                # Unwrap DDP model before saving state_dict
                to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save(to_save, save_path)

                print(f"  ✔ Saved best segmentation model → {save_path}")

            # WRITE CSV LOG
            lr_backbone = optimizer.param_groups[1]["lr"]
            lr_head = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            csv_row = [
                epoch + 1,
                f"{epoch_time:.2f}",
                avg_loss_ce, avg_loss_mask, avg_loss_dice,
                avg_val_loss,
                mIoU, mDice, pAcc,
                IoU_fg, IoU_bg,
                Dice_fg, Dice_bg,
                lr_backbone, lr_head
            ]

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(csv_row)

    # DDP Cleanup
    if world_size > 1:
        dist.destroy_process_group()
        
    if is_main_process:
        writer.close()
        print("\n[DETR-SEG] ✔ Segmentation Training Completed.\n")


# ============================================================
# INFERENCE (Updated for FP16, weight loading, and num_workers=12)
# ============================================================

def test_model(config): # Renamed from infer_segmentation
    # This now infers on the images in segmentation_test/images
    print("\n[DETR-SEG] 🔍 Running Segmentation Inference on DEDICATED TEST Set (No CSV)...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = SegArgs(config)

    model, _, postprocessors = build_segmentation_model(args)
    model.to(device)

    weight_path = os.path.join(config["project_dir"], "weights_seg", f"{config['run_name']}_best_seg.pt")
    if not os.path.exists(weight_path):
        print(f"[ERROR] Missing weights: {weight_path}")
        return

    # Load state dict (UNCHANGED)
    state_dict = torch.load(weight_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()

    # Use the dedicated TestSegmentationDataset
    dataset = TestSegmentationDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=12)

    # Save to a dedicated test prediction directory
    save_dir = os.path.join(config["project_dir"], "seg_test_predictions")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (img_tensor_list, target_list) in enumerate(loader):
            samples_nested = nested_tensor_from_tensor_list(img_tensor_list)
            samples = samples_nested.to(device)
            
            with autocast():
                outputs = model(samples)
            
            target_dict = target_list[0] 
            orig_target_sizes = target_dict["orig_size"].unsqueeze(0).to(device)
            max_h, max_w = samples_nested.tensors.shape[-2:]
            max_target_sizes = torch.as_tensor([[max_h, max_w]], device=device)

            pred_list = postprocessors["segm"]([outputs], target_sizes=orig_target_sizes, max_target_sizes=max_target_sizes)
            pred = pred_list[0]

            pred_mask = (pred["masks"][0].cpu().numpy() > 0.5).astype(np.uint8)
            mask_img = (pred_mask * 255).astype(np.uint8)

            # Use the filename from the target dictionary for saving
            file_identifier = target_dict["file_identifier"]
            if file_identifier.endswith(".jpg"): 
                file_identifier = file_identifier.replace(".jpg", ".png")
                
            save_path = os.path.join(save_dir, file_identifier)
            cv2.imwrite(save_path, mask_img)

    print(f"[DETR-SEG] ✔ Segmentation inference completed on TEST set → {save_dir}\n")


# ============================================================
# VISUALIZATION (RENAMED to visualize_predictions)
# Uses TEST set predictions
# ============================================================

def visualize_predictions(config): # Renamed from visualize_segmentation
    print("[DETR-SEG] 🎨 Visualizing segmentation masks for TEST set...")

    # Directories
    pred_dir = os.path.join(config["project_dir"], "seg_test_predictions") # Read from test prediction dir
    out_dir = os.path.join(config["project_dir"], "seg_visualized_test") # Save to test visualization dir
    os.makedirs(out_dir, exist_ok=True)
    
    # Corrected path to source images
    image_dir = os.path.join("segmentation_test", "images") 

    # Load the test dataset for file paths
    dataset = TestSegmentationDataset()
    
    # Iterate over files in the dataset
    for i in range(len(dataset)):
        
        # Get raw image path/name from dataset
        img_name = dataset.image_files[i]
        img_path = os.path.join(image_dir, img_name)
        
        if not os.path.exists(img_path): continue

        img = cv2.imread(img_path)
        if img is None: continue

        mask_name = img_name.replace(".jpg", ".png") # Mask file is a PNG prediction
        mask_path = os.path.join(pred_dir, mask_name)
        
        if not os.path.exists(mask_path): continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure mask is the same size as the original image
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) 
        
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        # Blend only on the foreground pixels
        alpha = 0.5
        mask_bool = mask > 0
        mask_3ch_bool = np.dstack([mask_bool] * 3)

        blended_part = cv2.addWeighted(img, 1.0 - alpha, mask_colored, alpha, 0)

        overlay = img.copy() 
        overlay[mask_3ch_bool] = blended_part[mask_3ch_bool]
        
        save_path = os.path.join(out_dir, f"seg_vis_{img_name}")
        cv2.imwrite(save_path, overlay)

    print(f"[DETR-SEG] ✔ Saved segmented overlays for TEST set → {out_dir}\n")