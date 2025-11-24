import time
from ..util.coco_eval import CocoEvaluator             
from .coco_custom import CocoDetection
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from ..util import box_ops
import numpy as np

"""
=========================================================
DETR TRAINING / EVALUATION / INFERENCE PIPELINE (FINAL)
=========================================================
"""

import os
import io
import cv2
import torch
import torch.optim as optim
import torchvision.transforms as T
import pandas as pd

from PIL import Image
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ============================================================
# INTERNAL IMPORTS
# ============================================================

from .data_utils import (
    SpacecraftDataset,
    TestSpacecraftDataset,
    DATA_ROOT,
    NUM_CLASSES,
    CLASS_NAMES,
    detr_collate_fn,
    get_detr_transforms
)

from ..detr_model.detr import SetCriterion, PostProcess     
from ..detr_model.matcher import HungarianMatcher            
from ..detr_model.__init__ import build_model                
from .util.misc import nested_tensor_from_tensor_list      

# ============================================================
# HELPER FUNCTIONS (FOR DDP AWARENESS)
# ============================================================

def is_main_process():
    """Returns True if the current process is the main process (Rank 0)"""
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

# ============================================================
# DETR ARGUMENT WRAPPER
# ============================================================

class Args:
    """Wrapper replicating DETR argument interface."""

    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_file = "coco"
        self.frozen_weights = None

        self.num_classes = config.get("num_classes", NUM_CLASSES)

        # Backbone config
        self.backbone = "resnet34"
        self.dilation = False
        self.position_embedding = "sine"
        self.masks = False

        # Transformer config
        self.num_queries = 50
        self.hidden_dim = 256
        self.dropout = 0.3
        self.nheads = 8
        self.dim_feedforward = 2048
        self.enc_layers = 6
        self.dec_layers = 6
        self.pre_norm = False

        #self.aux_loss = True
        self.aux_loss = False

        # Loss weights
        self.set_cost_class = 1
        self.set_cost_bbox = 7
        self.set_cost_giou = 2
        self.bbox_loss_coef = 7
        self.giou_loss_coef = 2
        self.eos_coef = 0.2

        # Learning rate
        self.lr_backbone = 1e-5

        # Gradient clipping
        self.clip_max_norm = 0.1


# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(config):
    # ====================================================
    # 1. DDP INITIALIZATION & DEVICE SETUP (MUST BE FIRST)
    # ====================================================
    is_ddp_run = "LOCAL_RANK" in os.environ
    
    if is_ddp_run:
        local_rank = int(os.environ["LOCAL_RANK"])
        # Initialize the process group
        dist.init_process_group(backend='nccl', init_method='env://')
        # Set the default device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        
        if is_main_process():
            print(f"\n[DDP] Initializing DDP with {world_size} processes.")
            print(f"[DDP] Primary process (Rank 0) on device: {device}")
    else:
        # Fallback for single-GPU testing without torchrun
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        if is_main_process():
            print(f"\n[DDP] DDP not detected. Running single-process on device: {device}")
            
    if is_main_process():
        print(f"\n[DETR] 🧠 Starting Training...")


    # Build DETR Arguments
    args = Args(config)
    args.num_classes = NUM_CLASSES

    # ====================================================
    # 2. MODEL, CRITERION, AND DDP WRAPPER (ONCE)
    # ====================================================
    model, criterion, postprocessors = build_model(args)
    
    # Move modules to the assigned GPU
    model.to(device)
    criterion.to(device)
    
    # Wrap the model in DistributedDataParallel (DDP)
    if is_ddp_run:
        # DDP handles moving gradients and syncing weights
        model = DDP(model, device_ids=[local_rank])
    # model_without_ddp is the unwrapped model for saving/uncommon cases
    model_without_ddp = model.module if is_ddp_run else model 

    # ====================================================
    # 3. DATA LOADERS with DISTRIBUTED SAMPLER
    # ====================================================
    train_set = SpacecraftDataset(
        csv_file=config["train_csv"],
        root_dir=DATA_ROOT,
        transform=get_detr_transforms(),
    )

    val_set = SpacecraftDataset(
        csv_file=config["val_csv"],
        root_dir=DATA_ROOT,
        transform=get_detr_transforms(),
    )
    
    # Training Data Loader with DistributedSampler (required for correct shuffling)
    if is_ddp_run:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_set, shuffle=False)
        
        train_loader = DataLoader(
            train_set,
            batch_size=config["batch"],
            collate_fn=detr_collate_fn,
            num_workers=os.cpu_count() // world_size,
            sampler=train_sampler # Use the sampler instead of shuffle
        )
        val_loader = DataLoader(
            val_set,
            batch_size=config["batch"],
            collate_fn=detr_collate_fn,
            num_workers=os.cpu_count() // world_size,
            sampler=val_sampler # Use the sampler
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=config["batch"],
            collate_fn=detr_collate_fn,
            num_workers=os.cpu_count(),
            shuffle=True
        )
        val_loader = DataLoader(
            val_set,
            batch_size=config["batch"],
            collate_fn=detr_collate_fn,
            num_workers=os.cpu_count(),
            shuffle=False
        )


    # Optimizer (Use the DDP-wrapped model's parameters)
    param_dicts = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = optim.AdamW(param_dicts, lr=5e-4, weight_decay=1e-4)

    # Logging setup (only the RANK 0 process should log/write files)
    log_dir = os.path.join(config["project_dir"], "tensorboard")
    if is_main_process():
        writer = SummaryWriter(log_dir=log_dir)
        # Setup CSV Metrics Logger (Only rank 0 writes the file)
        log_path = os.path.join(config["project_dir"], f"{config['run_name']}_results.csv")
        csv_header = "epoch,time,train/loss_ce,train/loss_bbox,train/loss_giou,val/loss_total,metrics/mAP50,metrics/mAP50-95,metrics/AR100,lr/backbone,lr/head"
        with open(log_path, "w") as f:
            f.write(csv_header + "\n")
        print(f"[LOG] Created metrics log file: {log_path}")
    else:
        writer = None # Avoid logging on non-rank 0 processes
        log_path = None
    
    # ----------------------------------------------------------
    # EPOCH LOOP
    # ----------------------------------------------------------
    best_loss = float("inf")
    if is_main_process():
        print(f"[DETR] Training for {config['epochs']} epochs...\n")

    for epoch in range(config["epochs"]):
        
        epoch_start_time = time.time()

        # Initialize dictionary to track individual loss components for the epoch
        total_losses_epoch = {k: 0.0 for k in criterion.weight_dict.keys()}
        total_train_loss = 0.0
        
        # ⭐️ DDP: Set epoch for Distributed Sampler to ensure shuffling is correct
        if is_ddp_run:
            train_loader.sampler.set_epoch(epoch)
        
        # ----------------------------
        # TRAINING
        # ----------------------------
        model.train()

        for samples, targets in train_loader:

            optimizer.zero_grad() # Clear gradients

            # Ensure data is moved to the correct device
            # This handles both NestedTensor (DETR default) and list of Tensors
            if isinstance(samples, (list, torch.Tensor)):
                samples = nested_tensor_from_tensor_list(samples)
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = model(samples)

            # Loss calculation
            loss_dict = criterion(outputs, targets)
            weights = criterion.weight_dict
            
            # Calculate the total weighted loss 
            loss = sum(loss_dict[k] * weights[k] for k in loss_dict.keys() if k in weights)
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                
            optimizer.step()
            
            # Accumulate losses (accumulate the batch loss)
            total_train_loss += loss.item()
            for k in total_losses_epoch.keys():
                if k in loss_dict:
                    total_losses_epoch[k] += loss_dict[k].item()
            
        # ----------------------------------------------------
        # DDP LOSS REDUCTION & AVERAGING (AFTER TRAINING EPOCH)
        # ----------------------------------------------------
        num_batches = len(train_loader)
        
        if is_ddp_run:
            # Manually reduce losses across processes
            total_train_loss_tensor = torch.tensor(total_train_loss, device=device)
            dist.all_reduce(total_train_loss_tensor, op=dist.ReduceOp.SUM)
            total_train_loss = total_train_loss_tensor.item()
            
            for k in total_losses_epoch.keys():
                loss_tensor = torch.tensor(total_losses_epoch[k], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                total_losses_epoch[k] = loss_tensor.item()
                
            # Total number of effective batches processed across all ranks
            effective_batches = num_batches * world_size 
        else:
            effective_batches = num_batches
            
        # Calculate final averages (must be done on all ranks before barrier/main_process check)
        avg_train_loss = total_train_loss / effective_batches
        avg_loss_ce = total_losses_epoch['loss_ce'] / effective_batches
        avg_loss_bbox = total_losses_epoch['loss_bbox'] / effective_batches
        avg_loss_giou = total_losses_epoch['loss_giou'] / effective_batches

        # ----------------------------------------------------
        # DDP BARRIER: Ensure all processes finish training before validation/saving
        # ----------------------------------------------------
        if is_ddp_run:
            dist.barrier()
        
        # ----------------------------
        # VALIDATION & METRICS (Only on Rank 0)
        # ----------------------------
        
        if is_main_process():
            
            # Use the unwrapped model for evaluation
            model_to_evaluate = model_without_ddp
            model_to_evaluate.eval()
            
            # 1. Validation Loss Calculation
            avg_val_loss = 0.0
            val_num_batches = len(val_loader)
            with torch.no_grad():
                for samples_val, targets_val in val_loader:
                    if isinstance(samples_val, (list, torch.Tensor)):
                        samples_val = nested_tensor_from_tensor_list(samples_val)
                    samples_val = samples_val.to(device)
                    targets_val = [{k: v.to(device) for k, v in t.items()} for t in targets_val]
                    
                    outputs_val = model_to_evaluate(samples_val)
                    loss_dict_val = criterion(outputs_val, targets_val)
                    weights_val = criterion.weight_dict
                    
                    loss_val = sum(loss_dict_val[k] * weights_val[k] for k in loss_dict_val.keys() if k in weights_val)
                    
                    avg_val_loss += loss_val.item()
                        
            avg_val_loss /= val_num_batches

            # 2. COCO mAP Evaluation Setup (Same as evaluate_map logic)
            ann_file = config["coco_val_json"]
            img_root = os.path.join(DATA_ROOT, "images")
            
            coco_dataset = CocoDetection(img_folder=img_root, ann_file=ann_file)
            
            def coco_collate(batch):
                imgs, targets = zip(*batch)
                return list(imgs), list(targets)

            # Use validation batch size for COCO loader
            coco_loader = DataLoader(
                coco_dataset,
                batch_size=config["batch"], 
                shuffle=False,
                collate_fn=coco_collate,
                num_workers=4
            )
            
            evaluator = CocoEvaluator(coco_dataset.coco, ["bbox"])

            # COCO Evaluation loop
            with torch.no_grad():
                for imgs, targets in coco_loader:
                    # Prepare inputs
                    imgs_tensor = [T.ToTensor()(img).to(device) for img in imgs]
                    samples_coco = nested_tensor_from_tensor_list(imgs_tensor)

                    outputs_coco = model_to_evaluate(samples_coco)

                    sizes = torch.tensor(
                        [[t["orig_size"][0], t["orig_size"][1]] for t in targets],
                        dtype=torch.float32,
                        device=device
                    )

                    results = postprocessors["bbox"](outputs_coco, sizes)

                    mapped = {
                        t["image_id"].item(): r
                        for t, r in zip(targets, results)
                    }

                    evaluator.update(mapped)

            # Summarize metrics
            evaluator.synchronize_between_processes()
            evaluator.accumulate()
            evaluator.summarize()
            stats = evaluator.coco_eval["bbox"].stats

            mAP_50_95 = stats[0]
            mAP_50 = stats[1]
            AR_100 = stats[8] # Assuming AR_100 corresponds to AR max detections

            # Reset model to train mode after validation
            model.train()
            
            # ----------------------------
            # LOGGING & SAVING (Only on Rank 0)
            # ----------------------------
            
            # Get Learning Rates
            lr_backbone = optimizer.param_groups[1]["lr"]
            lr_head = optimizer.param_groups[0]["lr"]

            # Use the calculated average losses and metrics
            epoch_time = time.time() - epoch_start_time

            # Log to TensorBoard
            writer.add_scalar("train/loss_total", avg_train_loss, epoch + 1)
            writer.add_scalar("train/loss_ce", avg_loss_ce, epoch + 1)
            writer.add_scalar("train/loss_bbox", avg_loss_bbox, epoch + 1)
            writer.add_scalar("train/loss_giou", avg_loss_giou, epoch + 1)
            writer.add_scalar("val/loss_total", avg_val_loss, epoch + 1)
            writer.add_scalar("metrics/mAP50", mAP_50, epoch + 1)
            writer.add_scalar("metrics/mAP50-95", mAP_50_95, epoch + 1)
            writer.add_scalar("metrics/AR100", AR_100, epoch + 1)
            writer.add_scalar("lr/backbone", lr_backbone, epoch + 1)
            writer.add_scalar("lr/head", lr_head, epoch + 1)


            # Format the log line
            log_line = f"{epoch+1},{epoch_time:.2f},{avg_loss_ce:.5f},{avg_loss_bbox:.5f},{avg_loss_giou:.5f},{avg_val_loss:.5f},{mAP_50:.4f},{mAP_50_95:.4f},{AR_100:.4f},{lr_backbone:.8f},{lr_head:.8f}"
            
            # Write to CSV
            with open(log_path, "a") as f:
                f.write(log_line + "\n")
                
            print(
                f"Epoch [{epoch+1}/{config['epochs']}] "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"mAP@.5: {mAP_50:.3f}"
            )
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss

                save_dir = os.path.join(config["project_dir"], "weights")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{config['run_name']}_best.pt")

                # ⭐️ DDP FIX: Use model_without_ddp (which is model.module) for saving
                torch.save(model_without_ddp.state_dict(), save_path)

                print(f"  ✔ Saved best model → {save_path}")

    
    if is_main_process() and writer:
        writer.close()
    
    # ----------------------------------------------------
    # DDP CLEANUP (MUST BE LAST)
    # ----------------------------------------------------
    if is_ddp_run:
        dist.destroy_process_group()
    
    if is_main_process():
        print("\n[DETR] ✔ Training Completed.\n")


# ============================================================
# CLASSIFICATION METRICS (Confusion Matrix and Report)
# ============================================================

def generate_classification_metrics(config):
    if not is_main_process():
        return

    print("\n[DETR] 🔢 Generating Classification Report & Confusion Matrix...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and weights
    args = Args(config)
    model, _, postprocessors = build_model(args)
    model.to(device)
    
    weight_file = os.path.join(
        config["project_dir"], "weights", f"{config['run_name']}_best.pt"
    )
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.eval()

    # Data
    val_set = SpacecraftDataset(
        csv_file=config["val_csv"],
        root_dir=DATA_ROOT,
        transform=get_detr_transforms(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch"],
        shuffle=False,
        collate_fn=detr_collate_fn,
        num_workers=4,
    )

    all_targets = []
    all_predictions = []
    
    # Thresholds for classification analysis
    CONF_THRESHOLD = 0.5 
    IOU_THRESHOLD = 0.5 

    with torch.no_grad():
        for samples, targets in val_loader:
            # Ensure data is handled correctly (using nested_tensor_from_tensor_list)
            if isinstance(samples, (list, torch.Tensor)):
                samples = nested_tensor_from_tensor_list(samples)
            samples = samples.to(device)
            targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)

            
            
            
            
            # 1. Convert the list of target dictionaries (targets_gpu) into a single tensor of original sizes
            target_sizes = torch.stack([t["orig_size"] for t in targets_gpu], dim=0)

            # 2. Pass the correctly formatted tensor (target_sizes) to the post-processor
            results = postprocessors['bbox'](outputs, target_sizes) 

            for target, result in zip(targets, results):
                gt_boxes_norm = target["boxes"].cpu()
                gt_labels = target["labels"].cpu().tolist()
                
                pred_scores = result["scores"].cpu()
                pred_boxes_norm = result["boxes"].cpu()
                pred_labels = result["labels"].cpu()

                # Filter predictions by confidence threshold
                high_conf_indices = pred_scores > CONF_THRESHOLD
                pred_boxes_filt = pred_boxes_norm[high_conf_indices]
                pred_labels_filt = pred_labels[high_conf_indices]
                
                if len(gt_labels) == 0:
                    continue

                # Convert normalized (cx, cy, w, h) to (xmin, ymin, xmax, ymax) format for IoU
                gt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes_norm)
                pred_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes_filt)

                if pred_boxes_xyxy.numel() == 0:
                    # If no predictions, all GT boxes are False Negatives (FN)
                    for _ in gt_labels:
                        all_targets.append(_)
                        all_predictions.append(NUM_CLASSES) # Background/FN class
                    continue

                # Calculate IoU between all GT and filtered predictions (IoU matrix shape: (num_gt, num_pred_filtered))
                iou_matrix = box_ops.box_iou(gt_boxes_xyxy, pred_boxes_xyxy)[0]

                # Match each GT box to its best prediction (if IoU > threshold)
                for gt_idx in range(len(gt_labels)):
                    best_iou, best_pred_idx = iou_matrix[gt_idx].max(dim=0)

                    all_targets.append(gt_labels[gt_idx])

                    if best_iou >= IOU_THRESHOLD:
                        # Matched prediction: use its label
                        predicted_label = pred_labels_filt[best_pred_idx].item()
                        all_predictions.append(predicted_label)
                    else:
                        # No match or IoU too low: False Negative
                        all_predictions.append(NUM_CLASSES) # Background/FN class
                        
    # Adjusting CLASS_NAMES for the report (labels 0-9 + 10 for background)
    report_class_names = CLASS_NAMES + ["Background/FN"]
    
    # 1. Classification Report
    report = classification_report(
        all_targets, 
        all_predictions, 
        target_names=report_class_names, 
        labels=list(range(NUM_CLASSES + 1)), # Include the 'Background/FN' class
        zero_division=0
    )
    
    report_path = os.path.join(
        config["project_dir"], 
        f"{config['run_name']}_classification_report.txt"
    )
    with open(report_path, "w") as f:
        f.write(f"Classification Report (Confidence > {CONF_THRESHOLD}, IoU > {IOU_THRESHOLD}):\n\n")
        f.write(report)
        
    print(f"[INFO] Saved classification report → {report_path}")
    
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_predictions, labels=list(range(NUM_CLASSES + 1)))

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=report_class_names,
        yticklabels=report_class_names,
        title='Confusion Matrix',
        ylabel='True label',
        xlabel='Predicted label'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    cm_path = os.path.join(
        config["project_dir"], 
        f"{config['run_name']}_confusion_matrix.png"
    )
    plt.savefig(cm_path)
    plt.close(fig) # Close figure to free memory
    
    print(f"[INFO] Saved confusion matrix plot → {cm_path}")
        
    print("[DETR] ✔ Classification metrics completed.")


# ============================================================
# INFERENCE — TEST IMAGES
# ============================================================

def test_model(config):
    if not is_main_process():
        return
        
    print("\n[DETR] 🔍 Running Inference...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = Args(config)
    args.num_classes = NUM_CLASSES

    model, _, postprocessors = build_model(args)
    model.to(device)

    weight_file = os.path.join(
        config["project_dir"], "weights", f"{config['run_name']}_best.pt"
    )

    if not os.path.exists(weight_file):
        print(f"[ERROR] Missing weight file: {weight_file}")
        return

    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.eval()

    print(f"[INFO] Loaded weights: {weight_file}")

    # Test dataset
    test_set = TestSpacecraftDataset(
        img_dir=config["test_images_full_path"],
        transform=get_detr_transforms()
    )

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    results = []

    with torch.no_grad():
        for img_tensor, name, h, w in test_loader:
            img_tensor = img_tensor.to(device)

            # DETR input format
            samples = nested_tensor_from_tensor_list([img_tensor.squeeze(0)])
            samples = samples.to(device)

            outputs = model(samples)

            # Extract scalar values
            orig_sizes = torch.tensor([[h.item(), w.item()]], dtype=torch.float32, device=device) 
            
            # Access the first item in the batch (which is size 1)
            pred = postprocessors["bbox"](outputs, orig_sizes)[0]

            scores = pred["scores"]
            labels = pred["labels"]
            boxes = pred["boxes"]

            # Handle Zero Detections
            if scores.numel() == 0:
                print(f"[WARN] No predictions found for image: {name[0]}")
                # Log a "no detection" entry
                results.append({
                    "Image name": name[0],
                    "Class": "N/A",
                    "Bounding box": "(0, 0, 0, 0)"
                })
                continue

            # Select best prediction
            best = int(torch.argmax(scores))
            
            current_name = name[0] 

            # Convert prediction back to original image coordinates (xmin, ymin, xmax, ymax)
            # The boxes from postprocessors are already scaled to orig_size
            xmin, ymin, xmax, ymax = boxes[best].cpu().round().int().tolist()

            cls_id = labels[best].item()
            cls_name = CLASS_NAMES[cls_id]

            # Clip coordinates to image bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(xmax, w.item())
            ymax = min(ymax, h.item())

            bbox_str = f"({xmin}, {ymin}, {xmax}, {ymax})"

            results.append({
                "Image name": current_name,
                "Class": cls_name,
                "Bounding box": bbox_str
            })

    out_csv = os.path.join(
        config["project_dir"],
        f"{config['run_name']}_test_predictions.csv"
    )

    pd.DataFrame(results).to_csv(out_csv, index=False)

    print(f"[DETR] ✔ Inference complete. Saved → {out_csv}\n")


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_predictions(config):
    if not is_main_process():
        return
        
    print("[VISUALIZER] Rendering bounding boxes...")

    csv_path = os.path.join(
        config["project_dir"],
        f"{config['run_name']}_test_predictions.csv"
    )

    if not os.path.exists(csv_path):
        print(f"[ERROR] Missing predictions CSV: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    out_dir = os.path.join(config["project_dir"], "predictions_visualized")
    os.makedirs(out_dir, exist_ok=True)

    for _, row in df.iterrows():
        img_name = row["Image name"]
        cls = row["Class"]
        bbox = row["Bounding box"]

        img_path = os.path.join(config["test_images_full_path"], img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        try:
            vals = bbox.strip("()").split(",")
            xmin, ymin, xmax, ymax = [int(v) for v in vals]
        except:
            continue

        if cls != "N/A":
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, cls, (xmin, max(0, ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

        save_path = os.path.join(out_dir, img_name)
        cv2.imwrite(save_path, img)

    print(f"[VISUALIZER] ✔ Saved annotated outputs → {out_dir}\n")


# ============================================================
# COCO mAP EVALUATION (USING coco_custom.py)
# ============================================================

def evaluate_map(config):
    if not is_main_process():
        return
        
    print("\n[DETR] 📊 Running COCO mAP Evaluation...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = Args(config)
    args.num_classes = NUM_CLASSES

    model, _, postprocessors = build_model(args)
    model.to(device)

    weights = os.path.join(
        config["project_dir"], "weights", f"{config['run_name']}_best.pt"
    )

    if not os.path.exists(weights):
        print(f"[ERROR] Missing: {weights}")
        return

    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    print(f"[INFO] Loaded weights: {weights}")

    # Load COCO-format val JSON
    ann_file = config["coco_val_json"]
    img_root = os.path.join(DATA_ROOT, "images")

    print(f"[EVAL] Using COCO annotation file: {ann_file}")
    print(f"[EVAL] Using image folder: {img_root}")

    coco_dataset = CocoDetection(
        img_folder=img_root,
        ann_file=ann_file
    )

    def coco_collate(batch):
        imgs, targets = zip(*batch)
        return list(imgs), list(targets)

    # Note: Using a batch DataLoader
    data_loader = DataLoader(
        coco_dataset,
        batch_size=config["batch"],
        shuffle=False,
        collate_fn=coco_collate
    )

    evaluator = CocoEvaluator(coco_dataset.coco, ["bbox"])

    # Main evaluation loop
    for imgs, targets in data_loader:

        imgs_tensor = [T.ToTensor()(img).to(device) for img in imgs]
        samples = nested_tensor_from_tensor_list(imgs_tensor)

        with torch.no_grad():
            outputs = model(samples)

        sizes = torch.tensor(
            [[t["orig_size"][0], t["orig_size"][1]] for t in targets],
            dtype=torch.float32,
            device=device
        )

        results = postprocessors["bbox"](outputs, sizes)

        mapped = {
            t["image_id"].item(): r
            for t, r in zip(targets, results)
        }

        evaluator.update(mapped)

    # Since we only run this on Rank 0, these sync/accumulate/summarize calls are local to the main process.
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.coco_eval["bbox"].stats

    mAP_50_95 = stats[0]
    mAP_50 = stats[1]
    
    # SIZE-SPECIFIC mAP
    mAP_small = stats[3]
    mAP_medium = stats[4]
    mAP_large = stats[5]
    
    AR = stats[8]

    print("========== FINAL COCO METRICS ==========")
    print(f"mAP@[0.50:0.95]: {mAP_50_95:.4f}")
    print(f"mAP@0.50:       {mAP_50:.4f}")
    print(f"mAP (Small):    {mAP_small:.4f}")
    print(f"mAP (Medium):   {mAP_medium:.4f}")
    print(f"mAP (Large):    {mAP_large:.4f}")
    print(f"AR@[0.50:0.95]: {AR:.4f}\n")

    # Log to TensorBoard
    writer = SummaryWriter(os.path.join(config["project_dir"], "tensorboard"))
    writer.add_scalar("coco/mAP_50_95", mAP_50_95)
    writer.add_scalar("coco/mAP_50", mAP_50)
    writer.add_scalar("coco/AR", AR)
    
    # Log size-specific mAP
    writer.add_scalar("coco/mAP_small", mAP_small)
    writer.add_scalar("coco/mAP_medium", mAP_medium)
    writer.add_scalar("coco/mAP_large", mAP_large)
    
    writer.close()

    print("[DETR] ✔ Logged metrics to TensorBoard.\n")