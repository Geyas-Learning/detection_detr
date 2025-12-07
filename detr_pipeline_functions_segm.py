import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils_segm import (
    SpacecraftSegmDataset,
    DATA_ROOT,
    SEG_NUM_CLASSES,
    detr_collate_fn_segm,
)

import numpy as np
import cv2
from PIL import Image

from util.misc import nested_tensor_from_tensor_list
from detr_model.__init__ import build_segmentation_model as build_model
from detr_model.segmentation import PostProcessSegm

class Args:
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_file = "custom"
        self.frozen_weights = None
        self.num_classes = SEG_NUM_CLASSES  # 2 classes: body, panel

        self.masks = True
        self.mask_loss_coef = 2.0
        self.dice_loss_coef = 2.0

        self.lr = 5e-5
        self.lr_backbone = 1e-5
        self.weight_decay = 1e-4
        self.num_queries = 100
        self.epochs = config["epochs"]
        self.batch_size = config["batch"]
        self.hidden_dim = 256
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 2048
        self.enc_layers = 6
        self.dec_layers = 6
        self.pre_norm = False
        self.aux_loss = True
        self.bbox_loss_coef = 2.0
        self.giou_loss_coef = 1.0
        self.eos_coef = 0.1

        self.backbone = "resnet34"
        self.position_embedding = "sine"
        self.dilation = False
        self.set_cost_class = 1.0
        self.set_cost_bbox = 5.0
        self.set_cost_giou = 2.0

def train_model(config):
    args = Args(config)

    train_csv = config["train_csv"]
    val_csv = config["val_csv"]

    train_dataset = SpacecraftSegmDataset(train_csv, split="train", root_dir=DATA_ROOT)
    val_dataset = SpacecraftSegmDataset(val_csv, split="val", root_dir=DATA_ROOT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,          # e.g. 4 on GPU
        collate_fn=detr_collate_fn_segm,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=detr_collate_fn_segm,
        shuffle=False,
        num_workers=4,
    )

    device = torch.device(args.device)
    model, criterion, _ = build_model(args)
    model.to(device)
    criterion.to(device)

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
    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # -------- gradient accumulation settings --------
    # effective_batch_size = args.batch_size * accum_steps
    accum_steps = max(1, int(16 // args.batch_size))  # try to approximate effective batch 16
    print(f"[TRAIN] Segmentation training on {args.device} for {args.epochs} epochs "
          f"(batch={args.batch_size}, accum_steps={accum_steps}, effective_batch≈{args.batch_size * accum_steps})")

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # ---------- TRAIN ----------
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        step_in_epoch = 0

        for step, (samples, targets) in enumerate(train_loader):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )

            # scale for accumulation
            loss = loss / accum_steps
            loss.backward()
            total_train_loss += loss.item()
            step_in_epoch += 1

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # final partial step
        if step_in_epoch % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / max(1, step_in_epoch)

        # ---------- VALIDATION ----------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for samples, targets in val_loader:
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(samples)
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(
                    loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                )
                total_val_loss += losses.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))

        print(f"[EPOCH {epoch+1}/{args.epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(config["project_dir"], f'{config["run_name"]}_best.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"[SAVE] Model saved with new best val loss: {best_val_loss:.4f}")


def test_model(config):
    args = Args(config)
    device = torch.device(args.device)
    model, _, _ = build_model(args)

    weights_path = os.path.join(
        config["project_dir"], f'{config["run_name"]}_best.pth'
    )
    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights not found: {weights_path}")
        return

    state_dict = torch.load(weights_path, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_img_dir = config["test_images_full_path"]  # "./segmentation_test/images"
    image_files = [
        f for f in os.listdir(test_img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    test_set = [(os.path.join(test_img_dir, f), f) for f in image_files]

    postprocessor = PostProcessSegm()
    output_dir = os.path.join(config["project_dir"], "predicted_masks")
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"[INFERENCE] Starting inference on {len(test_set)} images. "
        f"Saving masks to {output_dir}..."
    )

    with torch.no_grad():
        for img_path, file_name in test_set:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            orig_size = torch.as_tensor([int(h), int(w)], device=device)

            img_tensor = (
                torch.as_tensor(np.array(img), dtype=torch.float32)
                .permute(2, 0, 1) / 255.0
            )
            samples = nested_tensor_from_tensor_list([img_tensor]).to(device)

            outputs = model(samples)
            # build dummy results list to pass through PostProcessSegm
            results = [{"scores": outputs["pred_logits"][0].softmax(-1).max(-1).values,
                        "labels": outputs["pred_logits"][0].softmax(-1).argmax(-1)}]

            orig_target_sizes = orig_size.unsqueeze(0)
            max_h, max_w = samples.tensors.shape[-2:]
            max_target_sizes = torch.as_tensor([[max_h, max_w]], device=device)

            results = postprocessor(results, outputs, orig_target_sizes, max_target_sizes)

            # Combine body/panel into color mask for saving
            masks = results[0]["masks"]  # (num_queries,1,H,W)
            labels = results[0]["labels"]
            scores = results[0]["scores"]

            body_mask = np.zeros((h, w), dtype=np.uint8)
            panel_mask = np.zeros((h, w), dtype=np.uint8)

            for q in range(len(labels)):
                cls = int(labels[q])
                if cls >= SEG_NUM_CLASSES:
                    continue
                m = masks[q, 0].cpu().numpy()
                if cls == 0:
                    body_mask = np.logical_or(body_mask, m).astype(np.uint8)
                elif cls == 1:
                    panel_mask = np.logical_or(panel_mask, m).astype(np.uint8)

            out_img = np.zeros((h, w, 3), dtype=np.uint8)
            out_img[body_mask == 1] = [0, 0, 255]   # body = red
            out_img[panel_mask == 1] = [255, 0, 0]  # panel = blue

            out_path = os.path.join(
                output_dir, file_name.replace("_img.jpg", "_layer.jpg")
            )
            cv2.imwrite(out_path, out_img)

    print(
        "[INFERENCE] Inference complete. Use your metric script to compute IoU thresholds."
    )

def visualize_predictions(config):
    """
    Placeholder visualization function for the segmentation pipeline.

    This implementation does not create any visual overlays. It only reminds you
    where the predicted mask images are stored after running test_model(config).
    """
    output_dir = os.path.join(config["project_dir"], "predicted_masks")
    print("[VISUALIZATION] Skipped for segmentation pipeline.")
    print(f"[VISUALIZATION] Check predicted mask PNGs in: {output_dir}")

