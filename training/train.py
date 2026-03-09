"""
train.py — Main Training Loop (Kaggle-Ready)
=============================================
Trains the FibrilMask2Former pipeline on the FibrilSynth dataset.

Features:
  - Mixed precision (torch.cuda.amp) — MANDATORY on T4/P100
  - Gradient accumulation  (simulates batch_size=8 on single GPU)
  - Gradient checkpointing in backbone (saves ~30% VRAM)
  - Three-phase training: warmup → joint finetune → fine-detail
  - Saves best checkpoint by val loss
  - Optional W&B logging
  - Resume from checkpoint

Run locally (CPU debug, no GPU needed):
    python training/train.py --config configs/debug.yaml

Run on Kaggle (GPU, full training):
    python training/train.py --config configs/kaggle.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Project root on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.dataset import get_dataloaders
from models.mask2former import build_model
from training.losses import Mask2FormerLoss, HungarianMatcher


# ─────────────────────────────────────────────────────────────────────────────
# Default Configs
# ─────────────────────────────────────────────────────────────────────────────

DEBUG_CONFIG = {
    # Quick sanity check on CPU — 2 images, 2 epochs
    # NOTE: Swin-T requires image_size divisible by (patch_size × window_size) = 4×7 = 28
    # Valid sizes: 224 (8×28), 448 (16×28). 256 is NOT valid — causes assertion error.
    "data_root": "data/synthetic",
    "image_size": 224,
    "batch_size": 1,
    "accum_steps": 1,
    "num_workers": 0,
    "epochs": 2,
    "warmup_epochs": 1,

    "backbone": "swin_tiny",
    "pretrained": False,      # False for CPU debug (skip download)
    "num_queries": 10,
    "hidden_dim": 256,
    "nheads": 4,
    "dec_layers": 2,
    "num_classes": 1,

    "lr_backbone": 1e-4,
    "lr_head": 1e-3,
    "weight_decay": 0.05,
    "grad_clip": 1.0,

    "loss_weights": {"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0},

    "checkpoint_dir": "checkpoints",
    "use_wandb": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mixed_precision": torch.cuda.is_available(),
    "seed": 42,
}

KAGGLE_CONFIG = {
    # Full training on Kaggle T4/P100 (~10-14 hrs, high quality)
    "data_root": "/kaggle/input/fibrilsynth/data/synthetic",
    "image_size": 512,
    "batch_size": 1,
    "accum_steps": 8,
    "num_workers": 2,
    "epochs": 30,
    "warmup_epochs": 5,
    "backbone": "swin_tiny",
    "pretrained": True,
    "num_queries": 50,
    "hidden_dim": 256,
    "nheads": 8,
    "dec_layers": 6,
    "num_classes": 1,
    "lr_backbone": 1e-5,
    "lr_head": 1e-4,
    "weight_decay": 0.05,
    "grad_clip": 0.01,
    "loss_weights": {"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0},
    "checkpoint_dir": "/kaggle/working/checkpoints",
    "use_wandb": False,
    "device": "cuda",
    "mixed_precision": True,
    "seed": 42,
}

COLAB_FAST_CONFIG = {
    # ─── GOOGLE COLAB FREE T4 — target ~90 min training ───────────────────────
    # Strategy:
    #   • Smaller image (256px vs 512px)  → 4× faster per step
    #   • Fewer queries (25 vs 50)        → 2× faster transformer
    #   • Fewer decoder layers (3 vs 6)   → 2× faster forward pass
    #   • 200 training images, 10 epochs  → ~90 min on T4
    #   • Still produces useful model for demonstration / research paper
    # ─────────────────────────────────────────────────────────────────────────
    "data_root": "/content/fibril_data/data/synthetic",   # Colab path (see notebook)
    "image_size": 224,          # Swin-T native size — must be multiple of 28
    "batch_size": 2,            # Can fit 2 on T4 at 256px
    "accum_steps": 4,           # Effective batch = 8
    "num_workers": 2,
    "epochs": 10,               # 10 epochs × ~9min/epoch ≈ 90 min total
    "warmup_epochs": 2,

    "backbone": "swin_tiny",
    "pretrained": True,         # ImageNet weights → faster convergence
    "num_queries": 25,          # 25 instead of 50 → 2× faster decoder
    "hidden_dim": 256,
    "nheads": 8,
    "dec_layers": 3,            # 3 instead of 6 → 2× faster transformer
    "num_classes": 1,

    "lr_backbone": 5e-5,        # Slightly higher LR (fewer epochs to converge)
    "lr_head": 3e-4,
    "weight_decay": 0.05,
    "grad_clip": 0.1,

    "loss_weights": {"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0},

    "checkpoint_dir": "/content/drive/MyDrive/fibril_checkpoints",  # Save to Google Drive!
    "use_wandb": False,
    "device": "cuda",
    "mixed_precision": True,    # MANDATORY — halves VRAM usage on T4
    "seed": 42,
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Differential learning rates:
      - Backbone: lower LR (pretrained, shouldn't change much)
      - Decoder:  higher LR (randomly initialized)
    """
    backbone_params = list(model.backbone.parameters())
    other_params = [
        p for p in model.parameters()
        if not any(p is bp for bp in backbone_params)
    ]

    param_groups = [
        {"params": backbone_params, "lr": config["lr_backbone"]},
        {"params": other_params,    "lr": config["lr_head"]},
    ]

    return torch.optim.AdamW(
        param_groups,
        weight_decay=config["weight_decay"],
    )


def get_scheduler(optimizer, config: dict):
    """Cosine annealing with linear warmup."""
    total_steps = config["epochs"]
    warmup_steps = config["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_steps:
            return float(epoch) / float(max(1, warmup_steps))
        progress = (epoch - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_targets(batch: dict, device: str):
    """
    Extract GT labels and masks from batch dict.

    Returns:
        gt_labels: List of (N_i,) tensors on device
        gt_masks:  List of (N_i, H, W) tensors on device
    """
    gt_labels = [lab.to(device) for lab in batch["labels"]]
    gt_masks  = [msk.to(device) for msk in batch["masks"]]
    return gt_labels, gt_masks


def save_checkpoint(
    model, optimizer, scheduler, epoch: int,
    val_loss: float, best_val_loss: float,
    config: dict, is_best: bool = False,
):
    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "config": config,
    }

    # Always save latest
    torch.save(state, ckpt_dir / "latest_checkpoint.pth")

    # Save best separately
    if is_best:
        torch.save(state, ckpt_dir / "best_checkpoint.pth")
        print(f"    💾 Best checkpoint saved (val_loss={val_loss:.4f})")


def load_checkpoint(model, optimizer, scheduler, config: dict):
    """Load checkpoint if it exists (for resume training)."""
    ckpt_path = Path(config["checkpoint_dir"]) / "latest_checkpoint.pth"
    if not ckpt_path.exists():
        return 0, float("inf")

    print(f"[Train] Resuming from: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=config["device"])
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    return state["epoch"] + 1, state["best_val_loss"]


# ─────────────────────────────────────────────────────────────────────────────
# Train / Validate one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, criterion, optimizer, scaler,
    loader, config: dict, epoch: int,
) -> float:
    model.train()
    device = config["device"]
    accum = config["accum_steps"]
    use_amp = config["mixed_precision"]

    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"  [Train Ep {epoch+1}]", ncols=90, leave=False)

    for step, batch in enumerate(pbar):
        images = batch["images"].to(device)           # (B, 1, H, W)
        gt_labels, gt_masks = build_targets(batch, device)

        # ── Forward pass (with AMP) ───────────────────────────────────
        with autocast(device_type=device, enabled=use_amp):
            outputs = model(images)
            loss_dict = criterion(outputs, gt_labels, gt_masks)
            loss = loss_dict["total"] / accum

        # ── Backward pass ─────────────────────────────────────────────
        scaler.scale(loss).backward()

        # ── Optimizer step (every `accum` steps) ─────────────────────
        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss_dict["total"].item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{loss_dict['total'].item():.3f}",
            "ce":   f"{loss_dict['loss_ce'].item():.3f}",
            "dice": f"{loss_dict['loss_dice'].item():.3f}",
        })

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_one_epoch(
    model, criterion, loader, config: dict, epoch: int,
) -> float:
    model.eval()
    device = config["device"]
    use_amp = config["mixed_precision"]

    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"  [Val   Ep {epoch+1}]", ncols=90, leave=False)

    for batch in pbar:
        images = batch["images"].to(device)
        gt_labels, gt_masks = build_targets(batch, device)

        with autocast(device_type=device, enabled=use_amp):
            outputs = model(images)
            loss_dict = criterion(outputs, gt_labels, gt_masks)

        total_loss += loss_dict["total"].item()
        n_batches += 1
        pbar.set_postfix({"val_loss": f"{loss_dict['total'].item():.3f}"})

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict):
    set_seed(config["seed"])
    device = config["device"]

    print(f"\n{'='*65}")
    print(f"  Pulp Fibril Segmentation — Training")
    print(f"  Device:     {device}")
    print(f"  Backbone:   {config['backbone']}")
    print(f"  Epochs:     {config['epochs']}")
    print(f"  Batch size: {config['batch_size']} × {config['accum_steps']} accum")
    print(f"  AMP:        {config['mixed_precision']}")
    print(f"{'='*65}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    data_root = Path(PROJECT_ROOT) / config["data_root"] \
        if not Path(config["data_root"]).is_absolute() else Path(config["data_root"])

    train_loader, val_loader, _ = get_dataloaders(
        data_root=str(data_root),
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    print(f"[Data] Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(config, device=device)

    # Phase 1 Warmup: Freeze backbone
    if config["warmup_epochs"] > 0:
        print(f"[Train] Phase 1: Freezing backbone for {config['warmup_epochs']} warmup epochs")
        for p in model.backbone.parameters():
            p.requires_grad = False

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    criterion = Mask2FormerLoss(
        num_classes=config["num_classes"],
        weight_dict=config["loss_weights"],
        eos_coef=0.1,
    ).to(device)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    scaler = GradScaler(device, enabled=config["mixed_precision"])

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, config)

    # ── Optional W&B ──────────────────────────────────────────────────────────
    if config.get("use_wandb"):
        try:
            import wandb
            wandb.init(project="pulp-fibril-seg", config=config)
        except ImportError:
            print("[W&B] Not installed — skipping")
            config["use_wandb"] = False

    # ── Training Loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(start_epoch, config["epochs"]):

        # ── Switch to full training after warmup ──────────────────────
        if epoch == config["warmup_epochs"]:
            print(f"\n[Train] Phase 2: Unfreezing backbone for joint training")
            for p in model.backbone.parameters():
                p.requires_grad = True
            # Reset optimizer with differential LRs
            optimizer = get_optimizer(model, config)
            scheduler = get_scheduler(optimizer, config)
            scaler = GradScaler(device, enabled=config["mixed_precision"])

        t0 = time.time()
        train_loss = train_one_epoch(model, criterion, optimizer, scaler,
                                     train_loader, config, epoch)
        val_loss   = validate_one_epoch(model, criterion, val_loader, config, epoch)
        scheduler.step()

        elapsed = time.time() - t0
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Print epoch summary
        lr_bb = optimizer.param_groups[0]["lr"]
        lr_hd = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr_bb
        print(
            f"Epoch {epoch+1:3d}/{config['epochs']} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} {'✅ BEST' if is_best else ''} | "
            f"LR={lr_bb:.1e}/{lr_hd:.1e} | "
            f"{elapsed:.0f}s"
        )

        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss,
                        best_val_loss, config, is_best=is_best)

        if config.get("use_wandb"):
            import wandb
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

    # ── Save loss history ─────────────────────────────────────────────────────
    history_path = Path(config["checkpoint_dir"]) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint:    {config['checkpoint_dir']}/best_checkpoint.pth")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train FibrilMask2Former")
    parser.add_argument(
        "--config", type=str, default="debug",
        choices=["debug", "kaggle"],
        help="Which config to use: 'debug' (CPU, fast) or 'kaggle' (GPU, full)"
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data root path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = DEBUG_CONFIG.copy() if args.config == "debug" else KAGGLE_CONFIG.copy()

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.data_root is not None:
        config["data_root"] = args.data_root

    train(config)
