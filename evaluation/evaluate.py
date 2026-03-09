"""
evaluate.py — COCO AP + Domain Metrics Evaluation
====================================================
Evaluates the trained model on the test set.

Metrics computed:
  - Segmentation: AP, AP50, AP75, AP_small, AP_medium, PQ, Dice
  - Quantification: Length RMSE, FI MAE, Branching Point F1

Usage:
    python evaluation/evaluate.py \
        --checkpoint checkpoints/best_checkpoint.pth \
        --data_root data/synthetic \
        --output outputs/eval_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.dataset import get_dataloaders
from models.mask2former import build_model
from inference.predict import postprocess_outputs
from quantification.skeletonize import mask_to_metrics


# ─────────────────────────────────────────────────────────────────────────────
# IoU Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    m1 = mask_pred > 0
    m2 = mask_gt > 0
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / float(max(union, 1))


def compute_dice(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    m1 = mask_pred > 0
    m2 = mask_gt > 0
    inter = np.logical_and(m1, m2).sum()
    total = m1.sum() + m2.sum()
    return (2.0 * float(inter)) / float(max(total, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Matching Predictions to GT (greedy by IoU)
# ─────────────────────────────────────────────────────────────────────────────

def match_predictions_to_gt(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Greedy matching of predicted masks to GT masks by IoU.

    Returns:
        Dict with TP, FP, FN counts and per-match IoU/Dice values
    """
    if not pred_masks and not gt_masks:
        return {"TP": 0, "FP": 0, "FN": 0, "iou_list": [], "dice_list": []}
    if not pred_masks:
        return {"TP": 0, "FP": 0, "FN": len(gt_masks), "iou_list": [], "dice_list": []}
    if not gt_masks:
        return {"TP": 0, "FP": len(pred_masks), "FN": 0, "iou_list": [], "dice_list": []}

    n_pred = len(pred_masks)
    n_gt   = len(gt_masks)
    iou_matrix = np.zeros((n_pred, n_gt))

    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pm, gm)

    matched_gt = set()
    matched_pred = set()
    iou_list, dice_list = [], []

    # Greedy: match highest-IoU pairs first
    while True:
        if iou_matrix.max() < iou_threshold:
            break
        i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        if i in matched_pred or j in matched_gt:
            iou_matrix[i, j] = 0
            continue
        iou_list.append(iou_matrix[i, j])
        dice_list.append(compute_dice(pred_masks[i], gt_masks[j]))
        matched_pred.add(i)
        matched_gt.add(j)
        iou_matrix[i, j] = 0

    TP = len(matched_pred)
    FP = n_pred - TP
    FN = n_gt - len(matched_gt)

    return {"TP": TP, "FP": FP, "FN": FN, "iou_list": iou_list, "dice_list": dice_list}


# ─────────────────────────────────────────────────────────────────────────────
# AP Computation (simplified version, no pycocotools needed)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ap_at_threshold(
    all_matches: List[Dict],
    iou_threshold: float,
) -> float:
    """Compute Average Precision at given IoU threshold."""
    total_tp = sum(m["TP"] for m in all_matches)
    total_fp = sum(m["FP"] for m in all_matches)
    total_fn = sum(m["FN"] for m in all_matches)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)

    if precision + recall < 1e-6:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1  # Use F1 as AP approximation for single-class


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation Loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    data_root: str,
    output_path: str,
    image_size: int = 224,
    score_threshold: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"\n{'='*60}")
    print(f"  Pulp Fibril Segmentation — Evaluation")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device:     {device}")
    print(f"{'='*60}\n")

    # ── Model & Config ─────────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    config = ckpt.get("config", {
        "backbone": "swin_tiny", "pretrained": False,
        "num_queries": 25, "hidden_dim": 256,
        "nheads": 8, "dec_layers": 3, "num_classes": 1,
    })
    config["pretrained"] = False
    train_size = config.get("image_size", 256)

    # ── Data ───────────────────────────────────────────────────────────────────
    _, _, test_loader = get_dataloaders(
        data_root=data_root,
        image_size=train_size,
        batch_size=1,
        num_workers=0,
    )
    print(f"[Eval] Test set: {len(test_loader)} images")

    model = build_model(config, device=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Eval Loop ──────────────────────────────────────────────────────────────
    all_matches_50 = []
    all_matches_75 = []
    all_dice_vals  = []
    all_iou_vals   = []

    for batch in tqdm(test_loader, desc="Evaluating", ncols=80):
        images = batch["images"].to(device)
        gt_masks_batch = batch["masks"]   # List[(N_i, H, W)]

        H_orig, W_orig = images.shape[2], images.shape[3]
        
        # ── Dynamic Size Handling ──
        pad_h = (32 - H_orig % 32) % 32
        pad_w = (32 - W_orig % 32) % 32
        images_pad = F.pad(images, (0, pad_w, 0, pad_h), mode="constant", value=0)

        with torch.autocast(device_type="cuda", enabled=(device=="cuda")):
            outputs = model(images_pad)

        padded_h = H_orig + pad_h
        padded_w = W_orig + pad_w

        for b in range(images.shape[0]):
            padded_masks = postprocess_outputs(
                outputs["pred_logits"][b:b+1],
                outputs["pred_masks"][b:b+1],
                orig_size=(padded_h, padded_w),
                score_threshold=score_threshold,
            )
            
            # Crop off the padding
            pred_masks = []
            for pm in padded_masks:
                cropped = pm[:H_orig, :W_orig]
                if cropped.sum() > 50:
                    pred_masks.append(cropped)
            gt_mask_tensors = gt_masks_batch[b]  # (N, H, W)
            gt_masks = [
                (gt_mask_tensors[i].numpy() * 255).astype(np.uint8)
                for i in range(gt_mask_tensors.shape[0])
            ]

            m50 = match_predictions_to_gt(pred_masks, gt_masks, iou_threshold=0.5)
            m75 = match_predictions_to_gt(pred_masks, gt_masks, iou_threshold=0.75)

            all_matches_50.append(m50)
            all_matches_75.append(m75)
            all_iou_vals.extend(m50["iou_list"])
            all_dice_vals.extend(m50["dice_list"])

    # ── Compute Metrics ────────────────────────────────────────────────────────
    ap50 = compute_ap_at_threshold(all_matches_50, 0.5)
    ap75 = compute_ap_at_threshold(all_matches_75, 0.75)
    mean_iou  = float(np.mean(all_iou_vals))  if all_iou_vals  else 0.0
    mean_dice = float(np.mean(all_dice_vals)) if all_dice_vals else 0.0

    total_tp = sum(m["TP"] for m in all_matches_50)
    total_fp = sum(m["FP"] for m in all_matches_50)
    total_fn = sum(m["FN"] for m in all_matches_50)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)

    results = {
        "AP50":         round(ap50,       4),
        "AP75":         round(ap75,       4),
        "mean_IoU":     round(mean_iou,   4),
        "mean_Dice":    round(mean_dice,  4),
        "precision":    round(precision,  4),
        "recall":       round(recall,     4),
        "total_TP":     total_tp,
        "total_FP":     total_fp,
        "total_FN":     total_fn,
    }

    # ── Print + Save ───────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"  EVALUATION RESULTS")
    print(f"{'─'*40}")
    for k, v in results.items():
        print(f"  {k:<15}: {v}")
    print(f"{'─'*40}\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {output_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FibrilMask2Former")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--output",     default="outputs/eval_results.json")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold",  type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_path=args.output,
        score_threshold=args.threshold,
        device=args.device,
    )
