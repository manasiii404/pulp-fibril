"""
predict.py — Full Pipeline Inference
=====================================
Runs the complete 4-stage pipeline on a single input image:
  Stage 1: Real-ESRGAN super-resolution (2× upsampling)
  Stage 2+3: Swin-T + AG-UNet + Mask2Former (instance segmentation)
  Stage 4: Skeletonization + metrics

Outputs:
  - Color-coded mask overlay PNG
  - CSV metric report

Usage:
    python inference/predict.py \
        --image data/synthetic/images/00001.png \
        --checkpoint checkpoints/best_checkpoint.pth \
        --output outputs/ \
        --px_to_um 0.25
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.esrgan_sr import SuperResolutionModule
from models.mask2former import build_model
from quantification.skeletonize import mask_to_metrics
from inference.report_generator import save_report
from inference.visualize import visualize_predictions


# ─────────────────────────────────────────────────────────────────────────────
# Post-Processing: Convert Model Output → Binary Masks
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_outputs(
    pred_logits: torch.Tensor,   # (1, Q, num_cls+1)
    pred_masks: torch.Tensor,    # (1, Q, H, W)
    orig_size: tuple,            # (H_orig, W_orig)
    score_threshold: float = 0.5,
) -> list:
    """
    Convert raw Mask2Former output to a list of binary instance masks.

    Args:
        pred_logits:     Raw class logits per query
        pred_masks:      Raw mask logits per query
        orig_size:       Original image size to upsample masks to
        score_threshold: Minimum confidence score to keep an instance

    Returns:
        List of (H, W) uint8 binary masks (255 = instance)
    """
    # Compute class confidence scores: softmax → max over real classes (exclude "no object")
    scores = pred_logits[0].softmax(dim=-1)[:, :-1].max(dim=-1).values  # (Q,)

    # Binarize mask logits
    masks_sigmoid = pred_masks[0].sigmoid()  # (Q, H, W)

    # Upsample masks to original image size
    masks_up = F.interpolate(
        masks_sigmoid.unsqueeze(0),
        size=orig_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)   # (Q, H_orig, W_orig)

    # Filter: keep queries with score > threshold and non-empty masks
    instance_masks = []
    for q in range(scores.shape[0]):
        if scores[q].item() < score_threshold:
            continue
        mask_bin = (masks_up[q] > 0.5).cpu().numpy().astype(np.uint8) * 255
        if mask_bin.sum() > 100:   # Skip tiny predictions
            instance_masks.append(mask_bin)

    return instance_masks


# ─────────────────────────────────────────────────────────────────────────────
# Main Inference Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    image_path: str,
    checkpoint_path: str,
    output_dir: str,
    px_to_um: float = 0.25,
    score_threshold: float = 0.5,
    use_sr: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Run full 4-stage prediction pipeline on one image.

    Args:
        image_path:       Path to input grayscale .png image
        checkpoint_path:  Path to best_checkpoint.pth
        output_dir:       Directory to save outputs
        px_to_um:         Calibration: micrometers per pixel
        score_threshold:  Mask2Former query confidence threshold
        use_sr:           Apply Real-ESRGAN stage (disable for quick testing)
        device:           "cuda" or "cpu"

    Returns:
        metrics_list: List of per-fibril metric dicts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    img_name = Path(image_path).stem

    print(f"\n{'='*55}")
    print(f"  Pulp Fibril Prediction Pipeline")
    print(f"  Image:    {image_path}")
    print(f"  Device:   {device}")
    print(f"{'='*55}")

    # ── Load Input Image ──────────────────────────────────────────────────────
    orig_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if orig_image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    print(f"[1/4] Loaded image: {orig_image.shape}")

    # ── Stage 1: Super-Resolution ─────────────────────────────────────────────
    if use_sr and device == "cuda":
        print("[2/4] Applying Real-ESRGAN super-resolution (2×)...")
        sr_module = SuperResolutionModule(scale=2, device=device)
        sr_image = sr_module.enhance(orig_image)
        print(f"      Enhanced: {orig_image.shape} → {sr_image.shape}")
    else:
        print("[2/4] Skipping SR (CPU mode / disabled)")
        sr_image = orig_image

    # ── Stage 2+3: Instance Segmentation ─────────────────────────────────────
    print("[3/4] Running Mask2Former instance segmentation...")

    # Build model and load checkpoint
    config = {
        "backbone": "swin_tiny",
        "pretrained": False,      # Don't re-download during inference
        "num_queries": 50,
        "hidden_dim": 256,
        "nheads": 8,
        "dec_layers": 6,
        "num_classes": 1,
    }
    model = build_model(config, device=device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"      Checkpoint loaded: epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', 0):.4f}")

    # Preprocess: normalize, add batch + channel dims
    img_norm = sr_image.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm[None, None]).to(device)  # (1, 1, H, W)

    orig_h, orig_w = sr_image.shape

    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            outputs = model(img_tensor)

    # Convert outputs to binary masks
    instance_masks = postprocess_outputs(
        outputs["pred_logits"],
        outputs["pred_masks"],
        orig_size=(orig_h, orig_w),
        score_threshold=score_threshold,
    )
    print(f"      Detected {len(instance_masks)} fibril instances")

    # ── Stage 4: Skeletonization + Metrics ───────────────────────────────────
    print("[4/4] Computing fibril metrics (skeletonization + GNN)...")
    all_metrics = []

    for idx, mask in enumerate(instance_masks):
        m = mask_to_metrics(
            binary_mask=mask,
            intensity_image=sr_image,
            px_to_um=px_to_um,
            fibril_id=idx,
        )
        all_metrics.append(m)
        print(f"      Fibril {idx:3d}: "
              f"len={m['total_length_um']:.1f}μm  "
              f"tort={m['tortuosity']:.3f}  "
              f"branch={m['branching_count']}  "
              f"FI={m['fibrillation_index']:.2f}")

    # ── Save Outputs ──────────────────────────────────────────────────────────
    # Visualization
    vis_path = str(output_path / f"{img_name}_segmented.png")
    visualize_predictions(sr_image, instance_masks, save_path=vis_path, show=False)
    print(f"\n  ✅ Visualization saved: {vis_path}")

    # CSV Report
    report_path = str(output_path / f"{img_name}_report.csv")
    save_report(all_metrics, report_path)
    print(f"  ✅ Report saved:        {report_path}")
    print(f"{'='*55}\n")

    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Pulp Fibril Prediction Pipeline")
    parser.add_argument("--image",      required=True,  help="Path to input image (.png)")
    parser.add_argument("--checkpoint", required=True,  help="Path to best_checkpoint.pth")
    parser.add_argument("--output",     default="outputs", help="Output directory")
    parser.add_argument("--px_to_um",   type=float, default=0.25,
                        help="Calibration: micrometers per pixel")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Mask confidence threshold (0-1)")
    parser.add_argument("--no_sr",      action="store_true",
                        help="Skip super-resolution stage")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        px_to_um=args.px_to_um,
        score_threshold=args.threshold,
        use_sr=not args.no_sr,
        device=args.device,
    )
