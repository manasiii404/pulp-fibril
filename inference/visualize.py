"""
visualize.py — Color-coded Instance Mask Visualization
=======================================================
Overlays instance masks on the original image with:
  - Random distinct colors per fibril
  - Fibril ID labels
  - Skeleton overlay (optional)
  - Summary stats panel

Usage:
    from inference.visualize import visualize_predictions
    visualize_predictions(image, masks, save_path="output.png")

    # Check synthetic dataset
    python inference/visualize.py --mode check_synthetic --n 5
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Color Palette (distinct colors for up to 50 instances)
# ─────────────────────────────────────────────────────────────────────────────

def generate_color_palette(n: int, seed: int = 0) -> List[tuple]:
    """Generate N visually distinct RGB colors."""
    rng = random.Random(seed)
    colors = []
    for i in range(n):
        # Use golden ratio for hue distribution
        hue = (i * 0.618033988749895) % 1.0
        # Convert HSV → RGB: bright, saturated colors
        h = hue * 360
        s, v = 0.85, 0.95
        hi = int(h / 60) % 6
        f = h / 60 - int(h / 60)
        p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
        rgb_map = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
        r, g, b = rgb_map[hi]
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


# ─────────────────────────────────────────────────────────────────────────────
# Main Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_predictions(
    image: np.ndarray,
    masks: List[np.ndarray],
    metrics: Optional[List[dict]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Fibril Instance Segmentation",
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Create a color-coded fibril instance segmentation visualization.

    Args:
        image:     (H, W) uint8 grayscale image
        masks:     List of (H, W) uint8 binary instance masks
        metrics:   Optional list of metric dicts for annotation
        save_path: If given, saves PNG to this path
        show:      If True, display with matplotlib
        title:     Plot title
        alpha:     Mask overlay opacity (0-1)

    Returns:
        vis_bgr: (H, W, 3) uint8 BGR visualization image
    """
    H, W = image.shape
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.float32)

    colors = generate_color_palette(max(len(masks), 1))
    legend_patches = []

    for i, (mask, color) in enumerate(zip(masks, colors)):
        # Color this instance
        colored = np.zeros_like(vis)
        colored[mask > 127] = color

        vis = cv2.addWeighted(vis, 1.0, colored, alpha, 0)

        # Draw contour outline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis.astype(np.uint8), contours, -1, color, thickness=2)

        # Draw fibril ID label at centroid
        ys, xs = np.where(mask > 127)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            label_text = str(i)
            if metrics and i < len(metrics):
                label_text = f"{i}|{metrics[i].get('total_length_um', 0):.0f}μm"
            cv2.putText(
                vis.astype(np.uint8), label_text,
                (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1, cv2.LINE_AA
            )

        legend_patches.append(
            mpatches.Patch(
                color=[c / 255.0 for c in color],
                label=f"Fibril {i}",
            )
        )

    vis_uint8 = np.clip(vis, 0, 255).astype(np.uint8)

    # ── Matplotlib display ────────────────────────────────────────────────────
    if show or save_path:
        n_panels = 2
        fig, axes = plt.subplots(1, n_panels, figsize=(14, 6))
        fig.patch.set_facecolor("#1a1a2e")
        fig.suptitle(title, fontsize=14, fontweight="bold", color="white")

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original Image", color="white", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(vis_uint8[..., ::-1])  # BGR → RGB
        axes[1].set_title(f"Instance Segmentation ({len(masks)} fibrils)", color="white", fontsize=11)
        axes[1].axis("off")
        if legend_patches:
            legend = axes[1].legend(
                handles=legend_patches, loc="upper right",
                fontsize=7, ncol=2,
                facecolor="#2a2a4a", labelcolor="white",
                edgecolor="gray",
            )

        for ax in axes:
            ax.set_facecolor("#1a1a2e")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        if show:
            plt.show()
        plt.close()

    return vis_uint8


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Dataset Check Mode
# ─────────────────────────────────────────────────────────────────────────────

def check_synthetic_dataset(data_root: str, n: int = 5):
    """
    Show n random images from the synthetic dataset with their masks.
    Run this to quickly verify the generator is working correctly.
    """
    import json, os

    data_path = Path(data_root)
    annot_path = data_path / "annotations.json"

    if not annot_path.exists():
        print(f"ERROR: annotations.json not found at {annot_path}")
        print("Run: python data_pipeline/generate_dataset.py --n 20 --preview")
        return

    with open(annot_path) as f:
        coco = json.load(f)

    image_infos = random.sample(coco["images"], min(n, len(coco["images"])))
    id_to_annots = {}
    for a in coco["annotations"]:
        id_to_annots.setdefault(a["image_id"], []).append(a)

    for info in image_infos:
        img_path = data_path / "images" / info["file_name"]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        # Load masks
        masks_dir = data_path / "masks" / f"{info['id']:05d}"
        masks = []
        if masks_dir.exists():
            for mf in sorted(masks_dir.iterdir()):
                m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    masks.append(m)

        visualize_predictions(
            image, masks,
            title=f"FibrilSynth: {info['file_name']} ({len(masks)} fibrils)",
            show=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize fibril predictions")
    parser.add_argument("--mode", default="check_synthetic",
                        choices=["check_synthetic", "single"],
                        help="Visualization mode")
    parser.add_argument("--data_root", default="data/synthetic",
                        help="Synthetic dataset root (for check_synthetic mode)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of samples to show")
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path (for single mode)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "check_synthetic":
        data_root = Path(PROJECT_ROOT) / args.data_root
        check_synthetic_dataset(str(data_root), n=args.n)
    elif args.mode == "single" and args.image:
        image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        visualize_predictions(image, masks=[], show=True)
