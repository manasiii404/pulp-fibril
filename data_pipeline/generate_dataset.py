"""
generate_dataset.py — FibrilSynth Dataset CLI
==============================================
Generates N synthetic fibril images + COCO-format annotations.

Usage:
    python data_pipeline/generate_dataset.py --n 500 --output data/synthetic --seed 42
    python data_pipeline/generate_dataset.py --n 10  --output data/synthetic --preview
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Make sure project root is on path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.synthetic_gen import FibrilSynthGenerator, FibrilConfig


# ─────────────────────────────────────────────────────────────────────────────
# COCO Annotation Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_coco_skeleton(image_size: int) -> dict:
    """Create the COCO dataset skeleton (header + categories)."""
    return {
        "info": {
            "description": "FibrilSynth — Synthetic Pulp Fibril Microscopy Dataset",
            "version": "1.0",
            "year": 2026,
            "contributor": "DL Capstone Project",
        },
        "licenses": [],
        "categories": [
            {
                "id": 1,
                "name": "fibril",
                "supercategory": "fiber",
            }
        ],
        "images": [],
        "annotations": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Helper
# ─────────────────────────────────────────────────────────────────────────────

def visualize_sample(image: np.ndarray, masks: list, title: str = "FibrilSynth Sample"):
    """
    Show one generated image with color-coded instance masks overlaid.

    Args:
        image: (H, W) uint8 grayscale
        masks: List of (H, W) uint8 binary masks
        title: Plot title
    """
    # Convert grayscale to BGR for coloring
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.float32)

    # Generate distinct colors for each instance
    np.random.seed(0)
    colors = [
        tuple(np.random.randint(50, 255, 3).tolist())
        for _ in range(len(masks))
    ]

    legend_patches = []
    for i, (mask, color) in enumerate(zip(masks, colors)):
        colored_mask = np.zeros_like(vis)
        colored_mask[mask > 0] = color
        vis = cv2.addWeighted(vis, 1.0, colored_mask, 0.45, 0)
        legend_patches.append(
            mpatches.Patch(color=[c / 255.0 for c in color], label=f"Fiber {i+1}")
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Synthetic Image")
    axes[0].axis("off")

    axes[1].imshow(vis.astype(np.uint8)[..., ::-1])  # BGR → RGB
    axes[1].set_title(f"Instance Masks ({len(masks)} fibers)")
    axes[1].axis("off")
    if legend_patches:
        axes[1].legend(handles=legend_patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main Generation Function
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    n: int,
    output_dir: str,
    image_size: int = 512,
    seed: int = 42,
    preview: bool = False,
    preview_n: int = 3,
):
    """
    Generate N synthetic fibril images and save them with COCO annotations.

    Args:
        n:           Number of images to generate
        output_dir:  Root directory (creates images/ masks/ subdirs)
        image_size:  Square image side in pixels (512 recommended)
        seed:        Random seed for reproducibility
        preview:     If True, display first `preview_n` images
        preview_n:   How many images to show in preview mode
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    cfg = FibrilConfig(image_size=image_size)
    generator = FibrilSynthGenerator(config=cfg, seed=seed)

    coco = build_coco_skeleton(image_size)
    total_annotations = 0
    total_instances = 0

    print(f"\n{'='*60}")
    print(f"  FibrilSynth Dataset Generator")
    print(f"  Images:     {n}")
    print(f"  Output:     {output_path.resolve()}")
    print(f"  Image size: {image_size}×{image_size}")
    print(f"  Seed:       {seed}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for image_id in tqdm(range(n), desc="Generating", unit="img", ncols=80):
        # Generate one image
        image, masks, annotations = generator.generate(image_id=image_id)

        # ── Save image ─────────────────────────────────────────────────
        img_filename = f"{image_id:05d}.png"
        img_path = images_dir / img_filename
        cv2.imwrite(str(img_path), image)

        # ── Save per-instance masks ────────────────────────────────────
        img_masks_dir = masks_dir / f"{image_id:05d}"
        img_masks_dir.mkdir(exist_ok=True)

        for inst_idx, mask in enumerate(masks):
            mask_path = img_masks_dir / f"instance_{inst_idx:03d}.png"
            cv2.imwrite(str(mask_path), mask)

        # ── COCO image entry ───────────────────────────────────────────
        coco["images"].append({
            "id": image_id,
            "file_name": img_filename,
            "width": image_size,
            "height": image_size,
        })

        # ── COCO annotations ───────────────────────────────────────────
        coco["annotations"].extend(annotations)
        total_annotations += len(annotations)
        total_instances += len(masks)

        # ── Optional preview ───────────────────────────────────────────
        if preview and image_id < preview_n:
            visualize_sample(image, masks, title=f"Sample Image #{image_id}")

    # ── Save COCO JSON ─────────────────────────────────────────────────────
    annotations_path = output_path / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(coco, f, indent=2)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  ✅  Dataset generation complete!")
    print(f"  Images generated:       {n}")
    print(f"  Total fiber instances:  {total_instances}")
    print(f"  Total annotations:      {total_annotations}")
    print(f"  Avg instances/image:    {total_instances/n:.1f}")
    print(f"  Time elapsed:           {elapsed:.1f}s")
    print(f"  Annotations saved to:   {annotations_path.resolve()}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="FibrilSynth — Generate synthetic pulp fibril microscopy dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n", type=int, default=500,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--output", type=str, default="data/synthetic",
        help="Output directory (relative to project root)"
    )
    parser.add_argument(
        "--size", type=int, default=512,
        help="Square image size in pixels (512 for free-tier training)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Display first 3 images with masks after generation"
    )
    parser.add_argument(
        "--preview_n", type=int, default=3,
        help="How many images to preview"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve output relative to project root
    output = Path(PROJECT_ROOT) / args.output

    generate_dataset(
        n=args.n,
        output_dir=str(output),
        image_size=args.size,
        seed=args.seed,
        preview=args.preview,
        preview_n=args.preview_n,
    )
