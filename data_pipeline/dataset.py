"""
dataset.py — FibrilDataset PyTorch Dataset
==========================================
Loads the FibrilSynth synthetic dataset (COCO format) for training.

Returns per-image dict with:
  - image tensor (1, H, W) float32
  - list of instance mask tensors
  - list of bounding boxes
  - list of category labels

Supports train/val/test splits.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from data_pipeline.augmentation import (
    get_train_transforms,
    get_val_transforms,
    apply_augmentation_with_masks,
)


# ─────────────────────────────────────────────────────────────────────────────
# FibrilDataset
# ─────────────────────────────────────────────────────────────────────────────

class FibrilDataset(Dataset):
    """
    PyTorch Dataset for the FibrilSynth synthetic fibril dataset (COCO format).

    Directory layout expected:
        data_root/
            images/
                00000.png
                00001.png
                ...
            masks/
                00000/
                    instance_000.png
                    instance_001.png
                    ...
            annotations.json

    Args:
        data_root:   Path to the dataset root (contains images/ masks/ annotations.json)
        split:       "train", "val", or "test"
        image_size:  Target image size after augmentation
        val_frac:    Fraction of data for validation (default 0.15)
        test_frac:   Fraction of data for test     (default 0.10)
        seed:        Random seed for split reproducibility
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 512,
        val_frac: float = 0.15,
        test_frac: float = 0.10,
        seed: int = 42,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.masks_dir = self.data_root / "masks"
        self.image_size = image_size
        self.split = split

        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test'; got '{split}'"

        # ── Load COCO annotations ─────────────────────────────────────
        annot_path = self.data_root / "annotations.json"
        if not annot_path.exists():
            raise FileNotFoundError(
                f"annotations.json not found at {annot_path}\n"
                "Run generate_dataset.py first."
            )

        with open(annot_path, "r") as f:
            coco = json.load(f)

        # Build image_id → image info map
        self.id_to_image = {img["id"]: img for img in coco["images"]}

        # Build image_id → list of annotations map
        self.id_to_annots: Dict[int, List[dict]] = {}
        for annot in coco["annotations"]:
            iid = annot["image_id"]
            self.id_to_annots.setdefault(iid, []).append(annot)

        # All image IDs
        all_ids = sorted(self.id_to_image.keys())

        # ── Deterministic train/val/test split ───────────────────────
        rng = random.Random(seed)
        shuffled = all_ids.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_val - n_test

        if split == "train":
            self.image_ids = shuffled[:n_train]
        elif split == "val":
            self.image_ids = shuffled[n_train: n_train + n_val]
        else:
            self.image_ids = shuffled[n_train + n_val:]

        # ── Load augmentation transforms ─────────────────────────────
        if split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        print(f"[FibrilDataset] Split: {split:5s} | "
              f"Images: {len(self.image_ids):4d} | Size: {image_size}×{image_size}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dict:
            image:    torch.Tensor (1, H, W) float32   ← grayscale image
            masks:    torch.Tensor (N, H, W) float32   ← N instance masks
            boxes:    torch.Tensor (N, 4) float32      ← [x1, y1, x2, y2] normalized
            labels:   torch.Tensor (N,) int64          ← class IDs (all = 1 = fibril)
            image_id: int                              ← for COCO eval
        """
        image_id = self.image_ids[idx]
        image_info = self.id_to_image[image_id]
        annots = self.id_to_annots.get(image_id, [])

        # ── Load image ────────────────────────────────────────────────
        img_path = self.images_dir / image_info["file_name"]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Could not read image: {img_path}")

        H, W = image.shape

        # ── Load instance masks ───────────────────────────────────────
        masks_subdir = self.masks_dir / f"{image_id:05d}"
        masks = []

        for annot in annots:
            annot_idx = annots.index(annot)
            mask_path = masks_subdir / f"instance_{annot_idx:03d}.png"

            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None and np.sum(mask > 0) > 50:
                    masks.append(mask)
            else:
                # Reconstruct mask from polygon segmentation if file missing
                mask = self._polygon_to_mask(annot.get("segmentation", []), H, W)
                if mask is not None and np.sum(mask > 0) > 50:
                    masks.append(mask)

        # ── Apply augmentations ───────────────────────────────────────
        if len(masks) > 0:
            image_tensor, aug_masks = apply_augmentation_with_masks(
                self.transform, image, masks
            )
        else:
            result = self.transform(image=image)
            image_tensor = result["image"]
            aug_masks = []

        # If image is (H, W) after ToTensorV2, add channel dim
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 3 and image_tensor.shape[0] == 3:
            # If converted to 3-ch, take first channel (grayscale)
            image_tensor = image_tensor[:1]

        # ── Build mask tensor ─────────────────────────────────────────
        if len(aug_masks) > 0:
            mask_arrays = [
                (m > 127).astype(np.float32) for m in aug_masks
            ]
            masks_tensor = torch.from_numpy(np.stack(mask_arrays, axis=0))  # (N, H, W)
        else:
            # No instances — return dummy empty tensors
            masks_tensor = torch.zeros(
                (0, self.image_size, self.image_size), dtype=torch.float32
            )

        # ── Build bounding boxes from masks ───────────────────────────
        boxes = []
        valid_indices = []
        for i, mask_arr in enumerate(aug_masks):
            ys, xs = np.where(mask_arr > 127)
            if len(xs) == 0:
                continue
            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max()), float(ys.max())
            # Normalize to [0, 1]
            x1 /= self.image_size;  x2 /= self.image_size
            y1 /= self.image_size;  y2 /= self.image_size
            boxes.append([x1, y1, x2, y2])
            valid_indices.append(i)

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            masks_tensor = masks_tensor[valid_indices]
            labels_tensor = torch.ones(len(boxes), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            masks_tensor = torch.zeros(
                (0, self.image_size, self.image_size), dtype=torch.float32
            )
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        return {
            "image": image_tensor,          # (1, H, W)
            "masks": masks_tensor,          # (N, H, W)
            "boxes": boxes_tensor,          # (N, 4)
            "labels": labels_tensor,         # (N,)
            "image_id": image_id,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _polygon_to_mask(
        self,
        segmentation: List[List[float]],
        H: int,
        W: int,
    ) -> Optional[np.ndarray]:
        """
        Convert COCO polygon segmentation to binary mask.

        Args:
            segmentation: List of [x1,y1,x2,y2,...] polygon lists
            H, W: Image dimensions

        Returns:
            mask: (H, W) uint8 binary mask or None if no valid polygon
        """
        mask = np.zeros((H, W), dtype=np.uint8)
        for poly in segmentation:
            if len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], color=255)
        return mask if np.sum(mask > 0) > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Collate Function (handles variable-length mask lists)
# ─────────────────────────────────────────────────────────────────────────────

def fibril_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for FibrilDataset.

    Standard torch.stack won't work here because each image has
    a different number of fiber instances (N varies per image).
    We return lists for variable-length tensors.

    Args:
        batch: List of dicts from FibrilDataset.__getitem__

    Returns:
        Batched dict with images stacked and masks/boxes as lists
    """
    images = torch.stack([b["image"] for b in batch], dim=0)   # (B, 1, H, W)
    masks = [b["masks"] for b in batch]                         # List[(N_i, H, W)]
    boxes = [b["boxes"] for b in batch]                         # List[(N_i, 4)]
    labels = [b["labels"] for b in batch]                       # List[(N_i,)]
    image_ids = [b["image_id"] for b in batch]

    return {
        "images": images,
        "masks": masks,
        "boxes": boxes,
        "labels": labels,
        "image_ids": image_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_root: str,
    image_size: int = 512,
    batch_size: int = 1,
    num_workers: int = 2,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test DataLoaders.

    Args:
        data_root:   Path to dataset root
        image_size:  Square image size
        batch_size:  Batch size (1 recommended for T4/P100)
        num_workers: DataLoader workers
        val_frac:    Validation fraction
        test_frac:   Test fraction
        seed:        Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    common = dict(
        data_root=data_root,
        image_size=image_size,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    train_ds = FibrilDataset(split="train", **common)
    val_ds   = FibrilDataset(split="val",   **common)
    test_ds  = FibrilDataset(split="test",  **common)

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=fibril_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
