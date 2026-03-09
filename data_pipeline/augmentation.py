"""
augmentation.py — Albumentations Augmentation Pipeline
=======================================================
Compatible with albumentations >= 2.0.x
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Training Augmentations
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(image_size: int):
    return A.Compose(
        [
            # ── Spatial Transforms ────────────────────────────────────
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),

            # Elastic deformation — simulates fiber bending/stretching
            # NOTE: alpha_affine removed in albumentations v2
            A.ElasticTransform(
                alpha=80,
                sigma=6,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            ),

            # Grid distortion — simulates optical aberrations
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3),

            # ── Scale and Pad: Keep aspect ratio, consider whole image ──
            A.LongestMaxSize(max_size=image_size, p=1.0),
            A.PadIfNeeded(
                min_height=image_size, 
                min_width=image_size, 
                border_mode=cv2.BORDER_CONSTANT
            ),

            # ── Pixel-level Transforms ────────────────────────────────
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.6,
            ),

            A.RandomGamma(gamma_limit=(75, 130), p=0.4),

            # NOTE: GaussNoise v2 uses std_range instead of var_limit/mean
            A.GaussNoise(std_range=(0.02, 0.11), p=0.5),

            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.4),

            # NOTE: CoarseDropout v2 uses fill= not fill_value=
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(8, 24),
                hole_width_range=(8, 24),
                fill=235,
                p=0.3,
            ),

            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.Equalize(p=1.0),
            ], p=0.3),

            # ToTensorV2 internally handles / 255.0 for uint8 if normalization is absent,
            # but usually it doesn't divide unless Normalize is used. 
            # We explicitly scale down to 0-1 values.
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation / Test Augmentations
# ─────────────────────────────────────────────────────────────────────────────

def get_val_transforms(image_size: int):
    transforms = []
    
    if image_size is not None and image_size > 0:
        transforms.append(A.LongestMaxSize(max_size=image_size, p=1.0))
        
    transforms.extend([
        # Pad to nearest multiple of 32 for Swin compatibility natively
        A.PadIfNeeded(
            min_height=None, min_width=None, 
            pad_height_divisor=32, pad_width_divisor=32, 
            border_mode=cv2.BORDER_CONSTANT
        ),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms, additional_targets={"mask": "mask"})


# ─────────────────────────────────────────────────────────────────────────────
# ESRGAN Degradation Augmentations
# ─────────────────────────────────────────────────────────────────────────────

def get_degradation_transforms():
    return A.Compose(
        [
            A.GaussianBlur(blur_limit=(3, 9), p=0.8),
            A.GaussNoise(std_range=(0.06, 0.24), p=0.8),
            A.Downscale(scale_range=(0.25, 0.5), p=0.6),
            A.ImageCompression(quality_range=(50, 90), p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Apply augmentation to image + list of instance masks
# ─────────────────────────────────────────────────────────────────────────────

def apply_augmentation_with_masks(
    transform,
    image: np.ndarray,
    masks: list,
) -> tuple:
    """
    Apply an Albumentations transform to a grayscale image and a list of masks.

    Stacks all masks into a multi-channel array, applies transforms, then splits.

    Args:
        transform: Albumentations Compose object
        image:     (H, W) uint8 grayscale numpy array
        masks:     List of (H, W) uint8 binary masks

    Returns:
        aug_image: torch.Tensor (1, H, W) or (C, H, W)
        aug_masks: List of (H, W) uint8 numpy arrays
    """
    if len(masks) == 0:
        result = transform(image=image)
        return result["image"], []

    # Stack all masks into a single (H, W, N) float32 array
    mask_stack = np.stack(masks, axis=-1).astype(np.float32)

    result = transform(image=image, mask=mask_stack)

    aug_image = result["image"]          # torch.Tensor after ToTensorV2
    aug_mask_raw = result["mask"]        # Could be Tensor or ndarray depending on version

    # ── Convert mask back to numpy regardless of type ──────────────────────
    if isinstance(aug_mask_raw, torch.Tensor):
        aug_mask_np = aug_mask_raw.cpu().numpy()
    else:
        aug_mask_np = np.array(aug_mask_raw)

    # aug_mask_np shape: (H, W, N) or (H, W) if N=1
    # Split back into list of (H, W) uint8 masks
    if aug_mask_np.ndim == 2:
        # Single mask — shape (H, W)
        aug_masks = [(aug_mask_np > 0.5).astype(np.uint8) * 255]
    else:
        # Multiple masks — shape (H, W, N)
        aug_masks = [
            (aug_mask_np[..., i] > 0.5).astype(np.uint8) * 255
            for i in range(aug_mask_np.shape[-1])
        ]

    return aug_image, aug_masks
