"""
FibrilSynth — Synthetic Pulp Fibril Image Generator
=====================================================
Generates realistic grayscale microscopy images of wood pulp fibers
with hair-like fibrils, translucency, overlap, and noise.

Each image comes with:
  - Per-instance binary masks
  - COCO-format annotation dict

Author: Capstone Project
"""

import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


# ─────────────────────────────────────────────────
# Config Dataclass
# ─────────────────────────────────────────────────

@dataclass
class FibrilConfig:
    """All tunable parameters for synthetic generation."""

    # Image dimensions
    image_size: int = 512                  # Output image is square: size × size

    # Main fiber (trunk)
    fiber_count_range: Tuple = (2, 5)      # Number of main fibers per image
    fiber_width_range: Tuple = (8, 20)     # Trunk width in pixels
    fiber_length_range: Tuple = (200, 420) # Control point span
    fiber_curvature: float = 0.35          # How bendy the fiber is (0=straight)
    fiber_intensity_range: Tuple = (170, 220)  # Gray value of fiber body

    # Fibrils (hair-like branches)
    fibril_count_range: Tuple = (8, 25)    # Fibrils per fiber
    fibril_length_range: Tuple = (15, 55)  # Fibril length in pixels
    fibril_width_range: Tuple = (1, 3)     # Fibril width (sub-pixel thin)
    fibril_angle_spread: float = 60.0      # Degrees spread from fiber normal
    fibril_intensity_range: Tuple = (140, 195)

    # Translucency / Alpha
    fiber_alpha_range: Tuple = (0.55, 0.85)   # How opaque each fiber is
    fibril_alpha_range: Tuple = (0.30, 0.65)

    # Background
    background_intensity: int = 235        # Near-white background (microscopy)
    background_noise_std: float = 6.0      # Gaussian noise on background

    # Degradation (simulate real microscope)
    blur_sigma_range: Tuple = (0.5, 1.5)   # Slight optical blur
    contrast_gamma_range: Tuple = (0.85, 1.15)  # Exposure variation
    global_noise_std: float = 4.0          # Global shot noise


# ─────────────────────────────────────────────────
# Bézier / Spline Helpers
# ─────────────────────────────────────────────────

def random_spline(
    start: Tuple[int, int],
    length: int,
    curvature: float,
    n_ctrl: int = 5,
    img_size: int = 512
) -> np.ndarray:
    """
    Generate a smooth random curve as an array of (x, y) points.

    Args:
        start: (x, y) starting point
        length: Approximate pixel length of the curve
        curvature: Scale of random perturbation perpendicular to main axis
        n_ctrl: Number of internal control points
        img_size: Image boundary for clamping

    Returns:
        pts: (N, 2) array of float coordinates along the curve
    """
    angle = random.uniform(0, 2 * math.pi)
    dx = math.cos(angle)
    dy = math.sin(angle)

    step = length / (n_ctrl + 1)

    ctrl_x = [start[0]]
    ctrl_y = [start[1]]

    for i in range(1, n_ctrl + 2):
        px = start[0] + dx * step * i + random.gauss(0, curvature * step)
        py = start[1] + dy * step * i + random.gauss(0, curvature * step)
        ctrl_x.append(np.clip(px, 10, img_size - 10))
        ctrl_y.append(np.clip(py, 10, img_size - 10))

    ctrl_x = np.array(ctrl_x)
    ctrl_y = np.array(ctrl_y)

    # Fit a smooth spline through control points
    try:
        tck, u = splprep([ctrl_x, ctrl_y], s=0, k=min(3, len(ctrl_x) - 1))
        u_fine = np.linspace(0, 1, int(length * 2))
        x_fine, y_fine = splev(u_fine, tck)
    except Exception:
        # Fallback: linear interpolation
        x_fine = np.interp(np.linspace(0, 1, int(length * 2)),
                            np.linspace(0, 1, len(ctrl_x)), ctrl_x)
        y_fine = np.interp(np.linspace(0, 1, int(length * 2)),
                            np.linspace(0, 1, len(ctrl_y)), ctrl_y)

    pts = np.stack([x_fine, y_fine], axis=1)
    return pts


def spline_normals(pts: np.ndarray) -> np.ndarray:
    """
    Compute unit normal vectors perpendicular to each tangent along the spline.

    Args:
        pts: (N, 2) spline points

    Returns:
        normals: (N, 2) unit normal vectors
    """
    tangents = np.gradient(pts, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
    tangents = tangents / norms
    # Normal = perpendicular to tangent
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return normals


# ─────────────────────────────────────────────────
# Drawing Primitives
# ─────────────────────────────────────────────────

def draw_fiber_on_canvas(
    canvas: np.ndarray,
    alpha_canvas: np.ndarray,
    pts: np.ndarray,
    width: int,
    intensity: int,
    alpha: float,
) -> None:
    """
    Draw a thick curved fiber body onto a floating-point RGBA canvas.

    Args:
        canvas: (H, W) float32 grayscale accumulator
        alpha_canvas: (H, W) float32 alpha accumulator
        pts: (N, 2) spline points
        width: Stroke width in pixels
        intensity: Gray intensity value (0–255)
        alpha: Opacity of this fiber layer
    """
    overlay = np.zeros_like(canvas)
    pts_int = pts.astype(np.int32)

    for i in range(len(pts_int) - 1):
        cv2.line(
            overlay,
            tuple(pts_int[i]),
            tuple(pts_int[i + 1]),
            color=float(intensity),
            thickness=width,
            lineType=cv2.LINE_AA,
        )

    # Smooth the overlay edges slightly
    overlay = gaussian_filter(overlay, sigma=0.8)

    mask = overlay > 0
    canvas[mask] = canvas[mask] * (1 - alpha) + overlay[mask] * alpha
    alpha_canvas[mask] = np.maximum(alpha_canvas[mask], alpha)


def draw_fibril_on_canvas(
    canvas: np.ndarray,
    start: Tuple[int, int],
    angle_deg: float,
    length: int,
    width: int,
    intensity: int,
    alpha: float,
) -> Tuple[int, int]:
    """
    Draw a single hair-like fibril from a point on the fiber trunk.

    Args:
        canvas: (H, W) float32 grayscale canvas
        start: (x, y) attachment point on fiber trunk
        angle_deg: Direction angle in degrees
        length: Fibril length in pixels
        width: Fibril stroke width (1–3 px)
        intensity: Gray value
        alpha: Opacity

    Returns:
        end_pt: (x, y) tip of the fibril
    """
    rad = math.radians(angle_deg)
    # Add slight random secondary bend at midpoint
    bend = random.uniform(-20, 20)
    mid_x = start[0] + int(math.cos(rad) * length * 0.5)
    mid_y = start[1] + int(math.sin(rad) * length * 0.5)
    end_x = mid_x + int(math.cos(math.radians(angle_deg + bend)) * length * 0.5)
    end_y = mid_y + int(math.sin(math.radians(angle_deg + bend)) * length * 0.5)

    overlay = np.zeros_like(canvas)
    cv2.line(overlay, start, (mid_x, mid_y),
             color=float(intensity), thickness=width, lineType=cv2.LINE_AA)
    cv2.line(overlay, (mid_x, mid_y), (end_x, end_y),
             color=float(intensity * 0.8), thickness=max(1, width - 1),
             lineType=cv2.LINE_AA)

    overlay = gaussian_filter(overlay, sigma=0.5)
    mask = overlay > 0
    canvas[mask] = canvas[mask] * (1 - alpha) + overlay[mask] * alpha

    return (end_x, end_y)


# ─────────────────────────────────────────────────
# Mask Extraction
# ─────────────────────────────────────────────────

def extract_instance_mask(
    canvas_before: np.ndarray,
    canvas_after: np.ndarray,
    threshold: float = 5.0
) -> np.ndarray:
    """
    Extract a binary instance mask for one fiber by comparing canvas before
    and after drawing it.

    Args:
        canvas_before: Canvas state BEFORE drawing this fiber
        canvas_after: Canvas state AFTER drawing this fiber
        threshold: Minimum pixel difference to count as "fiber pixel"

    Returns:
        mask: (H, W) uint8 binary mask (255 = fibril pixel)
    """
    diff = np.abs(canvas_after.astype(np.float32) - canvas_before.astype(np.float32))
    mask = (diff > threshold).astype(np.uint8) * 255
    return mask


def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """
    Convert a binary mask to a COCO-format polygon list.

    Args:
        mask: (H, W) uint8 binary mask

    Returns:
        polygons: List of [x1,y1,x2,y2,...] flat coordinate lists
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:  # Skip tiny noise contours
            continue
        # Simplify contour slightly
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) >= 4:
            flat = approx.reshape(-1).tolist()
            polygons.append(flat)
    return polygons


def mask_to_bbox(mask: np.ndarray) -> List[float]:
    """
    Compute COCO-format bounding box [x, y, width, height] from a binary mask.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]


# ─────────────────────────────────────────────────
# Core Generator Class
# ─────────────────────────────────────────────────

class FibrilSynthGenerator:
    """
    Main synthetic fibril image generator.

    Usage:
        gen = FibrilSynthGenerator(config=FibrilConfig())
        image, masks, annot = gen.generate(image_id=0)

    Returns:
        image:  (H, W) uint8 grayscale image
        masks:  List of (H, W) uint8 binary masks, one per fiber instance
        annot:  COCO annotation dict for this image
    """

    def __init__(self, config: FibrilConfig = None, seed: int = None):
        self.cfg = config or FibrilConfig()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self, image_id: int = 0):
        """
        Generate one synthetic fibril image with instance masks.

        Returns:
            image:   (H, W) uint8 grayscale
            masks:   List[(H, W) uint8]  — one per fiber instance
            coco_annotations: List[dict]  — COCO annotation dicts for this image
        """
        cfg = self.cfg
        H = W = cfg.image_size

        # ── Step 1: Create background ──────────────────────────────────
        bg = np.full((H, W), cfg.background_intensity, dtype=np.float32)
        bg_noise = np.random.normal(0, cfg.background_noise_std, (H, W))
        canvas = np.clip(bg + bg_noise, 0, 255).astype(np.float32)
        alpha_canvas = np.zeros((H, W), dtype=np.float32)

        instance_masks = []
        coco_annotations = []
        annotation_id = image_id * 1000  # Unique annot IDs per image

        n_fibers = random.randint(*cfg.fiber_count_range)

        for fiber_idx in range(n_fibers):

            # ── Step 2: Generate main fiber trunk spline ───────────────
            start_x = random.randint(30, W - 30)
            start_y = random.randint(30, H - 30)
            fiber_len = random.randint(*cfg.fiber_length_range)
            fiber_width = random.randint(*cfg.fiber_width_range)
            fiber_intensity = random.randint(*cfg.fiber_intensity_range)
            fiber_alpha = random.uniform(*cfg.fiber_alpha_range)

            trunk_pts = random_spline(
                start=(start_x, start_y),
                length=fiber_len,
                curvature=cfg.fiber_curvature,
                img_size=cfg.image_size,
            )

            # Save canvas state BEFORE drawing this fiber
            canvas_before = canvas.copy()

            # Draw trunk
            draw_fiber_on_canvas(
                canvas, alpha_canvas,
                pts=trunk_pts,
                width=fiber_width,
                intensity=fiber_intensity,
                alpha=fiber_alpha,
            )

            # ── Step 3: Generate fibrils off trunk ─────────────────────
            normals = spline_normals(trunk_pts)
            n_fibrils = random.randint(*cfg.fibril_count_range)

            # Pick random points along the trunk for fibril attachment
            attach_indices = sorted(random.sample(
                range(5, len(trunk_pts) - 5), min(n_fibrils, len(trunk_pts) - 10)
            ))

            for idx in attach_indices:
                pt = trunk_pts[idx]
                norm = normals[idx]

                # Both sides of the fiber
                for side in [1, -1]:
                    if random.random() < 0.7:  # Not every attachment spawns both sides
                        base_angle = math.degrees(math.atan2(
                            float(norm[1] * side), float(norm[0] * side)
                        ))
                        angle_jitter = random.uniform(
                            -cfg.fibril_angle_spread / 2,
                             cfg.fibril_angle_spread / 2
                        )
                        fibril_angle = base_angle + angle_jitter
                        fibril_len = random.randint(*cfg.fibril_length_range)
                        fibril_w = random.randint(*cfg.fibril_width_range)
                        fibril_intensity = random.randint(*cfg.fibril_intensity_range)
                        fibril_alpha = random.uniform(*cfg.fibril_alpha_range)

                        attach_pt = (int(np.clip(pt[0], 0, W - 1)),
                                     int(np.clip(pt[1], 0, H - 1)))

                        draw_fibril_on_canvas(
                            canvas,
                            start=attach_pt,
                            angle_deg=fibril_angle,
                            length=fibril_len,
                            width=fibril_w,
                            intensity=fibril_intensity,
                            alpha=fibril_alpha,
                        )

            # ── Step 4: Extract instance mask ──────────────────────────
            canvas_after = canvas.copy()
            mask = extract_instance_mask(canvas_before, canvas_after, threshold=4.0)

            # Only keep instances with enough pixels
            if np.sum(mask > 0) < 100:
                continue

            instance_masks.append(mask)

            # Build COCO annotation
            polygons = mask_to_polygon(mask)
            bbox = mask_to_bbox(mask)
            area = float(np.sum(mask > 0))

            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,          # 1 = "fibril"
                "segmentation": polygons,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })
            annotation_id += 1

        # ── Step 5: Apply global degradation ──────────────────────────
        # Optical blur
        blur_sigma = random.uniform(*cfg.blur_sigma_range)
        canvas = gaussian_filter(canvas, sigma=blur_sigma)

        # Gamma / contrast variation
        gamma = random.uniform(*cfg.contrast_gamma_range)
        canvas = np.clip(canvas, 1, 255)
        canvas = 255.0 * (canvas / 255.0) ** gamma

        # Global shot noise
        noise = np.random.normal(0, cfg.global_noise_std, canvas.shape)
        canvas = np.clip(canvas + noise, 0, 255)

        image = canvas.astype(np.uint8)

        return image, instance_masks, coco_annotations
