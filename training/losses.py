"""
losses.py — Loss Functions + Hungarian Matcher
===============================================
Implements the Mask2Former loss suite:
  1. Focal Loss        — handles class imbalance (many "no-object" queries)
  2. Dice Loss         — measures mask overlap quality
  3. Binary Focal Loss — per-pixel mask supervision
  4. HungarianMatcher  — optimal 1-to-1 matching of queries to GT instances

The total loss is:
  L = λ_cls × L_focal + λ_mask × L_binary_focal + λ_dice × L_dice
  Weights: cls=2.0, mask=5.0, dice=5.0 (from original paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss (Classification)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for query classification.
    Down-weights easy "no-object" predictions, focuses on hard fibril queries.

    FL(p) = -α × (1-p_t)^γ × log(p_t)

    Args:
        alpha:    Class balance weight (default 0.25)
        gamma:    Focusing parameter (default 2.0)
        reduction: "mean" | "sum" | "none"
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  (N, C) raw logits
            targets: (N,) integer class labels

        Returns:
            loss: Scalar focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Dice Loss (Mask Quality)
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary mask quality.

    Dice = 2 × |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice

    Better than BCE for imbalanced binary masks (fibrils are thin → few positive pixels).

    Args:
        smooth: Laplace smoothing to avoid division by zero
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  (N, H, W) — raw logits (before sigmoid)
            targets: (N, H, W) — binary ground truth masks (0.0 or 1.0)

        Returns:
            loss: Scalar dice loss (1 - dice_coefficient)
        """
        inputs_sig = inputs.sigmoid().flatten(1)    # (N, HW)
        targets_flat = targets.flatten(1).float()   # (N, HW)

        intersection = (inputs_sig * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            inputs_sig.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        return (1 - dice).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Binary Focal Loss (Per-pixel mask supervision)
# ─────────────────────────────────────────────────────────────────────────────

def binary_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Binary focal loss for per-pixel mask prediction.

    Args:
        inputs:  (N, H, W) raw logits
        targets: (N, H, W) binary targets (0 or 1)
        alpha, gamma: Focal loss hyperparameters

    Returns:
        Scalar loss
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Hungarian Matcher
# ─────────────────────────────────────────────────────────────────────────────

class HungarianMatcher(nn.Module):
    """
    Optimal bipartite matching between predicted queries and ground truth instances.

    Solves: σ* = argmin_{σ} Σ C(pred_σ(i), GT_i)

    Where cost C = cost_class + cost_mask + cost_dice

    This ensures:
      - Each GT fibril is matched to exactly ONE query
      - No two queries match the same GT instance
      - Unmatched queries → "no-object" class

    Args:
        cost_class: Weight for classification cost
        cost_mask:  Weight for mask binary focal cost
        cost_dice:  Weight for dice cost
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,   # (B, Q, num_classes+1)
        pred_masks: torch.Tensor,    # (B, Q, H, W)
        gt_labels: List[torch.Tensor],  # List[(N_i,)] per image
        gt_masks: List[torch.Tensor],   # List[(N_i, H, W)] per image
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute optimal matching for each image in the batch.

        Returns:
            List of (query_indices, gt_indices) tuples — one per image.
            query_indices[j] should be matched to gt_indices[j].
        """
        B, Q = pred_logits.shape[:2]
        results = []

        for b in range(B):
            q_logits = pred_logits[b]     # (Q, num_classes+1)
            q_masks  = pred_masks[b]      # (Q, H, W)
            labels   = gt_labels[b]       # (N,)
            masks    = gt_masks[b]        # (N, H, W)

            N = len(labels)
            if N == 0:
                results.append((
                    torch.zeros(0, dtype=torch.int64),
                    torch.zeros(0, dtype=torch.int64),
                ))
                continue

            # ── Classification cost (negative softmax prob of GT class) ──
            # We want the query whose class prob matches GT label
            class_probs = q_logits.softmax(dim=-1)          # (Q, num_cls+1)
            cost_class = -class_probs[:, labels]             # (Q, N) — neg prob = lower is better

            # ── Mask costs (downsample to speed up matching) ─────────
            # Downsample to 64×64 for efficiency during matching
            H, W = q_masks.shape[1:]
            ds = 64
            if H > ds or W > ds:
                q_masks_ds = F.interpolate(
                    q_masks.unsqueeze(0), size=(ds, ds), mode="bilinear", align_corners=False
                ).squeeze(0)   # (Q, ds, ds)
                m_masks_ds = F.interpolate(
                    masks.float().unsqueeze(0), size=(ds, ds), mode="bilinear", align_corners=False
                ).squeeze(0)   # (N, ds, ds)
            else:
                q_masks_ds = q_masks
                m_masks_ds = masks.float()

            # Binary focal cost matrix
            q_flat = q_masks_ds.flatten(1)     # (Q, ds*ds)
            m_flat = m_masks_ds.flatten(1)     # (N, ds*ds)

            # cost_mask[q, n] = binary_focal_loss(pred_q, gt_n)
            cost_mask_matrix = torch.zeros(Q, N, device=q_logits.device)
            cost_dice_matrix = torch.zeros(Q, N, device=q_logits.device)

            for n in range(N):
                gt_n = m_flat[n].unsqueeze(0).expand(Q, -1)   # (Q, ds*ds)
                # Binary focal
                prob = q_flat.sigmoid()
                bce = F.binary_cross_entropy_with_logits(q_flat, gt_n, reduction="none")
                p_t = prob * gt_n + (1 - prob) * (1 - gt_n)
                focal = (0.25 * ((1 - p_t) ** 2.0) * bce).mean(dim=1)  # (Q,)
                cost_mask_matrix[:, n] = focal

                # Dice cost
                inter = (q_flat.sigmoid() * gt_n).sum(dim=1)
                dice = 1 - (2 * inter + 1) / (q_flat.sigmoid().sum(1) + gt_n.sum(1) + 1)
                cost_dice_matrix[:, n] = dice

            # ── Combined cost matrix ──────────────────────────────────
            C = (
                self.cost_class * cost_class.cpu()
                + self.cost_mask  * cost_mask_matrix.cpu()
                + self.cost_dice  * cost_dice_matrix.cpu()
            ).numpy()   # (Q, N)

            # ── Solve assignment (Hungarian algorithm) ────────────────
            q_idx, gt_idx = linear_sum_assignment(C)

            results.append((
                torch.tensor(q_idx,  dtype=torch.int64),
                torch.tensor(gt_idx, dtype=torch.int64),
            ))

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Combined Mask2Former Loss
# ─────────────────────────────────────────────────────────────────────────────

class Mask2FormerLoss(nn.Module):
    """
    Full Mask2Former loss combining classification + mask losses.
    Applied at final decoder layer AND all auxiliary layers.

    L_total = L_final + Σ L_aux_i  (sum over auxiliary decoder layers)

    Args:
        num_classes:     Number of object classes (1 = fibril)
        matcher:         HungarianMatcher instance
        weight_dict:     Loss component weights
        eos_coef:        "No object" class weight (down-weight)
    """

    def __init__(
        self,
        num_classes: int = 1,
        matcher: HungarianMatcher = None,
        weight_dict: dict = None,
        eos_coef: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher(
            cost_class=1.0, cost_mask=1.0, cost_dice=1.0
        )
        self.weight_dict = weight_dict or {
            "loss_ce": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 5.0,
        }

        # Class weights: down-weight "no object" class
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef   # Last class = "no object"
        self.register_buffer("empty_weight", empty_weight)

    def _get_target_classes(
        self,
        pred_logits: torch.Tensor,
        matched_indices: List[Tuple],
        gt_labels: List[torch.Tensor],
    ) -> torch.Tensor:
        """Build (B*Q,) target class tensor, unmatched queries → no-object class."""
        B, Q, _ = pred_logits.shape
        target = torch.full(
            (B, Q), self.num_classes,  # Default: "no object" class index
            dtype=torch.int64,
            device=pred_logits.device,
        )
        for b, (q_idx, gt_idx) in enumerate(matched_indices):
            if len(q_idx) == 0:
                continue
            target[b, q_idx] = gt_labels[b][gt_idx]
        return target.flatten()  # (B*Q,)

    def _compute_mask_losses(
        self,
        pred_masks: torch.Tensor,
        matched_indices: List[Tuple],
        gt_masks: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute binary focal + dice losses for matched query-GT pairs."""
        all_pred, all_gt = [], []

        for b, (q_idx, gt_idx) in enumerate(matched_indices):
            if len(q_idx) == 0:
                continue
            # Match predictions to GT
            pm = pred_masks[b][q_idx]  # (K, H, W)
            gm = gt_masks[b][gt_idx].float()    # (K, H, W)

            # Downsample GT to match pred size if needed
            if pm.shape[1:] != gm.shape[1:]:
                gm = F.interpolate(
                    gm.unsqueeze(0), size=pm.shape[1:],
                    mode="bilinear", align_corners=False
                ).squeeze(0)

            all_pred.append(pm)
            all_gt.append(gm)

        if not all_pred:
            device = pred_masks.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        preds = torch.cat(all_pred, dim=0)  # (total_matched, H, W)
        gts   = torch.cat(all_gt,   dim=0)

        loss_focal = binary_focal_loss(preds, gts)
        loss_dice  = DiceLoss()(preds, gts)

        return loss_focal, loss_dice

    def forward(
        self,
        outputs: Dict,
        gt_labels: List[torch.Tensor],
        gt_masks: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total Mask2Former loss.

        Args:
            outputs:   Dict with "pred_logits", "pred_masks", "aux_outputs"
            gt_labels: List of (N_i,) tensors — class IDs per image
            gt_masks:  List of (N_i, H, W) tensors — GT binary masks

        Returns:
            Dict of individual + total losses
        """
        all_losses = {}
        total_loss = torch.tensor(0.0, device=outputs["pred_logits"].device)

        # Compute for final output + all auxiliary outputs
        for layer_idx, layer_out in enumerate(
            outputs["aux_outputs"] + [{"pred_logits": outputs["pred_logits"],
                                       "pred_masks":  outputs["pred_masks"]}]
        ):
            suffix = f"_aux{layer_idx}" if layer_idx < len(outputs["aux_outputs"]) else ""

            pred_logits = layer_out["pred_logits"]
            pred_masks_raw  = layer_out["pred_masks"]

            # Upsample predicted masks to match GT resolution (H, W)
            H, W = gt_masks[0].shape[-2:]
            pred_masks = F.interpolate(
                pred_masks_raw,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

            # Match queries to GT
            matched = self.matcher(pred_logits, pred_masks, gt_labels, gt_masks)

            # Classification loss
            tgt_classes = self._get_target_classes(pred_logits, matched, gt_labels)
            loss_ce = F.cross_entropy(
                pred_logits.flatten(0, 1),
                tgt_classes,
                weight=self.empty_weight,
            )

            # Mask losses
            loss_mask, loss_dice = self._compute_mask_losses(pred_masks, matched, gt_masks)

            # Weighted combination
            layer_loss = (
                self.weight_dict["loss_ce"] * loss_ce
                + self.weight_dict["loss_mask"] * loss_mask
                + self.weight_dict["loss_dice"] * loss_dice
            )

            all_losses[f"loss_ce{suffix}"]   = loss_ce
            all_losses[f"loss_mask{suffix}"] = loss_mask
            all_losses[f"loss_dice{suffix}"] = loss_dice
            total_loss = total_loss + layer_loss

        all_losses["total"] = total_loss
        return all_losses
