"""
swin_backbone.py — Stage 2: Swin-Transformer Feature Extractor
==============================================================
Wraps timm's Swin-Tiny (28M params, fits T4 VRAM comfortably).
Returns 4-scale hierarchical feature maps: C1, C2, C3, C4.

Input:  (B, 1, H, W) grayscale
Output: List[(B, Ci, Hi, Wi)] multi-scale feature maps

Using Swin-Tiny:
  C1: (B, 96,  H/4,  W/4)
  C2: (B, 192, H/8,  W/8)
  C3: (B, 384, H/16, W/16)
  C4: (B, 768, H/32, W/32)

Pretrained: ImageNet-1K (via timm). The first patch-embed conv
is patched to accept 1-channel grayscale input.
"""

import torch
import torch.nn as nn
from typing import List, Dict

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARNING] timm not installed. Run: pip install timm")


# Feature channel dims per Swin variant
SWIN_CHANNELS = {
    "swin_tiny":  [96, 192, 384, 768],
    "swin_small": [96, 192, 384, 768],
    "swin_base":  [128, 256, 512, 1024],
}


class SwinBackbone(nn.Module):
    """
    Swin-Transformer backbone for multi-scale feature extraction.

    Args:
        variant:    "swin_tiny" | "swin_small" | "swin_base"
                    Use swin_tiny for T4 (15GB VRAM)
                    Use swin_small for P100 (16GB VRAM)
        pretrained: Load ImageNet-1K pretrained weights (recommended)
        in_channels: 1 for grayscale, 3 for RGB
        out_indices:  Which stages to return features from (0-indexed)
        freeze_stages: Freeze first N stages (0 = freeze none)
    """

    def __init__(
        self,
        variant: str = "swin_tiny",
        pretrained: bool = True,
        in_channels: int = 1,
        out_indices: tuple = (0, 1, 2, 3),
        freeze_stages: int = 0,
    ):
        super().__init__()
        self.variant = variant
        self.out_indices = out_indices
        self.out_channels = SWIN_CHANNELS[variant]

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required: pip install timm")

        # ── Load timm model ──────────────────────────────────────────
        timm_name_map = {
            "swin_tiny":  "swin_tiny_patch4_window7_224",
            "swin_small": "swin_small_patch4_window7_224",
            "swin_base":  "swin_base_patch4_window7_224",
        }
        timm_name = timm_name_map[variant]

        self.swin = timm.create_model(
            timm_name,
            pretrained=pretrained,
            features_only=True,        # Return intermediate features
            out_indices=out_indices,   # Which stages
            dynamic_img_size=True,     # Accept any H×W (recalculates pos bias)
            strict_img_size=False,     # Disable runtime size assertions for non-square paddings
        )

        # ── Adapt first patch embedding for grayscale input ──────────
        if in_channels != 3:
            self._adapt_input_channels(in_channels)

        # ── Gradient checkpointing (saves ~30% VRAM) ─────────────────
        if hasattr(self.swin, "set_grad_checkpointing"):
            self.swin.set_grad_checkpointing(enable=True)

        # ── Freeze stages if requested ────────────────────────────────
        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

        print(
            f"[SwinBackbone] Loaded {variant} | "
            f"pretrained={pretrained} | in_channels={in_channels} | "
            f"out_channels={self.out_channels}"
        )

    def _adapt_input_channels(self, in_channels: int):
        """
        Patch the first patch embedding conv to accept `in_channels` inputs.

        For grayscale (in_channels=1):
          The pretrained 3-channel weights are averaged to 1 channel,
          preserving the learned filter patterns.
        """
        # Find the patch embedding conv
        patch_embed = None
        for name, module in self.swin.named_modules():
            if isinstance(module, nn.Conv2d) and "patch_embed" in name:
                patch_embed = (name, module)
                break

        if patch_embed is None:
            # Try direct attribute access
            if hasattr(self.swin, "patch_embed"):
                pe = self.swin.patch_embed
                if hasattr(pe, "proj"):
                    patch_embed = ("patch_embed.proj", pe.proj)

        if patch_embed is not None:
            name, conv = patch_embed
            old_weight = conv.weight.data  # (out, 3, kH, kW)
            new_weight = old_weight.mean(dim=1, keepdim=True)  # (out, 1, kH, kW)

            # Create new conv with in_channels channels
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=conv.bias is not None,
            )
            new_conv.weight.data = new_weight
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data.clone()

            # Replace in model
            parts = name.split(".")
            obj = self.swin
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], new_conv)

            print(f"[SwinBackbone] Adapted patch embed: 3→{in_channels} channels ✅")

    def _freeze_stages(self, n_stages: int):
        """Freeze first n_stages stages of the backbone."""
        frozen = 0
        for i, stage in enumerate(self.swin.stages if hasattr(self.swin, "stages") else []):
            if i < n_stages:
                for param in stage.parameters():
                    param.requires_grad = False
                frozen += 1

        # Always freeze patch embedding
        if hasattr(self.swin, "patch_embed"):
            for param in self.swin.patch_embed.parameters():
                param.requires_grad = False
        print(f"[SwinBackbone] Froze patch_embed + {frozen} stages")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: (B, 1, H, W) grayscale image tensor

        Returns:
            features: List of tensors at requested out_indices
                      Typically: [C1, C2, C3, C4]
                      C1: (B, 96,  H/4,  W/4)
                      C2: (B, 192, H/8,  W/8)
                      C3: (B, 384, H/16, W/16)
                      C4: (B, 768, H/32, W/32)
        """
        import torch.nn.functional as F

        # ── Resize to nearest valid Swin-T size ─────────────────────
        # Swin-T requires H and W divisible by (patch_size × window_size) = 4×7 = 28
        # e.g. 224 (8×28), 448 (16×28). If input is 256, resize to 224.
        _, _, H, W = x.shape
        stride = 28  # patch_size(4) × window_size(7)
        target_H = max(stride, round(H / stride) * stride)
        target_W = max(stride, round(W / stride) * stride)
        if target_H != H or target_W != W:
            x = F.interpolate(x, size=(target_H, target_W),
                              mode="bilinear", align_corners=False)

        # timm swin returns features as (B, H, W, C) — need to permute to (B, C, H, W)
        raw_features = self.swin(x)

        features = []
        for feat in raw_features:
            if feat.ndim == 4 and feat.shape[-1] != feat.shape[1]:
                # (B, H, W, C) → (B, C, H, W)
                feat = feat.permute(0, 3, 1, 2).contiguous()
            features.append(feat)

        return features

    @property
    def feature_channels(self) -> List[int]:
        """Return output channel dimensions for each stage."""
        return [self.out_channels[i] for i in self.out_indices]

