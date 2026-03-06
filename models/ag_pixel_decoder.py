"""
ag_pixel_decoder.py — Attention-Gated U-Net Pixel Decoder
==========================================================
Replaces the standard FPN pixel decoder in Mask2Former with an
Attention-Gated U-Net. This makes the model focus on high-frequency
fibril pixels and suppress background noise — solving the low-contrast problem.

Architecture:
    C4 (coarsest) → Up → AG(C3+gate) → Up → AG(C2+gate) → Up → AG(C1+gate) → output

Attention Gate Formula:
    α = σ(W_x·x + W_g·g + b)
    x_hat = α ⊙ x

Input:   features: List[(B, Ci, Hi, Wi)]  — from Swin backbone
Output:  (B, hidden_dim, H/4, W/4)         — per-pixel embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """Two consecutive ConvBNReLU blocks — standard U-Net building block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Additive Attention Gate (Oktay et al., 2018).

    Filters skip-connection features (x) using a gating signal (g)
    from the decoder. Learns to highlight fibril-relevant regions
    and suppress background noise.

    Args:
        F_g: Channel dim of gating signal (from decoder)
        F_l: Channel dim of skip connection (from encoder)
        F_int: Intermediate channel dim (typically F_l // 2)

    Math:
        ψ = ReLU(W_g·g + W_x·x + b_g)
        α = σ(W_ψ·ψ + b_ψ)
        x_hat = α ⊙ x
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        # Gate path: 1×1 conv on decoder features
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # Skip path: 1×1 conv on encoder features
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # Attention coefficient: 1×1 conv → sigmoid
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: (B, F_g, Hg, Wg) — gating signal from decoder (coarse)
            x: (B, F_l, Hx, Wx) — skip features from encoder (fine)

        Returns:
            x_hat: (B, F_l, Hx, Wx) — attention-weighted features
        """
        # Upsample gate to match skip dimension if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)

        g1 = self.W_g(g)   # (B, F_int, H, W)
        x1 = self.W_x(x)   # (B, F_int, H, W)

        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)  # (B, 1, H, W) attention map

        return x * alpha   # Broadcast multiply — focus on fibril regions


# ─────────────────────────────────────────────────────────────────────────────
# Main AG-UNet Pixel Decoder
# ─────────────────────────────────────────────────────────────────────────────

class AGUNetPixelDecoder(nn.Module):
    """
    Attention-Gated U-Net Pixel Decoder for Mask2Former.

    Takes 4-scale features from the Swin-T backbone and produces
    high-resolution per-pixel embeddings for the Mask2Former decoder.

    Layer flow (bottom-up):
        C4 → conv → up
           ↓ AG(gate=up, skip=C3) → cat → DoubleConv → up
           ↓ AG(gate=up, skip=C2) → cat → DoubleConv → up
           ↓ AG(gate=up, skip=C1) → cat → DoubleConv
           ↓ final 1×1 conv → pixel embeddings

    Args:
        in_channels:  List of channel dims from backbone [C1, C2, C3, C4]
        hidden_dim:   Output embedding dimension (default 256)
    """

    def __init__(
        self,
        in_channels: List[int] = [96, 192, 384, 768],  # Swin-T defaults
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Lateral projections: bring all scales to hidden_dim
        self.lat4 = ConvBNReLU(in_channels[3], hidden_dim)  # C4 → 256
        self.lat3 = ConvBNReLU(in_channels[2], hidden_dim)  # C3 → 256
        self.lat2 = ConvBNReLU(in_channels[1], hidden_dim)  # C2 → 256
        self.lat1 = ConvBNReLU(in_channels[0], hidden_dim)  # C1 → 256

        # Attention gates
        # AG at level 3: gate from level 4 output, skip from C3
        self.ag3 = AttentionGate(F_g=hidden_dim, F_l=hidden_dim, F_int=hidden_dim // 2)
        self.ag2 = AttentionGate(F_g=hidden_dim, F_l=hidden_dim, F_int=hidden_dim // 2)
        self.ag1 = AttentionGate(F_g=hidden_dim, F_l=hidden_dim, F_int=hidden_dim // 2)

        # After concatenation (gated_skip + upsampled_decoder), reduce channels
        self.up3 = DoubleConv(hidden_dim * 2, hidden_dim)
        self.up2 = DoubleConv(hidden_dim * 2, hidden_dim)
        self.up1 = DoubleConv(hidden_dim * 2, hidden_dim)

        # Final output projection
        self.output_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # Multi-scale outputs for Mask2Former (pixel decoder returns multiple scales)
        self.aux_proj3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.aux_proj2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features: [C1, C2, C3, C4] from Swin backbone
                C1: (B, 96,  H/4,  W/4)
                C2: (B, 192, H/8,  W/8)
                C3: (B, 384, H/16, W/16)
                C4: (B, 768, H/32, W/32)

        Returns:
            pixel_embeddings:    (B, 256, H/4, W/4)  ← used by transformer decoder
            aux_features:        List of multi-scale features for auxiliary losses
        """
        c1, c2, c3, c4 = features

        # ── Lateral projections ────────────────────────────────────────
        p4 = self.lat4(c4)   # (B, 256, H/32, W/32)
        p3 = self.lat3(c3)   # (B, 256, H/16, W/16)
        p2 = self.lat2(c2)   # (B, 256, H/8,  W/8)
        p1 = self.lat1(c1)   # (B, 256, H/4,  W/4)

        # ── Decoder level 3 ───────────────────────────────────────────
        d4_up = F.interpolate(p4, size=p3.shape[2:], mode="bilinear", align_corners=True)
        p3_attend = self.ag3(g=d4_up, x=p3)   # Attention-gated skip
        d3 = self.up3(torch.cat([d4_up, p3_attend], dim=1))  # (B, 256, H/16, W/16)

        # ── Decoder level 2 ───────────────────────────────────────────
        d3_up = F.interpolate(d3, size=p2.shape[2:], mode="bilinear", align_corners=True)
        p2_attend = self.ag2(g=d3_up, x=p2)
        d2 = self.up2(torch.cat([d3_up, p2_attend], dim=1))  # (B, 256, H/8, W/8)

        # ── Decoder level 1 ───────────────────────────────────────────
        d2_up = F.interpolate(d2, size=p1.shape[2:], mode="bilinear", align_corners=True)
        p1_attend = self.ag1(g=d2_up, x=p1)
        d1 = self.up1(torch.cat([d2_up, p1_attend], dim=1))  # (B, 256, H/4, W/4)

        # ── Final embeddings ──────────────────────────────────────────
        pixel_embeddings = self.output_proj(d1)            # (B, 256, H/4, W/4)

        # Multi-scale aux features (for auxiliary losses during training)
        aux_features = [
            self.aux_proj3(d3),   # (B, 256, H/16, W/16)
            self.aux_proj2(d2),   # (B, 256, H/8,  W/8)
            pixel_embeddings,     # (B, 256, H/4,  W/4)
        ]

        return pixel_embeddings, aux_features
