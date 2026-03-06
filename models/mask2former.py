"""
mask2former.py — Stage 3: Full Instance Segmentation Model
===========================================================
Combines:
  1. Swin-T Backbone       → multi-scale feature maps
  2. AG-UNet Pixel Decoder → per-pixel embeddings (attention-gated)
  3. Transformer Decoder   → object queries → instance masks

Key innovations:
  - Masked Cross-Attention: each query only attends to its predicted mask region
  - Hungarian Matching:     1-to-1 GT ↔ query assignment during training
  - 50 object queries:      each learns ONE fibril instance

Input:  (B, 1, H, W) grayscale image
Output:
  - pred_masks:   (B, Q, H/4, W/4)  Q=50 predicted masks
  - pred_logits:  (B, Q, num_classes+1)  class scores per query
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

from models.swin_backbone import SwinBackbone
from models.ag_pixel_decoder import AGUNetPixelDecoder


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionEmbeddingSine2D(nn.Module):
    """
    2D sinusoidal positional encoding.
    Encodes spatial (x, y) position using sin/cos at different frequencies.
    Added to flattened feature maps before transformer attention.
    """

    def __init__(self, hidden_dim: int, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be even for sin/cos encoding"
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * 3.14159265

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map

        Returns:
            pe: (B, C, H, W) positional encoding
        """
        B, C, H, W = x.shape
        device = x.device

        y_embed = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0).expand(H, W)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (H + eps) * self.scale
            x_embed = x_embed / (W + eps) * self.scale

        dim_t = torch.arange(C // 2, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (C // 2))

        pe_x = x_embed.unsqueeze(-1) / dim_t                          # (H, W, C//2)
        pe_y = y_embed.unsqueeze(-1) / dim_t

        pe_x = torch.stack([pe_x[..., 0::2].sin(), pe_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pe_y = torch.stack([pe_y[..., 0::2].sin(), pe_y[..., 1::2].cos()], dim=-1).flatten(-2)

        pe = torch.cat([pe_y, pe_x], dim=-1)   # (H, W, C)
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        pe = pe.expand(B, -1, -1, -1)

        return pe


# ─────────────────────────────────────────────────────────────────────────────
# Masked Cross-Attention Layer (Key Mask2Former Innovation)
# ─────────────────────────────────────────────────────────────────────────────

class MaskedCrossAttentionLayer(nn.Module):
    """
    Cross-attention where each query ONLY attends within its predicted mask region.

    This is the core novelty of Mask2Former:
      Standard cross-attention: query looks at ALL pixels → noisy gradients
      Masked cross-attention:   query looks at ONLY its mask region → clean gradients

    For fibril segmentation:
      Query 1 learns Fibril A → only attends to pixels in Fibril A's mask
      Query 2 learns Fibril B → only attends to pixels in Fibril B's mask
      → Crossing fibers are separated naturally!

    Args:
        d_model:   Query/key/value embedding dimension
        nhead:     Number of attention heads
        dropout:   Dropout probability
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,           # (B, Q, d_model)
        key_value: torch.Tensor,        # (B, HW, d_model) — flattened pixel features
        attn_mask: Optional[torch.Tensor] = None,  # (B*nhead, Q, HW) — binary mask
    ) -> torch.Tensor:
        """
        Args:
            query:     (B, Q, d_model) object queries
            key_value: (B, HW, d_model) pixel embeddings
            attn_mask: (B*nhead, Q, HW) — True = ATTEND, False = IGNORE

        Returns:
            out: (B, Q, d_model) updated queries
        """
        # MultiheadAttention expects attn_mask as (Q, HW) or (B*nhead, Q, HW)
        # True = positions to be masked (IGNORED), so we invert our mask
        if attn_mask is not None:
            # Our mask: 1=fibril region, 0=background
            # PyTorch convention: True = MASK OUT (ignore), so invert
            attn_mask_bool = ~attn_mask.bool()
        else:
            attn_mask_bool = None

        attn_out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=attn_mask_bool,
        )

        out = self.norm(query + self.dropout(attn_out))
        return out


class TransformerDecoderLayer(nn.Module):
    """
    One Mask2Former decoder layer:
      1. Self-attention among queries
      2. Masked cross-attention with pixel features
      3. Feed-forward network
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, dim_ff: int = 2048, dropout: float = 0.0):
        super().__init__()

        # Self-attention: queries talk to each other
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Masked cross-attention: queries attend to masked pixel features
        self.cross_attn = MaskedCrossAttentionLayer(d_model, nhead, dropout)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        pixel_features: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        sa_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + self.dropout(sa_out))

        # Masked cross-attention
        queries = self.cross_attn(queries, pixel_features, attn_mask)

        # Feed-forward
        ff_out = self.ff(queries)
        queries = self.norm3(queries + self.dropout(ff_out))

        return queries


# ─────────────────────────────────────────────────────────────────────────────
# Mask Prediction Head
# ─────────────────────────────────────────────────────────────────────────────

class MaskHead(nn.Module):
    """
    Predicts per-query binary masks by dot-product between
    query embeddings and per-pixel embeddings.

    mask[q, i, j] = σ(query[q] · pixel[i, j])
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # Project query to mask space
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,        # (B, Q, hidden_dim)
        pixel_embeddings: torch.Tensor,  # (B, hidden_dim, H, W)
    ) -> torch.Tensor:
        """
        Returns:
            masks: (B, Q, H, W) logits — one mask per query
        """
        B, Q, D = queries.shape
        # Project queries
        mask_queries = self.mask_embed(queries)   # (B, Q, D)

        # Reshape pixel embeddings: (B, D, H, W) → (B, HW, D)
        B, D, H, W = pixel_embeddings.shape
        pixels_flat = pixel_embeddings.flatten(2).permute(0, 2, 1)  # (B, HW, D)

        # Dot product: (B, Q, D) × (B, D, HW) → (B, Q, HW)
        masks_flat = torch.einsum("bqd,bhd->bqh", mask_queries, pixels_flat)
        masks = masks_flat.view(B, Q, H, W)

        return masks


# ─────────────────────────────────────────────────────────────────────────────
# Full Mask2Former Model
# ─────────────────────────────────────────────────────────────────────────────

class FibrilMask2Former(nn.Module):
    """
    Complete instance segmentation model combining:
      - Swin-T backbone
      - AG-UNet pixel decoder
      - Mask2Former transformer decoder

    Optimized for free-tier hardware (T4/P100):
      - 50 object queries (vs 100 in original)
      - 6 decoder layers (vs 9)
      - 256 hidden dim

    Args:
        num_classes:    1 (fibril) + 1 (background) = 2
        num_queries:    Number of object queries (= max fibrils per image)
        hidden_dim:     Transformer embedding dimension
        nheads:         Attention heads
        num_dec_layers: Transformer decoder layers
        backbone_variant: "swin_tiny" | "swin_small"
    """

    def __init__(
        self,
        num_classes: int = 1,         # Binary: fibril or not
        num_queries: int = 50,        # Max fibrils per image
        hidden_dim: int = 256,
        nheads: int = 8,
        num_dec_layers: int = 6,
        backbone_variant: str = "swin_tiny",
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.nheads = nheads          # Store so forward() can use it (not hardcoded!)

        # ── Stage 1 can be called externally (ESRGAN), not part of this model ──

        # ── Stage 2: Swin-T Backbone ──────────────────────────────────
        self.backbone = SwinBackbone(
            variant=backbone_variant,
            pretrained=pretrained_backbone,
            in_channels=1,
        )
        backbone_channels = self.backbone.feature_channels  # [96, 192, 384, 768]

        # ── Pixel Decoder: AG-UNet ────────────────────────────────────
        self.pixel_decoder = AGUNetPixelDecoder(
            in_channels=backbone_channels,
            hidden_dim=hidden_dim,
        )

        # ── Positional encoding ───────────────────────────────────────
        self.pos_embed = PositionEmbeddingSine2D(hidden_dim)

        # ── Object Queries (learnable embeddings) ─────────────────────
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)

        # ── Input projection: pixel_embed → transformer dim ───────────
        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # ── Transformer Decoder Layers ────────────────────────────────
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dim_ff=hidden_dim * 8,
                dropout=0.0,
            )
            for _ in range(num_dec_layers)
        ])

        # ── Prediction Heads ──────────────────────────────────────────
        # Class prediction: query → class logit
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object"

        # Mask prediction: query × pixel → mask logit
        self.mask_head = MaskHead(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for non-pretrained components."""
        for p in self.class_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.normal_(self.query_feat.weight, std=0.02)

    def forward(
        self,
        images: torch.Tensor,
        gt_masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """
        Forward pass.

        Args:
            images:   (B, 1, H, W) grayscale image tensors
            gt_masks: Optional — used during training for mask-guided attention init

        Returns:
            dict with keys:
                "pred_logits": (B, Q, num_classes+1)   class scores
                "pred_masks":  (B, Q, H/4, W/4)        mask logits
                "aux_outputs": List of intermediate predictions (for aux losses)
        """
        B = images.shape[0]

        # ── Extract multi-scale features ─────────────────────────────
        features = self.backbone(images)   # [C1, C2, C3, C4]

        # ── Decode to per-pixel embeddings ────────────────────────────
        pixel_emb, aux_feats = self.pixel_decoder(features)
        # pixel_emb: (B, 256, H/4, W/4)

        # ── Prepare for transformer: flatten spatial dims ─────────────
        proj = self.input_proj(pixel_emb)                  # (B, 256, H/4, W/4)
        pos = self.pos_embed(proj)                          # (B, 256, H/4, W/4)
        proj_pos = proj + pos                               # Add positional info
        B, D, H4, W4 = proj_pos.shape
        pixel_flat = proj_pos.flatten(2).permute(0, 2, 1)  # (B, H4*W4, D)

        # ── Initialize object queries ─────────────────────────────────
        queries = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)   # (B, Q, D)
        query_pos = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)
        queries = queries + query_pos

        # ── Transformer decoder layers ────────────────────────────────
        aux_outputs = []
        pred_masks = None

        for i, layer in enumerate(self.decoder_layers):
            # Generate binary attention mask from current mask predictions
            if pred_masks is not None:
                # Binarize: 1 = predicted foreground region for each query
                attn_mask = (pred_masks.sigmoid() > 0.5).detach()  # (B, Q, H4, W4)
                attn_mask = attn_mask.flatten(2)                    # (B, Q, H4*W4)
                # Expand for multi-head: (B*nheads, Q, HW)
                # IMPORTANT: use self.nheads, NOT hardcoded 8
                attn_mask = attn_mask.unsqueeze(1).expand(
                    -1, self.nheads, -1, -1
                ).reshape(B * self.nheads, self.num_queries, H4 * W4)
            else:
                attn_mask = None   # First layer: no mask constraint

            queries = layer(queries, pixel_flat, attn_mask)

            # Predict masks at each layer (for auxiliary losses + next layer's mask)
            pred_masks = self.mask_head(queries, pixel_emb)       # (B, Q, H4, W4)
            pred_logits = self.class_head(queries)                  # (B, Q, num_cls+1)

            if i < len(self.decoder_layers) - 1:
                aux_outputs.append({
                    "pred_logits": pred_logits,
                    "pred_masks": pred_masks,
                })

        return {
            "pred_logits": pred_logits,    # (B, Q, num_classes+1)
            "pred_masks": pred_masks,       # (B, Q, H/4, W/4)
            "aux_outputs": aux_outputs,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: dict, device: str = "cuda") -> FibrilMask2Former:
    """
    Build the FibrilMask2Former model from a config dict.

    Args:
        config: Dict with keys matching FibrilMask2Former __init__ args
        device: "cuda" or "cpu"

    Returns:
        model: FibrilMask2Former on the specified device
    """
    model = FibrilMask2Former(
        num_classes=config.get("num_classes", 1),
        num_queries=config.get("num_queries", 50),
        hidden_dim=config.get("hidden_dim", 256),
        nheads=config.get("nheads", 8),
        num_dec_layers=config.get("dec_layers", 6),
        backbone_variant=config.get("backbone", "swin_tiny"),
        pretrained_backbone=config.get("pretrained", True),
    )

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] FibrilMask2Former built ✅")
    print(f"        Total params:     {total_params/1e6:.1f}M")
    print(f"        Trainable params: {trainable/1e6:.1f}M")
    print(f"        Device:           {device}\n")

    return model
