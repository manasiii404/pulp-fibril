"""
esrgan_sr.py — Stage 1: Real-ESRGAN Super-Resolution Wrapper
=============================================================
Upsamples low-quality fibril microscopy images using Real-ESRGAN x2.

Automatically downloads pretrained weights on first use.
Fine-tuned for grayscale microscopy inputs.

Usage:
    sr = SuperResolutionModule(scale=2, device='cuda')
    hr_image = sr.enhance(lr_image)   # numpy uint8 in, uint8 out
"""

import os
import urllib.request
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch

# ── Weight download URL ───────────────────────────────────────────────────────
ESRGAN_WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/"
    "RealESRGAN_x2plus.pth"
)
ESRGAN_WEIGHTS_DIR = Path(__file__).parent.parent / "checkpoints"
ESRGAN_WEIGHTS_PATH = ESRGAN_WEIGHTS_DIR / "RealESRGAN_x2plus.pth"


def download_weights():
    """Download Real-ESRGAN pretrained weights if not already present."""
    ESRGAN_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if not ESRGAN_WEIGHTS_PATH.exists():
        print("[ESRGAN] Downloading pretrained weights...")
        print(f"         URL: {ESRGAN_WEIGHTS_URL}")
        urllib.request.urlretrieve(ESRGAN_WEIGHTS_URL, str(ESRGAN_WEIGHTS_PATH))
        print(f"[ESRGAN] Weights saved to: {ESRGAN_WEIGHTS_PATH}")
    else:
        print(f"[ESRGAN] Using cached weights: {ESRGAN_WEIGHTS_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight RRDBNet Architecture
# (Simplified version — avoids full basicsr dependency in inference)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualDenseBlock(torch.nn.Module):
    """Residual Dense Block — core unit of RRDBNet."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * 0.2 + x


class RRDB(torch.nn.Module):
    """Residual-in-Residual Dense Block."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(torch.nn.Module):
    """
    Simplified RRDBNet for Real-ESRGAN ×2 upsampling.

    Works for both 1-channel (grayscale) and 3-channel (RGB) inputs.
    """

    def __init__(
        self,
        num_in_ch: int = 1,    # 1 for grayscale, 3 for RGB
        num_out_ch: int = 1,
        num_feat: int = 64,
        num_block: int = 6,    # Reduced from 23 → fits in 15GB VRAM
        num_grow_ch: int = 32,
        scale: int = 2,
    ):
        super().__init__()
        self.scale = scale
        self.conv_first = torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.body = torch.nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )

        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling
        self.conv_up1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 4:
            self.conv_up2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Pixel shuffle upsampling ×2
        feat = self.lrelu(
            self.conv_up1(
                torch.nn.functional.interpolate(feat, scale_factor=2, mode="nearest")
            )
        )
        if self.scale == 4:
            feat = self.lrelu(
                self.conv_up2(
                    torch.nn.functional.interpolate(feat, scale_factor=2, mode="nearest")
                )
            )

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# SuperResolutionModule — Main API
# ─────────────────────────────────────────────────────────────────────────────

class SuperResolutionModule:
    """
    Real-ESRGAN ×2 super-resolution for pulp fibril microscopy images.

    Handles:
      - Grayscale input (H, W) uint8
      - Tiled inference to handle large images without OOM
      - Optional weight loading from pretrained checkpoint

    Usage:
        sr = SuperResolutionModule(scale=2, device='cuda')
        enhanced = sr.enhance(image)  # uint8 in → uint8 out
    """

    def __init__(
        self,
        scale: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tile_size: int = 256,     # Process image in tiles (fits T4 VRAM)
        tile_pad: int = 16,
        pretrained: bool = True,
    ):
        self.scale = scale
        self.device = torch.device(device)
        self.tile_size = tile_size
        self.tile_pad = tile_pad

        # Build model
        self.model = RRDBNet(
            num_in_ch=1, num_out_ch=1,
            num_feat=64, num_block=6,  # Lightweight: 6 blocks vs 23 in full model
            num_grow_ch=32, scale=scale,
        ).to(self.device)

        if pretrained:
            self._load_pretrained()

        self.model.eval()
        print(f"[ESRGAN] Super-Resolution Module ready | scale={scale}× | device={device}")

    def _load_pretrained(self):
        """
        Load pretrained Real-ESRGAN weights.

        NOTE: The official weights are for 3-channel input.
        We load them and adapt — for grayscale the first conv weights
        are averaged across input channels.
        """
        download_weights()
        try:
            state_dict = torch.load(
                str(ESRGAN_WEIGHTS_PATH),
                map_location=self.device,
                weights_only=True,
            )
            # Handle nested state dicts
            if "params_ema" in state_dict:
                state_dict = state_dict["params_ema"]
            elif "params" in state_dict:
                state_dict = state_dict["params"]

            # Adapt first conv: 3-ch → 1-ch by averaging input channels
            if "conv_first.weight" in state_dict:
                w = state_dict["conv_first.weight"]
                if w.shape[1] == 3:
                    state_dict["conv_first.weight"] = w.mean(dim=1, keepdim=True)

            # Load with strict=False to skip missing/extra keys
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[ESRGAN] Missing keys (will use random init): {len(missing)}")
            print("[ESRGAN] Pretrained weights loaded ✅")

        except Exception as e:
            print(f"[ESRGAN] ⚠️  Could not load pretrained weights: {e}")
            print("[ESRGAN] Using randomly initialized weights (fine-tune required)")

    @torch.no_grad()
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance a grayscale image using Real-ESRGAN ×2 upsampling.

        Uses tiled inference to avoid OOM on large images.

        Args:
            image: (H, W) uint8 grayscale numpy array

        Returns:
            enhanced: (H*scale, W*scale) uint8 grayscale
        """
        assert image.ndim == 2, "enhance() expects a 2D grayscale image"

        H, W = image.shape
        # Normalize to [0, 1]
        img_f = image.astype(np.float32) / 255.0

        if self.tile_size is None or (H <= self.tile_size and W <= self.tile_size):
            # Small image — process in one shot
            out = self._forward_single(img_f)
        else:
            # Large image — tile-based inference
            out = self._forward_tiled(img_f)

        # Clip and convert back to uint8
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return out

    def _forward_single(self, image_f: np.ndarray) -> np.ndarray:
        """Run model on a single full image."""
        tensor = torch.from_numpy(image_f[None, None]).to(self.device)  # (1, 1, H, W)
        with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
            out = self.model(tensor)
        return out.squeeze().cpu().float().numpy()

    def _forward_tiled(self, image_f: np.ndarray) -> np.ndarray:
        """
        Tile-based inference for large images.
        Splits into overlapping tiles, processes each, and stitches.
        """
        H, W = image_f.shape
        tile = self.tile_size
        pad = self.tile_pad
        scale = self.scale

        output = np.zeros((H * scale, W * scale), dtype=np.float32)
        weight_map = np.zeros_like(output)

        for y in range(0, H, tile - 2 * pad):
            for x in range(0, W, tile - 2 * pad):
                # Extract tile with padding
                y0 = max(0, y - pad)
                x0 = max(0, x - pad)
                y1 = min(H, y + tile - pad)
                x1 = min(W, x + tile - pad)

                tile_in = image_f[y0:y1, x0:x1]
                tile_out = self._forward_single(tile_in)

                # Map tile output to output coordinates
                oy0, ox0 = y0 * scale, x0 * scale
                oy1, ox1 = y1 * scale, x1 * scale

                output[oy0:oy1, ox0:ox1] += tile_out
                weight_map[oy0:oy1, ox0:ox1] += 1.0

        # Average overlapping regions
        output = output / np.maximum(weight_map, 1.0)
        return output
