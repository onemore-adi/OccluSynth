"""
OccluSynthCompleter — 3D U-Net that completes the SDF of a partial voxel grid.

The contribution: given a tri-state partial observation (free / surface /
occluded, plus unobservable), predict the complete signed distance field —
including the OCCLUDED region behind observed surfaces, which is absent from
every depth image and therefore unreachable by any 2D inpainting network.

Supervision is masked: L1 on SURFACE ∪ OCCLUDED voxels only.  UNOBSERVABLE
voxels (out of every frustum) are excluded entirely — the network is never
told, and never graded on, what it cannot in principle know.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# voxel states — match occlusynth.fusion.tsdf
UNOBSERVABLE = 0
FREE         = 1
SURFACE      = 2
OCCLUDED     = 3


def _conv_block(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    """3×3×3 conv → GroupNorm(8) → GELU."""
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
    )


class _EncoderBlock(nn.Module):
    """stride-2 downsample conv + refinement conv."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.down   = _conv_block(c_in, c_out, stride=2)
        self.refine = _conv_block(c_out, c_out)

    def forward(self, x):
        return self.refine(self.down(x))


class _DecoderBlock(nn.Module):
    """trilinear ×2 upsample → concat skip → two convs."""

    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.conv1 = _conv_block(c_in + c_skip, c_out)
        self.conv2 = _conv_block(c_out, c_out)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear",
                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class OccluSynthCompleter(nn.Module):
    """
    Encoder-decoder 3D U-Net.

    Input:  (B, 3, D, H, W) — channels: sdf, weight, p_observed
    Output: (B, 1, D, H, W) — completed SDF (metres, unbounded)

    Encoder: 4 blocks, channels [32, 64, 128, 256], stride-2 conv per block
    Decoder: 4 blocks with skip connections, trilinear upsample + conv
    All convolutions 3×3×3, GroupNorm(8), GELU; final 1×1×1 conv, no activation.

    ~11M parameters — fits MPS memory at batch=4, crop=96³.
    """

    def __init__(self, in_channels: int = 3, base: int = 32):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8]      # [32, 64, 128, 256]

        self.stem = nn.Sequential(_conv_block(in_channels, chs[0]),
                                  _conv_block(chs[0], chs[0]))
        self.enc1 = _EncoderBlock(chs[0], chs[1])
        self.enc2 = _EncoderBlock(chs[1], chs[2])
        self.enc3 = _EncoderBlock(chs[2], chs[3])
        self.enc4 = _EncoderBlock(chs[3], chs[3])

        self.dec4 = _DecoderBlock(chs[3], chs[3], chs[3])
        self.dec3 = _DecoderBlock(chs[3], chs[2], chs[2])
        self.dec2 = _DecoderBlock(chs[2], chs[1], chs[1])
        self.dec1 = _DecoderBlock(chs[1], chs[0], chs[0])

        self.head = nn.Conv3d(chs[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)        # (B,  32, D,    H,    W)
        s1 = self.enc1(s0)       # (B,  64, D/2,  ...)
        s2 = self.enc2(s1)       # (B, 128, D/4,  ...)
        s3 = self.enc3(s2)       # (B, 256, D/8,  ...)
        b  = self.enc4(s3)       # (B, 256, D/16, ...)

        d = self.dec4(b, s3)
        d = self.dec3(d, s2)
        d = self.dec2(d, s1)
        d = self.dec1(d, s0)
        return self.head(d)      # (B, 1, D, H, W)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                   state: torch.Tensor) -> torch.Tensor:
    """
    L1 on SURFACE ∪ OCCLUDED voxels only.

    Args:
        pred:   (B, 1, D, H, W) predicted SDF
        target: (B, D, H, W)    GT SDF
        state:  (B, D, H, W)    0=unobservable 1=free 2=surface 3=occluded
    """
    mask = (state == SURFACE) | (state == OCCLUDED)
    if mask.sum() == 0:
        return pred.sum() * 0.0          # degenerate crop — keep graph, no NaN
    return (pred[:, 0] - target).abs()[mask].mean()
