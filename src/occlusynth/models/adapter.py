"""
OccluSynth Adapter — learns to predict per-frame depth scale (a) and shift (b)
from VGGT-Omega's camera register tokens, without seeing GT depth at inference.

Architecture (to be implemented in the next sprint)
----------------------------------------------------
Input:  camera_token  (B, 1, 2048)  — the [CLS]-like token from VGGT-Omega
                                       for each frame; encodes scene geometry
Output: (a, b)        (B, 2)        — scale and shift in metric space

Training signal: the four closed-form fits in depth_calibration.py provide
per-frame (a, b) targets.  The adapter is trained to regress these from the
register token so that at inference time no GT depth anchors are needed.

This module is a placeholder.  The interface is fixed; the implementation
will be added once the data pipeline (scannet.py + sparse_sampler.py) is
producing clean (token, a, b) training pairs.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class DepthAdapter(nn.Module):
    """
    MLP adapter: camera_token (2048-d) → (scale a, shift b).

    Args:
        token_dim:  dimensionality of the VGGT camera register token (default 2048)
        hidden_dim: MLP hidden layer width (default 256)
        init_scale: initialise the scale head output bias to this value
                    (set to the global median scale ~7.4 to warm-start training)

    Example::

        adapter = DepthAdapter(init_scale=7.4)
        token   = vggt_result["camera_and_register_tokens"][:, :, 0]  # (B, N, 2048)
        a, b    = adapter(token[:, 0])   # per-scene scalar from first-frame token
    """

    def __init__(
        self,
        token_dim:  int   = 2048,
        hidden_dim: int   = 256,
        init_scale: float = 7.4,
    ) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Separate heads — scale must be > 0, shift can be negative
        self.scale_head = nn.Linear(hidden_dim // 2, 1)
        self.shift_head = nn.Linear(hidden_dim // 2, 1)

        # Warm-start: bias the scale output toward the known global median
        with torch.no_grad():
            self.scale_head.bias.fill_(init_scale)
            self.shift_head.bias.zero_()

    def forward(self, token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token: (B, token_dim) — one camera register token per frame

        Returns:
            a: (B,) — predicted scale factor  (enforced > 0 via softplus)
            b: (B,) — predicted shift in metres
        """
        h = self.trunk(token)
        a = nn.functional.softplus(self.scale_head(h).squeeze(-1))  # > 0
        b = self.shift_head(h).squeeze(-1)
        return a, b

    def predict_numpy(self, token: "np.ndarray") -> Tuple[float, float]:
        """
        Convenience wrapper for single-frame numpy inference.

        Args:
            token: (2048,) numpy float32

        Returns:
            (a, b) as Python floats
        """
        import numpy as np
        t = torch.from_numpy(token).float().unsqueeze(0)
        with torch.no_grad():
            a, b = self(t)
        return float(a[0]), float(b[0])
