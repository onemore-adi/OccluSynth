"""
Depth visualisation and Rerun viewer integration.

colorize()     — universal depth → RGB colourmap utility used everywhere
RerunViewer    — streams RGB frames + depth maps to the Rerun SDK viewer
                 (requires ``pip install rerun-sdk``)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ── colourisation ─────────────────────────────────────────────────────────────

def colorize(
    depth:  np.ndarray,
    vmin:   Optional[float] = None,
    vmax:   Optional[float] = None,
    cmap:   str             = "inferno",
    invalid_rgb: tuple[int, int, int] = (30, 30, 30),
) -> np.ndarray:
    """
    Map a float depth map to an (H, W, 3) uint8 RGB image.

    Invalid pixels (depth == 0) are rendered in ``invalid_rgb``
    (dark grey by default) so they are visually distinct from near objects.

    Args:
        depth:       (H, W) float array; 0 = invalid
        vmin:        clip minimum (defaults to 2nd percentile of valid pixels)
        vmax:        clip maximum (defaults to 98th percentile of valid pixels)
        cmap:        matplotlib colourmap name  (default 'inferno')
        invalid_rgb: colour for invalid / zero-depth pixels

    Returns:
        (H, W, 3) uint8
    """
    import matplotlib.cm as cm

    valid = depth > 0
    if not valid.any():
        return np.full((*depth.shape, 3), invalid_rgb, dtype=np.uint8)

    if vmin is None:
        vmin = float(np.percentile(depth[valid], 2))
    if vmax is None:
        vmax = float(np.percentile(depth[valid], 98))

    norm  = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgba  = cm.get_cmap(cmap)(norm)
    rgb   = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb[~valid] = invalid_rgb
    return rgb


# ── Rerun viewer ──────────────────────────────────────────────────────────────

class RerunViewer:
    """
    Streams reconstruction data to the Rerun SDK viewer.

    Requires ``pip install rerun-sdk``.  Gracefully degrades if not installed.

    Usage::

        viewer = RerunViewer("occlusynth")
        viewer.log_rgb(stem, rgb_path)
        viewer.log_depth(stem, depth_m, K)
        viewer.log_pointcloud(stem, pts_xyz, pts_rgb)
        viewer.log_camera(stem, c2w, K, (H, W))
    """

    def __init__(self, app_id: str = "occlusynth") -> None:
        self.app_id   = app_id
        self._rr      = None
        self._enabled = False
        self._try_init()

    def _try_init(self) -> None:
        try:
            import rerun as rr
            rr.init(self.app_id, spawn=True)
            self._rr      = rr
            self._enabled = True
        except ImportError:
            print("[RerunViewer] rerun-sdk not installed — viewer disabled.")
            print("              Install with: pip install rerun-sdk")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── logging methods ───────────────────────────────────────────────────────

    def log_rgb(self, stem: str, rgb: np.ndarray) -> None:
        """Log an (H, W, 3) uint8 RGB frame."""
        if not self._enabled:
            return
        self._rr.log(f"world/camera/{stem}/rgb",
                     self._rr.Image(rgb))

    def log_depth(
        self,
        stem:     str,
        depth_m:  np.ndarray,
        K:        np.ndarray,
    ) -> None:
        """Log a float32 depth map in metres + its intrinsics."""
        if not self._enabled:
            return
        self._rr.log(
            f"world/camera/{stem}/depth",
            self._rr.DepthImage(depth_m, meter=1.0),
        )

    def log_pointcloud(
        self,
        stem:     str,
        pts_xyz:  np.ndarray,
        pts_rgb:  Optional[np.ndarray] = None,
    ) -> None:
        """Log an (N, 3) point cloud with optional (N, 3) uint8 colours."""
        if not self._enabled:
            return
        self._rr.log(
            f"world/pointcloud/{stem}",
            self._rr.Points3D(pts_xyz, colors=pts_rgb),
        )

    def log_camera(
        self,
        stem:      str,
        c2w:       np.ndarray,
        K:         np.ndarray,
        image_hw:  tuple[int, int],
    ) -> None:
        """Log camera pose + pinhole intrinsics."""
        if not self._enabled:
            return
        H, W = image_hw
        self._rr.log(
            f"world/camera/{stem}",
            self._rr.Transform3D(
                translation=c2w[:3, 3],
                mat3x3=c2w[:3, :3],
            ),
        )
        self._rr.log(
            f"world/camera/{stem}",
            self._rr.Pinhole(
                focal_length=(K[0, 0], K[1, 1]),
                principal_point=(K[0, 2], K[1, 2]),
                width=W, height=H,
            ),
        )
