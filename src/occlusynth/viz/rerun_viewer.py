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

        viewer = RerunViewer("occlusynth")                       # spawn GUI
        viewer = RerunViewer("occlusynth", spawn=False,          # headless → .rrd
                             save_path="demo_outputs/scene.rrd")
        viewer.log_rgb(stem, rgb_path)
        viewer.log_depth(stem, depth_m, K)
        viewer.log_pointcloud(stem, pts_xyz, pts_rgb)
        viewer.log_camera(stem, c2w, K, (H, W))
        viewer.log_visibility(result, frames_data)               # green/red/amber

    Open a saved recording later with::  rerun demo_outputs/scene.rrd
    """

    def __init__(
        self,
        app_id: str = "occlusynth",
        spawn: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        self.app_id    = app_id
        self.spawn     = spawn
        self.save_path = save_path
        self._rr       = None
        self._enabled  = False
        self._try_init()

    def _try_init(self) -> None:
        try:
            import rerun as rr
            rr.init(self.app_id, spawn=self.spawn and self.save_path is None)
            if self.save_path is not None:
                from pathlib import Path
                Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
                rr.save(self.save_path)
                print(f"[RerunViewer] recording → {self.save_path}  "
                      f"(open with: rerun {self.save_path})")
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

    def log_visibility(
        self,
        result,
        frames_data: Optional[list] = None,
        voxel_size:  float = 0.05,
        free_subsample: int = 40_000,
    ) -> None:
        """
        Stream a VisibilityResult as three independently-toggleable voxel clouds:

            world/voxels/free      green  — observed empty space
            world/voxels/surface   red    — measured solid geometry
            world/voxels/occluded  amber  — hidden, completer inpaint targets

        Toggle FREE off in the Rerun UI to get the money shot: red surfaces with
        amber occlusion shadows behind them (amber = what the robot must imagine).
        Camera frusta are logged too.
        """
        if not self._enabled:
            return

        from occlusynth.fusion.tsdf import (
            FREE, SURFACE, OCCLUDED, CLASS_COLORS,
        )

        rng    = np.random.default_rng(0)
        radius = voxel_size * 0.5
        specs  = [
            ("free",     FREE,     free_subsample),
            ("surface",  SURFACE,  None),
            ("occluded", OCCLUDED, None),
        ]
        for name, label, cap in specs:
            m   = result.labels == label
            xyz = result.centers[m]
            if cap is not None and len(xyz) > cap:
                xyz = xyz[rng.choice(len(xyz), cap, replace=False)]
            if len(xyz) == 0:
                continue
            col = np.array(CLASS_COLORS[label], np.uint8)
            self._rr.log(
                f"world/voxels/{name}",
                self._rr.Points3D(xyz, colors=col, radii=radius),
            )

        # camera frusta for context
        if frames_data is not None:
            for i, fd in enumerate(frames_data):
                depth = np.asarray(fd["depth_m"])
                H, W  = depth.shape
                self.log_camera(f"f{i:02d}", np.asarray(fd["c2w"]),
                                np.asarray(fd["K"]), (H, W))
