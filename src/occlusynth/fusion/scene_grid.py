"""
scene_grid — dense (sdf, weight, p_observed, state) volumes for the completer.

``fuse_visibility()`` is demo-oriented: it returns flat per-voxel arrays and a
PLY, but not the grid origin or the raw channel volumes.  The completer needs
exactly those, shaped (nx, ny, nz), plus the world origin so the GT SDF from
``mesh_to_tsdf()`` can be sampled on an identical grid.  This module wraps the
same VisibilityVoxelGrid integration loop without modifying it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .tsdf import TSDFConfig, VisibilityVoxelGrid


@dataclass
class SceneGrid:
    """Dense voxel volumes of one scene, all C-order (nx, ny, nz)."""
    origin:     np.ndarray   # (3,) float64 — world-space grid corner
    voxel_size: float
    sdf:        np.ndarray   # float32 — normalized TSDF in [-1, 1] (surface band), +1 elsewhere
    weight:     np.ndarray   # float32 — TSDF fusion weight
    p_observed: np.ndarray   # float32 — soft visibility in [0, 1]
    state:      np.ndarray   # uint8   — 0=unobservable 1=free 2=surface 3=occluded

    @property
    def dims(self) -> tuple:
        return self.sdf.shape

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, origin=self.origin, voxel_size=self.voxel_size,
            sdf=self.sdf, weight=self.weight,
            p_observed=self.p_observed, state=self.state,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SceneGrid":
        z = np.load(path)
        return cls(origin=z["origin"], voxel_size=float(z["voxel_size"]),
                   sdf=z["sdf"], weight=z["weight"],
                   p_observed=z["p_observed"], state=z["state"])


def fuse_visibility_grid(
    frames_data: List[Dict],
    config: Optional[TSDFConfig] = None,
) -> SceneGrid:
    """
    Run visibility-aware fusion and return the dense channel volumes.

    Same integration as ``fuse_visibility()`` (RGB accumulation skipped — the
    completer is geometry-only), but exposes origin + (nx, ny, nz) volumes.
    """
    cfg = config or TSDFConfig()
    grid = VisibilityVoxelGrid.from_frames(frames_data, cfg)

    for fd in frames_data:
        grid.integrate(
            np.asarray(fd["depth_m"], np.float32),
            np.asarray(fd["K"], np.float64),
            np.asarray(fd["c2w"], np.float64),
            rgb=None, depth_max=cfg.depth_max, depth_min=cfg.depth_min,
        )

    result = grid.classify()
    dims = grid.dims
    return SceneGrid(
        origin=grid.origin.copy(),
        voxel_size=grid.voxel_size,
        sdf=grid.tsdf.reshape(dims).astype(np.float32),
        weight=grid.weight.reshape(dims).astype(np.float32),
        p_observed=result.p_observed.reshape(dims).astype(np.float32),
        state=result.labels.reshape(dims).astype(np.uint8),
    )
