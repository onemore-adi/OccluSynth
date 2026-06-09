"""
ScanNet v2 dataset I/O — frames_25k subset.

Per-scene layout expected on disk:
    <scene_dir>/
        color/<stem>.jpg          RGB frames
        depth/<stem>.png          uint16 millimetres
        pose/<stem>.txt           4×4 camera-to-world (metres)
        intrinsics_color.txt      4×4 colour camera K
        intrinsics_depth.txt      4×4 depth camera K
        label/<stem>.png          NYU40 semantic labels  (optional)
        instance/<stem>.png       instance masks          (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

# ── physical constants ────────────────────────────────────────────────────────

DEPTH_MM_SCALE: float = 1000.0   # ScanNet PNG → metres
DEPTH_MAX_M:    float = 3.5      # Kinect v1 reliable range


# ── low-level loaders ────────────────────────────────────────────────────────

def load_gt_depth(scene_dir: str | Path, stem: str) -> np.ndarray:
    """
    Load ScanNet GT depth map → float32 metres.

    Args:
        scene_dir: path to a single scene directory
        stem:      frame stem, e.g. '000200'  (no extension)

    Returns:
        (H, W) float32 array; 0.0 = invalid / out-of-range pixel
    """
    path = Path(scene_dir) / "depth" / (stem + ".png")
    raw  = np.array(Image.open(path), dtype=np.float32)
    d    = raw / DEPTH_MM_SCALE
    d[d > DEPTH_MAX_M] = 0.0
    return d


def load_gt_pose(scene_dir: str | Path, stem: str) -> np.ndarray:
    """
    Load ScanNet camera-to-world pose matrix.

    Returns:
        (4, 4) float64 camera-to-world transform  [R | t; 0 | 1]
        Translation is in metres.
    """
    path = Path(scene_dir) / "pose" / (stem + ".txt")
    rows: list[list[float]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    return np.array(rows, dtype=np.float64)


def load_gt_intrinsics(
    scene_dir: str | Path,
    sensor: str = "depth",
) -> np.ndarray:
    """
    Load camera intrinsics.

    Args:
        scene_dir: path to scene directory
        sensor:    'color' or 'depth'

    Returns:
        (3, 3) float32 K matrix
    """
    path = Path(scene_dir) / f"intrinsics_{sensor}.txt"
    rows: list[list[float]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    return np.array(rows, dtype=np.float32)[:3, :3]


def pick_frames(scene_dir: str | Path, n_frames: int) -> List[str]:
    """
    Pick n_frames evenly-spaced stems that have colour + depth + pose on disk.

    Trims 2 frames from each end to avoid boundary/startup frames that
    ScanNet sometimes has with incomplete sensor data.

    Returns:
        Sorted list of stem strings (e.g. ['000200', '001200', ...])
    """
    scene_dir = Path(scene_dir)
    color_stems = {p.stem for p in (scene_dir / "color").glob("*.jpg")}
    depth_stems = {p.stem for p in (scene_dir / "depth").glob("*.png")}
    pose_stems  = {p.stem for p in (scene_dir / "pose").glob("*.txt")}

    valid = sorted(color_stems & depth_stems & pose_stems)
    pool  = valid[2:-2] if len(valid) > 10 else valid
    n     = min(n_frames, len(pool))
    idx   = np.linspace(0, len(pool) - 1, n, dtype=int)
    return [pool[i] for i in idx]


# ── ScanNetScene dataclass ────────────────────────────────────────────────────

@dataclass
class ScanNetScene:
    """
    Lightweight handle for one ScanNet scene directory.

    Usage::

        scene = ScanNetScene("data/scannet/tasks/scannet_frames_25k/scene0000_00")
        stems = scene.pick_frames(6)
        depth = scene.depth(stems[0])    # (480, 640) float32 metres
        pose  = scene.pose(stems[0])     # (4, 4) float64 c2w
        K     = scene.K_depth            # (3, 3) float32
    """

    scene_dir: str | Path

    def __post_init__(self) -> None:
        self.scene_dir = Path(self.scene_dir)
        if not self.scene_dir.exists():
            raise FileNotFoundError(f"Scene not found: {self.scene_dir}")

    @property
    def scene_id(self) -> str:
        return self.scene_dir.name

    @property
    def K_depth(self) -> np.ndarray:
        return load_gt_intrinsics(self.scene_dir, sensor="depth")

    @property
    def K_color(self) -> np.ndarray:
        return load_gt_intrinsics(self.scene_dir, sensor="color")

    def pick_frames(self, n_frames: int) -> List[str]:
        return pick_frames(self.scene_dir, n_frames)

    def depth(self, stem: str) -> np.ndarray:
        return load_gt_depth(self.scene_dir, stem)

    def pose(self, stem: str) -> np.ndarray:
        return load_gt_pose(self.scene_dir, stem)

    def color_path(self, stem: str) -> Path:
        return self.scene_dir / "color" / (stem + ".jpg")

    def all_stems(self) -> List[str]:
        return sorted(p.stem for p in (self.scene_dir / "color").glob("*.jpg"))

    def __repr__(self) -> str:
        n = len(self.all_stems())
        return f"ScanNetScene('{self.scene_id}', {n} frames)"
