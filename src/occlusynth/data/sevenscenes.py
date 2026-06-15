"""
7-Scenes dataset I/O.

Per-sequence layout expected on disk:
    <data_root>/7scenes/<scene>/<seq>/
        frame-XXXXXX.color.png     RGB  640×480
        frame-XXXXXX.depth.png     uint16 millimetres  640×480
        frame-XXXXXX.pose.txt      4×4 camera-to-world (metres)

All 7-Scenes sequences share one Kinect v1 camera model (no per-scene
intrinsics file); K_7SCENES is the dataset's published fixed calibration.

Depth convention matches ScanNet: 0 = invalid, clip at 3.5 m.
Pose convention matches ScanNet: 4×4 camera-to-world, metres.

Reference:
    Shotton et al., "Scene Coordinate Regression Forests for Camera
    Relocalization in RGB-D Images", CVPR 2013.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

# ── fixed camera model ────────────────────────────────────────────────────────
# Kinect v1, 640×480, same sensor used in ScanNet v2.
# Source: Microsoft Research 7-Scenes dataset page.
K_7SCENES: np.ndarray = np.array(
    [[585.0,   0.0, 320.0],
     [  0.0, 585.0, 240.0],
     [  0.0,   0.0,   1.0]],
    dtype=np.float32,
)

DEPTH_MM_SCALE: float = 1000.0   # uint16 mm → metres
DEPTH_MAX_M:    float = 3.5       # Kinect v1 reliable range
IMG_H:          int   = 480
IMG_W:          int   = 640


# ── low-level loaders ────────────────────────────────────────────────────────

def _stem_path(seq_dir: Path, stem: str, suffix: str) -> Path:
    return seq_dir / f"frame-{stem}{suffix}"


def load_depth_raw(seq_dir: Path, stem: str) -> np.ndarray:
    """Raw uint16 depth as float32 (mm units; 0 = invalid, 65535 = invalid)."""
    return np.array(
        Image.open(_stem_path(seq_dir, stem, ".depth.png")),
        dtype=np.float32,
    )


def load_depth(seq_dir: Path, stem: str) -> np.ndarray:
    """Depth map in metres; 0.0 = invalid pixel."""
    raw = load_depth_raw(seq_dir, stem)
    d   = raw / DEPTH_MM_SCALE
    d[d > DEPTH_MAX_M] = 0.0
    return d


def load_pose(seq_dir: Path, stem: str) -> np.ndarray:
    """4×4 camera-to-world transform (metres), identical format to ScanNet."""
    rows: list[list[float]] = []
    with open(_stem_path(seq_dir, stem, ".pose.txt")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    return np.array(rows, dtype=np.float64)


def pick_frames_7scenes(
    seq_dir: Path,
    n_frames: int,
    skip_invalid_poses: bool = True,
) -> List[str]:
    """
    Pick n_frames evenly-spaced stems that have all three files on disk.

    Trims 2 frames from each end (same as ScanNet) to avoid boundary frames.
    Optionally filters out frames whose pose.txt contains inf/nan (tracking
    loss), which 7-Scenes marks with large or infinite values.
    """
    seq_dir = Path(seq_dir)
    color_stems = {p.name.split(".")[0].replace("frame-", "")
                   for p in seq_dir.glob("frame-*.color.png")}
    depth_stems = {p.name.split(".")[0].replace("frame-", "")
                   for p in seq_dir.glob("frame-*.depth.png")}
    pose_stems  = {p.name.split(".")[0].replace("frame-", "")
                   for p in seq_dir.glob("frame-*.pose.txt")}

    valid = sorted(color_stems & depth_stems & pose_stems)

    if skip_invalid_poses:
        valid = [s for s in valid
                 if np.isfinite(load_pose(seq_dir, s)).all()]

    pool = valid[2:-2] if len(valid) > 10 else valid
    n    = min(n_frames, len(pool))
    if n == 0:
        raise ValueError(f"No valid frames in {seq_dir}")
    idx = np.linspace(0, len(pool) - 1, n, dtype=int)
    return [pool[i] for i in idx]


# ── SevenScenesSequence dataclass ─────────────────────────────────────────────

@dataclass
class SevenScenesSequence:
    """
    Lightweight handle for one 7-Scenes sequence directory.

    Usage::

        seq = SevenScenesSequence("data/7scenes/chess/seq-01")
        stems = seq.pick_frames(6)
        depth = seq.depth(stems[0])      # (480, 640) float32 metres
        pose  = seq.pose(stems[0])       # (4, 4) float64 c2w
        K     = seq.K                    # (3, 3) float32 — fixed for all 7-Scenes
    """

    seq_dir: str | Path

    def __post_init__(self) -> None:
        self.seq_dir = Path(self.seq_dir)
        if not self.seq_dir.exists():
            raise FileNotFoundError(f"Sequence directory not found: {self.seq_dir}")

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def scene_name(self) -> str:
        return self.seq_dir.parent.name

    @property
    def seq_name(self) -> str:
        return self.seq_dir.name

    @property
    def sequence_id(self) -> str:
        return f"{self.scene_name}/{self.seq_name}"

    # ── camera ────────────────────────────────────────────────────────────────

    @property
    def K(self) -> np.ndarray:
        """Fixed (3, 3) Kinect v1 intrinsics for all 7-Scenes sequences."""
        return K_7SCENES.copy()

    # ── frame access ──────────────────────────────────────────────────────────

    def all_stems(self) -> List[str]:
        """All 6-digit stems that have color + depth + pose files on disk."""
        color = {p.name.split(".")[0].replace("frame-", "")
                 for p in self.seq_dir.glob("frame-*.color.png")}
        depth = {p.name.split(".")[0].replace("frame-", "")
                 for p in self.seq_dir.glob("frame-*.depth.png")}
        pose  = {p.name.split(".")[0].replace("frame-", "")
                 for p in self.seq_dir.glob("frame-*.pose.txt")}
        return sorted(color & depth & pose)

    def pick_frames(self, n_frames: int) -> List[str]:
        return pick_frames_7scenes(self.seq_dir, n_frames)

    def depth_raw(self, stem: str) -> np.ndarray:
        """(480, 640) float32 in mm units — suitable as uncalibrated RANSAC input."""
        return load_depth_raw(self.seq_dir, stem)

    def depth(self, stem: str) -> np.ndarray:
        """(480, 640) float32 metres; 0.0 = invalid."""
        return load_depth(self.seq_dir, stem)

    def pose(self, stem: str) -> np.ndarray:
        """(4, 4) float64 camera-to-world transform."""
        return load_pose(self.seq_dir, stem)

    def color_path(self, stem: str) -> Path:
        return _stem_path(self.seq_dir, stem, ".color.png")

    def is_valid_pose(self, stem: str) -> bool:
        return bool(np.isfinite(self.pose(stem)).all())

    def __repr__(self) -> str:
        n = len(self.all_stems())
        return f"SevenScenesSequence('{self.sequence_id}', {n} frames)"
