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

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.utils.data
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


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class ScanNetDataset(torch.utils.data.Dataset):
    """
    Multi-view ScanNet dataset for VGGT adapter training.

    Each sample is a sequence of ``n_frames`` from one scene, returned as a
    dict of tensors:

    =========  ============================  ====================================
    Key        Shape                         Notes
    =========  ============================  ====================================
    rgb        (N, 3, H_c, W_c) float32     [0, 1], resized to target_size
    depth_gt   (N, H_c, W_c)   float32      metres; 0 = invalid; INTER_NEAREST
    intrinsics (N, 3, 3)        float32      colour K scaled to target_size
    pose       (N, 4, 4)        float32      camera-to-world, metres
    scene_id   str
    frame_idx  list[str]                     frame stems, length N
    =========  ============================  ====================================

    All three spatial tensors (rgb, depth_gt, intrinsics) share the same
    (H_c, W_c) coordinate frame.  Depth is resized with ``cv2.INTER_NEAREST``
    so the 0-sentinel for missing pixels is never averaged with valid depth.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        n_frames: int = 6,
        split: str = "train",       # "train" | "val" | "all"
        target_size: int = 512,     # longest side in pixels after resize
        val_fraction: float = 0.2,
    ) -> None:
        if split not in {"train", "val", "all"}:
            raise ValueError(f"split must be 'train', 'val', or 'all'; got {split!r}")

        if root is None:
            from occlusynth.utils.device import get_config, get_repo_root
            cfg = get_config()
            root = get_repo_root() / cfg["scannet_frames_25k"]

        self.root        = Path(root)
        self.n_frames    = n_frames
        self.target_size = target_size

        all_scenes = sorted(p.name for p in self.root.iterdir() if p.is_dir())

        if split == "all":
            self.scenes = all_scenes
        elif split == "val":
            self.scenes = [s for s in all_scenes if self._scene_is_val(s, val_fraction)]
        else:
            self.scenes = [s for s in all_scenes if not self._scene_is_val(s, val_fraction)]

    # ── split helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _scene_is_val(scene_id: str, val_fraction: float) -> bool:
        """Deterministic hash-based scene split (no official file required)."""
        digest = hashlib.md5(scene_id.encode()).digest()
        h = int.from_bytes(digest[:2], "little")   # 0..65535
        return h / 65536.0 < val_fraction

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> dict:
        scene_id  = self.scenes[idx]
        scene_dir = self.root / scene_id
        scene     = ScanNetScene(scene_dir)
        stems     = pick_frames(scene_dir, self.n_frames)

        # Compute resize scale from the actual colour image (all frames identical).
        sample_img        = Image.open(scene.color_path(stems[0]))
        orig_W, orig_H    = sample_img.size          # PIL gives (W, H)
        scale             = self.target_size / max(orig_H, orig_W)
        new_W, new_H      = round(orig_W * scale), round(orig_H * scale)

        # Scale colour intrinsics once for the whole scene.
        # fx, fy scale by `scale`; cx, cy scale by `scale`; bottom row unchanged.
        K_orig   = scene.K_color                     # (3, 3) float32
        K_scaled = K_orig.copy()
        K_scaled[0, 0] *= scale    # fx
        K_scaled[1, 1] *= scale    # fy
        K_scaled[0, 2] *= scale    # cx
        K_scaled[1, 2] *= scale    # cy
        K_tensor = torch.from_numpy(K_scaled)        # (3, 3) float32

        rgb_list, depth_list, pose_list = [], [], []

        for stem in stems:
            # ── RGB → (3, H_c, W_c) float32 [0, 1] ──────────────────────────
            img = (
                Image.open(scene.color_path(stem))
                .resize((new_W, new_H), Image.BILINEAR)
            )
            rgb = torch.from_numpy(
                np.array(img, dtype=np.float32) / 255.0
            ).permute(2, 0, 1)               # (H, W, 3) → (3, H, W)
            rgb_list.append(rgb)

            # ── GT depth → (H_c, W_c) float32 metres ────────────────────────
            # INTER_NEAREST: never mix valid depth with the 0-sentinel —
            # bilinear would create phantom geometry at occlusion boundaries.
            depth_raw = scene.depth(stem)           # (480, 640) float32, 0=invalid
            depth_resized = cv2.resize(
                depth_raw, (new_W, new_H), interpolation=cv2.INTER_NEAREST
            )
            depth_list.append(torch.from_numpy(depth_resized))

            # ── pose → (4, 4) float32 ────────────────────────────────────────
            pose_list.append(
                torch.from_numpy(scene.pose(stem).astype(np.float32))
            )

        # Broadcast the single K to every frame (contiguous copy via .expand + stack).
        K_seq = K_tensor.unsqueeze(0).expand(len(stems), -1, -1).contiguous()

        return {
            "rgb":        torch.stack(rgb_list),    # (N, 3, H_c, W_c)
            "depth_gt":   torch.stack(depth_list),  # (N, H_c, W_c)
            "intrinsics": K_seq,                    # (N, 3, 3)
            "pose":       torch.stack(pose_list),   # (N, 4, 4)
            "scene_id":   scene_id,
            "frame_idx":  stems,
        }

    # ── collate ───────────────────────────────────────────────────────────────

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Collate a list of sequence dicts produced by ``__getitem__``.

        Tensor fields are stacked along a new batch dimension B:

        =========  ============================
        Key        Shape
        =========  ============================
        rgb        (B, N, 3, H_c, W_c)
        depth_gt   (B, N, H_c, W_c)
        intrinsics (B, N, 3, 3)
        pose       (B, N, 4, 4)
        scene_id   list[str], length B
        frame_idx  list[list[str]], length B
        =========  ============================
        """
        return {
            "rgb":        torch.stack([item["rgb"]        for item in batch]),
            "depth_gt":   torch.stack([item["depth_gt"]   for item in batch]),
            "intrinsics": torch.stack([item["intrinsics"] for item in batch]),
            "pose":       torch.stack([item["pose"]       for item in batch]),
            "scene_id":   [item["scene_id"]  for item in batch],
            "frame_idx":  [item["frame_idx"] for item in batch],
        }
