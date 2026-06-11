#!/usr/bin/env python
"""
generate_completer_data.py — build 96³ training crops for the 3D voxel completer.

Per scene:
  1. fuse the partial grid with fuse_visibility_grid() (6 frames, GT depth,
     GT poses) — cached to data/completer_grids/<scene>_n<N>.npz
  2. sample the GT SDF with mesh_to_tsdf() on the *same* origin/dims
  3. sample random 96³ crops with occluded_fraction ≥ 0.10
  4. save .npz crops:
       input  (3, 96, 96, 96) float32 — (sdf, weight, p_observed)
       target (96, 96, 96)    float32 — GT signed distance, metres
       state  (96, 96, 96)    uint8   — 0=unobservable 1=free 2=surface 3=occluded

Scene grids smaller than 96 in any axis (typical in z) are centre-padded with
UNOBSERVABLE voxels (sdf=+1, weight=0, p_observed=0); the GT SDF is computed
on the padded grid directly, so the target stays exact everywhere.  Padded
voxels carry state=0 and are excluded from the loss.

Split: deterministic md5 hash (ScanNetDataset._scene_is_val, val_fraction=0.2)
over the 50 scenes with downloaded _vh_clean_2.ply → 40 train / 10 val.

Rejection rules:
  * occluded_fraction < 0.10           — teaches nothing about completion
  * >50% exactly-zero GT SDF values    — degenerate mesh region

Usage:
    .venv312/bin/python scripts/generate_completer_data.py
    .venv312/bin/python scripts/generate_completer_data.py --scenes scene0000_00
"""

import argparse
import time
from pathlib import Path

import numpy as np

from occlusynth.data import ScanNetDataset
from occlusynth.fusion import (OCCLUDED, TSDFConfig, SceneGrid,
                               fuse_visibility_grid, mesh_to_tsdf)
from occlusynth.utils import get_repo_root

CROP = 96
OCC_MIN_FRAC = 0.10
ZERO_MAX_FRAC = 0.50
VAL_SEED = 0


def completer_split(root: Path):
    """(train_scenes, val_scenes) over the scenes that have a GT mesh."""
    scans = root / "data/scannet/scans"
    dataset = ScanNetDataset(split="all")
    have_frames = set(dataset.scenes)
    scenes = sorted(p.name for p in scans.iterdir()
                    if (p / f"{p.name}_vh_clean_2.ply").exists()
                    and p.name in have_frames)
    val = [s for s in scenes if ScanNetDataset._scene_is_val(s, 0.2)]
    train = [s for s in scenes if s not in val]
    return train, val


def build_scene_arrays(scene: str, n_frames: int, cache_dir: Path,
                       root: Path) -> tuple[SceneGrid, np.ndarray]:
    """Partial grid + GT SDF, both centre-padded to ≥ CROP per axis. Cached."""
    cache = cache_dir / f"{scene}_n{n_frames}.npz"
    if cache.exists():
        z = np.load(cache)
        grid = SceneGrid(origin=z["origin"], voxel_size=float(z["voxel_size"]),
                         sdf=z["sdf"], weight=z["weight"],
                         p_observed=z["p_observed"], state=z["state"])
        return grid, z["gt_sdf"]

    dataset = ScanNetDataset(n_frames=n_frames, split="all")
    item = dataset[dataset.scenes.index(scene)]
    depth = item["depth_gt"].numpy()
    poses = item["pose"].numpy()
    K = item["intrinsics"][0].numpy()
    frames = [{"depth_m": depth[i], "K": K, "c2w": poses[i]}
              for i in range(len(item["frame_idx"]))]

    cfg = TSDFConfig()
    grid = fuse_visibility_grid(frames, cfg)

    # centre-pad every axis to at least CROP with UNOBSERVABLE defaults
    dims = np.array(grid.dims)
    padded = np.maximum(dims, CROP)
    lo = (padded - dims) // 2
    pad = [(int(l), int(p - d - l)) for l, p, d in zip(lo, padded, dims)]
    origin = grid.origin - lo * grid.voxel_size

    grid = SceneGrid(
        origin=origin,
        voxel_size=grid.voxel_size,
        sdf=np.pad(grid.sdf, pad, constant_values=1.0),
        weight=np.pad(grid.weight, pad, constant_values=0.0),
        p_observed=np.pad(grid.p_observed, pad, constant_values=0.0),
        state=np.pad(grid.state, pad, constant_values=0),
    )

    mesh = root / f"data/scannet/scans/{scene}/{scene}_vh_clean_2.ply"
    gt_sdf = mesh_to_tsdf(str(mesh), grid.voxel_size, grid.origin, grid.dims)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, origin=grid.origin, voxel_size=grid.voxel_size,
                        sdf=grid.sdf, weight=grid.weight,
                        p_observed=grid.p_observed, state=grid.state,
                        gt_sdf=gt_sdf)
    return grid, gt_sdf


def sample_crops(grid: SceneGrid, gt_sdf: np.ndarray, n_crops: int,
                 rng: np.random.Generator, max_attempts: int = 400):
    """Yield (corner_ijk, stats) for accepted crops."""
    dims = np.array(grid.dims)
    hi = dims - CROP  # inclusive corner range
    out = []
    attempts = 0
    while len(out) < n_crops and attempts < max_attempts:
        attempts += 1
        c = np.array([rng.integers(0, h + 1) for h in hi])
        sl = tuple(slice(int(c[a]), int(c[a]) + CROP) for a in range(3))
        state = grid.state[sl]
        occ_frac = float((state == OCCLUDED).mean())
        if occ_frac < OCC_MIN_FRAC:
            continue
        zero_frac = float((gt_sdf[sl] == 0.0).mean())
        if zero_frac > ZERO_MAX_FRAC:
            continue
        out.append((c, {"occ_frac": occ_frac, "zero_frac": zero_frac}))
    return out, attempts


def write_crop(path: Path, grid: SceneGrid, gt_sdf: np.ndarray,
               corner: np.ndarray, scene: str) -> None:
    sl = tuple(slice(int(corner[a]), int(corner[a]) + CROP) for a in range(3))
    # fp16 storage halves disk; loaders cast back to float32. SDF values are
    # O(metres) so fp16's ~1e-3 relative error is far below voxel size.
    inp = np.stack([grid.sdf[sl], grid.weight[sl], grid.p_observed[sl]]
                   ).astype(np.float16)
    np.savez_compressed(
        path,
        input=inp,                                  # (3, 96, 96, 96) float16
        target=gt_sdf[sl].astype(np.float16),       # (96, 96, 96) float16
        state=grid.state[sl],                       # (96, 96, 96) uint8
        # provenance — lets eval place the crop back into the world frame
        scene=scene,
        corner_ijk=corner.astype(np.int32),
        world_origin=(grid.origin + corner * grid.voxel_size),
        voxel_size=grid.voxel_size,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_frames", type=int, default=6)
    p.add_argument("--crops_per_scene", type=int, default=10)
    p.add_argument("--val_crops_per_scene", type=int, default=10)
    p.add_argument("--scenes", nargs="*", default=None,
                   help="restrict to these scenes (debugging)")
    args = p.parse_args()

    root = get_repo_root()
    cache_dir = root / "data/completer_grids"
    out_root = root / "data/completer_crops"

    train, val = completer_split(root)
    print(f"split: {len(train)} train / {len(val)} val scenes")

    jobs = [("train", s) for s in train] + [("val", s) for s in val]
    if args.scenes:
        jobs = [(sp, s) for sp, s in jobs if s in set(args.scenes)]

    totals = {"train": 0, "val": 0}
    for split, scene in jobs:
        t0 = time.time()
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        grid, gt_sdf = build_scene_arrays(scene, args.n_frames, cache_dir, root)

        n = args.crops_per_scene if split == "train" else args.val_crops_per_scene
        # val crops: fixed seed → identical crops every run; train seeds are
        # per-scene md5 (process-stable, unlike built-in str hash)
        import hashlib
        seed = (VAL_SEED if split == "val" else
                int.from_bytes(hashlib.md5(scene.encode()).digest()[:4], "little"))
        rng = np.random.default_rng(seed)
        crops, attempts = sample_crops(grid, gt_sdf, n, rng)

        for i, (corner, stats) in enumerate(crops):
            write_crop(out_dir / f"{scene}_crop{i:02d}.npz",
                       grid, gt_sdf, corner, scene)
        totals[split] += len(crops)
        occ = np.mean([s["occ_frac"] for _, s in crops]) if crops else 0.0
        print(f"  [{split}] {scene}: {len(crops)}/{n} crops "
              f"({attempts} attempts, mean occ {occ*100:.0f}%, "
              f"grid {grid.dims}, {time.time()-t0:.0f}s)")

    print(f"\ndone: {totals['train']} train crops, {totals['val']} val crops")
    for split in ("train", "val"):
        files = list((out_root / split).glob("*.npz"))
        size = sum(f.stat().st_size for f in files) / 1e6
        print(f"  {split}: {len(files)} files, {size:.0f} MB")


if __name__ == "__main__":
    main()
