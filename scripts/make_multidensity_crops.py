#!/usr/bin/env python
"""
make_multidensity_crops.py — visibility-level data augmentation for the completer.

The completer was trained on crops from grids fused at ONE view density (n=6).
With only ~418 train crops it overfits, and it never sees the same geometry at
different levels of observation. This builds an enriched TRAIN set: every train
scene fused at n=6, n=10 and n=20 views. Same ground-truth target, different
partial input — so the model learns to complete regardless of how much was
observed. That is exactly the dynamic / varying-coverage robustness the pitch
claims, turned into training signal.

Honest-metrics guarantee: the VAL set is left byte-identical (copied straight
from data/completer_crops/val, the original n6 crops), so trunc_l1 / sign_acc
stay directly comparable to the v1 and v2 runs.

Output: data/completer_crops_md/{train,val}
    train:  {scene}_crop##.npz            (existing n6, copied)
            {scene}_n10_crop##.npz        (new)
            {scene}_n20_crop##.npz        (new)
    val:    {scene}_crop##.npz            (n6, copied — unchanged)

    .venv312/bin/python scripts/make_multidensity_crops.py
    .venv312/bin/python scripts/make_multidensity_crops.py --densities 10 20 --crops 15
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_completer_data import (build_scene_arrays, completer_split,
                                      sample_crops, write_crop)
from occlusynth.utils import get_repo_root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--densities", type=int, nargs="*", default=[10, 20],
                   help="extra view counts to fuse train scenes at (n6 is copied)")
    p.add_argument("--crops", type=int, default=15,
                   help="target crops per (scene, density)")
    args = p.parse_args()

    root = get_repo_root()
    cache_dir = root / "data/completer_grids"
    src = root / "data/completer_crops"
    dst = root / "data/completer_crops_md"
    (dst / "train").mkdir(parents=True, exist_ok=True)
    (dst / "val").mkdir(parents=True, exist_ok=True)

    train, val = completer_split(root)
    print(f"split: {len(train)} train / {len(val)} val scenes")

    # 1) copy the existing n6 crops verbatim (train + val) — keeps val identical
    n_copied = 0
    for split in ("train", "val"):
        for f in sorted((src / split).glob("*.npz")):
            out = dst / split / f.name
            if not out.exists():
                shutil.copy2(f, out)
            n_copied += 1
    print(f"copied {n_copied} existing n6 crops")

    # 2) add higher-density crops for TRAIN scenes only
    totals = {n: 0 for n in args.densities}
    for n in args.densities:
        for scene in train:
            t0 = time.time()
            # resumable: skip if this (scene, density) was already written
            if list((dst / "train").glob(f"{scene}_n{n}_crop*.npz")):
                continue
            try:
                grid, gt_sdf = build_scene_arrays(scene, n, cache_dir, root)
            except Exception as e:                       # missing frames/mesh
                print(f"  [n{n}] {scene}: SKIP ({type(e).__name__}: {e})")
                continue
            seed = int.from_bytes(
                hashlib.md5(f"{scene}_n{n}".encode()).digest()[:4], "little")
            rng = np.random.default_rng(seed)
            crops, attempts = sample_crops(grid, gt_sdf, args.crops, rng)
            for i, (corner, _) in enumerate(crops):
                write_crop(dst / "train" / f"{scene}_n{n}_crop{i:02d}.npz",
                           grid, gt_sdf, corner, scene)
            totals[n] += len(crops)
            print(f"  [n{n}] {scene}: +{len(crops)}/{args.crops} crops "
                  f"({attempts} att, grid {grid.dims}, {time.time()-t0:.0f}s)")

    # 3) report
    tr = len(list((dst / "train").glob("*.npz")))
    va = len(list((dst / "val").glob("*.npz")))
    base = len(list((src / "train").glob("*.npz")))
    print(f"\ndone. train {tr} crops ({base} → {tr}, "
          f"{tr / max(base,1):.2f}x)  |  val {va} crops (unchanged)")
    for n, c in totals.items():
        print(f"  n{n}: +{c} crops")


if __name__ == "__main__":
    main()
