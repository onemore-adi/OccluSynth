#!/usr/bin/env python
"""
check_completer_alignment.py — verify mesh_to_tsdf() and fuse_visibility()
share the same world frame, voxel for voxel.

Logs both grids to one Rerun recording:
  gt_sdf/outside  blue   — GT SDF > 0 (thin shell near surface, subsampled)
  gt_sdf/inside   orange — GT SDF < 0
  partial/free    green
  partial/surface red
  partial/occluded amber

Pass criterion: red surface voxels sit on the GT SDF zero-crossing —
median |GT_SDF| at surface voxels < 0.075 m (1.5 voxels).

Usage:
    .venv312/bin/python scripts/check_completer_alignment.py --scene scene0000_00
"""

import argparse
from pathlib import Path

import numpy as np

from occlusynth.fusion import (TSDFConfig, fuse_visibility_grid, mesh_to_tsdf,
                               FREE, SURFACE, OCCLUDED, CLASS_COLORS)
from occlusynth.utils import get_repo_root

from run_visibility import build_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0000_00")
    p.add_argument("--n_frames", type=int, default=6)
    args = p.parse_args()

    root = get_repo_root()
    mesh_path = root / f"data/scannet/scans/{args.scene}/{args.scene}_vh_clean_2.ply"

    print(f"[1/3] fusing partial grid ({args.scene}, {args.n_frames} frames, GT depth)")
    frames, _ = build_frames(args.scene, args.n_frames, use_gt_depth=True)
    cfg = TSDFConfig()
    grid = fuse_visibility_grid(frames, cfg)
    nx, ny, nz = grid.dims
    print(f"      grid {nx}×{ny}×{nz}, origin {grid.origin.round(3)}")

    print(f"[2/3] sampling GT SDF from {mesh_path.name}")
    gt_sdf = mesh_to_tsdf(str(mesh_path), cfg.voxel_size, grid.origin, grid.dims)

    # ── numeric alignment check ──────────────────────────────────────────────
    surf = grid.state == SURFACE
    abs_at_surf = np.abs(gt_sdf[surf])
    med = float(np.median(abs_at_surf))
    frac_band = float((abs_at_surf < 1.5 * cfg.voxel_size).mean())
    print(f"      surface voxels: {surf.sum():,}")
    print(f"      median |GT_SDF| at surface voxels: {med*100:.2f} cm "
          f"(must be < 7.5 cm)")
    print(f"      fraction within 1.5 voxels of zero-crossing: {frac_band*100:.1f}%")

    # ── Rerun ────────────────────────────────────────────────────────────────
    print("[3/3] writing Rerun recording")
    import rerun as rr
    out = root / "demo_outputs/completer_alignment"
    out.mkdir(parents=True, exist_ok=True)
    rrd = out / f"{args.scene}_alignment.rrd"
    rr.init("occlusynth-completer-alignment", spawn=False)
    rr.save(str(rrd))

    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                          indexing="ij")
    centers = np.stack(
        [grid.origin[0] + (I + 0.5) * cfg.voxel_size,
         grid.origin[1] + (J + 0.5) * cfg.voxel_size,
         grid.origin[2] + (K + 0.5) * cfg.voxel_size], axis=-1).astype(np.float32)

    rng = np.random.default_rng(0)

    def log_pts(path, mask, color, cap=60_000, radius=0.012):
        pts = centers[mask]
        if len(pts) > cap:
            pts = pts[rng.choice(len(pts), cap, replace=False)]
        if len(pts):
            rr.log(path, rr.Points3D(pts, colors=color, radii=radius), static=True)

    # GT SDF shells near the zero-crossing (full field would be 1M+ points)
    shell = np.abs(gt_sdf) < 2 * cfg.voxel_size
    log_pts("gt_sdf/outside", shell & (gt_sdf >= 0), (70, 120, 240))   # blue
    log_pts("gt_sdf/inside",  shell & (gt_sdf < 0),  (250, 130, 40))   # orange

    log_pts("partial/free",     grid.state == FREE,     CLASS_COLORS[FREE], cap=30_000)
    log_pts("partial/surface",  grid.state == SURFACE,  CLASS_COLORS[SURFACE])
    log_pts("partial/occluded", grid.state == OCCLUDED, CLASS_COLORS[OCCLUDED], cap=40_000)

    print(f"      → {rrd}   (open with: rerun {rrd})")

    ok = med < 1.5 * cfg.voxel_size
    print(f"\nALIGNMENT {'OK' if ok else 'FAILED — origins differ, fix before '
          'generating training data'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
