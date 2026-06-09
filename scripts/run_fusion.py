#!/usr/bin/env python
"""
run_fusion.py — TSDF fusion with GT poses + VGGT-predicted (or GT) depth.

Orchestration only.  All real logic lives in src/occlusynth/.

Usage:
    python scripts/run_fusion.py --scene scene0000_00 --n_frames 20
    python scripts/run_fusion.py --scene scene0000_00 --n_frames 20 --use_gt_depth
"""

import argparse
import time
from pathlib import Path

import numpy as np

# ── library imports ───────────────────────────────────────────────────────────
from occlusynth.data    import ScanNetScene, sample_anchors
from occlusynth.models  import (VGGTWrapper, fit_perframe_ransac,
                                 resize_to_gt)
from occlusynth.fusion  import TSDFConfig, fuse
from occlusynth.utils   import get_repo_root


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene",        default="scene0000_00")
    p.add_argument("--n_frames",     type=int, default=20)
    p.add_argument("--use_gt_depth", action="store_true",
                   help="use ScanNet GT depth (skip VGGT inference)")
    p.add_argument("--out_dir",      default=None)
    args = p.parse_args()

    root      = get_repo_root()
    scene_dir = root / "data/scannet/tasks/scannet_frames_25k" / args.scene
    out_dir   = Path(args.out_dir) if args.out_dir else root / "demo_outputs/tsdf_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = ScanNetScene(scene_dir)
    stems = scene.pick_frames(args.n_frames)
    K_depth = scene.K_depth
    K_color = scene.K_color

    print(f"Scene       : {scene}")
    print(f"Frames      : {len(stems)}  ({stems[0]} … {stems[-1]})")
    print(f"Depth source: {'GT (ScanNet)' if args.use_gt_depth else 'VGGT-Omega'}")
    print(f"Pose source : GT (ScanNet)  ← see docs/architecture.md")

    # ── GT poses — always used ────────────────────────────────────────────────
    gt_poses = [scene.pose(s) for s in stems]
    span     = np.linalg.norm(gt_poses[-1][:3, 3] - gt_poses[0][:3, 3])
    print(f"Traj span   : {span:.3f} m")

    # ── depth ─────────────────────────────────────────────────────────────────
    if args.use_gt_depth:
        depths_m = [scene.depth(s) for s in stems]
        K_fusion = K_depth
        depth_tag = "gtdepth"
    else:
        wrapper  = VGGTWrapper()
        result   = wrapper.predict_cached(args.scene, stems, scene_dir,
                                          cache_dir=root / "demo_outputs/pred_cache")
        vggt_raw = result["depth"]   # (N, H_pred, W_pred) — arbitrary scale

        print("\nEstimating per-frame depth scale (RANSAC):")
        depths_m = []
        for i, s in enumerate(stems):
            gt_d = scene.depth(s)
            pred_resized = resize_to_gt(vggt_raw[i], gt_d)
            p_a, g_a, _ = sample_anchors(pred_resized, gt_d, n=500,
                                          rng=np.random.default_rng(i))
            a, b, n_in, ratio = fit_perframe_ransac(p_a, g_a)
            print(f"  [{s}]  a={a:.4f}  b={b:.4f}  inliers={ratio*100:.0f}%")
            depths_m.append((pred_resized * a + b).astype(np.float32))

        K_fusion  = K_color
        depth_tag = "vggtdepth"

    # ── build frames_data ─────────────────────────────────────────────────────
    frames_data = [
        {
            "rgb_path": scene.color_path(s),
            "depth_m":  depths_m[i],
            "K":        K_fusion,
            "c2w":      gt_poses[i],   # ← GT POSE, always
        }
        for i, s in enumerate(stems)
    ]

    # ── fuse ──────────────────────────────────────────────────────────────────
    tag = f"{args.scene}_n{len(stems)}_{depth_tag}_gtpose"
    cfg = TSDFConfig()
    print(f"\nFusing {len(frames_data)} frames  "
          f"(voxel={cfg.voxel_size*100:.0f}cm)…")
    t0  = time.time()
    out = fuse(frames_data, out_dir, tag, cfg)
    print(f"Fusion time : {time.time()-t0:.1f}s")
    print(f"Output      : {out}")


if __name__ == "__main__":
    main()
