#!/usr/bin/env python
"""
run_fusion.py — TSDF fusion with GT poses + VGGT-predicted (or GT) depth.

Orchestration only.  All real logic lives in src/occlusynth/.

Usage:
    python scripts/run_fusion.py --scene scene0000_00 --n_frames 6
    python scripts/run_fusion.py --scene scene0000_00 --n_frames 6 --use_gt_depth

Depth coordinate frame: 382×512 (ScanNetDataset target_size=512).
Intrinsics: colour K scaled to 382×512 — consistent with depth frame.
Poses: ScanNet GT (always).  See docs/architecture.md §Camera Pose Strategy.
"""

import argparse
import time
from pathlib import Path

import numpy as np

# ── library imports ───────────────────────────────────────────────────────────
from occlusynth.data   import ScanNetDataset
from occlusynth.models import (
    VGGTWrapper,
    load_grounding,
    ground_scene,
    apply_metric_correction,
)
from occlusynth.models.depth_calibration import resize_to_gt
from occlusynth.fusion import TSDFConfig, fuse
from occlusynth.utils  import get_repo_root


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene",        default="scene0000_00")
    p.add_argument("--n_frames",     type=int, default=6)
    p.add_argument("--use_gt_depth", action="store_true",
                   help="use ScanNet GT depth (skip VGGT inference)")
    p.add_argument("--out_dir",      default=None)
    args = p.parse_args()

    root         = get_repo_root()
    grounding_dir = root / "demo_outputs/grounding"
    cache_dir    = root / "demo_outputs/pred_cache"
    out_dir      = Path(args.out_dir) if args.out_dir else root / "demo_outputs/tsdf_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── dataset — unified 382×512 colour-camera frame ─────────────────────────
    dataset = ScanNetDataset(n_frames=args.n_frames, split="all")
    try:
        scene_idx = dataset.scenes.index(args.scene)
    except ValueError:
        raise SystemExit(f"Scene '{args.scene}' not found in data root.")

    item       = dataset[scene_idx]
    stems      = item["frame_idx"]
    gt_depth   = item["depth_gt"].numpy()    # (N, H_c, W_c) float32, metres
    poses      = item["pose"].numpy()        # (N, 4, 4) float32 c2w
    K          = item["intrinsics"][0].numpy()  # (3, 3) — same for all frames

    scene_dir  = dataset.root / args.scene

    span = np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3])
    print(f"Scene       : {args.scene}")
    print(f"Frames      : {len(stems)}  ({stems[0]} … {stems[-1]})")
    print(f"Resolution  : {gt_depth.shape[1]}×{gt_depth.shape[2]}  "
          f"(colour-camera frame, K scaled)")
    print(f"Depth source: {'GT (ScanNet)' if args.use_gt_depth else 'VGGT-Omega (RANSAC-calibrated)'}")
    print(f"Pose source : GT (ScanNet)  ← see docs/architecture.md")
    print(f"Traj span   : {span:.3f} m")

    # ── depth ─────────────────────────────────────────────────────────────────
    if args.use_gt_depth:
        depths_m  = [gt_depth[i] for i in range(len(stems))]
        depth_tag = "gtdepth"

    else:
        wrapper = VGGTWrapper()
        result  = wrapper.predict_cached(
            args.scene, stems, scene_dir, cache_dir=cache_dir
        )
        vggt_raw = result["depth"]   # (N, H_pred, W_pred) arbitrary scale

        # Load persisted (a, b) params if available; otherwise compute + save.
        json_path = grounding_dir / f"{args.scene}_grounding.json"
        if json_path.exists():
            params = load_grounding(json_path)
            print(f"\nLoaded grounding params from {json_path.name}")
        else:
            print(f"\nNo grounding JSON found — running ground_scene() …")
            json_path = ground_scene(
                args.scene, dataset, wrapper,
                out_dir=grounding_dir, cache_dir=cache_dir,
            )
            params = load_grounding(json_path)
            print(f"  Saved → {json_path}")

        print("\nPer-frame scale factors (a·d_pred + b → metres):")
        depths_m = []
        for i, stem in enumerate(stems):
            a, b = params[stem]
            pred_resized = resize_to_gt(vggt_raw[i], gt_depth[i])  # → 382×512
            depth_m = apply_metric_correction(pred_resized, a, b)
            depths_m.append(depth_m)
            print(f"  [{stem}]  a={a:.4f}  b={b:+.4f}")

        depth_tag = "vggtdepth"

    # ── build frames_data ─────────────────────────────────────────────────────
    frames_data = [
        {
            "rgb_path": scene_dir / "color" / f"{stem}.jpg",
            "depth_m":  depths_m[i],
            "K":        K,
            "c2w":      poses[i],    # ← GT POSE, always
        }
        for i, stem in enumerate(stems)
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
