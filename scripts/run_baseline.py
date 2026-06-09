#!/usr/bin/env python
"""
run_baseline.py — "Before adapter" VGGT-Omega baseline on a ScanNet scene.

Orchestration only.  All real logic lives in src/occlusynth/.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --scene scene0231_00 --n_frames 6 --out_dir demo_outputs/my_run
"""

import argparse
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image

# ── library imports (everything real is here) ─────────────────────────────────
from occlusynth.data     import ScanNetScene, sample_anchors
from occlusynth.models   import VGGTWrapper, fit_perframe_scalar, evaluate, resize_to_gt
from occlusynth.viz      import colorize
from occlusynth.utils    import get_repo_root


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene",    default="scene0000_00")
    p.add_argument("--n_frames", type=int, default=6)
    p.add_argument("--out_dir",  default=None)
    args = p.parse_args()

    root      = get_repo_root()
    scene_dir = root / "data/scannet/tasks/scannet_frames_25k" / args.scene
    out_dir   = Path(args.out_dir) if args.out_dir else root / "demo_outputs/scannet_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────────
    scene = ScanNetScene(scene_dir)
    stems = scene.pick_frames(args.n_frames)
    print(f"Scene: {scene}  frames: {stems}")

    # ── predict ───────────────────────────────────────────────────────────────
    wrapper = VGGTWrapper()
    result  = wrapper.predict_cached(args.scene, stems, scene_dir,
                                     cache_dir=root / "demo_outputs/pred_cache")
    depth_pred = result["depth"]      # (N, H, W)
    H, W       = result["image_hw"]

    # ── per-frame analysis ────────────────────────────────────────────────────
    gt_depths = [scene.depth(s) for s in stems]
    all_gt_v  = np.concatenate([g[g > 0] for g in gt_depths])
    vmin, vmax = float(np.percentile(all_gt_v, 2)), float(np.percentile(all_gt_v, 98))

    print(f"\n{'Frame':<10} {'Scale':>8} {'ARE':>8} {'δ<1.25':>8}")
    print("-" * 40)
    for i, stem in enumerate(stems):
        pred_resized = resize_to_gt(depth_pred[i], gt_depths[i])
        p_a, g_a, _  = sample_anchors(pred_resized, gt_depths[i], n=500,
                                       rng=np.random.default_rng(i))
        scale = fit_perframe_scalar(p_a, g_a)
        m     = evaluate(pred_resized, gt_depths[i], scale)
        print(f"{stem:<10} {scale:>8.4f} {m['ARE']:>8.4f} {m['delta_125']*100:>7.1f}%")

        # save overlay
        scaled_vis = colorize(pred_resized * scale, vmin, vmax)
        gt_vis     = colorize(gt_depths[i],          vmin, vmax)
        rgb        = np.array(Image.open(scene.color_path(stem)).convert("RGB"))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, img, title in zip(axes,
                                   [rgb, gt_vis, scaled_vis],
                                   ["RGB Input", "GT Depth", "VGGT Depth (scaled)"]):
            ax.imshow(img); ax.set_title(title, fontweight="bold"); ax.axis("off")
        fig.suptitle(f"{args.scene} | {stem} | scale={scale:.4f} ARE={m['ARE']:.4f}",
                     fontsize=11)
        fig.savefig(out_dir / f"frame_{i:02d}_{stem}_overview.png",
                    bbox_inches="tight", dpi=110)
        plt.close(fig)

    print(f"\nOutputs → {out_dir}/")


if __name__ == "__main__":
    main()
