#!/usr/bin/env python
"""
run_scale_fit.py — Four closed-form depth scale fits on ScanNet anchor pixels.

Orchestration only.  All real logic lives in src/occlusynth/.

Usage:
    python scripts/run_scale_fit.py
    python scripts/run_scale_fit.py --scene scene0231_00 --n_frames 6 --n_anchors 500
    python scripts/run_scale_fit.py --use_gt_depth   # sanity check
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── library imports ───────────────────────────────────────────────────────────
from occlusynth.data   import ScanNetScene, sample_anchors
from occlusynth.models import (
    VGGTWrapper,
    fit_global_scalar, fit_perframe_scalar, fit_perframe_ls, fit_perframe_ransac,
    evaluate, resize_to_gt,
)
from occlusynth.viz    import colorize
from occlusynth.utils  import get_repo_root


METHOD_NAMES = ["GlobalScalar", "PerFrameScalar", "PerFrameLS", "PerFrameRANSAC"]


def run_fits(pred_at_gt, gt_depth, anchors, s_global):
    """Run all four fits for one frame; return list of result dicts."""
    p_a, g_a = anchors["p"], anchors["g"]

    s_pf           = fit_perframe_scalar(p_a, g_a)
    a_ls, b_ls, _  = fit_perframe_ls(p_a, g_a)
    a_rs, b_rs, n_in, in_ratio = fit_perframe_ransac(p_a, g_a)

    results = [
        {"method": "GlobalScalar",    "a": s_global, "b": 0.0},
        {"method": "PerFrameScalar",  "a": s_pf,     "b": 0.0},
        {"method": "PerFrameLS",      "a": a_ls,     "b": b_ls},
        {"method": "PerFrameRANSAC",  "a": a_rs,     "b": b_rs,
         "n_inliers": n_in, "n_anchors": len(p_a), "inlier_ratio": in_ratio},
    ]
    for r in results:
        r["metrics"] = evaluate(pred_at_gt, gt_depth, r["a"], r["b"])
    return results


def print_table(stems, per_frame_results):
    print(f"\n  {'frame':<10} {'method':<20} {'a':>10} {'b':>10} "
          f"{'ARE':>8} {'RMSE':>8} {'δ<1.25':>8}")
    print(f"  {'':-<10} {'':-<20} {'':-<10} {'':-<10} {'':-<8} {'':-<8} {'':-<8}")
    for stem, frame_res in zip(stems, per_frame_results):
        for r in frame_res:
            extra = (f"  inliers={r['n_inliers']}/{r['n_anchors']}"
                     if "n_inliers" in r else "")
            print(f"  {stem:<10} {r['method']:<20} {r['a']:>10.4f} {r['b']:>10.4f} "
                  f"{r['metrics']['ARE']:>8.4f} {r['metrics']['RMSE']:>8.4f} "
                  f"{r['metrics']['delta_125']*100:>7.1f}%{extra}")


def print_summary(per_frame_results):
    print(f"\n  {'Method':<20} {'mean ARE':>10} {'mean RMSE':>10} "
          f"{'δ<1.05':>8} {'δ<1.10':>8} {'δ<1.25':>8}")
    print(f"  {'':-<20} {'':-<10} {'':-<10} {'':-<8} {'':-<8} {'':-<8}")
    for mk in METHOD_NAMES:
        rows = [r for fr in per_frame_results for r in fr if r["method"] == mk]
        m_are  = np.mean([r["metrics"]["ARE"]       for r in rows])
        m_rmse = np.mean([r["metrics"]["RMSE"]      for r in rows])
        d105   = np.mean([r["metrics"]["delta_105"] for r in rows])
        d110   = np.mean([r["metrics"]["delta_110"] for r in rows])
        d125   = np.mean([r["metrics"]["delta_125"] for r in rows])
        print(f"  {mk:<20} {m_are:>10.4f} {m_rmse:>10.4f} "
              f"{d105*100:>7.1f}% {d110*100:>7.1f}% {d125*100:>7.1f}%")


def save_summary_bar(per_frame_results, out_path):
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    are_vals = [np.mean([r["metrics"]["ARE"] for fr in per_frame_results
                         for r in fr if r["method"] == mk]) for mk in METHOD_NAMES]
    d125_vals = [np.mean([r["metrics"]["delta_125"] for fr in per_frame_results
                          for r in fr if r["method"] == mk]) for mk in METHOD_NAMES]
    x = np.arange(len(METHOD_NAMES))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, vals, ylabel, fmt in [
        (ax1, are_vals,        "Mean ARE ↓",       "{:.4f}"),
        (ax2, [v*100 for v in d125_vals], "δ<1.25 % ↑", "{:.1f}%"),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(METHOD_NAMES, rotation=15, ha="right")
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    fmt.format(v), ha="center", va="bottom", fontsize=9)
    fig.suptitle("Scale Fit Method Comparison", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene",        default="scene0000_00")
    p.add_argument("--n_frames",     type=int, default=6)
    p.add_argument("--n_anchors",    type=int, default=500)
    p.add_argument("--use_gt_depth", action="store_true")
    p.add_argument("--out_dir",      default=None)
    args = p.parse_args()

    root      = get_repo_root()
    scene_dir = root / "data/scannet/tasks/scannet_frames_25k" / args.scene
    out_dir   = Path(args.out_dir) if args.out_dir else root / "demo_outputs/scale_fit"
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = ScanNetScene(scene_dir)
    stems = scene.pick_frames(args.n_frames)
    print(f"Scene: {scene}  stems: {stems}")

    # ── get predictions ───────────────────────────────────────────────────────
    if args.use_gt_depth:
        depth_stack = np.stack([scene.depth(s) for s in stems])
    else:
        wrapper     = VGGTWrapper()
        result      = wrapper.predict_cached(args.scene, stems, scene_dir,
                                             cache_dir=root / "demo_outputs/pred_cache")
        depth_stack = result["depth"]

    # ── load GT + resize predictions ──────────────────────────────────────────
    gt_depths  = [scene.depth(s) for s in stems]
    pred_at_gt = [resize_to_gt(depth_stack[i], gt_depths[i]) for i in range(len(stems))]

    # ── sample anchors ────────────────────────────────────────────────────────
    anchors = [
        dict(zip(("p","g","yx"),
                 sample_anchors(pred_at_gt[i], gt_depths[i], n=args.n_anchors,
                                rng=np.random.default_rng(i * 1000 + 7))))
        for i in range(len(stems))
    ]
    all_p = np.concatenate([a["p"] for a in anchors])
    all_g = np.concatenate([a["g"] for a in anchors])

    # ── fit method 1 (global) ─────────────────────────────────────────────────
    s_global = fit_global_scalar(all_p, all_g)
    print(f"\nGlobal scalar: {s_global:.6f}  (d_metric = {s_global:.4f} × d_pred)")

    # ── fit methods 2/3/4 per frame ───────────────────────────────────────────
    per_frame_results = [
        run_fits(pred_at_gt[i], gt_depths[i], anchors[i], s_global)
        for i in range(len(stems))
    ]

    print_table(stems, per_frame_results)
    print("\nSummary (mean across frames):")
    print_summary(per_frame_results)

    save_summary_bar(per_frame_results, out_dir / "summary_bar.png")
    print(f"\nOutputs → {out_dir}/")


if __name__ == "__main__":
    main()
