#!/usr/bin/env python
"""
run_scale_fit.py — Compare four closed-form depth scale fits on ScanNet.

Data flows through ScanNetDataset (GT depth at 382×512, colour-camera frame)
so the comparison lives on the same coordinate grid as eventual TSDF fusion.

Usage:
    python scripts/run_scale_fit.py
    python scripts/run_scale_fit.py --scene scene0231_00 --n_frames 6
    python scripts/run_scale_fit.py --use_gt_depth   # sanity check: fit GT to GT
    python scripts/run_scale_fit.py --save_grounding  # also persist RANSAC (a,b) JSON
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from occlusynth.data   import ScanNetDataset
from occlusynth.models import (
    VGGTWrapper,
    fit_global_scalar, fit_perframe_scalar, fit_perframe_ls, fit_perframe_ransac,
    evaluate, resize_to_gt,
    fit_metric_depth, save_grounding,
)
from occlusynth.utils  import get_repo_root, get_config


METHOD_NAMES = ["GlobalScalar", "PerFrameScalar", "PerFrameLS", "PerFrameRANSAC"]


# ── per-frame fit helper ──────────────────────────────────────────────────────

def run_fits(pred_aligned, gt, p_anchors, g_anchors, s_global):
    """Run all four fits for one frame; return list of result dicts."""
    s_pf           = fit_perframe_scalar(p_anchors, g_anchors)
    a_ls, b_ls, _  = fit_perframe_ls(p_anchors, g_anchors)
    a_rs, b_rs, n_in, in_ratio = fit_perframe_ransac(p_anchors, g_anchors)

    results = [
        {"method": "GlobalScalar",   "a": s_global, "b": 0.0},
        {"method": "PerFrameScalar", "a": s_pf,     "b": 0.0},
        {"method": "PerFrameLS",     "a": a_ls,     "b": b_ls},
        {"method": "PerFrameRANSAC", "a": a_rs,     "b": b_rs,
         "n_inliers": n_in, "n_anchors": len(p_anchors), "inlier_ratio": in_ratio},
    ]
    for r in results:
        r["metrics"] = evaluate(pred_aligned, gt, r["a"], r["b"])
    return results


# ── table printing ────────────────────────────────────────────────────────────

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
        (ax1, are_vals,                "Mean ARE ↓",  "{:.4f}"),
        (ax2, [v * 100 for v in d125_vals], "δ<1.25 % ↑", "{:.1f}%"),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(METHOD_NAMES, rotation=15, ha="right")
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    fmt.format(v), ha="center", va="bottom", fontsize=9)
    fig.suptitle("Scale Fit Method Comparison", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene",          default="scene0000_00")
    p.add_argument("--n_frames",       type=int, default=6)
    p.add_argument("--n_anchors",      type=int, default=500)
    p.add_argument("--use_gt_depth",   action="store_true",
                   help="fit GT depth to itself — sanity check, should give ARE≈0")
    p.add_argument("--save_grounding", action="store_true",
                   help="persist RANSAC (a,b) to demo_outputs/grounding/")
    p.add_argument("--out_dir",        default=None)
    args = p.parse_args()

    root    = get_repo_root()
    cfg     = get_config()
    out_dir = Path(args.out_dir) if args.out_dir else root / "demo_outputs/scale_fit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── data: ScanNetDataset for GT depth at colour-camera resolution ─────────
    ds = ScanNetDataset(split="all", n_frames=args.n_frames)
    try:
        scene_idx = ds.scenes.index(args.scene)
    except ValueError:
        raise SystemExit(f"Scene '{args.scene}' not found in dataset root {ds.root}")

    item  = ds[scene_idx]
    stems = item["frame_idx"]
    gt_np = item["depth_gt"].numpy()        # (N, H_c, W_c) float32, metres, 0=invalid
    H_c, W_c = gt_np.shape[1], gt_np.shape[2]
    print(f"Scene: {args.scene}  frames: {stems}  GT resolution: {H_c}×{W_c}")

    # ── predictions ───────────────────────────────────────────────────────────
    if args.use_gt_depth:
        pred_np = gt_np.copy()
    else:
        wrapper    = VGGTWrapper()
        scene_dir  = ds.root / args.scene
        vggt_result = wrapper.predict_cached(
            args.scene, stems, scene_dir,
            cache_dir=root / cfg["cache_dir"],
        )
        pred_np = vggt_result["depth"]      # (N, H_p, W_p) — may differ from GT res

    # Align predictions to the GT (colour-camera) resolution.
    # bilinear is correct for VGGT predictions (continuous); GT depth uses
    # INTER_NEAREST in ScanNetDataset, which is already done.
    pred_aligned = np.stack([
        resize_to_gt(pred_np[i], gt_np[i]) if pred_np[i].shape != gt_np[i].shape
        else pred_np[i]
        for i in range(len(stems))
    ])                                      # (N, H_c, W_c)

    # ── sample anchors ────────────────────────────────────────────────────────
    from occlusynth.data import sample_anchors
    anchors = [
        dict(zip(("p", "g", "yx"),
                 sample_anchors(pred_aligned[i], gt_np[i], n=args.n_anchors,
                                rng=np.random.default_rng(i * 1000 + 7))))
        for i in range(len(stems))
    ]
    all_p = np.concatenate([a["p"] for a in anchors])
    all_g = np.concatenate([a["g"] for a in anchors])

    # ── global scalar (method 1 — needs cross-frame pool) ────────────────────
    s_global = fit_global_scalar(all_p, all_g)
    print(f"\nGlobal scalar: {s_global:.6f}  (d_metric = {s_global:.4f} × d_pred)")

    # ── per-frame fits (methods 2/3/4) ────────────────────────────────────────
    per_frame_results = [
        run_fits(pred_aligned[i], gt_np[i], anchors[i]["p"], anchors[i]["g"], s_global)
        for i in range(len(stems))
    ]

    print_table(stems, per_frame_results)
    print("\nSummary (mean across frames):")
    print_summary(per_frame_results)

    save_summary_bar(per_frame_results, out_dir / "summary_bar.png")

    # ── optional JSON persistence ─────────────────────────────────────────────
    if args.save_grounding:
        ransac_params = {
            stem: (r["a"], r["b"])
            for stem, fr in zip(stems, per_frame_results)
            for r in fr if r["method"] == "PerFrameRANSAC"
        }
        grounding_dir = root / "demo_outputs/grounding"
        json_path = save_grounding(ransac_params, grounding_dir, args.scene)
        print(f"\nGrounding params saved → {json_path}")

    print(f"\nOutputs → {out_dir}/")


if __name__ == "__main__":
    main()
