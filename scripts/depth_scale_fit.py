#!/usr/bin/env python
"""
depth_scale_fit.py — Four closed-form depth scale fits on ScanNet anchor pixels.

Methods (all closed-form, no gradient descent):
  1. global_scalar    — one scale for the whole dataset (median ratio over all anchors)
  2. perframe_scalar  — per-frame median ratio (d_gt / d_pred)
  3. perframe_ls      — per-frame scale + shift via normal equations
  4. perframe_ransac  — per-frame RANSAC scale + shift, outlier-robust

Evaluation metrics per method per frame:
  - ARE  : mean absolute relative error  mean(|d_scaled - d_gt| / d_gt)
  - RMSE : root mean squared error  sqrt(mean((d_scaled - d_gt)²))
  - δ<1.05, δ<1.10, δ<1.25 : inlier fractions at three accuracy thresholds

Usage:
    # First run: runs VGGT inference, caches raw depth to .npy, fits + plots
    python scripts/depth_scale_fit.py

    # Subsequent runs: loads cached predictions, no VGGT inference needed
    python scripts/depth_scale_fit.py   # auto-detects cache

    # Choose a different scene / frame count
    python scripts/depth_scale_fit.py --scene scene0231_00 --n_frames 6

    # Skip VGGT entirely — fit against GT depth only (sanity check, should be near perfect)
    python scripts/depth_scale_fit.py --use_gt_depth
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VGGT_SRC   = os.path.join(REPO_ROOT, "vggt", "vggt-omega")
CKPT       = os.path.join(VGGT_SRC, "checkpoints", "vggt_omega_1b_512.pt")
DATA_ROOT  = os.path.join(REPO_ROOT, "data", "scannet", "tasks", "scannet_frames_25k")
OUT_DIR    = os.path.join(REPO_ROOT, "demo_outputs", "scale_fit")
CACHE_DIR  = os.path.join(REPO_ROOT, "demo_outputs", "pred_cache")
sys.path.insert(0, VGGT_SRC)

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

N_ANCHORS    = 500       # anchor pixels per frame
IMG_RES      = 512       # VGGT input resolution
DEPTH_MM     = 1000.0    # ScanNet PNG → metres
DEPTH_MAX_M  = 3.5       # clip depth beyond reliable sensor range
RANSAC_ITERS = 500       # RANSAC iterations
RANSAC_THR   = 0.10      # inlier threshold: relative error < 10 %
RANSAC_SEED  = 42


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def sep(msg=""):
    w = 64
    print(f"\n── {msg} {'─' * max(0, w - len(msg) - 4)}" if msg else "─" * w)


def load_gt_depth(scene_dir, stem):
    """Load ScanNet GT depth → float32 metres.  0 = invalid."""
    raw = np.array(Image.open(os.path.join(scene_dir, "depth", stem + ".png")),
                   dtype=np.float32)
    d = raw / DEPTH_MM
    d[d > DEPTH_MAX_M] = 0.0
    return d   # (480, 640)


def load_gt_intrinsics(scene_dir):
    """Return 3×3 K for the depth camera."""
    rows = []
    with open(os.path.join(scene_dir, "intrinsics_depth.txt")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    return np.array(rows[:3], dtype=np.float32)[:, :3]


def pick_frames(scene_dir, n_frames):
    c = {os.path.splitext(f)[0]
         for f in os.listdir(os.path.join(scene_dir, "color")) if f.endswith(".jpg")}
    d = {os.path.splitext(f)[0]
         for f in os.listdir(os.path.join(scene_dir, "depth")) if f.endswith(".png")}
    p = {os.path.splitext(f)[0]
         for f in os.listdir(os.path.join(scene_dir, "pose"))  if f.endswith(".txt")}
    valid = sorted(c & d & p)
    pool  = valid[2:-2] if len(valid) > 10 else valid
    idx   = np.linspace(0, len(pool) - 1, min(n_frames, len(pool)), dtype=int)
    return [pool[i] for i in idx]


# ─────────────────────────────────────────────────────────────────────────────
# VGGT inference (with .npy cache)
# ─────────────────────────────────────────────────────────────────────────────

def run_vggt_cached(scene_id, stems, scene_dir):
    """
    Run VGGT-Omega and return raw (unscaled) depth array (N, H_pred, W_pred).
    Results are cached to CACHE_DIR/<scene_id>_<stems_hash>.npy so subsequent
    runs are instant.
    """
    import hashlib, torch
    stem_key  = hashlib.md5("_".join(stems).encode()).hexdigest()[:8]
    cache_key = f"{scene_id}_{stem_key}_res{IMG_RES}"
    cache_d   = os.path.join(CACHE_DIR, cache_key + "_depth.npy")
    cache_c   = os.path.join(CACHE_DIR, cache_key + "_conf.npy")
    cache_hw  = os.path.join(CACHE_DIR, cache_key + "_hw.npy")

    if os.path.isfile(cache_d):
        print("  [cache hit] loading predictions from disk …")
        depth = np.load(cache_d)
        conf  = np.load(cache_c)
        H, W  = np.load(cache_hw)
        print(f"  depth shape : {depth.shape}  H={H} W={W}")
        return depth, conf, (int(H), int(W))

    print("  [cache miss] running VGGT-Omega inference …")
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  device : {device}")

    model = VGGTOmega()
    t0 = time.time()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    print(f"  checkpoint loaded  {time.time()-t0:.1f}s")
    model = model.to(device).eval()

    color_paths = [os.path.join(scene_dir, "color", s + ".jpg") for s in stems]
    images  = __import__("vggt_omega.utils.load_fn", fromlist=["load_and_preprocess_images"])
    images  = images.load_and_preprocess_images(color_paths, image_resolution=IMG_RES)
    H, W    = images.shape[-2:]
    images_b = images.unsqueeze(0).to(device)

    t0 = time.time()
    with torch.inference_mode():
        preds = model(images_b)
    print(f"  inference  {time.time()-t0:.1f}s")

    depth = preds["depth"].float().cpu()[0, :, :, :, 0].numpy()   # (N, H, W)
    conf  = preds["depth_conf"].float().cpu()[0].numpy()           # (N, H, W)

    np.save(cache_d,  depth)
    np.save(cache_c,  conf)
    np.save(cache_hw, np.array([H, W]))
    print(f"  predictions cached → {CACHE_DIR}/")
    return depth, conf, (H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Anchor sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_anchors(pred_hw, gt_hw, n=N_ANCHORS, rng=None):
    """
    Stratified anchor sampling from valid pixels (both pred > 0 and gt > 0).

    Divides the image into an n_cells × n_cells grid, picks one valid pixel
    per cell when possible, then top-ups from random sampling to reach n anchors.

    Returns:
        p  : (n,) float32 — raw VGGT depth at anchor pixels
        g  : (n,) float32 — GT depth (metres) at anchor pixels
        yx : (n, 2) int   — (row, col) pixel coordinates of anchors
    """
    if rng is None:
        rng = np.random.default_rng(42)

    valid = (pred_hw > 0) & (gt_hw > 0)
    if valid.sum() < 10:
        raise ValueError(f"Only {valid.sum()} valid pixels — need at least 10.")

    H, W = pred_hw.shape

    # ── stratified: grid sampling ─────────────────────────────────────────────
    n_cells = max(1, int(np.sqrt(n)))
    cy_edges = np.linspace(0, H, n_cells + 1, dtype=int)
    cx_edges = np.linspace(0, W, n_cells + 1, dtype=int)

    strat_yx = []
    for ci in range(n_cells):
        for cj in range(n_cells):
            patch = valid[cy_edges[ci]:cy_edges[ci+1],
                          cx_edges[cj]:cx_edges[cj+1]]
            ys_local, xs_local = np.where(patch)
            if len(ys_local) == 0:
                continue
            pick = rng.integers(len(ys_local))
            strat_yx.append((ys_local[pick] + cy_edges[ci],
                             xs_local[pick] + cx_edges[cj]))

    # ── top-up from random sampling if stratified didn't reach n ─────────────
    all_ys, all_xs = np.where(valid)
    all_flat = all_ys * W + all_xs
    strat_flat = set(r * W + c for r, c in strat_yx)
    topup_pool = np.array([f for f in all_flat if f not in strat_flat])

    need = max(0, n - len(strat_yx))
    if need > 0 and len(topup_pool) > 0:
        chosen = rng.choice(topup_pool, min(need, len(topup_pool)), replace=False)
        for f in chosen:
            strat_yx.append((f // W, f % W))

    yx = np.array(strat_yx[:n], dtype=int)   # (n, 2)
    p  = pred_hw[yx[:, 0], yx[:, 1]].astype(np.float32)
    g  = gt_hw[  yx[:, 0], yx[:, 1]].astype(np.float32)
    return p, g, yx


# ─────────────────────────────────────────────────────────────────────────────
# The four closed-form fits
# ─────────────────────────────────────────────────────────────────────────────

def fit_global_scalar(all_pred, all_gt):
    """
    Method 1 — Global scalar.
    Single scale s across the whole dataset (all frames pooled).

    Closed form: s = median(d_gt / d_pred)  over all N_frames × N_anchors pairs.
    Rationale: median is robust to depth outliers and bimodal distributions;
    equivalent to minimising the L1 loss on log-depth ratios.

    Returns:
        s (float) — apply as  d_scaled = s * d_pred
    """
    ratios = all_gt / np.maximum(all_pred, 1e-9)
    return float(np.median(ratios))


def fit_perframe_scalar(pred_anchors, gt_anchors):
    """
    Method 2 — Per-frame scalar via median ratio.
    One scale per frame, independent of other frames.

    Closed form: s_f = median(d_gt_f / d_pred_f)
    One degree of freedom.  Fast and robust.  Correct when the only
    error is a global multiplicative factor (no additive bias).

    Returns:
        s (float) — apply as  d_scaled = s * d_pred
    """
    ratios = gt_anchors / np.maximum(pred_anchors, 1e-9)
    return float(np.median(ratios))


def fit_perframe_ls(pred_anchors, gt_anchors):
    """
    Method 3 — Per-frame scale + shift via least-squares (normal equations).
    Models  d_gt ≈ a * d_pred + b.

    Closed form: [a, b]^T = (A^T A)^{-1} A^T g
    where  A = [[p_0, 1], [p_1, 1], …, [p_n, 1]]  (n × 2)
           g = [g_0, g_1, …, g_n]^T                (n,)

    Two degrees of freedom — captures a constant additive bias in addition
    to the multiplicative scale error (e.g. camera-to-wall air gap, lens
    distortion offset).

    Returns:
        a (float) — scale
        b (float) — shift  →  d_scaled = a * d_pred + b
        residual_std (float) — std of fit residuals (quality indicator)
    """
    n    = len(pred_anchors)
    A    = np.stack([pred_anchors, np.ones(n, dtype=np.float32)], axis=1)  # (n, 2)
    g    = gt_anchors.astype(np.float64)
    A    = A.astype(np.float64)
    # Normal equations: x = (A^T A)^{-1} A^T g
    ATA  = A.T @ A                       # (2, 2)
    ATg  = A.T @ g                       # (2,)
    x    = np.linalg.solve(ATA, ATg)     # [a, b]
    resid = g - A @ x
    return float(x[0]), float(x[1]), float(resid.std())


def fit_perframe_ransac(pred_anchors, gt_anchors,
                        n_iter=RANSAC_ITERS, thr=RANSAC_THR, seed=RANSAC_SEED):
    """
    Method 4 — Per-frame RANSAC scale + shift.
    Same linear model as LS (d_gt = a * d_pred + b), but robust to gross
    outliers (sensor dropouts, reflective surfaces, transparent objects).

    Algorithm:
        for n_iter iterations:
            sample 2 anchor pairs (minimum to uniquely solve a 2-param model)
            solve [a, b] exactly from those 2 points
            count inliers: |a*p + b - g| / g < thr
        refit [a, b] via LS on the best inlier set (reduces variance)

    Inlier threshold thr is relative (fraction of GT depth), so it scales
    automatically with scene depth — no per-scene tuning needed.

    Returns:
        a (float)         — scale
        b (float)         — shift
        n_inliers (int)   — number of inliers in the best consensus set
        inlier_ratio(float) — n_inliers / total anchors
    """
    rng     = np.random.default_rng(seed)
    n       = len(pred_anchors)
    p       = pred_anchors.astype(np.float64)
    g       = gt_anchors.astype(np.float64)

    best_inliers = np.zeros(n, dtype=bool)
    best_count   = 0

    for _ in range(n_iter):
        # Minimal sample: 2 points → unique solution for a, b
        idx = rng.choice(n, 2, replace=False)
        p_s, g_s = p[idx], g[idx]
        dp = p_s[1] - p_s[0]
        if abs(dp) < 1e-9:
            continue           # degenerate sample, skip

        a_cand = (g_s[1] - g_s[0]) / dp
        b_cand = g_s[0] - a_cand * p_s[0]

        # Inliers: relative error below threshold
        pred_scaled = a_cand * p + b_cand
        rel_err     = np.abs(pred_scaled - g) / np.maximum(g, 1e-9)
        inliers     = rel_err < thr
        count       = inliers.sum()

        if count > best_count:
            best_count   = count
            best_inliers = inliers

    # Refit on inlier set using LS for lower variance
    if best_count >= 2:
        a, b, _ = fit_perframe_ls(p[best_inliers].astype(np.float32),
                                   g[best_inliers].astype(np.float32))
    else:
        # Fallback: degenerate case, use median scalar + zero shift
        a = float(np.median(g / np.maximum(p, 1e-9)))
        b = 0.0

    inlier_ratio = best_count / n
    return a, b, best_count, inlier_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def resize_to_gt(pred_hw, gt_hw):
    """Bilinear-resize pred depth to GT spatial resolution."""
    H_gt, W_gt = gt_hw.shape
    pred_pil = Image.fromarray(pred_hw.astype(np.float32))
    pred_pil = pred_pil.resize((W_gt, H_gt), Image.BILINEAR)
    return np.array(pred_pil, dtype=np.float32)


def evaluate(pred_full, gt_full, a, b=0.0):
    """
    Apply  d_scaled = a * d_pred + b  and compute metrics at valid pixels.

    Returns dict with:
        ARE   : mean |d_scaled - d_gt| / d_gt       (lower is better)
        RMSE  : sqrt(mean (d_scaled - d_gt)²)        (metres, lower is better)
        delta_105 : fraction where max(r, 1/r) < 1.05  (higher is better)
        delta_110 : fraction where max(r, 1/r) < 1.10
        delta_125 : fraction where max(r, 1/r) < 1.25
        n_valid : number of evaluated pixels
    """
    valid = (gt_full > 0) & (pred_full > 0)
    if valid.sum() == 0:
        return {k: float("nan") for k in
                ["ARE", "RMSE", "delta_105", "delta_110", "delta_125", "n_valid"]}

    p = pred_full[valid].astype(np.float64)
    g = gt_full[valid].astype(np.float64)

    scaled = a * p + b
    scaled = np.maximum(scaled, 1e-6)   # avoid log(0)

    rel_err = np.abs(scaled - g) / g
    ratio   = np.maximum(scaled / g, g / scaled)

    return {
        "ARE"       : float(np.mean(rel_err)),
        "RMSE"      : float(np.sqrt(np.mean((scaled - g) ** 2))),
        "delta_105" : float((ratio < 1.05).mean()),
        "delta_110" : float((ratio < 1.10).mean()),
        "delta_125" : float((ratio < 1.25).mean()),
        "n_valid"   : int(valid.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def colorize(d, vmin, vmax, cmap="inferno"):
    import matplotlib.cm as cm
    valid = d > 0
    norm  = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    rgba  = cm.get_cmap(cmap)(norm)
    rgb   = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb[~valid] = 30
    return rgb


def plot_per_frame_comparison(frame_idx, stem, pred_raw_pred_res,
                               gt_hw, results, vmin, vmax, out_path):
    """
    5-row × 4-column grid:
      cols: method 1 (global), method 2 (pf scalar), method 3 (pf LS), method 4 (pf RANSAC)
      rows: scaled depth image | abs error map | ratio map | error histogram | metrics text
    """
    method_names = ["Global Scalar", "PerFrame Scalar", "PerFrame LS", "PerFrame RANSAC"]
    n_methods    = 4
    n_rows       = 5

    fig = plt.figure(figsize=(20, 18))
    gs  = GridSpec(n_rows, n_methods, figure=fig,
                   hspace=0.35, wspace=0.08,
                   height_ratios=[3, 3, 3, 2, 1.2])

    pred_gt_res = resize_to_gt(pred_raw_pred_res, gt_hw)
    H_gt, W_gt  = gt_hw.shape
    valid_mask  = (gt_hw > 0) & (pred_gt_res > 0)

    for col, (name, res) in enumerate(zip(method_names, results)):
        a, b = res["a"], res["b"]
        scaled   = np.where(pred_gt_res > 0, a * pred_gt_res + b, 0).astype(np.float32)
        scaled   = np.maximum(scaled, 0)

        abs_err  = np.where(valid_mask, np.abs(scaled - gt_hw), 0).astype(np.float32)
        ratio_map = np.where(valid_mask,
                             np.maximum(scaled / np.maximum(gt_hw, 1e-6),
                                        gt_hw   / np.maximum(scaled, 1e-6)),
                             1.0).astype(np.float32)

        # Row 0: scaled depth
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(colorize(scaled, vmin, vmax))
        ax0.set_title(name, fontweight="bold", fontsize=10)
        ax0.axis("off")
        ax0.set_ylabel("Scaled depth", fontsize=8)

        # Row 1: absolute error map  (clip at 0.5 m for visibility)
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(abs_err, cmap="hot", vmin=0, vmax=0.5)
        ax1.axis("off")
        if col == 0:
            ax1.set_ylabel("Abs error (m)", fontsize=8)

        # Row 2: ratio map  (clip ratio at 2 for visibility)
        ax2 = fig.add_subplot(gs[2, col])
        ratio_vis = np.clip(ratio_map, 1.0, 2.0)
        ax2.imshow(ratio_vis, cmap="RdYlGn_r", vmin=1.0, vmax=1.5)
        ax2.axis("off")
        if col == 0:
            ax2.set_ylabel("max(p/g, g/p)", fontsize=8)

        # Row 3: error histogram
        ax3 = fig.add_subplot(gs[3, col])
        errs_valid = abs_err[valid_mask]
        if len(errs_valid):
            ax3.hist(errs_valid, bins=60, range=(0, 0.5),
                     color="#3498db", edgecolor="none", density=True)
            ax3.axvline(float(np.median(errs_valid)), color="#e74c3c",
                        linewidth=1.5, label=f"med {np.median(errs_valid):.3f}m")
        ax3.set_xlabel("Abs err (m)", fontsize=8)
        ax3.set_xlim(0, 0.5)
        ax3.legend(fontsize=7)
        ax3.set_yticks([])

        # Row 4: metrics text box
        m  = res["metrics"]
        ax4 = fig.add_subplot(gs[4, col])
        ax4.axis("off")
        txt = (f"ARE    {m['ARE']:.4f}\n"
               f"RMSE   {m['RMSE']:.4f} m\n"
               f"δ<1.05 {m['delta_105']*100:.1f}%\n"
               f"δ<1.10 {m['delta_110']*100:.1f}%\n"
               f"δ<1.25 {m['delta_125']*100:.1f}%")
        if "n_inliers" in res:
            txt += f"\nRANSAC inliers {res['inlier_ratio']*100:.0f}%"
        ax4.text(0.5, 0.5, txt, ha="center", va="center",
                 fontsize=8, fontfamily="monospace",
                 transform=ax4.transAxes,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", alpha=0.8))

    fig.suptitle(f"Depth Scale Fit Comparison  |  scene  |  frame {stem}",
                 fontsize=13, fontweight="bold", y=1.005)
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def plot_anchor_scatter(stems, all_anchor_data, all_method_fits, out_path):
    """
    One column per frame, one row per method.
    Each cell: scatter(d_pred_anchors, d_gt_anchors) + fitted line.
    """
    n_frames  = len(stems)
    n_methods = 4
    method_names = ["Global Scalar", "PerFrame Scalar", "PerFrame LS", "PerFrame RANSAC"]
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]

    fig, axes = plt.subplots(n_methods, n_frames,
                             figsize=(4 * n_frames, 4 * n_methods),
                             sharex="col", sharey="col")
    # ensure 2-D indexing even for 1 frame
    if n_frames == 1:
        axes = axes[:, np.newaxis]

    for col, (stem, ad) in enumerate(zip(stems, all_anchor_data)):
        p_anch, g_anch = ad["p"], ad["g"]
        x_line = np.linspace(p_anch.min(), p_anch.max(), 200)

        for row, (name, mfits, color) in enumerate(
                zip(method_names, all_method_fits[col], colors)):
            ax  = axes[row, col]
            ax.scatter(p_anch, g_anch, s=4, alpha=0.35, color="#aaaaaa",
                       rasterized=True, label="anchors")

            a, b = mfits["a"], mfits["b"]
            ax.plot(x_line, a * x_line + b, color=color, linewidth=2,
                    label=f"y={a:.3f}x+{b:.3f}")
            ax.plot(x_line, x_line, color="black", linewidth=0.8,
                    linestyle="--", alpha=0.4, label="identity")

            m = mfits["metrics"]
            ax.set_title(f"{name}\nARE={m['ARE']:.4f}  δ<1.25={m['delta_125']*100:.0f}%",
                         fontsize=8, fontweight="bold")
            ax.set_xlabel("d_pred (raw)", fontsize=7)
            ax.set_ylabel("d_gt (m)",     fontsize=7)
            ax.legend(fontsize=6, loc="upper left")
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.text(-0.35, 0.5, name, transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=9, fontweight="bold")

        axes[0, col].set_title(f"Frame {stem}\n{axes[0,col].get_title()}",
                                fontsize=8, fontweight="bold")

    fig.suptitle("Anchor Scatter: d_pred vs d_gt  with fitted lines",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_summary_bar(summary_df, out_path):
    """
    Grouped bar chart: ARE and δ<1.25 for each method, averaged across frames.
    """
    methods = list(summary_df.keys())
    are_vals    = [summary_df[m]["mean_ARE"]    for m in methods]
    delta_vals  = [summary_df[m]["mean_d125"]   for m in methods]
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(methods))
    w = 0.5
    bars = ax1.bar(x, are_vals, width=w, color=colors, edgecolor="white", linewidth=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("Mean ARE (lower ↓ better)", fontsize=10)
    ax1.set_title("Mean Absolute Relative Error", fontweight="bold")
    ax1.grid(axis="y", alpha=0.4)
    for bar, val in zip(bars, are_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(x, [v * 100 for v in delta_vals],
                    width=w, color=colors, edgecolor="white", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("δ < 1.25  (%, higher ↑ better)", fontsize=10)
    ax2.set_title("Accuracy: δ < 1.25", fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.4)
    for bar, val in zip(bars2, delta_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Method Comparison — Average Over All Frames",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene",      default="scene0000_00")
    p.add_argument("--n_frames",   type=int, default=6)
    p.add_argument("--n_anchors",  type=int, default=N_ANCHORS)
    p.add_argument("--use_gt_depth", action="store_true",
                   help="use ScanNet GT depth as 'predictions' — sanity check")
    args = p.parse_args()

    scene_dir = os.path.join(DATA_ROOT, args.scene)
    sep("Setup")
    print(f"  scene    : {args.scene}")
    print(f"  n_frames : {args.n_frames}")
    print(f"  anchors  : {args.n_anchors} per frame")
    print(f"  outdir   : {OUT_DIR}")

    # ── 1. frame selection ────────────────────────────────────────────────────
    stems = pick_frames(scene_dir, args.n_frames)
    sep(f"Frames ({len(stems)})")
    print(f"  {stems}")

    # ── 2. get predictions ────────────────────────────────────────────────────
    sep("Predictions")
    if args.use_gt_depth:
        print("  using GT depth as predictions (identity sanity check)")
        gt0 = load_gt_depth(scene_dir, stems[0])
        H, W = gt0.shape
        pred_stack = np.stack([load_gt_depth(scene_dir, s) for s in stems])
        pred_res = (H, W)
    else:
        pred_stack, conf_stack, pred_res = run_vggt_cached(args.scene, stems, scene_dir)

    H_pred, W_pred = pred_res

    # ── 3. load GT depths and resize predictions ──────────────────────────────
    gt_depths  = [load_gt_depth(scene_dir, s) for s in stems]
    pred_at_gt = [resize_to_gt(pred_stack[i], gt_depths[i]) for i in range(len(stems))]

    # ── 4. sample anchors ─────────────────────────────────────────────────────
    sep("Sampling anchors")
    anchors = []   # list of dicts  {p, g, yx}
    for i, stem in enumerate(stems):
        rng_i = np.random.default_rng(i * 1000 + 7)
        p_a, g_a, yx_a = sample_anchors(pred_at_gt[i], gt_depths[i],
                                          n=args.n_anchors, rng=rng_i)
        anchors.append({"p": p_a, "g": g_a, "yx": yx_a})
        print(f"  [{stem}]  sampled {len(p_a)} anchors  "
              f"pred range [{p_a.min():.3f}, {p_a.max():.3f}]  "
              f"gt range [{g_a.min():.3f}, {g_a.max():.3f}] m")

    # Global pool for method 1
    all_pred_anchors = np.concatenate([a["p"] for a in anchors])
    all_gt_anchors   = np.concatenate([a["g"] for a in anchors])

    # ── 5. fit — method 1: global scalar ──────────────────────────────────────
    sep("Fit 1 — Global Scalar")
    s_global = fit_global_scalar(all_pred_anchors, all_gt_anchors)
    print(f"  global scale  : {s_global:.6f}")
    print(f"  interpretation: d_scaled = {s_global:.4f} × d_pred")

    # ── 6. fit — methods 2/3/4 per frame; collect results ────────────────────
    sep("Fit 2/3/4 — Per-frame fits + evaluation")
    print(f"  {'frame':<10} {'method':<20} {'a':>10} {'b':>10} "
          f"{'ARE':>8} {'RMSE':>8} {'δ<1.25':>8} {'extra'}")
    print(f"  {'-'*8:<10} {'-'*18:<20} {'-'*10} {'-'*10} "
          f"{'-'*8} {'-'*8} {'-'*8}")

    per_frame_results = []   # list of [res_global, res_scalar, res_ls, res_ransac]
    all_method_fits   = []   # parallel to per_frame_results, for scatter plot

    for i, stem in enumerate(stems):
        p_a, g_a = anchors[i]["p"], anchors[i]["g"]
        pred_full = pred_at_gt[i]
        gt_full   = gt_depths[i]

        frame_results = []
        frame_fits    = []

        # --- global scalar (same a for all frames) ---
        m_global = evaluate(pred_full, gt_full, s_global)
        r_global = {"a": s_global, "b": 0.0, "metrics": m_global,
                    "method": "GlobalScalar"}
        frame_results.append(r_global)
        frame_fits.append(r_global)
        print(f"  {stem:<10} {'GlobalScalar':<20} {s_global:>10.4f} {'0':>10} "
              f"{m_global['ARE']:>8.4f} {m_global['RMSE']:>8.4f} "
              f"{m_global['delta_125']*100:>7.1f}%")

        # --- per-frame scalar ---
        s_pf = fit_perframe_scalar(p_a, g_a)
        m_pf = evaluate(pred_full, gt_full, s_pf)
        r_pf = {"a": s_pf, "b": 0.0, "metrics": m_pf, "method": "PerFrameScalar"}
        frame_results.append(r_pf)
        frame_fits.append(r_pf)
        print(f"  {stem:<10} {'PerFrameScalar':<20} {s_pf:>10.4f} {'0':>10} "
              f"{m_pf['ARE']:>8.4f} {m_pf['RMSE']:>8.4f} "
              f"{m_pf['delta_125']*100:>7.1f}%")

        # --- per-frame LS ---
        a_ls, b_ls, resid_std = fit_perframe_ls(p_a, g_a)
        m_ls = evaluate(pred_full, gt_full, a_ls, b_ls)
        r_ls = {"a": a_ls, "b": b_ls, "metrics": m_ls,
                "resid_std": resid_std, "method": "PerFrameLS"}
        frame_results.append(r_ls)
        frame_fits.append(r_ls)
        print(f"  {stem:<10} {'PerFrameLS':<20} {a_ls:>10.4f} {b_ls:>10.4f} "
              f"{m_ls['ARE']:>8.4f} {m_ls['RMSE']:>8.4f} "
              f"{m_ls['delta_125']*100:>7.1f}%  resid_std={resid_std:.4f}")

        # --- per-frame RANSAC ---
        a_rs, b_rs, n_in, in_ratio = fit_perframe_ransac(p_a, g_a)
        m_rs = evaluate(pred_full, gt_full, a_rs, b_rs)
        r_rs = {"a": a_rs, "b": b_rs, "metrics": m_rs,
                "n_inliers": n_in, "inlier_ratio": in_ratio,
                "method": "PerFrameRANSAC"}
        frame_results.append(r_rs)
        frame_fits.append(r_rs)
        print(f"  {stem:<10} {'PerFrameRANSAC':<20} {a_rs:>10.4f} {b_rs:>10.4f} "
              f"{m_rs['ARE']:>8.4f} {m_rs['RMSE']:>8.4f} "
              f"{m_rs['delta_125']*100:>7.1f}%  "
              f"inliers={n_in}/{args.n_anchors}({in_ratio*100:.0f}%)")

        per_frame_results.append(frame_results)
        all_method_fits.append(frame_fits)

    # ── 7. summary table ──────────────────────────────────────────────────────
    sep("Summary — averaged across frames")
    method_keys  = ["GlobalScalar", "PerFrameScalar", "PerFrameLS", "PerFrameRANSAC"]
    summary      = {mk: {"AREs": [], "RMSEs": [], "d105s": [], "d110s": [], "d125s": []}
                    for mk in method_keys}

    for frame_results in per_frame_results:
        for r in frame_results:
            mk = r["method"]
            m  = r["metrics"]
            summary[mk]["AREs"].append(m["ARE"])
            summary[mk]["RMSEs"].append(m["RMSE"])
            summary[mk]["d105s"].append(m["delta_105"])
            summary[mk]["d110s"].append(m["delta_110"])
            summary[mk]["d125s"].append(m["delta_125"])

    summary_stats = {}
    print(f"\n  {'Method':<20} {'mean ARE':>10} {'mean RMSE':>10} "
          f"{'δ<1.05':>8} {'δ<1.10':>8} {'δ<1.25':>8}")
    print(f"  {'-'*18:<20} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for mk in method_keys:
        s = summary[mk]
        mean_ARE   = float(np.mean(s["AREs"]))
        mean_RMSE  = float(np.mean(s["RMSEs"]))
        mean_d105  = float(np.mean(s["d105s"]))
        mean_d110  = float(np.mean(s["d110s"]))
        mean_d125  = float(np.mean(s["d125s"]))
        summary_stats[mk] = {"mean_ARE": mean_ARE, "mean_RMSE": mean_RMSE,
                              "mean_d105": mean_d105, "mean_d110": mean_d110,
                              "mean_d125": mean_d125}
        print(f"  {mk:<20} {mean_ARE:>10.4f} {mean_RMSE:>10.4f} "
              f"{mean_d105*100:>7.1f}% {mean_d110*100:>7.1f}% {mean_d125*100:>7.1f}%")

    # ── 8. figures ─────────────────────────────────────────────────────────────
    sep("Saving figures")

    # Shared depth range from GT
    all_gt_valid = np.concatenate([g[g > 0] for g in gt_depths])
    vmin_m = float(np.percentile(all_gt_valid, 2))
    vmax_m = float(np.percentile(all_gt_valid, 98))

    for i, stem in enumerate(stems):
        fpath = os.path.join(OUT_DIR, f"frame_{i:02d}_{stem}_comparison.png")
        plot_per_frame_comparison(
            i, stem, pred_stack[i], gt_depths[i],
            per_frame_results[i], vmin_m, vmax_m, fpath)
        print(f"  saved frame_{i:02d}_{stem}_comparison.png")

    scatter_path = os.path.join(OUT_DIR, "anchor_scatter.png")
    plot_anchor_scatter(stems, anchors, all_method_fits, scatter_path)
    print(f"  saved anchor_scatter.png")

    bar_path = os.path.join(OUT_DIR, "summary_bar.png")
    plot_summary_bar(summary_stats, bar_path)
    print(f"  saved summary_bar.png")

    # ── 9. numeric report ─────────────────────────────────────────────────────
    sep("Writing report")
    report_lines = [
        "=" * 70,
        "  Depth Scale Fit Report — Closed-Form Methods",
        f"  Scene: {args.scene}   Frames: {len(stems)}   Anchors/frame: {args.n_anchors}",
        "=" * 70,
        "",
        "  Four methods compared (all closed-form, no gradient descent):",
        "  1. GlobalScalar  : s = median(d_gt/d_pred) across all frames pooled",
        "  2. PerFrameScalar: s_f = median(d_gt/d_pred) per frame (1 param)",
        "  3. PerFrameLS    : [a,b] = lstsq on anchors per frame (2 params)",
        "  4. PerFrameRANSAC: [a,b] via RANSAC + LS refit on inliers (2 params)",
        "",
        f"  {'Method':<20} {'mean ARE':>10} {'mean RMSE (m)':>14} "
        f"{'δ<1.05':>8} {'δ<1.10':>8} {'δ<1.25':>8}",
        f"  {'-'*20} {'-'*10} {'-'*14} {'-'*8} {'-'*8} {'-'*8}",
    ]
    for mk in method_keys:
        s = summary_stats[mk]
        report_lines.append(
            f"  {mk:<20} {s['mean_ARE']:>10.4f} {s['mean_RMSE']:>14.4f} "
            f"{s['mean_d105']*100:>7.1f}% {s['mean_d110']*100:>7.1f}% "
            f"{s['mean_d125']*100:>7.1f}%"
        )
    report_lines += [
        "",
        "  Global scalar (fixed value for reference):",
        f"    s_global = {s_global:.6f}  (d_scaled = {s_global:.4f} × d_pred)",
        "",
        "  Interpretation:",
        "  · ARE < 0.05 is excellent for indoor TSDF fusion",
        "  · δ<1.25 > 95% means 95% of pixels within 25% of GT",
        "  · The OccluSynth adapter should beat PerFrameLS/RANSAC",
        "    by predicting these params from image features.",
        "=" * 70,
    ]
    report_txt = "\n".join(report_lines)
    print(report_txt)
    with open(os.path.join(OUT_DIR, "fit_report.txt"), "w") as f:
        f.write(report_txt + "\n")
    print(f"\n  report  → {OUT_DIR}/fit_report.txt")

    sep("Output files")
    for fn in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, fn))
        print(f"  {fn:<48} {sz/1024:6.1f} KB")


if __name__ == "__main__":
    main()
