"""
Closed-form depth scale calibration — four methods.

All functions are pure numpy, no torch dependency.
They consume 'anchor' pixel pairs (d_pred, d_gt) produced by
occlusynth.data.sparse_sampler.sample_anchors().

Methods
-------
1. fit_global_scalar   — one scale across the whole dataset
2. fit_perframe_scalar — per-frame median ratio  (1 param)
3. fit_perframe_ls     — per-frame scale + shift, normal equations  (2 params)
4. fit_perframe_ransac — per-frame RANSAC + LS refit on inliers  (2 params)

Evaluation
----------
evaluate(pred_full, gt_full, a, b=0.0) → dict of ARE, RMSE, δ<1.05/1.10/1.25

All fits model:    d_metric = a * d_pred + b
Methods 1 & 2 fix b = 0.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PIL import Image

# ── defaults (match the baseline experiment) ──────────────────────────────────
RANSAC_ITERS: int   = 500
RANSAC_THR:   float = 0.10   # relative error threshold for inlier test
RANSAC_SEED:  int   = 42


# ── image utility ─────────────────────────────────────────────────────────────

def resize_to_gt(pred_hw: np.ndarray, gt_hw: np.ndarray) -> np.ndarray:
    """
    Bilinear-resize ``pred_hw`` to the spatial resolution of ``gt_hw``.

    Used to align VGGT predictions (e.g. 448×592) to ScanNet GT depth (480×640)
    before computing anchor values and evaluation metrics.

    Args:
        pred_hw: (H_p, W_p) float32
        gt_hw:   (H_g, W_g) float32  — only shape is used

    Returns:
        (H_g, W_g) float32
    """
    H_g, W_g = gt_hw.shape
    img = Image.fromarray(pred_hw.astype(np.float32))
    img = img.resize((W_g, H_g), Image.BILINEAR)
    return np.array(img, dtype=np.float32)


# ── fit functions ────────────────────────────────────────────────────────────

def fit_global_scalar(
    all_pred: np.ndarray,
    all_gt:   np.ndarray,
) -> float:
    """
    Method 1 — Global scalar (dataset-wide).

    Pools all anchor pairs from all frames and fits a single scale factor::

        s = median(d_gt / d_pred)   over all N_frames × N_anchors pairs

    The median is equivalent to minimising the L1 loss on log-depth ratios.
    It is robust to outlier frames but cannot capture per-frame scale drift.

    Args:
        all_pred: (N_total,) float32 — predicted depth at all anchors
        all_gt:   (N_total,) float32 — GT depth (metres) at all anchors

    Returns:
        s (float) — apply as  d_metric = s * d_pred
    """
    ratios = all_gt / np.maximum(all_pred, 1e-9)
    return float(np.median(ratios))


def fit_perframe_scalar(
    pred_anchors: np.ndarray,
    gt_anchors:   np.ndarray,
) -> float:
    """
    Method 2 — Per-frame scalar via median ratio.

    One scale per frame; ignores scale drift between frames::

        s_f = median(d_gt_f / d_pred_f)

    One degree of freedom.  Fast and robust.  Correct when depth error is
    purely multiplicative (no additive air-gap bias).

    Args:
        pred_anchors: (N,) float32
        gt_anchors:   (N,) float32

    Returns:
        s (float) — apply as  d_metric = s * d_pred
    """
    ratios = gt_anchors / np.maximum(pred_anchors, 1e-9)
    return float(np.median(ratios))


def fit_perframe_ls(
    pred_anchors: np.ndarray,
    gt_anchors:   np.ndarray,
) -> Tuple[float, float, float]:
    """
    Method 3 — Per-frame scale + shift via least-squares (normal equations).

    Models  d_gt ≈ a * d_pred + b  and solves::

        [a, b]ᵀ = (AᵀA)⁻¹ Aᵀ g
        A = [[p₀, 1], [p₁, 1], …]    (N × 2)
        g = [g₀, g₁, …]ᵀ             (N,)

    Two degrees of freedom — captures a constant additive bias in addition
    to the multiplicative scale error.  Common sources of additive bias:
    camera–wall air gap at scene boundaries, lens distortion, sensor offset.

    Args:
        pred_anchors: (N,) float32
        gt_anchors:   (N,) float32

    Returns:
        a            (float) — scale factor
        b            (float) — additive shift in metres
        residual_std (float) — std of fit residuals (quality indicator)
    """
    n   = len(pred_anchors)
    A   = np.stack([pred_anchors, np.ones(n, dtype=np.float32)], axis=1).astype(np.float64)
    g   = gt_anchors.astype(np.float64)

    # Normal equations
    ATA   = A.T @ A          # (2, 2)
    ATg   = A.T @ g          # (2,)
    x     = np.linalg.solve(ATA, ATg)    # [a, b]
    resid = g - A @ x

    return float(x[0]), float(x[1]), float(resid.std())


def fit_perframe_ransac(
    pred_anchors: np.ndarray,
    gt_anchors:   np.ndarray,
    n_iter:       int   = RANSAC_ITERS,
    thr:          float = RANSAC_THR,
    seed:         int   = RANSAC_SEED,
) -> Tuple[float, float, int, float]:
    """
    Method 4 — Per-frame RANSAC scale + shift, robust to gross outliers.

    Same linear model as LS (d_gt = a * d_pred + b), but resistant to
    sensor dropouts, reflective / transparent surfaces that produce
    spurious depth readings.

    Algorithm::

        for n_iter iterations:
            draw 2 anchor pairs  (minimal sample for 2-param model)
            solve [a, b] exactly from those 2 points
            count inliers: |a*p + b - g| / g < thr
        refit [a, b] via LS on the best inlier set  (variance reduction)

    The inlier threshold ``thr`` is *relative* (fraction of GT depth), so
    it self-scales across scenes of different depths — no per-scene tuning.

    Args:
        pred_anchors: (N,) float32
        gt_anchors:   (N,) float32
        n_iter:       number of RANSAC trials (default 500)
        thr:          relative error threshold for inlier test (default 0.10 = 10%)
        seed:         RNG seed for reproducibility

    Returns:
        a            (float) — scale factor
        b            (float) — additive shift in metres
        n_inliers    (int)   — consensus set size
        inlier_ratio (float) — n_inliers / N
    """
    rng = np.random.default_rng(seed)
    n   = len(pred_anchors)
    p   = pred_anchors.astype(np.float64)
    g   = gt_anchors.astype(np.float64)

    best_inliers = np.zeros(n, dtype=bool)
    best_count   = 0

    for _ in range(n_iter):
        idx = rng.choice(n, 2, replace=False)
        p_s, g_s = p[idx], g[idx]
        dp = p_s[1] - p_s[0]
        if abs(dp) < 1e-9:          # degenerate: identical predicted depths
            continue

        a_c = (g_s[1] - g_s[0]) / dp
        b_c = g_s[0] - a_c * p_s[0]

        pred_scaled = a_c * p + b_c
        rel_err     = np.abs(pred_scaled - g) / np.maximum(g, 1e-9)
        inliers     = rel_err < thr
        count       = int(inliers.sum())

        if count > best_count:
            best_count   = count
            best_inliers = inliers

    # Refit on inlier set with LS for lower variance
    if best_count >= 2:
        a, b, _ = fit_perframe_ls(
            p[best_inliers].astype(np.float32),
            g[best_inliers].astype(np.float32),
        )
    else:
        a = float(np.median(g / np.maximum(p, 1e-9)))
        b = 0.0

    return a, b, best_count, best_count / n


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    pred_full: np.ndarray,
    gt_full:   np.ndarray,
    a:         float,
    b:         float = 0.0,
) -> Dict[str, float]:
    """
    Apply  d_metric = a * d_pred + b  and compute depth accuracy metrics
    at all valid pixels (gt > 0 AND pred > 0).

    Args:
        pred_full: (H, W) float32 — raw predicted depth
        gt_full:   (H, W) float32 — GT depth in metres; 0 = invalid
        a:         scale factor
        b:         additive shift in metres (0 for scalar-only methods)

    Returns dict:
        ARE       mean |d_scaled - d_gt| / d_gt          ↓ lower is better
        RMSE      √mean(d_scaled - d_gt)²  (metres)      ↓ lower is better
        delta_105 fraction where max(p/g, g/p) < 1.05    ↑ higher is better
        delta_110 fraction where max(p/g, g/p) < 1.10
        delta_125 fraction where max(p/g, g/p) < 1.25
        n_valid   number of valid pixels evaluated
    """
    valid = (gt_full > 0) & (pred_full > 0)
    if not valid.any():
        nan = float("nan")
        return dict(ARE=nan, RMSE=nan, delta_105=nan,
                    delta_110=nan, delta_125=nan, n_valid=0)

    p      = pred_full[valid].astype(np.float64)
    g      = gt_full[valid].astype(np.float64)
    scaled = np.maximum(a * p + b, 1e-6)   # clamp to avoid /0

    # Python 3.14 uses LOAD_FAST_BORROW which does NOT increment the refcount
    # of local variables when loading them onto the evaluation stack.  NumPy's
    # ufuncs check refcount == 1 as a signal that the input buffer is safe to
    # reuse as the output ("buffer stealing").  Any in-function arithmetic on
    # `scaled` (e.g.  scaled - g, scaled / g) would therefore corrupt the
    # `scaled` array in-place, producing wrong RMSE and delta values while
    # leaving ARE correct (since ARE is computed before the first use of scaled).
    # Fix: mark scaled as non-writable so numpy always allocates a fresh output.
    scaled.flags.writeable = False

    diff    = scaled - g          # safe: numpy can't reuse scaled (non-writable)
    rel_err = np.abs(diff) / g
    ratio   = np.maximum(scaled / g, g / scaled)

    return {
        "ARE":       float(np.mean(rel_err)),
        "RMSE":      float(np.sqrt(np.mean(diff ** 2))),
        "delta_105": float((ratio < 1.05).mean()),
        "delta_110": float((ratio < 1.10).mean()),
        "delta_125": float((ratio < 1.25).mean()),
        "n_valid":   int(valid.sum()),
    }
