"""
Anchor / sparse-pixel sampling for depth calibration.

'Anchors' are pixels at which both predicted depth and GT depth are valid.
They are the input to the four closed-form calibration fits in
occlusynth.models.depth_calibration.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def sample_anchors(
    pred_hw: np.ndarray,
    gt_hw:   np.ndarray,
    n:       int = 500,
    rng:     Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified anchor sampling from valid pixels (pred > 0 AND gt > 0).

    Strategy
    --------
    1. Divide the image into ⌊√n⌋ × ⌊√n⌋ grid cells.
    2. Pick one valid pixel per cell (uniform random within cell).
    3. Top-up from the remaining valid pixels until exactly n anchors.
    4. If fewer than n valid pixels exist, return all of them.

    Stratification avoids anchors clustering in textured foreground objects
    and ensures spatial coverage — important for the shift term in LS / RANSAC.

    Args:
        pred_hw: (H, W) float32 — predicted depth (arbitrary scale); 0 = invalid
        gt_hw:   (H, W) float32 — GT depth in metres; 0 = invalid
        n:       number of anchors to sample
        rng:     optional numpy random Generator (for reproducibility)

    Returns:
        p   (n,) float32  — predicted depth at anchor pixels
        g   (n,) float32  — GT depth (metres) at anchor pixels
        yx  (n, 2) int    — pixel coordinates [(row, col), ...]

    Raises:
        ValueError if fewer than 10 valid pixels are found.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    valid = (pred_hw > 0) & (gt_hw > 0)
    n_valid = int(valid.sum())
    if n_valid < 10:
        raise ValueError(
            f"Only {n_valid} valid pixels (pred>0 & gt>0) — need ≥ 10. "
            "Check that predictions and GT depths are on the same spatial grid."
        )

    H, W = pred_hw.shape

    # ── step 1: stratified grid sampling ──────────────────────────────────────
    n_cells  = max(1, int(np.sqrt(n)))
    cy_edges = np.linspace(0, H, n_cells + 1, dtype=int)
    cx_edges = np.linspace(0, W, n_cells + 1, dtype=int)

    strat_yx: list[tuple[int, int]] = []
    strat_flat: set[int] = set()

    for ci in range(n_cells):
        for cj in range(n_cells):
            patch = valid[cy_edges[ci]:cy_edges[ci + 1],
                          cx_edges[cj]:cx_edges[cj + 1]]
            ys_local, xs_local = np.where(patch)
            if len(ys_local) == 0:
                continue
            pick = int(rng.integers(len(ys_local)))
            r = int(ys_local[pick] + cy_edges[ci])
            c = int(xs_local[pick] + cx_edges[cj])
            strat_yx.append((r, c))
            strat_flat.add(r * W + c)

    # ── step 2: top-up with random sampling ──────────────────────────────────
    need = max(0, n - len(strat_yx))
    if need > 0:
        all_ys, all_xs = np.where(valid)
        all_flat = all_ys * W + all_xs
        topup_pool = np.array([f for f in all_flat if f not in strat_flat])
        if len(topup_pool) > 0:
            chosen = rng.choice(topup_pool,
                                min(need, len(topup_pool)),
                                replace=False)
            for f in chosen:
                strat_yx.append((int(f // W), int(f % W)))

    # ── assemble ──────────────────────────────────────────────────────────────
    yx = np.array(strat_yx[:n], dtype=np.int32)          # (n, 2)
    p  = pred_hw[yx[:, 0], yx[:, 1]].astype(np.float32)  # (n,)
    g  = gt_hw[  yx[:, 0], yx[:, 1]].astype(np.float32)  # (n,)
    return p, g, yx
