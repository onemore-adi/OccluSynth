"""
Metric grounding — closed-form RANSAC scale+shift calibration for VGGT depth.

Takes VGGT raw depth predictions and ScanNet GT depth (both in the same
(H_c, W_c) coordinate frame as produced by ScanNetDataset) and fits
per-frame linear corrections  d_metric = a * d_pred + b.

Public API
----------
fit_metric_depth   fit per-frame (a, b) from aligned pred/GT arrays
apply_metric_correction   apply (a, b) to a depth map
ground_scene       full pipeline: VGGT → fit → JSON persistence
load_grounding     read persisted (a, b) params from disk

Persistence format
------------------
out_dir/<scene_id>_grounding.json ::

    {
        "<stem>": [a, b],
        ...
    }

a  (float) scale factor   — d_metric = a * d_pred + b
b  (float) shift (metres) — typically small; captures sensor offset / air gap
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from occlusynth.data.sparse_sampler import sample_anchors
from occlusynth.models.depth_calibration import (
    fit_perframe_ransac,
    resize_to_gt,
)


# ── core fit ──────────────────────────────────────────────────────────────────

def fit_metric_depth(
    pred_frames: np.ndarray,
    gt_frames:   np.ndarray,
    stems:       Sequence[str],
    n_anchors:   int = 500,
) -> Dict[str, Tuple[float, float]]:
    """
    Fit per-frame RANSAC (a, b) for a sequence of aligned depth maps.

    VGGT predictions may be at a different spatial resolution than GT depth
    (e.g. cached at 448×592 vs GT at 382×512).  Each prediction frame is
    bilinearly resized to match its GT frame before anchor sampling.
    Bilinear is correct here because VGGT predictions are continuous — unlike
    GT depth where INTER_NEAREST is mandatory to avoid mixing valid depth with
    the 0-sentinel at occlusion boundaries.

    Args:
        pred_frames: (N, H_p, W_p) float32 — raw VGGT depth, arbitrary scale
        gt_frames:   (N, H_g, W_g) float32 — GT depth in metres; 0 = invalid
        stems:       length-N sequence of frame stem strings (e.g. '000200')
        n_anchors:   stratified anchor count per frame (default 500)

    Returns:
        dict mapping each stem → (a, b):
            a  scale factor   (d_metric = a * d_pred + b)
            b  shift in metres
    """
    N = len(stems)
    assert pred_frames.shape[0] == N and gt_frames.shape[0] == N

    result: Dict[str, Tuple[float, float]] = {}
    for i, stem in enumerate(stems):
        pred = pred_frames[i]   # (H_p, W_p)
        gt   = gt_frames[i]     # (H_g, W_g)

        # Align prediction to GT frame if resolutions differ.
        if pred.shape != gt.shape:
            pred = resize_to_gt(pred, gt)   # bilinear — safe for continuous preds

        p_a, g_a, _ = sample_anchors(pred, gt, n=n_anchors,
                                     rng=np.random.default_rng(i * 1000 + 7))
        a, b, _, _ = fit_perframe_ransac(p_a, g_a)
        result[stem] = (float(a), float(b))

    return result


# ── application ───────────────────────────────────────────────────────────────

def apply_metric_correction(
    depth: np.ndarray,
    a:     float,
    b:     float,
) -> np.ndarray:
    """
    Apply  d_metric = max(a * d_pred + b, 0)  element-wise.

    The max-0 clamp prevents negative depth from appearing when b is negative
    and the prediction is near zero.  Invalid pixels (depth == 0) stay 0.

    Args:
        depth: (H, W) float32 — raw VGGT depth
        a:     scale factor
        b:     shift in metres

    Returns:
        (H, W) float32 — metric-space depth; 0 = invalid
    """
    corrected = np.maximum(a * depth.astype(np.float64) + b, 0.0)
    # Preserve the 0-sentinel: pixels that were invalid stay invalid.
    corrected[depth == 0] = 0.0
    return corrected.astype(np.float32)


# ── persistence ───────────────────────────────────────────────────────────────

def _json_path(out_dir: Path, scene_id: str) -> Path:
    return out_dir / f"{scene_id}_grounding.json"


def save_grounding(
    params:   Dict[str, Tuple[float, float]],
    out_dir:  Path | str,
    scene_id: str,
) -> Path:
    """Persist {stem: [a, b]} to <out_dir>/<scene_id>_grounding.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _json_path(out_dir, scene_id)
    payload = {stem: list(ab) for stem, ab in params.items()}
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_grounding(json_path: Path | str) -> Dict[str, Tuple[float, float]]:
    """Load persisted grounding params.  Returns {stem: (a, b)}."""
    data = json.loads(Path(json_path).read_text())
    return {stem: (float(ab[0]), float(ab[1])) for stem, ab in data.items()}


# ── per-scene evaluation ─────────────────────────────────────────────────────

def eval_scene(
    scene_id:  str,
    dataset:   "ScanNetDataset",
    wrapper:   "VGGTWrapper",
    cache_dir: "Path | None" = None,
    n_anchors: int = 500,
) -> dict:
    """
    Fit RANSAC (a, b) for every frame in one scene and return per-scene stats.

    Returns a dict with:
        scene_id        str
        n_frames        int
        per_frame       list of {stem, a, b, ARE, RMSE, delta_125,
                                 n_inliers, inlier_ratio}
        mean_ARE        float   — primary quality metric
        max_ARE         float   — worst-case frame (catches pathological frames)
        mean_RMSE       float
        mean_delta_125  float
        mean_a          float   — average scale factor
        std_a           float   — scale instability across frames
        mean_inlier_ratio float
        flag            str     — "OK" | "WARN" | "DEGRADE"
        flag_reasons    list[str]
    """
    from occlusynth.utils.device import get_config, get_repo_root
    from occlusynth.models.depth_calibration import (
        fit_perframe_ransac, evaluate, resize_to_gt,
    )
    from occlusynth.data.sparse_sampler import sample_anchors

    if cache_dir is None:
        cfg = get_config()
        cache_dir = get_repo_root() / cfg["cache_dir"]

    idx       = dataset.scenes.index(scene_id)
    item      = dataset[idx]
    stems     = item["frame_idx"]
    gt_np     = item["depth_gt"].numpy()

    scene_dir   = dataset.root / scene_id
    vggt_result = wrapper.predict_cached(
        scene_id, stems, scene_dir, cache_dir=cache_dir
    )
    pred_np = vggt_result["depth"]

    per_frame = []
    for i, stem in enumerate(stems):
        pred = pred_np[i]
        gt   = gt_np[i]
        if pred.shape != gt.shape:
            pred = resize_to_gt(pred, gt)

        p_a, g_a, _ = sample_anchors(pred, gt, n=n_anchors,
                                     rng=np.random.default_rng(i * 1000 + 7))
        a, b, n_in, in_ratio = fit_perframe_ransac(p_a, g_a)
        m = evaluate(pred, gt, a, b)
        per_frame.append({
            "stem":         stem,
            "a":            a,
            "b":            b,
            "ARE":          m["ARE"],
            "RMSE":         m["RMSE"],
            "delta_125":    m["delta_125"],
            "n_inliers":    n_in,
            "inlier_ratio": in_ratio,
        })

    ares          = [f["ARE"]          for f in per_frame]
    rmses         = [f["RMSE"]         for f in per_frame]
    d125s         = [f["delta_125"]    for f in per_frame]
    scales        = [f["a"]            for f in per_frame]
    inlier_ratios = [f["inlier_ratio"] for f in per_frame]

    mean_ARE   = float(np.mean(ares))
    max_ARE    = float(np.max(ares))
    mean_a     = float(np.mean(scales))
    std_a      = float(np.std(scales))
    mean_ir    = float(np.mean(inlier_ratios))

    # Flag pathological scenes.
    # DEGRADE: fit is unreliable (mean error > 10%).
    # WARN:    some frames are hard but aggregate is usable.
    reasons  = []
    degrade  = False
    if mean_ARE > 0.10:
        reasons.append(f"mean_ARE={mean_ARE:.3f} > 0.10")
        degrade = True
    if max_ARE > 0.15:
        reasons.append(f"max_ARE={max_ARE:.3f} > 0.15")
    if mean_ir < 0.70:
        reasons.append(f"mean_inlier_ratio={mean_ir:.2f} < 0.70")
    if std_a / max(mean_a, 1e-6) > 0.20:
        reasons.append(f"scale_cv={std_a/mean_a:.2f} > 0.20 (unstable scale)")

    flag = "DEGRADE" if degrade else ("WARN" if reasons else "OK")

    return {
        "scene_id":          scene_id,
        "n_frames":          len(stems),
        "per_frame":         per_frame,
        "mean_ARE":          mean_ARE,
        "max_ARE":           max_ARE,
        "mean_RMSE":         float(np.mean(rmses)),
        "mean_delta_125":    float(np.mean(d125s)),
        "mean_a":            mean_a,
        "std_a":             std_a,
        "mean_inlier_ratio": mean_ir,
        "flag":              flag,
        "flag_reasons":      reasons,
    }


# ── full pipeline ─────────────────────────────────────────────────────────────

def ground_scene(
    scene_id:  str,
    dataset:   "ScanNetDataset",  # noqa: F821 — avoid circular import at module level
    wrapper:   "VGGTWrapper",     # noqa: F821
    out_dir:   Path | str,
    cache_dir: Path | str | None = None,
    n_anchors: int = 500,
) -> Path:
    """
    Run metric grounding for one scene and persist (a, b) to disk.

    1. Looks up the scene in ``dataset`` and retrieves GT depth at the
       colour-camera resolution (H_c, W_c) — the same frame as ``rgb`` and
       ``intrinsics`` in the dataset item.
    2. Runs VGGT prediction (or loads from cache).
    3. Fits per-frame (a, b) via RANSAC on stratified anchor pixels.
    4. Writes ``<out_dir>/<scene_id>_grounding.json`` and returns the path.

    Args:
        scene_id:  ScanNet scene ID, e.g. 'scene0000_00'
        dataset:   ScanNetDataset instance whose root contains scene_id
        wrapper:   VGGTWrapper for inference / cache lookup
        out_dir:   directory for the output JSON
        cache_dir: VGGT prediction cache directory (default: from get_config())
        n_anchors: stratified anchors per frame (default 500)

    Returns:
        Path to the written JSON file.
    """
    from occlusynth.utils.device import get_config, get_repo_root

    if cache_dir is None:
        cfg = get_config()
        cache_dir = get_repo_root() / cfg["cache_dir"]

    # ── dataset item → GT depth as numpy ──────────────────────────────────────
    try:
        scene_idx = dataset.scenes.index(scene_id)
    except ValueError:
        raise ValueError(
            f"'{scene_id}' not found in dataset (split='{dataset.scenes[:3]}...'). "
            "Use split='all' or ensure the scene is in the chosen split."
        )

    item  = dataset[scene_idx]
    stems = item["frame_idx"]
    gt_np = item["depth_gt"].numpy()        # (N, H_c, W_c) float32

    # ── VGGT predictions ──────────────────────────────────────────────────────
    scene_dir  = dataset.root / scene_id
    vggt_result = wrapper.predict_cached(
        scene_id, stems, scene_dir, cache_dir=Path(cache_dir)
    )
    pred_np = vggt_result["depth"]          # (N, H_p, W_p) float32

    # ── fit ───────────────────────────────────────────────────────────────────
    params = fit_metric_depth(pred_np, gt_np, stems, n_anchors=n_anchors)

    # ── persist ───────────────────────────────────────────────────────────────
    return save_grounding(params, out_dir, scene_id)
