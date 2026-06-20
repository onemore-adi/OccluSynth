"""
Stage 0 regression test — pins the RANSAC metric grounding numbers on scene0000_00.

All measurements are taken at the unified 382×512 colour-camera resolution
(ScanNetDataset default).  If this test breaks after a data-pipeline change,
the resolution or coordinate frame shifted — check ScanNetDataset first.

Run:  pytest tests/test_metric_grounding.py -v
"""

import json
import tempfile

import numpy as np
import pytest
import torch

from occlusynth.data import ScanNetDataset
from occlusynth.models import (
    fit_metric_depth,
    apply_metric_correction,
    save_grounding,
    load_grounding,
)
from occlusynth.models.depth_calibration import evaluate, resize_to_gt
from occlusynth.models.vggt_wrapper import VGGTWrapper
from occlusynth.utils import get_config, get_repo_root


# ── shared fixtures ───────────────────────────────────────────────────────────

SCENE_ID  = "scene0000_00"
N_FRAMES  = 6
N_ANCHORS = 500


@pytest.fixture(scope="module")
def grounding_data():
    """
    Load GT depth (382×512) and VGGT cached predictions for SCENE_ID.
    Returns (pred_np, gt_np, stems, params) where params = {stem: (a, b)}.
    """
    root = get_repo_root()
    cfg  = get_config()

    ds   = ScanNetDataset(split="all", n_frames=N_FRAMES)
    idx  = ds.scenes.index(SCENE_ID)
    item = ds[idx]
    stems = item["frame_idx"]
    gt_np = item["depth_gt"].numpy()           # (N, 382, 512)

    wrapper    = VGGTWrapper()
    scene_dir  = ds.root / SCENE_ID
    result     = wrapper.predict_cached(
        SCENE_ID, stems, scene_dir, cache_dir=root / cfg["cache_dir"]
    )
    pred_np = result["depth"]                  # (N, 448, 592) from cache

    params = fit_metric_depth(pred_np, gt_np, stems, n_anchors=N_ANCHORS)
    return pred_np, gt_np, stems, params


# ── resolution guard ──────────────────────────────────────────────────────────

def test_gt_resolution():
    """GT depth must be at 382×512 — the colour-camera frame after resize."""
    ds   = ScanNetDataset(split="all", n_frames=1)
    idx  = ds.scenes.index(SCENE_ID)
    item = ds[idx]
    H, W = item["depth_gt"].shape[-2:]
    assert (H, W) == (382, 512), (
        f"Expected (382, 512), got ({H}, {W}). "
        "ScanNetDataset target_size or colour image resolution changed."
    )


# ── fit quality ───────────────────────────────────────────────────────────────

def test_ransac_mean_are(grounding_data):
    """
    RANSAC mean ARE across 6 frames must be < 0.03.
    Canonical value at 382×512: 0.0242.
    """
    pred_np, gt_np, stems, params = grounding_data

    ares = []
    for i, stem in enumerate(stems):
        pred = pred_np[i]
        gt   = gt_np[i]
        if pred.shape != gt.shape:
            pred = resize_to_gt(pred, gt)
        a, b = params[stem]
        m = evaluate(pred, gt, a, b)
        ares.append(m["ARE"])

    mean_are = float(np.mean(ares))
    assert mean_are < 0.03, (
        f"RANSAC mean ARE = {mean_are:.4f} ≥ 0.03. "
        "Regression: data pipeline or fit code changed."
    )


def test_ransac_mean_are_tighter(grounding_data):
    """
    Tighter bound: mean ARE < 0.026 (gives room above the 0.0242 canonical).
    Fails only if the implementation regresses meaningfully, not on noise.
    """
    pred_np, gt_np, stems, params = grounding_data

    ares = []
    for i, stem in enumerate(stems):
        pred = pred_np[i]
        gt   = gt_np[i]
        if pred.shape != gt.shape:
            pred = resize_to_gt(pred, gt)
        a, b = params[stem]
        ares.append(evaluate(pred, gt, a, b)["ARE"])

    mean_are = float(np.mean(ares))
    assert mean_are < 0.026, (
        f"RANSAC mean ARE = {mean_are:.4f} ≥ 0.026 (canonical: 0.0242). "
        "Either the fit regressed or the data path changed resolution."
    )


def test_delta125(grounding_data):
    """δ<1.25 must be ≥ 99.5 % across all frames (canonical: 99.7 %)."""
    pred_np, gt_np, stems, params = grounding_data

    d125s = []
    for i, stem in enumerate(stems):
        pred = pred_np[i]
        gt   = gt_np[i]
        if pred.shape != gt.shape:
            pred = resize_to_gt(pred, gt)
        a, b = params[stem]
        d125s.append(evaluate(pred, gt, a, b)["delta_125"])

    mean_d125 = float(np.mean(d125s))
    assert mean_d125 >= 0.995, (
        f"Mean δ<1.25 = {mean_d125*100:.1f}% < 99.5% (canonical: 99.7%). Regression."
    )


def test_scale_factors_plausible(grounding_data):
    """
    Per-frame scale factors must be in a sane range.
    VGGT internal units → metres is roughly 6–9× for indoor scenes.
    A wildly different value means the pipeline is consuming the wrong array.
    """
    _, _, _, params = grounding_data
    for stem, (a, b) in params.items():
        assert 4.0 < a < 15.0, f"Frame {stem}: scale a={a:.3f} outside [4, 15]"
        assert -1.0 < b < 1.5,  f"Frame {stem}: shift b={b:.3f} outside [-1, 1.5]"


# ── apply_metric_correction ───────────────────────────────────────────────────

def test_apply_correction_no_negative_depth(grounding_data):
    pred_np, gt_np, stems, params = grounding_data
    for i, stem in enumerate(stems):
        pred = pred_np[i]
        if pred.shape != gt_np[i].shape:
            pred = resize_to_gt(pred, gt_np[i])
        a, b = params[stem]
        out = apply_metric_correction(pred, a, b)
        assert out.min() >= 0.0, f"Frame {stem}: negative depth after correction"


def test_apply_correction_preserves_invalid(grounding_data):
    """Pixels that were 0 (invalid) in the raw prediction stay 0."""
    pred_np, _, stems, params = grounding_data
    pred = pred_np[0].copy()
    a, b = params[stems[0]]
    out = apply_metric_correction(pred, a, b)
    was_zero = pred == 0.0
    assert (out[was_zero] == 0.0).all(), "apply_metric_correction corrupted 0-sentinels"


# ── JSON persistence ──────────────────────────────────────────────────────────

def test_grounding_json_roundtrip(grounding_data):
    """save_grounding / load_grounding must be a lossless round-trip."""
    _, _, _, params = grounding_data
    with tempfile.TemporaryDirectory() as tmp:
        path   = save_grounding(params, tmp, SCENE_ID)
        loaded = load_grounding(path)

    assert set(loaded.keys()) == set(params.keys())
    for stem in params:
        a_orig, b_orig = params[stem]
        a_load, b_load = loaded[stem]
        assert abs(a_orig - a_load) < 1e-9, f"a mismatch for {stem}"
        assert abs(b_orig - b_load) < 1e-9, f"b mismatch for {stem}"


def test_grounding_json_schema(grounding_data):
    """JSON file must be {stem: [a, b]} — readable by run_fusion.py."""
    _, _, _, params = grounding_data
    with tempfile.TemporaryDirectory() as tmp:
        path = save_grounding(params, tmp, SCENE_ID)
        raw  = json.loads(path.read_text())

    for stem, ab in raw.items():
        assert isinstance(stem, str)
        assert isinstance(ab, list) and len(ab) == 2
        assert all(isinstance(v, float) for v in ab)
