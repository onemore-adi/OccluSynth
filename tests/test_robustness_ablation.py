"""
Robustness ablation regression tests.

Reads demo_outputs/robustness/ablation_summary.json produced by
scripts/run_robustness_ablation.py.  Skips with an actionable message
if the file does not exist.

Run experiments first:
    python scripts/run_robustness_ablation.py

Then pin numbers:
    pytest tests/test_robustness_ablation.py -v
"""

import json
from pathlib import Path

import numpy as np
import pytest

from occlusynth.utils import get_repo_root

RESULTS_PATH = get_repo_root() / "demo_outputs/robustness/ablation_summary.json"


@pytest.fixture(scope="module")
def ablation():
    if not RESULTS_PATH.exists():
        pytest.skip(
            f"Ablation results not found at {RESULTS_PATH}. "
            "Run: python scripts/run_robustness_ablation.py"
        )
    return json.loads(RESULTS_PATH.read_text())


# ── noise sensitivity ─────────────────────────────────────────────────────────

def test_noise_baseline_are(ablation):
    """At σ=0 (no noise), RANSAC mean ARE must be < 0.035."""
    idx = ablation["noise"]["sigmas"].index(0.0)
    are = ablation["noise"]["RANSAC"]["mean"][idx]
    assert are < 0.035, f"Baseline (σ=0) RANSAC ARE = {are:.4f} ≥ 0.035"


def test_ransac_beats_ls_on_clean_data(ablation):
    """At σ=0, RANSAC ARE must be strictly lower than LS ARE."""
    idx   = ablation["noise"]["sigmas"].index(0.0)
    ls    = ablation["noise"]["LS"]["mean"][idx]
    ransac = ablation["noise"]["RANSAC"]["mean"][idx]
    assert ransac < ls, (
        f"RANSAC ({ransac:.4f}) ≥ LS ({ls:.4f}) on clean data. "
        "RANSAC should beat LS when anchors are noise-free."
    )


def test_ls_stable_under_gaussian_noise(ablation):
    """
    LS ARE at σ=0.25 m must be < 1.5× its σ=0 baseline.
    LS averages zero-mean noise, so its degradation should be modest.
    """
    sigmas = ablation["noise"]["sigmas"]
    ls_means = ablation["noise"]["LS"]["mean"]
    base  = ls_means[sigmas.index(0.0)]
    worst = ls_means[sigmas.index(0.25)]
    assert worst < base * 1.5, (
        f"LS ARE at σ=0.25 m ({worst:.4f}) > 1.5× baseline ({base:.4f}). "
        "LS is expected to be stable under homogeneous Gaussian noise."
    )


def test_ransac_degrades_at_large_gaussian_noise(ablation):
    """
    At σ=0.25 m, RANSAC ARE must exceed LS ARE.
    This is the key finding: RANSAC's inlier model assumes structural outliers,
    not homogeneous Gaussian noise.  At σ=0.25 m (>10% of ~2 m mean depth),
    RANSAC's inlier test misidentifies noisy anchors, degrading the fit.
    In practice ScanNet noise is structural — RANSAC remains the right choice.
    """
    sigmas = ablation["noise"]["sigmas"]
    i = sigmas.index(0.25)
    ls_are  = ablation["noise"]["LS"]["mean"][i]
    rs_are  = ablation["noise"]["RANSAC"]["mean"][i]
    assert rs_are > ls_are, (
        f"At σ=0.25 m: RANSAC ARE ({rs_are:.4f}) ≤ LS ARE ({ls_are:.4f}). "
        "Expected RANSAC to degrade under large homogeneous Gaussian noise."
    )


# ── anchor count sensitivity ──────────────────────────────────────────────────

def test_100_anchors_sufficient(ablation):
    """
    ARE at N=100 anchors must be within 0.003 of N=500 for both methods.
    Answers 'how few anchors do we need?' — 100 is safe.
    """
    counts = ablation["anchor_count"]["counts"]
    i500   = counts.index(500)
    i100   = counts.index(100)
    for method in ("LS", "RANSAC"):
        base  = ablation["anchor_count"][method]["mean"][i500]
        at100 = ablation["anchor_count"][method]["mean"][i100]
        delta = abs(at100 - base)
        assert delta < 0.003, (
            f"{method}: ARE at N=100 ({at100:.4f}) differs from N=500 "
            f"({base:.4f}) by {delta:.4f} > 0.003."
        )


def test_50_anchors_still_reasonable(ablation):
    """ARE at N=50 must be < 0.005 above N=500 for both methods."""
    counts = ablation["anchor_count"]["counts"]
    i500   = counts.index(500)
    i50    = counts.index(50)
    for method in ("LS", "RANSAC"):
        base = ablation["anchor_count"][method]["mean"][i500]
        at50 = ablation["anchor_count"][method]["mean"][i50]
        delta = abs(at50 - base)
        assert delta < 0.005, (
            f"{method}: ARE at N=50 ({at50:.4f}) differs from N=500 "
            f"({base:.4f}) by {delta:.4f} > 0.005."
        )


def test_ransac_better_than_ls_at_all_anchor_counts(ablation):
    """
    On clean data (σ=0), RANSAC must beat LS at every anchor count tested.
    Checks the method ranking is consistent regardless of N.
    """
    counts = ablation["anchor_count"]["counts"]
    for i, n in enumerate(counts):
        ls  = ablation["anchor_count"]["LS"]["mean"][i]
        rs  = ablation["anchor_count"]["RANSAC"]["mean"][i]
        assert rs < ls, (
            f"N={n}: RANSAC ARE ({rs:.4f}) ≥ LS ARE ({ls:.4f}). "
            "RANSAC should beat LS at every anchor count on clean data."
        )


# ── schema ────────────────────────────────────────────────────────────────────

def test_schema(ablation):
    assert "n_frames" in ablation and ablation["n_frames"] >= 60
    assert "noise" in ablation and "anchor_count" in ablation
    assert len(ablation["noise"]["sigmas"]) == 5
    assert 0.25 in ablation["noise"]["sigmas"]
    assert 100 in ablation["anchor_count"]["counts"]
    assert 50  in ablation["anchor_count"]["counts"]
