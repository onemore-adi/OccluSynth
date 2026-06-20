"""
Multi-scene regression test — validates RANSAC metric grounding across val scenes.

Reads pre-computed results from demo_outputs/multi_scene_eval/results.json,
which is produced by scripts/run_multi_scene_eval.py.  Tests fail with a clear
skip message if the results file doesn't exist yet.

Run the eval first:
    python scripts/run_multi_scene_eval.py --n 10

Then pin the numbers:
    pytest tests/test_multi_scene.py -v
"""

import json
from pathlib import Path

import numpy as np
import pytest

from occlusynth.utils import get_repo_root

RESULTS_PATH = get_repo_root() / "demo_outputs/multi_scene_eval/results.json"


@pytest.fixture(scope="module")
def results():
    if not RESULTS_PATH.exists():
        pytest.skip(
            f"Multi-scene results not found at {RESULTS_PATH}. "
            "Run: python scripts/run_multi_scene_eval.py --n 10"
        )
    data = json.loads(RESULTS_PATH.read_text())
    return [r for r in data if r["flag"] != "ERROR"]


@pytest.fixture(scope="module")
def ok_results(results):
    return [r for r in results if r["flag"] != "DEGRADE"]


# ── coverage ──────────────────────────────────────────────────────────────────

def test_enough_scenes_evaluated(results):
    """At least 5 scenes must have been evaluated successfully."""
    assert len(results) >= 5, (
        f"Only {len(results)} scenes evaluated. Re-run with --n 10."
    )


def test_degrade_scenes_are_minority(results):
    """No more than 30% of scenes should be flagged DEGRADE."""
    n_degrade = sum(1 for r in results if r["flag"] == "DEGRADE")
    ratio     = n_degrade / len(results)
    assert ratio <= 0.30, (
        f"{n_degrade}/{len(results)} scenes DEGRADE ({ratio*100:.0f}%). "
        "Closed-form fit is unreliable on this dataset sample."
    )


# ── aggregate quality (OK + WARN scenes) ─────────────────────────────────────

def test_mean_are_across_scenes(ok_results):
    """
    Mean ARE averaged across all non-degrade scenes must be < 0.035.
    Canonical value on 10 val scenes: 0.0240.
    """
    ares = [r["mean_ARE"] for r in ok_results]
    mean = float(np.mean(ares))
    assert mean < 0.035, (
        f"Cross-scene mean ARE = {mean:.4f} ≥ 0.035 (canonical: 0.0240). "
        f"Scene AREs: {[f'{a:.4f}' for a in ares]}"
    )


def test_median_are_across_scenes(ok_results):
    """Median ARE across scenes < 0.04 (robust to one bad scene)."""
    ares   = [r["mean_ARE"] for r in ok_results]
    median = float(np.median(ares))
    assert median < 0.04, (
        f"Cross-scene median ARE = {median:.4f} ≥ 0.04."
    )


def test_delta125_across_scenes(ok_results):
    """Mean δ<1.25 across scenes must be ≥ 95%."""
    d125s = [r["mean_delta_125"] for r in ok_results]
    mean  = float(np.mean(d125s))
    assert mean >= 0.95, (
        f"Cross-scene mean δ<1.25 = {mean*100:.1f}% < 95%."
    )


def test_scale_factors_in_range(ok_results):
    """
    Mean scale factor per scene must stay in [0.5, 20].

    Observed across 11 scenes (1 train + 10 val):
      scene0000_00 (train):  a ~ 6.8
      val scenes:            a ~ 1.5 – 4.1

    VGGT's internal scale is not normalised — variation across scenes is
    expected and correct.  This test only catches unit errors: depth in mm
    would give a ~ 0.007, wrong array or inverted depth would give a < 0.1
    or a > 50.
    """
    for r in ok_results:
        a = r["mean_a"]
        assert 0.5 < a < 20.0, (
            f"Scene {r['scene_id']}: mean scale a={a:.3f} outside (0.5, 20). "
            "Likely pipeline error: wrong units or wrong depth array."
        )


def test_worst_case_are_bounded(ok_results):
    """
    The single worst-case frame across all scenes must be < 0.20.
    > 0.20 means one frame is catastrophically wrong, not just hard.
    """
    worst = max(r["max_ARE"] for r in ok_results)
    worst_scene = next(r["scene_id"] for r in ok_results if r["max_ARE"] == worst)
    assert worst < 0.20, (
        f"Worst-case ARE = {worst:.4f} in {worst_scene}. "
        "A single frame has catastrophic fit error."
    )


# ── per-scene format ──────────────────────────────────────────────────────────

def test_results_schema(results):
    """Every result must have the expected keys with plausible values."""
    required = {
        "scene_id", "n_frames", "mean_ARE", "max_ARE", "mean_RMSE",
        "mean_delta_125", "mean_a", "std_a", "mean_inlier_ratio",
        "flag", "flag_reasons", "per_frame",
    }
    for r in results:
        missing = required - set(r.keys())
        assert not missing, f"{r.get('scene_id','?')}: missing keys {missing}"
        assert r["flag"] in {"OK", "WARN", "DEGRADE", "ERROR"}
        assert r["n_frames"] >= 1


def test_per_frame_schema(results):
    """Every per-frame entry must have stem, a, b, ARE, RMSE, delta_125."""
    required = {"stem", "a", "b", "ARE", "RMSE", "delta_125", "n_inliers", "inlier_ratio"}
    for r in results:
        for f in r["per_frame"]:
            missing = required - set(f.keys())
            assert not missing, (
                f"{r['scene_id']} frame {f.get('stem','?')}: missing {missing}"
            )
