"""
tests/test_sevenscenes.py — 7-Scenes loader correctness.

All tests that require data on disk are skipped when the data is absent;
the shape / constant tests run unconditionally.

Test categories:
  (a) K constant has correct shape and expected focal length
  (b) When data present: depth/pose shapes, depth range, pick_frames count
  (c) When data present: ARE is finite after RANSAC grounding (mm→m)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from occlusynth.data.sevenscenes import (
    IMG_H, IMG_W,
    K_7SCENES,
    DEPTH_MAX_M,
    SevenScenesSequence,
    load_depth,
    load_depth_raw,
    load_pose,
)

# Default probe sequence — skipped if not present
_SEQ_DIR = ROOT / "data" / "7scenes" / "chess" / "seq-01"
_HAS_DATA = _SEQ_DIR.exists() and any(_SEQ_DIR.glob("frame-*.depth.png"))


# ---------------------------------------------------------------------------
# (a) Constants — always run
# ---------------------------------------------------------------------------

class TestConstants:
    def test_k_shape(self):
        assert K_7SCENES.shape == (3, 3)

    def test_k_focal_length(self):
        assert abs(K_7SCENES[0, 0] - 585.0) < 1e-3, "fx expected 585"
        assert abs(K_7SCENES[1, 1] - 585.0) < 1e-3, "fy expected 585"

    def test_k_principal_point(self):
        assert abs(K_7SCENES[0, 2] - 320.0) < 1e-3, "cx expected 320"
        assert abs(K_7SCENES[1, 2] - 240.0) < 1e-3, "cy expected 240"

    def test_k_bottom_row(self):
        np.testing.assert_array_equal(K_7SCENES[2], [0, 0, 1])

    def test_image_dims(self):
        assert IMG_H == 480 and IMG_W == 640


# ---------------------------------------------------------------------------
# (b) Loader shapes and value range — require data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_DATA, reason="7-Scenes chess/seq-01 not downloaded")
class TestLoaderShapes:
    @pytest.fixture(scope="class")
    def seq(self):
        return SevenScenesSequence(_SEQ_DIR)

    @pytest.fixture(scope="class")
    def stem(self, seq):
        return seq.pick_frames(1)[0]

    def test_depth_shape(self, seq, stem):
        d = seq.depth(stem)
        assert d.shape == (IMG_H, IMG_W), f"expected ({IMG_H}, {IMG_W}), got {d.shape}"
        assert d.dtype == np.float32

    def test_depth_raw_shape(self, seq, stem):
        d = seq.depth_raw(stem)
        assert d.shape == (IMG_H, IMG_W)

    def test_depth_range(self, seq, stem):
        d = seq.depth(stem)
        assert d.min() >= 0.0, "depth must be non-negative"
        assert d.max() <= DEPTH_MAX_M + 1e-6, f"depth must be clipped at {DEPTH_MAX_M} m"

    def test_depth_raw_in_mm(self, seq, stem):
        """Raw depth is in mm; valid pixels should be in the ~500–3500 mm range."""
        raw = seq.depth_raw(stem)
        valid = raw[raw > 0]
        if len(valid) > 10:
            assert valid.min() > 100, "raw mm depth should be >> 1 for indoor scenes"
            assert valid.max() < 65535, "65535 is an invalid sentinel, not a real measurement"

    def test_pose_shape(self, seq, stem):
        p = seq.pose(stem)
        assert p.shape == (4, 4), f"expected (4, 4), got {p.shape}"
        assert p.dtype == np.float64

    def test_pose_last_row(self, seq, stem):
        p = seq.pose(stem)
        np.testing.assert_allclose(p[3], [0, 0, 0, 1], atol=1e-6,
                                   err_msg="last row of pose must be [0,0,0,1]")

    def test_pose_valid_rotation(self, seq, stem):
        R = seq.pose(stem)[:3, :3]
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 0.01, f"rotation determinant should be 1, got {det:.4f}"

    def test_k_property(self, seq):
        K = seq.K
        assert K.shape == (3, 3)
        np.testing.assert_array_equal(K, K_7SCENES)

    def test_pick_frames_count(self, seq):
        n = 6
        stems = seq.pick_frames(n)
        assert len(stems) == n, f"expected {n} stems, got {len(stems)}"

    def test_pick_frames_unique(self, seq):
        stems = seq.pick_frames(6)
        assert len(set(stems)) == len(stems), "pick_frames returned duplicate stems"

    def test_color_path_exists(self, seq, stem):
        assert seq.color_path(stem).exists()

    def test_all_stems_nonempty(self, seq):
        assert len(seq.all_stems()) > 0

    def test_repr(self, seq):
        r = repr(seq)
        assert "chess" in r and "seq-01" in r


# ---------------------------------------------------------------------------
# (c) RANSAC grounding: ARE finite after calibration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_DATA, reason="7-Scenes chess/seq-01 not downloaded")
class TestRansacGrounding:
    def test_are_finite(self):
        """RANSAC calibration of mm→m must yield finite ARE for 7-Scenes depth."""
        from occlusynth.models.depth_calibration import evaluate
        from occlusynth.models.metric_grounding import fit_metric_depth

        seq   = SevenScenesSequence(_SEQ_DIR)
        stems = seq.pick_frames(4)

        raw_frames = np.stack([seq.depth_raw(s) for s in stems])   # mm
        gt_frames  = np.stack([seq.depth(s)     for s in stems])   # metres

        params = fit_metric_depth(raw_frames, gt_frames, stems)

        for s in stems:
            a, b = params[s]
            assert np.isfinite(a), f"scale factor 'a' is not finite for {s}"
            assert np.isfinite(b), f"shift 'b' is not finite for {s}"
            # a should be close to 1/1000 = 0.001
            assert 0.0005 < a < 0.002, (
                f"expected a ≈ 0.001 (mm→m), got {a:.6f} for stem {s}")

            m = evaluate(raw_frames[stems.index(s)], gt_frames[stems.index(s)], a, b)
            assert np.isfinite(m["ARE"]), f"ARE is not finite for stem {s}"
            assert m["ARE"] < 0.01, (
                f"ARE after mm→m calibration should be near 0, got {m['ARE']:.4f}")
            assert m["delta_125"] > 0.95, (
                f"δ<1.25 should be near 1.0 after calibration, got {m['delta_125']:.4f}")
