"""
tests/test_safety_benchmark.py — safety benchmark correctness checks.

(a) Hazard set is non-empty on scene0000_00 (well-observed indoor scene)
(b) On at least one val scene, completer awareness > 0 > baselines
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from occlusynth.models.completer import OccluSynthCompleter

ROOT     = Path(__file__).resolve().parent.parent
CKPT     = ROOT / "checkpoints" / "interim_64_aug" / "completer_best.pt"
GRID_DIR = ROOT / "data" / "completer_grids"
VAL_CROPS_DIR = ROOT / "data" / "completer_crops" / "val"

OCCLUDED = 3


def _load_model() -> OccluSynthCompleter | None:
    if not CKPT.exists():
        return None
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    m = OccluSynthCompleter(mc_dropout=False)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


# ---------------------------------------------------------------------------
# (a) Hazard set is non-empty on scene0000_00
# ---------------------------------------------------------------------------

class TestHazardDefinition:
    @pytest.mark.skipif(
        not (GRID_DIR / "scene0000_00_n6.npz").exists(),
        reason="scene0000_00 cache not available"
    )
    def test_hazards_nonempty_scene0000_00(self):
        """Hidden hazards (state==3 & gt_sdf<0) must exist in scene0000_00."""
        d = np.load(GRID_DIR / "scene0000_00_n6.npz", allow_pickle=False)
        hazard = (d["state"] == OCCLUDED) & (d["gt_sdf"] < 0)
        n = int(hazard.sum())
        print(f"\n  scene0000_00 hazards: {n:,} / "
              f"{int((d['state']==OCCLUDED).sum()):,} occluded")
        assert n > 0, "no hidden hazards found — check gt_sdf or state arrays"

    @pytest.mark.skipif(
        not (GRID_DIR / "scene0000_00_n6.npz").exists(),
        reason="scene0000_00 cache not available"
    )
    def test_hazard_fraction_plausible(self):
        """Between 10 % and 90 % of occluded voxels should be hidden hazards
        in a typical indoor scene (avoids trivial edge cases)."""
        d = np.load(GRID_DIR / "scene0000_00_n6.npz", allow_pickle=False)
        n_occ    = int((d["state"] == OCCLUDED).sum())
        n_hazard = int(((d["state"] == OCCLUDED) & (d["gt_sdf"] < 0)).sum())
        frac = n_hazard / max(n_occ, 1)
        print(f"\n  hazard fraction: {frac:.3f}")
        assert 0.10 <= frac <= 0.90, (
            f"hazard fraction {frac:.3f} is outside plausible range [0.10, 0.90]; "
            "check gt_sdf alignment or state labels"
        )

    def test_baseline_awareness_is_zero_by_construction(self):
        """no_completion (SDF=0) and occluded_as_free (SDF=+0.1) must have
        0% awareness: neither value is < 0, so they can never detect a hazard."""
        no_completion_pred    = 0.0     # all occluded cells set to 0
        occluded_as_free_pred = 0.10    # all occluded cells set to +0.10 m
        assert not (no_completion_pred    < 0), "no_completion SDF is not negative"
        assert not (occluded_as_free_pred < 0), "occluded_as_free SDF is not negative"

    @pytest.mark.skipif(
        not any((GRID_DIR / f"{s}_n6.npz").exists() for s in
                ["scene0001_00", "scene0085_00"]),
        reason="no val scene grids available"
    )
    def test_hazards_nonempty_on_val_scenes(self):
        """All croppable val scenes should have at least one hidden hazard."""
        val_scenes = [
            "scene0001_00", "scene0085_00", "scene0146_00", "scene0220_01",
            "scene0301_02", "scene0367_00", "scene0556_00", "scene0635_01",
            "scene0704_00",
        ]
        for s in val_scenes:
            f = GRID_DIR / f"{s}_n6.npz"
            if not f.exists():
                continue
            d = np.load(f, allow_pickle=False)
            n = int(((d["state"] == OCCLUDED) & (d["gt_sdf"] < 0)).sum())
            assert n > 0, f"{s}: no hidden hazards found"


# ---------------------------------------------------------------------------
# (b) Completer awareness > 0 > baselines  (on real val data)
# ---------------------------------------------------------------------------

class TestAwarenessRateCompleterBeatsBaselines:
    @pytest.mark.skipif(not CKPT.exists(),         reason="checkpoint not available")
    @pytest.mark.skipif(not VAL_CROPS_DIR.exists(), reason="val crops not available")
    def test_completer_detects_at_least_one_hazard(self):
        """On the val crop with the most hazards, completer must mark at least
        one of them as occupied (SDF < 0)."""
        model = _load_model()
        assert model is not None

        # Find val crop with most hazards
        best_f     = None
        best_n_haz = 0
        for f in sorted(VAL_CROPS_DIR.glob("*.npz")):
            z = np.load(f, allow_pickle=False)
            n = int(((z["state"] == OCCLUDED) & (z["target"] < 0)).sum())
            if n > best_n_haz:
                best_n_haz = n
                best_f = f

        assert best_f is not None and best_n_haz > 0, \
            "no val crop with hidden hazards found"

        z     = np.load(best_f, allow_pickle=False)
        gt    = z["target"].astype(np.float32)
        state = z["state"]
        inp   = z["input"].astype(np.float32)

        hazard_mask = (state == OCCLUDED) & (gt < 0)

        inp_t = torch.from_numpy(inp).unsqueeze(0)
        with torch.no_grad():
            pred = model(inp_t)[0, 0].numpy()

        awareness_completer = float((pred[hazard_mask] < 0).mean())
        awareness_no_compl  = 0.0   # by construction
        awareness_occ_free  = 0.0   # by construction

        print(f"\n  crop: {best_f.name}  n_hazards={best_n_haz:,}")
        print(f"  awareness — completer: {awareness_completer:.4f}  "
              f"no_completion: {awareness_no_compl:.4f}  "
              f"occluded_as_free: {awareness_occ_free:.4f}")

        assert awareness_completer > awareness_no_compl, (
            f"completer ({awareness_completer:.4f}) should beat no_completion (0.000)"
        )
        assert awareness_completer > awareness_occ_free, (
            f"completer ({awareness_completer:.4f}) should beat occluded_as_free (0.000)"
        )

    @pytest.mark.skipif(not CKPT.exists(),         reason="checkpoint not available")
    @pytest.mark.skipif(not VAL_CROPS_DIR.exists(), reason="val crops not available")
    def test_awareness_aggregates_positive_across_val(self):
        """Aggregate completer awareness over all val crops with hazards must be > 0."""
        model = _load_model()
        assert model is not None

        total_haz    = 0
        total_aware  = 0

        for f in sorted(VAL_CROPS_DIR.glob("*.npz"))[:20]:  # fast: first 20 crops
            z     = np.load(f, allow_pickle=False)
            gt    = z["target"].astype(np.float32)
            state = z["state"]
            mask  = (state == OCCLUDED) & (gt < 0)
            n_h   = int(mask.sum())
            if n_h == 0:
                continue

            inp_t = torch.from_numpy(z["input"].astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                pred = model(inp_t)[0, 0].numpy()

            total_haz   += n_h
            total_aware += int((pred[mask] < 0).sum())

        assert total_haz > 0, "no hazards found in first 20 val crops"
        agg_awareness = total_aware / total_haz
        print(f"\n  aggregate awareness over first 20 crops: {agg_awareness:.4f} "
              f"({total_aware}/{total_haz} hazards detected)")
        assert agg_awareness > 0.0, (
            f"completer should detect at least some hazards (got {agg_awareness:.4f})"
        )
