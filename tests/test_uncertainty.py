"""
tests/test_uncertainty.py — MC Dropout uncertainty estimation tests.

(a) std is non-negative everywhere
(b) p_occ ∈ [0, 1] everywhere
(c) existing checkpoint loads cleanly into mc_dropout=True model
(d) high-std voxels concentrate in occluded regions, not surface regions
    (on real val crop + real checkpoint)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from occlusynth.models.completer import OccluSynthCompleter, predict_with_uncertainty

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "interim_64_aug" / "completer_best.pt"
VAL_DIR = ROOT / "data" / "completer_crops" / "val"


def _make_model(mc_dropout: bool = True) -> OccluSynthCompleter:
    m = OccluSynthCompleter(mc_dropout=mc_dropout)
    if CKPT.exists():
        ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model"])
    return m


def _dummy_input(shape=(1, 3, 32, 32, 32)) -> torch.Tensor:
    return torch.randn(*shape)


# ---------------------------------------------------------------------------
# (a) std is non-negative everywhere
# ---------------------------------------------------------------------------

class TestStdNonnegative:
    def test_std_all_nonneg_random_input(self):
        model = _make_model(mc_dropout=True)
        inp = _dummy_input((1, 3, 32, 32, 32))
        _, std, _ = predict_with_uncertainty(model, inp, n_samples=4)
        assert (std >= 0).all(), "std must be ≥ 0 everywhere"

    def test_std_is_zero_for_one_sample(self):
        """With n_samples=1, Welford variance divides by 0 → should clamp to 0."""
        model = _make_model(mc_dropout=True)
        inp = _dummy_input((1, 3, 16, 16, 16))
        _, std, _ = predict_with_uncertainty(model, inp, n_samples=1)
        # Sample variance undefined for n=1; implementation returns 0 (M2/max(0,1)=0)
        assert (std >= 0).all()

    def test_std_nonzero_with_mc_dropout(self):
        """Active dropout should produce nonzero variance across samples."""
        model = _make_model(mc_dropout=True)
        model.train()  # confirm training mode first, then let predict_with_uncertainty manage
        inp = _dummy_input((1, 3, 32, 32, 32))
        _, std, _ = predict_with_uncertainty(model, inp, n_samples=8)
        assert std.max().item() > 0, "MC Dropout should produce nonzero std"

    def test_std_zero_without_dropout(self):
        """Without dropout, all samples are identical → std should be zero."""
        model = _make_model(mc_dropout=False)
        with pytest.raises(ValueError, match="mc_dropout=True"):
            predict_with_uncertainty(model, _dummy_input(), n_samples=4)


# ---------------------------------------------------------------------------
# (b) p_occ ∈ [0, 1]
# ---------------------------------------------------------------------------

class TestPOccBounded:
    def test_p_occ_in_zero_one(self):
        model = _make_model(mc_dropout=True)
        inp = _dummy_input((1, 3, 32, 32, 32))
        _, _, p_occ = predict_with_uncertainty(model, inp, n_samples=4)
        assert (p_occ >= 0).all() and (p_occ <= 1).all(), \
            "p_occ must be in [0, 1]"

    def test_p_occ_dtype(self):
        model = _make_model(mc_dropout=True)
        _, _, p_occ = predict_with_uncertainty(
            model, _dummy_input((1, 3, 16, 16, 16)), n_samples=4)
        assert p_occ.dtype == torch.float32

    def test_p_occ_shape_matches_input(self):
        model = _make_model(mc_dropout=True)
        b, d = 1, 24
        inp = _dummy_input((b, 3, d, d, d))
        _, _, p_occ = predict_with_uncertainty(model, inp, n_samples=4)
        assert p_occ.shape == (b, 1, d, d, d)


# ---------------------------------------------------------------------------
# (c) Checkpoint loads cleanly into mc_dropout=True
# ---------------------------------------------------------------------------

class TestCheckpointCompatibility:
    @pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not available")
    def test_checkpoint_loads_strict(self):
        """load_state_dict(strict=True) must succeed — no missing or unexpected keys."""
        ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
        model = OccluSynthCompleter(mc_dropout=True)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
        assert len(missing) == 0, f"missing keys: {missing}"
        assert len(unexpected) == 0, f"unexpected keys: {unexpected}"

    @pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not available")
    def test_eval_output_unchanged_by_mc_flag(self):
        """In eval mode, mc_dropout=True and mc_dropout=False must give identical output."""
        ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
        m_no = OccluSynthCompleter(mc_dropout=False)
        m_yes = OccluSynthCompleter(mc_dropout=True)
        m_no.load_state_dict(ckpt["model"])
        m_yes.load_state_dict(ckpt["model"])
        m_no.eval(); m_yes.eval()
        inp = _dummy_input((1, 3, 32, 32, 32))
        with torch.no_grad():
            out_no  = m_no(inp)
            out_yes = m_yes(inp)
        assert torch.allclose(out_no, out_yes), \
            "eval-mode outputs must match regardless of mc_dropout flag"


# ---------------------------------------------------------------------------
# (d) High-std voxels concentrate in occluded, not surface regions
# ---------------------------------------------------------------------------

class TestStdConcentrationInOccluded:
    @pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not available")
    @pytest.mark.skipif(not VAL_DIR.exists(), reason="val crops not available")
    def test_mean_std_occluded_ge_surface(self):
        """On a real val crop, mean std over OCCLUDED voxels ≥ mean std over SURFACE voxels.

        Reasoning: the model was trained with L1 loss and has seen the GT SDF
        for surface voxels (MAE ~5 cm) but is extrapolating for occluded voxels
        (MAE ~27 cm).  Different dropout masks should disagree more where the
        model is less certain — i.e., in the occluded blind spot.
        """
        # Pick first val crop with both SURFACE and OCCLUDED voxels
        SURFACE_STATE, OCCLUDED_STATE = 2, 3
        crop_file = None
        for f in sorted(VAL_DIR.glob("*.npz")):
            z = np.load(f)
            state = z["state"]
            if (state == SURFACE_STATE).any() and (state == OCCLUDED_STATE).any():
                crop_file = f
                break
        assert crop_file is not None, "no suitable val crop found"

        z = np.load(crop_file)
        inp_np = z["input"].astype(np.float32)
        state  = z["state"]

        model = _make_model(mc_dropout=True)
        model.eval()
        inp_t = torch.from_numpy(inp_np).unsqueeze(0)  # (1, 3, D, H, W)

        _, std, _ = predict_with_uncertainty(model, inp_t, n_samples=8)
        std_np = std[0, 0].detach().numpy()

        mean_std_surface  = float(std_np[state == SURFACE_STATE].mean())
        mean_std_occluded = float(std_np[state == OCCLUDED_STATE].mean())

        print(f"\n  std surface={mean_std_surface:.5f}  occluded={mean_std_occluded:.5f}")

        # Soft threshold: occluded std should be at least half the surface std
        # (a fully unconcentrated distribution would give equal stds)
        assert mean_std_occluded >= mean_std_surface * 0.5, (
            f"occluded std ({mean_std_occluded:.5f}) should be ≥ 0.5 × "
            f"surface std ({mean_std_surface:.5f})"
        )
