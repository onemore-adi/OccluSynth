"""
OccluSynthCompleter unit tests — shape, parameter budget, and (critically)
that the loss mask supervises SURFACE ∪ OCCLUDED and nothing else.

    pytest tests/test_completer.py -v
"""

import numpy as np
import pytest
import torch

from occlusynth.models.completer import (OccluSynthCompleter, masked_l1_loss,
                                         UNOBSERVABLE, FREE, SURFACE, OCCLUDED)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return OccluSynthCompleter().eval()


def test_forward_shape(model):
    x = torch.randn(4, 3, 64, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, 1, 64, 64, 64)
    assert torch.isfinite(y).all()


def test_forward_crop96(model):
    """The training crop size must pass through the 4-level U-Net cleanly."""
    x = torch.randn(1, 3, 96, 96, 96)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 96, 96, 96)


def test_param_budget(model):
    n = sum(p.numel() for p in model.parameters())
    assert 8e6 <= n <= 15e6, f"{n/1e6:.1f}M params outside 8-15M target"


# ── loss mask ────────────────────────────────────────────────────────────────

def _random_state(shape, rng):
    return torch.from_numpy(
        rng.integers(0, 4, size=shape).astype(np.int64))


def test_loss_zero_when_pred_equals_target():
    rng = np.random.default_rng(0)
    target = torch.randn(2, 16, 16, 16)
    pred = target.unsqueeze(1).clone()
    state = _random_state((2, 16, 16, 16), rng)
    assert masked_l1_loss(pred, target, state).item() == 0.0


def test_loss_ignores_unobservable_and_free():
    """Corrupting the prediction ONLY outside the mask must not move the loss."""
    rng = np.random.default_rng(1)
    target = torch.randn(2, 16, 16, 16)
    state = _random_state((2, 16, 16, 16), rng)
    pred = target.unsqueeze(1).clone()
    unsupervised = (state == UNOBSERVABLE) | (state == FREE)
    pred[:, 0][unsupervised] += 100.0
    assert masked_l1_loss(pred, target, state).item() == 0.0


def test_loss_nonzero_on_supervised_voxels():
    """Corrupting the prediction in the SURFACE/OCCLUDED region must register."""
    rng = np.random.default_rng(2)
    target = torch.randn(2, 16, 16, 16)
    state = _random_state((2, 16, 16, 16), rng)
    pred = target.unsqueeze(1).clone()
    supervised = (state == SURFACE) | (state == OCCLUDED)
    pred[:, 0][supervised] += 1.0
    assert masked_l1_loss(pred, target, state).item() == pytest.approx(1.0)


def test_loss_degenerate_crop_no_nan():
    """All-unsupervised crop: loss must be 0 with a valid graph, never NaN."""
    pred = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
    target = torch.randn(1, 8, 8, 8)
    state = torch.full((1, 8, 8, 8), FREE, dtype=torch.int64)
    loss = masked_l1_loss(pred, target, state)
    assert loss.item() == 0.0
    loss.backward()                      # graph intact
    assert torch.isfinite(pred.grad).all()


def test_gradients_flow(model):
    x = torch.randn(1, 3, 32, 32, 32)
    target = torch.randn(1, 32, 32, 32)
    state = torch.full((1, 32, 32, 32), OCCLUDED, dtype=torch.int64)
    m = OccluSynthCompleter()
    loss = masked_l1_loss(m(x), target, state)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
