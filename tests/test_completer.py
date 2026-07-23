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


# ── v2: state channels, occupancy head, completion_loss ─────────────────────

from occlusynth.models.completer import add_state_channels, completion_loss


def test_v2_backward_compat_old_checkpoint_shape():
    """Default constructor must be byte-compatible with v1 checkpoints."""
    old = OccluSynthCompleter()
    new = OccluSynthCompleter()
    new.load_state_dict(old.state_dict())  # raises on any shape change


def test_add_state_channels():
    rng = np.random.default_rng(2)
    inp = torch.randn(2, 3, 8, 8, 8)
    state = _random_state((2, 8, 8, 8), rng)
    x = add_state_channels(inp, state)
    assert x.shape == (2, 7, 8, 8, 8)
    assert torch.equal(x[:, :3], inp)
    # one-hot channels sum to one everywhere and match the state ids
    assert torch.equal(x[:, 3:].sum(1), torch.ones(2, 8, 8, 8))
    assert torch.equal(x[:, 3:].argmax(1), state)


def test_v2_forward_two_channels():
    m = OccluSynthCompleter(in_channels=7, occ_head=True).eval()
    x = torch.randn(1, 7, 32, 32, 32)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 2, 32, 32, 32)


def test_completion_loss_zero_when_perfect():
    """Perfect SDF + saturated correct logits + free-space margin met → ~0."""
    target = torch.randn(1, 16, 16, 16) * 0.2
    state = torch.full((1, 16, 16, 16), OCCLUDED, dtype=torch.long)
    logit = torch.where(target < 0, 50.0, -50.0).unsqueeze(1)
    pred = torch.cat([target.unsqueeze(1), logit], dim=1)
    total, parts = completion_loss(pred, target, state)
    assert total.item() < 1e-4
    assert parts["free"].item() == 0.0  # no FREE voxels present


def test_completion_loss_free_space_hinge():
    """Predicting solid inside observed free space must be penalised."""
    target = torch.full((1, 8, 8, 8), 0.5)
    state = torch.full((1, 8, 8, 8), FREE, dtype=torch.long)
    state[0, 0, 0, 0] = SURFACE  # avoid the degenerate-crop early-out
    bad = torch.full((1, 2, 8, 8, 8), -0.5)
    good = torch.full((1, 2, 8, 8, 8), 0.5)
    bad[0, 0, 0, 0, 0] = good[0, 0, 0, 0, 0] = target[0, 0, 0, 0]
    _, parts_bad = completion_loss(bad, target, state)
    _, parts_good = completion_loss(good, target, state)
    assert parts_bad["free"].item() > 0.1
    assert parts_good["free"].item() == 0.0


def test_completion_loss_truncation_caps_far_field():
    """A 3 m far-field error must cost no more than the 0.3 m truncation."""
    target = torch.full((1, 8, 8, 8), 3.0)
    state = torch.full((1, 8, 8, 8), OCCLUDED, dtype=torch.long)
    pred = torch.zeros(1, 1, 8, 8, 8)
    total, parts = completion_loss(pred, target, state, w_occ=0.0, w_free=0.0)
    assert parts["sdf"].item() <= 0.30 + 1e-6
