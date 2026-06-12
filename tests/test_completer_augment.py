"""
Augmentation tests — the 8-variant gravity-preserving group for completer crops.

The critical property: input, target, and state must receive the IDENTICAL
transform in the same call.  An orientation mismatch between input and target
is a silent killer — the loss still decreases while the network learns garbage.

    pytest tests/test_completer_augment.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_completer import CompleterCropDataset, augment_crop  # noqa: E402

D = 8  # tiny cube — fast


def _coordinate_arrays(rng):
    """target is a deterministic field; input ch0 mirrors it; state quantizes it.
    Any inconsistency between the transforms of the three arrays then shows up
    as a direct mismatch — no reimplementation of the transform needed."""
    tgt = rng.standard_normal((D, D, D)).astype(np.float32)
    inp = np.stack([tgt, 2 * tgt, -tgt])
    state = (np.abs(tgt) * 2).astype(np.uint8) % 4
    return inp, tgt, state


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("f", [0, 1])
def test_all_arrays_transform_identically(k, f):
    """Rotated input's target must equal the rotated original target."""
    inp, tgt, state = _coordinate_arrays(np.random.default_rng(0))
    ai, at, as_ = augment_crop(inp, tgt, state, k, f)
    # input channel 0 was the target ⇒ must still be, voxel for voxel
    np.testing.assert_array_equal(ai[0], at)
    np.testing.assert_array_equal(ai[1], 2 * at)
    np.testing.assert_array_equal(ai[2], -at)
    # state was a function of target ⇒ relationship must survive
    np.testing.assert_array_equal(as_, (np.abs(at) * 2).astype(np.uint8) % 4)


def test_identity_variant():
    inp, tgt, state = _coordinate_arrays(np.random.default_rng(1))
    ai, at, as_ = augment_crop(inp, tgt, state, 0, 0)
    np.testing.assert_array_equal(at, tgt)
    np.testing.assert_array_equal(ai, inp)
    np.testing.assert_array_equal(as_, state)


def test_eight_distinct_variants():
    """The group has exactly 8 elements on an asymmetric field."""
    _, tgt, _ = _coordinate_arrays(np.random.default_rng(2))
    inp = np.stack([tgt, tgt, tgt])
    state = np.zeros_like(tgt, np.uint8)
    seen = {augment_crop(inp, tgt, state, k, f)[1].tobytes()
            for k in range(4) for f in range(2)}
    assert len(seen) == 8


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("f", [0, 1])
def test_z_axis_never_transformed(k, f):
    """No z-flips, no z-rotations: ceilings must stay above floors.

    A field that depends only on z (gravity height) must be invariant under
    every variant — yaw rotations and horizontal flips permute x/y only.
    """
    z_field = np.broadcast_to(np.arange(D, dtype=np.float32),
                              (D, D, D)).copy()       # tgt[x,y,z] = z
    inp = np.stack([z_field] * 3)
    state = z_field.astype(np.uint8)
    ai, at, as_ = augment_crop(inp, z_field, state, k, f)
    np.testing.assert_array_equal(at, z_field)
    np.testing.assert_array_equal(ai, inp)
    np.testing.assert_array_equal(as_, state)


def test_sdf_values_permuted_never_rescaled():
    """Isometries permute SDF samples; the multiset of values is unchanged."""
    inp, tgt, state = _coordinate_arrays(np.random.default_rng(3))
    for k in range(4):
        for f in range(2):
            _, at, _ = augment_crop(inp, tgt, state, k, f)
            np.testing.assert_array_equal(np.sort(at, axis=None),
                                          np.sort(tgt, axis=None))


# ── dataset-level guarantees ─────────────────────────────────────────────────

def _write_fake_crops(d: Path, n=3, size=16):
    rng = np.random.default_rng(7)
    d.mkdir(parents=True)
    for i in range(n):
        np.savez(d / f"fake_crop{i:02d}.npz",
                 input=rng.standard_normal((3, size, size, size)).astype(np.float16),
                 target=rng.standard_normal((size, size, size)).astype(np.float16),
                 state=rng.integers(0, 4, (size, size, size)).astype(np.uint8))


def test_val_dataset_ignores_augment_flag(tmp_path):
    """Val arrays must stay byte-identical even if augment=True is passed —
    every historical val number stays comparable."""
    _write_fake_crops(tmp_path / "val")
    ds_plain = CompleterCropDataset(tmp_path / "val", crop_size=16,
                                    train=False, augment=False)
    ds_aug   = CompleterCropDataset(tmp_path / "val", crop_size=16,
                                    train=False, augment=True)
    assert ds_aug.augment is False
    for i in range(len(ds_plain)):
        a, b = ds_plain[i], ds_aug[i]
        for x, y in zip(a, b):
            assert torch.equal(x, y)


def test_augment_off_is_legacy_behaviour(tmp_path):
    """train=True, augment=False (the default) must return raw crops."""
    _write_fake_crops(tmp_path / "train")
    ds = CompleterCropDataset(tmp_path / "train", crop_size=16,
                              train=True, augment=False)
    raw = np.load(sorted((tmp_path / "train").glob("*.npz"))[0])
    inp, tgt, state = ds[0]
    np.testing.assert_array_equal(inp.numpy(), raw["input"].astype(np.float32))
    np.testing.assert_array_equal(tgt.numpy(), raw["target"].astype(np.float32))


def test_augmented_samples_stay_consistent(tmp_path):
    """End-to-end: even through random augmentation, input ch0 ↔ target
    correspondence built into the fake data must survive __getitem__."""
    d = tmp_path / "train"
    d.mkdir()
    rng = np.random.default_rng(11)
    tgt = rng.standard_normal((16, 16, 16)).astype(np.float16)
    np.savez(d / "fake_crop00.npz",
             input=np.stack([tgt, tgt, tgt]),
             target=tgt,
             state=np.zeros((16, 16, 16), np.uint8))
    ds = CompleterCropDataset(d, crop_size=16, train=True, augment=True)
    np.random.seed(0)
    for _ in range(16):  # covers multiple variants
        inp, t, _ = ds[0]
        assert torch.equal(inp[0], t)
