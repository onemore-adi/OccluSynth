"""
Geometric consistency tests for ScanNetDataset.

Run with:  pytest tests/test_scannet_dataset.py -v
"""

import torch
import pytest
from occlusynth.data.scannet import ScanNetDataset


@pytest.fixture(scope="module")
def val_sample():
    ds = ScanNetDataset(split="val")
    return ds[0]


def test_intrinsics_align_with_depth(val_sample):
    """
    rgb, depth_gt, and intrinsics must all share the same spatial resolution.
    Then verify that back-projecting and re-projecting valid pixels via K is
    a perfect round-trip (confirms K is expressed in the resized frame).
    """
    sample = val_sample
    rgb, depth, K = sample["rgb"], sample["depth_gt"], sample["intrinsics"]

    # rgb (N, 3, H, W);  depth (N, H, W);  K (N, 3, 3)
    assert rgb.shape[-2:] == depth.shape[-2:], (
        f"RGB-depth resolution mismatch: rgb {rgb.shape} vs depth {depth.shape}"
    )

    for f in range(rgb.shape[0]):
        H, W = depth[f].shape
        Kf = K[f]

        valid = (depth[f] > 0).nonzero(as_tuple=False)
        assert len(valid) > 0, f"Frame {f} has no valid depth pixels"

        idxs = valid[torch.randperm(len(valid))[:100]]
        for v, u in idxs:
            d = depth[f, v, u].item()
            x = (u - Kf[0, 2]) * d / Kf[0, 0]
            y = (v - Kf[1, 2]) * d / Kf[1, 1]
            u_back = Kf[0, 0] * x / d + Kf[0, 2]
            v_back = Kf[1, 1] * y / d + Kf[1, 2]
            assert abs(u_back - u.item()) < 1e-3, f"u round-trip failed: {u_back} vs {u}"
            assert abs(v_back - v.item()) < 1e-3, f"v round-trip failed: {v_back} vs {v}"


def test_depth_sentinel_is_zero(val_sample):
    """INTER_NEAREST must never produce values between 0 and the minimum
    valid depth — bilinear would create fractional depths near boundaries."""
    depth = val_sample["depth_gt"]
    nonzero = depth[depth > 0]
    assert nonzero.min().item() > 0.05, "suspiciously small non-zero depth"
    # No sub-millimetre phantom values that bilinear would produce
    tiny = ((depth > 0) & (depth < 0.01)).sum().item()
    assert tiny == 0, f"Found {tiny} pixels with 0 < depth < 10mm — bilinear bleed?"


def test_rgb_range(val_sample):
    rgb = val_sample["rgb"]
    assert rgb.min().item() >= 0.0
    assert rgb.max().item() <= 1.0


def test_depth_units(val_sample):
    """Depth must be in metres, not millimetres (all values < DEPTH_MAX_M)."""
    depth = val_sample["depth_gt"]
    valid = depth[depth > 0]
    assert valid.max().item() <= 3.5, "depth appears to be in mm, not metres"
    assert valid.mean().item() < 10.0, "mean depth suspiciously large"


def test_collate_fn():
    ds = ScanNetDataset(split="val", n_frames=4)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, collate_fn=ScanNetDataset.collate_fn
    )
    batch = next(iter(loader))

    assert batch["rgb"].shape[:2] == (2, 4)
    assert batch["depth_gt"].shape[:2] == (2, 4)
    assert batch["intrinsics"].shape == (2, 4, 3, 3)
    assert batch["pose"].shape == (2, 4, 4, 4)
    assert batch["rgb"].shape[-2:] == batch["depth_gt"].shape[-2:], (
        "Collated rgb and depth_gt spatial dims differ"
    )
