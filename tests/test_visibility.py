"""
Visibility-aware voxel grid regression tests.

These pin the core occlusion logic of VisibilityVoxelGrid with small synthetic
scenes (no ScanNet data / VGGT required), so they run fast and always.

    pytest tests/test_visibility.py -v
"""

import numpy as np
import pytest

from occlusynth.fusion import (
    VisibilityVoxelGrid, TSDFConfig,
    FREE, SURFACE, OCCLUDED, UNOBSERVABLE,
)


def _identity_cam(H=60, W=80, f=60.0):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
    c2w = np.eye(4)
    return K, c2w, (H, W)


def _grid_from(depth, K, c2w, cfg=None):
    cfg = cfg or TSDFConfig()
    frames = [{"depth_m": depth, "K": K, "c2w": c2w}]
    g = VisibilityVoxelGrid.from_frames(frames, cfg)
    g.integrate(depth, K, c2w, rgb=None, depth_max=cfg.depth_max,
                depth_min=cfg.depth_min)
    return g


# ── channels ──────────────────────────────────────────────────────────────────

def test_three_channels_present():
    """Grid exposes (sdf, weight, p_observed) — the channels TSDF + completer need."""
    K, c2w, (H, W) = _identity_cam()
    depth = np.full((H, W), 2.0, np.float32)
    g = _grid_from(depth, K, c2w)
    r = g.classify()
    M = int(np.prod(r.dims))
    assert g.tsdf.shape == (M,)
    assert g.weight.shape == (M,)
    assert r.p_observed.shape == (M,)
    assert 0.0 <= r.p_observed.min() and r.p_observed.max() <= 1.0


# ── carving ─────────────────────────────────────────────────────────────────────

def test_flat_wall_free_in_front_occlusion_only_behind():
    """A flat wall facing the camera: free space in front, surface at the wall,
    and any occluded voxels are only the thin shell behind the wall (within the
    bbox pad) — never in front of it."""
    K, c2w, (H, W) = _identity_cam()
    depth = np.full((H, W), 2.0, np.float32)
    cfg = TSDFConfig()
    r = _grid_from(depth, K, c2w, cfg).classify()
    assert r.counts["free"] > 0
    assert r.counts["surface"] > 0
    # all FREE voxels are in front of the wall (z < wall)
    free = r.centers[r.labels == FREE]
    assert free[:, 2].max() <= 2.0 + cfg.surface_trunc
    # any OCCLUDED voxels sit behind the wall, never in the free region in front
    occ = r.centers[r.labels == OCCLUDED]
    if len(occ):
        assert occ[:, 2].min() >= 2.0 - cfg.surface_trunc


def test_near_object_casts_occlusion_shadow():
    """A near object occluding part of a far wall creates OCCLUDED voxels in its
    shadow — these are the completer's inpaint targets."""
    K, c2w, (H, W) = _identity_cam()
    depth = np.full((H, W), 2.5, np.float32)
    depth[20:40, 30:50] = 1.0                       # near box in the centre
    r = _grid_from(depth, K, c2w).classify()
    assert r.counts["occluded"] > 0
    occ = r.centers[r.labels == OCCLUDED]
    # shadow sits behind the box (z > 1.0), not in front of the camera
    assert occ[:, 2].min() > 1.0


def test_occluded_distinct_from_unobservable():
    """OCCLUDED (in frustum, behind surface) must be a different label than
    UNOBSERVABLE (never in any valid-depth frustum). The completer inpaints the
    former only."""
    K, c2w, (H, W) = _identity_cam()
    depth = np.full((H, W), 2.5, np.float32)
    depth[20:40, 30:50] = 1.0
    r = _grid_from(depth, K, c2w).classify()
    # voxels far outside the narrow frustum exist and are UNOBSERVABLE
    assert r.counts["unobservable"] > 0
    assert OCCLUDED != UNOBSERVABLE
    # the two sets are disjoint by construction
    assert not np.any((r.labels == OCCLUDED) & (r.labels == UNOBSERVABLE))


def test_invalid_depth_gives_no_evidence():
    """Pixels with depth==0 contribute no free/surface/occluded evidence."""
    K, c2w, (H, W) = _identity_cam()
    valid = np.full((H, W), 2.0, np.float32)
    cfg = TSDFConfig()
    g = VisibilityVoxelGrid.from_frames(
        [{"depth_m": valid, "K": K, "c2w": c2w}], cfg)   # size with valid frame
    invalid = np.zeros((H, W), np.float32)
    g.integrate(invalid, K, c2w, rgb=None,
                depth_max=cfg.depth_max, depth_min=cfg.depth_min)
    assert g.free.sum() == 0 and g.surf.sum() == 0 and g.occ.sum() == 0


def test_surface_band_matches_truncation():
    """Surface voxels lie within ±surface_trunc of the measured depth.

    Fronto-parallel wall viewed head-on: obliquity correction ≈ 1, so the band
    is the thin world-space surface_trunc (2 voxels), not the fat sdf_trunc.
    """
    K, c2w, (H, W) = _identity_cam()
    depth = np.full((H, W), 2.0, np.float32)
    cfg = TSDFConfig()
    r = _grid_from(depth, K, cfg=cfg, c2w=c2w).classify()
    surf = r.centers[r.labels == SURFACE]
    assert surf[:, 2].min() >= 2.0 - cfg.surface_trunc - cfg.voxel_size
    assert surf[:, 2].max() <= 2.0 + cfg.surface_trunc + cfg.voxel_size


def test_oblique_wall_band_stays_thin():
    """
    A side wall viewed at a grazing angle must NOT get a fat surface band.

    Without obliquity correction the projective sdf = d−z underestimates the
    true distance by 1/cosθ, so a grazing wall's amber band balloons (the bug
    this fixes). With the normal correction the band thickness perpendicular to
    the wall must stay within a few voxels of surface_trunc.
    """
    H, W = 80, 100
    f = 70.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
    c2w = np.eye(4)
    cfg = TSDFConfig()

    # Side wall at x = X0 (normal ‖ x), spanning a range of depths → grazing.
    X0 = 1.4
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    # ray dir (in cam) ∝ ((u-cx)/f, (v-cy)/f, 1); it meets plane x=X0 at z = X0 / ((u-cx)/f)
    xn = (us - W / 2) / f
    with np.errstate(divide="ignore"):
        z_hit = np.where(xn > 0.05, X0 / np.maximum(xn, 1e-6), 0.0)
    depth = np.where((z_hit > 0.3) & (z_hit < 3.0), z_hit, 0.0).astype(np.float32)

    g = _grid_from(depth, K, c2w, cfg)
    r = g.classify()
    surf = r.centers[r.labels == SURFACE]
    assert len(surf) > 50
    # thickness perpendicular to the wall = spread in x around X0
    x_spread = float(surf[:, 0].max() - surf[:, 0].min())
    # uncorrected this blows past ~0.5 m; corrected it must stay tight (≤ ~5 voxels)
    assert x_spread <= 0.30, f"oblique surface band too thick in x: {x_spread:.2f} m"


# ── empty-grid guard ─────────────────────────────────────────────────────────

def test_from_frames_raises_on_no_depth():
    """Sizing the grid with zero valid depth must fail loudly, not silently."""
    K, c2w, (H, W) = _identity_cam()
    depth = np.zeros((H, W), np.float32)
    with pytest.raises(ValueError):
        VisibilityVoxelGrid.from_frames([{"depth_m": depth, "K": K, "c2w": c2w}],
                                        TSDFConfig())
