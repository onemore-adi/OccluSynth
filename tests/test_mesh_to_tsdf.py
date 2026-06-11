"""
mesh_to_tsdf() tests — GT SDF voxelization for the completer.

Two layers:
  * synthetic box mesh — fast, no data needed, pins sign convention + values
  * scene0000_00 alignment — surface voxels from fuse_visibility() must sit on
    the GT mesh's SDF zero-crossing (skipped when ScanNet data is absent)

    pytest tests/test_mesh_to_tsdf.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from occlusynth.fusion import (TSDFConfig, SURFACE,
                               fuse_visibility_grid, mesh_to_tsdf)

ROOT = Path(__file__).resolve().parents[1]
SCENE = "scene0000_00"
MESH = ROOT / f"data/scannet/scans/{SCENE}/{SCENE}_vh_clean_2.ply"
VOXEL = 0.05


# ── synthetic: unit box ───────────────────────────────────────────────────────

def _box_mesh_path(tmp_path):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)  # corner at origin
    path = tmp_path / "box.ply"
    o3d.io.write_triangle_mesh(str(path), mesh)
    return path


def test_box_sign_convention(tmp_path):
    """Positive outside the box, negative inside, |sdf| = distance to wall."""
    path = _box_mesh_path(tmp_path)
    origin = np.array([-0.5, -0.5, -0.5])
    dims = (40, 40, 40)  # spans [-0.5, 1.5] at 5 cm
    sdf = mesh_to_tsdf(str(path), VOXEL, origin, dims)
    assert sdf.shape == dims
    assert sdf.dtype == np.float32

    def at(p):  # world point → voxel index
        idx = np.floor((np.asarray(p) - origin) / VOXEL).astype(int)
        return sdf[tuple(idx)]

    assert at([0.5, 0.5, 0.5]) < 0          # box centre: inside
    assert at([-0.3, 0.5, 0.5]) > 0         # outside the box
    # box centre is 0.5 m from every face
    assert abs(abs(at([0.5, 0.5, 0.5])) - 0.5) < 2 * VOXEL
    # outside distance to nearest face
    assert abs(at([-0.3, 0.5, 0.5]) - 0.3) < 2 * VOXEL


def test_box_zero_crossing_on_wall(tmp_path):
    """|sdf| is small exactly at the box walls."""
    path = _box_mesh_path(tmp_path)
    origin = np.array([-0.5, -0.5, -0.5])
    sdf = mesh_to_tsdf(str(path), VOXEL, origin, (40, 40, 40))
    # voxels whose centre lies within half a voxel of the x=0 wall
    I = np.arange(40)
    xc = origin[0] + (I + 0.5) * VOXEL
    wall = np.abs(xc) < VOXEL / 2 + 1e-9
    # interior span of the wall in y/z
    assert np.all(np.abs(sdf[wall][:, 12:28, 12:28]) < VOXEL)


# ── scene0000_00: alignment with fuse_visibility ──────────────────────────────

@pytest.fixture(scope="module")
def scene_grid_and_gt():
    from occlusynth.data import ScanNetDataset

    dataset = ScanNetDataset(n_frames=6, split="all")
    item = dataset[dataset.scenes.index(SCENE)]
    depth = item["depth_gt"].numpy()
    poses = item["pose"].numpy()
    K = item["intrinsics"][0].numpy()
    frames = [{"depth_m": depth[i], "K": K, "c2w": poses[i]}
              for i in range(len(item["frame_idx"]))]

    cfg = TSDFConfig()
    grid = fuse_visibility_grid(frames, cfg)
    gt_sdf = mesh_to_tsdf(str(MESH), cfg.voxel_size, grid.origin, grid.dims)
    return grid, gt_sdf


@pytest.mark.skipif(not MESH.exists(), reason="ScanNet scene0000_00 not downloaded")
def test_surface_at_zero_crossing(scene_grid_and_gt):
    """Surface voxels from fuse_visibility() sit on the GT SDF zero-crossing.

    If the world origins of the two grids differ, this median jumps by the
    offset — it is the gate before generating any training data.
    """
    grid, gt_sdf = scene_grid_and_gt
    abs_at_surf = np.abs(gt_sdf[grid.state == SURFACE])
    assert len(abs_at_surf) > 1000
    # median absolute GT SDF at surface voxels < 1.5 * voxel_size
    assert float(np.median(abs_at_surf)) < 1.5 * VOXEL  # < 0.075 m


@pytest.mark.skipif(not MESH.exists(), reason="ScanNet scene0000_00 not downloaded")
def test_gt_sdf_shape_matches_partial_grid(scene_grid_and_gt):
    grid, gt_sdf = scene_grid_and_gt
    assert gt_sdf.shape == grid.dims
    assert np.isfinite(gt_sdf).all()
