"""
mesh_to_tsdf — voxelize a GT ScanNet mesh into a dense SDF grid.

Produces the completer's supervision target: the *complete* signed distance
field of the scene, sampled on exactly the same 5 cm grid that
``fuse_visibility()`` builds from partial RGB-D observations.  Alignment is
everything here — the GT grid must share the partial grid's world origin and
dims voxel-for-voxel, or the completer learns a systematic offset.

Sign convention (matches open3d RaycastingScene):
    positive  = outside the mesh (free space)
    negative  = inside the mesh  (occupied)

Note: ScanNet ``_vh_clean_2.ply`` meshes are open surfaces (no closed walls
on the unscanned side), so the inside/outside sign from ray-parity can be
unreliable far from the surface.  Near the surface — the band the completer
is supervised on — the sign is solid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_CHUNK = 500_000  # query points per RaycastingScene call (memory cap)


def mesh_to_tsdf(
    mesh_path: str,
    voxel_size: float,
    world_origin: np.ndarray,
    grid_dims: tuple,
) -> np.ndarray:
    """
    Sample the GT mesh's signed distance on a dense voxel grid.

    Args:
        mesh_path:    path to ``<scene>_vh_clean_2.ply``.
        voxel_size:   metres per voxel; must match ``fuse_visibility()`` (0.05).
        world_origin: (3,) grid *corner* — ``VisibilityVoxelGrid.origin`` for
                      the same scene.  Voxel centres are at
                      ``origin + (idx + 0.5) * voxel_size``.
        grid_dims:    (nx, ny, nz) — same dims as the partial grid.

    Returns:
        float32 array of shape ``grid_dims`` — signed distance in metres at
        each voxel centre, C-order layout identical to the partial grid.
    """
    import open3d as o3d

    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"GT mesh not found: {mesh_path}")

    mesh = o3d.t.io.read_triangle_mesh(str(mesh_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    origin = np.asarray(world_origin, np.float64).reshape(3)
    nx, ny, nz = (int(d) for d in grid_dims)

    # voxel centres, C-order with indexing="ij" — same layout as
    # VisibilityVoxelGrid.centers()
    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                          indexing="ij")
    queries = np.stack(
        [origin[0] + (I.ravel() + 0.5) * voxel_size,
         origin[1] + (J.ravel() + 0.5) * voxel_size,
         origin[2] + (K.ravel() + 0.5) * voxel_size],
        axis=1,
    ).astype(np.float32)

    sdf = np.empty(len(queries), np.float32)
    for s in range(0, len(queries), _CHUNK):
        chunk = o3d.core.Tensor(queries[s:s + _CHUNK])
        sdf[s:s + _CHUNK] = scene.compute_signed_distance(chunk).numpy()

    return sdf.reshape(nx, ny, nz)
