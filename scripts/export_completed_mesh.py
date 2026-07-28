#!/usr/bin/env python
"""
export_completed_mesh.py — before/after meshes of occluded-region completion.

Produces the demo's "object permanence" shot: the mesh a TSDF-only system sees
(*before*) next to the same scene with the completer's predicted geometry
spliced into the occluded volume (*after*), with the recreated geometry
vertex-coloured amber so it is unmistakable in renders.

Field construction (metres, same conventions as eval_geometry.py):
  before = sdf_norm * SURFACE_TRUNC          (occluded/free/unobs are +trunc)
  after  = before, with OCCLUDED voxels replaced by the completer's predicted
           SDF clipped to ±SURFACE_TRUNC     (keeps zero-crossing interpolation
                                              symmetric with the TSDF band)

Marching cubes runs at level 0 on both fields (skimage, .venv312). Vertices are
world-frame: origin + half-voxel + MC coords, so the outputs overlay directly
on the ScanNet GT mesh (<scene>_vh_clean_2.ply) in the same frame.

Outputs (demo_outputs/before_after/):
  <scene>_before.ply           measured geometry only (light grey)
  <scene>_after.ply            measured (grey) + completed (amber)
  <scene>_completed_only.ply   completed geometry alone — for turntable.py
                               --overlay on a darkened before-mesh
  <scene>_stats.json           vertex/triangle counts, % occluded predicted solid

Usage:
    .venv312/bin/python scripts/export_completed_mesh.py --scene scene0635_01
    .venv312/bin/python scripts/export_completed_mesh.py --scene scene0000_00 --no_smooth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from occlusynth.models.completer import OccluSynthCompleter
from occlusynth.utils import get_repo_root

FREE, SURFACE, OCCLUDED, UNOBSERVABLE = 1, 2, 3, 0
SURFACE_TRUNC = 0.10   # metres — denormalises the stored SDF (from TSDFConfig)

MEASURED_RGB  = (0.788, 0.804, 0.827)   # light grey — measured geometry
COMPLETED_RGB = (0.878, 0.631, 0.000)   # amber #E0A100 — imagined (HERO colour)


def run_tiled_inference(model, partial_inp, device, tile_xy=96, overlap=16,
                        state=None):
    """Full-scene completion via overlapping x-y tiles (mirrors run_safety_benchmark).

    When ``state`` is given the tiles get one-hot state channels appended
    (7-channel v2 input); prediction channel 0 is the SDF either way."""
    _, nx, ny, nz = partial_inp.shape
    n_ch = 3 if state is None else 7
    step = tile_xy - overlap
    x_starts = list(range(0, nx, step))
    y_starts = list(range(0, ny, step))
    if nx > tile_xy and (nx - x_starts[-1]) < tile_xy:
        x_starts[-1] = nx - tile_xy
    if ny > tile_xy and (ny - y_starts[-1]) < tile_xy:
        y_starts[-1] = ny - tile_xy

    acc = np.zeros((nx, ny, nz), dtype=np.float64)
    wts = np.zeros((nx, ny, nz), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for xs in x_starts:
            xe = min(xs + tile_xy, nx)
            xs = max(xs, 0)
            for ys in y_starts:
                ye  = min(ys + tile_xy, ny)
                ys_ = max(ys, 0)
                tile = np.zeros((1, n_ch, tile_xy, tile_xy, nz), dtype=np.float32)
                tw, th = xe - xs, ye - ys_
                tile[0, :3, :tw, :th, :nz] = partial_inp[:, xs:xe, ys_:ye, :nz]
                if state is not None:
                    st = state[xs:xe, ys_:ye, :nz]
                    for s in range(4):        # one-hot: unobs, free, surf, occ
                        tile[0, 3 + s, :tw, :th, :nz] = (st == s)
                pred = model(torch.from_numpy(tile).to(device))[0, 0].cpu().numpy()
                acc[xs:xe, ys_:ye, :nz] += pred[:tw, :th, :nz]
                wts[xs:xe, ys_:ye, :nz] += 1.0
    pos = wts > 0
    out = np.zeros((nx, ny, nz), dtype=np.float32)
    out[pos] = (acc[pos] / wts[pos]).astype(np.float32)
    return out


def field_to_mesh(field_m, origin, voxel_size, smooth_iters=10, min_component=30,
                  iso=0.0):
    """Marching cubes → open3d mesh (uniform grey), small shards removed.

    ``iso`` shifts the level set (solid becomes pred < iso): a positive level
    grows predicted solids (closes holes, risks bulging), a negative one shrinks
    them (tighter, more conservative geometry)."""
    import open3d as o3d
    from skimage import measure

    verts, faces, _, _ = measure.marching_cubes(
        field_m, level=iso, spacing=(voxel_size,) * 3)
    # MC lattice points are voxel centres: world = origin + half-voxel + coords
    verts_w = origin[None, :] + 0.5 * voxel_size + verts

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts_w),
        o3d.utility.Vector3iVector(faces.astype(np.int32)))

    if min_component > 1:
        cluster, tri_count, _ = mesh.cluster_connected_triangles()
        keep = np.asarray(tri_count)[np.asarray(cluster)] >= min_component
        mesh.remove_triangles_by_mask(~keep)
        mesh = mesh.remove_unreferenced_vertices()

    if smooth_iters > 0:
        # Scope=Vertex: default All would also smooth vertex colors later
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=smooth_iters,
            filter_scope=o3d.geometry.FilterScope.Vertex)
    mesh.compute_vertex_normals()
    return mesh


def color_new_geometry(mesh_after, mesh_before, voxel_size):
    """Amber vertices = geometry farther than one voxel from the before-mesh —
    i.e. surface that did not exist before completion. Returns the amber mask."""
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_before))
    q = o3d.core.Tensor(np.asarray(mesh_after.vertices, dtype=np.float32))
    dist = scene.compute_distance(q).numpy()
    new_mask = dist > voxel_size
    colors = np.where(new_mask[:, None],
                      np.array(COMPLETED_RGB), np.array(MEASURED_RGB))
    mesh_after.vertex_colors = o3d.utility.Vector3dVector(colors)
    return new_mask


def completed_only(mesh, new_mask):
    """Sub-mesh of triangles whose vertices are all new (completed) geometry."""
    import open3d as o3d
    faces = np.asarray(mesh.triangles)
    keep = new_mask[faces].all(axis=1)
    sub = o3d.geometry.TriangleMesh(
        mesh.vertices, o3d.utility.Vector3iVector(faces[keep]))
    sub.vertex_colors = mesh.vertex_colors
    sub = sub.remove_unreferenced_vertices()
    sub.compute_vertex_normals()
    return sub


def main() -> None:
    p = argparse.ArgumentParser(description="Export before/after completion meshes")
    p.add_argument("--scene", required=True)
    p.add_argument("--grid_dir", default="data/completer_grids")
    p.add_argument("--n_frames", type=int, default=6,
                   help="which cached fusion to use (<scene>_n<N>.npz); denser "
                        "fusions give a cleaner measured mesh for demo renders")
    p.add_argument("--ckpt", default="checkpoints/interim_64_aug/completer_best.pt")
    p.add_argument("--v2", action="store_true",
                   help="checkpoint is a v2 completer (7-channel input, occ head)")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--smooth_iters", type=int, default=10,
                   help="Taubin smoothing iterations (volume-preserving, so it "
                        "does not shrink geometry). Applied IDENTICALLY to the "
                        "before and after meshes — keep it symmetric so the "
                        "comparison stays honest.")
    p.add_argument("--no_smooth", action="store_true",
                   help="Skip Taubin smoothing (raw 5 cm voxel look)")
    p.add_argument("--iso", type=float, default=0.0,
                   help="marching-cubes level in metres for the COMPLETED field "
                        "(solid = pred < iso); positive grows predicted solids "
                        "(closes holes, risks bulging), negative shrinks them")
    p.add_argument("--tag", default=None,
                   help="suffix for output filenames (keeps variants side by side)")
    p.add_argument("--min_component", type=int, default=30,
                   help="Drop disconnected components smaller than this many "
                        "triangles (declutters sparse-view shard noise; 0 = keep all)")
    p.add_argument("--out_dir", default="demo_outputs/before_after")
    args = p.parse_args()

    import open3d as o3d

    root = get_repo_root()
    grid_file = root / args.grid_dir / f"{args.scene}_n{args.n_frames}.npz"
    if not grid_file.exists():
        sys.exit(f"scene grid not found: {grid_file} "
                 "(generate with scripts/generate_completer_data.py)")
    ckpt_path = root / args.ckpt
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path}")

    d          = np.load(grid_file, allow_pickle=False)
    origin     = d["origin"].astype(np.float64)
    vox        = float(d["voxel_size"])
    sdf_norm   = d["sdf"].astype(np.float32)
    state      = d["state"]
    smooth     = 0 if args.no_smooth else args.smooth_iters

    device = torch.device(args.device)
    ckpt   = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model  = (OccluSynthCompleter(in_channels=7, occ_head=True) if args.v2
              else OccluSynthCompleter(mc_dropout=False)).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[export] {args.scene}  dims={state.shape}  vox={vox} m  "
          f"ckpt ep={ckpt.get('epoch')}  v2={args.v2}")

    partial_inp = np.stack([sdf_norm, d["weight"].astype(np.float32),
                            d["p_observed"].astype(np.float32)])
    compl = run_tiled_inference(model, partial_inp, device,
                                state=state if args.v2 else None)

    before_m = sdf_norm * SURFACE_TRUNC
    occ      = state == OCCLUDED
    after_m  = before_m.copy()
    # subtract iso so marching cubes still runs at level 0: solid becomes
    # (pred < iso) in the completed region while OBSERVED surfaces are untouched
    after_m[occ] = np.clip(compl[occ] - args.iso, -SURFACE_TRUNC, SURFACE_TRUNC)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_b = field_to_mesh(before_m, origin, vox, smooth, args.min_component)
    mesh_a = field_to_mesh(after_m,  origin, vox, smooth, args.min_component)
    mesh_b.vertex_colors = o3d.utility.Vector3dVector(
        np.tile(MEASURED_RGB, (len(mesh_b.vertices), 1)))
    new_mask = color_new_geometry(mesh_a, mesh_b, vox)
    mesh_c   = completed_only(mesh_a, new_mask)

    counts = {tag: {"verts": len(m.vertices), "tris": len(m.triangles)}
              for tag, m in (("before", mesh_b), ("after", mesh_a),
                             ("completed_only", mesh_c))}

    # Pad every mesh with two unreferenced vertices at the union bbox corners so
    # turntable.py frames all three identically (crossfade-safe). Unreferenced
    # vertices are never rasterised on a TriangleMesh.
    mins = np.min([np.asarray(m.get_min_bound()) for m in (mesh_b, mesh_a)], axis=0)
    maxs = np.max([np.asarray(m.get_max_bound()) for m in (mesh_b, mesh_a)], axis=0)
    for m in (mesh_b, mesh_a, mesh_c):
        v = np.vstack([np.asarray(m.vertices), mins, maxs])
        c = np.vstack([np.asarray(m.vertex_colors), [MEASURED_RGB] * 2])
        n = np.vstack([np.asarray(m.vertex_normals), [[0.0, 0.0, 1.0]] * 2])
        m.vertices       = o3d.utility.Vector3dVector(v)
        m.vertex_colors  = o3d.utility.Vector3dVector(c)
        m.vertex_normals = o3d.utility.Vector3dVector(n)

    paths = {}
    for tag, m in (("before", mesh_b), ("after", mesh_a), ("completed_only", mesh_c)):
        path = out_dir / f"{args.scene}_{tag}{args.tag or ''}.ply"
        o3d.io.write_triangle_mesh(str(path), m, write_vertex_normals=True)
        paths[tag] = str(path.relative_to(root))
        print(f"  {tag:15s} {len(m.vertices):8d} verts  "
              f"{len(m.triangles):8d} tris  → {paths[tag]}")

    stats = {
        "scene": args.scene,
        "ckpt": args.ckpt, "epoch": ckpt.get("epoch"),
        "dims": list(state.shape), "voxel_size_m": vox,
        "n_occluded_voxels": int(occ.sum()),
        "pct_occluded_predicted_solid": round(float((compl[occ] < 0).mean()) * 100, 2),
        "before": counts["before"],
        "after":  counts["after"],
        "completed_only": counts["completed_only"],
        "files": paths,
    }
    stats_path = out_dir / f"{args.scene}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"  stats → {stats_path.relative_to(root)}")


if __name__ == "__main__":
    main()
