#!/usr/bin/env python
"""make_slide_meshes.py — the three frame-aligned mesh panels for the deck/video.

Renders, at identical camera and scale so the panels can be compared directly:

  mesh_before.png   measured geometry only (TSDF fusion)      — grey
  mesh_after.png    measured + completer's predicted geometry — grey + amber
  mesh_gt.png       ScanNet ground truth                      — the answer key

The before/after meshes come from ``export_completed_mesh.py``, which applies
IDENTICAL smoothing and shard-culling to both, so the visual difference between
the panels is the completion and nothing else. Ground truth is rendered from the
untouched ScanNet mesh.

Every mesh is padded with two unreferenced vertices at the shared bounding-box
corners, so Open3D's automatic framing puts all three at the same scale
(unreferenced vertices are never rasterised).

Usage:
    .venv312/bin/python scripts/make_slide_meshes.py \
        --before demo_outputs/before_after/scene0000_00_before_polish.ply \
        --after  demo_outputs/before_after/scene0000_00_after_polish.ply \
        --gt     data/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.ply \
        --out_dir demo_video/renders/hb5/slide_assets
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent
GT_RGB = (0.62, 0.65, 0.70)      # neutral grey for the ground-truth panel

# Rendering runs one mesh per subprocess: Open3D's windowed Visualizer does not
# survive repeated create/destroy in a single process (and headless EGL is
# unavailable on macOS), so a loop in-process silently yields blank frames.
_WORKER = r'''
import sys
import numpy as np, open3d as o3d
ply, png, w, h, zoom = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
bg = [int(sys.argv[6][i:i+2], 16) / 255.0 for i in (0, 2, 4)]
m = o3d.io.read_triangle_mesh(ply)
m.compute_vertex_normals()
vis = o3d.visualization.Visualizer()
vis.create_window(width=w, height=h, visible=False)
vis.add_geometry(m)
o = vis.get_render_option()
o.background_color = np.array(bg)
o.mesh_show_back_face = True
c = vis.get_view_control()
c.set_up([0, 0, 1]); c.set_front([0.58, -0.68, 0.45]); c.set_zoom(zoom)
for _ in range(12):
    vis.poll_events(); vis.update_renderer()
vis.capture_screen_image(png, do_render=True)
vis.destroy_window()
'''


def padded(mesh, mins, maxs, rgb=None):
    """Pad with the shared bbox corners so every panel frames identically."""
    if rgb is not None or not mesh.has_vertex_colors():
        base = np.tile(rgb or GT_RGB, (len(mesh.vertices), 1))
        mesh.vertex_colors = o3d.utility.Vector3dVector(base)
    mesh.compute_vertex_normals()
    v = np.vstack([np.asarray(mesh.vertices), mins, maxs])
    c = np.vstack([np.asarray(mesh.vertex_colors), [GT_RGB] * 2])
    n = np.vstack([np.asarray(mesh.vertex_normals), [[0, 0, 1.0]] * 2])
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    mesh.vertex_normals = o3d.utility.Vector3dVector(n)
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out_dir", default="demo_video/renders/hb5/slide_assets")
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=820)
    ap.add_argument("--zoom", type=float, default=0.44)
    ap.add_argument("--bg", default="0B0E14",
                    help="background hex — matches the deck card colour")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [("mesh_before.png", args.before, None),
             ("mesh_after.png", args.after, None),
             ("mesh_gt.png", args.gt, GT_RGB)]

    meshes = {}
    for name, path, rgb in items:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            sys.exit(f"missing mesh: {p}")
        meshes[name] = (o3d.io.read_triangle_mesh(str(p)), rgb)

    # shared framing box across all three panels
    mins = np.min([np.asarray(m.get_min_bound()) for m, _ in meshes.values()], axis=0)
    maxs = np.max([np.asarray(m.get_max_bound()) for m, _ in meshes.values()], axis=0)
    print(f"shared bbox min={np.round(mins,2)} max={np.round(maxs,2)}")

    with tempfile.TemporaryDirectory() as td:
        worker = Path(td) / "_render_worker.py"
        worker.write_text(_WORKER)
        for name, (mesh, rgb) in meshes.items():
            padded(mesh, mins, maxs, rgb)
            tmp_ply = Path(td) / f"{name}.ply"
            o3d.io.write_triangle_mesh(str(tmp_ply), mesh, write_vertex_normals=True)
            png = out_dir / name
            subprocess.run([sys.executable, str(worker), str(tmp_ply), str(png),
                            str(args.width), str(args.height), str(args.zoom),
                            args.bg],
                           check=True, capture_output=True)
            try:
                shown = png.relative_to(ROOT)
            except ValueError:      # --out_dir outside the repo
                shown = png
            print(f"  {name:18s} {len(mesh.vertices):7d} verts → {shown}")


if __name__ == "__main__":
    main()
