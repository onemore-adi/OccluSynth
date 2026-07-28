#!/usr/bin/env python3
"""render_gt_turntable.py — ScanNet ground-truth mesh as a turntable that is
frame-for-frame aligned with clips/shot09_{before,after}_mesh.mp4.

Those two clips were rendered by turntable.py at its defaults (12 s, 30 fps,
0.5 deg/frame, elevation 28, zoom 0.45, auto-up) and share an identical bounding
box, which is what fixes their camera. The GT mesh has a *different* extent, so
rendering it on its own would frame it differently and the 3-up comparison would
not line up.

Fix: add eight background-coloured anchor points at the before/after bounding-box
corners. They are invisible against the backdrop but force Open3D to fit the
camera to the same box, so the orbit matches exactly.

Run:  .venv312/bin/python render_gt_turntable.py
Out:  clips/shot09_gt_mesh.mp4
"""
import glob
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
GT_PLY = os.path.join(ROOT, "demo_video/renders/data/mesh_gt.ply")
REF_PLY = os.path.join(ROOT, "demo_outputs/before_after_n40/scene0000_00_before.ply")
OUT = os.path.join(ROOT, "clips/shot09_gt_mesh.mp4")

SECONDS, FPS, ROT_PER_FRAME = 12.0, 30, 0.5
ELEVATION, ZOOM = 28.0, 0.45
BG_HEX = "0E1116"
WIDTH, HEIGHT = 1920, 1080
# neutral grey: GT is shown as pure geometry, in the same shading language as the
# measured panels — the column header does the labelling, not the colour.
GT_GREY = [0.78, 0.79, 0.81]


def main():
    import open3d as o3d

    bg = np.array([int(BG_HEX[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])

    mesh = o3d.io.read_triangle_mesh(GT_PLY)
    if len(mesh.triangles) == 0:
        sys.exit(f"no triangles in {GT_PLY}")
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(GT_GREY)

    # camera anchor: the before/after bounding box, drawn in the background colour
    ref = o3d.io.read_triangle_mesh(REF_PLY)
    rb = ref.get_axis_aligned_bounding_box()
    lo, hi = np.asarray(rb.get_min_bound()), np.asarray(rb.get_max_bound())
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    anchor = o3d.geometry.PointCloud()
    anchor.points = o3d.utility.Vector3dVector(corners)
    anchor.colors = o3d.utility.Vector3dVector(np.tile(bg, (8, 1)))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=WIDTH, height=HEIGHT, visible=True)
    vis.add_geometry(mesh)
    vis.add_geometry(anchor)
    opt = vis.get_render_option()
    opt.background_color = bg
    opt.point_size = 1.0
    ctr = vis.get_view_control()

    center = (lo + hi) / 2.0
    extents = hi - lo
    up_idx = int(np.argmin(extents))
    print(f"  ref extents X={extents[0]:.2f} Y={extents[1]:.2f} Z={extents[2]:.2f}"
          f"  -> up = {'XYZ'[up_idx]}")
    up_vec = np.zeros(3)
    up_vec[up_idx] = 1.0
    h1 = np.zeros(3)
    h1[(up_idx + 1) % 3] = 1.0
    h2 = np.cross(up_vec, h1)
    elev = np.deg2rad(ELEVATION)

    tmp = os.path.join(ROOT, "clips", "_gt_frames")
    os.makedirs(tmp, exist_ok=True)
    for old in glob.glob(os.path.join(tmp, "*.png")):
        os.remove(old)

    for _ in range(8):
        vis.poll_events()
        vis.update_renderer()

    frames = int(round(SECONDS * FPS))
    for i in range(frames):
        angle = np.deg2rad(i * ROT_PER_FRAME)
        front = (np.cos(elev) * (np.cos(angle) * h1 + np.sin(angle) * h2)
                 + np.sin(elev) * up_vec)
        front = (front / np.linalg.norm(front)).tolist()
        ctr.set_lookat(center.tolist())
        ctr.set_front(front)
        ctr.set_up(up_vec.tolist())
        ctr.set_zoom(ZOOM)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(tmp, f"f{i:05d}.png"), do_render=True)
        if i % 60 == 0:
            print(f"  frame {i}/{frames}")
    vis.destroy_window()

    norm = (f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x{BG_HEX},"
            f"format=yuv420p")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "f%05d.png"), "-vf", norm,
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium", OUT],
                   check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
