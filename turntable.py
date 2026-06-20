#!/usr/bin/env python3
"""
turntable.py — turn a .ply into a smooth orbiting video clip.

Run with the env that has open3d (your .venv312):
    .venv312/bin/python turntable.py --ply VOXELS.ply --out clips/shot04_fusion.mp4 --seconds 24

Works for BOTH the voxel grid PLY (green/red/amber points) and the marching-cubes
mesh PLY. It opens a render window, slowly rotates the model, saves a frame per
tick, then stitches the frames into an mp4 with ffmpeg. No manual camera wrangling.

If the spin is too fast/slow, change --rot_per_frame (try 1.0–6.0; higher = faster).
If the window opens tiny on a Retina Mac, that's fine — frames capture at full pixels.
"""
import argparse, glob, os, subprocess, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="input .ply (mesh or point cloud)")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--rot_per_frame", type=float, default=0.5,
                    help="orbit speed in degrees per frame — 0.5 = 1 full rotation per 24 s at 30 fps")
    ap.add_argument("--elevation", type=float, default=28.0,
                    help="camera elevation above the floor plane in degrees (try 15–45)")
    ap.add_argument("--zoom", type=float, default=0.45,
                    help="zoom level; smaller = tighter (try 0.3–0.65)")
    ap.add_argument("--up", choices=["auto", "x", "y", "z"], default="auto",
                    help="world up axis; auto picks the bounding-box axis with smallest extent")
    ap.add_argument("--point_size", type=float, default=4.0)
    ap.add_argument("--bg", default="0E1116", help="background hex (matches the cards)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    # Overlay support — load a second PLY (e.g. voxel grid) on top of the primary
    ap.add_argument("--overlay", metavar="PLY", default=None,
                    help="second .ply to render on top of the primary geometry")
    ap.add_argument("--mesh_color", default="1e2530",
                    help="hex colour for the primary mesh when an overlay is present "
                         "(keep it dark so the overlay pops); default is near-black navy")
    ap.add_argument("--wireframe", action="store_true",
                    help="show the primary mesh as wireframe so the overlay is visible through walls")
    args = ap.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        sys.exit("open3d not found — run this with your .venv312 python "
                 "(e.g. .venv312/bin/python turntable.py ...)")

    def hex_color(h):
        return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

    def load_ply(path, muted_color=None):
        """Return an Open3D geometry from a PLY, mesh or point cloud."""
        m = o3d.io.read_triangle_mesh(path)
        if len(m.triangles) > 0:
            m.compute_vertex_normals()
            if muted_color:
                m.paint_uniform_color(muted_color)
            elif not m.has_vertex_colors():
                m.paint_uniform_color([0.7, 0.7, 0.72])
            return m
        pc = o3d.io.read_point_cloud(path)
        if len(pc.points) == 0:
            sys.exit(f"No geometry found in {path}")
        return pc

    # Primary geometry — dim the mesh to background if an overlay will be drawn on top
    mesh_col = hex_color(args.mesh_color) if args.overlay else None
    geom = load_ply(args.ply, muted_color=mesh_col)

    # Optional overlay (e.g. voxel grid on top of the GT mesh)
    overlay_geom = load_ply(args.overlay) if args.overlay else None

    bg = tuple(int(args.bg[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    frames = int(round(args.seconds * args.fps))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=args.width, height=args.height, visible=True)
    vis.add_geometry(geom)
    if overlay_geom is not None:
        vis.add_geometry(overlay_geom)
    opt = vis.get_render_option()
    opt.background_color = np.array(bg)
    opt.point_size = args.point_size
    if args.wireframe:
        opt.mesh_show_wireframe = True
        opt.mesh_show_back_face = True
    ctr = vis.get_view_control()

    # --- Determine the scene's up axis -----------------------------------
    # Use the combined bounding box of all geometry so the camera centres correctly
    # even when a second PLY (overlay) extends the scene slightly.
    all_geoms = [geom] + ([overlay_geom] if overlay_geom else [])
    mins = np.min([np.asarray(g.get_axis_aligned_bounding_box().get_min_bound())
                   for g in all_geoms], axis=0)
    maxs = np.max([np.asarray(g.get_axis_aligned_bounding_box().get_max_bound())
                   for g in all_geoms], axis=0)
    center  = (mins + maxs) / 2.0
    extents = maxs - mins

    if args.up == "auto":
        up_idx = int(np.argmin(extents))   # shortest dimension = room height = up
        print(f"  extents  X={extents[0]:.2f}  Y={extents[1]:.2f}  Z={extents[2]:.2f}")
        print(f"  auto up  → {'XYZ'[up_idx]} axis (smallest extent)")
    else:
        up_idx = {"x": 0, "y": 1, "z": 2}[args.up]

    up_vec = np.zeros(3);  up_vec[up_idx] = 1.0
    # Two orthogonal horizontal directions for the orbit circle
    h1 = np.zeros(3);  h1[(up_idx + 1) % 3] = 1.0
    h2 = np.cross(up_vec, h1)   # completes a right-handed frame

    elev = np.deg2rad(args.elevation)
    center_list = center.tolist()
    up_list = up_vec.tolist()

    outdir = os.path.dirname(os.path.abspath(args.out)) or "."
    tmp = os.path.join(outdir, "_turntable_frames")
    os.makedirs(tmp, exist_ok=True)
    for old in glob.glob(os.path.join(tmp, "*.png")):
        os.remove(old)

    # Warm-up so the renderer is fully initialised before we override the camera.
    for _ in range(8):
        vis.poll_events(); vis.update_renderer()

    for i in range(frames):
        angle = np.deg2rad(i * args.rot_per_frame)
        # Camera sits on a cone: radius cos(elev), height sin(elev), orbiting up_vec.
        # `front` = direction FROM scene-center TO camera (Open3D's convention).
        front = (np.cos(elev) * (np.cos(angle) * h1 + np.sin(angle) * h2)
                 + np.sin(elev) * up_vec)
        front = (front / np.linalg.norm(front)).tolist()

        ctr.set_lookat(center_list)
        ctr.set_front(front)
        ctr.set_up(up_list)
        ctr.set_zoom(args.zoom)
        vis.poll_events(); vis.update_renderer()
        vis.capture_screen_image(os.path.join(tmp, f"f{i:05d}.png"), do_render=True)
    vis.destroy_window()

    norm = (f"scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x{args.bg},format=yuv420p")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
         "-i", os.path.join(tmp, "f%05d.png"),
         "-vf", norm, "-c:v", "libx264", "-crf", "18", "-preset", "medium", args.out],
        check=True)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
