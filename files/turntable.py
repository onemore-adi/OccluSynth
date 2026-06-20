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
    ap.add_argument("--rot_per_frame", type=float, default=3.0,
                    help="orbit speed; higher spins faster (try 1.0–6.0)")
    ap.add_argument("--point_size", type=float, default=4.0)
    ap.add_argument("--bg", default="0E1116", help="background hex (matches the cards)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        sys.exit("open3d not found — run this with your .venv312 python "
                 "(e.g. .venv312/bin/python turntable.py ...)")

    # Load as mesh if it has faces, else as a point cloud (voxel grids are point clouds).
    mesh = o3d.io.read_triangle_mesh(args.ply)
    if len(mesh.triangles) > 0:
        geom = mesh
        geom.compute_vertex_normals()
        if not geom.has_vertex_colors():
            geom.paint_uniform_color([0.7, 0.7, 0.72])
    else:
        geom = o3d.io.read_point_cloud(args.ply)
        if len(geom.points) == 0:
            sys.exit(f"No geometry found in {args.ply}")

    bg = tuple(int(args.bg[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    frames = int(round(args.seconds * args.fps))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=args.width, height=args.height, visible=True)
    vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.background_color = np.array(bg)
    opt.point_size = args.point_size
    ctr = vis.get_view_control()

    outdir = os.path.dirname(os.path.abspath(args.out)) or "."
    tmp = os.path.join(outdir, "_turntable_frames")
    os.makedirs(tmp, exist_ok=True)
    for old in glob.glob(os.path.join(tmp, "*.png")):
        os.remove(old)

    # let the view settle before capturing
    for _ in range(8):
        vis.poll_events(); vis.update_renderer()

    for i in range(frames):
        ctr.rotate(args.rot_per_frame, 0.0)   # horizontal orbit
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
