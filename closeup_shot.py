#!/usr/bin/env python3
"""
closeup_shot.py — cinematic single-object completion shot (the "sofa shot").

The camera starts at the EXACT pose of a real RGB frame (so the composited
video can crossfade photo → mesh with pixel-aligned perspective), holds, then
flies in an arc to BEHIND the object while the completed (amber) geometry
grows outward from the measured (grey) surfaces — wireframe scaffold first,
solid surface following — ending on a view no camera ever saw.

Run with the env that has open3d + scipy (your .venv312):
    .venv312/bin/python closeup_shot.py --scene scene0000_00 --frame 003200 \
        --out clips/shot10_closeup_raw.mp4

Inputs come from scripts/export_completed_mesh.py:
    demo_outputs/before_after/<scene>_before.ply          (grey, measured)
    demo_outputs/before_after/<scene>_completed_only.ply  (amber, predicted)

Use --still 0,45,150,290 to dump inspection PNGs instead of the full video.
"""
import argparse, glob, os, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

AMBER      = (0.878, 0.631, 0.000)   # solid completed surface
AMBER_WIRE = (1.000, 0.760, 0.180)   # scaffold lines, slightly brighter
GREY       = (0.788, 0.804, 0.827)
GT_GREEN   = (0.250, 0.850, 0.500)   # ground-truth verification wireframe


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def lookat_rotation(eye, target, up):
    """World→camera rotation, OpenCV convention (x right, y down, z forward)."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])   # rows = camera axes in world frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0000_00")
    ap.add_argument("--frame", default="003200", help="RGB frame stem for the start pose")
    ap.add_argument("--ply_dir", default="demo_outputs/before_after")
    ap.add_argument("--out", default="clips/shot10_closeup_raw.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--crop_radius", type=float, default=2.6,
                    help="x-y radius (m) of the hero crop around the object centre")
    ap.add_argument("--crop_top", type=float, default=2.1,
                    help="keep geometry up to this height (m) above the floor "
                         "(cuts floating ceiling/curtain shards)")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds at the photo pose")
    ap.add_argument("--flight", type=float, default=7.5, help="seconds of camera arc")
    ap.add_argument("--settle", type=float, default=3.0, help="seconds on the end pose")
    ap.add_argument("--grow_start", type=float, default=2.0)
    ap.add_argument("--grow_end", type=float, default=8.0)
    ap.add_argument("--end_radius", type=float, default=3.8, help="camera distance at the end (m)")
    ap.add_argument("--end_height", type=float, default=2.2, help="camera height above object centre (m)")
    ap.add_argument("--arc_deg", type=float, default=180.0, help="azimuth sweep of the flight")
    ap.add_argument("--scaffold_lead", type=float, default=0.45,
                    help="metres the wireframe scaffold runs ahead of the solid front")
    ap.add_argument("--gt_overlay", action="store_true",
                    help="reveal the ScanNet GT mesh as a green wireframe during "
                         "the settle phase — the verification beat")
    ap.add_argument("--gt_tris", type=int, default=20000,
                    help="decimate the cropped GT mesh to this many triangles")
    ap.add_argument("--bg", default="0E1116")
    ap.add_argument("--still", default=None,
                    help="comma-separated frame indices → PNGs next to --out, no video")
    args = ap.parse_args()

    import open3d as o3d
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation, Slerp

    from occlusynth.data.scannet import load_gt_depth, load_gt_pose, load_gt_intrinsics

    scene_dir = ROOT / "data/scannet/tasks/scannet_frames_25k" / args.scene
    c2w = load_gt_pose(scene_dir, args.frame)
    K_d = load_gt_intrinsics(scene_dir, "depth")
    depth = load_gt_depth(scene_dir, args.frame)

    # ── object centre: robust median of the frame's central unprojected depth ──
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    box = (us > W * 0.2) & (us < W * 0.8) & (vs > H * 0.2) & (vs < H * 0.8)
    valid = box & (depth > 0.5) & (depth < 4.0)
    z = depth[valid]
    x = (us[valid] - K_d[0, 2]) * z / K_d[0, 0]
    y = (vs[valid] - K_d[1, 2]) * z / K_d[1, 1]
    pts_w = (c2w[:3, :3] @ np.stack([x, y, z]) + c2w[:3, 3:4]).T
    center = np.median(pts_w, axis=0)

    # ── load + crop hero region ────────────────────────────────────────────────
    def load_crop(name, radius):
        m = o3d.io.read_triangle_mesh(str(ROOT / args.ply_dir / f"{args.scene}_{name}.ply"))
        lo = center - [radius, radius, 10.0]
        hi = center + [radius, radius, 10.0]
        m = m.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
        m.compute_vertex_normals()
        return m

    before = load_crop("before", args.crop_radius)
    compl  = load_crop("completed_only", args.crop_radius)
    floor_z = np.percentile(np.asarray(before.vertices)[:, 2], 5)
    zbox = o3d.geometry.AxisAlignedBoundingBox(
        center - [args.crop_radius, args.crop_radius, 10.0],
        np.array([center[0] + args.crop_radius, center[1] + args.crop_radius,
                  floor_z + args.crop_top]))
    def drop_shards(m, min_tris):
        """Cropping disconnects fragments at the box edges — remove them."""
        cluster, tri_count, _ = m.cluster_connected_triangles()
        keep = np.asarray(tri_count)[np.asarray(cluster)] >= min_tris
        m.remove_triangles_by_mask(~keep)
        return m.remove_unreferenced_vertices()

    before = drop_shards(before.crop(zbox), 90)
    compl  = drop_shards(compl.crop(zbox), 40)
    before.paint_uniform_color(GREY)
    compl.paint_uniform_color(AMBER)
    print(f"[closeup] centre {np.round(center, 2)}  floor z {floor_z:.2f}  "
          f"before {len(before.vertices)}v  completed {len(compl.vertices)}v")
    look = center.copy(); look[2] = floor_z + 0.55   # keep the object vertically centred

    # ── growth ordering: distance of each completed vertex to measured geometry ─
    g = cKDTree(np.asarray(before.vertices)).query(np.asarray(compl.vertices))[0]
    tris = np.asarray(compl.triangles)
    g_tri = g[tris].max(axis=1)
    g_max = float(g_tri.max()) if len(g_tri) else 1.0

    edges = np.unique(np.sort(np.concatenate(
        [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    g_edge = g[edges].max(axis=1)

    # ── optional GT verification wireframe (same world frame as the grid) ──────
    gt_edges, gt_verts = None, None
    if args.gt_overlay:
        gt_path = ROOT / f"data/scannet/scans/{args.scene}/{args.scene}_vh_clean_2.ply"
        gt = o3d.io.read_triangle_mesh(str(gt_path)).crop(zbox)
        gt = gt.simplify_quadric_decimation(target_number_of_triangles=args.gt_tris)
        gt_verts = o3d.utility.Vector3dVector(np.asarray(gt.vertices))
        gt_tris_a = np.asarray(gt.triangles)
        gt_edges = np.unique(np.sort(np.concatenate(
            [gt_tris_a[:, [0, 1]], gt_tris_a[:, [1, 2]], gt_tris_a[:, [2, 0]]]),
            axis=1), axis=0)
        rng = np.random.default_rng(7)
        gt_edges = gt_edges[rng.permutation(len(gt_edges))]
        print(f"[closeup] GT overlay: {len(gt_edges)} wireframe edges")

    # ── camera path ────────────────────────────────────────────────────────────
    w2c0 = np.linalg.inv(c2w)
    eye0 = c2w[:3, 3]
    up = np.array([0.0, 0.0, 1.0])

    rel0 = eye0 - look
    r0, th0 = np.hypot(rel0[0], rel0[1]), np.arctan2(rel0[1], rel0[0])
    z0 = eye0[2]

    n_hold   = int(round(args.hold * args.fps))
    n_flight = int(round(args.flight * args.fps))
    n_settle = int(round(args.settle * args.fps))
    n_total  = n_hold + n_flight + n_settle

    R0 = Rotation.from_matrix(w2c0[:3, :3])

    def camera_at(i):
        """Return 4x4 world→camera extrinsic for frame i."""
        if i < n_hold:
            return w2c0
        t = min((i - n_hold) / max(n_flight - 1, 1), 1.0)
        s = smoothstep(t)
        th = th0 + np.deg2rad(args.arc_deg) * s
        r  = r0 + (args.end_radius - r0) * s
        zc = z0 + (look[2] + args.end_height - z0) * s
        eye = look + np.array([r * np.cos(th), r * np.sin(th), 0.0])
        eye[2] = zc
        R_look = Rotation.from_matrix(lookat_rotation(eye, look, up))
        # blend away from the photo's exact roll over the first quarter of the arc
        blend = smoothstep(t / 0.25) if t < 0.25 else 1.0
        R = Slerp([0, 1], Rotation.concatenate([R0, R_look]))([blend])[0]
        E = np.eye(4)
        E[:3, :3] = R.as_matrix()
        E[:3, 3] = -R.as_matrix() @ eye
        return E

    # ── intrinsics scaled to the render window (photo K, height-fit, centred) ──
    K_c = load_gt_intrinsics(scene_dir, "color")
    import cv2
    rgb = cv2.imread(str(scene_dir / "color" / f"{args.frame}.jpg"))
    ch, cw = rgb.shape[:2]
    s = args.height / ch
    fx, fy = K_c[0, 0] * s, K_c[1, 1] * s
    cx = K_c[0, 2] * s + (args.width - cw * s) / 2.0
    cy = K_c[1, 2] * s
    intr = o3d.camera.PinholeCameraIntrinsic(args.width, args.height, fx, fy, cx, cy)
    print(f"[closeup] photo {cw}x{ch} scale {s:.3f} → render offset x {(args.width - cw*s)/2:.0f}px")

    # ── render loop ────────────────────────────────────────────────────────────
    bg = tuple(int(args.bg[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=args.width, height=args.height, visible=True)
    vis.add_geometry(before)
    opt = vis.get_render_option()
    opt.background_color = np.array(bg)
    opt.line_width = 1.5
    # The TSDF crust is a thin shell — without back faces it turns into
    # see-through slats whenever the camera crosses to its far side.
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    for _ in range(8):
        vis.poll_events(); vis.update_renderer()

    outdir = os.path.dirname(os.path.abspath(args.out)) or "."
    tmp = os.path.join(outdir, "_closeup_frames")
    os.makedirs(tmp, exist_ok=True)
    for old in glob.glob(os.path.join(tmp, "*.png")):
        os.remove(old)

    stills = ([int(x) for x in args.still.split(",")] if args.still else None)
    grow_f0, grow_f1 = args.grow_start * args.fps, args.grow_end * args.fps

    solid = o3d.geometry.TriangleMesh()
    wire = o3d.geometry.LineSet()
    gtwire = o3d.geometry.LineSet()
    have_solid = have_wire = have_gt = False
    gt_f0 = n_hold + n_flight + int(0.3 * args.fps)          # settle + 0.3 s
    gt_f1 = gt_f0 + int(1.2 * args.fps)                      # reveal over 1.2 s
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intr

    for i in range(n_total):
        r_t = g_max * smoothstep((i - grow_f0) / max(grow_f1 - grow_f0, 1))

        if have_solid:
            vis.remove_geometry(solid, reset_bounding_box=False); have_solid = False
        if have_wire:
            vis.remove_geometry(wire, reset_bounding_box=False); have_wire = False

        keep_t = g_tri <= r_t
        if keep_t.any():
            solid = o3d.geometry.TriangleMesh(
                compl.vertices, o3d.utility.Vector3iVector(tris[keep_t]))
            solid.paint_uniform_color(AMBER)
            solid.compute_vertex_normals()
            vis.add_geometry(solid, reset_bounding_box=False); have_solid = True

        keep_e = (g_edge <= r_t + args.scaffold_lead) & (g_edge > r_t)
        if keep_e.any() and r_t > 0:
            wire = o3d.geometry.LineSet(
                compl.vertices, o3d.utility.Vector2iVector(edges[keep_e]))
            wire.paint_uniform_color(AMBER_WIRE)
            vis.add_geometry(wire, reset_bounding_box=False); have_wire = True

        if gt_edges is not None and i >= gt_f0:
            if have_gt:
                vis.remove_geometry(gtwire, reset_bounding_box=False); have_gt = False
            n_show = int(smoothstep((i - gt_f0) / max(gt_f1 - gt_f0, 1)) * len(gt_edges))
            if n_show > 0:
                gtwire = o3d.geometry.LineSet(
                    gt_verts, o3d.utility.Vector2iVector(gt_edges[:n_show]))
                gtwire.paint_uniform_color(GT_GREEN)
                vis.add_geometry(gtwire, reset_bounding_box=False); have_gt = True

        cam.extrinsic = camera_at(i)
        ctr.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
        vis.poll_events(); vis.update_renderer()

        if stills is not None:
            if i in stills:
                p = os.path.join(outdir, f"closeup_still_{i:05d}.png")
                vis.capture_screen_image(p, do_render=True)
                print("wrote", p)
            if i >= max(stills):
                break
        else:
            vis.capture_screen_image(os.path.join(tmp, f"f{i:05d}.png"), do_render=True)

    vis.destroy_window()
    if stills is not None:
        return

    norm = (f"scale={args.width}:{args.height}:force_original_aspect_ratio=decrease,"
            f"pad={args.width}:{args.height}:(ow-iw)/2:(oh-ih)/2:color=0x{args.bg},format=yuv420p")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
         "-i", os.path.join(tmp, "f%05d.png"),
         "-vf", norm, "-c:v", "libx264", "-crf", "18", "-preset", "medium", args.out],
        check=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
