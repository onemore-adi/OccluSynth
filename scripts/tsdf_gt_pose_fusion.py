#!/usr/bin/env python
"""
tsdf_gt_pose_fusion.py — TSDF fusion using ScanNet GT poses + VGGT depth.

Design decision: GT poses are used instead of VGGT-predicted poses.
See docs/architecture.md §Camera Pose Strategy for the full rationale.

TL;DR: VGGT ATE = 70 cm, TSDF voxel = 5 cm.  Pose error must be < 2.5 cm.
Using ScanNet BundleFusion GT poses is the correct MVP choice.

Usage:
    python scripts/tsdf_gt_pose_fusion.py --scene scene0000_00 --n_frames 20
    python scripts/tsdf_gt_pose_fusion.py --scene scene0000_00 --use_gt_depth
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VGGT_SRC  = os.path.join(REPO_ROOT, "vggt", "vggt-omega")
DATA_ROOT = os.path.join(REPO_ROOT, "data", "scannet", "tasks", "scannet_frames_25k")
OUT_DIR   = os.path.join(REPO_ROOT, "demo_outputs", "tsdf_fusion")
sys.path.insert(0, VGGT_SRC)

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

# TSDF constants — see docs/architecture.md
VOXEL_SIZE    = 0.05          # 5 cm
SDF_TRUNC     = 4 * VOXEL_SIZE  # 20 cm (4× rule)
DEPTH_MAX     = 3.5           # metres (Kinect v1 reliable range)
DEPTH_MM_SCALE = 1000.0       # ScanNet PNG → metres
VGGT_DEPTH_SCALE = 7.39       # baseline; adapter will refine this per-scene

IMG_RESOLUTION = 512


# ── helpers ───────────────────────────────────────────────────────────────────

def sep(msg=""):
    w = 62
    if msg:
        print(f"\n── {msg} {'─' * max(0, w - len(msg) - 4)}")
    else:
        print("─" * w)


def load_gt_pose(scene_dir, stem):
    """
    Load ScanNet camera-to-world pose (4×4, metres).
    NOTE: This is the authoritative pose used for TSDF fusion.
          VGGT-Omega poses are NOT used — see docs/architecture.md.
    """
    txt = os.path.join(scene_dir, "pose", stem + ".txt")
    rows = []
    with open(txt) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    return np.array(rows, dtype=np.float64)   # (4, 4) c2w


def load_gt_depth(scene_dir, stem):
    """Load GT depth in metres (uint16 PNG / 1000)."""
    png = os.path.join(scene_dir, "depth", stem + ".png")
    d = np.array(Image.open(png), dtype=np.float32)
    return d / DEPTH_MM_SCALE


def load_gt_intrinsics(scene_dir, sensor="depth"):
    """Return 3×3 K matrix for 'color' or 'depth' sensor."""
    txt = os.path.join(scene_dir, f"intrinsics_{sensor}.txt")
    rows = []
    with open(txt) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    K = np.array(rows, dtype=np.float64)
    return K[:3, :3]


def c2w_to_w2c(c2w):
    """Invert a 4×4 camera-to-world matrix."""
    R = c2w[:3, :3].T
    t = -R @ c2w[:3, 3]
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3]  = t
    return w2c


def pick_frames(scene_dir, n_frames):
    """Pick n evenly-spaced frames that have color + depth + pose."""
    color_stems = {os.path.splitext(f)[0]
                   for f in os.listdir(os.path.join(scene_dir, "color"))
                   if f.endswith(".jpg")}
    depth_stems = {os.path.splitext(f)[0]
                   for f in os.listdir(os.path.join(scene_dir, "depth"))
                   if f.endswith(".png")}
    pose_stems  = {os.path.splitext(f)[0]
                   for f in os.listdir(os.path.join(scene_dir, "pose"))
                   if f.endswith(".txt")}
    valid = sorted(color_stems & depth_stems & pose_stems)

    pool = valid[2:-2] if len(valid) > 10 else valid
    idx  = np.linspace(0, len(pool) - 1, min(n_frames, len(pool)), dtype=int)
    return [pool[i] for i in idx]


# ── VGGT depth prediction ─────────────────────────────────────────────────────

def predict_vggt_depth(model, device, color_paths, img_resolution=IMG_RESOLUTION):
    """
    Run VGGT-Omega on a list of colour images.
    Returns depth maps in arbitrary VGGT scale — NOT metric.
    Caller is responsible for applying VGGT_DEPTH_SCALE to convert to metres.
    """
    images   = load_and_preprocess_images(color_paths, image_resolution=img_resolution)
    images_b = images.unsqueeze(0).to(device)

    with torch.inference_mode():
        preds = model(images_b)

    depth = preds["depth"].float().cpu()[0, :, :, :, 0].numpy()   # (N, H, W)
    conf  = preds["depth_conf"].float().cpu()[0].numpy()           # (N, H, W)
    return depth, conf, images.shape[-2:]   # depth, conf, (H, W)


# ── scale estimation ──────────────────────────────────────────────────────────

def estimate_depth_scale(pred_depth_hw, gt_depth_hw):
    """
    Least-squares scale: s = argmin ||s*pred - gt||  (valid pixels only).
    This is the per-scene scale the adapter will eventually learn.
    """
    valid = (gt_depth_hw > 0) & (pred_depth_hw > 0)
    if valid.sum() < 50:
        return VGGT_DEPTH_SCALE   # fallback to baseline
    p = pred_depth_hw[valid].flatten()
    g = gt_depth_hw[valid].flatten()
    return float(np.dot(p, g) / max(np.dot(p, p), 1e-9))


# ── TSDF fusion ───────────────────────────────────────────────────────────────

def fuse_tsdf(frames_data, out_dir, tag):
    """
    Fuse a list of (rgb_path, depth_m, K, c2w_pose) tuples into a TSDF volume.

    Tries open3d first (requires Python ≤ 3.12).  Falls back to a pure-numpy
    TSDF implementation that produces a point-cloud PLY — good enough for demo
    visualisation without the open3d wheel.

    Args:
        frames_data: list of dicts with keys:
            rgb_path  : str
            depth_m   : np.ndarray (H, W) float32, metres
            K         : np.ndarray (3, 3) intrinsics
            c2w       : np.ndarray (4, 4) camera-to-world  ← GT poses
        out_dir: output directory
        tag: string label for output filenames
    """
    try:
        import open3d as o3d
        return _fuse_open3d(frames_data, out_dir, tag, o3d)
    except ImportError:
        print("  [INFO] open3d not available (no Python 3.14 wheel yet).")
        print("         Using numpy TSDF fallback → coloured point cloud PLY.")
        print("         For full mesh: run in a Python 3.11/3.12 venv with open3d.")
        return _fuse_numpy_tsdf(frames_data, out_dir, tag)


def _fuse_open3d(frames_data, out_dir, tag, o3d):
    print(f"  open3d version : {o3d.__version__}")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE,
        sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i, fd in enumerate(frames_data):
        rgb = np.array(Image.open(fd["rgb_path"]).convert("RGB"))
        H, W = fd["depth_m"].shape
        if rgb.shape[:2] != (H, W):
            rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))
        depth = fd["depth_m"].copy()
        depth[depth > DEPTH_MAX] = 0

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb.astype(np.uint8)),
            o3d.geometry.Image((depth * 1000).astype(np.uint16)),
            depth_scale=1000.0, depth_trunc=DEPTH_MAX,
            convert_rgb_to_intensity=False,
        )
        K = fd["K"]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=W, height=H, fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2])
        volume.integrate(rgbd, intrinsic, c2w_to_w2c(fd["c2w"]))
        if (i + 1) % 5 == 0 or i == 0:
            print(f"    integrated {i+1}/{len(frames_data)} frames")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    ply_path = os.path.join(out_dir, f"{tag}_mesh.ply")
    o3d.io.write_triangle_mesh(ply_path, mesh)
    print(f"  mesh saved → {ply_path}")
    print(f"  vertices: {len(mesh.vertices):,}   triangles: {len(mesh.triangles):,}")
    return ply_path


def _fuse_numpy_tsdf(frames_data, out_dir, tag):
    """
    Minimal numpy TSDF: back-project each depth frame into world space,
    accumulate coloured 3-D points, write as ASCII PLY.
    Not a true signed-distance volume — no marching cubes — but produces a
    dense coloured point cloud that visualises correctly in MeshLab / CloudCompare.
    """
    all_pts, all_cols = [], []

    for i, fd in enumerate(frames_data):
        depth = fd["depth_m"]               # (H, W)
        K     = fd["K"]                     # (3, 3)
        c2w   = fd["c2w"]                   # (4, 4)
        rgb   = np.array(Image.open(fd["rgb_path"]).convert("RGB"))

        H, W  = depth.shape
        # Resize RGB → depth resolution
        if rgb.shape[:2] != (H, W):
            rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))

        # Valid pixels
        valid = (depth > 0) & (depth < DEPTH_MAX)
        ys, xs = np.where(valid)
        zs = depth[ys, xs]

        # Back-project to camera space
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        Xc = (xs - cx) * zs / fx
        Yc = (ys - cy) * zs / fy
        Zc = zs
        pts_cam = np.stack([Xc, Yc, Zc, np.ones_like(Zc)], axis=1)  # (N, 4)

        # Transform to world space
        pts_world = (c2w @ pts_cam.T).T[:, :3]   # (N, 3)

        cols = rgb[ys, xs]   # (N, 3) uint8

        # Sub-sample to keep PLY manageable (max 50k pts per frame)
        if len(pts_world) > 50_000:
            idx = np.random.default_rng(i).choice(len(pts_world), 50_000, replace=False)
            pts_world = pts_world[idx]
            cols      = cols[idx]

        all_pts.append(pts_world)
        all_cols.append(cols)
        print(f"    back-projected frame {i+1}/{len(frames_data)}  "
              f"({len(pts_world):,} points)")

    all_pts  = np.concatenate(all_pts,  axis=0)
    all_cols = np.concatenate(all_cols, axis=0)

    # Write ASCII PLY
    ply_path = os.path.join(out_dir, f"{tag}_pointcloud.ply")
    n = len(all_pts)
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(all_pts, all_cols):
            f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} "
                    f"{col[0]} {col[1]} {col[2]}\n")

    size_mb = os.path.getsize(ply_path) / 1e6
    print(f"  point cloud saved → {ply_path}  ({size_mb:.1f} MB, {n:,} pts)")
    print(f"  Visualise with: meshlab {ply_path}")
    return ply_path


def _save_fusion_inputs(frames_data, out_dir, tag):
    """Save depth maps and pose list when open3d is unavailable."""
    import json
    poses_record = []
    for i, fd in enumerate(frames_data):
        depth_path = os.path.join(out_dir, f"{tag}_depth_{i:02d}.npy")
        np.save(depth_path, fd["depth_m"])
        poses_record.append({
            "frame": i,
            "c2w": fd["c2w"].tolist(),
            "K": fd["K"].tolist(),
        })
    with open(os.path.join(out_dir, f"{tag}_poses.json"), "w") as f:
        json.dump(poses_record, f, indent=2)
    print(f"  saved {len(frames_data)} depth maps + poses to {out_dir}/")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene",      default="scene0000_00")
    p.add_argument("--n_frames",   type=int, default=20,
                   help="number of frames to fuse (default 20)")
    p.add_argument("--use_gt_depth", action="store_true",
                   help="use ScanNet GT depth instead of VGGT-predicted depth")
    p.add_argument("--ckpt", default=os.path.join(VGGT_SRC, "checkpoints",
                                                   "vggt_omega_1b_512.pt"))
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    scene_dir = os.path.join(DATA_ROOT, args.scene)
    tag = f"{args.scene}_n{args.n_frames}_{'gtdepth' if args.use_gt_depth else 'vggtdepth'}_gtpose"

    sep("Config")
    print(f"  scene          : {args.scene}")
    print(f"  n_frames       : {args.n_frames}")
    print(f"  depth source   : {'GT (ScanNet)' if args.use_gt_depth else 'VGGT-Omega (scaled)'}")
    print(f"  pose source    : GT (ScanNet) ← see docs/architecture.md")
    print(f"  voxel size     : {VOXEL_SIZE*100:.0f} cm")
    print(f"  sdf trunc      : {SDF_TRUNC*100:.0f} cm")

    # ── frame selection ───────────────────────────────────────────────────────
    stems = pick_frames(scene_dir, args.n_frames)
    sep(f"Frames ({len(stems)} selected)")
    print(f"  {stems[0]} … {stems[-1]}")

    # ── GT intrinsics (use depth camera K, which matches depth resolution) ────
    K_depth = load_gt_intrinsics(scene_dir, sensor="depth")
    K_color = load_gt_intrinsics(scene_dir, sensor="color")
    print(f"\n  depth K  fx={K_depth[0,0]:.1f}  fy={K_depth[1,1]:.1f}  "
          f"cx={K_depth[0,2]:.1f}  cy={K_depth[1,2]:.1f}")

    # ── load GT poses (THE authoritative source for TSDF) ────────────────────
    sep("Loading GT poses  ← TSDF uses these, NOT VGGT poses")
    gt_poses = [load_gt_pose(scene_dir, s) for s in stems]
    traj_span = np.linalg.norm(gt_poses[-1][:3,3] - gt_poses[0][:3,3])
    print(f"  loaded {len(gt_poses)} GT poses")
    print(f"  trajectory span : {traj_span:.3f} m")
    print(f"  first pose t    : {gt_poses[0][:3,3]}")
    print(f"  last  pose t    : {gt_poses[-1][:3,3]}")

    # ── depth: VGGT or GT ─────────────────────────────────────────────────────
    if args.use_gt_depth:
        sep("Using GT depth (ScanNet)")
        gt_depths = [load_gt_depth(scene_dir, s) for s in stems]
        depths_m  = gt_depths
        K_for_fusion = K_depth
        depth_label = "GT depth"
    else:
        sep("Predicting depth with VGGT-Omega")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"  device : {device}")
        print("  loading model…")
        t0 = time.time()
        model = VGGTOmega()
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
        model = model.to(device).eval()
        print(f"  checkpoint loaded in {time.time()-t0:.1f}s")

        color_paths = [os.path.join(scene_dir, "color", s + ".jpg") for s in stems]
        t0 = time.time()
        vggt_depth, vggt_conf, (H, W) = predict_vggt_depth(model, device, color_paths)
        print(f"  inference done in {time.time()-t0:.1f}s  shape={vggt_depth.shape}")
        print(f"  VGGT depth range: [{vggt_depth.min():.3f}, {vggt_depth.max():.3f}]")

        # Per-frame scale estimation against GT depth
        print("\n  Estimating per-frame depth scale (VGGT → metres):")
        scales, depths_m_list = [], []
        for i, s in enumerate(stems):
            gt_d = load_gt_depth(scene_dir, s)

            # Resize VGGT depth to GT depth resolution for scale estimation
            from PIL import Image as PILImage
            vd_pil = PILImage.fromarray(vggt_depth[i]).resize(
                (gt_d.shape[1], gt_d.shape[0]), PILImage.NEAREST)
            vd_resized = np.array(vd_pil)

            scale = estimate_depth_scale(vd_resized, gt_d)
            scales.append(scale)
            # Use VGGT depth at pred resolution, scaled to metres
            depths_m_list.append(vggt_depth[i] * scale)

        scales = np.array(scales)
        print(f"  per-frame scales : {[f'{s:.3f}' for s in scales]}")
        print(f"  mean ± std       : {scales.mean():.3f} ± {scales.std():.3f}")

        depths_m = depths_m_list
        K_for_fusion = K_color   # color intrinsics match color image resolution
        depth_label = f"VGGT depth (mean scale {scales.mean():.3f}x)"

    # ── build frames_data for TSDF ────────────────────────────────────────────
    frames_data = []
    for i, s in enumerate(stems):
        frames_data.append({
            "rgb_path": os.path.join(scene_dir, "color", s + ".jpg"),
            "depth_m":  depths_m[i].astype(np.float32),
            "K":        K_for_fusion,
            "c2w":      gt_poses[i],   # ← GT POSE, always
        })

    # ── fuse ─────────────────────────────────────────────────────────────────
    sep(f"TSDF fusion  ({len(frames_data)} frames, {depth_label})")
    t0 = time.time()
    mesh = fuse_tsdf(frames_data, OUT_DIR, tag)
    print(f"  fusion time : {time.time()-t0:.1f}s")

    sep("Summary")
    print(f"  Pose source  : ScanNet GT  (ATE = 0, by definition)")
    print(f"  Depth source : {depth_label}")
    print(f"  Voxel size   : {VOXEL_SIZE*100:.0f} cm")
    print(f"  Frames fused : {len(frames_data)}")
    print(f"  Outputs in   : {OUT_DIR}/")
    if mesh is not None:
        print(f"  Mesh         : {tag}_mesh.ply")
    print()
    print("  NOTE: VGGT-Omega poses (ATE=70cm) are intentionally NOT used.")
    print("  See docs/architecture.md §Camera Pose Strategy for rationale.")


if __name__ == "__main__":
    main()
