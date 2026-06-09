"""
TSDF fusion — GT-pose pipeline.

Design decision: ScanNet GT poses are ALWAYS used for fusion.
VGGT-Omega predicted poses (ATE ≈ 70 cm) are incompatible with
5 cm voxels.  See docs/architecture.md §Camera Pose Strategy.

Two backends:
  open3d  — full marching-cubes mesh  (requires open3d; not yet on Python 3.14)
  numpy   — coloured point-cloud PLY  (fallback; always available)

Moved from: scripts/tsdf_gt_pose_fusion.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class TSDFConfig:
    """Fusion hyper-parameters — see docs/architecture.md §TSDF Fusion Parameters."""
    voxel_size:  float = 0.05   # 5 cm
    sdf_trunc:   float = 0.20   # 4 × voxel_size
    depth_max:   float = 3.5    # metres, Kinect v1 reliable range
    max_pts_per_frame: int = 50_000   # point-cloud fallback cap


# ── public API ────────────────────────────────────────────────────────────────

def fuse(
    frames_data: List[Dict],
    out_dir:     str | Path,
    tag:         str,
    config:      Optional[TSDFConfig] = None,
) -> Optional[Path]:
    """
    Fuse a list of RGB-D + pose frames into a 3D reconstruction.

    Args:
        frames_data: list of dicts, one per frame::

            {
                "rgb_path": str | Path,         # JPEG colour image
                "depth_m":  np.ndarray (H, W),  # depth in metres; 0 = invalid
                "K":        np.ndarray (3, 3),  # camera intrinsics
                "c2w":      np.ndarray (4, 4),  # camera-to-world  ← GT pose
            }

        out_dir: directory for output file(s)
        tag:     filename prefix  (e.g. 'scene0000_00_n20_gtdepth_gtpose')
        config:  TSDFConfig; defaults to 5 cm voxels

    Returns:
        Path to the output file (.ply mesh or point-cloud), or None on error.

    NOTE: Poses in ``frames_data`` must be ScanNet GT poses.
    VGGT-Omega poses are explicitly NOT supported here — the caller is
    responsible for loading them from ``pose/*.txt``.
    """
    cfg = config or TSDFConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import open3d as o3d
        return _fuse_open3d(frames_data, out_dir, tag, cfg, o3d)
    except ImportError:
        return _fuse_numpy_pointcloud(frames_data, out_dir, tag, cfg)


# ── backends ──────────────────────────────────────────────────────────────────

def _c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """Invert a 4×4 camera-to-world matrix."""
    R = c2w[:3, :3].T
    t = -R @ c2w[:3, 3]
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3]  = t
    return w2c


def _fuse_open3d(
    frames_data: List[Dict],
    out_dir:     Path,
    tag:         str,
    cfg:         TSDFConfig,
    o3d,
) -> Path:
    """Full marching-cubes mesh via open3d ScalableTSDFVolume."""
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.voxel_size,
        sdf_trunc=cfg.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i, fd in enumerate(frames_data):
        rgb = np.array(Image.open(fd["rgb_path"]).convert("RGB"))
        H, W = fd["depth_m"].shape
        if rgb.shape[:2] != (H, W):
            rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))

        depth = fd["depth_m"].copy().astype(np.float32)
        depth[depth > cfg.depth_max] = 0.0

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb.astype(np.uint8)),
            o3d.geometry.Image((depth * 1000).astype(np.uint16)),
            depth_scale=1000.0,
            depth_trunc=cfg.depth_max,
            convert_rgb_to_intensity=False,
        )
        K = fd["K"]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=W, height=H,
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        )
        volume.integrate(rgbd, intrinsic, _c2w_to_w2c(fd["c2w"]))

        if (i + 1) % 5 == 0 or i == 0:
            print(f"    [open3d] integrated {i+1}/{len(frames_data)}")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out_path = out_dir / f"{tag}_mesh.ply"
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    print(f"  mesh → {out_path}  "
          f"({len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris)")
    return out_path


def _fuse_numpy_pointcloud(
    frames_data: List[Dict],
    out_dir:     Path,
    tag:         str,
    cfg:         TSDFConfig,
) -> Path:
    """
    Fallback: back-project each frame → world-space points → ASCII PLY.

    Not a true TSDF (no marching cubes), but produces a dense coloured
    point cloud viewable in MeshLab / CloudCompare.  Used when open3d is
    unavailable (e.g. Python 3.14).
    """
    all_pts:  list[np.ndarray] = []
    all_cols: list[np.ndarray] = []

    for i, fd in enumerate(frames_data):
        depth = fd["depth_m"].astype(np.float32)
        K, c2w = fd["K"], fd["c2w"]
        rgb = np.array(Image.open(fd["rgb_path"]).convert("RGB"))

        H, W = depth.shape
        if rgb.shape[:2] != (H, W):
            rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))

        valid = (depth > 0) & (depth < cfg.depth_max)
        ys, xs = np.where(valid)
        zs = depth[ys, xs]

        # Back-project to camera space
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        pts_cam = np.stack(
            [(xs - cx) * zs / fx, (ys - cy) * zs / fy, zs, np.ones_like(zs)],
            axis=1,
        )   # (N, 4)

        # Camera → world
        pts_world = (c2w @ pts_cam.T).T[:, :3]   # (N, 3)
        cols = rgb[ys, xs]                        # (N, 3) uint8

        # Sub-sample to cap file size
        if len(pts_world) > cfg.max_pts_per_frame:
            rng = np.random.default_rng(i)
            idx = rng.choice(len(pts_world), cfg.max_pts_per_frame, replace=False)
            pts_world = pts_world[idx]
            cols      = cols[idx]

        all_pts.append(pts_world)
        all_cols.append(cols)
        print(f"    [numpy] back-projected {i+1}/{len(frames_data)}  "
              f"({len(pts_world):,} pts)")

    pts  = np.concatenate(all_pts,  axis=0)
    cols = np.concatenate(all_cols, axis=0)
    n    = len(pts)

    out_path = out_dir / f"{tag}_pointcloud.ply"
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(pts, cols):
            f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} "
                    f"{int(col[0])} {int(col[1])} {int(col[2])}\n")

    size_mb = out_path.stat().st_size / 1e6
    print(f"  point cloud → {out_path}  ({size_mb:.1f} MB, {n:,} pts)")
    return out_path
