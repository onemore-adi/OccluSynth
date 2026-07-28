"""
TSDF fusion — GT-pose pipeline.

Design decision: ScanNet GT poses are ALWAYS used for fusion.
VGGT-Omega predicted poses (ATE ≈ 70 cm) are incompatible with
5 cm voxels.  See docs/architecture.md §Camera Pose Strategy.

Three paths:
  fuse()             surface reconstruction —
                       open3d  → marching-cubes mesh (Python 3.12 `.venv312`)
                       numpy   → coloured point-cloud PLY (fallback)
  fuse_visibility()  visibility-aware dense voxel grid with (sdf, weight,
                       p_observed) channels.  Carves free space, marks the
                       surface band, and — crucially — distinguishes voxels
                       that are *occluded* (inside a camera frustum but behind
                       a measured surface, the completer's inpaint targets)
                       from voxels that are *out-of-frustum* (never observable,
                       leave alone).  See VisibilityVoxelGrid below.

Moved from: scripts/tsdf_gt_pose_fusion.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class TSDFConfig:
    """Fusion hyper-parameters — see docs/architecture.md §TSDF Fusion Parameters."""
    voxel_size:  float = 0.05   # 5 cm
    sdf_trunc:   float = 0.20   # 4 × voxel_size — open3d mesh (fuse()) band
    surface_trunc: float = 0.10 # 2 × voxel_size — visibility surface band, cosine-tightened
                                #   per-voxel on oblique surfaces (see integrate())
    depth_max:   float = 3.5    # metres, Kinect v1 reliable range
    depth_min:   float = 0.1    # metres, ignore closer (sensor noise / self)
    max_pts_per_frame: int = 50_000   # point-cloud fallback cap
    bbox_pad:    float = 0.15   # metres of padding around the observed bbox
    free_subsample: int = 40_000  # cap on FREE voxels emitted to PLY / viewer


# ── voxel visibility states ───────────────────────────────────────────────────

UNOBSERVABLE = 0   # never inside any valid-depth frustum  → not renderable, leave alone
FREE         = 1   # observed empty space (ray passed through in front of a surface) → green
SURFACE      = 2   # within the cosine-tightened surface band of a measured surface → red (solid geometry)
OCCLUDED     = 3   # inside a frustum but always behind a surface → amber, COMPLETER INPAINT TARGET

# green / red / amber — the demo palette (amber = uncertainty, "what the robot imagines")
CLASS_COLORS: Dict[int, tuple[int, int, int]] = {
    FREE:     ( 54, 200,  84),   # green  — confirmed empty, safe to traverse
    SURFACE:  (220,  52,  52),   # red    — measured solid geometry
    OCCLUDED: (240, 176,  48),   # amber  — hidden volume the completer must imagine
}
CLASS_NAMES: Dict[int, str] = {
    UNOBSERVABLE: "unobservable",
    FREE:         "free",
    SURFACE:      "surface",
    OCCLUDED:     "occluded",
}


# ── visibility-aware voxel grid ───────────────────────────────────────────────

@dataclass
class VisibilityResult:
    """Classified output of a VisibilityVoxelGrid."""
    centers:    np.ndarray          # (M, 3) float32 — world-space voxel centres
    labels:     np.ndarray          # (M,)   int8    — UNOBSERVABLE/FREE/SURFACE/OCCLUDED
    p_observed: np.ndarray          # (M,)   float32 — soft visibility ∈ [0, 1]
    colors:     np.ndarray          # (M, 3) uint8   — accumulated surface RGB (surface only)
    dims:       tuple               # (nx, ny, nz)
    counts:     Dict[str, int]      # voxels per class
    tsdf:       Optional[np.ndarray] = None  # (M,) float32
    weight:     Optional[np.ndarray] = None  # (M,) float32

    def mask(self, label: int) -> np.ndarray:
        return self.labels == label


def _invert_rt(c2w: np.ndarray) -> np.ndarray:
    return _c2w_to_w2c(c2w)


def _compute_normal_map(
    depth: np.ndarray,
    K:     np.ndarray,
    depth_min: float,
    depth_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-pixel surface normals in CAMERA coordinates, from depth gradients.

    Returns:
        normals (H, W, 3) float32 — unit normals (sign arbitrary; we only use |n·r|)
        valid   (H, W)    bool    — pixel + its 4-neighbours all have valid depth

    Used to correct the projective TSDF obliquity artifact: the along-ray
    distance d−z overestimates the true perpendicular distance to a tilted
    surface by 1/cosθ.  Knowing the normal lets us recover the perpendicular
    distance so the truncation band is uniform in world space.
    """
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.astype(np.float32)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    P = np.stack([x, y, z], axis=-1)                     # (H, W, 3) camera-space points

    dPdx = np.zeros_like(P)
    dPdy = np.zeros_like(P)
    dPdx[:, 1:-1, :] = P[:, 2:, :] - P[:, :-2, :]
    dPdy[1:-1, :, :] = P[2:, :, :] - P[:-2, :, :]
    n  = np.cross(dPdx, dPdy)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n  = n / np.maximum(nn, 1e-8)

    valid = (z > depth_min) & (z < depth_max)
    nb = np.zeros_like(valid)
    nb[1:-1, 1:-1] = (valid[1:-1, 1:-1]
                      & valid[1:-1, 2:] & valid[1:-1, :-2]
                      & valid[2:, 1:-1] & valid[:-2, 1:-1])
    return n.astype(np.float32), nb


class VisibilityVoxelGrid:
    """
    Dense 5 cm voxel grid with three accumulated channels per voxel:

        tsdf       truncated signed distance  (surface geometry)
        weight     TSDF fusion weight         (surface confidence)
        p_observed visibility evidence        (free + surface vs occluded)

    Integration is **projective** — the analytic, vectorised equivalent of
    per-pixel DDA ray-casting.  For every voxel we project its centre into the
    camera and compare its camera-Z against the measured depth at that pixel:

        sdf = depth_measured − z_camera                    (along the optical axis)

    Because sdf is along the optical axis, a fixed band ``|sdf| ≤ trunc`` balloons
    in world space for surfaces viewed off-normal (stretched by 1/cosθ — a grazing
    wall becomes 0.5 m thick and steals voxels from free/occluded supervision).
    We therefore **cosine-tighten** the band per voxel by the ray-to-normal angle,
    ``trunc_eff = trunc · |n·r̂|`` (clamped), so oblique surfaces get a tighter band
    and the world-space surface shell stays ~2 voxels thick everywhere:

        sdf >  +trunc_eff   →  voxel is in FRONT of the surface  →  free space
                               (a ray to that pixel passes straight through it)
        |sdf| ≤  trunc_eff  →  voxel straddles the surface        →  TSDF update
        sdf <  −trunc_eff   →  voxel is BEHIND the surface        →  occluded this frame

    with ``trunc = surface_trunc`` (2 voxels) tightened toward 1 voxel on oblique
    views.  A voxel that projects outside the image, behind the camera, or onto a
    pixel with no depth reading contributes *no evidence* for that frame.  Across all
    frames we accumulate, per voxel, how often it was seen as free / surface /
    occluded.  The final label is decided by priority surface > free > occluded,
    and a voxel that received evidence in *no* frame is UNOBSERVABLE
    (out-of-frustum) — which is what separates "hidden, go inpaint it" from
    "never observable, leave it alone".
    """

    def __init__(self, origin, dims, voxel_size, surface_trunc):
        self.origin        = np.asarray(origin, np.float64)       # (3,) grid corner
        self.dims          = tuple(int(d) for d in dims)          # (nx, ny, nz)
        self.voxel_size    = float(voxel_size)
        self.surface_trunc = float(surface_trunc)                 # world-space surface half-band

        M = int(np.prod(self.dims))
        self.tsdf   = np.ones(M,  np.float32)        # +1 = far in front (empty)
        self.weight = np.zeros(M, np.float32)
        self.color  = np.zeros((M, 3), np.float32)
        self.free   = np.zeros(M, np.float32)        # frames seen as free space
        self.surf   = np.zeros(M, np.float32)        # frames seen as surface band
        self.occ    = np.zeros(M, np.float32)        # frames seen as behind-surface
        self._centers: Optional[np.ndarray] = None

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_frames(cls, frames_data: List[Dict], cfg: TSDFConfig) -> "VisibilityVoxelGrid":
        """Size the grid to the bbox of all camera centres + back-projected surfaces."""
        lo = np.full(3, +np.inf)
        hi = np.full(3, -np.inf)
        rng = np.random.default_rng(0)
        any_surface = False

        for fd in frames_data:
            c2w  = np.asarray(fd["c2w"], np.float64)
            cam_c = c2w[:3, 3]
            lo = np.minimum(lo, cam_c)
            hi = np.maximum(hi, cam_c)

            depth = np.asarray(fd["depth_m"], np.float32)
            K     = np.asarray(fd["K"], np.float64)
            H, W  = depth.shape
            valid = (depth > cfg.depth_min) & (depth < cfg.depth_max)
            ys, xs = np.where(valid)
            if len(xs) == 0:
                continue
            any_surface = True
            if len(xs) > 4000:
                sel = rng.choice(len(xs), 4000, replace=False)
                ys, xs = ys[sel], xs[sel]
            zs = depth[ys, xs]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            pc = np.stack([(xs - cx) * zs / fx,
                           (ys - cy) * zs / fy,
                           zs, np.ones_like(zs)], axis=1)        # (n, 4)
            pw = (c2w @ pc.T).T[:, :3]
            lo = np.minimum(lo, pw.min(0))
            hi = np.maximum(hi, pw.max(0))

        if not any_surface or not np.all(np.isfinite(lo)):
            raise ValueError("No valid depth in any frame — cannot size voxel grid.")

        lo -= cfg.bbox_pad
        hi += cfg.bbox_pad
        origin = np.floor(lo / cfg.voxel_size) * cfg.voxel_size
        dims   = np.ceil((hi - origin) / cfg.voxel_size).astype(int) + 1
        return cls(origin, dims, cfg.voxel_size, cfg.surface_trunc)

    # ── geometry ──────────────────────────────────────────────────────────────

    def centers(self) -> np.ndarray:
        """(M, 3) world-space voxel centres, C-order matching the flat channels."""
        if self._centers is None:
            nx, ny, nz = self.dims
            I, J, Kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                                   indexing="ij")
            cx = self.origin[0] + (I + 0.5) * self.voxel_size
            cy = self.origin[1] + (J + 0.5) * self.voxel_size
            cz = self.origin[2] + (Kk + 0.5) * self.voxel_size
            self._centers = np.stack([cx.ravel(), cy.ravel(), cz.ravel()],
                                     axis=1).astype(np.float64)
        return self._centers

    # ── integration ───────────────────────────────────────────────────────────

    def integrate(
        self,
        depth:     np.ndarray,
        K:         np.ndarray,
        c2w:       np.ndarray,
        rgb:       Optional[np.ndarray] = None,
        depth_max: float = 3.5,
        depth_min: float = 0.1,
    ) -> None:
        """Fold one RGB-D frame into the grid (projective TSDF + visibility).

        The surface band ``|d − z| ≤ trunc`` is cosine-tightened per voxel by the
        ray-to-normal angle, so oblique surfaces keep a ~uniform world-space shell
        thickness instead of ballooning by 1/cosθ.
        """
        centers = self.centers()                        # (M, 3)
        w2c = _invert_rt(np.asarray(c2w, np.float64))
        cam = (w2c[:3, :3] @ centers.T + w2c[:3, 3:4]).T   # (M, 3) camera coords
        z   = cam[:, 2]

        H, W = depth.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        trunc = self.surface_trunc

        front = z > 1e-6
        u = np.full(len(z), -1.0); v = np.full(len(z), -1.0)
        u[front] = fx * cam[front, 0] / z[front] + cx
        v[front] = fy * cam[front, 1] / z[front] + cy

        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        in_img = (front
                  & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
                  & (z < depth_max + 1.0))            # generous margin; band is corrected below

        d = np.zeros(len(z), np.float32)
        idx = np.where(in_img)[0]
        d[idx] = depth[vi[idx], ui[idx]]
        valid = in_img & (d > depth_min) & (d < depth_max)

        # ── obliquity correction: tighten the band on oblique surfaces ─────────
        # Projective sdf = d − z is measured along the optical axis, so a fixed
        # band |d−z| ≤ trunc balloons in world space for surfaces viewed off-
        # normal (stretched by 1/cosθ — a grazing wall becomes 0.5 m thick).  We
        # scale the per-voxel band by cosθ = |n·r̂|, so oblique surfaces get a
        # proportionally tighter band and the world-space shell stays ~uniform.
        # Tightening only ever REMOVES voxels from the surface band, so noisy
        # depth-normals are safe: the worst case is the untightened ``trunc``.
        sdf = (d - z).astype(np.float32)
        trunc_eff = np.full(len(sdf), trunc, np.float32)
        nmap, nvalid = _compute_normal_map(depth, K, depth_min, depth_max)
        vidx = np.where(valid)[0]
        if len(vidx):
            cv    = cam[vidx]
            rhat  = cv / np.maximum(np.linalg.norm(cv, axis=1)[:, None], 1e-8)
            nv    = nmap[vi[vidx], ui[vidx]]
            ok    = nvalid[vi[vidx], ui[vidx]]
            cos_t = np.abs(np.einsum("ij,ij->i", nv, rhat))       # |cosθ|, ray ↔ normal
            # clamp floor (0.5) stops grazing bands collapsing into holes
            cos_t = np.where(ok, np.clip(cos_t, 0.5, 1.0), 1.0).astype(np.float32)
            trunc_eff[vidx] = trunc * cos_t

        free_m = valid & (sdf >  trunc_eff)
        surf_m = valid & (np.abs(sdf) <= trunc_eff)
        occ_m  = valid & (sdf < -trunc_eff)

        # visibility evidence
        self.free[free_m] += 1.0
        self.surf[surf_m] += 1.0
        self.occ[occ_m]   += 1.0

        # TSDF surface update (weighted running mean) on the surface band only
        if surf_m.any():
            sd    = np.clip(sdf[surf_m] / trunc_eff[surf_m], -1.0, 1.0).astype(np.float32)
            w_old = self.weight[surf_m]
            w_new = w_old + 1.0
            self.tsdf[surf_m] = (self.tsdf[surf_m] * w_old + sd) / w_new
            if rgb is not None:
                col = rgb[vi[surf_m], ui[surf_m]].astype(np.float32)
                self.color[surf_m] = (self.color[surf_m] * w_old[:, None] + col) / w_new[:, None]
            self.weight[surf_m] = w_new

    # ── classification ─────────────────────────────────────────────────────────

    def classify(self) -> VisibilityResult:
        """Collapse the accumulators into per-voxel labels + soft p_observed."""
        M = len(self.free)
        labels = np.full(M, UNOBSERVABLE, np.int8)
        # priority: surface > free > occluded  (later assignment wins)
        labels[self.occ  > 0] = OCCLUDED
        labels[self.free > 0] = FREE
        labels[self.surf > 0] = SURFACE

        total = self.free + self.surf + self.occ
        with np.errstate(invalid="ignore", divide="ignore"):
            p_obs = np.where(total > 0, (self.free + self.surf) / total, 0.0)
        p_obs = p_obs.astype(np.float32)

        colors = np.clip(self.color, 0, 255).astype(np.uint8)

        counts = {CLASS_NAMES[c]: int((labels == c).sum())
                  for c in (UNOBSERVABLE, FREE, SURFACE, OCCLUDED)}

        return VisibilityResult(
            centers=self.centers().astype(np.float32),
            labels=labels,
            p_observed=p_obs,
            colors=colors,
            dims=self.dims,
            counts=counts,
            tsdf=self.tsdf,
            weight=self.weight,
        )


def fuse_visibility(
    frames_data: List[Dict],
    out_dir:     str | Path,
    tag:         str,
    config:      Optional[TSDFConfig] = None,
    viewer=None,
) -> tuple[VisibilityResult, Path]:
    """
    Build a visibility-aware voxel grid from RGB-D + GT-pose frames.

    Args:
        frames_data: same schema as :func:`fuse` (rgb_path, depth_m, K, c2w).
        out_dir:     directory for the coloured voxel PLY.
        tag:         filename prefix.
        config:      TSDFConfig; defaults to 5 cm voxels / 20 cm truncation.
        viewer:      optional RerunViewer — if given, the three classes are
                     streamed as separate, toggleable point clouds.

    Returns:
        (VisibilityResult, path_to_ply)
    """
    cfg = config or TSDFConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = VisibilityVoxelGrid.from_frames(frames_data, cfg)
    nx, ny, nz = grid.dims
    print(f"  grid: {nx}×{ny}×{nz} = {nx*ny*nz:,} voxels @ {cfg.voxel_size*100:.0f} cm "
          f"(origin {grid.origin.round(2)})")

    for i, fd in enumerate(frames_data):
        depth = np.asarray(fd["depth_m"], np.float32)
        H, W  = depth.shape
        rgb = np.array(Image.open(fd["rgb_path"]).convert("RGB"))
        if rgb.shape[:2] != (H, W):
            rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))
        grid.integrate(depth, np.asarray(fd["K"], np.float64),
                       np.asarray(fd["c2w"], np.float64),
                       rgb=rgb, depth_max=cfg.depth_max, depth_min=cfg.depth_min)
        print(f"    [visibility] integrated {i+1}/{len(frames_data)}")

    result = grid.classify()
    c = result.counts
    obs = c["free"] + c["surface"]
    print(f"  classes: free={c['free']:,}  surface={c['surface']:,}  "
          f"occluded={c['occluded']:,}  unobservable={c['unobservable']:,}")
    if c["occluded"] + obs > 0:
        print(f"  occluded fraction of observable volume: "
              f"{c['occluded'] / max(c['occluded'] + obs, 1) * 100:.1f}%")

    ply_path = _write_visibility_ply(result, out_dir, tag, cfg)

    if viewer is not None and getattr(viewer, "enabled", False):
        viewer.log_visibility(result, frames_data, voxel_size=cfg.voxel_size,
                              free_subsample=cfg.free_subsample)

    return result, ply_path


def _write_visibility_ply(
    result: VisibilityResult,
    out_dir: Path,
    tag:     str,
    cfg:     TSDFConfig,
) -> Path:
    """Write a colour-coded voxel point cloud (green/red/amber). FREE is subsampled."""
    rng = np.random.default_rng(0)
    parts_xyz: list[np.ndarray] = []
    parts_rgb: list[np.ndarray] = []

    for label in (FREE, SURFACE, OCCLUDED):
        m = result.mask(label)
        xyz = result.centers[m]
        if label == FREE and len(xyz) > cfg.free_subsample:
            sel = rng.choice(len(xyz), cfg.free_subsample, replace=False)
            xyz = xyz[sel]
        if len(xyz) == 0:
            continue
        col = np.tile(np.array(CLASS_COLORS[label], np.uint8), (len(xyz), 1))
        parts_xyz.append(xyz)
        parts_rgb.append(col)

    pts  = np.concatenate(parts_xyz, axis=0) if parts_xyz else np.zeros((0, 3))
    cols = np.concatenate(parts_rgb, axis=0) if parts_rgb else np.zeros((0, 3), np.uint8)
    n = len(pts)

    out_path = out_dir / f"{tag}_visibility.ply"
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    lines = [f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}"
             for p, c in zip(pts, cols)]
    out_path.write_text(header + "\n".join(lines) + ("\n" if n else ""))
    print(f"  visibility cloud → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB, {n:,} pts)")
    return out_path


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
