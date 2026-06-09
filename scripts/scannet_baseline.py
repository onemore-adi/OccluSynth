#!/usr/bin/env python
"""
scannet_baseline.py — "Before adapter" VGGT-Omega baseline on ScanNet scene0000_00.

Outputs saved to demo_outputs/scannet_baseline/:
  frame_XX_rgb.jpg          – input colour frame
  frame_XX_depth_pred.png   – VGGT predicted depth (inferno colormap)
  frame_XX_depth_gt.png     – ScanNet GT depth (same colormap + same scale)
  frame_XX_depth_overlay.png– pred/GT side-by-side with error strip
  camera_trajectory.png     – predicted vs GT camera path (top-down XZ view)
  depth_scale_analysis.png  – scatter: pred depth vs GT depth with fitted scale
  baseline_report.txt       – numeric summary (scale ratio, ATE, etc.)
"""

import os, sys, time, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VGGT_SRC    = os.path.join(REPO_ROOT, "vggt", "vggt-omega")
CKPT        = os.path.join(VGGT_SRC, "checkpoints", "vggt_omega_1b_512.pt")
SCENE_DIR   = os.path.join(REPO_ROOT, "data", "scannet", "tasks",
                            "scannet_frames_25k", "scene0000_00")
OUT_DIR     = os.path.join(REPO_ROOT, "demo_outputs", "scannet_baseline")
sys.path.insert(0, VGGT_SRC)

os.makedirs(OUT_DIR, exist_ok=True)

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

# ── helpers ───────────────────────────────────────────────────────────────────
DEPTH_SCALE_MM = 1000.0   # ScanNet depth PNG is uint16 millimetres → /1000 = metres
IMG_RESOLUTION = 512

def sep(msg=""):
    w = 62
    if msg:
        print(f"\n── {msg} {'─' * max(0, w - len(msg) - 4)}")
    else:
        print("─" * w)

def load_gt_pose(frame_stem):
    txt = os.path.join(SCENE_DIR, "pose", frame_stem + ".txt")
    rows = []
    with open(txt) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    return np.array(rows, dtype=np.float32)   # 4×4 c2w

def load_gt_depth(frame_stem):
    png = os.path.join(SCENE_DIR, "depth", frame_stem + ".png")
    d = np.array(Image.open(png), dtype=np.float32)
    return d / DEPTH_SCALE_MM   # metres, 0 = invalid

def load_gt_intrinsics_color():
    txt = os.path.join(SCENE_DIR, "intrinsics_color.txt")
    rows = []
    with open(txt) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    K = np.array(rows, dtype=np.float32)
    return K[:3, :3]   # 3×3

def colorize_depth(depth_m, vmin=None, vmax=None, cmap="inferno"):
    """Return H×W×3 uint8 image; zero/invalid pixels shown in dark grey."""
    valid = depth_m > 0
    if vmin is None:
        vmin = depth_m[valid].min() if valid.any() else 0
    if vmax is None:
        vmax = depth_m[valid].max() if valid.any() else 1
    norm = np.clip((depth_m - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    rgba = cm.get_cmap(cmap)(norm)
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb[~valid] = 30   # dark grey for invalid
    return rgb

def align_depth_to_gt(pred, gt):
    """
    Compute scale s such that  s * pred ≈ gt  (least-squares, valid pixels only).
    Returns scale, inlier mask, per-pixel absolute relative error.
    """
    valid = (gt > 0) & (pred > 0)
    if valid.sum() < 50:
        return 1.0, valid, np.zeros_like(pred)
    p = pred[valid].flatten()
    g = gt[valid].flatten()
    scale = np.dot(p, g) / max(np.dot(p, p), 1e-9)
    scaled_pred = pred * scale
    rel_err = np.abs(scaled_pred - gt) / np.maximum(gt, 1e-6)
    return scale, valid, rel_err

def c2w_to_w2c(c2w):
    """Invert a 4×4 camera-to-world → world-to-camera."""
    R = c2w[:3, :3].T
    t = -R @ c2w[:3, 3]
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R
    w2c[:3, 3]  = t
    return w2c


# ─────────────────────────────────────────────────────────────────────────────
def main():
    sep("Setup")
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"  device : {device}")
    print(f"  scene  : scene0000_00")
    print(f"  ckpt   : {os.path.basename(CKPT)}")
    print(f"  outdir : {OUT_DIR}")

    # ── 1. pick 6 frames at ~evenly-spaced timestamps ────────────────────────
    sep("Frame selection")
    all_color = sorted(f for f in os.listdir(os.path.join(SCENE_DIR, "color"))
                       if f.endswith(".jpg"))
    all_depth = sorted(f for f in os.listdir(os.path.join(SCENE_DIR, "depth"))
                       if f.endswith(".png"))

    # available frame stems (those with both color + depth + pose)
    stems_color = {os.path.splitext(f)[0] for f in all_color}
    stems_depth = {os.path.splitext(f)[0] for f in all_depth}
    stems_pose  = {os.path.splitext(f)[0] for f in
                   os.listdir(os.path.join(SCENE_DIR, "pose")) if f.endswith(".txt")}
    valid_stems = sorted(stems_color & stems_depth & stems_pose)

    # pick 6 evenly spaced — skip first and last 2 to avoid boundary frames
    pool = valid_stems[2:-2] if len(valid_stems) > 10 else valid_stems
    indices = np.linspace(0, len(pool) - 1, 6, dtype=int)
    chosen  = [pool[i] for i in indices]

    print(f"  total valid frames : {len(valid_stems)}")
    print(f"  selected (6)       : {chosen}")
    # approximate time gap assuming ScanNet ~10 fps for the 25k subset spacing
    frame_nums = [int(s) for s in chosen]
    gaps = np.diff(frame_nums)
    print(f"  frame-number gaps  : {gaps}  (each ~{gaps[0]/25:.1f}s at 25fps)")

    color_paths = [os.path.join(SCENE_DIR, "color", s + ".jpg") for s in chosen]

    # ── 2. load model + checkpoint ────────────────────────────────────────────
    sep("Loading model")
    model = VGGTOmega()
    print("  loading checkpoint …")
    t0 = time.time()
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print(f"  checkpoint loaded in {time.time()-t0:.1f}s")
    model = model.to(device).eval()

    # ── 3. preprocess images ──────────────────────────────────────────────────
    sep("Preprocessing")
    images = load_and_preprocess_images(color_paths, image_resolution=IMG_RESOLUTION)
    H, W   = images.shape[-2:]
    print(f"  input tensor : {tuple(images.shape)}  (H={H}, W={W})")
    images_b = images.unsqueeze(0).to(device)   # (1, 6, 3, H, W)

    # ── 4. inference ─────────────────────────────────────────────────────────
    sep("Inference")
    t0 = time.time()
    with torch.inference_mode():
        preds = model(images_b)
    elapsed = time.time() - t0
    print(f"  forward pass : {elapsed:.2f}s  ({elapsed/6:.2f}s per frame)")

    # ── 5. decode outputs ─────────────────────────────────────────────────────
    pose_enc = preds["pose_enc"].float().cpu()          # (1, 6, 9)
    depth_b  = preds["depth"].float().cpu()             # (1, 6, H, W, 1)
    conf_b   = preds["depth_conf"].float().cpu()        # (1, 6, H, W)

    extr, intr = encoding_to_camera(pose_enc, (H, W))  # (1,6,3,4), (1,6,3,3)
    extr = extr[0].numpy()   # (6, 3, 4)  — world-to-camera (OpenCV)
    intr = intr[0].numpy()   # (6, 3, 3)

    depth_pred = depth_b[0, :, :, :, 0].numpy()        # (6, H, W)  raw predicted
    depth_conf = conf_b[0].numpy()                      # (6, H, W)

    print(f"  depth pred range : [{np.nanmin(depth_pred):.3f}, {np.nanmax(depth_pred):.3f}]")
    print(f"  conf  range      : [{depth_conf.min():.3f}, {depth_conf.max():.3f}]")

    # ── 6. load GT ─────────────────────────────────────────────────────────────
    sep("Loading GT data")
    gt_poses  = [load_gt_pose(s) for s in chosen]     # list of 4×4 c2w
    gt_depths = [load_gt_depth(s) for s in chosen]    # list of (480,640) metres
    gt_K      = load_gt_intrinsics_color()

    print(f"  GT depth range (frame 0) : [{gt_depths[0][gt_depths[0]>0].min():.3f}, "
          f"{gt_depths[0][gt_depths[0]>0].max():.3f}] m")

    # ── 7. per-frame analysis + visualisation ─────────────────────────────────
    sep("Per-frame depth analysis")

    # resize GT depth to pred resolution for comparison
    def resize_gt_depth(d_hw, target_hw):
        from PIL import Image as PILImage
        d16 = (d_hw * DEPTH_SCALE_MM).clip(0, 65535).astype(np.uint16)
        img = PILImage.fromarray(d16, mode="I;16").resize(
            (target_hw[1], target_hw[0]), PILImage.NEAREST)
        return np.array(img, dtype=np.float32) / DEPTH_SCALE_MM

    scales, rel_errs, are_inliers = [], [], []
    for i, stem in enumerate(chosen):
        dp = depth_pred[i]   # (H, W)
        gt_resized = resize_gt_depth(gt_depths[i], (H, W))

        scale, valid, rel_err = align_depth_to_gt(dp, gt_resized)
        scales.append(scale)
        # masked median abs rel error (only where GT valid and pred > 0)
        mask = valid & (dp > 0)
        median_are = float(np.median(rel_err[mask])) if mask.sum() > 0 else np.nan
        are_inliers.append(median_are)
        rel_errs.append(rel_err)
        print(f"  [{stem}]  scale={scale:.4f}  median_ARE={median_are:.4f}"
              f"  valid_px={mask.sum()}")

    # ── 8. save per-frame images ───────────────────────────────────────────────
    sep("Saving per-frame visualisations")

    # shared depth colormap range across all frames (from GT, for comparability)
    all_gt_valid = np.concatenate([d[d>0] for d in gt_depths])
    vmin_m = float(np.percentile(all_gt_valid, 2))
    vmax_m = float(np.percentile(all_gt_valid, 98))

    for i, stem in enumerate(chosen):
        idx = f"frame_{i:02d}_{stem}"

        # RGB
        rgb_img = Image.open(color_paths[i]).convert("RGB")
        rgb_img.save(os.path.join(OUT_DIR, f"{idx}_rgb.jpg"), quality=92)

        # Predicted depth (scaled to metres using per-frame scale)
        dp_scaled = depth_pred[i] * scales[i]
        pred_vis  = colorize_depth(dp_scaled, vmin=vmin_m, vmax=vmax_m)
        Image.fromarray(pred_vis).save(os.path.join(OUT_DIR, f"{idx}_depth_pred.png"))

        # GT depth (resized to pred resolution)
        gt_resized = resize_gt_depth(gt_depths[i], (H, W))
        gt_vis     = colorize_depth(gt_resized, vmin=vmin_m, vmax=vmax_m)
        Image.fromarray(gt_vis).save(os.path.join(OUT_DIR, f"{idx}_depth_gt.png"))

        # Side-by-side comparison panel
        conf_vis = depth_conf[i]
        rel_vis  = np.clip(rel_errs[i], 0, 1)

        fig = plt.figure(figsize=(18, 5))
        gs  = GridSpec(1, 5, figure=fig, wspace=0.05)

        titles  = ["RGB Input", "GT Depth", "Pred Depth (scaled)", "Depth Error |rel|", "Confidence"]
        imgs    = [np.array(rgb_img.resize((W, H))), gt_vis, pred_vis,
                   plt.cm.RdYlGn_r(rel_vis)[:, :, :3],
                   plt.cm.viridis(conf_vis / max(conf_vis.max(), 1e-6))[:, :, :3]]

        for col, (title, img_data) in enumerate(zip(titles, imgs)):
            ax = fig.add_subplot(gs[0, col])
            if img_data.dtype != np.uint8:
                img_data = (img_data * 255).astype(np.uint8)
            ax.imshow(img_data)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

        fig.suptitle(
            f"VGGT-Omega Baseline  |  scene0000_00  |  frame {stem}  "
            f"|  scale={scales[i]:.4f}  med_ARE={are_inliers[i]:.3f}",
            fontsize=11, y=1.02
        )
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"{idx}_depth_overlay.png"),
                    bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"  saved {idx}")

    # ── 9. camera trajectory plot ──────────────────────────────────────────────
    sep("Camera trajectory plot")

    # GT: c2w → camera centres in world
    gt_centres = np.array([p[:3, 3] for p in gt_poses])         # (6, 3)

    # Pred: extr is (6, 3, 4) world-to-camera.  centre = -R^T @ t
    pred_centres = np.array([
        -extr[i, :3, :3].T @ extr[i, :3, 3] for i in range(6)
    ])                                                            # (6, 3)

    # Align pred trajectory to GT (Procrustes on centroids + scale)
    gt_c   = gt_centres   - gt_centres.mean(0)
    pred_c = pred_centres - pred_centres.mean(0)
    M = gt_c.T @ pred_c
    U, S, Vt = np.linalg.svd(M)
    R_align  = U @ Vt
    s_align  = S.sum() / max(np.sum(pred_c**2), 1e-9)
    pred_aligned = s_align * (pred_c @ R_align.T) + gt_centres.mean(0)

    ate = float(np.mean(np.linalg.norm(pred_aligned - gt_centres, axis=1)))
    print(f"  ATE (after Procrustes alignment) : {ate:.4f} m")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (h_ax, v_ax, title) in zip(
        axes,
        [("X", "Z", "Top-down view (X-Z)"), ("X", "Y", "Side view (X-Y)")]
    ):
        hi = {"X": 0, "Y": 1, "Z": 2}[h_ax]
        vi = {"X": 0, "Y": 1, "Z": 2}[v_ax]

        ax.plot(gt_centres[:, hi], gt_centres[:, vi],
                "o-", color="#2ecc71", linewidth=2, markersize=8, label="GT")
        ax.plot(pred_aligned[:, hi], pred_aligned[:, vi],
                "s--", color="#e74c3c", linewidth=2, markersize=8, label="Pred (aligned)")

        for j in range(6):
            ax.annotate(str(j),
                        (gt_centres[j, hi], gt_centres[j, vi]),
                        fontsize=8, color="#27ae60", ha="center", va="bottom")
            ax.annotate(str(j),
                        (pred_aligned[j, hi], pred_aligned[j, vi]),
                        fontsize=8, color="#c0392b", ha="center", va="top")

        ax.set_xlabel(f"{h_ax} (m)"); ax.set_ylabel(f"{v_ax} (m)")
        ax.set_title(title, fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.4)
        ax.set_aspect("equal")

    fig.suptitle(
        f"Camera Trajectory  |  scene0000_00  |  ATE={ate:.4f} m",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "camera_trajectory.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  saved camera_trajectory.png")

    # ── 10. depth scale scatter plot ──────────────────────────────────────────
    sep("Depth scale analysis")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, stem in enumerate(chosen):
        ax  = axes[i]
        dp  = depth_pred[i]
        gt_resized = resize_gt_depth(gt_depths[i], (H, W))
        valid = (gt_resized > 0) & (dp > 0)

        # sub-sample for speed
        rng = np.random.default_rng(42)
        idx_flat = np.where(valid.flatten())[0]
        if len(idx_flat) > 4000:
            idx_flat = rng.choice(idx_flat, 4000, replace=False)

        pred_s = dp.flatten()[idx_flat]
        gt_s   = gt_resized.flatten()[idx_flat]

        ax.scatter(pred_s, gt_s, s=3, alpha=0.4, color="#3498db", rasterized=True)

        # fitted line
        s = scales[i]
        x_line = np.linspace(pred_s.min(), pred_s.max(), 100)
        ax.plot(x_line, s * x_line, color="#e74c3c", linewidth=1.5,
                label=f"fit  y={s:.3f}x")
        ax.plot(x_line, x_line, color="grey", linewidth=1, linestyle="--", alpha=0.5,
                label="identity")

        ax.set_xlabel("Pred depth (raw)", fontsize=8)
        ax.set_ylabel("GT depth (m)", fontsize=8)
        ax.set_title(f"{stem}  scale={s:.4f}  ARE={are_inliers[i]:.3f}",
                     fontsize=8, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Depth Scale Analysis: predicted vs GT depth\n"
                 "(scale factor = slope of best-fit line)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "depth_scale_analysis.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  saved depth_scale_analysis.png")

    # ── 11. text report ───────────────────────────────────────────────────────
    sep("Writing report")

    scale_mean   = float(np.mean(scales))
    scale_std    = float(np.std(scales))
    are_mean     = float(np.nanmean(are_inliers))
    are_std      = float(np.nanstd(are_inliers))

    # predicted intrinsics vs GT
    pred_fx = float(intr[0, 0, 0])
    pred_fy = float(intr[0, 1, 1])
    gt_fx   = float(gt_K[0, 0])
    gt_fy   = float(gt_K[1, 1])

    report_lines = [
        "=" * 62,
        "  VGGT-Omega ScanNet Baseline Report — scene0000_00",
        "=" * 62,
        "",
        f"  Date           : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Checkpoint     : vggt_omega_1b_512.pt",
        f"  Device         : {device}",
        f"  Frames used    : {chosen}",
        f"  Image res      : {H}×{W}",
        "",
        "── Depth Scale (pred units → metres) ─────────────────",
        f"  Per-frame scales : {[f'{s:.4f}' for s in scales]}",
        f"  Mean scale       : {scale_mean:.4f}  ± {scale_std:.4f}",
        f"  Interpretation   : multiply pred depth by {scale_mean:.3f} to get metres",
        "",
        "── Depth Accuracy (after scale alignment) ─────────────",
        f"  Per-frame median ARE : {[f'{e:.4f}' for e in are_inliers]}",
        f"  Mean median ARE      : {are_mean:.4f}  ± {are_std:.4f}",
        f"  (ARE < 0.10 is good; < 0.20 is usable for indoor scenes)",
        "",
        "── Camera Pose Accuracy ────────────────────────────────",
        f"  ATE (Procrustes-aligned) : {ate:.4f} m",
        f"  GT translation range     : "
        f"{np.linalg.norm(gt_centres[-1]-gt_centres[0]):.3f} m (start→end)",
        "",
        "── Intrinsics Comparison ───────────────────────────────",
        f"  GT   fx / fy  : {gt_fx:.1f} / {gt_fy:.1f}",
        f"  Pred fx / fy  : {pred_fx:.1f} / {pred_fy:.1f}  (frame 0)",
        f"  fx ratio      : {pred_fx/max(gt_fx,1):.4f}   fy ratio: {pred_fy/max(gt_fy,1):.4f}",
        "",
        "── Output Files ────────────────────────────────────────",
        "  frame_XX_rgb.jpg            input colour frames",
        "  frame_XX_depth_pred.png     VGGT predicted depth (inferno)",
        "  frame_XX_depth_gt.png       ScanNet GT depth (same scale)",
        "  frame_XX_depth_overlay.png  5-panel: RGB / GT / Pred / Error / Conf",
        "  camera_trajectory.png       GT vs predicted path (2 views)",
        "  depth_scale_analysis.png    scatter pred vs GT with fitted scale",
        "",
        "── OccluSynth Adapter Notes ────────────────────────────",
        f"  → Predicted depth is in arbitrary scale; adapter should learn",
        f"    the {scale_mean:.3f}× correction automatically.",
        f"  → ARE of {are_mean:.3f} is the baseline to beat after adapter training.",
        f"  → Pose ATE of {ate:.3f} m is the camera tracking baseline.",
        "=" * 62,
    ]

    report_txt = "\n".join(report_lines)
    print(report_txt)

    with open(os.path.join(OUT_DIR, "baseline_report.txt"), "w") as f:
        f.write(report_txt + "\n")
    print(f"\n  report saved → {OUT_DIR}/baseline_report.txt")

    sep("Done")
    print(f"  All outputs in: {OUT_DIR}")
    all_files = sorted(os.listdir(OUT_DIR))
    for fn in all_files:
        sz = os.path.getsize(os.path.join(OUT_DIR, fn))
        print(f"    {fn:<45}  {sz/1024:6.1f} KB")


if __name__ == "__main__":
    main()
