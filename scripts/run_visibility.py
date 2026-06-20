#!/usr/bin/env python
"""
run_visibility.py — visibility-aware voxel fusion + the green/red/amber demo.

Builds a dense 5 cm voxel grid with (sdf, weight, p_observed) channels from
ScanNet GT poses, classifies every voxel as free / surface / occluded /
unobservable, then:

  1. writes a colour-coded voxel PLY  (demo_outputs/visibility/)
  2. streams it to Rerun as three toggleable clouds + camera frusta (.rrd)
  3. renders a headless matplotlib "money shot" PNG (docs/images/)

Usage:
    python scripts/run_visibility.py --scene scene0000_00 --use_gt_depth
    python scripts/run_visibility.py --scene scene0000_00            # VGGT depth
    python scripts/run_visibility.py --scene scene0000_00 --no_rerun  # PNG only

Run with the open3d/rerun env:  .venv312/bin/python scripts/run_visibility.py ...
"""

import argparse
from pathlib import Path

import numpy as np

from occlusynth.data    import ScanNetDataset
from occlusynth.models  import (VGGTWrapper, load_grounding, ground_scene,
                                apply_metric_correction)
from occlusynth.models.depth_calibration import resize_to_gt
from occlusynth.fusion  import (TSDFConfig, fuse_visibility,
                                FREE, SURFACE, OCCLUDED, CLASS_COLORS)
from occlusynth.utils   import get_repo_root


def build_frames(scene, n_frames, use_gt_depth):
    """Return (frames_data, scene_dir) — same schema fuse_visibility expects."""
    root          = get_repo_root()
    grounding_dir = root / "demo_outputs/grounding"
    cache_dir     = root / "demo_outputs/pred_cache"

    dataset = ScanNetDataset(n_frames=n_frames, split="all")
    item    = dataset[dataset.scenes.index(scene)]
    stems   = item["frame_idx"]
    gt_depth = item["depth_gt"].numpy()
    poses    = item["pose"].numpy()
    K        = item["intrinsics"][0].numpy()
    scene_dir = dataset.root / scene

    if use_gt_depth:
        depths = [gt_depth[i] for i in range(len(stems))]
    else:
        wrapper = VGGTWrapper()
        result  = wrapper.predict_cached(scene, stems, scene_dir, cache_dir=cache_dir)
        vggt_raw = result["depth"]
        json_path = grounding_dir / f"{scene}_grounding.json"
        if not json_path.exists():
            json_path = ground_scene(scene, dataset, wrapper,
                                     out_dir=grounding_dir, cache_dir=cache_dir)
        params = load_grounding(json_path)
        depths = []
        for i, stem in enumerate(stems):
            a, b = params[stem]
            depths.append(apply_metric_correction(resize_to_gt(vggt_raw[i], gt_depth[i]), a, b))

    frames = [{
        "rgb_path": scene_dir / "color" / f"{stem}.jpg",
        "depth_m":  depths[i],
        "K":        K,
        "c2w":      poses[i],
    } for i, stem in enumerate(stems)]
    return frames, stems


def money_shot_png(result, out_png, scene, use_gt_depth, frames=None):
    """
    Headless money shot: a top-down cross-section (the canonical occlusion
    figure) beside a 3D view.  green=free, red=surface, amber=occluded.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    cen, lab = result.centers, result.labels

    def col(label):
        return np.array(CLASS_COLORS[label]) / 255.0

    def sub(xyz, cap):
        return xyz[rng.choice(len(xyz), cap, replace=False)] if len(xyz) > cap else xyz

    fig = plt.figure(figsize=(15, 7))

    # ── Panel 1: horizontal cross-section at surface mid-height ────────────────
    surf_all = cen[lab == SURFACE]
    z_mid = float(np.median(surf_all[:, 2])) if len(surf_all) else float(np.median(cen[:, 2]))
    band  = np.abs(cen[:, 2] - z_mid) < 0.12          # ~5-voxel slab
    ax1 = fig.add_subplot(1, 2, 1)
    for label, name, a, s in [
        (FREE,     "free (observed empty)",   0.30, 8),
        (OCCLUDED, "occluded (inpaint target)", 0.85, 14),
        (SURFACE,  "surface (measured)",      1.0,  16),
    ]:
        pts = cen[band & (lab == label)]
        if len(pts):
            ax1.scatter(pts[:, 0], pts[:, 1], s=s, c=[col(label)], alpha=a,
                        label=name, edgecolors="none")
    if frames is not None:
        cams = np.array([fd["c2w"][:3, 3] for fd in frames])
        ax1.scatter(cams[:, 0], cams[:, 1], marker="^", s=90, c="black",
                    label="cameras", zorder=5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax1.set_title(f"top-down cross-section @ z={z_mid:.2f} m", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, markerscale=1.3)

    # ── Panel 2: 3D — measured surface + occlusion shadow ─────────────────────
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    occ  = sub(cen[lab == OCCLUDED], 30000)
    surf = sub(cen[lab == SURFACE],  30000)
    # red surface reads as the solid structure; amber occlusion = the imagined volume behind it
    ax2.scatter(*occ.T,  s=5, c=[col(OCCLUDED)], alpha=0.15)
    ax2.scatter(*surf.T, s=9, c=[col(SURFACE)],  alpha=0.95)
    ax2.view_init(elev=20, azim=-62)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    ax2.set_title("3D: surface + occlusion shadow", fontsize=10)
    ax2.set_box_aspect((1, 1, 1))

    c = result.counts
    src = "GT depth" if use_gt_depth else "VGGT RANSAC depth"
    occ_frac = c["occluded"] / max(c["occluded"] + c["free"] + c["surface"], 1) * 100
    fig.suptitle(
        f"OccluSynth visibility voxels — {scene} ({src})\n"
        f"free {c['free']:,}   surface {c['surface']:,}   "
        f"occluded {c['occluded']:,} ({occ_frac:.0f}% of observable)   "
        f"unobservable {c['unobservable']:,}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  money shot → {out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene",        default="scene0000_00")
    p.add_argument("--n_frames",     type=int, default=6)
    p.add_argument("--use_gt_depth", action="store_true",
                   help="use ScanNet GT depth (crispest occlusion; skips VGGT)")
    p.add_argument("--no_rerun",     action="store_true",
                   help="skip Rerun logging, write PLY + PNG only")
    p.add_argument("--voxel",        type=float, default=0.05)
    args = p.parse_args()

    root    = get_repo_root()
    out_dir = root / "demo_outputs/visibility"
    tag     = f"{args.scene}_n{args.n_frames}_{'gt' if args.use_gt_depth else 'vggt'}"

    print(f"Scene        : {args.scene}")
    print(f"Depth source : {'GT (ScanNet)' if args.use_gt_depth else 'VGGT RANSAC'}")
    print(f"Pose source  : GT (ScanNet)")

    frames, stems = build_frames(args.scene, args.n_frames, args.use_gt_depth)
    print(f"Frames       : {len(frames)}  ({stems[0]} … {stems[-1]})")

    viewer = None
    if not args.no_rerun:
        from occlusynth.viz.rerun_viewer import RerunViewer
        rrd = out_dir / f"{tag}.rrd"
        viewer = RerunViewer("occlusynth-visibility", spawn=False, save_path=str(rrd))

    cfg = TSDFConfig(voxel_size=args.voxel,
                     sdf_trunc=4 * args.voxel,
                     surface_trunc=2 * args.voxel)   # 2× voxel, cosine-tightened on oblique surfaces
    result, ply = fuse_visibility(frames, out_dir, tag, cfg, viewer=viewer)

    money_shot_png(result, root / "docs/images" / f"visibility_{tag}.png",
                   args.scene, args.use_gt_depth, frames=frames)

    print("\nDone.")
    print(f"  PLY  : {ply}")
    if viewer is not None:
        print(f"  RRD  : {out_dir / (tag + '.rrd')}   (open with: rerun <path>)")


if __name__ == "__main__":
    main()
