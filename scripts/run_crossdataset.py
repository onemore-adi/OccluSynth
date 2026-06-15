#!/usr/bin/env python
"""
run_crossdataset.py — cross-dataset generalisation probe on 7-Scenes.

Demonstrates that the frozen OccluSynth pipeline (RANSAC depth grounding +
fuse_visibility) generalises to the 7-Scenes dataset [Shotton et al., CVPR 2013]
without any retraining.  The 3D U-Net completer is NOT run here — only the
upstream pipeline (depth calibration, visibility voxel grid) is evaluated.

Depth grounding strategy
------------------------
7-Scenes depth maps are stored as uint16 millimetres; they are treated as
"uncalibrated" raw depth and fed into the same per-frame RANSAC fit used for
VGGT predictions on ScanNet.  RANSAC recovers a ≈ 0.001 (the mm→m factor),
confirming the calibration is dataset-agnostic.  After correction, ARE ≈ 0
(depth is from the same Kinect sensor, so there is no real scale ambiguity).
This is an honest sanity check: a real cross-dataset use case would supply
predictions from an uncalibrated source (e.g. monocular depth estimation).

Output
------
  demo_outputs/crossdataset/<scene>_<seq>/  PLY + .rrd
  docs/images/crossdataset_<scene>_<seq>.png
  demo_outputs/crossdataset/results.json

Usage:
    .venv312/bin/python scripts/run_crossdataset.py
    .venv312/bin/python scripts/run_crossdataset.py \\
        --seqs chess/seq-01 fire/seq-01 \\
        --data_root data/7scenes \\
        --n_frames 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from occlusynth.data.sevenscenes import SevenScenesSequence
from occlusynth.models.depth_calibration import evaluate
from occlusynth.models.metric_grounding import (
    apply_metric_correction,
    fit_metric_depth,
)
from occlusynth.fusion import FREE, SURFACE, OCCLUDED, CLASS_COLORS
from occlusynth.fusion.tsdf import TSDFConfig, fuse_visibility
from occlusynth.utils import get_repo_root


# ---------------------------------------------------------------------------
# Depth grounding
# ---------------------------------------------------------------------------

def ground_sequence(seq: SevenScenesSequence, stems: list[str]) -> dict:
    """
    Fit per-frame RANSAC (a, b) using raw-mm depth as 'uncalibrated predictions'
    and GT-metres depth as reference.  Returns per-frame params + aggregate metrics.
    """
    raw_frames = np.stack([seq.depth_raw(s) for s in stems])   # mm, float32
    gt_frames  = np.stack([seq.depth(s)     for s in stems])   # metres, float32

    params = fit_metric_depth(raw_frames, gt_frames, stems)

    # Evaluate over all frames
    are_list, delta_list = [], []
    for i, s in enumerate(stems):
        a, b = params[s]
        m = evaluate(raw_frames[i], gt_frames[i], a, b)
        if m["n_valid"] > 0 and np.isfinite(m["ARE"]):
            are_list.append(m["ARE"])
            delta_list.append(m["delta_125"])

    return {
        "params": {s: list(params[s]) for s in stems},
        "mean_ARE":      float(np.mean(are_list))      if are_list  else float("nan"),
        "mean_delta125": float(np.mean(delta_list))    if delta_list else float("nan"),
        "n_frames":      len(stems),
    }


# ---------------------------------------------------------------------------
# Visibility fusion
# ---------------------------------------------------------------------------

def build_frames_7scenes(
    seq: SevenScenesSequence,
    stems: list[str],
    params: dict[str, list],
) -> list[dict]:
    """Build the fuse_visibility frame schema from calibrated 7-Scenes depth."""
    K = seq.K   # (3, 3) fixed
    frames = []
    for s in stems:
        a, b    = params[s]
        raw_mm  = seq.depth_raw(s)
        depth_m = apply_metric_correction(raw_mm, a, b)
        frames.append({
            "rgb_path": seq.color_path(s),
            "depth_m":  depth_m,
            "K":        K,
            "c2w":      seq.pose(s),
        })
    return frames


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def cross_section_png(result, out_png: Path, seq: SevenScenesSequence,
                      frames: list[dict]) -> None:
    """Top-down cross-section: green=free, red=surface, amber=occluded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    cen, lab = result.centers, result.labels

    def col(label):
        return np.array(CLASS_COLORS[label]) / 255.0

    fig, ax = plt.subplots(figsize=(8, 8))

    surf_all = cen[lab == SURFACE]
    z_mid = float(np.median(surf_all[:, 2])) if len(surf_all) else float(np.median(cen[:, 2]))
    band  = np.abs(cen[:, 2] - z_mid) < 0.12

    for label, name, alpha, s in [
        (FREE,     "free (observed empty)",      0.25,  8),
        (OCCLUDED, "occluded (blind spot)",       0.80, 14),
        (SURFACE,  "surface (measured)",          1.00, 16),
    ]:
        pts = cen[band & (lab == label)]
        if len(pts) > 30_000:
            pts = pts[rng.choice(len(pts), 30_000, replace=False)]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=s, c=[col(label)],
                       alpha=alpha, label=name, edgecolors="none")

    cams = np.array([fd["c2w"][:3, 3] for fd in frames])
    ax.scatter(cams[:, 0], cams[:, 1], marker="^", s=90, c="black",
               label="cameras", zorder=5)

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    c = result.counts
    obs = c["free"] + c["surface"]
    occ_frac = c["occluded"] / max(c["occluded"] + obs, 1) * 100
    ax.set_title(
        f"7-Scenes · {seq.sequence_id} · top-down @ z={z_mid:.2f} m\n"
        f"free {c['free']:,}  surface {c['surface']:,}  "
        f"occluded {c['occluded']:,} ({occ_frac:.0f}% of observable)  "
        f"unobs {c['unobservable']:,}",
        fontsize=9,
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  PNG  → {out_png}")


# ---------------------------------------------------------------------------
# Per-sequence pipeline
# ---------------------------------------------------------------------------

def run_sequence(
    seq: SevenScenesSequence,
    n_frames: int,
    out_root: Path,
    no_rerun: bool,
) -> dict:
    tag = f"{seq.scene_name}_{seq.seq_name}"
    print(f"\n{'─'*60}")
    print(f"Sequence : {seq.sequence_id}  ({seq})")

    stems = seq.pick_frames(n_frames)
    print(f"Frames   : {len(stems)}  ({stems[0]} … {stems[-1]})")

    # ── depth grounding ──────────────────────────────────────────────────────
    print("Grounding depth (RANSAC mm→m) …")
    ground = ground_sequence(seq, stems)
    a_vals = [ground["params"][s][0] for s in stems]
    b_vals = [ground["params"][s][1] for s in stems]
    print(f"  mean a = {np.mean(a_vals):.5f}  (expected ≈0.001 = mm→m)")
    print(f"  mean b = {np.mean(b_vals):.5f} m")
    print(f"  ARE    = {ground['mean_ARE']:.4f}   δ<1.25 = {ground['mean_delta125']:.4f}")

    # ── visibility fusion ────────────────────────────────────────────────────
    frames = build_frames_7scenes(seq, stems, ground["params"])

    seq_out = out_root / tag
    viewer  = None
    if not no_rerun:
        from occlusynth.viz.rerun_viewer import RerunViewer
        rrd_path = seq_out / f"{tag}.rrd"
        viewer = RerunViewer(f"occlusynth-7scenes-{tag}", spawn=False,
                             save_path=str(rrd_path))

    cfg = TSDFConfig(voxel_size=0.05, sdf_trunc=0.20, surface_trunc=0.10)
    result, ply = fuse_visibility(frames, seq_out, tag, cfg, viewer=viewer)

    if not no_rerun and viewer is not None:
        print(f"  RRD  → {seq_out / (tag + '.rrd')}   (open with: rerun <path>)")

    cross_section_png(result, ROOT / "docs/images" / f"crossdataset_{tag}.png",
                      seq, frames)

    c = result.counts
    return {
        "sequence_id": seq.sequence_id,
        "n_frames":    len(stems),
        "grounding":   ground,
        "voxel_counts": c,
        "ply": str(ply),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--seqs", nargs="+",
        default=["chess/seq-01", "fire/seq-01"],
        help="sequences relative to data_root, e.g. chess/seq-01 fire/seq-01",
    )
    ap.add_argument("--data_root", default="data/7scenes",
                    help="path to the 7-Scenes root (default: data/7scenes)")
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--no_rerun", action="store_true",
                    help="skip Rerun logging (PNG + PLY only)")
    args = ap.parse_args()

    root      = get_repo_root()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    out_root  = root / "demo_outputs" / "crossdataset"
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = []

    for rel in args.seqs:
        scene, seq_name = rel.split("/", 1)
        seq_dir = data_root / scene / seq_name
        if not seq_dir.exists():
            print(f"[SKIP] {seq_dir} not found — download 7-Scenes first "
                  f"(see README.md §7-Scenes data).")
            continue

        seq = SevenScenesSequence(seq_dir)
        r   = run_sequence(seq, args.n_frames, out_root, args.no_rerun)
        all_results.append(r)

    if not all_results:
        print("\nNo sequences processed. Check --data_root and --seqs.")
        return

    # ── summary table ────────────────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print(f"{'Sequence':<22} {'ARE↓':>8} {'δ<1.25↑':>9} "
          f"{'free':>8} {'surf':>8} {'occ':>8} {'occ%':>7}")
    print("─" * 75)
    for r in all_results:
        g = r["grounding"]
        c = r["voxel_counts"]
        obs  = c["free"] + c["surface"]
        frac = c["occluded"] / max(c["occluded"] + obs, 1) * 100
        print(f"{r['sequence_id']:<22} {g['mean_ARE']:>8.4f} "
              f"{g['mean_delta125']:>9.4f} "
              f"{c['free']:>8,} {c['surface']:>8,} "
              f"{c['occluded']:>8,} {frac:>6.1f}%")

    out_json = out_root / "results.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nresults → {out_json}")


if __name__ == "__main__":
    main()
