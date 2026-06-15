#!/usr/bin/env python
"""
eval_geometry.py — Chamfer-L1, F-score@5cm, completion ratio on ScanNet val scenes.

Evaluates two methods:
  TSDF-only   partial reconstruction (no completion); no predicted points in occluded space
  Completer   OccluSynthCompleter 3D U-Net via tiled inference

Metrics are reported separately for:
  surface     state == SURFACE  (visible geometry — sanity check)
  occluded    state == OCCLUDED (hidden geometry — the key comparison)

Published Atlas / NeuralRecon numbers are cited with a "(reported, †)" tag.
These are full-scene numbers from dense video; our protocol differs — see footnote.

Output: demo_outputs/geometry_eval/results.json + stdout table.

Usage:
    .venv312/bin/python scripts/eval_geometry.py --device mps
    .venv312/bin/python scripts/eval_geometry.py --device mps --scenes scene0000_00
    .venv312/bin/python scripts/eval_geometry.py --no_completer   # TSDF-only, skips GPU
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from occlusynth.models.completer import OccluSynthCompleter
from occlusynth.utils import get_repo_root

SURFACE       = 2
OCCLUDED      = 3
SURFACE_TRUNC = 0.10    # metres — denormalises the stored SDF (from TSDFConfig)
EXTRACT_BAND  = 0.025   # 2.5 cm half-voxel band around the zero crossing
TAU           = 0.05    # F-score threshold (5 cm)
VAL_FRACTION  = 0.20

# ── Published reference numbers ───────────────────────────────────────────────
# IMPORTANT: verify these from the original papers before presenting.
# Atlas:       Murez et al., ECCV 2020, Table 1, ScanNet 3D reconstruction benchmark
# NeuralRecon: Sun et al., CVPR 2021, Table 1, ScanNet 3D reconstruction benchmark
#
# Chamfer-L1 = (Accuracy + Completeness) / 2, where Accuracy = mean d(pred→GT)
# and Completeness = mean d(GT→pred), both in cm at 5 cm threshold.
# Protocol: full-scene reconstruction from dense video — NOT comparable to our
# 6-frame partial-completion, occluded-region-only evaluation.
PUBLISHED = {
    "Atlas (ECCV 2020)†": {
        "chamfer_l1_cm":    6.5,    # approx — verify Table 1
        "fscore_5cm":       0.396,  # approx — verify Table 1
        "completion_ratio": None,
        "region":           "full scene",
    },
    "NeuralRecon (CVPR 2021)†": {
        "chamfer_l1_cm":    5.2,    # approx — verify Table 1
        "fscore_5cm":       0.434,  # approx — verify Table 1
        "completion_ratio": None,
        "region":           "full scene",
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _scene_is_val(scene_id: str) -> bool:
    h = int.from_bytes(hashlib.md5(scene_id.encode()).digest()[:2], "little")
    return h / 65536.0 < VAL_FRACTION


def run_tiled_inference(
    model: OccluSynthCompleter,
    partial_inp: np.ndarray,     # (3, nx, ny, nz) float32
    device: torch.device,
    tile_xy: int = 96,
    overlap: int = 16,
) -> np.ndarray:
    """Run completer on a full scene grid via overlapping tiles. Returns (nx,ny,nz) metres."""
    _, nx, ny, nz = partial_inp.shape
    step = tile_xy - overlap

    x_starts = list(range(0, nx, step))
    y_starts = list(range(0, ny, step))
    if nx > tile_xy and (nx - x_starts[-1]) < tile_xy:
        x_starts[-1] = nx - tile_xy
    if ny > tile_xy and (ny - y_starts[-1]) < tile_xy:
        y_starts[-1] = ny - tile_xy

    acc = np.zeros((nx, ny, nz), dtype=np.float64)
    wts = np.zeros((nx, ny, nz), dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for xs in x_starts:
            xe  = min(xs + tile_xy, nx)
            xs  = max(xs, 0)
            for ys in y_starts:
                ye  = min(ys + tile_xy, ny)
                ys_ = max(ys, 0)
                tile = np.zeros((1, 3, tile_xy, tile_xy, nz), dtype=np.float32)
                tw, th = xe - xs, ye - ys_
                tile[0, :, :tw, :th, :nz] = partial_inp[:, xs:xe, ys_:ye, :nz]
                pred = model(torch.from_numpy(tile).to(device))[0, 0].cpu().numpy()
                acc[xs:xe, ys_:ye, :nz] += pred[:tw, :th, :nz]
                wts[xs:xe, ys_:ye, :nz] += 1.0

    pos = wts > 0
    out = np.zeros((nx, ny, nz), dtype=np.float32)
    out[pos] = (acc[pos] / wts[pos]).astype(np.float32)
    return out


def extract_surface_pts(
    sdf_m: np.ndarray,
    state: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    region: int,
    band: float = EXTRACT_BAND,
) -> np.ndarray:
    """World-frame centres of voxels within `band` metres of sdf_m's zero crossing
    in the given region (SURFACE or OCCLUDED). Returns (N, 3) float32."""
    mask = (state == region) & (np.abs(sdf_m) < band)
    idx  = np.argwhere(mask)
    if len(idx) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return (origin + (idx + 0.5) * voxel_size).astype(np.float32)


def geometry_metrics(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    tau: float = TAU,
) -> dict:
    """Chamfer-L1 (cm), F-score@tau, completion_ratio (= recall@tau).

    Chamfer-L1 = (mean d(pred→GT) + mean d(GT→pred)) / 2
    F-score    = harmonic mean of precision and recall at distance tau
    completion_ratio = recall = fraction of GT surface pts within tau of any pred pt
    """
    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return {
            "chamfer_l1_cm":    None,
            "fscore_5cm":       0.0,
            "completion_ratio": 0.0,
            "precision_5cm":    0.0,
            "n_pred":           int(len(pred_pts)),
            "n_gt":             int(len(gt_pts)),
        }

    d_p2g, _ = KDTree(gt_pts).query(pred_pts)
    d_g2p, _ = KDTree(pred_pts).query(gt_pts)

    chamfer = (float(d_p2g.mean()) + float(d_g2p.mean())) / 2.0 * 100.0
    prec    = float((d_p2g < tau).mean())
    recall  = float((d_g2p < tau).mean())
    fscore  = (2.0 * prec * recall / (prec + recall)) if (prec + recall) > 0.0 else 0.0

    return {
        "chamfer_l1_cm":    round(chamfer, 3),
        "fscore_5cm":       round(fscore,  4),
        "completion_ratio": round(recall,  4),
        "precision_5cm":    round(prec,    4),
        "n_pred":           int(len(pred_pts)),
        "n_gt":             int(len(gt_pts)),
    }


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def eval_scene(
    scene_id: str,
    model: OccluSynthCompleter | None,
    device: torch.device,
    grid_dir: Path,
) -> dict | None:
    grid_path = grid_dir / f"{scene_id}_n6.npz"
    if not grid_path.exists():
        return None

    d          = np.load(grid_path, allow_pickle=False)
    origin     = d["origin"]
    vox        = float(d["voxel_size"])
    sdf_norm   = d["sdf"]           # normalised [-1, 1]
    state      = d["state"]
    gt_sdf     = d["gt_sdf"]        # metres
    partial_m  = sdf_norm * SURFACE_TRUNC   # metres; only meaningful at SURFACE voxels

    # Completer inference (tiled)
    if model is not None:
        inp   = np.stack([sdf_norm, d["weight"], d["p_observed"]])
        compl = run_tiled_inference(model, inp, device)
    else:
        compl = None

    result = {}
    for region_name, region_id in (("surface", SURFACE), ("occluded", OCCLUDED)):
        gt_pts    = extract_surface_pts(gt_sdf,   state, origin, vox, region_id)

        # TSDF-only: sdf_norm == 0 in OCCLUDED (no measurement), so no predicted surface there
        tsdf_pts  = (
            extract_surface_pts(partial_m, state, origin, vox, region_id)
            if region_id == SURFACE
            else np.zeros((0, 3), dtype=np.float32)
        )

        compl_pts = (
            extract_surface_pts(compl, state, origin, vox, region_id)
            if compl is not None
            else np.zeros((0, 3), dtype=np.float32)
        )

        result[region_name] = {
            "tsdf_only": geometry_metrics(tsdf_pts,  gt_pts),
            "completer": geometry_metrics(compl_pts, gt_pts),
            "_pts": {           # kept for aggregate pooling, not serialised
                "gt": gt_pts, "tsdf": tsdf_pts, "compl": compl_pts,
            },
        }

    return result


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _fmt(val, suffix="") -> str:
    if val is None:
        return "—"
    return f"{val:.3f}{suffix}"


def print_table(agg: dict) -> None:
    W = (30, 10, 15, 14, 14)
    hdr = (f"{'Method':<{W[0]}}{'Region':<{W[1]}}"
           f"{'Chamfer-L1↓':>{W[2]}}{'F-score@5cm↑':>{W[3]}}{'Compl.Ratio↑':>{W[4]}}")
    sep = "─" * sum(W)
    print(f"\n{hdr}\n{sep}")

    rows = [
        ("TSDF-only (no completion)", "surface",  agg["surface"]["tsdf_only"]),
        ("TSDF-only (no completion)", "occluded", agg["occluded"]["tsdf_only"]),
        ("OccluSynth Completer",      "surface",  agg["surface"]["completer"]),
        ("OccluSynth Completer",      "occluded", agg["occluded"]["completer"]),
    ]
    for method, region, m in rows:
        cham = _fmt(m["chamfer_l1_cm"], " cm")
        fsc  = _fmt(m["fscore_5cm"])
        cr   = _fmt(m["completion_ratio"])
        print(f"{method:<{W[0]}}{region:<{W[1]}}{cham:>{W[2]}}{fsc:>{W[3]}}{cr:>{W[4]}}")

    print(f"{'─── Published (†, different protocol) ' + '─' * (sum(W) - 38)}")
    for name, pub in PUBLISHED.items():
        rgn  = pub.get("region", "full scene")
        cham = _fmt(pub.get("chamfer_l1_cm"), " cm")
        fsc  = _fmt(pub.get("fscore_5cm"))
        cr   = "—"
        print(f"{name:<{W[0]}}{rgn:<{W[1]}}{cham:>{W[2]}}{fsc:>{W[3]}}{cr:>{W[4]}}")

    print(
        "\n† Published numbers: full-scene reconstruction from dense video."
        "\n  OccluSynth: 6-frame partial input, occluded-region-only evaluation."
        "\n  Published Chamfer / F-score figures are approximate — verify Table 1"
        "\n  of each paper before citing in a presentation."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",         default="checkpoints/interim_64_aug/completer_best.pt")
    ap.add_argument("--device",       default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--scenes",       nargs="*", default=None,
                    help="scene IDs (default: all val scenes in completer_grids/)")
    ap.add_argument("--no_completer", action="store_true",
                    help="skip completer inference (reports TSDF-only + published only)")
    args = ap.parse_args()

    root     = get_repo_root()
    grid_dir = root / "data" / "completer_grids"
    out_dir  = root / "demo_outputs" / "geometry_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    device   = torch.device(args.device)

    # Determine which scenes to evaluate
    if args.scenes:
        scene_ids = args.scenes
    else:
        scene_ids = sorted(
            p.stem.replace("_n6", "")
            for p in grid_dir.glob("*_n6.npz")
            if _scene_is_val(p.stem.replace("_n6", ""))
        )
    print(f"scenes ({len(scene_ids)}): {scene_ids}")

    # Load model
    model = None
    if not args.no_completer:
        ckpt_path = root / args.ckpt
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint not found at {ckpt_path}; running TSDF-only")
        else:
            ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            model = OccluSynthCompleter().to(device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            print(f"checkpoint: epoch {ckpt.get('epoch')}, "
                  f"val_loss {ckpt.get('val_loss', float('nan')):.4f}")

    # Per-scene evaluation + accumulate point clouds for micro-average aggregate
    scene_results: dict = {}
    pool: dict[str, dict[str, list]] = {
        r: {"gt": [], "tsdf": [], "compl": []}
        for r in ("surface", "occluded")
    }

    for sid in scene_ids:
        print(f"  {sid} … ", end="", flush=True)
        sr = eval_scene(sid, model, device, grid_dir)
        if sr is None:
            print("grid not found, skip")
            continue

        for rname in ("surface", "occluded"):
            pts = sr[rname].pop("_pts")      # consume; not stored in JSON
            pool[rname]["gt"].append(pts["gt"])
            pool[rname]["tsdf"].append(pts["tsdf"])
            pool[rname]["compl"].append(pts["compl"])

        of  = sr["occluded"]["tsdf_only"]["fscore_5cm"]
        ocf = sr["occluded"]["completer"]["fscore_5cm"]
        sf  = sr["surface"]["tsdf_only"]["fscore_5cm"]
        scf = sr["surface"]["completer"]["fscore_5cm"]
        print(f"surf F(tsdf/compl)={sf:.3f}/{scf:.3f}  "
              f"occ F(tsdf/compl)={of:.3f}/{ocf:.3f}")
        scene_results[sid] = sr

    if args.device == "mps":
        torch.mps.empty_cache()

    # Micro-average aggregate (pool all point clouds across scenes)
    def _stack(lst: list) -> np.ndarray:
        return np.vstack(lst) if lst else np.zeros((0, 3), np.float32)

    agg: dict = {}
    for rname in ("surface", "occluded"):
        gt_all    = _stack(pool[rname]["gt"])
        tsdf_all  = _stack(pool[rname]["tsdf"])
        compl_all = _stack(pool[rname]["compl"])
        agg[rname] = {
            "tsdf_only": geometry_metrics(tsdf_all,  gt_all),
            "completer": geometry_metrics(compl_all, gt_all),
        }

    print_table(agg)

    out = {
        "aggregate":  agg,
        "scenes":     scene_results,
        "published":  PUBLISHED,
        "_meta": {
            "ckpt":           str(args.ckpt),
            "n_scenes":       len(scene_results),
            "extract_band_m": EXTRACT_BAND,
            "tau_m":          TAU,
            "note": (
                "TSDF-only has 0 predicted points in the occluded region by construction "
                "(sdf_norm == 0 there). Published numbers use a different evaluation "
                "protocol (full-scene, dense video). Verify Table 1 of each paper."
            ),
        },
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nresults → {out_path}")


if __name__ == "__main__":
    main()
