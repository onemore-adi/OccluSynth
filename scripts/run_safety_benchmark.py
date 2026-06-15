#!/usr/bin/env python
"""
run_safety_benchmark.py — OccluSynth occlusion safety benchmark on ScanNet val scenes.

Defines the first reproducible occlusion-safety benchmark using ScanNet geometry only
(no Habitat-Sim, no semantic labels, no simulation).  Split is deterministic via the
same md5-hash used for completer training; see docs/safety_benchmark.md.

Hidden hazard definition
------------------------
  hazard = (state == OCCLUDED) AND (gt_sdf < 0)

These are voxels that are (a) behind a measured surface in every camera view
(so no line-of-sight sensor can see them) and (b) physically inside the GT mesh
(so a robot passing through would collide).  They are exactly the obstacles that
any observation-only system — no_completion or occluded_as_free — is structurally
incapable of detecting.

Metric 1 — Hazard-awareness rate
---------------------------------
  fraction of hidden hazards that the method marks as occupied (SDF < 0)

  no_completion:    SDF = 0     → awareness = 0.000  (by construction)
  occluded_as_free: SDF = +0.10 → awareness = 0.000  (by construction)
  OccluSynth:       completed_sdf < 0              → reported honestly

Metric 2 — Path collision-avoidance rate (requires Prompt 1 planner)
----------------------------------------------------------------------
  naive_path: A* with λ=0 (occluded treated as free)
  risk_path:  A* with λ=4.0 (penalises high-p_occ occluded)

  2D hazard map: projected to floor plan over the robot height band.
  avoidance = (hazards_on_naive - hazards_on_risk) / max(hazards_on_naive, 1)

Usage:
    .venv312/bin/python scripts/run_safety_benchmark.py
    .venv312/bin/python scripts/run_safety_benchmark.py --device mps
    .venv312/bin/python scripts/run_safety_benchmark.py --no_planner  # Metric 1 only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from occlusynth.models.completer import OccluSynthCompleter, predict_with_uncertainty
from occlusynth.planning import (
    PlannerConfig,
    astar,
    build_cost_map,
    farthest_free_pair,
    path_geom_length,
)
from occlusynth.utils import get_repo_root

OCCLUDED = 3
VAL_FRACTION = 0.2


# ---------------------------------------------------------------------------
# Split helper (mirrors ScanNetDataset._scene_is_val exactly)
# ---------------------------------------------------------------------------

def _scene_is_val(scene_id: str) -> bool:
    digest = hashlib.md5(scene_id.encode()).digest()
    h = int.from_bytes(digest[:2], "little")
    return h / 65536.0 < VAL_FRACTION


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------

def run_tiled_inference(
    model: OccluSynthCompleter,
    partial_inp: np.ndarray,     # (3, nx, ny, nz) float32
    device: torch.device,
    tile_xy: int = 96,
    overlap: int = 16,
) -> np.ndarray:
    """Run completer on the full scene grid via overlapping 96-voxel x-y tiles.

    z dimension is always 96 (pre-padded by generate_completer_data.py), so
    no z-tiling is needed.  Predictions are averaged over overlapping regions.

    Returns completed_sdf: (nx, ny, nz) float32.
    """
    _, nx, ny, nz = partial_inp.shape
    step = tile_xy - overlap

    x_starts = list(range(0, nx, step))
    y_starts = list(range(0, ny, step))
    # Ensure last tile covers the edge
    if nx > tile_xy and (nx - x_starts[-1]) < tile_xy:
        x_starts[-1] = nx - tile_xy
    if ny > tile_xy and (ny - y_starts[-1]) < tile_xy:
        y_starts[-1] = ny - tile_xy

    acc = np.zeros((nx, ny, nz), dtype=np.float64)
    wts = np.zeros((nx, ny, nz), dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for xs in x_starts:
            xe = min(xs + tile_xy, nx)
            xs = max(xs, 0)
            for ys in y_starts:
                ye = min(ys + tile_xy, ny)
                ys_ = max(ys, 0)

                tile = np.zeros((1, 3, tile_xy, tile_xy, nz), dtype=np.float32)
                tw = xe - xs
                th = ye - ys_
                tile[0, :, :tw, :th, :nz] = partial_inp[:, xs:xe, ys_:ye, :nz]
                t  = torch.from_numpy(tile).to(device)
                pred = model(t)[0, 0].cpu().numpy()  # (tile_xy, tile_xy, nz)

                acc[xs:xe, ys_:ye, :nz] += pred[:tw, :th, :nz]
                wts[xs:xe, ys_:ye, :nz] += 1.0

    pos = wts > 0
    out = np.zeros((nx, ny, nz), dtype=np.float32)
    out[pos] = (acc[pos] / wts[pos]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Per-scene metrics
# ---------------------------------------------------------------------------

def _hazard_awareness(completed_sdf: np.ndarray, state: np.ndarray,
                      gt_sdf: np.ndarray) -> dict:
    """Metric 1: hazard-awareness rates for all three methods."""
    hazard = (state == OCCLUDED) & (gt_sdf < 0)
    n_hazards = int(hazard.sum())
    if n_hazards == 0:
        return {"n_hazards": 0,
                "n_occluded": int((state == OCCLUDED).sum()),
                "awareness_completer": None,
                "awareness_no_completion": 0.0,
                "awareness_occluded_as_free": 0.0}

    # no_completion:    predicted SDF = 0.0  → occupied iff 0 < 0  = False
    # occluded_as_free: predicted SDF = +0.1 → occupied iff 0.1 < 0 = False
    # completer:        completed_sdf < 0
    aware_c = float((completed_sdf[hazard] < 0).mean())
    return {
        "n_hazards":               n_hazards,
        "n_occluded":              int((state == OCCLUDED).sum()),
        "awareness_completer":     aware_c,
        "awareness_no_completion": 0.0,
        "awareness_occluded_as_free": 0.0,
    }


def _planner_avoidance(
    state: np.ndarray,
    gt_sdf: np.ndarray,
    completed_sdf: np.ndarray,
    voxel_size: float,
) -> dict:
    """Metric 2: path collision-avoidance rate.

    Compares naive path (λ=0, ignores risk) vs risk-graded path (λ=4.0).
    Counts 2D floor-plan cells that are hidden hazards on the naive path
    but avoided by the risk-graded path.
    """
    cfg_naive = PlannerConfig(lambda_risk=0.0)
    cfg_risk  = PlannerConfig(lambda_risk=4.0)

    try:
        naive_cm = build_cost_map(state, completed_sdf, cfg_naive,
                                  voxel_size=voxel_size)
        risk_cm  = build_cost_map(state, completed_sdf, cfg_risk,
                                  voxel_size=voxel_size)
    except ValueError:
        return {"planner": "no_free_voxels"}

    n_traversable = np.isfinite(naive_cm).sum()
    if n_traversable < 4:
        return {"planner": "too_few_traversable"}

    try:
        start, goal = farthest_free_pair(naive_cm)
    except ValueError:
        return {"planner": "no_free_pair"}

    naive_path = astar(naive_cm, start, goal)
    risk_path  = astar(risk_cm,  start, goal)

    if naive_path is None or risk_path is None:
        return {"planner": "no_path"}

    # 2D hazard map: any OCCLUDED+GT-occupied voxel in robot height band
    # Reuse the height band already computed inside build_cost_map
    free_zs  = np.argwhere(state == 1)
    if len(free_zs) == 0:
        return {"planner": "no_floor"}
    z_floor = int(free_zs[:, 2].min())
    z_lo    = z_floor + int(round(cfg_risk.robot_height_lo / voxel_size))
    z_hi    = min(z_floor + int(round(cfg_risk.robot_height_hi / voxel_size)),
                  state.shape[2] - 1)

    band_state  = state[:, :, z_lo:z_hi + 1]
    band_gt     = gt_sdf[:, :, z_lo:z_hi + 1]
    hazard_2d   = np.any((band_state == OCCLUDED) & (band_gt < 0), axis=2)

    naive_haz  = sum(1 for c in naive_path if hazard_2d[c])
    risk_haz   = sum(1 for c in risk_path  if hazard_2d[c])
    avoided    = max(0, naive_haz - risk_haz)
    avoid_rate = avoided / max(naive_haz, 1)

    naive_len = path_geom_length(naive_path, voxel_size)
    risk_len  = path_geom_length(risk_path,  voxel_size)

    return {
        "planner":               "ok",
        "naive_path_cells":      len(naive_path),
        "risk_path_cells":       len(risk_path),
        "naive_path_m":          round(naive_len, 3),
        "risk_path_m":           round(risk_len, 3),
        "naive_hazard_cells_2d": naive_haz,
        "risk_hazard_cells_2d":  risk_haz,
        "hazards_avoided":       avoided,
        "collision_avoidance_rate": round(avoid_rate, 4),
        "start":                 list(start),
        "goal":                  list(goal),
    }


# ---------------------------------------------------------------------------
# PNG per-scene output
# ---------------------------------------------------------------------------

def _save_scene_png(
    naive_cm: np.ndarray,
    risk_cm: np.ndarray,
    hazard_2d: np.ndarray,
    naive_path: list | None,
    risk_path: list | None,
    start: tuple,
    goal: tuple,
    scene: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    nx, ny = naive_cm.shape

    for ax, cm, path, title in [
        (axes[0], naive_cm, naive_path, f"Naive path (λ=0) — {scene}"),
        (axes[1], risk_cm,  risk_path,  f"Risk-graded path (λ=4) — {scene}"),
    ]:
        vis = cm.copy()
        finite_max = vis[np.isfinite(vis)].max() if np.isfinite(vis).any() else 6.0
        vis[~np.isfinite(vis)] = finite_max + 2.0

        im = ax.imshow(vis.T, origin="lower", cmap="YlOrRd",
                       vmin=1.0, vmax=finite_max + 2.5, aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Hazard overlay (orange dots)
        haz_xy = np.argwhere(hazard_2d)
        if len(haz_xy):
            ax.scatter(haz_xy[:, 0], haz_xy[:, 1], s=2, c="orange",
                       alpha=0.4, zorder=2, label="Hidden hazard")

        if path:
            xs_ = [p[0] for p in path]
            ys_ = [p[1] for p in path]
            ax.plot(xs_, ys_, "w-", lw=1.5, zorder=3)
        ax.plot(start[0], start[1], "go", ms=8, zorder=4)
        ax.plot(goal[0],  goal[1],  "r*", ms=12, zorder=4)
        ax.set_title(title, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="OccluSynth occlusion safety benchmark"
    )
    p.add_argument("--grid_dir", default="data/completer_grids")
    p.add_argument("--ckpt", default="checkpoints/interim_64_aug/completer_best.pt")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--no_planner", action="store_true",
                   help="Skip Metric 2 (path collision-avoidance)")
    p.add_argument("--no_png",    action="store_true")
    p.add_argument("--val_only",  action="store_true", default=True,
                   help="Restrict to val scenes (default True)")
    p.add_argument("--all_scenes", action="store_true",
                   help="Run on all cached scene grids, not just val")
    args = p.parse_args()

    root    = get_repo_root()
    out_dir = root / "demo_outputs" / "safety_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = root / "docs" / "images" / "safety_benchmark"

    # ── load model ────────────────────────────────────────────────────────────
    ckpt_path = root / args.ckpt
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path}")

    device = torch.device(args.device)
    ckpt   = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model  = OccluSynthCompleter(mc_dropout=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[benchmark] checkpoint ep={ckpt.get('epoch')} "
          f"val_loss={ckpt.get('val_loss', '?'):.4f}")

    # ── enumerate scenes ──────────────────────────────────────────────────────
    grid_dir = root / args.grid_dir
    all_grids = sorted(grid_dir.glob("*_n6.npz"))

    if args.all_scenes:
        scenes = [f.name.replace("_n6.npz", "") for f in all_grids]
    else:
        scenes = [f.name.replace("_n6.npz", "") for f in all_grids
                  if _scene_is_val(f.name.replace("_n6.npz", ""))]

    print(f"[benchmark] running on {len(scenes)} {'val' if not args.all_scenes else 'all'} "
          f"scenes: {', '.join(scenes)}")
    print(f"[benchmark] device={args.device}  planner={'on' if not args.no_planner else 'off'}")

    # ── per-scene loop ────────────────────────────────────────────────────────
    scene_results = {}
    agg = {"n_hazards": 0, "aware_num": 0.0,
           "naive_haz": 0, "risk_haz": 0, "n_planner_scenes": 0}

    hdr = (f"{'Scene':<16}  {'Hazards':>9}  {'Aware-C':>8}  "
           f"{'Aware-B':>8}  {'Avoid%':>7}  {'Dims'}")
    print("\n" + hdr)
    print("─" * len(hdr))

    for scene in scenes:
        grid_file = grid_dir / f"{scene}_n6.npz"
        data = np.load(grid_file, allow_pickle=False)

        state      = data["state"]            # uint8  (nx, ny, nz)
        gt_sdf     = data["gt_sdf"]           # float32 (nx, ny, nz)
        voxel_size = float(data["voxel_size"])
        nx, ny, nz = state.shape

        partial_inp = np.stack([
            data["sdf"].astype(np.float32),
            data["weight"].astype(np.float32),
            data["p_observed"].astype(np.float32),
        ], axis=0)   # (3, nx, ny, nz)

        # ── tiled inference ────────────────────────────────────────────────
        completed_sdf = run_tiled_inference(
            model, partial_inp, device, tile_xy=96, overlap=16)

        # ── Metric 1 ───────────────────────────────────────────────────────
        m1 = _hazard_awareness(completed_sdf, state, gt_sdf)

        # ── Metric 2 ───────────────────────────────────────────────────────
        m2 = {}
        if not args.no_planner and m1["n_hazards"] > 0:
            m2 = _planner_avoidance(state, gt_sdf, completed_sdf, voxel_size)

        # ── save per-scene PNG ─────────────────────────────────────────────
        if not args.no_png and not args.no_planner and m2.get("planner") == "ok":
            cfg_n = PlannerConfig(lambda_risk=0.0)
            cfg_r = PlannerConfig(lambda_risk=4.0)
            naive_cm = build_cost_map(state, completed_sdf, cfg_n, voxel_size)
            risk_cm  = build_cost_map(state, completed_sdf, cfg_r, voxel_size)
            free_zs  = np.argwhere(state == 1)
            z_fl = int(free_zs[:, 2].min())
            z_lo = z_fl + int(round(cfg_r.robot_height_lo / voxel_size))
            z_hi = min(z_fl + int(round(cfg_r.robot_height_hi / voxel_size)),
                       nz - 1)
            band_state = state[:, :, z_lo:z_hi + 1]
            band_gt    = gt_sdf[:, :, z_lo:z_hi + 1]
            haz_2d     = np.any((band_state == OCCLUDED) & (band_gt < 0), axis=2)
            start_t    = tuple(m2["start"])
            goal_t     = tuple(m2["goal"])
            naive_path = astar(naive_cm, start_t, goal_t)
            risk_path  = astar(risk_cm,  start_t, goal_t)
            _save_scene_png(
                naive_cm, risk_cm, haz_2d, naive_path, risk_path,
                start_t, goal_t, scene,
                png_dir / f"{scene}.png"
            )

        # ── accumulate aggregates ─────────────────────────────────────────
        n_h = m1["n_hazards"]
        agg["n_hazards"] += n_h
        if m1["awareness_completer"] is not None:
            agg["aware_num"] += m1["awareness_completer"] * n_h
        if m2.get("planner") == "ok":
            agg["naive_haz"] += m2["naive_hazard_cells_2d"]
            agg["risk_haz"]  += m2["risk_hazard_cells_2d"]
            agg["n_planner_scenes"] += 1

        # ── store + print ─────────────────────────────────────────────────
        scene_results[scene] = {"split": "val", **m1, "planner": m2}

        avoid_str = (f"{m2['collision_avoidance_rate']*100:>6.1f}%"
                     if m2.get("planner") == "ok" else "   n/a ")
        aware_c_str = (f"{m1['awareness_completer']:.4f}"
                       if m1["awareness_completer"] is not None else "   N/A")
        print(f"{scene:<16}  {n_h:>9,}  {aware_c_str:>8}  "
              f"{'0.0000':>8}  {avoid_str}  {nx}×{ny}×{nz}")

    print("─" * len(hdr))

    # ── aggregate ─────────────────────────────────────────────────────────────
    tot_haz   = agg["n_hazards"]
    agg_aware = agg["aware_num"] / max(tot_haz, 1) if tot_haz else float("nan")
    agg_avoid = (max(0, agg["naive_haz"] - agg["risk_haz"])
                 / max(agg["naive_haz"], 1)) if agg["naive_haz"] > 0 else float("nan")

    avoid_agg_str = f"{agg_avoid*100:>6.1f}%" if not np.isnan(agg_avoid) else "   n/a "
    print(f"{'AGGREGATE':<16}  {tot_haz:>9,}  {agg_aware:.4f}  "
          f"{'0.0000':>8}  {avoid_agg_str}  —")

    print(f"""
Notes:
  • no_completion and occluded_as_free have 0.000 hazard awareness by construction
    (they never predict occupied SDF in the OCCLUDED region).
  • Avoid% = fraction of naive-path hidden hazards that the risk-graded path avoids.
  • Completer checkpoint: ep={ckpt.get('epoch')} (interim 64³; A100 96³ improves numbers).
  • Split: deterministic md5 hash, val_fraction=0.20 — see docs/safety_benchmark.md.
""")

    # ── save results ──────────────────────────────────────────────────────────
    results = {
        "_meta": {
            "benchmark": "occlusynth_safety_v1",
            "ckpt":      str(ckpt_path),
            "epoch":     int(ckpt.get("epoch", -1)),
            "val_loss":  float(ckpt.get("val_loss", 0.0)),
            "split":     "val (md5 hash, val_fraction=0.20)",
            "n_scenes":  len(scenes),
            "hazard_def":
                "state==OCCLUDED and gt_sdf<0 (occluded-by-observation AND inside GT mesh)",
            "baseline_awareness_note":
                "no_completion and occluded_as_free always have 0.000 awareness "
                "— they structurally cannot detect any occluded hazard",
        },
        "aggregate": {
            "n_scenes":                len(scenes),
            "n_total_hazards":         tot_haz,
            "awareness_completer":     round(agg_aware, 4) if not np.isnan(agg_aware) else None,
            "awareness_no_completion": 0.0,
            "awareness_occluded_as_free": 0.0,
            "collision_avoidance_rate":
                round(agg_avoid, 4) if not np.isnan(agg_avoid) else None,
            "n_planner_scenes":        agg["n_planner_scenes"],
        },
        "scenes": scene_results,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[benchmark] results → {out_path}")
    if not args.no_png and agg["n_planner_scenes"] > 0:
        print(f"[benchmark] PNGs → {png_dir}/")


if __name__ == "__main__":
    main()
