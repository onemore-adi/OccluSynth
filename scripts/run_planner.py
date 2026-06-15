#!/usr/bin/env python
"""
run_planner.py — risk-graded A* planner on an OccluSynth scene grid.

Loads a cached SceneGrid (.npz) produced by generate_completer_data or
run_visibility, builds a 2D cost map over the robot height band, runs A*,
and outputs:
  - PNG heatmap + path:  docs/images/planner_<scene>.png
  - Rerun .rrd:          demo_outputs/planner_<scene>.rrd   (optional)

Usage:
    .venv312/bin/python scripts/run_planner.py
    .venv312/bin/python scripts/run_planner.py --scene scene0000_00
    .venv312/bin/python scripts/run_planner.py --start 10 20 --goal 130 140
    .venv312/bin/python scripts/run_planner.py --lambda_risk 8.0 --sdf_source gt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from occlusynth.planning import (
    PlannerConfig,
    astar,
    build_cost_map,
    farthest_free_pair,
    path_geom_length,
)
from occlusynth.utils import get_repo_root


# ---------------------------------------------------------------------------
# PNG output
# ---------------------------------------------------------------------------

def _save_png(
    cost_map: np.ndarray,
    path: list,
    start: tuple,
    goal: tuple,
    out_path: Path,
    scene: str,
    voxel_size: float,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[run_planner] matplotlib not available — skipping PNG")
        return

    nx, ny = cost_map.shape
    vis = cost_map.copy()
    finite_max = vis[np.isfinite(vis)].max() if np.isfinite(vis).any() else 6.0
    vis[~np.isfinite(vis)] = finite_max + 2.0   # render walls as max+2

    fig, ax = plt.subplots(figsize=(10, 10 * ny / nx))
    # cost map: transpose so x→right, y→up; imshow uses (row=y, col=x)
    im = ax.imshow(
        vis.T,
        origin="lower",
        cmap="YlOrRd",
        aspect="equal",
        vmin=1.0,
        vmax=finite_max + 2.5,
    )
    plt.colorbar(im, ax=ax, label="Cost (inf=wall)")

    # Path overlay
    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="white", linewidth=1.5, zorder=3)

    # Start / goal markers
    ax.plot(start[0], start[1], "go", markersize=10, zorder=4, label="Start")
    ax.plot(goal[0],  goal[1],  "r*", markersize=14, zorder=4, label="Goal")

    ax.set_title(f"Risk-Graded A* — {scene}\n"
                 f"path length {path_geom_length(path, voxel_size):.2f} m  "
                 f"({len(path)} cells)")
    ax.set_xlabel("x voxel index")
    ax.set_ylabel("y voxel index")
    ax.legend(loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[run_planner] PNG saved → {out_path}")


# ---------------------------------------------------------------------------
# Rerun output
# ---------------------------------------------------------------------------

def _save_rrd(
    cost_map: np.ndarray,
    path: list,
    origin: np.ndarray,
    voxel_size: float,
    z_band_lo_idx: int,
    out_path: Path,
    scene: str,
) -> None:
    try:
        import rerun as rr
    except ImportError:
        print("[run_planner] rerun not available — skipping .rrd")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rr.init(f"occlusynth_planner_{scene}", spawn=False)
    rr.save(str(out_path))

    nx, ny = cost_map.shape
    z_world = origin[2] + (z_band_lo_idx + 0.5) * voxel_size

    # Cost map as a 2D point cloud at floor height
    finite = np.isfinite(cost_map)
    ijs = np.argwhere(finite)
    if len(ijs):
        xs = origin[0] + (ijs[:, 0] + 0.5) * voxel_size
        ys = origin[1] + (ijs[:, 1] + 0.5) * voxel_size
        zs = np.full(len(ijs), z_world)
        pts = np.stack([xs, ys, zs], axis=1)
        costs = cost_map[ijs[:, 0], ijs[:, 1]]
        # Normalise costs to [0,1] for colour mapping
        c_max = costs.max()
        c_norm = (costs - 1.0) / max(c_max - 1.0, 1e-6)
        r = (c_norm * 255).clip(0, 255).astype(np.uint8)
        g = ((1 - c_norm) * 200).clip(0, 255).astype(np.uint8)
        b = np.zeros(len(ijs), dtype=np.uint8)
        colors = np.stack([r, g, b], axis=1)
        rr.log("planner/cost_map", rr.Points3D(pts, colors=colors, radii=voxel_size * 0.45))

    # Impassable walls
    walls = np.argwhere(~finite)
    if len(walls):
        wx = origin[0] + (walls[:, 0] + 0.5) * voxel_size
        wy = origin[1] + (walls[:, 1] + 0.5) * voxel_size
        wz = np.full(len(walls), z_world)
        wpts = np.stack([wx, wy, wz], axis=1)
        rr.log("planner/walls", rr.Points3D(wpts, colors=[50, 50, 50], radii=voxel_size * 0.45))

    # Path as line strip
    if path and len(path) >= 2:
        pxs = origin[0] + (np.array([p[0] for p in path]) + 0.5) * voxel_size
        pys = origin[1] + (np.array([p[1] for p in path]) + 0.5) * voxel_size
        pzs = np.full(len(path), z_world + 0.02)
        path_pts = np.stack([pxs, pys, pzs], axis=1)
        rr.log("planner/path", rr.LineStrips3D([path_pts], colors=[[255, 255, 0]]))
        rr.log("planner/start", rr.Points3D([path_pts[0]],  colors=[[0, 220, 0]],  radii=0.08))
        rr.log("planner/goal",  rr.Points3D([path_pts[-1]], colors=[[220, 0, 0]],  radii=0.08))

    print(f"[run_planner] Rerun .rrd saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Risk-graded A* planner on OccluSynth voxel grid")
    p.add_argument("--scene",   default="scene0000_00")
    p.add_argument("--grid_dir", default="data/completer_grids")
    p.add_argument("--sdf_source", choices=["gt", "completer"], default="gt",
                   help="Which SDF to use as completed_sdf (gt=ground-truth from cache)")
    p.add_argument("--lambda_risk", type=float, default=4.0)
    p.add_argument("--robot_height_lo", type=float, default=0.10)
    p.add_argument("--robot_height_hi", type=float, default=0.50)
    p.add_argument("--start", type=int, nargs=2, metavar=("I", "J"), default=None)
    p.add_argument("--goal",  type=int, nargs=2, metavar=("I", "J"), default=None)
    p.add_argument("--use_uncertainty", action="store_true",
                   help="Use calibrated MC Dropout p_occ (requires completer checkpoint)")
    p.add_argument("--ckpt", default="checkpoints/interim_64_aug/completer_best.pt",
                   help="Completer checkpoint for --use_uncertainty")
    p.add_argument("--n_samples", type=int, default=16,
                   help="MC Dropout samples for --use_uncertainty")
    p.add_argument("--no_rrd",  action="store_true")
    p.add_argument("--no_png",  action="store_true")
    args = p.parse_args()

    root = get_repo_root()

    # ------------------------------------------------------------------
    # Load scene grid
    # ------------------------------------------------------------------
    grid_path = root / args.grid_dir / f"{args.scene}_n6.npz"
    if not grid_path.exists():
        sys.exit(f"[run_planner] cache not found: {grid_path}\n"
                 "Run scripts/run_visibility.py first to generate it.")

    print(f"[run_planner] loading {grid_path}")
    data = np.load(grid_path, allow_pickle=False)

    state        = data["state"]          # uint8  (nx, ny, nz)
    origin       = data["origin"]         # float64 (3,)
    voxel_size   = float(data["voxel_size"])
    nx, ny, nz   = state.shape

    if args.sdf_source == "gt":
        if "gt_sdf" not in data:
            sys.exit("[run_planner] gt_sdf key not found in cache. "
                     "Re-generate with scripts/generate_completer_data.py or use --sdf_source completer.")
        completed_sdf = data["gt_sdf"].astype(np.float32)
        print(f"[run_planner] using GT SDF as completed_sdf")
    else:
        # Placeholder: use partial SDF from fusion (the sdf key)
        completed_sdf = data["sdf"].astype(np.float32)
        print(f"[run_planner] using partial fusion SDF as completed_sdf "
              "(for best results load a trained completer checkpoint)")

    print(f"[run_planner] grid {nx}×{ny}×{nz}  voxel {voxel_size*100:.0f} cm  "
          f"origin {origin}")

    # ------------------------------------------------------------------
    # Optional: MC Dropout p_occ_volume
    # ------------------------------------------------------------------
    p_occ_volume = None
    if args.use_uncertainty:
        print("[run_planner] computing MC Dropout p_occ_volume ...")
        try:
            import torch
            from occlusynth.models.completer import (
                OccluSynthCompleter, predict_with_uncertainty
            )
            ckpt_path = root / args.ckpt
            if not ckpt_path.exists():
                print(f"[run_planner] WARNING: checkpoint not found at {ckpt_path}, "
                      "falling back to sdf<0 estimate")
            else:
                ckpt  = torch.load(str(ckpt_path), map_location="cpu",
                                   weights_only=False)
                model = OccluSynthCompleter(mc_dropout=True)
                model.load_state_dict(ckpt["model"])
                model.eval()

                # Tile the full grid in x and y using 96-voxel windows with 16-voxel
                # overlap; assemble a full p_occ volume (nz fixed at full depth)
                CROP = 96
                OVERLAP = 16
                STEP = CROP - OVERLAP
                p_occ_vol = np.zeros((nx, ny, nz), dtype=np.float32)
                weight_vol = np.zeros((nx, ny, nz), dtype=np.float32)

                inp_full = np.stack([
                    data["sdf"].astype(np.float32),
                    data["weight"].astype(np.float32),
                    data["p_observed"].astype(np.float32),
                ], axis=0)  # (3, nx, ny, nz)

                x_starts = list(range(0, max(nx - CROP, 0) + 1, STEP)) or [0]
                y_starts = list(range(0, max(ny - CROP, 0) + 1, STEP)) or [0]
                n_tiles  = len(x_starts) * len(y_starts)
                print(f"[run_planner] tiling {nx}×{ny} into {n_tiles} crops ...")

                with torch.no_grad():
                    for xi, xs in enumerate(x_starts):
                        for yi, ys in enumerate(y_starts):
                            xe = min(xs + CROP, nx)
                            ye = min(ys + CROP, ny)
                            xs_eff = xe - CROP if xe == nx else xs
                            ys_eff = ye - CROP if ye == ny else ys
                            # Pad z to CROP if needed
                            crop = inp_full[:, xs_eff:xs_eff + CROP,
                                           ys_eff:ys_eff + CROP, :CROP]
                            if crop.shape[1] < CROP or crop.shape[2] < CROP:
                                continue
                            z_crop = min(nz, CROP)
                            c = np.zeros((1, 3, CROP, CROP, CROP), dtype=np.float32)
                            c[0, :, :, :, :z_crop] = crop[:, :, :, :z_crop]
                            t = torch.from_numpy(c)
                            _, _, p_t = predict_with_uncertainty(
                                model, t, n_samples=args.n_samples)
                            p = p_t[0, 0, :, :, :z_crop].numpy()
                            p_occ_vol[xs_eff:xs_eff + CROP,
                                      ys_eff:ys_eff + CROP, :z_crop] += p
                            weight_vol[xs_eff:xs_eff + CROP,
                                       ys_eff:ys_eff + CROP, :z_crop] += 1.0

                pos = weight_vol > 0
                p_occ_vol[pos] /= weight_vol[pos]
                p_occ_volume = p_occ_vol
                print(f"[run_planner] p_occ_volume computed — "
                      f"mean over occluded voxels: "
                      f"{p_occ_volume[state == 3].mean():.4f}")
        except Exception as exc:
            print(f"[run_planner] WARNING: uncertainty estimation failed ({exc}), "
                  "falling back to sdf<0 estimate")

    # ------------------------------------------------------------------
    # Build cost map
    # ------------------------------------------------------------------
    cfg = PlannerConfig(
        lambda_risk=args.lambda_risk,
        robot_height_lo=args.robot_height_lo,
        robot_height_hi=args.robot_height_hi,
    )
    cost_map = build_cost_map(state, completed_sdf, cfg, voxel_size=voxel_size,
                              p_occ_volume=p_occ_volume)
    if args.use_uncertainty and p_occ_volume is not None:
        print("[run_planner] cost map uses calibrated MC Dropout p_occ")

    n_traversable = np.isfinite(cost_map).sum()
    n_wall        = (cost_map == np.inf).sum()
    print(f"[run_planner] cost map {nx}×{ny}: "
          f"traversable={n_traversable}  wall={n_wall}")

    # ------------------------------------------------------------------
    # Determine z_lo for Rerun (floor band low index)
    # ------------------------------------------------------------------
    free_zs  = np.argwhere(state == 1)
    z_floor  = int(free_zs[:, 2].min())
    z_lo_idx = z_floor + int(round(cfg.robot_height_lo / voxel_size))

    # ------------------------------------------------------------------
    # Start / goal
    # ------------------------------------------------------------------
    if args.start is not None and args.goal is not None:
        start = tuple(args.start)
        goal  = tuple(args.goal)
        for name, cell in [("start", start), ("goal", goal)]:
            ci, cj = cell
            if not (0 <= ci < nx and 0 <= cj < ny):
                sys.exit(f"[run_planner] {name}={cell} out of bounds ({nx},{ny})")
            if cost_map[ci, cj] == np.inf:
                sys.exit(f"[run_planner] {name}={cell} is a wall (cost=inf)")
    else:
        print("[run_planner] finding farthest free pair ...")
        start, goal = farthest_free_pair(cost_map)
        print(f"[run_planner] auto start={start}  goal={goal}")

    # ------------------------------------------------------------------
    # A* search
    # ------------------------------------------------------------------
    print(f"[run_planner] running A* from {start} to {goal} ...")
    path = astar(cost_map, start, goal)

    if path is None:
        sys.exit(f"[run_planner] no path found from {start} to {goal}")

    geom_len  = path_geom_length(path, voxel_size)
    path_cost = sum(
        np.sqrt((path[k+1][0]-path[k][0])**2 + (path[k+1][1]-path[k][1])**2)
        * cost_map[path[k+1]]
        for k in range(len(path)-1)
    )

    # Count occluded cells on path
    free_zs_col = np.argwhere(state == 1)
    z_hi_idx = z_floor + int(round(cfg.robot_height_hi / voxel_size))
    z_hi_idx = min(z_hi_idx, nz - 1)
    band_state = state[:, :, z_lo_idx : z_hi_idx + 1]
    n_occluded_on_path = sum(
        1 for (i, j) in path if np.any(band_state[i, j] == 3)
    )

    print(f"\n=== Planner results — {args.scene} ===")
    print(f"  Path cells   : {len(path)}")
    print(f"  Geom length  : {geom_len:.3f} m")
    print(f"  Path cost    : {path_cost:.3f}")
    print(f"  Occluded cells on path : {n_occluded_on_path} / {len(path)}")
    print(f"  lambda_risk  : {cfg.lambda_risk}")
    print(f"  Height band  : {cfg.robot_height_lo:.2f}–{cfg.robot_height_hi:.2f} m above floor")
    print(f"  z floor idx  : {z_floor}  band idx: {z_lo_idx}..{z_hi_idx}")

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    if not args.no_png:
        png_out = root / "docs" / "images" / f"planner_{args.scene}.png"
        _save_png(cost_map, path, start, goal, png_out, args.scene, voxel_size)

    if not args.no_rrd:
        rrd_out = root / "demo_outputs" / f"planner_{args.scene}.rrd"
        _save_rrd(cost_map, path, origin, voxel_size, z_lo_idx, rrd_out, args.scene)

    print("[run_planner] done.")


if __name__ == "__main__":
    main()
