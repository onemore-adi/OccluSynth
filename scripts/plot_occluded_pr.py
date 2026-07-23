#!/usr/bin/env python
"""
plot_occluded_pr.py — precision–recall curve for occluded-region occupancy.

The "how we chose the operating point" artifact: sweep the SDF threshold τ used
to call an occluded voxel solid (completed_sdf < τ) and score against ground
truth (gt_sdf < 0), pooled over the 10 held-out val scenes. τ = 0 (the natural
sign threshold, used by the safety benchmark) is marked as our operating point.

Voxel-level occupancy PR is the safety-relevant view (it matches the hidden-
hazard definition in run_safety_benchmark.py); the surface F-score@5cm in
eval_geometry.py is the reconstruction-quality view. Quote them separately.

Usage:
    .venv312/bin/python scripts/plot_occluded_pr.py [--device mps]

Outputs:
    demo_outputs/pr_curve/results.json
    docs/images/occluded_pr_curve.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from occlusynth.models.completer import OccluSynthCompleter
from occlusynth.utils import get_repo_root
from run_safety_benchmark import _scene_is_val, run_tiled_inference

OCCLUDED = 3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid_dir", default="data/completer_grids")
    p.add_argument("--ckpt", default="checkpoints/interim_64_aug/completer_best.pt")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--n_thresholds", type=int, default=61)
    args = p.parse_args()

    root = get_repo_root()
    device = torch.device(args.device)
    ckpt = torch.load(str(root / args.ckpt), map_location="cpu", weights_only=False)
    model = OccluSynthCompleter(mc_dropout=False).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[pr] ckpt ep={ckpt.get('epoch')}")

    grid_dir = root / args.grid_dir
    scenes = [f.name.replace("_n6.npz", "") for f in sorted(grid_dir.glob("*_n6.npz"))
              if _scene_is_val(f.name.replace("_n6.npz", ""))]
    print(f"[pr] {len(scenes)} val scenes: {', '.join(scenes)}")

    preds, labels = [], []
    for scene in scenes:
        d = np.load(grid_dir / f"{scene}_n6.npz", allow_pickle=False)
        occ = d["state"] == OCCLUDED
        inp = np.stack([d["sdf"].astype(np.float32),
                        d["weight"].astype(np.float32),
                        d["p_observed"].astype(np.float32)])
        compl = run_tiled_inference(model, inp, device)
        preds.append(compl[occ])
        labels.append(d["gt_sdf"][occ] < 0)
        print(f"  {scene}: {occ.sum():>8d} occluded voxels, "
              f"{labels[-1].sum():>8d} solid in GT")

    pred = np.concatenate(preds)
    lab = np.concatenate(labels)
    n_pos = int(lab.sum())
    print(f"[pr] pooled: {len(lab)} occluded voxels, {n_pos} GT-solid "
          f"({100 * n_pos / len(lab):.1f}%)")

    taus = np.linspace(-0.15, 0.30, args.n_thresholds)
    rows = []
    for tau in taus:
        hit = pred < tau
        tp = int((hit & lab).sum())
        prec = tp / max(int(hit.sum()), 1)
        rec = tp / max(n_pos, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append({"tau_m": round(float(tau), 4), "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4)})

    op = min(rows, key=lambda r: abs(r["tau_m"]))          # tau = 0
    best = max(rows, key=lambda r: r["f1"])

    out_dir = root / "demo_outputs" / "pr_curve"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({
        "_meta": {"ckpt": args.ckpt, "epoch": ckpt.get("epoch"),
                  "n_scenes": len(scenes),
                  "n_occluded_voxels": int(len(lab)), "n_gt_solid": n_pos,
                  "definition": "predicted solid = completed_sdf < tau; "
                                "positive = occluded voxel with gt_sdf < 0"},
        "operating_point_tau0": op, "best_f1": best, "curve": rows,
    }, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="#0E1116")
    ax.set_facecolor("#0E1116")
    rec = [r["recall"] for r in rows]
    prc = [r["precision"] for r in rows]
    ax.plot(rec, prc, color="#E0A100", lw=2.5, label="OccluSynth (sweep of τ)")
    ax.scatter([op["recall"]], [op["precision"]], s=90, zorder=5,
               color="#FFFFFF", edgecolor="#E0A100", lw=2,
               label=f"operating point τ=0  (P={op['precision']:.2f}, R={op['recall']:.2f})")
    ax.scatter([0], [0], s=90, marker="x", color="#C0272D", lw=2.5,
               label="any observation-only system (P=0, R=0)")
    base_rate = n_pos / len(lab)
    ax.axhline(base_rate, color="#5A6472", ls="--", lw=1.2)
    ax.text(0.99, base_rate + 0.012, f"chance precision = {base_rate:.2f}",
            color="#8B95A5", ha="right", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#5A6472")
    ax.tick_params(colors="#C9CDD3")
    ax.set_xlabel("Recall — share of hidden solid geometry detected", color="#C9CDD3")
    ax.set_ylabel("Precision — share of predictions that are real", color="#C9CDD3")
    ax.set_title("Occluded-region occupancy: precision vs recall\n"
                 f"(10 held-out ScanNet scenes, {len(lab):,} occluded voxels)",
                 color="#F4F1EA")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.15, color="#5A6472")
    leg = ax.legend(loc="upper right", facecolor="#161B22", edgecolor="#5A6472")
    for t in leg.get_texts():
        t.set_color("#C9CDD3")

    png = root / "docs" / "images" / "occluded_pr_curve.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[pr] τ=0 operating point: P={op['precision']:.3f} R={op['recall']:.3f} "
          f"F1={op['f1']:.3f}")
    print(f"[pr] best F1 on curve:    P={best['precision']:.3f} R={best['recall']:.3f} "
          f"F1={best['f1']:.3f} at τ={best['tau_m']}")
    print(f"[pr] wrote {png.relative_to(root)} and "
          f"{(out_dir / 'results.json').relative_to(root)}")


if __name__ == "__main__":
    main()
