#!/usr/bin/env python
"""
eval_calibration.py — calibration analysis for OccluSynth MC Dropout uncertainty.

Evaluates whether p_occ (fraction of stochastic SDF samples with SDF<0) is a
well-calibrated occupancy probability on the val set.  Reports Expected
Calibration Error (ECE) and saves a reliability diagram.

Definition
----------
For OCCLUDED voxels only (the blind spot — the region we claim to know about):
  - predicted: p_occ ∈ [0,1] from predict_with_uncertainty()
  - actual:    gt_sdf < 0  (GT mesh SDF is negative inside solid geometry)
  - ECE = Σ_b (|B_b|/N) × |mean_confidence_b − freq_occupied_b|
            over equal-width bins of confidence.

A perfectly calibrated model hits the diagonal; ECE=0 is ideal.

Note on the headline claim
--------------------------
The model uses *inference-only* MC Dropout — no fine-tuning with a calibration
loss.  Inference-only dropout is a known heuristic and is not expected to give
ECE < 0.05 out of the box.  This script reports the actual ECE honestly.  If
the deck claims ECE < 0.05, that claim requires temperature scaling or a
dedicated calibration fine-tune step (run documented in scripts/train_completer.py
with the --calibrate flag once available).  For now, use this script to measure
the baseline.

Usage:
    .venv312/bin/python scripts/eval_calibration.py
    .venv312/bin/python scripts/eval_calibration.py --n_samples 32 --n_bins 15
    .venv312/bin/python scripts/eval_calibration.py --fast  # n_samples=4 for speed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from occlusynth.models.completer import OccluSynthCompleter, predict_with_uncertainty
from occlusynth.utils import get_repo_root

OCCLUDED = 3


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: np.ndarray,   # p_occ values ∈ [0,1]
    labels: np.ndarray,        # ground-truth binary occupancy (0 or 1)
    n_bins: int = 10,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expected Calibration Error (ECE) with equal-width bins.

    Returns:
        ece:         scalar ECE
        bin_confs:   mean predicted confidence per bin
        bin_freqs:   GT occupancy frequency per bin
        bin_counts:  number of samples per bin
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs  = np.zeros(n_bins)
    bin_freqs  = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (confidences >= lo) & (confidences < hi if b < n_bins - 1 else confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_counts[b] = mask.sum()
        bin_confs[b]  = confidences[mask].mean()
        bin_freqs[b]  = labels[mask].mean()

    n = len(confidences)
    ece = float((bin_counts / n * np.abs(bin_confs - bin_freqs)).sum())
    return ece, bin_confs, bin_freqs, bin_counts


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def save_reliability_diagram(
    bin_confs: np.ndarray,
    bin_freqs: np.ndarray,
    bin_counts: np.ndarray,
    ece: float,
    n_samples: int,
    n_crops: int,
    out_path: Path,
    tag: str = "",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[eval_calibration] matplotlib not available — skipping PNG")
        return

    fig = plt.figure(figsize=(7, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)

    # --- calibration plot ---
    ax = fig.add_subplot(gs[0])
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")

    mask = bin_counts > 0
    xs = bin_confs[mask]
    ys = bin_freqs[mask]

    # Shade calibration gap
    ax.fill_between(xs, xs, ys,
                    where=(ys > xs), alpha=0.15, color="royalblue",
                    label="Underconfident")
    ax.fill_between(xs, xs, ys,
                    where=(ys <= xs), alpha=0.15, color="tomato",
                    label="Overconfident")
    ax.plot(xs, ys, "o-", color="steelblue", lw=2, ms=6, label="Model")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted p_occ (confidence)")
    ax.set_ylabel("GT occupancy frequency")
    ax.set_title(
        f"Reliability Diagram — OCCLUDED voxels{(' — ' + tag) if tag else ''}\n"
        f"ECE = {ece:.4f}  |  {n_samples} samples  |  {n_crops} val crops"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.98, 0.02, f"ECE={ece:.4f}", ha="right", va="bottom",
            transform=ax.transAxes,
            fontsize=12, color="steelblue",
            bbox=dict(boxstyle="round", fc="white", ec="steelblue", alpha=0.8))

    # --- histogram ---
    ax2 = fig.add_subplot(gs[1])
    n_bins = len(bin_counts)
    width  = 1.0 / n_bins
    centers = bin_confs  # already computed as mean per bin
    # For bins with counts, use centre-of-bin as x position for histogram
    bin_centres = np.linspace(width / 2, 1 - width / 2, n_bins)
    ax2.bar(bin_centres, bin_counts, width=width * 0.9,
            color="steelblue", alpha=0.7)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Predicted p_occ")
    ax2.set_ylabel("Sample count")
    ax2.set_title("Bin populations")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval_calibration] reliability diagram → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Calibration analysis for OccluSynth MC Dropout uncertainty"
    )
    p.add_argument("--ckpt", default="checkpoints/interim_64_aug/completer_best.pt")
    p.add_argument("--data_dir", default="data/completer_crops/val")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--n_samples", type=int, default=16,
                   help="MC Dropout samples per crop (default 16)")
    p.add_argument("--n_bins", type=int, default=10,
                   help="ECE bins (default 10)")
    p.add_argument("--fast", action="store_true",
                   help="Use n_samples=4 for quick testing")
    p.add_argument("--max_crops", type=int, default=None,
                   help="Limit number of val crops (for quick runs)")
    args = p.parse_args()

    if args.fast:
        args.n_samples = 4

    root = get_repo_root()
    out_dir = root / "demo_outputs" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ───────────────────────────────────────────────────────────
    ckpt_path = root / args.ckpt
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path}")

    device = torch.device(args.device)
    ckpt   = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model  = OccluSynthCompleter(mc_dropout=True).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[eval_calibration] checkpoint ep={ckpt.get('epoch')} "
          f"val_loss={ckpt.get('val_loss', '?'):.4f}")
    print(f"[eval_calibration] n_samples={args.n_samples}  n_bins={args.n_bins}")

    # ── collect p_occ vs GT occupancy for OCCLUDED voxels ────────────────────
    val_dir = root / args.data_dir
    files   = sorted(val_dir.glob("*.npz"))
    if args.max_crops:
        files = files[: args.max_crops]
    print(f"[eval_calibration] evaluating {len(files)} val crops ...")

    all_confs  = []   # p_occ for occluded voxels
    all_labels = []   # GT occupancy (gt_sdf < 0) for occluded voxels

    for i, f in enumerate(files):
        z      = np.load(f)
        inp_np = z["input"].astype(np.float32)
        gt_np  = z["target"].astype(np.float32)   # GT SDF in metres
        state  = z["state"]                        # uint8

        inp_t = torch.from_numpy(inp_np).unsqueeze(0).to(device)

        _, _, p_occ_t = predict_with_uncertainty(model, inp_t,
                                                  n_samples=args.n_samples)
        p_occ = p_occ_t[0, 0].cpu().numpy()   # (D, H, W)

        occ_mask = state == OCCLUDED
        if not occ_mask.any():
            continue

        confs  = p_occ[occ_mask].astype(np.float64)
        labels = (gt_np[occ_mask] < 0).astype(np.float64)

        all_confs.append(confs)
        all_labels.append(labels)

        if (i + 1) % 10 == 0 or i == len(files) - 1:
            n_done = len(all_confs)
            print(f"  crop {i+1}/{len(files)}  "
                  f"occluded voxels collected: {sum(len(c) for c in all_confs):,}")

    if not all_confs:
        sys.exit("[eval_calibration] no OCCLUDED voxels found in val set")

    confs_all  = np.concatenate(all_confs)
    labels_all = np.concatenate(all_labels)

    gt_occ_rate = labels_all.mean()
    print(f"\n[eval_calibration] total occluded voxels: {len(confs_all):,}")
    print(f"[eval_calibration] GT occupancy rate (how much occ space is solid): "
          f"{gt_occ_rate:.4f}")

    # ── compute ECE ──────────────────────────────────────────────────────────
    ece, bin_confs, bin_freqs, bin_counts = compute_ece(
        confs_all, labels_all, n_bins=args.n_bins
    )

    print(f"\n=== Calibration results ===")
    print(f"  ECE          : {ece:.4f}")
    print(f"  Mean p_occ   : {confs_all.mean():.4f}")
    print(f"  GT occ. rate : {gt_occ_rate:.4f}")
    print(f"  n_samples    : {args.n_samples}")
    print(f"  val crops    : {len(files)}")

    # Reconcile with deck claim
    threshold = 0.05
    print(f"\n  Deck claim: ECE < {threshold}")
    if ece < threshold:
        print(f"  → SATISFIED (ECE={ece:.4f} < {threshold})")
    else:
        print(f"  → NOT SATISFIED with inference-only MC Dropout "
              f"(ECE={ece:.4f} ≥ {threshold}).")
        print(f"     To achieve ECE < {threshold}: apply temperature scaling or "
              f"fine-tune with a calibration objective.")

    # ── save outputs ─────────────────────────────────────────────────────────
    tag   = f"ep{ckpt.get('epoch', 0)}_s{args.n_samples}"
    png   = root / "docs" / "images" / f"calibration_{tag}.png"
    save_reliability_diagram(
        bin_confs, bin_freqs, bin_counts,
        ece=ece,
        n_samples=args.n_samples,
        n_crops=len(files),
        out_path=png,
        tag=tag,
    )

    results = {
        "ece":            float(ece),
        "gt_occ_rate":    float(gt_occ_rate),
        "mean_p_occ":     float(confs_all.mean()),
        "n_occluded_vox": int(len(confs_all)),
        "n_val_crops":    len(files),
        "n_samples":      args.n_samples,
        "n_bins":         args.n_bins,
        "ckpt":           str(ckpt_path),
        "epoch":          int(ckpt.get("epoch", -1)),
        "deck_claim_ece_lt_0.05": bool(ece < 0.05),
        "note": (
            "inference-only MC Dropout; no calibration fine-tune applied. "
            "ECE reflects raw stochastic SDF sampling, not a probabilistic output head."
        ),
        "bin_confs":  bin_confs.tolist(),
        "bin_freqs":  bin_freqs.tolist(),
        "bin_counts": bin_counts.tolist(),
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n[eval_calibration] results → {results_path}")


if __name__ == "__main__":
    main()
