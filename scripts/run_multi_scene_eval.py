#!/usr/bin/env python
"""
run_multi_scene_eval.py — Validate RANSAC metric grounding across N val scenes.

Runs the closed-form fit on N deterministically selected val-split scenes,
records per-scene mean/worst-case ARE, and flags pathological scenes.

Usage:
    python scripts/run_multi_scene_eval.py             # 10 scenes
    python scripts/run_multi_scene_eval.py --n 5       # 5 scenes (faster)
    python scripts/run_multi_scene_eval.py --n 0       # all 307 val scenes
    python scripts/run_multi_scene_eval.py --scenes scene0006_00 scene0050_00

VGGT predictions are cached after the first run, so subsequent runs are fast.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from occlusynth.data import ScanNetDataset
from occlusynth.models import eval_scene
from occlusynth.models.vggt_wrapper import VGGTWrapper
from occlusynth.utils import get_repo_root, get_config


# ── scene selection ───────────────────────────────────────────────────────────

def pick_eval_scenes(ds: ScanNetDataset, n: int) -> list[str]:
    """
    Return n scenes evenly spread across the val split (deterministic).
    Spreading avoids clustering in one part of the dataset hash space.
    n=0 returns all val scenes.
    """
    if n == 0 or n >= len(ds.scenes):
        return list(ds.scenes)
    idx = np.linspace(0, len(ds.scenes) - 1, n, dtype=int)
    return [ds.scenes[i] for i in idx]


# ── display ───────────────────────────────────────────────────────────────────

FLAG_ICON = {"OK": "✓", "WARN": "⚠", "DEGRADE": "✗"}


def print_table(results: list[dict]) -> None:
    hdr = (f"  {'scene_id':<20} {'frames':>6} {'mean ARE':>10} {'max ARE':>9} "
           f"{'RMSE (m)':>9} {'δ<1.25':>8} {'mean a':>8} {'inlier%':>8}  flag")
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    for r in results:
        icon = FLAG_ICON[r["flag"]]
        reasons = "; ".join(r["flag_reasons"]) if r["flag_reasons"] else ""
        note = f"  [{reasons}]" if reasons else ""
        print(
            f"  {r['scene_id']:<20} {r['n_frames']:>6} "
            f"{r['mean_ARE']:>10.4f} {r['max_ARE']:>9.4f} "
            f"{r['mean_RMSE']:>9.4f} {r['mean_delta_125']*100:>7.1f}% "
            f"{r['mean_a']:>8.2f} {r['mean_inlier_ratio']*100:>7.1f}%"
            f"  {icon}{note}"
        )


def print_aggregate(results: list[dict]) -> None:
    ok      = [r for r in results if r["flag"] == "OK"]
    warn    = [r for r in results if r["flag"] == "WARN"]
    degrade = [r for r in results if r["flag"] == "DEGRADE"]

    ares   = [r["mean_ARE"]  for r in results]
    rmses  = [r["mean_RMSE"] for r in results]
    d125s  = [r["mean_delta_125"] for r in results]

    print(f"\n  Scenes evaluated : {len(results)}")
    print(f"  OK / WARN / DEGRADE : {len(ok)} / {len(warn)} / {len(degrade)}")
    print(f"\n  Mean   ARE  across scenes : {np.mean(ares):.4f}")
    print(f"  Median ARE  across scenes : {np.median(ares):.4f}")
    print(f"  Worst  ARE  (scene)       : {max(ares):.4f}  ({results[np.argmax(ares)]['scene_id']})")
    print(f"  Mean   RMSE across scenes : {np.mean(rmses):.4f} m")
    print(f"  Mean   δ<1.25             : {np.mean(d125s)*100:.1f}%")

    if degrade:
        print("\n  DEGRADE scenes (excluded from aggregate):")
        for r in degrade:
            print(f"    {r['scene_id']} — {'; '.join(r['flag_reasons'])}")
    if warn:
        print("\n  WARN scenes:")
        for r in warn:
            print(f"    {r['scene_id']} — {'; '.join(r['flag_reasons'])}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n",       type=int, default=10,
                   help="number of val scenes to evaluate (0 = all 307)")
    p.add_argument("--scenes",  nargs="+", default=None,
                   help="explicit scene IDs to evaluate (overrides --n)")
    p.add_argument("--n_frames",  type=int, default=6)
    p.add_argument("--n_anchors", type=int, default=500)
    p.add_argument("--out_dir",   default=None)
    args = p.parse_args()

    root    = get_repo_root()
    cfg     = get_config()
    out_dir = Path(args.out_dir) if args.out_dir else root / "demo_outputs/multi_scene_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / cfg["cache_dir"]

    ds = ScanNetDataset(split="val", n_frames=args.n_frames)

    if args.scenes:
        eval_scenes = args.scenes
        # allow scenes outside val split (e.g. scene0000_00 is in train by hash)
        ds_all = ScanNetDataset(split="all", n_frames=args.n_frames)
        for sid in eval_scenes:
            if sid not in ds.scenes and sid not in ds_all.scenes:
                sys.exit(f"Scene '{sid}' not found in dataset root.")
        # rebuild ds to include any requested train-split scenes
        ds = ds_all
    else:
        eval_scenes = pick_eval_scenes(ds, args.n)

    print(f"Evaluating {len(eval_scenes)} scene(s), {args.n_frames} frames each.")
    print(f"Cache dir : {cache_dir}")
    print(f"Output dir: {out_dir}\n")

    # Load model once; it stays in memory across all scenes.
    wrapper = VGGTWrapper()

    results  = []
    t_start  = time.time()

    for i, scene_id in enumerate(eval_scenes):
        t_scene = time.time()
        print(f"[{i+1}/{len(eval_scenes)}] {scene_id} ...", end=" ", flush=True)
        try:
            r = eval_scene(
                scene_id, ds, wrapper,
                cache_dir=cache_dir,
                n_anchors=args.n_anchors,
            )
            elapsed = time.time() - t_scene
            icon    = FLAG_ICON[r["flag"]]
            reasons = f"  [{'; '.join(r['flag_reasons'])}]" if r["flag_reasons"] else ""
            print(f"ARE={r['mean_ARE']:.4f}  max={r['max_ARE']:.4f}  "
                  f"δ<1.25={r['mean_delta_125']*100:.1f}%  "
                  f"a={r['mean_a']:.2f}±{r['std_a']:.2f}  "
                  f"{elapsed:.0f}s  {icon}{reasons}")
            results.append(r)
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({
                "scene_id": scene_id, "n_frames": args.n_frames,
                "mean_ARE": float("nan"), "max_ARE": float("nan"),
                "mean_RMSE": float("nan"), "mean_delta_125": float("nan"),
                "mean_a": float("nan"), "std_a": float("nan"),
                "mean_inlier_ratio": float("nan"),
                "flag": "ERROR", "flag_reasons": [str(exc)],
                "per_frame": [],
            })

    total_time = time.time() - t_start
    wrapper.unload()

    # ── table + aggregate ─────────────────────────────────────────────────────
    valid = [r for r in results if r["flag"] != "ERROR"]
    print(f"\n{'='*80}")
    print(f"  Multi-scene eval: {len(eval_scenes)} scenes, {args.n_frames} frames each")
    print(f"  Total wall time: {total_time/60:.1f} min")
    print(f"{'='*80}\n")
    print_table(valid)
    print()
    print_aggregate(valid)

    # ── save JSON ─────────────────────────────────────────────────────────────
    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nFull results → {out_json}")

    # ── presentation table (markdown) ─────────────────────────────────────────
    md_lines = [
        "| Scene | Frames | mean ARE ↓ | max ARE ↓ | RMSE (m) ↓ | δ<1.25 ↑ | Scale a | Flag |",
        "|-------|--------|-----------|---------|----------|--------|---------|------|",
    ]
    for r in valid:
        reasons = "; ".join(r["flag_reasons"]) if r["flag_reasons"] else "—"
        md_lines.append(
            f"| {r['scene_id']} | {r['n_frames']} "
            f"| {r['mean_ARE']:.3f} | {r['max_ARE']:.3f} "
            f"| {r['mean_RMSE']:.3f} | {r['mean_delta_125']*100:.1f}% "
            f"| {r['mean_a']:.2f}±{r['std_a']:.2f} "
            f"| {r['flag']} |"
        )

    ok_results = [r for r in valid if r["flag"] == "OK"]
    if ok_results:
        md_lines += [
            "",
            f"**Aggregate (OK scenes only, N={len(ok_results)}):** "
            f"mean ARE {np.mean([r['mean_ARE'] for r in ok_results]):.3f} | "
            f"mean δ<1.25 {np.mean([r['mean_delta_125'] for r in ok_results])*100:.1f}%",
        ]

    md_path = out_dir / "results_table.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Markdown table → {md_path}")


if __name__ == "__main__":
    main()
