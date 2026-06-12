#!/usr/bin/env python
"""
eval_completer.py — completer vs baselines on the val crops, split by voxel class.

The split is the entire point: SURFACE voxels were observed (all methods should
tie — sanity check), OCCLUDED voxels were never seen by any camera and only the
completer can say anything about them.

Methods:
  completer        OccluSynthCompleter prediction (metres)
  no_completion    occluded SDF = 0     (zero-thickness-wall assumption);
                   surface voxels keep the partial TSDF (denormalised)
  occluded_as_free occluded SDF = +0.1  (naive free-space assumption);
                   surface voxels as above

Metrics, per class:
  mae_cm           mean |pred − gt| in cm
  sign_acc         free/occupied classification accuracy (sign of SDF)
  completion_ratio fraction of voxels within 5 cm of GT

Outputs (demo_outputs/completer_eval/):
  results.json     full metrics table
  <scene>.rrd      input states / predicted zero-band / GT zero-band,
                   same world frame, toggleable

Usage:
    .venv312/bin/python scripts/eval_completer.py --device mps \
        --ckpt checkpoints/completer_best.pt
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from occlusynth.fusion import FREE, SURFACE, OCCLUDED, CLASS_COLORS
from occlusynth.fusion.tsdf import TSDFConfig
from occlusynth.models.completer import OccluSynthCompleter
from occlusynth.utils import get_repo_root

SURFACE_TRUNC = TSDFConfig().surface_trunc   # 0.10 m — denormalises input TSDF


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    err = np.abs(pred - gt)
    return {
        "mae_cm": float(err.mean() * 100),
        "sign_acc": float(((pred > 0) == (gt > 0)).mean()),
        "completion_ratio": float((err < 0.05).mean()),
    }


def baseline_pred(inp_sdf: np.ndarray, state: np.ndarray,
                  occluded_value: float) -> np.ndarray:
    """Surface keeps the (denormalised) partial TSDF; occluded gets a constant."""
    pred = inp_sdf * SURFACE_TRUNC          # normalized [-1,1] → metres
    pred[state == OCCLUDED] = occluded_value
    return pred


def log_scene_rrd(out_dir: Path, scene: str, crops: list) -> Path:
    """One .rrd: input states + predicted/GT SDF zero-bands in world frame."""
    import rerun as rr
    rrd = out_dir / f"{scene}.rrd"
    rr.init("occlusynth-completer-eval", spawn=False)
    rr.save(str(rrd))

    def centers_of(crop, mask):
        idx = np.argwhere(mask)
        return (crop["world_origin"] +
                (idx + 0.5) * crop["voxel_size"]).astype(np.float32)

    band = 0.025  # |sdf| < half a voxel → zero-crossing shell
    for crop in crops:
        state, pred, gt = crop["state"], crop["pred"], crop["target"]
        tag = f"crop{crop['i']:02d}"
        for path, mask, color in [
            (f"input/surface/{tag}",  state == SURFACE,  CLASS_COLORS[SURFACE]),
            (f"input/occluded/{tag}", state == OCCLUDED, CLASS_COLORS[OCCLUDED]),
            (f"predicted/zero_band/{tag}", np.abs(pred) < band, (70, 120, 240)),
            (f"gt/zero_band/{tag}",        np.abs(gt) < band,   (160, 160, 160)),
            # the contribution: occupied space recovered inside the occlusion
            (f"predicted/occupied_in_occluded/{tag}",
             (state == OCCLUDED) & (pred < 0), (240, 80, 200)),
            (f"gt/occupied_in_occluded/{tag}",
             (state == OCCLUDED) & (gt < 0), (120, 60, 160)),
        ]:
            pts = centers_of(crop, mask)
            if len(pts):
                rr.log(path, rr.Points3D(pts, colors=color, radii=0.012),
                       static=True)
    return rrd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/completer_best.pt")
    p.add_argument("--data_dir", default="data/completer_crops/val")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--no_rerun", action="store_true")
    args = p.parse_args()

    root = get_repo_root()
    out_dir = root / "demo_outputs/completer_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = OccluSynthCompleter().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"checkpoint: {args.ckpt} (epoch {ckpt.get('epoch')}, "
          f"val {ckpt.get('val_loss'):.4f})")

    files = sorted(Path(args.data_dir).glob("*.npz"))
    print(f"val crops: {len(files)}")

    # accumulate flat (pred, gt) per method per class
    acc = {m: {"surface": [[], []], "occluded": [[], []]}
           for m in ("completer", "no_completion", "occluded_as_free")}
    by_scene = defaultdict(list)

    with torch.no_grad():
        for f in files:
            z = np.load(f)
            inp = z["input"].astype(np.float32)
            gt = z["target"].astype(np.float32)
            state = z["state"]

            x = torch.from_numpy(inp).unsqueeze(0).to(device)
            pred = model(x)[0, 0].cpu().numpy()

            preds = {
                "completer": pred,
                "no_completion": baseline_pred(inp[0], state, 0.0),
                "occluded_as_free": baseline_pred(inp[0], state, 0.1),
            }
            for cls_name, cls in (("surface", SURFACE), ("occluded", OCCLUDED)):
                m = state == cls
                for method, pr in preds.items():
                    acc[method][cls_name][0].append(pr[m])
                    acc[method][cls_name][1].append(gt[m])

            scene = str(z["scene"])
            by_scene[scene].append({
                "i": len(by_scene[scene]), "state": state, "pred": pred,
                "target": gt, "world_origin": z["world_origin"],
                "voxel_size": float(z["voxel_size"]),
            })
    if args.device == "mps":
        torch.mps.empty_cache()

    results = {m: {c: metrics(np.concatenate(v[0]), np.concatenate(v[1]))
                   for c, v in d.items()} for m, d in acc.items()}

    # ── report ───────────────────────────────────────────────────────────────
    hdr = f"{'method':<18}{'class':<10}{'MAE (cm)':>10}{'sign acc':>10}{'compl<5cm':>11}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for m in ("no_completion", "occluded_as_free", "completer"):
        for c in ("surface", "occluded"):
            r = results[m][c]
            print(f"{m:<18}{c:<10}{r['mae_cm']:>10.2f}{r['sign_acc']:>10.3f}"
                  f"{r['completion_ratio']:>11.3f}")

    beats = all(
        results["completer"]["occluded"]["mae_cm"] < results[b]["occluded"]["mae_cm"]
        for b in ("no_completion", "occluded_as_free"))
    print(f"\ncompleter beats both baselines on occluded MAE: {beats}")

    results["_meta"] = {"ckpt": str(args.ckpt), "epoch": int(ckpt.get("epoch", -1)),
                        "n_val_crops": len(files),
                        "beats_both_baselines_occluded_mae": bool(beats)}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"results → {out_dir / 'results.json'}")

    if not args.no_rerun:
        for scene, crops in sorted(by_scene.items()):
            rrd = log_scene_rrd(out_dir, scene, crops)
            print(f"  rrd → {rrd}")


if __name__ == "__main__":
    main()
