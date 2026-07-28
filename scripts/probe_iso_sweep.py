#!/usr/bin/env python
"""Fast parameter-preserving probes for completer mesh quality.

At a 5-10 min budget you cannot retrain 14.7M params, so every probe here runs
on EXISTING weights and changes only how the model is applied:

  iso sweep   marching-cubes level / solidity threshold. The net's SDF is
              regressed-to-mean (blurry) which shrinks solids -> holes. One
              inference pass yields the whole precision/recall curve, so the
              best iso level is free to find.
  D4 TTA      average predictions over the 8 yaw/flip variants the model was
              TRAINED with (SDF is invariant under those isometries). Pure
              variance reduction -> flatter grounds, plainer walls.

Reported per (ckpt, tta, iso):
  solid_precision  high = does not balloon into empty space
  solid_recall     high = holes covered / furniture backs closed
  free_violation   % of observed-FREE voxels called solid (balloon into seen space)
  surf_l1_cm       truncated L1 on observed SURFACE (crispness where seen)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from occlusynth.models.completer import OccluSynthCompleter, add_state_channels

FREE, SURFACE, OCCLUDED = 1, 2, 3


def load_model(p, device):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    v2 = ck["model"]["stem.0.0.weight"].shape[1] == 7
    m = (OccluSynthCompleter(in_channels=7, occ_head=True) if v2
         else OccluSynthCompleter())
    m.load_state_dict(ck["model"]); m.to(device).eval()
    return m, v2, ck.get("epoch")


def _fwd(model, v2, inp, state, device):
    x = torch.from_numpy(np.ascontiguousarray(inp)).unsqueeze(0).to(device)
    if v2:
        s = torch.from_numpy(np.ascontiguousarray(state.astype(np.int64))).unsqueeze(0).to(device)
        x = add_state_channels(x, s)
    return model(x)[0, 0].cpu().numpy()


def predict(model, v2, inp, state, device, tta=False):
    """Plain forward, or mean over the 8 D4 (yaw x flip) variants.

    Crop axes are (x,y,z); input has a leading channel axis. Forward transform
    is rot90(k) then optional x-flip; the inverse un-flips before un-rotating.
    """
    if not tta:
        return _fwd(model, v2, inp, state, device)
    acc = None
    for k in range(4):
        for f in (0, 1):
            i_t = np.rot90(inp, k, axes=(1, 2))
            s_t = np.rot90(state, k, axes=(0, 1))
            if f:
                i_t, s_t = np.flip(i_t, axis=1), np.flip(s_t, axis=0)
            p = _fwd(model, v2, i_t, s_t, device)
            if f:
                p = np.flip(p, axis=0)
            p = np.rot90(p, -k, axes=(0, 1))          # invert rotation
            acc = p.astype(np.float64) if acc is None else acc + p
    return (acc / 8.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--data_dir", default=str(ROOT / "data/completer_crops/val"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tta", action="store_true", help="also evaluate with D4 TTA")
    ap.add_argument("--limit", type=int, default=None, help="use only N val crops")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    files = sorted(Path(args.data_dir).glob("*.npz"))
    if args.limit:
        files = files[:args.limit]
    # iso levels in metres: negative thickens solids (closes holes)
    isos = [-0.04, -0.03, -0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]
    print(f"val crops: {len(files)}  |  iso levels: {isos}")

    results = {}
    for ck in args.ckpts:
        model, v2, epoch = load_model(ck, device)
        name = Path(ck).parent.name
        for tta in ([False, True] if args.tta else [False]):
            occ_p, occ_g, free_p, surf_e = [], [], [], []
            for f in files:
                z = np.load(f)
                inp = z["input"].astype(np.float32)
                gt = z["target"].astype(np.float32)
                st = z["state"]
                with torch.no_grad():
                    pred = predict(model, v2, inp, st, device, tta=tta)
                occ = st == OCCLUDED
                occ_p.append(pred[occ]); occ_g.append(gt[occ] < 0)
                free_p.append(pred[st == FREE])
                sm = st == SURFACE
                surf_e.append(np.abs(np.clip(pred[sm], -.3, .3) - np.clip(gt[sm], -.3, .3)))
            occ_p = np.concatenate(occ_p); occ_g = np.concatenate(occ_g)
            free_p = np.concatenate(free_p)
            surf_l1 = float(np.concatenate(surf_e).mean() * 100)

            tag = f"{name}{'+tta' if tta else ''}"
            rows = []
            for iso in isos:
                ps = occ_p < iso
                tp = float((ps & occ_g).sum())
                prec = tp / ps.sum() if ps.sum() else float("nan")
                rec = tp / occ_g.sum() if occ_g.sum() else float("nan")
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                rows.append({"iso": iso, "precision": prec, "recall": rec, "f1": f1,
                             "free_violation_pct": float((free_p < iso).mean() * 100)})
            results[tag] = {"ckpt": ck, "epoch": epoch, "v2": bool(v2),
                            "surf_l1_cm": surf_l1, "sweep": rows}
            best = max(rows, key=lambda r: r["f1"])
            print(f"\n=== {tag}  (epoch {epoch}, surf_l1 {surf_l1:.2f} cm) ===")
            print(f"{'iso(m)':>8}{'prec':>9}{'recall':>9}{'F1':>9}{'free_viol%':>12}")
            for r in rows:
                mark = "  <-- best F1" if r is best else ""
                print(f"{r['iso']:>8.3f}{r['precision']:>9.3f}{r['recall']:>9.3f}"
                      f"{r['f1']:>9.3f}{r['free_violation_pct']:>12.2f}{mark}")
        if args.device == "mps":
            torch.mps.empty_cache()

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
