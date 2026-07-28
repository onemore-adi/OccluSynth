#!/usr/bin/env python
"""
train_completer.py — train the 3D voxel completer on 96³ ScanNet crops.

Loss: L1(sdf_pred, sdf_gt) masked to SURFACE ∪ OCCLUDED voxels;
UNOBSERVABLE and FREE are excluded entirely.

Local MPS debug (mandatory before any cloud run):
    .venv312/bin/python scripts/train_completer.py --device mps --epochs 2 \
        --batch_size 2 --crop_size 64 --data_dir data/completer_crops --fast_dev_run

Cloud A100:
    python scripts/train_completer.py --device cuda --epochs 50 --batch_size 4 \
        --crop_size 96

Logging: wandb (project occlusynth-completer); falls back to offline mode when
no API key is configured, --no_wandb disables it entirely.  Every 10 steps the
train loss is logged; each epoch logs val loss + a cross-section slice image
(input SDF | predicted SDF | GT SDF) saved under checkpoints/slices/ too.

Checkpoints: best val loss → checkpoints/completer_best.pt, plus
completer_epoch{N:03d}.pt every 10 epochs.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from occlusynth.models.completer import (OccluSynthCompleter, masked_l1_loss,
                                         completion_loss, add_state_channels,
                                         SURFACE, OCCLUDED)


# ── data ──────────────────────────────────────────────────────────────────────

def augment_crop(inp: np.ndarray, tgt: np.ndarray, state: np.ndarray,
                 k: int, f: int):
    """
    Apply one of 8 gravity-preserving variants: k yaw rotations (90° about
    the z axis) then an optional x-flip — the dihedral group D4 in the
    horizontal plane (an x-flip composed with rotations also yields y-flips).

    The z axis is NEVER transformed: flipping z would put ceilings below
    floors, and the completer's job is exploiting gravity-aligned priors
    (floors continue under tables).  SDF values are invariant under these
    isometries, so all four arrays are only permuted, never rescaled.

    Crop axis order is (x, y, z) — z last; input carries a leading channel
    axis.  All arrays MUST receive the identical transform in the same call:
    an input/target orientation mismatch still trains to a decreasing loss
    while learning garbage.
    """
    def t(a, ax_x, ax_y):
        a = np.rot90(a, k, axes=(ax_x, ax_y))
        if f:
            a = np.flip(a, axis=ax_x)
        return a

    return t(inp, 1, 2), t(tgt, 0, 1), t(state, 0, 1)


class CompleterCropDataset(Dataset):
    """96³ .npz crops; optionally sub-crops to a smaller cube (debug runs).

    ``augment`` enables the 8-variant yaw/flip augmentation — train only:
    it is force-disabled when ``train=False`` so val arrays stay
    byte-identical across runs and every val number remains comparable.
    """

    def __init__(self, crop_dir: Path, crop_size: int = 96,
                 train: bool = True, augment: bool = False):
        self.files = sorted(Path(crop_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz crops in {crop_dir}")
        self.crop_size = crop_size
        self.train = train
        self.augment = augment and train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        z = np.load(self.files[idx])
        inp = z["input"].astype(np.float32)      # (3, 96, 96, 96)
        tgt = z["target"].astype(np.float32)     # (96, 96, 96)
        state = z["state"]                       # (96, 96, 96) uint8

        c = self.crop_size
        full = inp.shape[1]
        if c < full:
            if self.train:                       # random sub-crop
                o = np.random.randint(0, full - c + 1, size=3)
            else:                                # deterministic centre crop
                o = np.full(3, (full - c) // 2)
            sl = (slice(o[0], o[0] + c), slice(o[1], o[1] + c),
                  slice(o[2], o[2] + c))
            inp, tgt, state = inp[(slice(None),) + sl], tgt[sl], state[sl]

        if self.augment:
            inp, tgt, state = augment_crop(inp, tgt, state,
                                           k=np.random.randint(4),
                                           f=np.random.randint(2))

        return (torch.from_numpy(np.ascontiguousarray(inp)),
                torch.from_numpy(np.ascontiguousarray(tgt)),
                torch.from_numpy(np.ascontiguousarray(state.astype(np.int64))))


# ── slice visualisation ───────────────────────────────────────────────────────

def slice_figure(inp_sdf, pred, tgt, state, path: Path):
    """Middle-z cross-section: input SDF | prediction | GT | state."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = inp_sdf.shape[-1] // 2
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [(inp_sdf[..., k], "input SDF (partial)", "coolwarm", (-1, 1)),
              (pred[..., k],    "predicted SDF",       "coolwarm", (-0.5, 0.5)),
              (tgt[..., k],     "GT SDF",              "coolwarm", (-0.5, 0.5)),
              (state[..., k],   "state (0=unobs 1=free 2=surf 3=occ)",
               "viridis", (0, 3))]
    for ax, (img, title, cmap, (lo, hi)) in zip(axes, panels):
        im = ax.imshow(img.T, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ── train / eval loops ────────────────────────────────────────────────────────

def run_val(model, loader, device, max_batches=None, v2=False,
            w_occ=0.5, w_free=0.2, trunc=0.30, w_near=2.0, free_margin=0.10):
    """Returns (loss, sample, metrics). `loss` stays the checkpoint-selection
    criterion (masked L1 for v1, completion_loss total for v2); metrics adds
    architecture-independent numbers so v1 and v2 runs can be compared:
    truncated L1 (±0.30 m) and occluded-region sign accuracy."""
    model.eval()
    losses, sample = [], None
    tl1, sacc = [], []
    with torch.no_grad():
        for bi, (inp, tgt, state) in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            inp, tgt, state = inp.to(device), tgt.to(device), state.to(device)
            if v2:
                inp = add_state_channels(inp, state)
            pred = model(inp)
            if v2:
                total, _ = completion_loss(pred, tgt, state,
                                           trunc=trunc, w_near=w_near,
                                           w_occ=w_occ, w_free=w_free,
                                           free_margin=free_margin)
                losses.append(total.item())
            else:
                losses.append(masked_l1_loss(pred, tgt, state).item())

            sup = (state == SURFACE) | (state == OCCLUDED)
            if sup.any():
                d = (pred[:, 0].clamp(-0.3, 0.3) - tgt.clamp(-0.3, 0.3)).abs()
                tl1.append(d[sup].mean().item())
            occ = state == OCCLUDED
            if occ.any():
                sacc.append((((pred[:, 0] < 0) == (tgt < 0))[occ])
                            .float().mean().item())
            if sample is None:
                sample = (inp[0, 0].cpu().numpy(), pred[0, 0].cpu().numpy(),
                          tgt[0].cpu().numpy(), state[0].cpu().numpy())
    model.train()
    metrics = {"trunc_l1": float(np.mean(tl1)) if tl1 else float("nan"),
               "occ_sign_acc": float(np.mean(sacc)) if sacc else float("nan")}
    return float(np.mean(losses)), sample, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/completer_crops")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--augment", action="store_true",
                   help="train-only 8-variant augmentation: 4 yaw rotations × "
                        "2 horizontal flips (never z — preserves gravity)")
    p.add_argument("--v2", action="store_true",
                   help="completer v2: one-hot state input channels (7ch), "
                        "occupancy head, truncated near-surface-weighted SDF "
                        "loss + free-space hinge (see completion_loss)")
    p.add_argument("--w_occ", type=float, default=0.5,
                   help="v2 occupancy-BCE weight (0 disables the occ head term)")
    p.add_argument("--w_free", type=float, default=0.2,
                   help="v2 observed-free-space hinge weight (0 disables it)")
    p.add_argument("--trunc", type=float, default=0.30,
                   help="v2 SDF supervision truncation in metres")
    p.add_argument("--w_near", type=float, default=2.0,
                   help="v2 extra loss weight within 10 cm of the GT surface")
    p.add_argument("--free_margin", type=float, default=0.10,
                   help="v2 free-space hinge margin in metres")
    p.add_argument("--fast_dev_run", action="store_true",
                   help="limit to 8 train / 2 val batches per epoch")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--init_from", default=None,
                   help="checkpoint to warm-start model weights from "
                        "(optimizer/scheduler restart fresh)")
    args = p.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CompleterCropDataset(data_dir / "train", args.crop_size,
                                    train=True, augment=args.augment)
    val_ds   = CompleterCropDataset(data_dir / "val",   args.crop_size,
                                    train=False, augment=False)
    pin = args.device == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=pin,
                          drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=pin)
    print(f"data: {len(train_ds)} train / {len(val_ds)} val crops @ "
          f"{args.crop_size}³, device {device}, "
          f"augment={'on (train only)' if args.augment else 'off'}")

    model = (OccluSynthCompleter(in_channels=7, occ_head=True) if args.v2
             else OccluSynthCompleter()).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params")
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"warm start: {args.init_from} "
              f"(epoch {ck.get('epoch')}, val {ck.get('val_loss'):.4f})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    wb = None
    if not args.no_wandb:
        import wandb
        if not (os.environ.get("WANDB_API_KEY") or
                (Path.home() / ".netrc").exists()):
            os.environ.setdefault("WANDB_MODE", "offline")
        wb = wandb.init(project="occlusynth-completer", config=vars(args))
        print(f"wandb: mode={wb.settings.mode}")

    best_val = float("inf")
    step = 0
    max_train_b = 8 if args.fast_dev_run else None
    max_val_b   = 2 if args.fast_dev_run else None

    for epoch in range(args.epochs):
        t0 = time.time()
        ep_losses = []
        for bi, (inp, tgt, state) in enumerate(train_dl):
            if max_train_b and bi >= max_train_b:
                break
            inp, tgt, state = inp.to(device), tgt.to(device), state.to(device)
            if args.v2:
                inp = add_state_channels(inp, state)
                loss, parts = completion_loss(model(inp), tgt, state,
                                              trunc=args.trunc, w_near=args.w_near,
                                              w_occ=args.w_occ, w_free=args.w_free,
                                              free_margin=args.free_margin)
            else:
                loss = masked_l1_loss(model(inp), tgt, state)
                parts = None
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())
            step += 1
            if step % 10 == 0:
                if wb:
                    log = {"train/loss": loss.item(),
                           "train/lr": sched.get_last_lr()[0]}
                    if parts:
                        log.update({f"train/{k}": v.item()
                                    for k, v in parts.items()})
                    wb.log(log, step=step)
                print(f"  e{epoch} s{step}: train {loss.item():.4f}")
        sched.step()

        val_loss, sample, vm = run_val(model, val_dl, device, max_val_b,
                                       v2=args.v2, w_occ=args.w_occ,
                                       w_free=args.w_free, trunc=args.trunc,
                                       w_near=args.w_near,
                                       free_margin=args.free_margin)
        if args.device == "mps":
            torch.mps.empty_cache()

        slice_path = slice_figure(*sample,
                                  ckpt_dir / "slices" / f"epoch{epoch:03d}.png")
        if wb:
            import wandb
            wb.log({"val/loss": val_loss,
                    "val/trunc_l1": vm["trunc_l1"],
                    "val/occ_sign_acc": vm["occ_sign_acc"],
                    "val/slice": wandb.Image(str(slice_path))}, step=step)
        print(f"epoch {epoch}: train {np.mean(ep_losses):.4f}  "
              f"val {val_loss:.4f}  trunc_l1 {vm['trunc_l1']:.4f}  "
              f"sign_acc {vm['occ_sign_acc']:.3f}  ({time.time()-t0:.0f}s)")

        ckpt = {"model": model.state_dict(), "epoch": epoch,
                "val_loss": val_loss, "config": vars(args)}
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, ckpt_dir / "completer_best.pt")
            print(f"  ↑ new best val {val_loss:.4f} → completer_best.pt")
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, ckpt_dir / f"completer_epoch{epoch+1:03d}.pt")

    if wb:
        wb.finish()
    print(f"done. best val {best_val:.4f}")


if __name__ == "__main__":
    main()
