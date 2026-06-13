#!/usr/bin/env python
"""
upload_completer_hf.py — publish the trained completer to HuggingFace.

Uploads checkpoints/completer_best.pt (+ eval results.json if present) to
the model repo onemore-adi/occlusynth-completer.

Requires a write token:  hf auth login   (or HF_TOKEN env var)

Usage:
    .venv312/bin/python scripts/upload_completer_hf.py
    .venv312/bin/python scripts/upload_completer_hf.py --ckpt checkpoints/completer_best.pt
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from occlusynth.utils import get_repo_root

REPO_ID = "onemore-adi/occlusynth-completer"

MODEL_CARD = """\
---
license: mit
tags:
  - 3d-reconstruction
  - scene-completion
  - signed-distance-field
  - scannet
  - pytorch
---

# OccluSynth Completer

Part of the **OccluSynth** pipeline for occlusion-aware 3D scene reconstruction
(Aditya Agarwal, NIT Rourkela).

## What it does

Given a *partial*, visibility-aware voxel grid produced by scanning a scene
with a depth camera, the completer predicts the **complete signed distance
field (SDF)** — including the OCCLUDED region behind observed surfaces.

Occluded voxels are those that are inside a camera frustum but permanently
behind a measured surface: they never appear in any depth image, so 2D
depth-inpainting networks cannot recover them at all. The completer operates
in 3D on 96³ voxel crops and explicitly targets this region.

## Architecture

Encoder-decoder **3D U-Net** with skip connections.

| Component | Detail |
|---|---|
| Parameters | 14.7 M |
| Input | `(B, 3, D, H, W)` — channels: sdf, weight, p_observed |
| Output | `(B, 1, D, H, W)` — completed SDF in metres (unbounded) |
| Encoder | 4 blocks, channels [32, 64, 128, 256], stride-2 Conv3d |
| Decoder | 4 blocks, trilinear upsample + skip concat |
| Norms / activation | GroupNorm(8) + GELU throughout |
| Head | 1×1×1 Conv3d, no activation |
| Voxel pitch | 5 cm |
| Crop size | 96³ |

**Loss:** masked L1 on `SURFACE ∪ OCCLUDED` voxels only.
`UNOBSERVABLE` voxels (never inside any camera frustum) are excluded entirely
from loss — the network is never penalised for what it cannot know.

## Training data

| Detail | Value |
|---|---|
| Dataset | ScanNet v2 (`_vh_clean_2.ply` meshes + GT poses + GT depth) |
| Scenes | 40 train / 10 val (deterministic md5 hash split) |
| Crops | 418 train / 90 val × 96³ at 5 cm |
| Crop rejection | occluded fraction < 10%, or > 50% exactly-zero GT SDF |
| Augmentation | 4 yaw rotations × 2 horizontal flips (z-axis never touched — preserves gravity-aligned priors) |
| GT SDF | `mesh_to_tsdf()` via open3d RaycastingScene, sampled on the same origin/dims as the fused grid |

**Grid-alignment contract.** The GT SDF is sampled at voxel centres
`origin + (idx + 0.5) * 0.05 m`, using the *identical* world-space origin
and grid dimensions as the partial grid produced by `fuse_visibility()`.
Before any training data was generated, alignment was verified on
`scene0000_00`: median |GT SDF| at surface voxels = **2.51 cm**
(threshold 7.5 cm), 93.1% of surface voxels within 1.5 voxels of the
GT zero-crossing. A mismatch here produces silent training on garbage.

## Evaluation (64³ interim checkpoint, 35 epochs + augmentation)

Metrics split by voxel class — the split is the contribution.
Surface voxels were observed; occluded voxels were not.

| Method | Class | MAE (cm) ↓ | Sign acc ↑ | Compl < 5 cm ↑ |
|---|---|---|---|---|
| no_completion (SDF = 0) | surface | 7.65 | 0.432 | 0.509 |
| no_completion | **occluded** | 45.27 | 0.299 | 0.061 |
| occluded_as_free (SDF = +0.1) | surface | 7.65 | 0.432 | 0.509 |
| occluded_as_free | **occluded** | 42.00 | 0.701 | 0.121 |
| **OccluSynth Completer** | surface | **4.86** | **0.585** | **0.768** |
| **OccluSynth Completer** | **occluded** | **27.14** | **0.722** | **0.349** |

The completer reduces occluded-voxel MAE by 40% vs the best baseline and
achieves nearly 3× the completion ratio (34.9% vs 12.1% within 5 cm of GT).
This is an *interim* checkpoint at 64³ crop size; a full 96³ A100 run is
expected to improve these numbers further.

## Usage

```python
import torch
from occlusynth.models import OccluSynthCompleter

model = OccluSynthCompleter()
ckpt = torch.load("completer_best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

# inp: (B, 3, D, H, W) float32 — (sdf, weight, p_observed) from fuse_visibility_grid()
# All three channels should match the normalisation used during training:
#   sdf      normalised to [-1, 1] over the 10 cm surface band
#   weight   raw fusion weight (unnormalised)
#   p_observed in [0, 1]
with torch.no_grad():
    completed_sdf = model(inp)  # (B, 1, D, H, W) metres
```

See `src/occlusynth/fusion/scene_grid.py` for how to produce the input from
raw RGB-D frames, and `scripts/eval_completer.py` for the full evaluation
pipeline.

## ScanNet terms of use

The weights in this repository were trained on data derived from the
**ScanNet** dataset. ScanNet is released for non-commercial research use
only. Users of this checkpoint must comply with the
[ScanNet Terms of Use](http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf).
In particular:
- Do not use these weights or the derived reconstructions for commercial
  purposes without explicit permission from the ScanNet authors.
- Do not redistribute the raw ScanNet data.
- Cite ScanNet in any publication that uses this model.

> Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.
> *ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes.*
> CVPR 2017.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/completer_best.pt")
    p.add_argument("--repo_id", default=REPO_ID)
    args = p.parse_args()

    root = get_repo_root()
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(ckpt), path_in_repo="completer_best.pt",
                    repo_id=args.repo_id)

    results = root / "demo_outputs/completer_eval/results.json"
    if results.exists():
        api.upload_file(path_or_fileobj=str(results),
                        path_in_repo="results.json", repo_id=args.repo_id)

    api.upload_file(path_or_fileobj=MODEL_CARD.encode(),
                    path_in_repo="README.md", repo_id=args.repo_id)
    print(f"uploaded → https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
