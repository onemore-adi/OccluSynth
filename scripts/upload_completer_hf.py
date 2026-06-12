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
  - scannet
---

# OccluSynth Completer

3D U-Net (14.7M params) that completes the signed distance field of a partial,
visibility-aware TSDF voxel grid — including OCCLUDED voxels that appear in no
depth image. Input: (sdf, weight, p_observed) 96^3 crops at 5 cm; output:
completed SDF in metres. Trained on 40 ScanNet scenes, supervised with masked
L1 on SURFACE and OCCLUDED voxels only.

```python
import torch
from occlusynth.models import OccluSynthCompleter

model = OccluSynthCompleter()
ckpt = torch.load("completer_best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
```

Part of the OccluSynth occlusion-aware reconstruction pipeline.
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
