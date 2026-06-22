# OccluSynth — Occlusion-Aware 3D Scene Reconstruction

- **Problem Statement Number** - 09
- **Problem Statement Title** - Occlusion-Aware 3D Scene Reconstruction in Partially Observable Real-World Environments
- **Team name** - onemore_adi
- **Team members (Names)** - Aditya Agarwal
- **Institute/College Name** - National Institute of Technology, Rourkela
- **Final Presentation Google Drive Link** - [`PDF Drive Link`](https://drive.google.com/file/d/1cRv_bYaVuypEfW7YYF2Wp7zmxNOf7FWG/view?usp=sharing)
- **Full Submission Demo Video Link** - [YouTube: Demo Video](https://youtu.be/Nx2NK8ceUPw)
- **Setup & Result Reproducibility Video Link** - [YouTube: Reproducibility Video](https://youtu.be/3Jw84Sa7_i8)

### Project Artefacts

- **Technical Documentation** - See the [`docs/`](docs/) folder:
  - [`docs/architecture.md`](docs/architecture.md) — full technical architecture, design decisions, camera-pose strategy, OSS stack
  - [`docs/ax.md`](docs/ax.md) — **agentic AI development writeup** (how this was built with open-weight models / agentic tooling)
  - [`docs/completer_sprint.md`](docs/completer_sprint.md) — 3D U-Net completer spec, contracts, status
  - [`docs/safety_benchmark.md`](docs/safety_benchmark.md) — risk-aware planner safety benchmark
  - [`docs/adapter_design.md`](docs/adapter_design.md) — depth-adapter scaffold design
- **Source Code** - All source in [`src/occlusynth/`](src/occlusynth/); orchestration scripts in [`scripts/`](scripts/); tests in [`tests/`](tests/). Installable via `pip install -e .` (see Installation below).
- **Models Used** -
  - [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) — open-weight feed-forward 3D geometry transformer (depth / pose); used as a frozen off-the-shelf predictor. GitHub: [facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- **Models Published** - [onemore-adi/occlusynth-completer](https://huggingface.co/onemore-adi/occlusynth-completer) — OccluSynth Completer (14.7 M-param 3D U-Net, occluded-region SDF completion), MIT licence. ⚠️ **Make this repo public on HuggingFace before submission.**
- **Datasets Used** -
  - [ScanNet v2](http://www.scan-net.org/) — primary RGB-D + GT mesh dataset (non-commercial research licence)
  - [Microsoft 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) — cross-dataset generalisation probe
- **Datasets Published** - Completer training crops are **generated locally** from ScanNet (`scripts/generate_completer_data.py`) and not redistributed (ScanNet licence). N/A.

### Attribution

OccluSynth builds on **[VGGT-Omega](https://github.com/facebookresearch/vggt)** (Meta AI) for feed-forward monocular depth/pose prediction. VGGT is used as a frozen, off-the-shelf predictor only.

**New work developed in this project (not in VGGT):**

- Visibility-aware TSDF fusion that labels every voxel `free` / `surface` / `occluded` / `unobservable` (`src/occlusynth/fusion/`)
- RANSAC metric-grounding to lift VGGT's scale-ambiguous depth to metric scale (`src/occlusynth/models/metric_grounding.py`)
- The **OccluSynth Completer** — a 3D U-Net that predicts SDF inside occluded volumes (`src/occlusynth/models/`, `scripts/train_completer.py`)
- A risk-graded A\* planner over the completed cost map (`scripts/run_planner.py`)
- Full evaluation harness (geometry, completer, robustness, cross-dataset, safety benchmark)

---

## What it does

OccluSynth fuses sparse RGB-D observations into a dense, visibility-aware 3D
voxel grid that explicitly labels every voxel as **free** (confirmed empty),
**surface** (measured solid geometry), **occluded** (behind a surface in every
view — the robot's "blind spot"), or **unobservable** (never in any frustum).
A 3D U-Net completer then predicts the SDF inside the occluded volume —
recovering geometry that is absent from every depth image.

## Pipeline

```
RGB-D frames + ScanNet GT poses
        │
        ▼
  VGGT-Omega depth  →  RANSAC scale fit  →  metric depth
        │
        ▼
  fuse_visibility()   — 5 cm voxel grid, (sdf, weight, p_observed)
        │
        ▼
  OccluSynthCompleter — 3D U-Net, predicts SDF in occluded regions
        │
        ▼
  Completed dense SDF  →  risk-graded planner (A* on cost map)
```

## Pose source

**All fusion uses ScanNet ground-truth camera poses**, not poses predicted by
VGGT-Omega. VGGT-Omega's absolute trajectory error on 6 frames is **~70 cm** —
14× the 5 cm voxel pitch — which produces blurred, doubled surfaces. ScanNet GT
poses are the output of BundleFusion, equivalent to assuming a calibrated
odometry source (the standard assumption for KinectFusion / BundleFusion /
ElasticFusion). In a real deployment this is replaced by a VIO / SLAM front-end.
See [`docs/architecture.md`](docs/architecture.md) §Camera Pose Strategy.

---

## Installation

```bash
# Python 3.12 environment for open3d / rerun / completer training
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -e .

# VGGT-Omega is an external dependency — clone separately (see docs/architecture.md)
# Place the checkpoint at: vggt/vggt-omega/checkpoints/vggt_omega_1b_512.pt
```

> Two environments are used: `.venv` (Python 3.14, VGGT inference / MPS) and
> `.venv312` (Python 3.12, open3d 0.19 fusion + completer training). See
> [`docs/architecture.md`](docs/architecture.md) for the full rationale.

## Quick start

```bash
# Visibility-aware voxel grid (GT depth, GT poses)
.venv312/bin/python scripts/run_visibility.py --scene scene0000_00 --use_gt_depth

# Alignment check (GT mesh ↔ fused grid)
.venv312/bin/python scripts/check_completer_alignment.py --scene scene0000_00

# Generate completer training crops
.venv312/bin/python scripts/generate_completer_data.py

# Train (MPS debug gate)
.venv312/bin/python scripts/train_completer.py --device mps --epochs 2 \
  --batch_size 2 --crop_size 64 --data_dir data/completer_crops --fast_dev_run

# Evaluate vs baselines
.venv312/bin/python scripts/eval_completer.py --device mps \
  --ckpt checkpoints/completer_best.pt
```

---

## Key results (64³ interim checkpoint)

| Method                   | Occluded MAE (cm) ↓ | Occluded sign acc ↑ | Compl < 5 cm ↑ |
| ------------------------ | ------------------- | ------------------- | -------------- |
| no_completion            | 45.27               | 0.299               | 0.061          |
| occluded_as_free         | 42.00               | 0.701               | 0.121          |
| **OccluSynth Completer** | **27.14**           | **0.722**           | **0.349**      |

Geometry surface/occluded split (`demo_outputs/geometry_eval/results.json`):

| Method         | Chamfer L1 ↓ | F-score@5cm ↑ | Occluded F-score ↑             |
| -------------- | ------------ | ------------- | ------------------------------ |
| TSDF-only      | 3.11 cm      | 74.1%         | 0.0% (cannot see behind walls) |
| **OccluSynth** | **1.77 cm**  | **83.5%**     | **32.0%**                      |

> Metrics are from the **interim 64³ MPS checkpoint** (`checkpoints/interim_64_aug/completer_best.pt`,
> epoch 32). The full 96³ A100 run is scripted and data-prepared
> (`scripts/train_completer.py --device cuda --crop_size 96`); it was not executed
> this phase due to compute access, not missing work, and is expected to improve all metrics.

## License

[MIT](LICENSE) © 2026 Aditya Agarwal
