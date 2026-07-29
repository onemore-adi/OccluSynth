# OccluSynth — Occlusion-Aware 3D Scene Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-142%20passing-brightgreen.svg)](tests/)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20model-occlusynth--completer-yellow)](https://huggingface.co/onemore-adi/occlusynth-completer)

**Reconstructs the parts of a scene no camera ever saw.** Conventional 3D
reconstruction treats "unobserved" as "empty" — a robot then plans straight
through the chair leg hidden behind a sofa. OccluSynth labels every voxel with
what the sensor *actually knew*, then predicts the geometry inside the blind
spot: **57.6% of hidden surface recovered within 5 cm, where observation-only
methods recover 0%.**

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
  - [`docs/render_quality.md`](docs/render_quality.md) — completer ablations, precision/recall frontier, and the mesh-export recipe
  - [`docs/ax.md`](docs/ax.md) — **agentic AI development writeup** (how this was built with open-weight models / agentic tooling)
  - [`docs/completer_sprint.md`](docs/completer_sprint.md) — 3D U-Net completer spec, contracts, status
  - [`docs/safety_benchmark.md`](docs/safety_benchmark.md) — risk-aware planner safety benchmark
  - [`docs/adapter_design.md`](docs/adapter_design.md) — depth-adapter scaffold design
- **Source Code** - All source in [`src/occlusynth/`](src/occlusynth/); orchestration scripts in [`scripts/`](scripts/); tests in [`tests/`](tests/). Installable via `pip install -e .` (see Installation below).
- **Models Used** -
  - [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) — open-weight feed-forward 3D geometry transformer (depth / pose); used as a frozen off-the-shelf predictor. GitHub: [facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- **Models Published** - [onemore-adi/occlusynth-completer](https://huggingface.co/onemore-adi/occlusynth-completer) — OccluSynth Completer (14.7 M-param 3D U-Net, occluded-region SDF completion), MIT licence.
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

## Contents

[What it does](#what-it-does) · [Pipeline](#pipeline) · [Repository layout](#repository-layout) ·
[Pose source](#pose-source) · [Installation](#installation) · [Quick start](#quick-start) ·
[Tests](#tests) · [Reproducing the results](#reproducing-the-results) ·
[Key results](#key-results-64-interim-checkpoint) · [Limitations](#limitations) · [License](#license)

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

## Repository layout

```
src/occlusynth/
  fusion/        visibility-aware TSDF fusion — the 4-state voxel labelling
  models/        completer (3D U-Net), metric grounding, depth calibration, VGGT wrapper
  planning/      risk-graded A* planner over the completed cost map
  data/          ScanNet / 7-Scenes loaders, sparse view sampling
  viz/           rerun + mesh visualisation helpers
scripts/         orchestration: fusion, training, evaluation, benchmarks, export
tests/           142 tests (unit + integration + planner)
docs/            technical documentation and figures
reproduce.sh     one-command re-run of every headline metric
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
  --ckpt checkpoints/interim_64_aug/completer_best.pt

# Figures & demo renders
.venv312/bin/python scripts/plot_occluded_pr.py            # occluded PR curve
.venv312/bin/python scripts/export_completed_mesh.py --scene scene0000_00 \
  --n_frames 40 --iso 0.04 --anchor --min_component 150 --smooth_iters 28
```

Render quality is dominated by fusion density (`--n_frames`); the marching-cubes
`--iso` level trades hole-closing against bulging, and `--anchor` drops completed
components not connected to measured surface. See
[`docs/render_quality.md`](docs/render_quality.md) for the analysis and the
recommended recipe.

## Tests

```bash
.venv312/bin/python -m pytest tests/ -q
```

142 tests covering fusion visibility semantics, the completer (architecture,
augmentation, loss masking), metric grounding, geometry evaluation, the planner,
the safety benchmark, and cross-dataset loading.

## Reproducing the results

Every headline metric re-runs from cached predictions and the released checkpoint —
no training and no GPU required:

```bash
./reproduce.sh
```

Writes a full transcript to `repro_full.log` plus one file per step
(`repro_step1.txt` … `repro_step6.txt`). Prerequisites: `.venv312` created with
`pip install -e .`, cached VGGT predictions in `demo_outputs/pred_cache/`, completer
crops in `data/completer_crops/`, and the checkpoint at
`checkpoints/interim_64_aug/completer_best.pt`.

---

## Key results (64³ interim checkpoint)

| Method                   | Occluded MAE (cm) ↓ | Occluded sign acc ↑ | Compl < 5 cm ↑ |
| ------------------------ | ------------------- | ------------------- | -------------- |
| no_completion            | 45.27               | 0.299               | 0.061          |
| occluded_as_free         | 42.00               | 0.701               | 0.121          |
| **OccluSynth Completer** | **27.14**           | **0.722**           | **0.349**      |

Geometry surface/occluded split (`demo_outputs/geometry_eval/results.json`, 10 held-out scenes):

| Method         | Chamfer L1 ↓ | Surface F-score@5cm ↑ | Occluded F-score@5cm ↑         |
| -------------- | ------------ | --------------------- | ------------------------------ |
| TSDF-only      | 3.05 cm      | 79.6%                 | 0.0% (cannot see behind walls) |
| **OccluSynth** | **2.20 cm**  | **84.7%**             | **37.2%**                      |

In the occluded region specifically, OccluSynth reaches **57.6% surface recall @5cm**
(vs **0%** for any observation-only method — no sensor measures behind a surface) at
**27.4% precision** — a deliberate recall-first operating point: a missed obstacle is a
collision, a phantom one is a slowdown. The full precision–recall sweep with per-voxel
confidence is in `docs/images/occluded_pr_curve.png` (`scripts/plot_occluded_pr.py`).

> Metrics are from the **interim 64³ MPS checkpoint** (`checkpoints/interim_64_aug/completer_best.pt`,
> epoch 32), trained on a 16 GB MacBook — no cloud GPU.

### Model exploration — four interventions, one frontier

Four independent attempts to improve the completer at 64³ all landed on the **same
precision/recall frontier**:

| # | Intervention | Outcome |
| - | ------------ | ------- |
| 1 | Multi-task loss (truncated near-surface SDF + occupancy BCE + free-space hinge) | Moves *along* the frontier. The free-space hinge cut observed-free violations 32.2% → 0.95%, paid for in recall |
| 2 | Explicit occluded/unobservable mask input channels + occupancy head | No frontier gain |
| 3 | 3.66× multi-density training data (each scene fused at 6/10/20 views) | No frontier gain |
| 4 | Stability-fixed retrain of (3) at 10× lower LR | Better val loss (0.1794), **still worse on the frontier** |

Intervention 4 is the instructive one: it improved its *own* validation loss yet lost at
every matched precision, and lost again when scored directly against ScanNet GT in the
dense regime (completed-geometry accuracy 11.2 cm vs 13.8 cm). Hence the standing rule in
[`docs/render_quality.md`](docs/render_quality.md): **validation loss is not a proxy for the
frontier — re-score with `scripts/probe_iso_sweep.py` before promoting a checkpoint.**

Reproduce via `train_completer.py --v2 --w_occ … --w_free …`,
`scripts/make_multidensity_crops.py`, and `scripts/probe_iso_sweep.py`.

Four interventions converging on one curve is itself the finding: the ceiling at this
resolution is voxel size and inherent ambiguity, not tuning. The remaining untested lever
is a **96³ run on GPU** (`--device cuda --crop_size 96`) — scripted and data-prepared, but
not executed (no compute access).

## Limitations

Stated plainly, because they bound what the numbers above mean:

- **Resolution.** All results come from a 64³ interim checkpoint trained on a
  16 GB laptop. The 96³ run is scripted and data-prepared but never executed.
- **Precision.** ~40–45% of predicted-solid voxels in the occluded region are wrong.
  Some of that is irreducible (hidden geometry is genuinely ambiguous), not all.
  The operating point is deliberately recall-first — see [Key results](#key-results-64-interim-checkpoint).
- **Poses.** Evaluation uses ScanNet GT camera poses; a deployment would substitute
  a VIO/SLAM front-end and inherit its drift (see [Pose source](#pose-source)).
- **Published-baseline comparisons.** Atlas and NeuralRecon numbers quoted in the
  docs use a different protocol (full-scene, dense video) and are **not** directly
  comparable. The structural claim — 0% vs 37.2% in the occluded region — is the
  defensible one.
- **Scale.** 10 held-out scenes and 90 validation crops: indicative, not conclusive.
- **Planner maturity.** The completed map and its per-voxel confidence are what ship;
  path-level collision avoidance is early-stage.

## Citation

```bibtex
@software{agarwal2026occlusynth,
  author  = {Agarwal, Aditya},
  title   = {OccluSynth: Occlusion-Aware 3D Scene Reconstruction
             in Partially Observable Real-World Environments},
  year    = {2026},
  url     = {https://github.com/onemore-adi/OccluSynth},
  license = {MIT}
}
```

## License

[MIT](LICENSE) © 2026 Aditya Agarwal

Note that dataset licences are separate: ScanNet v2 is non-commercial research-only,
and VGGT-Omega carries its own upstream licence.
