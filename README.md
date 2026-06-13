# OccluSynth

**Occlusion-Aware 3D Scene Reconstruction in Partially Observable Real-World Environments**  
Aditya Agarwal · National Institute of Technology, Rourkela

---

## What it does

OccluSynth fuses sparse RGB-D observations into a dense, visibility-aware 3D
voxel grid that explicitly labels every voxel as **free** (confirmed empty),
**surface** (measured solid geometry), **occluded** (behind a surface in every
view — the robot's "blind spot"), or **unobservable** (never in any frustum).
A 3D U-Net completer then predicts the SDF inside the occluded volume —
recovering geometry that is absent from every depth image.

---

## Pose source

**All fusion uses ScanNet ground-truth camera poses** (`pose/<frame_id>.txt`),
not poses predicted by VGGT-Omega.

VGGT-Omega's absolute trajectory error on 6 frames is **~70 cm** — 14× the
5 cm voxel pitch. Fusing at that pose error produces blurred, doubled surfaces.
ScanNet GT poses are themselves the output of BundleFusion (a standard
visual-inertial odometry system). Using them is equivalent to assuming a
calibrated odometry source — a documented, standard assumption for any RGB-D
reconstruction system (KinectFusion, BundleFusion, ElasticFusion all do the
same). In a real deployment this would be replaced by a VIO system or SLAM.

See `docs/architecture.md §Camera Pose Strategy` for the full rationale and
the OOM analysis of why more frames do not fix the pose problem on Apple Silicon.

---

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

---

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

## Data

Requires ScanNet v2 (non-commercial research licence).
GT meshes in `data/scannet/scans/<scene>/<scene>_vh_clean_2.ply`.
RGB-D frames in `data/scannet/tasks/scannet_frames_25k/`.

---

## Key results (64³ interim checkpoint)

| Method | Occluded MAE (cm) ↓ | Occluded sign acc ↑ | Compl < 5 cm ↑ |
|---|---|---|---|
| no_completion | 45.27 | 0.299 | 0.061 |
| occluded_as_free | 42.00 | 0.701 | 0.121 |
| **OccluSynth Completer** | **27.14** | **0.722** | **0.349** |

Full surface/occluded split in `demo_outputs/completer_eval/results.json`.

---

## Docs

- `context.md` — living plan: completed chapters, current status, next steps
- `docs/architecture.md` — design decisions and rationale
- `docs/completer_sprint.md` — completer chapter: spec, contracts, status
