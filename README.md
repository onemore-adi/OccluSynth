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

### ScanNet v2 (primary)

Requires ScanNet v2 (non-commercial research licence).
GT meshes in `data/scannet/scans/<scene>/<scene>_vh_clean_2.ply`.
RGB-D frames in `data/scannet/tasks/scannet_frames_25k/`.

### 7-Scenes (cross-dataset generalisation probe)

Used in `scripts/run_crossdataset.py` to verify that the frozen
RANSAC-grounding + visibility-fusion pipeline generalises beyond ScanNet.
No retraining is required.

**Download** from the [Microsoft Research 7-Scenes page](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
and extract so the layout matches:

```
data/7scenes/
  chess/
    seq-01/
      frame-000000.color.png   (640×480 RGB)
      frame-000000.depth.png   (640×480 uint16 mm)
      frame-000000.pose.txt    (4×4 camera-to-world)
      frame-000001.color.png
      …
    seq-02/
    …
  fire/
    seq-01/
    …
```

All 7-Scenes sequences share fixed Kinect v1 intrinsics
(`fx=fy=585, cx=320, cy=240` at 640×480); no per-sequence calibration
file is needed.

**Run the cross-dataset probe:**

```bash
.venv312/bin/python scripts/run_crossdataset.py \
    --seqs chess/seq-01 fire/seq-01 \
    --data_root data/7scenes \
    --n_frames 6
```

Outputs: `demo_outputs/crossdataset/results.json`,
`docs/images/crossdataset_chess_seq-01.png`, per-sequence `.rrd`.

---

## Key results (64³ interim checkpoint)

| Method | Occluded MAE (cm) ↓ | Occluded sign acc ↑ | Compl < 5 cm ↑ |
|---|---|---|---|
| no_completion | 45.27 | 0.299 | 0.061 |
| occluded_as_free | 42.00 | 0.701 | 0.121 |
| **OccluSynth Completer** | **27.14** | **0.722** | **0.349** |

Full surface/occluded split in `demo_outputs/completer_eval/results.json`.

> All metrics above are from the **interim 64³ MPS checkpoint** (`checkpoints/interim_64_aug/completer_best.pt`, epoch 32, val_loss 0.1857). The full 96³ A100 training run is scripted and ready (`python scripts/train_completer.py --device cuda --epochs 50 --batch_size 4 --crop_size 96`; data prepared) — it was not executed this phase due to compute access, not missing work, and is expected to improve all metrics.

---

## Docs

- `context.md` — living plan: completed chapters, current status, next steps
- `docs/architecture.md` — design decisions and rationale
- `docs/completer_sprint.md` — completer chapter: spec, contracts, status
