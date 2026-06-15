# OccluSynth — Architecture & Design Decisions

**Project:** Occlusion-Aware 3D Scene Reconstruction in Partially Observable Real-World Environments  
**Author:** Aditya Agarwal, NIT Rourkela

---

## System Overview

OccluSynth fuses sparse RGB-D observations from a moving camera into a dense,
occlusion-aware 3D reconstruction.  The pipeline has three stages:

```
RGB frames
    │
    ▼
┌─────────────────────────────┐
│  VGGT-Omega  (frozen)       │  feed-forward multi-view depth + pose
│  vggt_omega_1b_512.pt       │
└────────────┬────────────────┘
             │  depth maps (arbitrary scale)
             │  camera pose encoding
             ▼
┌─────────────────────────────┐
│  OccluSynth Adapter         │  learned scale correction + occlusion mask
│  (trained on ScanNet 25k)   │
└────────────┬────────────────┘
             │  metric depth maps  +  occlusion confidence
             │  camera poses  (GT or VGGT-estimated — see §Camera Poses)
             ▼
┌─────────────────────────────┐
│  TSDF Fusion                │  voxel size 5 cm
│  (Open3D ScalableTSDFVolume)│
└────────────┬────────────────┘
             │
             ▼
        Dense mesh  +  occlusion-masked regions
```

---

## Camera Pose Strategy

### Current assumption (MVP): Ground-Truth Poses from ScanNet

OccluSynth currently uses the **ScanNet ground-truth camera poses** supplied in
`pose/<frame_id>.txt` for TSDF fusion rather than the poses predicted by
VGGT-Omega.

**Why:**

| Metric | Value |
|--------|-------|
| TSDF voxel resolution | 5 cm |
| Max tolerable pose error (rule of thumb: ½ voxel) | **< 2.5 cm** |
| VGGT-Omega ATE on 6 frames (scene0000_00) | **70.4 cm** |
| VGGT-Omega ATE as % of trajectory | **21 %** |

A 70 cm pose error on a 5 cm voxel grid produces blurred and duplicated
surfaces in TSDF fusion — the reconstruction becomes unusable.  This is not a
flaw in VGGT-Omega; the model was designed for relative scene understanding,
not metric localisation.

**Is this assumption honest?**  
Yes.  The problem statement specifies *sparse depth completion* and
*occlusion-aware reconstruction* — it does not require the system to solve
simultaneous localisation and mapping (SLAM).  Real-world deployments of
RGB-D reconstruction (e.g., the original KinectFusion, BundleFusion, and all
ScanNet-trained models) rely on external odometry sources: wheel encoders, IMU,
visual-inertial odometry, or SLAM systems such as ORB-SLAM3.  ScanNet's GT
poses are themselves the output of BundleFusion.  Using them is equivalent to
assuming a good odometry source is available — a standard and documented
assumption.

**Document this assumption everywhere it matters:**
- This file (done)
- `README.md` — "Pose source" section
- Demo video narration: *"Poses from ScanNet BundleFusion odometry — in
  deployment this would be replaced by a VIO system."*

---

### Option 2 (tested + ruled out): More frames → OOM on Apple Silicon

We attempted to measure VGGT-Omega's ATE as a function of frame count on this
machine (Apple M-series, MPS backend, 512 px resolution):

| Frames | ATE (m) | Traj span (m) | Rel ATE | Status |
|--------|---------|---------------|---------|--------|
| 6      | 0.704   | 3.272         | 21.5 %  | ✓ ran  |
| 12     | —       | —             | —       | ✗ OOM: `Invalid buffer size: 9.52 GiB` |
| 20+    | —       | —             | —       | ✗ OOM (worse) |

**Root cause:** VGGT-Omega's inter-frame attention is O(N² × T²) in frames N
and tokens T.  At 512 px with patch size 16, each frame produces ~1,000 tokens.
12 frames × 1,000² = 144 M attention entries × fp32 = ~9.5 GiB — exceeds MPS
single-buffer limit.

**Conclusion:** Option 2 is hardware-dead on Apple Silicon at 512 px.  Even if
it ran, ATE of 21 % on 6 frames is unlikely to drop below 2 % (sub-5 cm) with
more frames — the model has no metric-scale anchor.  GT poses remain the only
correct MVP choice.

**If you have a CUDA GPU:** test with `--n_frames 20` — the transformer may
fit in 24 GB VRAM and ATE may improve somewhat, but metric-scale grounding
will still require an explicit anchor or loop closure.

---

### Option 3 (post-MVP): Pose refinement

Bundle adjustment (e.g., COLMAP, g2o) or depth-ICP post-processing over
VGGT's predicted poses and depth maps.  Estimated ATE improvement: 5–10×.
Estimated effort: 2–3 days.  **Not required for hackathon MVP.**

---

## Depth Scale

VGGT-Omega outputs depth in an arbitrary internal scale.  Baseline measurement
on scene0000_00 (6 frames, 500 anchors, GT depth at **382×512** colour-camera
frame via ScanNetDataset):

| Metric | Value |
|--------|-------|
| Global scale factor (pred × k = metres) | **7.40** |
| Per-frame RANSAC scale mean ± std | **6.81 ± 0.47** |
| Per-frame RANSAC mean ARE | **0.024** (2.4 %) |
| Per-frame RANSAC mean RMSE | **0.083 m** |
| Per-frame RANSAC δ<1.25 | **99.7 %** |

Numbers are regression-pinned in `tests/test_metric_grounding.py`.
The adapter's supervised signal is replacing the closed-form RANSAC fit with
a learned predictor.  ARE of 2.4 % is the **"before adapter"** baseline.

---

## TSDF Fusion Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Voxel size | 5 cm | Matches ScanNet annotation resolution |
| SDF truncation | 4 × voxel = 20 cm | Standard 4× rule |
| Depth max | 3.5 m | Reliable range of Kinect v1 sensor |
| Depth scale | ~7.4 × per-frame RANSAC | VGGT internal → metres |
| Pose source | ScanNet GT (`pose/*.txt`) | See §Camera Pose Strategy |

Two fusion paths in `src/occlusynth/fusion/tsdf.py`:
- `fuse()` — surface reconstruction (open3d marching-cubes mesh on `.venv312`; numpy point-cloud fallback)
- `fuse_visibility()` — visibility-aware dense voxel grid (below)

---

## Visibility-Aware Voxel Grid

`fuse_visibility()` builds a dense 5 cm grid over the scene bbox (camera centres +
back-projected surfaces + 15 cm pad) with three per-voxel channels: **sdf**,
**weight**, **p_observed**.

**Integration is projective** — the analytic, vectorised equivalent of per-pixel
DDA ray-casting. For every voxel we project its centre into each camera and
compare its camera-Z against the measured depth at that pixel:

| Condition | Meaning | Action |
|-----------|---------|--------|
| `sdf = d − z > +trunc_eff` | voxel in front of surface | **free** evidence (ray passes through) |
| `\|sdf\| ≤ trunc_eff` | voxel straddles surface | TSDF update + **surface** evidence |
| `sdf < −trunc_eff` | voxel behind surface | **occluded** evidence (this frame) |
| projects outside image / behind camera / onto invalid pixel | no observation | no evidence |

**Obliquity correction (important).** `sdf = d − z` is measured along the optical
axis, so a fixed band `|sdf| ≤ trunc` balloons in world space for surfaces viewed
off-normal — stretched by 1/cosθ, a grazing wall becomes ~0.5 m of false surface,
stealing voxels from free/occluded supervision and teaching a completer that walls
are half a metre thick. We **cosine-tighten** the band per voxel:
`trunc_eff = surface_trunc · |n·r̂|` (clamped to ≥ 0.5), with `surface_trunc = 2 ×
voxel = 10 cm` and `n` the local depth-map normal. Tightening only ever *removes*
voxels from the surface band, so noisy depth-normals are safe (worst case = the
untightened band). Far-wall band on scene0000_00: **25 cm → 15 cm** (5 → 3 voxels);
surface share of observable volume: **20.7 % → 6.3 %**.

Per voxel we accumulate free / surface / occluded counts across all frames, then
classify by priority **surface > free > occluded**. The critical distinction:

- **OCCLUDED** — received occluded evidence but never free/surface: inside a
  frustum yet always behind a measured surface. *These are the completer's
  inpaint targets.*
- **UNOBSERVABLE** — zero evidence of any kind: never inside a valid-depth
  frustum. The completer must **leave these alone** (no information to recover).

`p_observed = (free + surface) / (free + surface + occluded)` ∈ [0, 1] is the soft
visibility channel; 0 for fully-occluded, undefined→0 for unobservable.

**scene0000_00, 6 GT-depth frames:** free 104.0k · surface 20.1k (**6.3 % of
observable**) · occluded 194.9k (**61 % of observable**) · unobservable 909.6k.

Run: `.venv312/bin/python scripts/run_visibility.py --scene scene0000_00 --use_gt_depth`
→ coloured voxel PLY + Rerun `.rrd` + cross-section PNG (`docs/images/visibility_*.png`).
Green = free, red = surface (solid), amber = occluded (the volume the robot must
imagine — the planner detours around these). Pinned in `tests/test_visibility.py`.

---

## 3D Voxel Completer

`OccluSynthCompleter` in `src/occlusynth/models/completer.py`.

### The problem it solves

OCCLUDED voxels (in a camera frustum, permanently behind a measured surface)
never appear in any depth image.  A 2D depth-inpainting network cannot recover
them — they are not missing pixels, they are missing *rays*.  The completer
operates in 3D on voxel-grid crops and explicitly targets this class.

### Architecture

Encoder-decoder 3D U-Net with skip connections.

| Component | Detail |
|---|---|
| Parameters | **14.7 M** |
| Input | `(B, 3, D, H, W)` — sdf (normalised), weight, p_observed |
| Output | `(B, 1, D, H, W)` — completed SDF in metres |
| Encoder | 4 blocks, channels [32, 64, 128, 256], stride-2 Conv3d |
| Decoder | 4 blocks, trilinear upsample + skip concat |
| Norms / act | GroupNorm(8) + GELU |
| Head | 1×1×1 Conv3d, no activation (unbounded SDF) |
| Crop size | 96³ at 5 cm |

**Loss:** `masked_l1_loss(pred, target, state)` — L1 on SURFACE ∪ OCCLUDED
voxels only.  UNOBSERVABLE is excluded entirely.

### Supervision target

GT SDF from `mesh_to_tsdf()` (`src/occlusynth/fusion/mesh_to_tsdf.py`) via
open3d `RaycastingScene`, sampled on voxel centres
`origin + (idx + 0.5) * 0.05 m` — the **identical** origin/dims as the
partial grid from `fuse_visibility()`.

**Alignment gate:** before generating any training data, verified on
scene0000_00: median |GT SDF| at surface voxels = **2.51 cm** (threshold
7.5 cm), 93.1% within 1.5 voxels of the GT zero-crossing.
`test_surface_at_zero_crossing` pins this for regressions.

### Training data

40 train / 10 val ScanNet scenes (deterministic md5 split), 96³ fp16 crops,
≥10% occluded fraction, rejection for >50% exactly-zero GT SDF (degenerate
mesh region).  418 train / 90 val crops (~770 MB).  Val crops use a fixed
seed — byte-identical across regenerations so all val numbers are comparable.

**Augmentation** (train-only, `--augment`): 4 yaw rotations about z × optional
x-flip = 8 variants (dihedral group D4).  z is never flipped — flipping z
puts ceilings below floors and poisons the gravity-aligned priors the
completer must exploit.  All four arrays (3 input channels, target, state)
receive the identical transform in the same `__getitem__` call; 22 tests pin
this, including a coordinate-identity test that makes an orientation mismatch
impossible to hide.

### Evaluation (64³ interim, 35 epochs + augmentation)

| Method | Class | MAE (cm) ↓ | Sign acc ↑ | Compl < 5 cm ↑ |
|---|---|---|---|---|
| no_completion (SDF = 0) | surface | 7.65 | 0.432 | 0.509 |
| no_completion | **occluded** | 45.27 | 0.299 | 0.061 |
| occluded_as_free (SDF = +0.1) | surface | 7.65 | 0.432 | 0.509 |
| occluded_as_free | **occluded** | 42.00 | 0.701 | 0.121 |
| **OccluSynth Completer** | surface | **4.86** | **0.585** | **0.768** |
| **OccluSynth Completer** | **occluded** | **27.14** | **0.722** | **0.349** |

Completer beats both baselines on every occluded metric; surface columns are
a sanity check (similar across methods = not breaking observed geometry).
Full table in `demo_outputs/completer_eval/results.json`.

Scripts: `scripts/generate_completer_data.py`, `scripts/train_completer.py`,
`scripts/eval_completer.py`.  Checkpoint: `checkpoints/interim_64_aug/completer_best.pt`.

---

## Data

| Dataset | Split | Location |
|---------|-------|----------|
| ScanNet v2 — frames_25k | 1,513 scenes | `data/scannet/tasks/scannet_frames_25k/` |
| ScanNet label map | full | `data/scannet/scannetv2-labels.combined.tsv` |
| Full .sens / meshes | not downloaded | needed only for fusion evaluation |

Per-scene structure: `color/` (JPG), `depth/` (PNG uint16 mm), `pose/` (4×4
c2w float TXT), `intrinsics_color.txt`, `intrinsics_depth.txt`, `label/`,
`instance/`.

---

## Baseline (Before Adapter)

Run: `python scripts/run_scale_fit.py --save_grounding`  
Multi-scene: `python scripts/run_multi_scene_eval.py --n 10`  
Results cached in `demo_outputs/multi_scene_eval/results.json`.

### Single scene (scene0000_00, train split)

- Depth ARE (per-frame RANSAC): **0.024** | scale **6.81 ± 0.47** | δ<1.25 **99.7%**
- Camera ATE: **0.704 m** (unusable for TSDF — GT poses used instead)

### Multi-scene validation (10 val scenes, 6 frames each, GT at 382×512)

| Scene | mean ARE ↓ | max ARE ↓ | RMSE (m) ↓ | δ<1.25 ↑ | Scale a | Flag |
|-------|-----------|---------|----------|--------|---------|------|
| scene0001_00 | 0.026 | 0.042 | 0.105 | 99.0% | 2.93±0.21 | OK |
| scene0085_00 | 0.017 | 0.020 | 0.065 | 99.8% | 2.27±0.11 | OK |
| scene0146_00 | 0.017 | 0.046 | 0.097 | 98.9% | 2.06±0.06 | OK |
| scene0220_01 | 0.021 | 0.031 | 0.102 | 99.5% | 2.75±0.07 | OK |
| scene0301_02 | 0.029 | 0.058 | 0.113 | 98.1% | 1.83±0.10 | OK |
| scene0367_00 | 0.025 | 0.029 | 0.129 | 98.4% | 2.16±0.07 | OK |
| scene0471_01 | 0.017 | 0.048 | 0.056 | 98.8% | 1.53±0.03 | OK |
| scene0556_00 | 0.027 | 0.048 | 0.105 | 98.6% | 4.04±0.08 | OK |
| scene0635_01 | 0.030 | 0.040 | 0.075 | 98.8% | 1.81±0.30 | OK |
| scene0704_00 | 0.032 | 0.049 | 0.098 | 98.3% | 2.74±0.21 | OK |
| **Aggregate** | **0.024** | **0.058** | **0.094 m** | **98.8%** | — | 10/10 OK |

No pathological scenes detected. Scale factor `a` varies by scene (1.5–4.1 for val; ~6.8 for scene0000_00) — VGGT's output scale is not normalised across scenes, which is why the adapter must predict per-scene (a, b) rather than using a fixed global scalar. Regression-pinned in `tests/test_multi_scene.py`.

---

## Risk-Graded Planner

### Overview

The planner converts the completed 3D voxel grid into a 2D floor-plan cost map and
finds the minimum-cost path using 8-connected A*. Risk is derived directly from
the completed SDF — no Habitat-Sim, no navigation mesh, no prebuilt map.

**Source:** `src/occlusynth/planning/astar_planner.py`
**Script:** `scripts/run_planner.py`
**Tests:** `tests/test_planner.py` (18 tests)

### Cost map construction

The 3D state grid (axes: x, y, z with z = vertical/gravity) is collapsed to
a 2D floor-plan (nx, ny) by inspecting the **robot height band** — a z-slice
between `robot_height_lo` and `robot_height_hi` metres above the lowest FREE
voxel in the grid.  Default band: **0.10–0.50 m**.

Per (x, y) column in the height band:

| Column content | Cost |
|---|---|
| Any SURFACE voxel | `inf` — impassable wall |
| Any OCCLUDED voxel | `1.0 + λ × p_occupied` |
| Any FREE, no OCCLUDED | `1.0` — confirmed traversable |
| All UNOBSERVABLE | `6.0` — never in any frustum |

`p_occupied` = fraction of OCCLUDED voxels in the column whose completed
SDF < 0 (negative SDF = inside the mesh = physically occupied).
`λ` = `lambda_risk` (default **4.0**).

Monotonicity chain: `1.0 ≤ 1.0+λ·p_occ ≤ 5.0 < 6.0 < inf`.
A fully-occluded worst-case column (λ=4, p=1) costs 5.0 — still cheaper
than unknown (6.0) because at least we have a prediction.

### A* search

8-connected grid search.  Edge cost = move distance × destination cell cost
(diagonal move = √2 × cost; cardinal = 1 × cost).
Heuristic = Euclidean distance in voxel units (admissible: min finite cost ≥ 1.0).

Default start/goal: `farthest_free_pair()` — double-BFS over the traversable
subgraph (hop-count diameter approximation).  Override with `--start I J --goal I J`.

### Results on scene0000_00

```
Grid 146×153×96  voxel 5 cm
Traversable cells: 20,538  Wall cells: 1,800
Start (145,152) → Goal (0,16)
Path cells   : 244
Geom length  : 13.56 m
Path cost    : 673.5
Occluded cells on path: 91 / 244  (37 %)
Robot height band: z_idx 26–34  (0.10–0.50 m above floor)
```

Outputs: `docs/images/planner_scene0000_00.png` (cost-map heatmap + white path),
`demo_outputs/planner_scene0000_00.rrd` (Rerun 3D viewer).

### Configuration (`PlannerConfig`)

| Parameter | Default | Meaning |
|---|---|---|
| `lambda_risk` | 4.0 | Risk weight on p_occupied for occluded cells |
| `robot_height_lo` | 0.10 m | Lower bound of robot body band above floor |
| `robot_height_hi` | 0.50 m | Upper bound of robot body band above floor |

Run: `.venv312/bin/python scripts/run_planner.py --scene scene0000_00 [--lambda_risk 4.0]`
