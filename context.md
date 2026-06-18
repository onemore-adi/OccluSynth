# OccluSynth — Development Context

**Project:** Occlusion-aware 3D scene reconstruction using VGGT-Omega (Facebook) + ScanNet v2.  
**Repo:** https://github.com/onemore-adi/OccluSynth  
**Machine:** Apple Silicon MPS (no CUDA). Two Python environments:

**Two Python environments:**

| Venv | Python | Use |
|------|--------|-----|
| `.venv` | 3.14.3 | VGGT inference (MPS), metric grounding, all existing code |
| `.venv312` | 3.12.13 ARM | open3d 0.19 TSDF + marching cubes, rerun 0.33 viewer, completer training (also MPS) |

Activate: `source .venv/bin/activate` or `source .venv312/bin/activate`.

**open3d fusion verified:** `scene0000_00_n6_gtdepth_gtpose_mesh.ply` — 15,676 verts / 27,122 tris in 2.1s.
VGGT depth mesh: 19,533 verts / 33,755 tris — more geometry because VGGT fills in GT-invalid pixels (expected).
Both meshes at `demo_outputs/tsdf_fusion/`.  

---

## What is built and working

### Package: `src/occlusynth/` (pip install -e .)

```
src/occlusynth/
  data/
    scannet.py          ScanNetScene dataclass + loaders (depth/pose/K/frames)
    sparse_sampler.py   stratified 500-anchor sampling for depth calibration
  models/
    vggt_wrapper.py     VGGTWrapper: lazy load, .npy cache, predict_cached()
    depth_calibration.py  4 scale fits + evaluate() — see "Known Bug Fixed" below
    adapter.py          MLP scaffold: token(2048) → (scale, shift) — NOT TRAINED YET
  fusion/
    tsdf.py             GT-pose TSDF; numpy ASCII PLY fallback (no open3d on py3.14)
  utils/
    device.py           get_device() / get_config() / get_repo_root()
  viz/
    rerun_viewer.py     colorize() + Rerun SDK viewer (graceful fallback)
```

### Scripts (orchestration only, all logic in src/)

| Script | Purpose |
|--------|---------|
| `scripts/run_baseline.py` | "Before adapter" overview: RGB | GT depth | VGGT depth |
| `scripts/run_scale_fit.py` | Benchmark 4 scale methods; saves bar chart |
| `scripts/run_fusion.py` | TSDF fusion → PLY point cloud |

All three run end-to-end. Default scene: `scene0000_00`, 6 frames.

---

## Data & Model Locations

| Item | Path |
|------|------|
| ScanNet frames_25k | `data/scannet/tasks/scannet_frames_25k/<scene>/` |
| ScanNet GT meshes | `data/scannet/scans/<scene>/<scene>_vh_clean_2.ply` |
| ScanNet GT meshes (labelled) | `data/scannet/scans/<scene>/<scene>_vh_clean_2.labels.ply` |
| VGGT-Omega source | `vggt/vggt-omega/` (not in git — external dep) |
| VGGT checkpoint | `vggt/vggt-omega/checkpoints/vggt_omega_1b_512.pt` (4.3 GB) |
| Prediction cache | `demo_outputs/pred_cache/scene0000_00_91593293_res512_*.npy` |

**GT mesh download** (`data/scannet/scans/`) — 50 scenes being downloaded (40 train + 10 val, ~5 GB total):
```bash
python scripts/download_scannet_subset.py --out_dir data/scannet --mode scenes \
  --file_types _vh_clean_2.ply _vh_clean_2.labels.ply --yes \
  --scene_ids scene0000_00 scene0020_01 ...  # see list in download script
```
Re-runnable: skips complete files, retries failed ones (TUM server drops connections).

**Cache files** (6 × `.npy`, all must exist for cache hit):
`_depth (6,448,592)`, `_conf`, `_pose_enc (6,9)`, `_extrinsics (6,3,4)`, `_intrinsics (6,3,3)`, `_hw (2,)`.

**ScanNet GT depth** is uint16 PNG in **mm** → divide by 1000 → metres, clip at 3.5 m (invalid = 0.0).

---

## Architecture Decisions (documented in docs/architecture.md)

### 1. GT poses always used for fusion
VGGT ATE = 70 cm on a 3.27 m trajectory. TSDF voxels = 5 cm → pose error tolerance < 2.5 cm.
VGGT poses are unusable for fusion. ScanNet GT poses are always passed to `fuse()`.

### 2. MPS OOM at 12+ frames
VGGT attention is O(N²·tokens). At 12 frames → 9.52 GiB buffer → MPS hard limit.  
Safe limit: **6 frames** per inference on 16 GB M-series. Cache one batch, reuse it.  
Timing (MPS, M-series): model load **~15 s**, inference 6 frames **~14–23 min** (solo / under contention).  
Multi-scene eval (10 scenes): ~2–4 h wall time; cached on disk, subsequent runs are instant.

### 3. Depth scale fitting (4 methods)
VGGT outputs raw depth in arbitrary scale (~0.16–0.37 for indoor scenes where GT is 1.3–2.9 m).
Fit `d_metric = a × d_pred + b` using 500 stratified anchor pixels per frame.

**Results on scene0000_00 (6 frames, 500 anchors, GT depth at 382×512 colour-camera frame):**
| Method | mean ARE ↓ | mean RMSE ↓ | δ<1.25 ↑ |
|--------|-----------|------------|---------|
| GlobalScalar | 0.060 | 0.147 m | 96.8% |
| PerFrameScalar | 0.035 | 0.104 m | 99.7% |
| PerFrameLS | 0.026 | 0.085 m | 99.7% |
| **PerFrameRANSAC** | **0.024** | **0.083 m** | **99.7%** |

RANSAC best. Per-frame shift (`b`) captures depth-dependent bias.
Regression-pinned in `tests/test_metric_grounding.py` (mean ARE < 0.026).

---

## Critical Bug Fixed: Python 3.14 + NumPy Buffer Stealing

**Symptom:** `evaluate()` in `depth_calibration.py` returned ARE=correct, RMSE=39423, δ<1.25=49% when called inside a function, but correct values when the same code ran inline.

**Root cause:** Python 3.14 uses `LOAD_FAST_BORROW` — loads local variables onto the evaluation stack **without incrementing their refcount**. NumPy's ufuncs check `refcount == 1` as a signal to reuse the input buffer as the output ("buffer stealing"). So `rel_err = np.abs(scaled - g) / g` caused numpy to write `scaled - g` back into `scaled`'s buffer. Subsequent `(scaled - g)**2` then computed `(scaled_corrupted - g)**2 = (original_diff - g)**2` → catastrophically wrong.

**Fix:** `scaled.flags.writeable = False` immediately after creation. NumPy cannot steal a non-writable buffer; it always allocates a fresh output array.

```python
# In evaluate() — src/occlusynth/models/depth_calibration.py
scaled = np.maximum(a * p + b, 1e-6)
scaled.flags.writeable = False    # ← CRITICAL: prevents Python 3.14 LOAD_FAST_BORROW aliasing
diff = scaled - g
```

**This bug will affect any function in Python 3.14 that:**
1. Creates a numpy array as a local variable  
2. Uses it multiple times in arithmetic expressions  
3. Gets a suspiciously wrong result on only some metrics (the first-computed one is correct)

---

## Completed (metric grounding chapter closed)

- ✓ `ScanNetDataset` at 382×512, INTER_NEAREST depth, scaled K
- ✓ Per-frame RANSAC affine fit — ARE 0.024, δ<1.25 99.7% on scene0000_00
- ✓ Multi-scene eval: 10 val scenes, mean ARE 0.024, 0/10 DEGRADE
- ✓ Noise/anchor ablation: stable to σ=0.10 m; 100 anchors = 500 anchors
- ✓ TSDF fusion with open3d (`.venv312`): marching-cubes mesh, both GT and VGGT depth paths
- ✓ GT meshes: 50 scenes downloaded to `data/scannet/scans/` (99/100 PLYs; labels.ply for scene0704_00 pending TUM server)

**adapter.py is Phase 2 future work only.** The closed-form RANSAC fit is the shipping component.
See `docs/adapter_design.md` §5 and §8 for rationale.

---

## Next Steps (in priority order)

### 1. Visibility-aware voxel grid ✓ DONE
`fuse_visibility()` in `src/occlusynth/fusion/tsdf.py` — dense 5 cm grid, channels
(sdf, weight, p_observed). Projective TSDF (DDA-equivalent) carves free space, marks
the surface band, and separates OCCLUDED (in frustum, behind surface → inpaint target)
from UNOBSERVABLE (out of frustum → leave alone).
- **Obliquity correction**: surface band cosine-tightened per voxel (`trunc_eff = surface_trunc·|n·r̂|`,
  surface_trunc=2×voxel=10cm) so grazing walls don't balloon by 1/cosθ. Far-wall band 25→15cm; surface share 20.7%→6.3%.
- scene0000_00 6 frames: free 104.0k · surface 20.1k (6.3% of observable) · occluded 194.9k (61%) · unobservable 909.6k
- Run: `.venv312/bin/python scripts/run_visibility.py --scene scene0000_00 --use_gt_depth`
- Outputs: voxel PLY + Rerun `.rrd` + cross-section PNG (`docs/images/visibility_*.png`)
- Green=free, red=surface (solid), amber=occluded (what the robot imagines). Pinned in `tests/test_visibility.py` (8 tests).

### 2. 3D Voxel Completer ✓ DONE (interim 64³ checkpoint; A100 run improves numbers)
- ✓ `mesh_to_tsdf()` (`src/occlusynth/fusion/mesh_to_tsdf.py`) — GT SDF on the
  fused grid's origin/dims; alignment on scene0000_00: median |GT SDF| at
  surface voxels 2.51 cm, 93% within 1.5 voxels. `tests/test_mesh_to_tsdf.py`.
- ✓ 418 train / 90 val crops (`scripts/generate_completer_data.py`,
  `data/completer_crops/`); scene0471_01 + 5 train scenes yield 0 crops
  (observed region too small for the ≥10% occluded rule).
- ✓ `OccluSynthCompleter` 14.7M-param 3D U-Net + masked L1
  (`src/occlusynth/models/completer.py`, `tests/test_completer.py`).
- ✓ `scripts/train_completer.py` — MPS debug gate passed (2 epochs, loss ↓).
  96³ b=4 fits MPS memory but ~70 s/step (~4+ days) — not viable on MPS. The full 96³ run is scripted and ready (418 train / 90 val crops prepared); it was not executed this phase due to compute access, not missing work. Single command on an A100:
  `python scripts/train_completer.py --device cuda --epochs 50 --batch_size 4 --crop_size 96`
  All reported metrics are from the interim 64³ checkpoint (`checkpoints/interim_64_aug/completer_best.pt`, epoch 32, val_loss 0.1857); the 96³ run is expected to improve them further.
- ✓ `scripts/eval_completer.py` — baselines + surface/occluded split, validated
  end-to-end; `scripts/upload_completer_hf.py` ready (needs HF token).

Input: (sdf, weight, p_observed) voxel grid, 96³ crops, 3 channels
Output: completed SDF over the full crop
Loss: L1(sdf_pred, sdf_gt) masked to (state==SURFACE or state==OCCLUDED),
      UNOBSERVABLE voxels excluded entirely from loss
Target: GT SDF voxelized from _vh_clean_2.ply at same 5 cm grid + world origin
Training data: 40 train scenes × random 96³ crops with ≥10% occluded voxels

### 3. Risk-Graded Planner ✓ DONE
- ✓ `src/occlusynth/planning/` subpackage: `PlannerConfig`, `build_cost_map()`,
  `astar()`, `farthest_free_pair()`, `path_geom_length()`
- ✓ 2D cost map collapsing z over robot height band (0.10–0.50 m above lowest FREE
  voxel); columns classified SURFACE→inf, OCCLUDED→1+λ·p_occ, FREE→1, UNOBS→6
- ✓ 8-connected A* with Euclidean heuristic; edge cost = move_dist × dest_cost
- ✓ `scripts/run_planner.py` — loads cached SceneGrid, runs planner, saves PNG
  heatmap + Rerun .rrd
- ✓ `tests/test_planner.py` — 18 tests covering cost monotonicity, no-blocked-cell
  on path, start/goal connectivity, regression detour; all passing
- ✓ End-to-end on scene0000_00 (GT SDF): path 13.56 m / 244 cells; detour clearly
  visible in `docs/images/planner_scene0000_00.png`
- ✓ "Risk-Graded Planner" section added to `docs/architecture.md`

### 4. MC Dropout Uncertainty ✓ DONE
- ✓ `OccluSynthCompleter(mc_dropout=True)` — Dropout(p=0.2) after each decoder block,
  gated by flag; no new weight keys → existing checkpoints load with strict=True
- ✓ `predict_with_uncertainty(model, inp, n_samples=16)` — context-managed MC sampling
  (Welford online mean/var), returns (mean_sdf, std_sdf, p_occ)
- ✓ `tests/test_uncertainty.py` — 10 tests (std ≥ 0, p_occ ∈ [0,1], checkpoint compat,
  std concentration); all passing with real checkpoint + val data
- ✓ `scripts/eval_calibration.py` — ECE + reliability diagram PNG, honest reconciliation
  with claimed ECE < 0.05; actual value measured on interim 64³ checkpoint
- ✓ `build_cost_map()` gains optional `p_occ_volume` param (calibrated MC p_occ replaces
  sdf<0 sign estimate for OCCLUDED columns)
- ✓ `run_planner.py --use_uncertainty` — tiled completer inference over full scene grid,
  passes p_occ_volume to cost map; falls back gracefully

### 5. Occlusion Safety Benchmark ✓ DONE
- ✓ `scripts/run_safety_benchmark.py` — scene-level benchmark on all 10 val scenes
  via tiled completer inference (96×96 tiles); Metric 1 (hazard awareness rate) +
  Metric 2 (planner collision-avoidance rate); output table + JSON
- ✓ `tests/test_safety_benchmark.py` — 6 tests (hazards non-empty, fraction plausible,
  baselines=0 by construction, completer>0 on val crops); all passing
- ✓ `docs/safety_benchmark.md` — full spec, reproducible split, results table, design
  rationale (why ScanNet not Habitat-Sim)
- Results (interim checkpoint): baselines 0.000 awareness (structural), OccluSynth
  21.3% aggregate awareness; 15.5% collision avoidance on scene0556_00
- Result in `demo_outputs/safety_benchmark/results.json`

### 6. Demo video
Record: (a) raw VGGT depth, (b) RANSAC-calibrated depth, (c) GT depth, (d) open3d mesh,
(e) visibility voxel grid (green/red/amber — the money shot).
Show the **RANSAC calibration** improving depth scale (not an adapter — name it correctly).

---

## Quick Commands

```bash
cd /Users/onemore_adi/OccluSynth
source .venv/bin/activate

# Run all three pipelines (uses cached VGGT predictions — no GPU needed)
python scripts/run_baseline.py
python scripts/run_scale_fit.py
python scripts/run_fusion.py --use_gt_depth   # GT depth
python scripts/run_fusion.py                  # VGGT depth (RANSAC calibrated)

# Different scene or more frames
python scripts/run_scale_fit.py --scene scene0231_00 --n_frames 6
```

---

## Package Layout Reference

```toml
# pyproject.toml key settings
[tool.occlusynth]
vggt_omega_src     = "vggt/vggt-omega"
default_checkpoint = "vggt/vggt-omega/checkpoints/vggt_omega_1b_512.pt"
scannet_frames_25k = "data/scannet/tasks/scannet_frames_25k"
cache_dir          = "demo_outputs/pred_cache"
```

Key constants:
- `DEPTH_MM_SCALE = 1000.0` — ScanNet PNG → metres
- `DEPTH_MAX_M = 3.5` — Kinect v1 reliable range
- `RANSAC_ITERS = 500`, `RANSAC_THR = 0.10` (10% relative error threshold)
- VGGT resolution: 512 px longest side → cached output `(448, 592)`; ScanNetDataset resizes colour to `(382, 512)` and GT depth to `(382, 512)` via INTER_NEAREST

---

## Files NOT in git (local only)

- `data/` — ScanNet frames (too large)
- `vggt/` — VGGT-Omega source + 4.3 GB checkpoint (external dep)
- `demo_outputs/` — cached predictions and generated outputs
- `.venv/` — Python virtual environment
