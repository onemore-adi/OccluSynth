# Completer Sprint — 3D Voxel SDF Completion

Status document for the completer chapter. For project-wide context see
repo-root `context.md`; for design rationale see `docs/architecture.md`.
Last updated: 2026-06-13 (interim training complete, eval done, checkpoint on HF).

## What is being built

Given the partial, visibility-aware voxel grid from `fuse_visibility()`
(5 cm voxels, channels `(sdf, weight, p_observed)`, states
`0=unobservable / 1=free / 2=surface / 3=occluded`), predict the **complete
SDF** of the scene — including OCCLUDED voxels behind observed surfaces.

The completer must be **3D**: occluded voxels appear in *no* depth image
(they are behind a measured surface in every frame that sees them), so a 2D
depth-inpainting network cannot recover them by construction. This is the
project's core contribution.

```
Input:  (sdf, weight, p_observed) voxel grid, 96³ crops, 3 channels
Output: completed SDF over the full crop
Loss:   L1(sdf_pred, sdf_gt) masked to (state==SURFACE or state==OCCLUDED),
        UNOBSERVABLE voxels excluded entirely from loss
Target: GT SDF voxelized from _vh_clean_2.ply at same 5 cm grid + world origin
Data:   40 train scenes × random 96³ crops with ≥10% occluded voxels
```

## Component status

| Component | File(s) | Status |
|---|---|---|
| GT SDF voxelizer `mesh_to_tsdf()` | `src/occlusynth/fusion/mesh_to_tsdf.py` | ✅ done |
| Dense-grid helper `fuse_visibility_grid()` / `SceneGrid` | `src/occlusynth/fusion/scene_grid.py` | ✅ done |
| Alignment check (partial grid ↔ GT SDF) | `scripts/check_completer_alignment.py` | ✅ passed — median \|GT SDF\| at surface voxels **2.51 cm** (gate < 7.5 cm), 93.1% within 1.5 voxels; Rerun overlay in `demo_outputs/completer_alignment/` |
| Tests | `tests/test_mesh_to_tsdf.py` (4), `tests/test_completer.py` (8) | ✅ all passing (full suite 55) |
| Training data | `scripts/generate_completer_data.py` → `data/completer_crops/` | ✅ **418 train / 90 val** 96³ fp16 crops, ~770 MB |
| Model + loss | `src/occlusynth/models/completer.py` | ✅ 14.7M-param 3D U-Net, masked L1 |
| Training script | `scripts/train_completer.py` | ✅ MPS debug gate passed (2 epochs, no OOM, loss ↓) |
| Eval script | `scripts/eval_completer.py` | ✅ pipeline validated end-to-end with debug ckpt |
| HF upload script | `scripts/upload_completer_hf.py` | ✅ written; needs `hf auth login` |
| Interim local training (64³, aug, 35 ep, MPS) | `checkpoints/interim_64_aug/completer_best.pt` | ✅ done — best val 0.1857 (ep 32) |
| Final eval table + per-scene .rrd | `demo_outputs/completer_eval/` | ✅ done — completer beats both baselines |
| Checkpoint → HF `onemore-adi/occlusynth-completer` | | ✅ done |
| **Cloud A100 run (96³, 50 ep)** | RunPod | ⬜ **ready, not executed this phase** — script + 418/90 crops prepared; see command below; expected to improve all metrics |

## Key implementation facts

- **Grid alignment contract**: GT SDF is sampled at voxel centres
  `origin + (idx + 0.5) * 0.05` with the *same* origin/dims as the fused grid,
  C-order `indexing="ij"`. Any origin mismatch shows up as a jump in
  `test_surface_at_zero_crossing` — that test is the gate.
- **Padding**: scenes smaller than 96 in any axis (common in z) are
  centre-padded with UNOBSERVABLE (sdf=+1, weight=0, p_obs=0); the GT SDF is
  computed on the padded grid directly, so targets stay exact. Padded voxels
  carry state=0 → excluded from loss.
- **Crop rejection**: occluded_fraction < 0.10, or > 50% exactly-zero GT SDF.
  Six scenes (scene0161_01, 0216_00, 0238_01, 0371_01, 0387_01 train;
  scene0471_01 val) yield **zero crops** — observed coverage too small. This
  is correct behaviour, not a bug. Hence 9 (not 10) val scenes in eval.
- **Split**: deterministic md5 hash (`ScanNetDataset._scene_is_val`,
  val_fraction=0.2) over the 50 scenes with `_vh_clean_2.ply` → 40/10.
  Val crops use a fixed seed (identical every regeneration).
- **Input SDF channel is normalized** to [-1, 1] over the (cosine-tightened)
  surface band; the *target* is metric metres. Baselines denormalise the
  input via `surface_trunc = 0.10`.
- **Storage**: crops are fp16 (`input`, `target`) + uint8 (`state`) with
  provenance keys (`scene`, `corner_ijk`, `world_origin`, `voxel_size`) so
  eval can place crops back into the world frame.
- ScanNet meshes are open surfaces → ray-parity sign can be unreliable far
  from the surface; near the supervised band it is solid.

## Remaining work

1. **A100 run** — fully scripted and ready; training data (418 train / 90 val crops) prepared locally. Not executed this phase due to compute access, not missing work. On an A100:
   ```bash
   pip install -e . && pip install wandb
   python scripts/train_completer.py --device cuda --epochs 50 --batch_size 4 --crop_size 96
   ```
   Local 96³ b=4 fits MPS memory but ~70 s/step (~4+ days) — not viable on MPS. All reported metrics are from the interim 64³ checkpoint (`checkpoints/interim_64_aug/completer_best.pt`, epoch 32, val_loss 0.1857); the 96³ A100 run is expected to improve them further.
2. **Eval**:
   `python scripts/eval_completer.py --device cuda --ckpt checkpoints/completer_best.pt`

   Table format (results.json + stdout) — surface/occluded split is the point:

   | Metric | Surface voxels | Occluded voxels |
   |---|---|---|
   | MAE (cm) | _ | _ |
   | Sign accuracy | _ | _ |
   | Completion ratio (within 5 cm) | _ | _ |

   Baselines (both in the table): **no-completion** (occluded SDF=0) and
   **occluded-as-free** (occluded SDF=+0.1). Success = completer beats both
   on occluded voxels; surface columns should be similar across methods.
3. **Upload**: `python scripts/upload_completer_hf.py` →
   `onemore-adi/occlusynth-completer` (needs HF token).
4. **Next chapter: ✅ DONE** — risk-graded planner (`src/occlusynth/planning/`,
   `scripts/run_planner.py`, `tests/test_planner.py`). A* over the completed
   voxel cost map; no Habitat-Sim. Results on scene0000_00:
   path 13.56 m / 244 cells, 91 occluded on path. See `docs/architecture.md §Risk-Graded Planner`.

## What NOT to do

- No 2D depth-image inpainting — occluded voxels are in no depth image.
- No semantic head yet — geometry-only completion is the MVP.
- Do not train on the full 1,513 ScanNet scenes — 40 is sufficient.
- Do not use Habitat-Sim — the planner runs on reconstructed scenes.
- Do not modify `fuse_visibility()` or the depth calibration code — closed
  chapters. (`scene_grid.py` wraps, never edits.)

## Definition of done

- [x] `mesh_to_tsdf()` implemented and alignment-verified in Rerun
- [x] `test_surface_at_zero_crossing` passing
- [x] 400+ train crops generated, val crops generated
- [x] `OccluSynthCompleter` forward pass tested
- [x] Training loss decreasing after 2 local debug epochs
- [x] Training loss decreasing after 2 local debug epochs
- [x] Eval table showing improvement over both baselines on occluded voxels (64³ interim: occluded MAE 45.27 → 27.14 cm, completion ratio 6.1% → 34.9%)
- [x] One Rerun `.rrd` per (croppable) val scene in `demo_outputs/completer_eval/`
- [x] Checkpoint uploaded to HF as `onemore-adi/occlusynth-completer`
- [ ] Full 50-epoch 96³ A100 run — scripted and ready, not executed this phase due to compute access; `python scripts/train_completer.py --device cuda --epochs 50 --batch_size 4 --crop_size 96`
