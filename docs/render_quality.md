# Render quality: what moves it and what doesn't

Post-finals feedback on the demo renders: grounds not flat, holes left open,
walls noisy, space behind furniture not closed, occasional ballooning. This
note records the experiments run against that feedback (2026-07-28) and the
settings that actually improve the meshes, so the conclusions don't have to
be re-derived.

## TL;DR

1. **Completer hyperparameters are not the bottleneck.** Four checkpoints —
   v1 `interim_64_aug` (shipping), `v2_masktrunc_64_aug`, `md_64_aug`
   (3.6× training data), and a v2 fine-tune with the occupancy/free-space
   loss terms enabled — all sit on the **same occluded precision/recall
   frontier** (precision caps at 0.55–0.62 at every operating point).
   Loss reweighting moves the model *along* the frontier, not past it.
2. **Fusion density dominates.** Same scene, same completer, same iso level:
   at `--n_frames 6` 85 % of the volume is unobservable and the render is
   fragmented shards plus large hallucinated blobs; at `--n_frames 40`
   (58 % unobservable) floors come out flat, walls continuous, and completed
   geometry drops to a modest supporting role. Render demos from the densest
   available grid.
3. **The marching-cubes iso level is a free polish dial**
   (`export_completed_mesh.py --iso`, applied to *completed* voxels only —
   observed surfaces are untouched). Solid = `pred < iso`:
   - positive iso → grows predicted solids: closes holes / seals furniture
     backs, at the risk of bulging;
   - negative iso → shrinks them: tighter, more conservative geometry.
   ±0.01–0.02 m is the useful range. The full precision/recall sweep comes
   from a single inference pass via `scripts/probe_iso_sweep.py`.

## Evidence (90 val crops, occluded region)

| checkpoint | best-F1 iso | precision | recall | free-violation @ iso 0 |
|---|---|---|---|---|
| v1 `interim_64_aug` (shipping) | +0.02 | 0.556 | 0.309 | 8.7 % |
| `v2_masktrunc_64_aug` | +0.02 | 0.520 | 0.432 | 32.2 % |
| `md_64_aug` (3.6× data) | +0.02 | 0.520 | 0.282 | 7.4 % |
| v2 + w_occ 0.5 / w_free 0.2 (10-ep fine-tune) | +0.02 | 0.497 | 0.277 | **0.95 %** |

- `w_free` (observed-free-space hinge) works exactly as designed — it cut
  free-space violations from 32 % to under 1 % — but pays for it in recall:
  a different operating point on the same frontier, not a better model.
- D4 test-time augmentation (averaging the 8 yaw/flip variants the model was
  trained with) gives a small surface-L1 gain on v1 and nothing on v2; not
  worth the 8× inference cost.
- ~40 % of predicted-solid occluded voxels are wrong for *every* model —
  the honest ceiling of predicting geometry no camera saw. Getting past it
  needs different supervision or architecture, not loss weights.

## Recommended demo render recipe

```bash
# densest grid available for the scene; iso slightly positive to close holes
.venv312/bin/python scripts/export_completed_mesh.py \
    --scene scene0000_00 --n_frames 40 \
    --ckpt checkpoints/interim_64_aug/completer_best.pt \
    --iso 0.01
```

If bulging bothers more than holes, use `--iso -0.01` instead. v2 checkpoints
render via `--v2`. Note scene0635_01 only has an `n6` grid — regenerate a
denser one (`scripts/generate_completer_data.py`) before using it in renders.

## Reproducing the analysis

```bash
.venv312/bin/python scripts/probe_iso_sweep.py \
    checkpoints/interim_64_aug/completer_best.pt --tta
```

Training-side knobs used in the grid are exposed as
`train_completer.py --w_occ --w_free --trunc --w_near --free_margin`.
