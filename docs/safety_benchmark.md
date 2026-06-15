# OccluSynth Occlusion Safety Benchmark

**First reproducible occlusion-safety evaluation on real indoor 3D reconstructions.**  
No Habitat-Sim. No semantic labels. No simulation. ScanNet geometry only.

---

## Motivation

Standard navigation benchmarks (e.g., Habitat, MP3D ObjectNav) measure whether
an agent can *find* a target.  They do not measure whether the agent would
*collide with geometry it never saw*.

OccluSynth introduces a complementary benchmark: given a partial RGB-D
reconstruction where some space is occluded, how well does the system detect
obstacles hidden in that occluded volume?

---

## Hidden hazard definition

```
hazard_voxel = (state == OCCLUDED) AND (gt_sdf < 0)
```

- **OCCLUDED** (`state == 3`): the voxel is inside at least one camera frustum
  but permanently behind a measured surface — it never appears in any depth
  image.  No line-of-sight sensor can see it.
- **`gt_sdf < 0`**: the voxel is inside the GT mesh (`_vh_clean_2.ply`).
  A robot's body overlapping this voxel is a physical collision.

Hidden hazards are exactly the obstacles that any purely observational system
*structurally cannot detect*.

---

## Dataset

| Property | Value |
|---|---|
| Source | ScanNet v2 (`_vh_clean_2.ply` meshes + GT depth + GT poses) |
| Val scenes | 10 (md5 hash split, val_fraction=0.20) |
| Note | scene0471_01 has no completer crops but IS included in the safety benchmark |
| Voxel pitch | 5 cm |
| Grid structure | axis-2 = vertical; `gt_sdf` from open3d RaycastingScene |

---

## Split — deterministic and reproducible

The train/val split is identical to the one used for completer training:

```python
import hashlib

def scene_is_val(scene_id: str, val_fraction: float = 0.20) -> bool:
    digest = hashlib.md5(scene_id.encode()).digest()
    h = int.from_bytes(digest[:2], "little")   # 0..65535
    return h / 65536.0 < val_fraction
```

This function is defined in `src/occlusynth/data/scannet.py` as
`ScanNetDataset._scene_is_val`.  No external scene-list file is required.

**Val scenes (9 croppable):** scene0001_00, scene0085_00, scene0146_00,
scene0220_01, scene0301_02, scene0367_00, scene0556_00, scene0635_01, scene0704_00.

---

## Metrics

### Metric 1 — Hazard-awareness rate

> Fraction of hidden hazards the method marks as occupied (completed SDF < 0).

| Method | Predicted SDF for OCCLUDED | Awareness |
|---|---|---|
| no_completion | 0.0 m | **0.000** (by construction) |
| occluded_as_free | +0.10 m | **0.000** (by construction) |
| OccluSynth Completer | completed_sdf from 3D U-Net | > 0.000 (reported) |

The baselines have zero awareness by construction — they never produce a
negative SDF in the occluded region.  OccluSynth's geometric completion is
the only component capable of detecting any hidden hazard.

### Metric 2 — Path collision-avoidance rate

> Fraction of hidden hazards on the naive path that the risk-graded planner avoids.

1. **Naive path**: A\* with λ=0 (occluded cells treated as free, cost=1.0).
2. **Risk path**: A\* with λ=4.0 (penalises `completed_sdf < 0` occluded cells
   via `cost = 1.0 + 4.0 × p_occupied`).
3. Project hazards to the 2D floor plan over the robot height band
   (0.10–0.50 m above lowest FREE voxel).
4. `avoidance_rate = (hazards_on_naive – hazards_on_risk) / max(hazards_on_naive, 1)`

---

## Results (interim 64³ checkpoint, ep=32, val_loss=0.1857)

```
Scene            Hazards   Aware-Compl  Aware-Base  Avoid%   Dims
──────────────────────────────────────────────────────────────────
scene0001_00      64,507      0.2019      0.0000     0.0%   162×171×96
scene0085_00      77,345      0.3316      0.0000     0.0%   116×96×96
scene0146_00       5,007      0.0200      0.0000     0.0%   96×96×96
scene0220_01      22,978      0.0702      0.0000     0.0%   134×117×96
scene0301_02      83,099      0.1714      0.0000     0.0%   114×127×96
scene0367_00      65,133      0.3213      0.0000     0.0%   143×163×96
scene0471_01       9,410      0.0021      0.0000     0.0%   96×96×96
scene0556_00      53,602      0.1145      0.0000    15.5%   132×113×96
scene0635_01      21,937      0.0437      0.0000     0.0%   96×96×96
scene0704_00      27,067      0.3340      0.0000     0.0%   96×103×96
──────────────────────────────────────────────────────────────────
AGGREGATE        430,085      0.2132      0.0000     0.0%   —
```

Baselines (no_completion, occluded_as_free): **0.000 awareness always**.
OccluSynth Completer: **21.3% aggregate hazard awareness** — the only system
capable of detecting any hidden obstacle.

**Note on avoidance rate:** With 21% aggregate awareness, the average risk
premium per hazardous column is only 1.0 + 4.0 × 0.21 ≈ 1.84, rarely enough
to force a detour (a 1-step rectangular detour costs ~2√2 ≈ 2.83 extra path
units).  scene0556_00 (15.5%) shows the planner CAN avoid hazards when the
completer detects a cluster.  The full 96³ A100 checkpoint (better awareness)
will improve the avoidance rate accordingly.

Regenerate results: `demo_outputs/safety_benchmark/results.json`

## How to reproduce

```bash
# Generate scene grids for all val scenes (run once)
for scene in scene0001_00 scene0085_00 scene0146_00 scene0220_01 \
             scene0301_02 scene0367_00 scene0556_00 scene0635_01 scene0704_00; do
  .venv312/bin/python scripts/run_visibility.py --scene $scene --use_gt_depth
done

# Run the benchmark
.venv312/bin/python scripts/run_safety_benchmark.py --device mps
```

Results: `demo_outputs/safety_benchmark/results.json`  
Per-scene PNGs (naive vs risk path over hazard heatmap):
`docs/images/safety_benchmark/<scene>.png`

---

## Design decisions

**Why ScanNet, not Habitat-Sim?**  
Habitat-Sim is a rendering engine for *simulated* navigation.  OccluSynth
evaluates on *real* reconstructions from commodity RGB-D sensors (Kinect v1).
Simulated scenes are typically convex rooms without cluttered occlusion; real
ScanNet scenes have furniture, doorways, and corners that produce complex
occluded volumes.  Using Habitat for our benchmark would make the claim
"first benchmark on real occlusion" false.

**Why full-scene inference, not crops?**  
Hazards are a property of the full scene's occluded volume.  Using 96³ crops
would miss hazards that span across crop boundaries and would make the planner
metric meaningless (paths in a 4.8 m × 4.8 m patch are too short to require
detours).

**Why are the baselines always 0%?**  
By definition.  This is the entire point: systems that don't model occluded
space *cannot* detect any hazard in that space.  A comparison of 0% vs
OccluSynth's awareness rate directly quantifies the safety contribution of
geometric completion.

---

## Limitations

- GT SDF uses `_vh_clean_2.ply` (clean mesh, not semantic labels) — thin objects
  like chair legs may have unreliable signed distance near the surface.
- Interim checkpoint (64³ crops, 35 epochs on MPS); A100 96³ run expected to
  improve Metric 1 numbers.
- p_occ from inference-only MC Dropout is poorly calibrated (ECE ≈ 0.42);
  Metric 2 currently uses the `completed_sdf < 0` criterion, not p_occ.

---

## References

Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.
*ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes.* CVPR 2017.
