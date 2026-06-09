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
on scene0000_00 (6 frames, GT depth from ScanNet):

| Metric | Value |
|--------|-------|
| Mean scale factor (pred × k = metres) | **7.39** |
| Std across frames | 0.48 |
| Mean median absolute relative error (after scale) | **0.029** (2.9 %) |

The adapter's primary supervised signal is learning this `k` correction per
scene, together with the occlusion mask.  ARE of 2.9 % is the **"before
adapter"** baseline; the adapter should reduce this further.

---

## TSDF Fusion Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Voxel size | 5 cm | Matches ScanNet annotation resolution |
| SDF truncation | 4 × voxel = 20 cm | Standard 4× rule |
| Depth max | 3.5 m | Reliable range of Kinect v1 sensor |
| Depth scale | 7.39 × (learned) | VGGT internal → metres |
| Pose source | ScanNet GT (`pose/*.txt`) | See §Camera Pose Strategy |

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

Run: `python scripts/scannet_baseline.py`  
Outputs: `demo_outputs/scannet_baseline/`

Results on scene0000_00, 6 frames, real checkpoint:

- Depth ARE: **0.029** (excellent structure, wrong scale)
- Depth scale: **7.39×** (stable across frames)
- Camera ATE: **0.704 m** (unusable for TSDF — use GT poses)
- Confidence: **1.0–1.73** (model is consistently high-confidence)
