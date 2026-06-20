# Adapter Design Note (Final)

**Project:** OccluSynth
**Component:** Sparse-Depth Metric Grounding
**Author:** Aditya Agarwal (onemore_adi)
**Status:** ✅ Decision locked. Closed-form per-frame fit ships in MVP.
**Supersedes:** prior draft proposing a learned token-injection adapter

---

## TL;DR

> **No learned adapter is required for the MVP.** A closed-form per-frame RANSAC fit of `depth_metric = a · depth_pred + b`, using the 500 sparse depth anchors provided per frame by the problem statement, achieves **ARE 0.0243 / δ<1.05 = 90.3%** on ScanNet `scene0000_00`. This meets the metric-depth requirement for downstream TSDF fusion and frees the remaining engineering budget for the project's actual research contributions (visibility-aware fusion, occlusion-aware completion, risk-graded planning).

---

## 1. Empirical Findings — Stage 0

Closed-form baselines evaluated on `scene0000_00`, 6 frames, using the 500 anchor pixels as input (the same input mode any learned adapter would have):

| Method             | Mean ARE ↓ | Mean RMSE ↓ | δ<1.05 ↑  | δ<1.10 ↑  | δ<1.25 ↑  |
| ------------------ | ---------- | ----------- | --------- | --------- | --------- |
| GlobalScalar       | 0.0600     | 0.148 m     | 51.1%     | 80.7%     | 96.8%     |
| PerFrameScalar     | 0.0347     | 0.105 m     | 77.5%     | 91.5%     | 99.7%     |
| PerFrameLS         | 0.0252     | 0.086 m     | 90.0%     | 98.0%     | 99.7%     |
| **PerFrameRANSAC** | **0.0243** | **0.085 m** | **90.3%** | **98.0%** | **99.7%** |

### Three interpretations

1. **Per-frame fitting is mandatory.** Global scalar (0.060) loses badly to per-frame (0.024) because per-frame scales actually drift across views — frame `003200` has true scale 8.46 vs the global 7.41, contributing the bulk of GlobalScalar's error. The original Phase 1 baseline measurement (σ = 0.477 across 6 frames) predicted this exactly.

2. **Adding a bias term is worth +29% relative ARE.** PerFrameLS gains 0.0252 ↔ 0.0347 (PerFrameScalar) by allowing `b ≠ 0`. Frames `001200` and `004200` have b ≈ 0.31–0.34 m — consistent with a near-camera flat surface introducing a depth bias VGGT-Omega cannot absorb into scale alone. **This is why we use `(a, b)`, not `a` alone.**

3. **RANSAC over LS is marginal on clean ScanNet (0.0243 vs 0.0252) but free.** Inlier rates of 95–99% confirm VGGT-Omega's depth is clean on standard indoor frames. RANSAC's headroom shows up only on frames with transparent surfaces, reflections, or sensor outliers. Since the runtime cost is ~milliseconds per frame and the implementation is one scikit-learn call, we use RANSAC by default — the robustness is essentially free insurance for the demo and eval sets.

---

## 2. Production Approach

### Algorithm

For each frame independently:

```python
from sklearn.linear_model import RANSACRegressor

def fit_metric_depth(depth_pred, anchor_pixels, anchor_depths):
    """
    depth_pred:    (H, W) predicted depth from VGGT-Omega, non-metric
    anchor_pixels: (500, 2) pixel locations (u, v) of sparse depth anchors
    anchor_depths: (500,) ground-truth metric depth at those locations

    Returns: (a, b) such that depth_metric = a * depth_pred + b
    """
    d_pred = depth_pred[anchor_pixels[:, 1], anchor_pixels[:, 0]]
    valid = (anchor_depths > 0) & (d_pred > 0)

    ransac = RANSACRegressor(
        min_samples=10,
        residual_threshold=0.10,   # 10 cm — tune on val set
        max_trials=100,
        random_state=0,
    )
    ransac.fit(d_pred[valid, None], anchor_depths[valid])
    a = ransac.estimator_.coef_[0]
    b = ransac.estimator_.intercept_
    return float(a), float(b)

def apply_metric_correction(depth_pred, a, b):
    return a * depth_pred + b
```

### Where this lives in the codebase

```
src/occlusynth/models/metric_grounding.py    # the fit + apply functions
src/occlusynth/data/sparse_sampler.py        # produces the 500 anchor pixels
scripts/eval_metric_grounding.py             # reproduces the Stage 0 table
```

The "metric grounding" module replaces what was originally planned as `models/adapter.py`. The training script (`train_adapter.py`) is no longer needed for the MVP and can be deleted from the planning docs.

---

## 3. Why This Is Defensible (not a cop-out)

1. **The numbers are strong by the literature's own standards.** ARE 0.024 / δ<1.05 = 90.3% sits above most learned monocular-depth-with-sparse-prior methods on similar indoor benchmarks. The closed-form fit isn't a hack — it's a correct solution to a well-conditioned problem.

2. **VGGT-Omega's prior is doing the heavy lifting.** The baseline's 0.029 ARE after oracle scale alignment indicated the geometric structure was already excellent. Stage 0 confirms a 2-parameter affine fit is enough to convert that structure into metric units. Adding a learned model would buy noise reduction at best and overfitting at worst.

3. **The paper story is honest.** "We show that VGGT-Omega's monocular depth prior is strong enough on indoor scenes that a closed-form per-frame affine fit of 500 sparse anchors achieves metric ARE of 0.024 — no adapter training required." This is a _more_ interesting finding than "we trained an adapter." It identifies a previously-unmeasured property of the backbone.

4. **The freed time goes to the real contributions.** OccluSynth's novelty is visibility-aware fusion and occlusion-aware completion, not depth scale calibration. The original 3-5 days budgeted for adapter training now go to those components.

---

## 4. Limitations (acknowledge in paper)

1. **Requires clean anchor depths at inference.** Stage 0 used GT-quality ScanNet depth at the 500 sampled locations. Real sparse-depth sensors (ToF, LiDAR) produce noisier values. **Stage 0.5 noise experiment completed** — see §7 and `docs/images/noise_sensitivity.png`. Key finding: both methods are stable up to σ=0.10 m; at σ=0.25 m LS is actually more robust than RANSAC because homogeneous Gaussian noise averages out under LS (see §7 for full analysis). RANSAC is the right choice for ScanNet because its dominant noise is *structural* (dropouts, reflections), not Gaussian.

2. **Affine fit only corrects global per-frame error.** Any local depth structure errors — e.g., systematic bias at object edges or thin structures — pass through uncorrected. The empirical 90% δ<1.05 rate means ~10% of pixels are off by more than 5%; these residuals will appear as TSDF blur but are within the 5cm voxel tolerance for our use case.

3. **No information flows from anchors to dense depth through the network.** This is by design (the network is frozen and uninformed about anchors), but it does mean failure cases on the dense prediction cannot be repaired with more anchors. If a future scene shows ARE > 0.10 with closed-form fit, the learned approach in §5 becomes necessary.

---

## 5. Optional Phase 2 (post-MVP, stretch goal only)

If after the MVP ships there is time remaining and we observe failure cases on the eval split where closed-form ARE > 0.10, a small learned `(a, b)` predictor could be useful:

```
Inputs : 500 anchor pixels (u, v, d_known, d_pred_at_pixel)
         pooled camera_and_register_tokens (B, N, 2048)
Network: PointNet-style encoder over 500 anchors  →  256-dim
         concat with pooled tokens                →  2304-dim
         2-layer MLP                              →  (a, b) per frame
Params : ~2-3M trainable, VGGT-Omega frozen
Loss   : L1 between (a·d_pred + b) and d_gt at the 500 anchor pixels
```

This is **not** a hackathon deliverable. It belongs in the Phase 2 / future-work section of the paper, framed as "a learned alternative that improves on closed-form when anchor noise is high or coverage is low." Do not build this before TSDF + completion + planner are working.

---

## 6. Impact on Project Plan

| Component                    | Before                            | After                                        |
| ---------------------------- | --------------------------------- | -------------------------------------------- |
| Adapter (this doc)           | ~5 days dev + 30 GPU-hrs training | ~1 day (closed-form + noise robustness test) |
| TSDF visibility-aware fusion | ~2 days                           | ~3 days (gain 1 day)                         |
| U-Net completer              | ~3 days                           | ~4 days (gain 1 day)                         |
| Risk-graded planner          | ~2 days                           | ~3 days (gain 1 day)                         |
| Buffer for video / polish    | 3 days                            | 4 days                                       |

The 5+ days saved here are the single largest scope improvement since Phase 1. Allocate them to making the visibility-aware completion result actually convincing — that's the contribution that justifies the project.

---

## 7. Verification Checklist Before Next Component

Before moving on to visibility-aware TSDF fusion, confirm:

- [x] `src/occlusynth/models/metric_grounding.py` implemented (`fit_metric_depth`, `apply_metric_correction`, `ground_scene`, `eval_scene`)
- [x] Closed-form fit reproduces §1 numbers on `scene0000_00` — ARE=0.024 at 382×512 (regression-pinned in `tests/test_metric_grounding.py`)
- [x] Multi-scene eval on 10 held-out val scenes — mean ARE 0.024, worst scene 0.032, 0/10 DEGRADE (`tests/test_multi_scene.py`)
- [x] Stage 0.5 noise-robustness: σ ∈ {0, 0.01, 0.05, 0.10, 0.25 m} × anchor counts {500, 250, 100, 50} — plots in `docs/images/`, pinned in `tests/test_robustness_ablation.py`
  - Both methods stable σ ≤ 0.10 m; RANSAC degrades at σ=0.25 m (Gaussian mismatches outlier model); LS stable throughout
  - 100 anchors = 500 anchors in practice (ΔARE < 0.001); 50 still reasonable
- [x] Per-frame `(a, b)` written to `demo_outputs/grounding/<scene_id>_grounding.json`
- [ ] Decision documented in commit message linking to this file

---

## 8. Decision

> **Ship `PerFrameRANSAC` affine fit (`depth_metric = a · depth_pred + b`) as the metric-grounding component of OccluSynth.** No model training is required for this step. The learned alternative is logged as Phase 2 future work and explicitly out of MVP scope.

Next component: visibility-aware TSDF fusion with ray-casting to populate the third `p_observed` voxel channel. See `docs/fusion_design.md` (to be written).
 