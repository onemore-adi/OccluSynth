# OccluSynth — Numbers Cheat Sheet

**Print this. Glance at it during Q&A.**
Every figure below is reproducible from `scripts/`. Metric definitions in
[`PROJECT_REPORT.md` §5.5](PROJECT_REPORT.md).

> ⚠️ **One-line caveat to state up front if asked:** all numbers are from the
> **interim 64³ checkpoint** (`checkpoints/interim_64_aug/completer_best.pt`,
> epoch 32) trained on a 16 GB MacBook. The 96³ GPU run is scripted and
> data-prepared but **not executed** (no compute access).

---

## ⭐ The five numbers to memorise

| # | Number | What it is |
|---|---|---|
| 1 | **57.6%** | Hidden-surface recall @5 cm in the occluded region — **vs 0% for any observation-only method** |
| 2 | **37.2%** | Occluded F-score @5 cm — the headline "we score where nothing else can" |
| 3 | **2.20 cm** | Chamfer L1, improved from 3.05 cm (**–27.8%**) |
| 4 | **21.3%** | Hidden hazards anticipated across 430,085 hazards — baselines structurally 0% |
| 5 | **14.7 M** | Parameters — trains on a laptop, runs fast enough for a robot |

**The one-sentence claim:** *Every observation-only method scores exactly 0% behind
surfaces — not because they are bad, but because no sensor measures through a sofa.
We score 37.2%.*

---

## 1. Geometry evaluation — 10 held-out scenes

Source: `demo_outputs/geometry_eval/results.json` · `scripts/eval_geometry.py`

### Surface region (where cameras did see)

| Method | Chamfer L1 ↓ | F-score @5 cm ↑ | Completion ratio ↑ |
|---|---|---|---|
| TSDF-only | 3.05 cm | 79.6% | 73.8% |
| **OccluSynth** | **2.20 cm** | **84.7%** | **96.7%** |
| *Improvement* | *–27.8%* | *+5.1 pp* | *+22.9 pp* |

### Occluded region (the blind spot — the actual contribution)

| Method | Chamfer L1 ↓ | F-score @5 cm ↑ | Completion ratio ↑ |
|---|---|---|---|
| TSDF-only | n/a — no points exist | **0.0%** | **0.0%** |
| **OccluSynth** | 9.44 cm | **37.2%** | **57.6%** |

> TSDF-only scores 0% **by construction**, not by underperformance: it has zero
> predicted points behind surfaces. This is the structural point of the project.

**Operating point:** 57.6% recall at 27.4% precision — deliberately recall-first.
*A missed obstacle is a collision; a phantom one is a slowdown.*

---

## 2. Completer vs baselines — 90 validation crops

Source: `scripts/eval_completer.py` · occluded region

| Method | MAE ↓ | Sign accuracy ↑ | Within 5 cm ↑ |
|---|---|---|---|
| `no_completion` (hidden = zero-thickness) | 45.27 cm | 0.299 | 6.1% |
| `occluded_as_free` (hidden = empty) | 42.00 cm | 0.701 | 12.1% |
| **OccluSynth Completer** | **27.14 cm** | **0.722** | **34.9%** |

Surface region (sanity check): MAE 4.86 cm · within 5 cm 76.8%

**Beats both naive baselines on every occluded metric** — MAE cut ~40% vs
`no_completion`, and nearly **3× more voxels within 5 cm**.

---

## 3. Precision / recall frontier — voxel-level occupancy

Source: `scripts/probe_iso_sweep.py`, 90 val crops, shipping checkpoint.
Sweeping the iso threshold moves *along* this curve.

| Iso level | Precision | Recall | F1 |
|---|---|---|---|
| 0.00 (default) | 0.581 | 0.255 | 0.354 |
| +0.02 | 0.556 | 0.309 | 0.397 |
| **+0.04 (shipped)** | **0.535** | **0.351** | **0.424** |
| +0.06 | 0.514 | 0.391 | 0.444 |
| +0.10 | 0.477 | 0.472 | 0.475 |

**Calibration win:** moving 0.00 → +0.04 lifted recall **0.255 → 0.351 (+38%
relative)** with **zero retraining**.

> These are *voxel occupancy* P/R — distinct from the *surface* F-score in §1.
> Do not conflate the two in conversation.

---

## 4. Safety benchmark

Source: `scripts/run_safety_benchmark.py` · 10 scenes

| Metric | OccluSynth | Baseline |
|---|---|---|
| Hidden-hazard awareness | **21.3%** | **0.0%** |
| Total hazards evaluated | 430,085 | — |
| Collision avoidance (scene0556_00) | 15.5% | — |

---

## 5. Model and data

| Property | Value |
|---|---|
| Architecture | 3D U-Net, encoder–decoder with skip connections |
| Parameters | **14,688,577 (14.7 M)** |
| Encoder channels | 32 → 64 → 128 → 256 |
| Normalisation / activation | GroupNorm(8) / GELU |
| Input | 3 channels (SDF, weight, p_observed) |
| Output | 1 channel (completed SDF, metres) |
| Loss | Masked L1 over SURFACE ∪ OCCLUDED |
| Voxel size | 5 cm |
| Training crops | 418 train / 90 val (96³) |
| Multi-density variant | 1528 train crops (6/10/20 views) |
| Scenes | 35 train / 9 val — **zero overlap, verified** |
| Trained on | Apple MacBook, 16 GB, MPS — no cloud GPU |
| Best checkpoint | epoch 32, val loss 0.1857 |
| Published | `onemore-adi/occlusynth-completer` (HF, MIT) |

---

## 6. Input density — the biggest visual lever

Same scene, same model, same settings — **only frame count differs**:

| Frames | Unobservable volume | Result | Completion share of mesh |
|---|---|---|---|
| 6 | 85.1% | Fragmented shards | 38% |
| 40 | 58.4% | Reads as a real room | 24% |

> Sparse capture makes the completer's contribution look *proportionally larger*
> while making the mesh uglier. Dense capture is the better demo; both are honest
> if labelled.

---

## 7. Failed interventions — four, all on the same frontier

**This is a finding, not a gap.** State it confidently.

| # | Intervention | Outcome |
|---|---|---|
| 1 | Loss reweighting (occupancy BCE, free-space hinge, truncation, near-surface) | Slid along frontier. `w_free` cut free-space violations 32% → **0.95%**, paid in recall |
| 2 | Architecture v2 (state channels + occupancy head) | No frontier gain |
| 3 | 3.6× data (1528 multi-density crops) | No frontier gain |
| 4 | Stability-fixed retrain at 1e-4 | **Better val loss (0.1794), still worse on frontier** |

**Intervention 4 in detail** — the most instructive:

| Iso | Shipping checkpoint | Retrained |
|---|---|---|
| 0.00 | P 0.581 / R 0.255 | P 0.582 / R 0.225 |
| +0.04 | P 0.535 / R 0.351 | P 0.535 / R 0.330 |
| +0.10 | P 0.477 / R 0.472 | P 0.482 / R 0.457 |

Re-checked in the dense regime against ScanNet GT (n40 meshes):

| Checkpoint | Completed-geometry accuracy ↓ | Hidden surface recovered ↑ |
|---|---|---|
| **Shipping** | **11.2 cm** | **36.6%** |
| Retrained | 13.8 cm | 33.4% |

**Standing rule:** *validation loss is not a proxy for the frontier — always
re-score with `probe_iso_sweep.py` before promoting a checkpoint.*

**Also tested and rejected:** MC-dropout uncertainty gating (no information beyond
the prediction); 8× test-time augmentation (negligible gain, 8× cost).

---

## 8. Render pipeline improvements (no retraining)

| Change | Effect |
|---|---|
| Fusion density 6 → 40 frames | Unobservable 85% → 58% |
| Iso 0.00 → +0.04 | Recall 0.255 → 0.351 |
| Anchor filter | **1946 → 31 components**, only **6.8%** of solid volume removed |
| Anchor on held-out scene | 692 → 11 components, 4.6% removed |

**Anchor filter principle:** completed geometry touching measured surface is an
extension of evidence; geometry floating in the void is a hallucination.

---

## 9. Published baselines — ⚠️ different protocol

| Method | Chamfer L1 | F-score @5 cm | Region |
|---|---|---|---|
| Atlas (ECCV 2020) † | 6.5 cm | 39.6% | full scene |
| NeuralRecon (CVPR 2021) † | 5.2 cm | 43.4% | full scene |
| OccluSynth | 2.20 cm | 84.7% | surface region |

> † **Not directly comparable** — different evaluation protocol (full-scene, dense
> video input). Say this out loud before quoting. The defensible claim is the
> occluded-region result, where the comparison is structural: **0% vs 37.2%**.

---

## 10. Engineering

| Item | Value |
|---|---|
| Unit tests | 55 |
| Planner tests | 18 |
| External model | VGGT-Omega (Meta, 1B, frozen, open-weight) |
| Datasets | ScanNet v2 (primary), 7-Scenes (cross-dataset probe) |
| Licence | MIT (our code + published model) |

---

## Quick answers to likely questions

**"Why is occluded Chamfer 9.44 cm when surface is 2.20 cm?"**
Because we are *predicting geometry no camera measured*. 9.44 cm error on invented
geometry compares against infinite error for methods that predict nothing at all.

**"Why is precision only 27%?"**
Deliberate. Recall-first for a safety system — a missed obstacle is a collision, a
phantom one is a slowdown. The full curve is available; that operating point is a
choice, not a limit.

**"Did more data help?"**
No — and we tested it twice, including once where it improved validation loss and
*still* lost on the frontier. Four interventions, one frontier. That points at
resolution, not tuning.

**"Is the 96³ run done?"**
No. Scripted, data-prepared, never executed — no compute access. It is the most
credible remaining lever precisely *because* four 64³ interventions all plateaued.
