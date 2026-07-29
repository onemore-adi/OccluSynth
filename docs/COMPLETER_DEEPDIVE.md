# The OccluSynth Completer — Deep Dive

**The key innovation, dissected.** Use this when someone wants to go deeper than the
stage talk allows. Structured in **layers** — stop at whichever depth the question
was actually asking for.

> Layer 1 — the idea in 60 seconds
> Layer 2 — why this problem is hard
> Layer 3 — input/output contract
> Layer 4 — architecture
> Layer 5 — the loss function
> Layer 6 — training
> Layer 7 — results
> Layer 8 — what failed (four interventions, one frontier)
> Layer 9 — inference-time engineering
> Layer 10 — hardest questions

---

## Layer 1 — The idea in 60 seconds

The completer is a **14.7-million-parameter 3D U-Net** that takes a partially
observed voxel grid and predicts the geometry inside the **occluded** region —
space that is hidden behind a measured surface in *every* camera view.

It is the only component in the pipeline that outputs geometry no sensor measured.
Everything upstream measures; the completer *infers*.

The one-line contract: **in → partial 3D scan with labelled blind spots; out → a
complete signed distance field, including inside those blind spots.**

---

## Layer 2 — Why this problem is genuinely hard

**It is not inpainting.** The most common suggestion is "use a 2D inpainting model."
That cannot work here, and the reason is structural: the occluded region **appears in
no image at all**. Not at a bad angle, not partially — there is no pixel anywhere in
the dataset corresponding to the volume behind the sofa. 2D inpainting fills gaps
*inside an image plane*. Our gap is in space, not in the picture.

**It is not interpolation.** Interpolation fills between known values. Here we
extrapolate into a region where the model has *no measurements on the far side* —
only a boundary condition on the near side plus learned priors about how rooms behave.

**The supervision is inherently partial.** Ground-truth meshes tell us what is really
behind the sofa, but only where the ScanNet scan itself was complete.

**And it is fundamentally ambiguous.** Behind a given surface there might be a wall,
a box, a chair, or nothing. Multiple answers are consistent with the evidence. This
single fact explains most of the model's observed behaviour (Layer 10).

---

## Layer 3 — The input/output contract

### Input: 3 channels at every voxel

| Channel | Meaning |
|---|---|
| `sdf` | Truncated signed distance, normalised to [−1, 1] |
| `weight` | Fusion confidence — how many rays contributed |
| `p_observed` | Probability this voxel was actually observed |

### The four-state label (drives supervision, not fed as input in v1)

| State | Meaning | Role in training |
|---|---|---|
| `UNOBSERVABLE` (0) | Outside every camera frustum | **Excluded entirely** |
| `FREE` (1) | Ray passed through | Excluded (v1) / hinge term (v2) |
| `SURFACE` (2) | Ray terminated here | **Supervised** |
| `OCCLUDED` (3) | Behind a surface in all views | **Supervised — the target** |

### Output

A single channel: completed SDF in **metres**, unbounded, at full input resolution.

### Typical composition of a validation crop

| State | Share |
|---|---|
| Unobservable | 79.5% |
| Occluded | 11.1% |
| Free | 8.2% |
| Surface | 1.3% |

> **This table is worth memorising.** Only **11.1%** of the volume is the actual
> prediction target, and only **1.3%** is measured surface. The model reasons from a
> very thin band of evidence — and we deliberately refuse to score ourselves on the
> 79.5% we cannot know.

---

## Layer 4 — Architecture

```
Input (3, D, H, W)
   │
   ├─ stem      : 2 × [Conv3d 3³ → GroupNorm(8) → GELU]      →  32 ch, full res
   ├─ encoder 1 : stride-2 down + refine                      →  64 ch, ½
   ├─ encoder 2 : stride-2 down + refine                      → 128 ch, ¼
   ├─ encoder 3 : stride-2 down + refine                      → 256 ch, ⅛
   └─ encoder 4 : stride-2 down + refine                      → 256 ch, ¹⁄₁₆   ← bottleneck
                          │
   ┌──────────────────────┘
   ├─ decoder 4 : trilinear ×2 → concat skip(enc3) → 2 conv   → 256 ch
   ├─ decoder 3 : trilinear ×2 → concat skip(enc2) → 2 conv   → 128 ch
   ├─ decoder 2 : trilinear ×2 → concat skip(enc1) → 2 conv   →  64 ch
   ├─ decoder 1 : trilinear ×2 → concat skip(stem) → 2 conv   →  32 ch
   └─ head      : Conv3d 1×1×1, no activation                 →   1 ch (SDF, metres)
```

**Design decisions and their drivers:**

- **U-Net shape.** Output must be the same shape as input, and needs both global
  context (*"this is a room corner"*) and local precision (*"the surface is exactly
  here"*). The encoder supplies the former, skip connections preserve the latter.
- **4 downsampling stages.** At 5 cm voxels and a 64³ crop, the bottleneck sees a
  4³ region — each bottleneck voxel summarises ~0.8 m of space. Enough context to
  know "floor continues", small enough to train on a laptop.
- **GroupNorm, not BatchNorm.** Batch size is 4. BatchNorm is unreliable at small
  batch sizes; GroupNorm is batch-independent — and critically, behaves identically
  at training and inference, which matters for tiled whole-scene inference.
- **Trilinear upsampling, not transposed convolution.** Transposed convs produce
  checkerboard artefacts; on a geometry task those become visible surface ripples.
- **No output activation.** SDF is unbounded and signed; squashing it would cap how
  far the model can say "far away".
- **1×1×1 head.** The final layer only recombines channels — all spatial reasoning
  happened earlier.

**Parameter count: 14,688,577.** Deliberately small. Trains on a 16 GB MacBook, and a
robot can run it. Scaling up was never the interesting question here.

### Optional v2 variants (both default OFF, checkpoint-compatible)

- `in_channels=7` — appends the one-hot state. **Motivation:** without it,
  `OCCLUDED` and `UNOBSERVABLE` voxels have *identical* features (sdf=+1, w=0,
  p_obs=0), so the network cannot tell which region it is being asked to fill.
- `occ_head=True` — a second output channel giving a calibrated occupancy logit.

Both were tested. Neither moved the frontier (Layer 8).

---

## Layer 5 — The loss function

### v1 (shipping): masked L1

```
mask = (state == SURFACE) | (state == OCCLUDED)
loss = |pred − target|[mask].mean()
```

Deceptively simple, and every part is deliberate:

- **Masked** so `UNOBSERVABLE` is excluded — the network is never told, and never
  graded on, what it cannot in principle know. Supervising it would inflate every
  metric while teaching nothing transferable.
- **`SURFACE` included** so the model stays anchored to measured reality rather than
  drifting free in the hidden region.
- **L1 not L2** because L2 punishes large errors quadratically, which in an ambiguous
  problem pushes even harder toward blurry averages.

### v2 (built, tested, not shipped): three-term loss

| Term | Form | Intent |
|---|---|---|
| `sdf` | Truncated (±0.30 m) L1, up-weighted within 10 cm of surface | Focus capacity where marching cubes actually looks |
| `occ` | BCE on occupancy logit vs (GT < 0), batch pos-weighted | Sign flips change occupancy — what safety metrics score |
| `free` | Hinge `relu(margin − pred)` on FREE voxels | Directly penalise hallucinating solid into *observed empty* space |

The `free` term is legitimate supervision, not leakage: those voxels were *measured*
empty.

**Result:** it worked exactly as designed and still did not help. `w_free` cut
free-space violations from **32.2% → 0.95%** — a 34× improvement — but bought it by
predicting less solid overall, costing recall. A different operating point on the
same frontier.

---

## Layer 6 — Training

| Setting | Value | Why |
|---|---|---|
| Optimiser | AdamW, weight decay 1e-4 | Standard, robust |
| Learning rate | 1e-3, cosine annealed | — |
| Batch size | 4 | 16 GB unified memory limit |
| Crop size | 64³ (sub-cropped from 96³) | Memory |
| Epochs | ~35 | Best at epoch 32 |
| Hardware | Apple MacBook, 16 GB, MPS | **No cloud GPU used** |
| Best val loss | 0.1857 | — |

### Data

- **418 training / 90 validation crops**, from **35 train / 9 val scenes**
- **Zero scene overlap** — verified explicitly, not assumed
- ScanNet v2, 5 cm voxels

### Augmentation — and the one that is deliberately absent

Eight variants: 4 yaw rotations (90° about the vertical) × optional horizontal flip —
the dihedral group D4 in the horizontal plane.

**The vertical axis is never flipped.** Flipping z would put ceilings below floors and
destroy the single most valuable prior the model has: **gravity**. Floors continue
under tables *downward*. Rooms are not symmetric top-to-bottom. SDF values are
invariant under the permitted isometries, so arrays are only permuted, never rescaled.

> A good detail to volunteer — it shows the augmentation was reasoned about, not
> copy-pasted.

---

## Layer 7 — Results

### vs naive baselines (90 val crops, occluded region)

| Method | MAE ↓ | Sign acc ↑ | Within 5 cm ↑ |
|---|---|---|---|
| `no_completion` (zero-thickness walls) | 45.27 cm | 0.299 | 6.1% |
| `occluded_as_free` (assume empty) | 42.00 cm | 0.701 | 12.1% |
| **Completer** | **27.14 cm** | **0.722** | **34.9%** |

Nearly **3× more voxels within 5 cm** than the better baseline.

### Scene-level geometry (10 held-out scenes)

| Region | Metric | TSDF-only | OccluSynth |
|---|---|---|---|
| Surface | Chamfer L1 ↓ | 3.05 cm | **2.20 cm** |
| Surface | F-score @5 cm ↑ | 79.6% | **84.7%** |
| Surface | Completion ratio ↑ | 73.8% | **96.7%** |
| **Occluded** | **F-score @5 cm ↑** | **0.0%** | **37.2%** |
| **Occluded** | **Completion ratio ↑** | **0.0%** | **57.6%** |

> The 0% is **structural, not a failure of the baseline**: TSDF-only has zero
> predicted points behind surfaces by construction. Always say this out loud — it is
> the difference between an honest claim and an unfair one.

Note the completer also *improves the surface region* (2.20 vs 3.05 cm) — it fills
small measurement holes too, not just large occlusions.

### The precision/recall frontier

| Iso | Precision | Recall | F1 |
|---|---|---|---|
| 0.00 | 0.581 | 0.255 | 0.354 |
| **+0.04 (shipped)** | **0.535** | **0.351** | **0.424** |
| +0.10 | 0.477 | 0.472 | 0.475 |

**Recall-first by design:** a missed obstacle is a collision; a phantom obstacle is a
slowdown. Asymmetric costs, asymmetric operating point.

---

## Layer 8 — What failed: four interventions, one frontier

The most scientifically interesting part of the project. **Present it as a finding,
not an apology.**

| # | Intervention | Result |
|---|---|---|
| 1 | Loss reweighting grid (occ BCE, free hinge, truncation, near-surface) | Slid along frontier |
| 2 | Architecture v2 (state channels + occupancy head) | No gain |
| 3 | 3.6× data (1528 multi-density crops) | No gain |
| 4 | Stability-fixed retrain at 1e-4 | Better val loss, **worse frontier** |

**Intervention 4 in full**, because it is the most instructive story:

The multi-density run had *diverged* — validation loss climbed 0.184 → 0.256. Diagnosis:
learning rate too high for a warm restart. Fix: restart at 1e-4. It worked — training
stabilised, and validation loss reached **0.1794**, with `trunc_l1` and `sign_acc`
improving *together* (corroboration, not a single-metric fluke).

Then it was scored on the frontier:

| Iso | Shipping | Retrained |
|---|---|---|
| 0.00 | P 0.581 / R 0.255 | P 0.582 / R **0.225** |
| +0.04 | P 0.535 / R 0.351 | P 0.535 / R **0.330** |
| +0.10 | P 0.477 / R 0.472 | P 0.482 / R **0.457** |

Lower recall at *every* matched precision. One confound remained — the validation
crops are all sparse (n6, 79.5% unobservable) while we render dense — so it was
re-tested directly against ScanNet ground truth on dense n40 meshes:

| Checkpoint | Accuracy ↓ | Hidden surface recovered ↑ |
|---|---|---|
| **Shipping** | **11.2 cm** | **36.6%** |
| Retrained | 13.8 cm | 33.4% |

Worse in both regimes. Shipping checkpoint retained.

**The lesson, now a standing rule in the repo:** *validation loss is not a proxy for
the frontier. Always re-score before promoting a checkpoint.*

**Also tested and rejected:** MC-dropout uncertainty gating — the joint rule
`(sdf < iso) AND (std < τ)` never beat the threshold alone; best F1 occurred with the
gate disabled, meaning per-voxel dropout variance carries essentially no information
beyond the prediction itself. And 8× test-time augmentation: negligible gain, 8× cost.

**What four failures at one frontier actually mean:** the ceiling is not in tuning. It
is in **64³ resolution** and **inherent task ambiguity**. That is why the 96³ run is
the credible next step — it is the one lever never pulled.

---

## Layer 9 — Inference-time engineering

Three things that improved output with **no retraining at all**:

**1. Tiled whole-scene inference.** Rooms exceed any crop size, so inference runs in
overlapping 96³ tiles with 16-voxel overlap, averaged in the overlap regions.

**2. Iso-level calibration.** Because the model is a mean-regressor on an ambiguous
problem, its SDF is biased toward "empty". Shifting the surface threshold to +0.04 m
lifted recall **0.255 → 0.351 (+38% relative)**, free.

*Why this is safe:* the exporter writes completed values **only into OCCLUDED
voxels** — observed-free space keeps its measured TSDF by construction. So a higher
iso grows geometry only inside the blind spot, never into space a camera confirmed
empty.

**3. The anchor filter — the blob killer.** Rooted in how the model actually works: it
extrapolates from observed anchors. Therefore a predicted component *touching*
measured surface is an extension of evidence; one floating in the void is a
hallucination.

| Scene | Components before → after | Solid volume removed |
|---|---|---|
| scene0000_00 | 1946 → 31 | 6.8% |
| scene0704_00 (held out) | 692 → 11 | 4.6% |

**98% of components removed, ~7% of volume.** Almost everything it deletes is
speckle; almost nothing real is lost.

---

## Layer 10 — The hardest questions

**"Why does it produce smooth blobs instead of sharp furniture?"**
> The deepest question anyone asks, and the answer is mathematical. Behind a surface
> many completions are plausible — wall, box, chair, nothing. Trained with a
> regression loss, the output that minimises expected error is the **average of all
> plausible answers**, and the average of many different shapes is a smooth blob. It
> is not a bug in the architecture; it is what regression *provably* does under
> ambiguity. A generative model would give one sharp plausible answer instead — see
> below.

**"So why not use diffusion / Stable Diffusion?"**
> Four reasons. *Modality:* SD is a 2D image model; the hidden region is in no image.
> *Determinism:* a sampler gives a different room each run — for a safety planner,
> "plausible" is the failure mode. *Scale:* 3D diffusion needs orders of magnitude
> more than 418 crops. *Metrics:* our KPIs are expected-error metrics, for which the
> posterior mean — what regression learns — is provably optimal; a diffusion sample
> scores worse while looking better.
>
> The honest framing: *regression gives the average of all plausible rooms — accurate
> but soft. Diffusion gives one sharp plausible room — convincing but unverifiable.
> For safety we chose accurate.*

**"40% of your predictions are wrong. Is this usable?"**
> Yes, because of how it is consumed. The output is never presented as measured —
> it is visually distinct, kept in a separate state class, and the planner treats it
> as **risk**, not fact. The comparison is not against a perfect system; it is against
> systems that assume hidden space is *empty*, which is a confident wrong answer with
> no uncertainty attached at all.

**"Isn't 27 cm MAE in the occluded region terrible?"**
> Compare like with like. The naive baselines score 45 cm and 42 cm on the same
> voxels, and conventional reconstruction has no prediction there to score. Errors on
> invented geometry are not comparable to errors on measured geometry.

**"How do you know it isn't memorising?"**
> Three defences: validation scenes never appear in training (verified, not assumed);
> unobservable space is excluded from the loss so it cannot be rewarded for
> reproducing unseen ground truth; and the anchor filter's generalisation to a
> held-out scene (692 → 11 components) shows the behaviour transfers.

**"What would you do with more time?"**
> The 96³ run first — four 64³ interventions hitting one frontier is strong evidence
> that resolution, not tuning, is the bottleneck. Then semantic input features
> (feeding image/semantic cues, as SCFusion does) to strengthen priors. A generative
> variant is legitimate future work for *render quality*, but would need a separate
> safety argument.

---

## One-paragraph summary

> The OccluSynth Completer is a 14.7M-parameter 3D U-Net that predicts signed distance
> fields inside occluded volumes — geometry absent from every depth image and therefore
> unreachable by any 2D method. Trained on 418 crops from 35 ScanNet scenes on a 16 GB
> laptop, it recovers **57.6% of hidden surface within 5 cm** where conventional
> reconstruction recovers **0%**, while also improving measured-surface Chamfer error
> from 3.05 to 2.20 cm. Four independent attempts to improve it — loss reweighting,
> architectural changes, 3.6× data, and a stability-fixed retrain — all landed on the
> same precision/recall frontier, which is itself the finding: the ceiling is voxel
> resolution and inherent ambiguity, not tuning.
