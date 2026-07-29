# OccluSynth — Project Report

**Occlusion-Aware 3D Scene Reconstruction in Partially Observable Real-World Environments**
Problem Statement 09 · Team `onemore_adi` · Aditya Agarwal · NIT Rourkela

> Companion documents:
> [`NUMBERS_CHEATSHEET.md`](NUMBERS_CHEATSHEET.md) — every number, scannable ·
> [`COMPLETER_DEEPDIVE.md`](COMPLETER_DEEPDIVE.md) — the core innovation in depth ·
> [`render_quality.md`](render_quality.md) — render experiments ·
> [`architecture.md`](architecture.md) — system architecture

---

## Part 1 — The story

### 1.1 The observation that started it

Every 3D reconstruction system in wide use today shares one silent assumption:
**if I did not see it, it does not exist.**

Point a camera at a living room. You get back a beautiful 3D model of exactly
the surfaces light reached. The floor *under* the sofa? Missing. The space
*behind* the cabinet? Missing. The far side of every object? Missing. Not marked
"unknown" — simply absent, indistinguishable from empty air.

For a photogrammetry hobbyist that is a cosmetic annoyance. For a robot it is a
safety defect. A robot planning a path through a room reads "absent" as "free to
drive through." The sofa's shadow becomes an inviting shortcut. The reconstruction
is not merely incomplete; it is **confidently wrong in exactly the places that
matter**, and it gives the planner no signal that anything is missing.

That gap — between *unobserved* and *empty* — is the entire project.

### 1.2 The reframing: a third state

Conventional volumetric reconstruction is binary. Each little cube of space
(a **voxel**) is either occupied or free. OccluSynth's first contribution is to
refuse that binary and label every voxel as one of **four** states:

| State | Meaning | The robot's view |
|---|---|---|
| `free` | A camera ray passed *through* this space | "Confirmed empty — safe" |
| `surface` | A camera ray *stopped* here | "Confirmed solid — avoid" |
| `occluded` | Behind a measured surface in *every* view | **"My blind spot"** |
| `unobservable` | Never inside any camera's field of view | "Outside my knowledge" |

The crucial distinction is between `occluded` and `unobservable`. Occluded space
is *geometrically implied*: we know something blocked our view, we know roughly
where the blocker is, and physics tells us the hidden region is bounded and
structured. Floors continue under tables. Walls continue behind cabinets. Sofas
have backs. Unobservable space carries no such implication — it is simply outside
what the cameras ever surveyed.

That separation is what makes the problem tractable. **We only try to predict the
occluded region, and we never grade ourselves on the unobservable one.** The
network is never told, and never scored on, what it cannot in principle know.

### 1.3 Why 2D inpainting cannot solve this

The obvious first idea — and the one most people suggest — is to use a 2D image
inpainting model to "fill in the holes," then reconstruct from the filled images.

This cannot work, and the reason is worth stating precisely: **the occluded region
does not appear in any image.** Not partially. Not at a bad angle. There is no
pixel, in any frame, that corresponds to the volume behind the sofa. A 2D inpainter
fills gaps *within an image plane*; it has nothing to operate on here. The hole is
not in the picture — the hole is in space.

The completion therefore has to happen in 3D, in a representation that knows about
volume and visibility. That single constraint forced most of the architecture.

---

## Part 2 — What was built

The system is a five-stage pipeline. Each stage exists because the previous one
leaves a specific problem unsolved.

```
RGB frames
    │
    ├─▶ [1] VGGT-Omega ──────────▶ relative depth + camera poses
    │                                (scale-ambiguous)
    ├─▶ [2] RANSAC metric grounding ▶ depth in real metres
    │
    ├─▶ [3] Visibility-aware TSDF fusion ▶ 4-state voxel grid
    │                                       (free/surface/occluded/unobservable)
    ├─▶ [4] OccluSynth Completer (3D U-Net) ▶ predicted geometry in occluded space
    │                                          ★ the core innovation
    └─▶ [5] Risk-graded A* planner ─▶ paths that respect hidden hazards
```

**[1] Depth and pose.** We use Meta's VGGT-Omega, a 1-billion-parameter open-weight
model, *frozen and off-the-shelf*. It turns ordinary photographs into depth maps
and camera positions. We did not train it — reusing a strong open model here let
the effort go where the novelty is.

**[2] Metric grounding.** VGGT's depth is *scale-ambiguous*: it knows the room's
shape but not whether it is 3 metres or 30 across. A robot needs real metres. We
fit a single scale factor using **RANSAC**, a classic technique that finds the
best-fitting relationship while ignoring outliers.

**[3] Visibility-aware fusion.** This is where the four-state labelling happens.
We trace each camera ray through the voxel grid and mark what it proves. This
stage is the project's own contribution, not something inherited.

**[4] The completer.** A 14.7-million-parameter 3D U-Net that looks at the partial
grid and predicts the geometry inside occluded space. Detailed in
[`COMPLETER_DEEPDIVE.md`](COMPLETER_DEEPDIVE.md).

**[5] The planner.** An A* path planner over the *completed* map, which treats
predicted-but-uncertain geometry as risk rather than certainty.

---

## Part 3 — The experiments, and what actually drove each decision

This is the honest engineering record, including the parts that did not work.

### 3.1 Decision: predict a signed distance field, not occupancy

**Driver:** meshing quality. A binary occupied/empty grid gives blocky, stair-stepped
output. A **signed distance field** — where each voxel stores *how far* it is from
the nearest surface, negative inside objects — supports smooth surface extraction
at sub-voxel precision. It also gives the loss function a gradient to follow
everywhere, not just at boundaries.

### 3.2 Decision: mask the loss to SURFACE ∪ OCCLUDED

**Driver:** intellectual honesty, which turned out to also be good engineering.
Unobservable voxels are excluded from training entirely. If we had supervised them,
the network would learn to reproduce ground-truth geometry it had no evidence for —
inflating every metric while learning nothing transferable. The masked loss means
our numbers describe genuine inference, not memorisation.

### 3.3 The feedback that triggered the deepest investigation

After the metrics were already good, feedback on the renders was blunt: they did
not *look* convincing. Grounds were not flat, walls were noisy, holes remained,
geometry ballooned.

The instinct was to tune the model. That instinct was wrong, and proving it wrong
was the most valuable work in the project.

**Four independent training interventions were tried:**

1. **Loss reweighting** — a grid over occupancy-BCE weight, free-space hinge weight,
   truncation distance, and near-surface weighting.
2. **Architecture v2** — explicit one-hot state channels as input plus a dedicated
   occupancy output head.
3. **3.6× more data** — 1528 multi-density training crops instead of 418.
4. **A stability-fixed retrain** — the multi-density run had diverged; restarting it
   at a 10× lower learning rate fixed that and produced a genuinely better validation
   loss (0.1794, with corroborating metrics all moving together).

**All four landed on the same precision/recall frontier.** Not "slightly worse" —
*the same curve*. Reweighting the loss slid the operating point along it; none of
them lifted it. The fourth was the most instructive: it improved on its own
validation set yet still lost on the frontier, and lost again when scored directly
against ground-truth geometry (11.2 cm vs 13.8 cm accuracy). That produced a
standing rule now written into the docs: **validation loss is not a proxy for the
frontier — always re-score before promoting a checkpoint.**

### 3.4 What actually fixed the renders

Having exhausted the model, the investigation turned to everything around it.

**Finding 1 — input density dominates.** Holding the model, checkpoint and every
setting fixed, and changing *only* the number of fused input frames: at 6 frames,
85% of the volume is unobservable and the output is fragmented shards; at 40 frames,
58% is unobservable and the reconstruction reads as an actual room. The renders had
been generated at the sparse setting. This was never a model failure.

**Finding 2 — the decision threshold was miscalibrated.** Because the model is
trained with a regression loss on an ambiguous problem, it hedges: its predicted
distances are biased toward "empty". Shifting the surface-extraction threshold
raised recall from 0.26 to 0.35 with no retraining at all. Critically, this is
*safe* in the render path because completed geometry is only ever written into
occluded voxels — observed-empty space keeps its measured values by construction.

**Finding 3 — hallucinations are topologically distinguishable.** The model
extrapolates from observed anchors. So a predicted chunk *touching* measured
surface is an extension of evidence; one floating in the void is a hallucination.
Filtering on exactly that removed **1946 of 1977 connected components while costing
only 6.8% of solid volume** — near-total speck removal, almost no real geometry lost.

**Two things that were tried and rejected**, recorded so nobody repeats them:
MC-dropout uncertainty gating (the uncertainty carried no information beyond the
prediction itself) and 8× test-time augmentation (negligible gain, 8× the cost).

### 3.5 The architecture question: why not Stable Diffusion?

Asked often, and the answer is a genuine engineering trade, not a default.

- **Modality.** Stable Diffusion is a 2D image model; our data is a metric 3D grid.
- **Determinism.** A diffusion sampler produces *a* plausible room, different every
  run. For a safety planner, "plausible" is the failure mode — a confidently
  hallucinated clear path is exactly the accident we exist to prevent.
- **Scale.** 14.7M parameters trains on a 16 GB laptop. 3D diffusion needs orders
  of magnitude more data and compute than 418 training crops.
- **Metrics.** Our KPIs are expected-error metrics, and mathematically the posterior
  *mean* — which regression learns — is optimal for those. A diffusion sample scores
  worse on every number we report while looking better.

The honest synthesis: *regression gives the average of all plausible rooms — accurate
but soft. Diffusion would give one sharp plausible room — convincing but unverifiable.
For a safety system we chose accurate, and recovered most of the visual crispness
through calibration and filtering afterwards.*

---

## Part 4 — Honest limitations

- **Resolution.** All reported numbers come from an interim 64³ checkpoint trained
  on a 16 GB MacBook. The 96³ GPU run is scripted and data-prepared but was never
  executed (no compute access). Given four failed 64³ interventions, resolution is
  the most credible remaining lever.
- **Precision ceiling.** Roughly 40–45% of predicted-solid voxels are wrong. Some of
  that is irreducible — hidden geometry is genuinely ambiguous — but not all.
- **Poses.** Evaluation uses ScanNet ground-truth camera poses; a deployed system
  would need its own pose estimation, adding error.
- **Baseline comparisons.** Published numbers from Atlas and NeuralRecon use a
  different protocol (full-scene, dense video) and are *not* directly comparable.
- **Scale.** 10 held-out scenes and 90 validation crops — enough to be indicative,
  not enough to be conclusive.

---

## Part 5 — Every concept, in plain English

*No technical background assumed. Read top to bottom, or dip in.*

### 5.1 Representing 3D space

**Voxel.** A pixel is a square of a 2D image; a **voxel** is a cube of 3D space.
Imagine dividing a room into sugar cubes and labelling each one. Ours are 5 cm on a
side. A room becomes a 3D grid of roughly 200 × 180 × 96 of them.

**Point cloud.** A cloud of dots floating in 3D, each marking a measured surface
point. Like a swarm of fireflies frozen in the shape of a room. Simple, but no
notion of "inside" or "solid".

**Mesh.** A surface made of connected triangles — the standard way 3D models are
stored. Has actual surfaces, so it can be rendered and lit.

**Signed Distance Field (SDF).** Instead of storing "solid or not," each voxel
stores *how far it is to the nearest surface*, with a sign: **negative inside**
objects, **positive outside**, and **exactly zero on the surface**. Like a
topographic map where sea level is the object's skin — below sea level is inside
the object. This is powerful because the surface is where the value crosses zero,
which can be located far more precisely than the grid itself.

**TSDF (Truncated SDF).** The same, but distances are capped: anything beyond ±10 cm
is just recorded as "far". We only care about precision near surfaces, and capping
saves memory and keeps the training signal focused.

**Marching cubes.** The algorithm that converts an SDF grid into a triangle mesh by
finding the zero-crossing surface. Think of it as shrink-wrapping a surface onto the
place where the distance field flips sign.

**Iso level.** The value marching cubes treats as "the surface." Normally zero.
Nudging it slightly positive grows objects a little (closing small holes); slightly
negative shrinks them. Our calibration fix was exactly this adjustment.

### 5.2 Cameras and visibility

**RGB-D.** An ordinary colour image (RGB) plus a **depth** channel — how far away
each pixel is. Like a photo where every pixel also knows its distance.

**Camera pose.** Where the camera was and which way it pointed, in 3D. Needed to
place each photo's measurements into a shared world.

**Frustum.** The pyramid-shaped volume a camera can see. Anything outside every
camera's frustum is our `unobservable` state.

**Occlusion.** Something hidden behind something else. The heart of the project.

**Ray casting.** Tracing a straight line from the camera through space to see what
it hits. Each ray proves two things: everything it passed through is empty, and
everything behind where it stopped is unknown.

**Ground truth (GT).** The correct answer, known independently. ScanNet provides
carefully-scanned complete 3D models of real rooms, so we can check predictions
against reality.

### 5.3 The neural network

**Neural network.** A function with millions of adjustable numbers (**parameters**
or **weights**), tuned automatically so that given an input it produces a desired
output. Not programmed with rules — shown examples until it generalises.

**Convolution.** A small filter slid across the data looking for local patterns. In
2D it might detect edges; in **3D convolution** (ours) it detects volumetric patterns
like "this looks like a floor continuing" across a small cube of space at a time.

**U-Net.** A network shape, named for its diagram. The **encoder** half progressively
shrinks the data, forcing it to summarise the big picture ("this is a room corner").
The **decoder** half expands back to full resolution. **Skip connections** pass
fine detail directly across from encoder to decoder, so precision is not lost during
the squeeze. Originally from medical imaging; ideal for tasks whose output is the
same shape as the input.

**Bottleneck.** The narrowest middle point, where the network holds only the
compressed, most abstract summary.

**GroupNorm / GELU.** Housekeeping. **Normalisation** keeps internal numbers in a
sane range so training stays stable. An **activation function** (GELU) adds the
non-linearity that lets a network learn more than straight lines.

**Parameters.** Our completer has 14.7 million adjustable numbers. Modern language
models have hundreds of billions — ours is deliberately small so it trains on a
laptop and runs fast enough for a robot.

### 5.4 Training

**Training / inference.** Training is learning from examples (slow, once). Inference
is using the trained model on new data (fast, repeatedly).

**Loss function.** A score of how wrong the model currently is. Training = making
this number go down. **L1 loss** is simply average absolute error.
**BCE (Binary Cross-Entropy)** is the standard loss for yes/no questions.

**Gradient descent, epoch, batch.** Training nudges every parameter slightly in the
direction that reduces the loss. A **batch** is a handful of examples processed at
once; an **epoch** is one full pass through all the training data. We trained for
tens of epochs, each taking ~20 minutes.

**Learning rate.** The step size of those nudges. Too high and training thrashes and
diverges — which is *precisely* what happened to our multi-density run, and cutting
it 10× fixed it. Too low and training crawls.

**Train / validation split.** Train on some scenes, evaluate on *different* ones
held back. Otherwise you are testing whether the model memorised, not whether it
learned. Our validation scenes never appear in training — verified explicitly.

**Overfitting.** When a model memorises training data instead of learning general
patterns: training loss keeps falling while validation loss rises. We saw this
signature at one point and read it correctly.

**Data augmentation.** Artificially expanding the dataset by transforming examples.
We rotate crops in 90° steps around the vertical axis and mirror them — but
**never flip vertically**, because that would put ceilings below floors and destroy
the gravity prior the model needs ("floors continue underneath things").

**Checkpoint.** A saved snapshot of the trained parameters.

**Warm start.** Beginning training from an existing checkpoint rather than randomly,
to build on prior learning.

### 5.5 Measuring success

This is the vocabulary for our results table.

**Precision and recall.** The two ways of being right, always in tension:
- **Precision** — of everything the model *claimed* was solid, what fraction really
  was? Low precision = inventing things that are not there (ballooning).
- **Recall** — of everything that *really* was solid, what fraction did the model
  find? Low recall = missing real obstacles (holes).

The classic analogy: a fishing net. A **fine** net catches every fish (high recall)
but also scoops up seaweed (low precision). A **selective** net catches only fish
(high precision) but lets many escape (low recall). You cannot maximise both; you
choose the trade-off that suits the consequences.

**Our chosen trade-off, and why:** we deliberately favour recall. *A missed obstacle
is a collision; a phantom obstacle is a slowdown.* Those costs are not symmetric, so
neither is our operating point.

**F1 score.** A single number blending precision and recall (their harmonic mean),
for when you want one figure of merit rather than two.

**Precision/recall frontier.** Sweeping the decision threshold traces a curve of
achievable (precision, recall) pairs. Moving *along* the curve is just choosing a
trade-off. Moving the *whole curve* outward is genuine improvement. **Our central
finding is that four separate training interventions failed to move the curve** —
strong evidence the ceiling lies in resolution and inherent ambiguity, not in
tuning.

**Chamfer distance.** Average distance between two 3D shapes — for each point on one,
distance to the nearest point on the other. Lower is better; ours is in centimetres.

**F-score @ 5 cm.** Treats a predicted surface point as correct if it lands within
5 cm of the true surface, then combines precision and recall on that basis. The
standard yardstick in 3D reconstruction.

**Completion ratio.** What fraction of the true surface was recovered within a
tolerance — literally, how much of the room did we get.

**Sign accuracy.** For an SDF, whether the *sign* is right — i.e. did we get
inside-vs-outside correct, ignoring how far off the distance was. It matters because
sign flips change occupancy, which is what a robot actually acts on.

**MAE (Mean Absolute Error).** Average size of the error, ignoring direction.

**Ablation study.** Systematically removing or changing one component to see whether
it was actually contributing. Our four failed interventions were ablations, and
negative results from them are genuine findings.

**MC Dropout.** A trick for estimating uncertainty: randomly switch off parts of the
network several times and see how much the answer wobbles. Lots of wobble = low
confidence. We implemented it, tested it as a filter, and found it added nothing
beyond the prediction itself — an honest negative result.

**Test-time augmentation (TTA).** Running the model on several rotated copies of the
input and averaging, to reduce noise. Tested; not worth 8× the compute here.

---

*Report reflects the repository state as of 2026-07-29. Every number in the companion
cheat sheet is reproducible from the scripts in `scripts/`.*
