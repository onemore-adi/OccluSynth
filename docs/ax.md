# AX — Agentic AI Development & Open-Weight Models

> **Required submission artefact.** This document explains how OccluSynth used
> open-weight models and agentic development tooling. Some specifics below are
> marked `TODO/verify` — please confirm or edit them to match your exact setup
> before final submission.

This document has two halves:

1. **Open-weight models *in the solution*** — what OccluSynth runs at inference time.
2. **Agentic tooling *to build the solution*** — how the codebase itself was developed.

---

## 1. Open-weight models in the solution

| Model | Role | Weights | License |
|---|---|---|---|
| **VGGT-Omega (1B, 512)** | Feed-forward monocular depth + camera pose from sparse RGB | Open weights — `vggt/vggt-omega/checkpoints/vggt_omega_1b_512.pt` | Open (`TODO/verify` exact license + HF link) |
| **OccluSynth Completer** | 3D U-Net predicting SDF in occluded voxels (trained by us) | `checkpoints/interim_64_aug/completer_best.pt` | MIT (this repo) — not yet published to HF |

VGGT-Omega is used **frozen**, off the shelf, as a geometry predictor. We do not
fine-tune it. Its scale-ambiguous depth is lifted to metric scale by our own
RANSAC grounding step (`src/occlusynth/models/metric_grounding.py`), and its
pose output is deliberately **not** used for fusion (ATE ~70 cm ≫ 5 cm voxel
pitch — see [`architecture.md`](architecture.md) §Camera Pose Strategy).

The only model we *trained* is the OccluSynth Completer, a from-scratch 3D U-Net.

---

## 2. Agentic tooling used to build OccluSynth

### Setup / harness

The codebase was developed using **Claude Code** (Anthropic's agentic coding
CLI) driving the **Claude Opus** model, running locally against the repo on
Apple Silicon (MPS). Claude Code acts as the agentic harness: it reads files,
runs shell commands and tests, edits source, and iterates against real tool
output rather than producing code blind.

`TODO/verify`: list any other assistants/IDEs you used (e.g. Copilot, Cursor,
plain ChatGPT) so the attribution is complete.

### Agentic workflow — plan-driven loop

The central pattern was a **living-plan loop** anchored by
[`context.md`](../context.md): a single source-of-truth document tracking
completed chapters, current status, the two-environment setup, known bugs and
their fixes, and next steps. Each work session:

1. Re-read `context.md` to restore state (the agent starts each session cold).
2. Pick the next chapter; plan the change against the existing `src/` contracts.
3. Implement in `src/occlusynth/` with orchestration kept thin in `scripts/`.
4. Run the relevant script / `pytest` and read the actual output.
5. Update `context.md` and the per-sprint docs
   ([`completer_sprint.md`](completer_sprint.md), etc.).

This kept architecture decisions explicit and reviewable rather than buried in
chat history — the docs in `docs/` are themselves agentic-workflow artefacts.

### Tool use / tool chaining

- **Filesystem + grep** for navigating a multi-module package.
- **Bash** for running the two venvs (`.venv` for VGGT/MPS, `.venv312` for
  open3d/rerun/training), `pytest`, and the eval scripts — output fed straight
  back into the next reasoning step.
- **ffmpeg + open3d** chained from Python for the demo-video pipeline
  (`turntable.py` renders `.ply` orbits → `clipkit.sh`/`assemble_video.sh`
  stitch with captions). The turntable's up-axis auto-detection was an
  agentic debugging win — the camera bug was diagnosed by inspecting the
  mesh bounding-box extents, not by guessing.

### Memory / context handling

Two layers:
- **Persistent project memory** — durable facts about the project, environment,
  and hard-won gotchas (e.g. *never edit a module a running `num_workers>0`
  training job imports — it crashes the job*) were stored so they survive
  across sessions.
- **In-repo `context.md`** — the human-readable, version-controlled plan that
  doubles as the agent's working memory and as reviewer documentation.

### Reasoning & planning pipeline

Work was decomposed into **sprints/chapters** (visibility fusion → completer
data → completer training → evaluation → planner → cross-dataset → demo), each
with a written spec and explicit input/output contracts before implementation.
This is visible in the commit history and in the `docs/*_sprint.md` files.

### Multi-agent orchestration

`TODO/verify`: state honestly whether you used sub-agents / parallel agents. If
development was single-agent (one Claude Code session at a time), say so — that
is a perfectly valid answer and better than overclaiming.

---

## What worked

- **The `context.md` living-plan loop.** Because the agent re-derives state each
  session, a single well-maintained plan file was the highest-leverage artefact
  — it made cold restarts cheap and kept decisions auditable.
- **Thin scripts, fat `src/`.** Keeping all logic in `src/occlusynth/` with
  scripts as orchestration made the code testable and let the agent reuse
  contracts instead of re-deriving them.
- **Running real tools, not trusting output.** Every metric in the README came
  from actually executing the eval scripts and reading JSON, which caught
  several silent errors (e.g. TSDF-only scoring 0 on occluded regions *by
  construction* — surfaced by inspecting `n_pred`, then documented rather than
  hidden).
- **Inspecting data to debug geometry.** The turntable rotation bug was fixed by
  having the agent print bounding-box extents and reason about the up axis,
  instead of trial-and-error on camera flags.

## What did NOT work

- **Trusting VGGT-Omega pose.** The original plan was to fuse on predicted poses;
  ~70 cm ATE made that unusable. This forced the documented pivot to GT poses —
  a real limitation, not hidden.
- **Full-resolution training on Apple Silicon.** 96³ crops OOM on MPS; the agent
  could script the A100 run but not execute it this phase. The interim 64³
  checkpoint is what the reported KPIs come from.
- **Screen-recording 3D views by hand.** Jittery and unrepeatable — replaced by
  the deterministic `turntable.py` render pipeline.
- `TODO/verify`: add any agentic dead-ends you hit (a tool that wasn't worth the
  setup, a workflow you abandoned). Honest negatives score well here.

---

*See [`architecture.md`](architecture.md) for the full technical design and
[`README.md`](../README.md) for the artefact index.*
