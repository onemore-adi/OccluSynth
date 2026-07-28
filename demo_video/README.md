# OccluSynth Demo Video

The 6-minute animated demo, generated **entirely in code** against the real
OccluSynth repo so every data shot animates genuine voxel / mesh / planner
output — never a redrawn illustration. See [`../plan.md`](../plan.md) for the full
build plan and [`../OccluSynth_Demo_Video_Script.md`](../OccluSynth_Demo_Video_Script.md)
for the beat sheet.

This README is the operator's manual: **how to regenerate every clip from scratch.**

---

## 0. One source of truth for the look

Every colour, the frame rate, resolution and font live in
[`config/tokens.py`](config/tokens.py). **Nothing else may hardcode a colour or
fps** — every Manim and Blender script imports from `tokens.py`. This is what
guarantees continuity across all clips (plan.md §1). The hero amber is
`OCCLUDED = #E0A100` and must be that exact value everywhere it appears.

To import from anywhere in this tree:

```python
from config.tokens import OCCLUDED, SURFACE, FREE, VOID, FPS, RES, FONT
```

Run scripts with `demo_video/` on `PYTHONPATH` (e.g. run from this directory, or
`PYTHONPATH=demo_video ...`) so `config` resolves.

---

## 1. Directory layout (plan.md §2)

```
demo_video/
  config/tokens.py   # design tokens — single source of truth (colours, fps, res, font)
  export/            # small scripts: dump repo data -> animation-ready files in renders/data/
  manim/             # manim scene files (one file per clip group)
  blender/           # blender python scripts (one per 3D set piece) + procedural robot
  assets/robot/      # cached robot .blend
  audio/vo/          # narration wavs (per scene)
  audio/sfx/         # 3 sfx cues
  renders/           # per-clip mp4/pngs, named by shot id
  renders/data/      # exporter outputs — the only bridges between the repo and the video
  build/             # final assembly output
  build.sh           # ffmpeg concat + grade + audio mux (Phase 5)
  README.md          # this file
```

---

## 2. Prerequisites

| Tool | Used for | Install / run |
|---|---|---|
| Python 3 + repo env | exporters, smoke test | the OccluSynth repo environment |
| **Manim Community** | 2D / diagrams / data plots / titles / all text cards | `pip install manim`, then `manim -qk scene.py` |
| **Blender** (headless) | 3D set pieces + procedural robot | `blender -b -P script.py` |
| **ffmpeg** | concat, LUT, grain, vignette, audio mux | `bash build.sh` |

Everything except Blender rendering runs headless and fast. Blender runs locally
(MPS/CPU is fine for these short clips).

---

## 3. Regenerating everything — end to end

The pipeline is **repo data → `renders/data/` bridges → clips → assembled film**.
Run these phases in order (plan.md §7). All commands are run from `demo_video/`.

### Phase 1 — Export bridges (+ smoke test gate)

The smoke test is the gate that stops you sinking days into Blender against
missing data. Run it **first**, before writing/running any exporter:

Run the exporters with the repo's **`.venv312`** interpreter (Python 3.12 —
open3d / torch / skimage). The exporters read *cached* OccluSynth outputs; none
re-runs the ~14-min VGGT inference. `dump_meshes` does run the cached completer
checkpoint once (tiled 96³ forward, seconds on MPS) to produce the completed mesh,
and needs `scikit-image` for marching cubes (`../.venv312/bin/pip install
"numpy<2" scikit-image==0.24.0` — the numpy pin keeps the occlusynth stack intact).

```bash
# 1. Confirm the raw repo assets exist (caches, meshes, results.json, ...).
python export/smoke_test.py

# 2. Run the exporters that dump repo data into renders/data/:
../.venv312/bin/python export/dump_voxels.py     # -> renders/data/voxels_scene0000_00.npz
../.venv312/bin/python export/dump_meshes.py     # -> renders/data/mesh_{tsdf_only,occlusynth_completed,gt}.ply
../.venv312/bin/python export/dump_planner.py    # -> renders/data/planner.npz
../.venv312/bin/python export/dump_depth.py      # -> renders/data/depth_{rgb,vggt_raw,vggt_calibrated,gt}.png
../.venv312/bin/python export/dump_benchmark.py  # -> renders/data/benchmark_dots.npz

# 3. Confirm the bridges landed. Must exit 0 before any Blender work:
python export/smoke_test.py --check-exports
```

If the planner heatmap source check fails, regenerate it (cached SceneGrid, no
inference): `../.venv312/bin/python scripts/run_planner.py --scene scene0000_00`.

> **GATE:** do not start Phase 3 (Blender) until
> `python export/smoke_test.py --check-exports` exits 0.

### Phase 2 — Manim clips (fast, no Blender)

Each Manim scene imports `config/tokens.py` (via `manim/_manim_common.py`) and
renders 1920×1080 @ 24fps. Manim lives in its **own** venv (`.venv-manim`, Python
3.12) to keep it away from the occlusynth `numpy<2` pin; it needs the cairo/pango
system libraries. All counters use a Pango-`Text` updater (`value_text`) instead
of Manim's `DecimalNumber`, so **no LaTeX is required**.

One-time setup:

```bash
brew install cairo pango pkg-config          # system libs for pycairo/pango
python3.12 -m venv .venv-manim
./.venv-manim/bin/pip install manim
```

Render (run from `demo_video/`). `-q h` = 1920×1080; `--fps 24` forces 24fps;
the produced file lands under `media/videos/.../` and is copied to `renders/`:

```bash
M=./.venv-manim/bin/manim
# opaque .mp4 clips (VOID background)
$M -q h --fps 24 --format mp4 -o s3a_depth   manim/s3a_depth.py  S3A_Depth
$M -q h --fps 24 --format mp4 -o s4b_split   manim/s4b_split.py  S4B_Split
$M -q h --fps 24 --format mp4 -o s6a_dots    manim/s6_results.py S6A_Dots
$M -q h --fps 24 --format mp4 -o s6b_numbers manim/s6_results.py S6B_Numbers
$M -q h --fps 24 --format mp4 -o s7b_end     manim/text_cards.py S7B_End
# alpha .mov overlays ( -t = transparent, argb )
$M -q h --fps 24 --format mov -t -o s1d_text  manim/text_cards.py S1D_Freeze
$M -q h --fps 24 --format mov -t -o s2a_title manim/text_cards.py S2A_Novelty
$M -q h --fps 24 --format mov -t -o s5b_text  manim/text_cards.py S5B_Text
# then: cp media/videos/<stem>/1080p24/<id>.<ext> renders/<id>.<ext>
```

Clips produced here (plan.md §4) → output id:
**2A** novelty title → `s2a_title.mov`, **3A** depth compare → `s3a_depth.mp4`,
**4B** split-vs-baseline → `s4b_split.mp4`, **6A** 100-dot benchmark →
`s6a_dots.mp4`, **6B** numbers → `s6b_numbers.mp4`; text overlays **1D** →
`s1d_text.mov`, **5B** hero line → `s5b_text.mov`, **7B** end card →
`s7b_end.mp4`. 3A reads `renders/data/depth_*.png` and 6A reads
`renders/data/benchmark_dots.npz` (Phase 1 bridges).

> **Note:** always render Phase-2 clips against a clean `media/` — Manim's
> per-animation cache can serve stale frames after a code edit even with
> `--disable_caching`. `rm -rf media` before a re-render if in doubt.
>
> On-screen text comes from plan.md §4. The companion
> `OccluSynth_Demo_Video_Script.md` is not in the repo, so the 2A novelty-card
> wording (`manim/text_cards.py:S2A_Novelty`) is composed from the project thesis
> — reconcile it with the script if/when it is available.

### Phase 3 — Blender hero renders

Each Blender script imports colours/fps from `tokens.py` (via
`blender/_bl_common.py`) and writes one mp4 per shot id to `renders/`. Built and
tested with **Blender 5.1** (`brew install --cask blender`); the binary is
`/Applications/Blender.app/Contents/MacOS/Blender`. Rendering uses **EEVEE** with
the **Standard** view transform so the token colours land exactly.

> Blender 5.x's image settings have no FFMPEG muxer, so `_bl_common.render()`
> writes a PNG sequence to `renders/_frames/<id>/` and encodes to
> `renders/<id>.mp4` with the **system ffmpeg** (libx264, yuv420p — matching the
> Manim clips). `renders/_frames/` is a scratch dir (git-ignored).

```bash
BL=/Applications/Blender.app/Contents/MacOS/Blender

# Build + cache the procedural OS-1 robot first (assets/robot/os1.blend):
$BL -b -P blender/robot.py

# Hero / data shots:
$BL -b -P blender/s2c_amber_reveal.py                 # 2C ★ -> s2c_amber_reveal.mp4
$BL -b -P blender/s4a_growth.py                       # 4A ★ -> s4a_growth.mp4
$BL -b -P blender/s5_planner.py -- --shot 5a          # 5A   -> s5a_collide.mp4
$BL -b -P blender/s5_planner.py -- --shot 5b          # 5B   -> s5b_detour.mp4

# Room shots — ONE shared set + camera rig (build_room/base_camera in room_shots.py);
# 7A reuses Scene-1A's camera + set exactly for the bookend:
$BL -b -P blender/room_shots.py -- --shot 1a          # 1A -> s1a_drivein.mp4
$BL -b -P blender/room_shots.py -- --shot 1c          # 1C -> s1c_reveal.mp4
$BL -b -P blender/room_shots.py -- --shot 7a          # 7A -> s7a_callback.mp4
```

Fast smoke test of any shot: prefix with `OCCLU_PREVIEW_FRAMES=N` to cap the
frame count (e.g. `OCCLU_PREVIEW_FRAMES=2 $BL -b -P blender/s4a_growth.py`).

Data sources (Phase-1 bridges): 2C ← `voxels_scene0000_00.npz` (emissive
per-state cubes, keyframed amber fog-in); 4A ← `mesh_occlusynth_completed.ply`
(Build modifier, amber→red) vs ghosted `mesh_gt.ply`; 5A/5B ← `planner.npz`
(naive vs risk-graded path over the amber hazard field). The robot's one cyan LED
accent (`CYAN_LED` in `_bl_common.py`) is the only colour outside `tokens.py` —
it's a named constant per the robot spec, not hardcoded per scene.

### Phase 3B / 3C — Act 2 deep-dive segments (D0–D8)

Act 2 (~4:35) earns Act 1's claims component-by-component. Extra Phase-1 bridges
feed it: `scale_fit.npz`, `rays.npz`, `uncertainty.npz`, `costmap.npz`,
`completer_meta.json`, `geometry.json` (all covered by `smoke_test.py
--check-exports`).

**Phase 3B — Manim (no 3D):** `_act2_common.py` holds the shared pipeline diagram
+ 4-state legend. Segments render to `renders/d*.mp4`:

```bash
M=./.venv-manim/bin/manim
# OCCLU_HOLD pads each segment's finished-diagram tail to its VO-paced length
OCCLU_HOLD=6  $M -q h --fps 24 --format mp4 -o d0_bridge          manim/act2_overview.py D0_Bridge
OCCLU_HOLD=18 $M -q h --fps 24 --format mp4 -o d1_overview        manim/act2_overview.py D1_Overview
OCCLU_HOLD=15 $M -q h --fps 24 --format mp4 -o d2_perception      manim/d2_perception.py D2_Perception
OCCLU_HOLD=12 $M -q h --fps 24 --format mp4 -o d4_completion_arch manim/d4_completion.py D4_CompletionArch
OCCLU_HOLD=18 $M -q h --fps 24 --format mp4 -o d6_planner         manim/d6_planner.py    D6_Planner
OCCLU_HOLD=16 $M -q h --fps 24 --format mp4 -o d7_validation      manim/d7_d8_close.py   D7_Validation
OCCLU_HOLD=16 $M -q h --fps 24 --format mp4 -o d8_close           manim/d7_d8_close.py   D8_Close
```

**Phase 3C — Blender (reuse the voxel/mesh loaders):**

```bash
$BL -b -P blender/d3_fusion.py       # D3 ★ ray-cast fusion (rays.npz + voxels)
$BL -b -P blender/d4_volumes.py      # D4 input volumes + masked-loss region
$BL -b -P blender/d5_uncertainty.py  # D5 MC-dropout p_occ (uncertainty.npz)
```

D3/D4vol/D5 are short; `build.sh` freeze-pads them (they end on a full frame) to
their VO-paced target via `pad_target()`.

**Two-act timeline** (`build.sh` §6): Act 1 (s1a…s6b) → `d0_bridge` → D1…D8 →
`s7a_callback` → `s7b_end`, with a longer dissolve on the Act1→D0 handoff
(`xfade_into`). Final runtime **6:05**. Note: `build.sh` uses bash-3.2-safe case
functions (macOS system bash), not associative arrays.

### Phase 4 — Audio

Put per-scene narration wavs in `audio/vo/` and the 3 SFX cues in `audio/sfx/`
(sub-bass hit @ Scene 1 freeze, soft pad @ Scene 2 amber, mechanical click @
Scene 5 commit). See plan.md §5.

### Phase 5 — Assemble, grade, mux

```bash
bash build.sh
```

`build.sh` concatenates clips in script order (with the script's crossfades),
applies **one LUT derived from `tokens.py`**, adds grain + vignette, overlays the
alpha text `.mov`s on their timecodes, muxes `audio/vo` + `audio/sfx`, normalises
to -14 LUFS, and exports `build/OccluSynth_demo_final.mp4` (H.264, 1080p24).

> `build.sh` is currently a **Phase 0 stub** — it prints its plan and exits 0. It
> is implemented in Phase 5 (plan.md §6).

---

## 4. Regenerating a single clip

1. If the clip is data-driven, make sure its bridge in `renders/data/` is current
   — re-run the matching `export/dump_*.py`, then `python export/smoke_test.py
   --check-exports`.
2. Re-run just that clip's Manim scene or Blender script (Phase 2 / 3 above). It
   overwrites `renders/<shot-id>.*`.
3. Re-run `bash build.sh` to fold it back into the final film.

Clip → source map (plan.md §4):

| Shot id | Tool | Source data |
|---|---|---|
| 1A / 1C / 7A | Blender | procedural room set + `os1.blend` (7A reuses 1A camera) |
| 1D, 2A, 5B text, 7B | Manim | text overlays (alpha `.mov`) |
| 2C ★ amber reveal | Blender | `renders/data/voxels_scene0000_00.npz` |
| 3A depth compare | Manim | `renders/data/depth_*.png` |
| 4A ★ couch-growth | Blender | `renders/data/*.ply` (completed vs GT mesh) |
| 4B split | Manim | `renders/data/*.ply` / numbers |
| 5A / 5B ★ planner | Blender | `renders/data/planner.npz` |
| 6A dots | Manim | `renders/data/benchmark_*.npz` |
| 6B numbers | Manim | metrics from results.json |

★ = hero shot; spend the most polish on 2C (amber reveal) and 5B (detour).

---

## 5. Acceptance (plan.md §6, §9)

- Final runtime 5:35–6:00.
- Every data set piece sourced from real repo data via `export/` — no
  illustration stands in for a measured result.
- Every colour on screen matches `config/tokens.py`; amber `#E0A100` means
  hidden/imagined everywhere.
- Robot identical across 1A, 1C, 5A, 5B, 7A (same `os1.blend`).
- Scene 7A reuses Scene 1A camera + set (bookend matches).
- One LUT + grain applied globally to Blender **and** Manim footage.
- Hero reveals (2C amber, 5B detour) each hold ≥3 s.
