# OccluSynth Demo Video — Build Plan (generate with Claude)

**Purpose:** produce a **~6-minute** animated demo entirely in code, driven by Claude Code against the real `OccluSynth` repo, so every data shot animates *genuine* voxel/mesh/planner output — not redrawn illustrations. Titles, transitions, and abstract beats are code-generated too, so nothing needs a video editor's GUI except final trimming (optional).

**Two-act structure (this is the update).** The current render is a tight **~1:30 cinematic sizzle** — keep it as **Act 1**. It sells the *what* fast and consequence-first, with jargon deliberately stripped. On its own it's too short and too shallow for the judging criteria. So we add **Act 2 (~4:35): a component-by-component deep dive** that earns everything Act 1 claimed — one segment per stage, substance-first, with a real visualization behind every technical statement. Act 1 → a short "open the hood" bridge → Act 2 → the closing thesis card. Target total **6:00–6:15**.

> Rule of thumb: **Act 1 = the trailer. Act 2 = the paper.** Anything that felt like too much detail for Act 1 (RANSAC, masked L1, obliquity correction, MC-dropout, cost-map math) now has a home — always paired with a picture, never narrated over a black slide.

**Inputs this plan assumes exist** (from `context.md`):
- Fused voxel grid + states — `scripts/run_visibility.py` (free/surface/occluded/unobservable)
- Meshes — `demo_outputs/tsdf_fusion/*.ply`, GT `scene0000_00_vh_clean_2.ply`
- Completer eval — `scripts/eval_completer.py`
- Planner path + heatmap — `scripts/run_planner.py` (13.56 m / 244 cells, `docs/images/planner_scene0000_00.png`)
- Depth compare — `scripts/run_baseline.py` (RGB | GT | VGGT), `scripts/run_scale_fit.py`
- Safety benchmark — `scripts/run_safety_benchmark.py` → `demo_outputs/safety_benchmark/results.json` (21.3% awareness, 15.5% avoidance on scene0556_00)

**Companion docs:** `OccluSynth_Demo_Video_Script.md` (v2 cinematic cut — the beat sheet) and `OccluSynth_AI_Shot_Generation.md` (design bible — palette, robot spec, grade). This plan reuses their design tokens exactly.

---

## 0. Toolchain (all scriptable, all Claude-writable)

| Layer | Tool | Runs how | Used for |
|---|---|---|---|
| 2D / diagrams / data plots / titles | **Manim** (Community) | `manim -qk scene.py` | pipeline map, depth compare, cost map, 100-dot benchmark, number tickers, all text cards |
| 3D data set pieces | **Blender headless** | `blender -b -P script.py` | amber reveal, couch-growth completion, planner detour, robot |
| Robot | Blender Python (procedural) | in the Blender scripts | procedural OS-1 (no external asset needed) |
| Audio | your VO or TTS + a soft pad | — | narration + 3 SFX cues |
| Assembly / grade / grain | **ffmpeg** (+ optional DaVinci) | `bash build.sh` | concat, LUT, grain, vignette, audio mux |

Everything except Blender rendering runs headless and fast. Blender runs on your machine (MPS/CPU fine for these short clips).

---

## 1. Design tokens — single source of truth

Create `demo_video/config/tokens.py`. Claude generates it once; every Manim and Blender script imports it. Pulled verbatim from the Style Bible:

```python
# Palette (hex)
NAVY        = "#0A3D91"
NAVY_SHADOW = "#062A63"
FREE        = "#2E7D32"   # green  — seen empty
SURFACE     = "#C0272D"   # red    — measured solid
OCCLUDED    = "#E0A100"   # amber  — imagined (HERO colour)
UNOBS       = "#6E7681"   # grey   — no evidence
VOID        = "#0B0E14"   # near-black background
WARMWHITE   = "#F4F1EA"

FPS   = 24
RES   = (1920, 1080)      # 16:9
FONT  = "Inter"           # or Helvetica Now / Söhne
```

**Rule for Claude:** no scene may hardcode a colour or fps; import from `tokens.py`. This is what guarantees continuity across every clip.

---

## 2. Directory layout

```
demo_video/
  config/tokens.py
  export/            # small scripts to dump repo data → animation-ready files
  manim/             # manim scene files (one file per clip group)
  blender/           # blender python scripts (one per 3D set piece) + procedural robot
  assets/robot/      # cached robot .blend
  audio/vo/          # narration wavs (per scene)  + sfx/
  renders/           # per-clip mp4/pngs, named by shot id
  build/             # final assembly
  build.sh           # ffmpeg concat + grade + audio
  README.md          # how to regenerate everything
```

---

## 3. Data export step (do this before animating)

Claude writes small scripts in `export/` that dump exactly what the animators need, so Blender/Manim never re-run heavy inference:

- `export/dump_voxels.py` → `renders/data/voxels_scene0000_00.npz` = voxel centres (N,3) + state label (N,) + p_occ (N,). From the cached `SceneGrid`.
- `export/dump_meshes.py` → copies TSDF-only mesh, OccluSynth completed mesh, GT mesh into `renders/data/` as clean `.ply`.
- `export/dump_planner.py` → `renders/data/planner.npz` = cost map (H,W), start, goal, naive straight path, OccluSynth path, hazard cluster mask.
- `export/dump_depth.py` → three registered PNGs (RGB, VGGT raw, VGGT calibrated, GT) at identical crop for the split.
- `export/dump_benchmark.py` → reads `results.json` → the 21/100 hazard flags for the dot grid.

**Act 2 deep-dive bridges** (new — the deep dives need richer data than the sizzle):
- `export/dump_scale_fit.py` → `scale_fit.npz` = per-frame anchor pairs (d_pred, d_gt) + RANSAC inlier mask + fitted (a, b); the 4-method table (GlobalScalar…PerFrameRANSAC: ARE/RMSE/δ<1.05/δ<1.25); multi-scene ARE array (10 scenes); noise-ablation curve (σ ∈ {0,0.01,0.05,0.10,0.25}, anchors {500,250,100,50}).
- `export/dump_rays.py` → camera extrinsics/intrinsics + a downsampled ray set for the fusion ray-cast animation (reuses `voxels_scene0000_00.npz`).
- `export/dump_uncertainty.py` → `uncertainty.npz` = MC-dropout mean/std volume + p_occ volume + reliability-diagram bins + measured ECE (0.42). From `predict_with_uncertainty` / `eval_calibration.py`.
- `export/dump_costmap.py` → `costmap.npz` = z-collapsed cost-map layers, per-column state, graded cost, A* expansion order, naive + OccluSynth paths, hazard mask (may extend `dump_planner.py`).
- `export/dump_completer_meta.py` → `completer_meta.json` = U-Net layer shapes for the architecture diagram + training curve (val_loss per epoch → 0.1857 @ 32) + completer metrics table.
- `export/dump_geometry.py` → `geometry.json` = Chamfer-L1 / F@5cm / Occl-F tables, surface-vs-occluded split (TSDF-only vs OccluSynth).

**Acceptance:** each produces a file in `renders/data/` and prints its shape. These are the only bridges between the repo and the video. **Extend `smoke_test.py --check-exports` (§3.5) with a row for every new `.npz`/`.json` above** so the gate covers Act 2 too.

---

## 3.5 Smoke test — the gate before Blender

Run this **before Phase 3**. It verifies every source path the exporters depend on, and (with `--check-exports`) that the exporters actually produced their outputs. It fails loudly with a non-zero exit code, so you never sink days into Blender against missing data.

Claude creates `export/smoke_test.py` (source + Act 1 export rows as before, **plus the Act 2 export rows**).

**How it's used:**
1. Run `python export/smoke_test.py` **first** — before writing any exporter — to confirm the raw repo assets exist.
2. After the Phase 1 exporters run, run `python export/smoke_test.py --check-exports` to confirm the `renders/data/` bridges landed (Act 1 + Act 2).
3. Only when both are green does Phase 3 (Blender) begin.

---

## 4. Scene-by-scene build tasks

### ACT 1 — Cinematic sizzle (0:00 → ~1:25) — *already rendered; keep as-is*

This is the 1:30 you already have. Two changes for the 6-min cut: (1) tighten so it lands ~1:25, and (2) **the closing callback + thesis card (7A/7B) move to the very end of Act 2** — Act 1 now hands off to the D0 bridge instead of closing out. Everything else stays.

Scenes 1–7 (1A/1C/1D, 2A/2C, 3A, 4A/4B, 5A/5B, 6A/6B, 7A/7B) as originally built.

---

### ACT 2 — Component deep-dives (~1:40 → ~6:00) — *this is the runtime you're adding*

Substance-first. One segment per stage. Every technical claim is spoken **only while its visualization is on screen**. Manim carries diagrams and plots; Blender carries the 3D data. All import `tokens.py`. Keep the Act-1 4-state legend parked bottom-left through all of Act 2 so colours never need re-explaining.

- **D0 — Bridge / "open the hood" (~0:15)** · Manim · Act-1 final frame pulls back and resolves into the 5-stage pipeline diagram. Text: *"That's the demo. Here's how every stage actually works."* → `renders/d0_bridge.mp4`
- **D1 — System overview (~0:35)** · Manim · animated architecture — five stages light left→right (Perception → Fusion → Completion → Uncertainty → Planner); INPUT (RGB + sparse anchors) → OUTPUT (occlusion-aware SDF + collision-safe path) rail underneath; the four-state legend docks bottom-left for the rest of Act 2. → `renders/d1_overview.mp4`
- **D2 — Perception in depth (~0:45)** · Manim + `scale_fit.npz` · frozen VGGT-Omega block; raw-depth histogram near ~0.2 vs GT 1.3–2.9 m; 500 stratified anchors; **animated scatter of d_pred vs d_gt with the RANSAC line fitting live**, inliers green / outliers red; the 4-method bar chart animating GlobalScalar 0.060 → PerFrameRANSAC **0.024** ARE; a 10-scene strip (mean 0.024); the noise-robustness curve. → `renders/d2_perception.mp4`
- **D3 — Visibility-aware fusion in depth (~0:50)** ★ · Blender + `voxels_*.npz` + `dump_rays.py` · camera **rays cast into the 5 cm grid** — carving FREE (green), stopping at SURFACE (red), marking the shadow behind as OCCLUDED (amber), leaving out-of-frustum as UNOBSERVABLE (grey). Then the **obliquity correction** (20.7% → 6.3%). Cross-section slice. Real counts: free 104k · surface 20.1k · **occluded 194.9k (61%)** · unobs 909.6k. → `renders/d3_fusion.mp4`
- **D4 — Completion in depth (~0:50)** ★ · Manim (arch) + Blender (volumes) · animated **3D U-Net block diagram** (14.7 M); the **3 input channels** (sdf, weight, p_observed); the **masked-L1 region** on surface ∪ occluded; completer metrics table (MAE 45.27 → **27.14 cm**; sign-acc 0.30 → 0.72; compl<5cm 0.06 → 0.35); training curve to val_loss 0.186 @ epoch 32; honest tag: *interim 64³*. → `renders/d4_completion.mp4`
- **D5 — Uncertainty in depth (~0:35)** · Blender + Manim + `uncertainty.npz` · **MC-dropout** as 16 overlaid passes; per-voxel std → p_occ volume; **reliability diagram** with honest **ECE 0.42** and post-temperature target < 0.05. → `renders/d5_uncertainty.mp4`
- **D6 — Planner in depth (~0:40)** · Manim + `costmap.npz` · build the **2D cost map layer by layer**; colour columns by cost (SURFACE → ∞, OCCLUDED → 1 + λ·p_occ, FREE → 1, UNOBS → 6); overlay **8-connected A\*** expanding; the final **13.56 m / 244-cell** path; 15.5% avoidance. → `renders/d6_planner.mp4`
- **D7 — Validation & benchmark in depth (~0:35)** · Manim + `geometry.json` + `results.json` · the **ScanNet-native safety benchmark**; Metric 1 + Metric 2; geometry table (Chamfer 3.11 → **1.77 cm**, F@5cm 74.1 → 83.5%, Occl-F 0 → **32%**); 55+ tests / 18 planner tests; 7-Scenes portability. → `renders/d7_validation.mp4`
- **D8 — Honesty, roadmap & close (~0:30)** · Manim · three columns (Built & tested / Partial gaps / Phase 2); then **dissolve to the reused 7A callback and 7B thesis card** as the true ending. → `renders/d8_close.mp4`, then `renders/s7a_callback.mp4` + `renders/s7b_end.mp4`

**Act 2 ≈ 4:35.** Act 1 (~1:25) + D0 bridge + Act 2 → **~6:15**; trim segment tails to land 6:00–6:15. **Protect D3 (fusion rays) and D4 (completion).**

---

## 5. Audio

- **Act 1 VO:** consequence-first narration. `audio/vo/act1/`.
- **Act 2 VO:** a new, longer, substance-first track — one block per deep dive (D1–D8). `audio/vo/act2/`.
- **SFX (3 only, Act 1):** sub-bass hit, soft pad, mechanical click. `audio/sfx/`.
- **Music:** optional low ambient bed at ~-24 LUFS.
- Claude writes `build.sh` to mux VO + SFX to the assembled picture on each act's timecodes.

---

## 6. Assembly, grade, grain (ffmpeg)

`build.sh` does, in order:
1. Concatenate clips in **final timeline order**: Act 1 (s1…s6) → **D0 bridge** → Act 2 (D1…D8) → 7A callback → 7B thesis card. Short crossfades where the script says "wipe/dissolve"; a slightly longer, deliberate dissolve on the Act 1 → D0 handoff.
2. Apply **one LUT** (teal-navy shadows / warm highlights) to the whole timeline from the tokens.
3. Add subtle film grain + vignette globally.
4. Overlay the alpha text `.mov`s at their timecodes.
5. Mux audio (Act 1 + Act 2 VO + SFX); normalise to -14 LUFS; export `build/OccluSynth_demo_final.mp4` (H.264, 1080p24).

**Acceptance:** final runtime **6:00–6:15**; the Act 1 sizzle still reads as a self-contained ~1:25 opener; every hex on screen matches `tokens.py`; amber = `#E0A100` everywhere.

---

## 7. Execution phases (milestones for Claude Code)

- **Phase 0 — Scaffold.** *(done)*
- **Phase 1 — Export bridges + smoke test** (with Act 2 rows added). *(1 day)*
- **Phase 2 — Act 1 Manim clips.** *(done)*
- **GATE:** do not start any Blender phase until `python export/smoke_test.py --check-exports` exits 0.
- **Phase 3 — Act 1 Blender hero renders.** *(done)*
- **Phase 3B — Act 2 Manim segments:** D0, D1, D2, D4-arch, D6-costmap, D7, D8. *(1–2 days)*
- **Phase 3C — Act 2 Blender segments:** D3 ray-cast fusion, D4 volumes, D5 uncertainty. *(1–2 days; D3 + D4 priority)*
- **Phase 4 — Audio.** *(1 day)*
- **Phase 5 — Assemble + grade** (two-act timeline). *(1 day)*

**Act 2 Manim (3B) can run in parallel with Act 1 Blender (3) since it needs no 3D.**

---

## 9. Final QA checklist (Claude verifies after assembly)

- [ ] `python export/smoke_test.py --check-exports` exits 0, **including the Act 2 rows**.
- [ ] Runtime **6:00–6:15**; Act 1 still reads as a self-contained ~1:25 opener; Act1→D0 dissolve marks the tonal shift.
- [ ] Act 2 covers **all five components + benchmark + honesty** (D1–D8), each with a real visualization.
- [ ] Every data set piece (2C, 4A, 4B, 5A/5B, 6A/6B **and D2, D3, D4, D5, D6, D7**) sourced from real repo data via `export/`.
- [ ] Jargon appears **only in Act 2** — Act 1 stays consequence-first.
- [ ] Robot identical across 1A, 1C, 5A, 5B, 7A (same `os1.blend`).
- [ ] Scene 7A reuses Scene 1A camera + set; 7A/7B play at the **end**, after D8.
- [ ] All colours == `tokens.py`; amber `#E0A100` everywhere (Act 1 and Act 2).
- [ ] 4-state legend visible throughout Act 2.
- [ ] One LUT + grain applied globally across both acts, Blender **and** Manim.
- [ ] Every Act 2 technical claim is spoken while its visual is on screen.
- [ ] Honest gaps shown, not hidden: ECE 0.42 (D5), interim 64³ checkpoint (D4).
- [ ] 3 SFX present on the right frames; both VO tracks synced.
- [ ] No illustration stands in for a measured result.
- [ ] Hero reveals (2C amber, 5B detour) each hold ≥3 s; D3 rays and D4 completion given the most Act 2 time.
- [ ] Closing thesis card held long enough to read twice.

---

### Note on effort vs. payoff

Act 1 is done. The new work is **Act 2**, most of it Manim (Phase 3B) — fast, low-risk diagrams/plots/tables where the extra ~4.5 min of runtime mostly comes from. The Blender deep dives (Phase 3C) reuse the voxel/mesh loaders and camera rigs from Act 1. Protect **D3 (fusion rays)** and **D4 (completion)** — they visualize the actual contribution. Across both acts, the four shots worth the most polish: 2C amber, 5B detour, D3 fusion rays, D4 completion.
