# OccluSynth — Shot Capture Guide (the dumbed-down version)

You don't need to be good at video. You're making **8 short clips**. The two title/closing
cards and all captions are already done for you. This guide tells you, for each clip, the exact
command to type. Then one script glues everything together.

**The key trick:** don't try to screen-record the spinning 3D grid by hand — it'll be jittery and
frustrating. Instead, render the spin straight from the `.ply` file with `turntable.py`. Smooth,
repeatable, no mouse wrangling.

---

## Step 1 — Put these files in your repo root

From the kit I gave you, copy into the OccluSynth repo root:

```
turntable.py
clipkit.sh
assemble_video.sh
video_assets/        (title.png, close.png, results.png, lt1…lt8 .png)
```

Then make folders and make the scripts runnable:

```bash
cd /Users/onemore_adi/OccluSynth
mkdir -p clips
chmod +x clipkit.sh assemble_video.sh
brew install ffmpeg      # if you don't already have it
```

There are only **three ways** you'll ever make a clip:

| Tool | Use it for | Output |
|---|---|---|
| `turntable.py` | the 3D grid / mesh spinning | a smooth orbit `.mp4` |
| `clipkit.sh still` | any single image (chart, table, cross-section, card) | image held as `.mp4` |
| `clipkit.sh dissolve` | two images that crossfade (the planner before/after) | `.mp4` |

Every clip must land in `clips/` with the **exact name** shown below.

---

## Step 2 — Find your source files

The 3D shots need `.ply` files your pipeline already produced. Locate them:

```bash
# the GT mesh (used for the opening shot)
ls data/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.ply

# the marching-cubes mesh
ls demo_outputs/tsdf_fusion/

# the visibility voxel grid PLY + cross-section PNG.
# If you're not sure of the exact filename, just search:
find . -iname "*visib*.ply" -o -iname "*voxel*.ply"
ls docs/images/visibility_*.png
```

If the voxel-grid `.ply` doesn't exist yet, regenerate it (it prints where it saved):

```bash
.venv312/bin/python scripts/run_visibility.py --scene scene0000_00 --use_gt_depth
```

Note the path it prints for the PLY — you'll feed that to `turntable.py`.

---

## Step 3 — Make the 8 clips, one command each

Run these from the repo root. **Use your `.venv312` python for `turntable.py`** (that's the env
with open3d). Replace `PATH/TO/...` with the real paths you found in Step 2.

### shot 1 — the problem (24 s) · slow orbit of the GT mesh
```bash
.venv312/bin/python turntable.py \
  --ply data/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.ply \
  --out clips/shot01_problem.mp4 --seconds 24
```

### shot 2 — the third state (15 s) · the green/red/amber cross-section (EASY, it's just an image)
```bash
./clipkit.sh still docs/images/visibility_scene0000_00.png 15 clips/shot02_thirdstate.mp4
```

### shot 3 — perception (23 s) · the depth triptych, then the scale-fit bar chart
First find the two PNGs your scripts saved (check `demo_outputs/` — `run_baseline.py` and
`run_scale_fit.py` print their save paths). Then show one after the other:
```bash
./clipkit.sh seq PATH/TO/baseline_triptych.png 11 PATH/TO/scalefit_barchart.png 12 clips/shot03_perception.mp4
```
*Too fiddly? Just hold the bar chart for the whole shot — the narration carries it:*
```bash
./clipkit.sh still PATH/TO/scalefit_barchart.png 23 clips/shot03_perception.mp4
```

### shot 4 — visibility fusion (24 s) · THE money shot: orbit the voxel grid
```bash
.venv312/bin/python turntable.py \
  --ply PATH/TO/visibility_scene0000_00.ply \
  --out clips/shot04_fusion.mp4 --seconds 24 --point_size 5
```

### shot 5 — the completer (33 s) · the geometry results table
Easiest source: take the results slide you built (Prompt 0) and screenshot just the table
(macOS: **Cmd+Shift+4**, drag a box → saves a PNG to your Desktop). Then:
```bash
./clipkit.sh still ~/Desktop/geometry_table.png 33 clips/shot05_completer.mp4
```

### shot 6 — uncertainty (15 s) · the MC-dropout heatmap *(skip this if you didn't make one)*
```bash
./clipkit.sh still PATH/TO/calibration_or_puncertainty.png 15 clips/shot06_uncertainty.mp4
```
If you have no uncertainty image, **don't make this clip** — use the 90 s cut (Step 4), which drops it.

### shot 7 — the planner (28 s) · naive path crossfading into the safe path
Make two stills first by running the planner at two risk levels and renaming each output:
```bash
.venv312/bin/python scripts/run_planner.py --scene scene0000_00 --lambda_risk 0
mv docs/images/planner_scene0000_00.png planner_lam0.png
.venv312/bin/python scripts/run_planner.py --scene scene0000_00 --lambda_risk 4
mv docs/images/planner_scene0000_00.png planner_lam4.png
```
Then crossfade them:
```bash
./clipkit.sh dissolve planner_lam0.png planner_lam4.png 28 clips/shot07_planner.mp4
```

### shot 8 — results (14 s) · the results card (already made for you)
```bash
./clipkit.sh still video_assets/results.png 14 clips/shot08_results.mp4
```

You do **not** make shot 0 (title) or shot 9 (close) — the assembler uses the cards automatically.

---

## Step 4 — Record narration, then assemble

1. Open `OccluSynth_narration_teleprompter.md`, hit record on your phone or Mac, and read it
   straight through at the marked pace. Save it as `narration.m4a` in the repo root.
2. Build the video:
```bash
./assemble_video.sh                          # full ~3:10 cut
# or, if you skipped shot 3 and/or shot 6:
MODE=short ./assemble_video.sh               # ~90 s cut (drops shots 3 and 6)
```
Any clip you didn't make becomes a labelled grey "missing" slate, so the build always finishes —
you can record that one shot later and re-run.

Output: `OccluSynth_demo.mp4`.

---

## Troubleshooting

**The turntable spins too fast / too slow.** Add `--rot_per_frame N`. Try `1.5` for slow, `5` for
fast. Re-run; it only takes a moment.

**The turntable window opens tiny (Retina Mac).** That's normal — it captures at full pixel
resolution anyway, so the clip is still sharp. Don't resize or cover the window while it renders.

**`turntable.py` says "open3d not found".** You ran it with the wrong python. Use
`.venv312/bin/python turntable.py ...` (the env where open3d works), not plain `python`.

**The voxel grid is too dark / dim.** Bump `--point_size 6` and confirm the PLY actually has the
green/red/amber colours (it should, straight from `run_visibility.py`).

**Prefer to film the live Rerun window instead?** (Fallback — less smooth.) Open the grid:
`rerun PATH/TO/file.rrd`, collapse the side panels for a clean frame, then **Cmd+Shift+5 →
Record Selected Portion**, drag a box around the 3D view, hit Record, and slowly **left-drag** to
orbit (scroll = zoom, right-drag = pan). Stop from the menu bar. Trim the ends in QuickTime
(Edit → Trim). Save it as the matching `clips/shotNN_*.mp4`. The assembler will clean up the size.

**A static image looks stretched.** It won't — `clipkit.sh` always fits the image inside 1080p and
fills the rest with the slate colour (letterbox/pillarbox), never stretches.

**Audio drifts out of sync.** The video length is fixed by the script, so just re-read the
narration to the teleprompter timecodes; a slightly long take gets trimmed, a short one leaves a
little silence. Neither shifts the visuals.

---

## The whole thing, in order

```
copy kit into repo  →  mkdir clips
shot1  turntable  (GT mesh)
shot2  clipkit still   (cross-section png)
shot3  clipkit seq     (triptych + bar chart)   [or just the bar chart]
shot4  turntable  (voxel grid)          ← money shot
shot5  clipkit still   (geometry table screenshot)
shot6  clipkit still   (uncertainty png)         [optional — skip + use MODE=short]
shot7  clipkit dissolve (planner λ=0 → λ=4)
shot8  clipkit still   (results.png — provided)
record narration.m4a   →   ./assemble_video.sh   →   OccluSynth_demo.mp4
```
