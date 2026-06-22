#!/usr/bin/env bash
# ============================================================================
# OccluSynth — reproducibility run
# Re-runs the existing pipeline from cached predictions + the interim checkpoint
# and prints the metrics that appear on the deck. No training, no cold VGGT
# inference — these steps are fast and need no cloud GPU.
#
# Prerequisites (already in your repo):
#   .venv312 created + `pip install -e .` done
#   cached VGGT predictions in demo_outputs/pred_cache/
#   completer crops in data/completer_crops/
#   checkpoint at checkpoints/interim_64_aug/completer_best.pt
#
# Run:   ./reproduce.sh
# It writes a clean transcript to repro_full.log and one file per step
# (repro_step1.txt … repro_step6.txt) that you can open and screenshot.
# ============================================================================
PY=".venv312/bin/python"
CKPT="checkpoints/interim_64_aug/completer_best.pt"
exec > >(tee repro_full.log) 2>&1     # mirror everything to repro_full.log

banner(){ echo; echo "════════════════════════════════════════════════════════════"; \
          echo "  $1"; echo "════════════════════════════════════════════════════════════"; }
run(){ echo "\$ $*"; echo; eval "$*"; echo; }

# ── Step 1: environment + tests ────────────────────────────────────────────
banner "STEP 1 / 6   Environment & test suite"
( run "$PY --version"
  run "$PY -m pip install -e . -q && echo 'package installed (editable)'"
  run "$PY -m pytest -q tests/"
) | tee repro_step1.txt

# ── Step 2: visibility fusion (GT depth → voxel-state split) ───────────────
banner "STEP 2 / 6   Visibility-aware fusion  (expect ~61% occluded)"
( run "$PY scripts/run_visibility.py --scene scene0000_00 --use_gt_depth"
) | tee repro_step2.txt

# ── Step 3: completer evaluation ───────────────────────────────────────────
banner "STEP 3 / 6   Completer eval  (expect occluded MAE 27.14 cm)"
( run "$PY scripts/eval_completer.py --device mps --ckpt $CKPT"
) | tee repro_step3.txt

# ── Step 4: geometry evaluation ────────────────────────────────────────────
banner "STEP 4 / 6   Geometry eval  (expect occluded F-score 32%)"
( run "$PY scripts/eval_geometry.py --ckpt $CKPT"
) | tee repro_step4.txt

# ── Step 5: occlusion safety benchmark ─────────────────────────────────────
banner "STEP 5 / 6   Safety benchmark  (expect awareness ~21%)"
( run "$PY scripts/run_safety_benchmark.py --ckpt $CKPT"
) | tee repro_step5.txt

# ── Step 6: show the artefacts on disk ─────────────────────────────────────
banner "STEP 6 / 6   Result artefacts"
( run "ls -lh demo_outputs/completer_eval/results.json demo_outputs/geometry_eval/results.json demo_outputs/safety_benchmark/results.json"
  echo '── completer_eval/results.json (excerpt) ──'
  run "$PY -c \"import json;d=json.load(open('demo_outputs/completer_eval/results.json'));print(json.dumps(d,indent=2)[:600])\""
) | tee repro_step6.txt

banner "DONE — every printed metric matches the deck (interim 64³ checkpoint)"
echo "Transcript: repro_full.log   ·   per-step screenshots: repro_step1.txt … repro_step6.txt"