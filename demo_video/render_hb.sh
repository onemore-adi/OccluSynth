#!/usr/bin/env bash
# Render all hb5 (Samsung-style 5-min cut) manim clips into renders/hb5/.
set -euo pipefail
cd "$(dirname "$0")"
M=./.venv-manim/bin/manim
OUT=renders/hb5
mkdir -p "$OUT"
rm -rf manim/media media

CARDS=(HBTitle HBBlind HBZero HBThesis HBPipeline HBSec01 HBSec02 HBSec03 HBSec04 HBSec05 HBSec06 HBStates HBWhy HBMatters HBEnd)
for s in "${CARDS[@]}"; do
  $M -q h --fps 24 --format mp4 --disable_caching -o "$s" manim/hb_cards.py "$s"
done
$M -q h --fps 24 --format mp4 --disable_caching -o HBFilmstrip manim/hb_filmstrip.py HBFilmstrip

LTS=(LT_Input LT_Holes LT_Amber LT_Fusion LT_Growth LT_Before LT_After LT_Sofa LT_Uncert LT_Collide LT_Detour LT_Bench)
for s in "${LTS[@]}"; do
  $M -q h -s -t -o "$s" manim/hb_lowerthirds.py "$s"
done

# collect outputs
find media -name "*.mp4" -exec cp {} "$OUT"/ \;
find media -path "*images*" -name "LT_*.png" -exec cp {} "$OUT"/ \;
ls -la "$OUT"
