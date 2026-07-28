#!/usr/bin/env bash
# Conform every clip of the 5-min hackathon cut to 1920x1080 @ 24fps CFR yuv420p
# intermediates in renders/hb5/conformed/, with trims, freeze-pads and
# lower-third overlays applied. Run AFTER render_hb.sh.
set -eo pipefail
cd "$(dirname "$0")"
R=renders            # previous-video renders
H=renders/hb5        # new hb5 cards + LT pngs
C=../clips           # june/july clip library (repo root)
O=renders/hb5/conformed
mkdir -p "$O"

NORM="scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0B0E14,setsar=1,fps=24,format=yuv420p"
ENC="-c:v libx264 -crf 16 -preset medium -an"

conform() {  # <out> <in> [trim_start] [trim_dur]
  local out=$1 in=$2 ss=${3:-} d=${4:-}
  local pre=()
  [[ -n "$ss" ]] && pre+=(-ss "$ss")
  [[ -n "$d"  ]] && pre+=(-t "$d")
  ffmpeg -y -loglevel error "${pre[@]}" -i "$in" -vf "$NORM" $ENC "$O/$out.mp4"
  echo "conformed $out"
}

# <out> <in> <lt_png> [trim_start] [trim_dur] [pad_secs]
conform_lt() {
  local out=$1 in=$2 lt=$3 ss=${4:-} d=${5:-} pad=${6:-0}
  local pre=()
  [[ -n "$ss" ]] && pre+=(-ss "$ss")
  [[ -n "$d"  ]] && pre+=(-t "$d")
  # duration after trim+pad, for fade-out timing
  local dur
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$in")
  [[ -n "$d" ]] && dur=$d
  dur=$(python3 -c "print(float('$dur')+float('$pad'))")
  local fo
  fo=$(python3 -c "print(max(0.0, float('$dur')-1.5))")
  ffmpeg -y -loglevel error "${pre[@]}" -i "$in" -loop 1 -i "$H/$lt.png" -filter_complex \
    "[0:v]${NORM},tpad=stop_mode=clone:stop_duration=${pad}[base];\
     [1:v]format=rgba,fade=in:st=0.7:d=0.7:alpha=1,fade=out:st=${fo}:d=0.8:alpha=1[lt];\
     [base][lt]overlay=0:0:shortest=1,format=yuv420p[v]" \
    -map "[v]" -t "$dur" $ENC "$O/$out.mp4"
  echo "conformed $out (+$lt)"
}

# ---------------- ACT 1 ----------------
conform    c01_title     "$H/HBTitle.mp4"
conform    c02_drivein   "$R/s1a_drivein.mp4"
# s1c + s1d alpha text overlay (starts 0.7s in)
ffmpeg -y -loglevel error -i "$R/s1c_reveal.mp4" -itsoffset 0.7 -i "$R/s1d_text.mov" -filter_complex \
  "[0:v]${NORM}[base];[base][1:v]overlay=0:0,format=yuv420p[v]" -map "[v]" $ENC "$O/c03_reveal.mp4"
echo "conformed c03_reveal (+s1d)"
conform    c04_blind     "$H/HBBlind.mp4"
# NOTE: clips/shot10_closeup_raw.mp4 is misnamed — it shows the COMPLETED (amber)
# mesh from ~2s on. Use the genuinely raw TSDF-only turntable for the "holes" beat.
conform_lt c05_holes     "$C/shot09_before_mesh.mp4"    LT_Holes   4 8
conform    c06_zero      "$H/HBZero.mp4"
conform_lt c07_amber     "$R/s2c_amber_reveal.mp4"      LT_Amber
conform    c08_thesis    "$H/HBThesis.mp4"

# ---------------- ACT 2 ----------------
conform    c09_pipeline  "$H/HBPipeline.mp4"
conform    c10_overview  "$R/d1_overview.mp4"           0 13
conform    c11_sec01     "$H/HBSec01.mp4"
conform_lt c12_filmstrip "$H/HBFilmstrip.mp4"           LT_Input
conform    c13_sec02     "$H/HBSec02.mp4"
conform    c14_depth     "$R/s3a_depth.mp4"
conform    c15_anchors   "$R/d2_perception.mp4"         0 16
conform    c16_sec03     "$H/HBSec03.mp4"
conform_lt c17_fusion    "$R/d3_fusion.mp4"             LT_Fusion  "" "" 1.5
conform    c18_states    "$H/HBStates.mp4"
conform    c19_sec04     "$H/HBSec04.mp4"
conform    c20_arch      "$R/d4_completion_arch.mp4"    0 15
conform_lt c21_growth    "$R/s4a_growth.mp4"            LT_Growth  "" "" 1.2
conform    c22_sec05     "$H/HBSec05.mp4"
conform_lt c23_before    "$C/shot09_before_mesh.mp4"    LT_Before  0 9
conform_lt c24_fade      "$C/shot09_completion_fade.mp4" LT_After
conform_lt c25_sofa      "$C/shot10_sofa_closeup.mp4"   LT_Sofa    1.5 13
conform    c26_compare   "$C/comparison_sota3.mp4"
conform    c27_sec06     "$H/HBSec06.mp4"
conform_lt c28_uncert    "$R/d5_uncertainty.mp4"        LT_Uncert  "" "" 1.0
conform    c29_planner   "$R/d6_planner.mp4"            0 16.5
conform_lt c30_collide   "$R/s5a_collide.mp4"           LT_Collide
conform_lt c31_detour    "$R/s5b_detour.mp4"            LT_Detour

# ---------------- ACT 3 ----------------
conform    c32_why       "$H/HBWhy.mp4"
conform_lt c33_dots      "$R/s6a_dots.mp4"              LT_Bench
conform    c34_numbers   "$R/s6b_numbers.mp4"
conform    c35_matters   "$H/HBMatters.mp4"
conform    c36_callback  "$R/s7a_callback.mp4"
conform    c37_end       "$H/HBEnd.mp4"

echo "--- all conformed ---"
for f in "$O"/*.mp4; do
  printf "%-18s %ss\n" "$(basename "$f" .mp4)" "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")"
done
