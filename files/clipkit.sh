#!/usr/bin/env bash
# clipkit.sh — make demo clips from STILL images (charts, tables, cross-sections,
# planner PNGs). Output is always 1920x1080 / 30fps / h264, ready for assemble_video.sh.
#
#   ./clipkit.sh still     IMG SECS OUT
#   ./clipkit.sh seq       IMG_A SECS_A IMG_B SECS_B OUT     # show A, then B
#   ./clipkit.sh dissolve  IMG_A IMG_B SECS OUT              # A crossfades into B
#
# Examples:
#   ./clipkit.sh still docs/images/visibility_scene0000_00.png 15 clips/shot02_thirdstate.mp4
#   ./clipkit.sh dissolve planner_lam0.png planner_lam4.png 28 clips/shot07_planner.mp4
set -euo pipefail
BG=0E1116
NORM="scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x${BG},setsar=1,fps=30,format=yuv420p"

cmd="${1:-}"; shift || true
case "$cmd" in
  still)
    img="$1"; secs="$2"; out="$3"
    ffmpeg -y -loglevel error -loop 1 -framerate 30 -t "$secs" -i "$img" \
      -vf "$NORM" -c:v libx264 -crf 18 -preset medium "$out" ;;
  seq)
    a="$1"; sa="$2"; b="$3"; sb="$4"; out="$5"
    ffmpeg -y -loglevel error -loop 1 -framerate 30 -t "$sa" -i "$a" \
      -loop 1 -framerate 30 -t "$sb" -i "$b" \
      -filter_complex "[0:v]${NORM}[v0];[1:v]${NORM}[v1];[v0][v1]concat=n=2:v=1[v]" \
      -map "[v]" -c:v libx264 -crf 18 -preset medium "$out" ;;
  dissolve)
    a="$1"; b="$2"; secs="$3"; out="$4"
    half=$(awk "BEGIN{print $secs/2}")
    off=$(awk "BEGIN{print $secs/2 - 1.5}")
    ffmpeg -y -loglevel error -loop 1 -framerate 30 -t "$half" -i "$a" \
      -loop 1 -framerate 30 -t "$half" -i "$b" \
      -filter_complex "[0:v]${NORM}[v0];[1:v]${NORM}[v1];[v0][v1]xfade=transition=fade:duration=1.5:offset=${off},format=yuv420p[v]" \
      -map "[v]" -c:v libx264 -crf 18 -preset medium "$out" ;;
  *) echo "usage: ./clipkit.sh {still|seq|dissolve} ..."; exit 1 ;;
esac
echo "wrote $out"
