#!/usr/bin/env bash
# ============================================================================
# OccluSynth reproducibility-video assembler
# Everything here is a STILL (terminal screenshots + the two cards). Each is
# held for its slot, the step caption is burned in, narration is laid over,
# and the output is fixed to the summed timeline.
#
# Put your 6 terminal screenshots in SHOTS_DIR with these names (.png):
#   install  visibility  completer  geometry  safety  artifacts
# (Cmd+Shift+4 on macOS → drag a box around the relevant terminal output.)
# Cards come from ASSETS_DIR (repro_title.png, repro_close.png — provided).
# Record narration to NARRATION (read the teleprompter), then:
#   ./assemble_repro.sh        → OccluSynth_reproducibility.mp4
# ============================================================================
set -eo pipefail
SHOTS_DIR="${SHOTS_DIR:-./shots}"
ASSETS_DIR="${ASSETS_DIR:-./video_assets}"
NARRATION="${NARRATION:-./repro_narration.m4a}"
OUT="${OUT:-./OccluSynth_reproducibility.mp4}"
BG="0x0E1116"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

# name | source png | dur | caption | cap_in | cap_len
ROWS=(
  "title|@title|6|-|0|0"
  "install|install|20|rlt1_install|6|13"
  "visibility|visibility|18|rlt2_fusion|4|13"
  "completer|completer|22|rlt3_compl|7|14"
  "geometry|geometry|16|rlt4_geom|3|12"
  "safety|safety|16|rlt5_safety|3|12"
  "artifacts|artifacts|12|rlt6_json|2|9"
  "close|@close|8|-|0|0"
)

NORM="scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=${BG},setsar=1,fps=30,format=yuv420p"
src(){ case "$1" in
  @title) echo "$ASSETS_DIR/repro_title.png";;
  @close) echo "$ASSETS_DIR/repro_close.png";;
  *) for e in png PNG jpg jpeg; do [ -f "$SHOTS_DIR/$1.$e" ] && { echo "$SHOTS_DIR/$1.$e"; return; }; done; echo "__MISSING__:$1";; esac; }

echo ">> Normalizing stills..."
LIST="$WORK/list.txt"; : > "$LIST"; i=0
for r in "${ROWS[@]}"; do IFS='|' read -r name s dur cap cin clen <<< "$r"
  sp="$(src "$s")"; seg="$WORK/seg_$(printf '%02d' $i).mp4"
  if [[ "$sp" == __MISSING__:* ]]; then
    echo "!! missing screenshot '$name' — inserting placeholder slate"
    ffmpeg -y -loglevel error -f lavfi -i "color=c=${BG}:s=1920x1080:r=30:d=${dur}" \
      -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='[ missing: ${name} ]':fontcolor=0x9AA4B2:fontsize=48:x=(w-tw)/2:y=(h-th)/2" \
      -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p "$seg"
  else
    ffmpeg -y -loglevel error -loop 1 -framerate 30 -t "$dur" -i "$sp" \
      -vf "$NORM" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p "$seg"
  fi
  echo "file '$seg'" >> "$LIST"; i=$((i+1))
done

echo ">> Captions + audio..."
INPUTS=(-f concat -safe 0 -i "$LIST"); FILTER=""; LAST="0:v"; ci=1; t=0
for r in "${ROWS[@]}"; do IFS='|' read -r name s dur cap cin clen <<< "$r"
  if [ "$cap" != "-" ] && [ -f "$ASSETS_DIR/$cap.png" ] && [ "$clen" -gt 0 ]; then
    a=$((t+cin)); b=$((a+clen)); INPUTS+=(-i "$ASSETS_DIR/$cap.png")
    FILTER+="[${LAST}][${ci}:v]overlay=0:0:enable='between(t,${a},${b})'[v${ci}];"; LAST="v${ci}"; ci=$((ci+1))
  fi; t=$((t+dur)); done
FILTER="${FILTER%;}"; [ -z "$FILTER" ] && FILTER="[0:v]null[vout]" && LAST="vout"
TOTAL="$t"; AUDIO=()
[ -f "$NARRATION" ] && { INPUTS+=(-i "$NARRATION"); AUDIO=(-map "${ci}:a" -c:a aac -b:a 192k); echo "   narration: $NARRATION"; } \
                    || echo "!! no narration ($NARRATION) — silent video"

ffmpeg -y -loglevel error "${INPUTS[@]}" -filter_complex "$FILTER" \
  -map "[${LAST}]" "${AUDIO[@]}" -t "$TOTAL" \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -movflags +faststart "$OUT"
echo ">> Done: $OUT ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT")s)"