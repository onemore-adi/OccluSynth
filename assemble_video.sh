#!/usr/bin/env bash
# assemble_video.sh — stitch OccluSynth demo clips into a single video.
#
# Usage:
#   ./assemble_video.sh                   # full cut (~3:10)
#   MODE=short ./assemble_video.sh        # ~90s cut (drops shots 3 and 6)
#
# Requires: clips/ directory with shot*.mp4 files, ffmpeg, video_assets/ images.
# narration.m4a is optional — omit to get a silent video.
#
# Missing clips become a labelled grey slate; the build always completes.
set -euo pipefail

OUT=OccluSynth_demo.mp4
MODE=${MODE:-full}
BG=0E1116
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Shot manifest: (number name duration_secs)
declare -a SHOTS_FULL=(
  "1  shot01_problem      24"
  "2  shot02_thirdstate   15"
  "3  shot03_perception   23"
  "4  shot04_fusion       24"
  "5  shot05_completer    33"
  "6  shot06_uncertainty  15"
  "7  shot07_planner      28"
  "8  shot08_results      14"
)
declare -a SHOTS_SHORT=(
  "1  shot01_problem      24"
  "2  shot02_thirdstate   15"
  "4  shot04_fusion       24"
  "5  shot05_completer    33"
  "7  shot07_planner      28"
  "8  shot08_results      14"
)

[[ "$MODE" == "short" ]] && SHOTS=("${SHOTS_SHORT[@]}") || SHOTS=("${SHOTS_FULL[@]}")

TITLE_SECS=5
CLOSE_SECS=6
NORM="scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x${BG},setsar=1,fps=30,format=yuv420p"

# -- helpers ------------------------------------------------------------------
norm_clip() {          # <in> <out>
    ffmpeg -y -loglevel error -i "$1" \
      -vf "${NORM},format=yuv420p" \
      -c:v libx264 -crf 18 -preset medium -an "$2"
}

make_slate() {         # <label> <secs> <out>
    local label="$1" secs="$2" out="$3"
    local tmp_png
    tmp_png=$(mktemp /tmp/slate_XXXXXX.png)
    .venv312/bin/python3 - "$label" "$tmp_png" <<'PYEOF'
import sys
from PIL import Image, ImageDraw, ImageFont
label, path = sys.argv[1], sys.argv[2]
img = Image.new("RGB", (1920, 1080), (60, 60, 68))
d = ImageDraw.Draw(img)
try:
    fnt = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
except Exception:
    fnt = ImageFont.load_default()
txt = f"MISSING  {label}"
bbox = d.textbbox((0, 0), txt, font=fnt)
x = (1920 - (bbox[2]-bbox[0])) // 2
y = (1080 - (bbox[3]-bbox[1])) // 2
d.text((x, y), txt, font=fnt, fill=(200, 200, 200))
img.save(path)
PYEOF
    ffmpeg -y -loglevel error -loop 1 -framerate 30 -t "$secs" -i "$tmp_png" \
      -vf "format=yuv420p" -c:v libx264 -crf 18 -preset medium "$out"
    rm -f "$tmp_png"
}

overlay_lt() {         # <clip_in> <lt_png> <out>
    ffmpeg -y -loglevel error -i "$1" -i "$2" \
      -filter_complex "[0:v][1:v]overlay=0:H-h[v]" \
      -map "[v]" -c:v libx264 -crf 18 -preset medium "$3"
}

still_clip() {         # <png> <secs> <out>
    ffmpeg -y -loglevel error -loop 1 -framerate 30 -t "$2" -i "$1" \
      -vf "${NORM},format=yuv420p" \
      -c:v libx264 -crf 18 -preset medium "$3"
}

# -- title card ---------------------------------------------------------------
echo "Building title card..."
still_clip video_assets/title.png $TITLE_SECS "$TMP/seg_00_title.mp4"

# -- content shots ------------------------------------------------------------
SEG=1
for entry in "${SHOTS[@]}"; do
    read -r n name secs <<< "$entry"
    src="clips/${name}.mp4"
    lt="video_assets/lt${n}.png"
    seg_raw="$TMP/seg_${SEG}_raw.mp4"
    seg_out="$TMP/seg_${SEG}_${name}.mp4"

    if [[ -f "$src" ]]; then
        echo "Processing shot ${n}: ${name}..."
        norm_clip "$src" "$seg_raw"
    else
        echo "  !! $src missing — inserting slate"
        make_slate "$name" "$secs" "$seg_raw"
    fi

    if [[ -f "$lt" ]]; then
        overlay_lt "$seg_raw" "$lt" "$seg_out"
    else
        cp "$seg_raw" "$seg_out"
    fi
    SEG=$((SEG + 1))
done

# -- close card ---------------------------------------------------------------
echo "Building close card..."
CLOSE_SEG="$TMP/seg_${SEG}_close.mp4"
still_clip video_assets/close.png $CLOSE_SECS "$CLOSE_SEG"

# -- concat list --------------------------------------------------------------
LIST="$TMP/concat.txt"
for f in "$TMP"/seg_*.mp4; do
    echo "file '$f'" >> "$LIST"
done
# sort by filename so segments stay in order
sort "$LIST" -o "$LIST"

echo "Concatenating segments..."
ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" \
  -c:v libx264 -crf 18 -preset medium "$TMP/video_only.mp4"

# -- narration ----------------------------------------------------------------
if [[ -f narration.m4a ]]; then
    echo "Mixing narration..."
    ffmpeg -y -loglevel error \
      -i "$TMP/video_only.mp4" -i narration.m4a \
      -c:v copy -c:a aac -b:a 192k \
      -map 0:v:0 -map 1:a:0 \
      -shortest "$OUT"
else
    echo "(no narration.m4a found — outputting silent video)"
    cp "$TMP/video_only.mp4" "$OUT"
fi

echo ""
echo "Done → $OUT"
