#!/usr/bin/env bash
#
# build.sh — final assembly, grade, grain and audio mux for the OccluSynth demo.
# Implements plan.md §6:
#   1. Concatenate the picture clips in script order with short crossfades.
#   2. Apply ONE LUT/grade (teal-navy shadows / warm highlights) derived from
#      config/tokens.py — no colour is hardcoded here.
#   3. Add subtle film grain + a global vignette.
#   4. Overlay the alpha text .mov overlays at their timecodes.
#   5. Mux audio/vo + audio/sfx; normalise to -14 LUFS.
#   6. Export build/OccluSynth_demo_final.mp4 (H.264, 1080p24).
#
# Robust to work-in-progress: any missing picture clip, overlay, or audio file is
# skipped with a warning so the film assembles from whatever exists so far.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDERS="$HERE/renders"
BUILD="$HERE/build"
AUDIO="$HERE/audio"
CONFIG="$HERE/config"
TMP="$BUILD/_tmp"
NORM="$TMP/norm"
mkdir -p "$NORM"

FFMPEG="$(command -v ffmpeg || echo /opt/homebrew/bin/ffmpeg)"
FFPROBE="$(command -v ffprobe || echo /opt/homebrew/bin/ffprobe)"
PY="$(command -v python3)"

XFADE="${XFADE:-0.4}"          # crossfade / dissolve duration (s)
FPS=24
W=1920; H=1080
FINAL="$BUILD/OccluSynth_demo_final.mp4"

# --- final two-act timeline (plan.md §6): Act1 → D0 → D1..D8 → 7A → 7B ---
BASE=(
  # Act 1 — cinematic sizzle (7A/7B moved to the very end)
  s1a_drivein s1c_reveal s2a_title s2c_amber_reveal s3a_depth s4a_growth \
  s4b_split s5a_collide s5b_detour s6a_dots s6b_numbers \
  # bridge + Act 2 deep dives
  d0_bridge d1_overview d2_perception d3_fusion d4_completion_arch d4_volumes \
  d5_uncertainty d6_planner d7_validation d8_close \
  # closing callback + thesis card
  s7a_callback s7b_end)

# alpha overlays:  "mov_id : base_clip_id : offset_seconds_into_that_clip"
OVERLAYS=("s1d_text:s1c_reveal:0.4" "s5b_text:s5b_detour:0.6")

# sfx cues (plan.md §5): "sfx_file : base_clip_id : offset"
SFX=("sub_bass:s1c_reveal:1.0" "pad:s2c_amber_reveal:0.5" "click:s5b_detour:3.0")

# Freeze-pad targets (s): short Blender Act-2 clips (which end on a full frame) are
# held to their VO-paced length so the two-act cut lands 6:00-6:15. Manim Act-2
# segments already carry their hold internally (OCCLU_HOLD), so they are not padded.
# (case functions, not assoc arrays — macOS ships bash 3.2)
pad_target() { case "$1" in
  d3_fusion) echo 48 ;; d4_volumes) echo 18 ;; d5_uncertainty) echo 30 ;;
  s7a_callback) echo 10 ;; *) echo "" ;; esac; }

# A longer, deliberate dissolve on the Act 1 -> D0 handoff (tonal shift).
xfade_into() { case "$1" in d0_bridge) echo 1.0 ;; *) echo "$XFADE" ;; esac; }

echo "== OccluSynth build.sh =="
echo "   ffmpeg: $FFMPEG"

# --------------------------------------------------------------------------
# Grade parameters derived from config/tokens.py (NAVY shadows / WARMWHITE highs).
# --------------------------------------------------------------------------
eval "$("$PY" - "$CONFIG" <<'PY'
import sys, importlib.util, pathlib
cfg = pathlib.Path(sys.argv[1]) / "tokens.py"
spec = importlib.util.spec_from_file_location("tokens", cfg)
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
def rgb(h): h=h.lstrip('#'); return [int(h[i:i+2],16)/255 for i in (0,2,4)]
navy, warm = rgb(t.NAVY), rgb(t.WARMWHITE)
ks, kh = 0.35, 0.15
sh = [ (c-0.5)*2*ks for c in navy ]      # push shadows toward navy/teal
hi = [ (c-0.5)*2*kh for c in warm ]      # push highlights warm
print(f'CB_RS={sh[0]:.4f}; CB_GS={sh[1]:.4f}; CB_BS={sh[2]:.4f}')
print(f'CB_RH={hi[0]:.4f}; CB_GH={hi[1]:.4f}; CB_BH={hi[2]:.4f}')
print(f'AMBER="{t.OCCLUDED}"; VOIDHEX="{t.VOID}"')
PY
)"
VOIDFF="0x${VOIDHEX#\#}"
GRADE="colorbalance=rs=${CB_RS}:gs=${CB_GS}:bs=${CB_BS}:rh=${CB_RH}:gh=${CB_GH}:bh=${CB_BH}"
GRAIN="noise=alls=6:allf=t+u"
VIGNETTE="vignette=PI/5"
echo "   grade (from tokens): $GRADE"
echo "   hero amber (tokens): $AMBER"

# --------------------------------------------------------------------------
# 1. Normalise every existing base clip to a canonical stream (1080p24, yuv420p,
#    silent audio). Alpha .mov title cards are flattened over VOID.
# --------------------------------------------------------------------------
PRESENT=(); DURS=()
for id in "${BASE[@]}"; do
  src=""
  [ -f "$RENDERS/$id.mp4" ] && src="$RENDERS/$id.mp4"
  [ -z "$src" ] && [ -f "$RENDERS/$id.mov" ] && src="$RENDERS/$id.mov"
  if [ -z "$src" ]; then echo "   [skip] missing base clip: $id"; continue; fi
  out="$NORM/$id.mp4"
  if [[ "$src" == *.mov ]]; then
    # flatten alpha title over VOID background (all inputs before filter/maps)
    "$FFMPEG" -y -v error \
      -f lavfi -i "color=c=${VOIDFF}:s=${W}x${H}:r=${FPS}" \
      -i "$src" \
      -f lavfi -i anullsrc=r=48000:cl=stereo \
      -filter_complex \
      "[1:v]format=rgba[fg];[0:v][fg]overlay=shortest=1,format=yuv420p[v]" \
      -map "[v]" -map 2:a -shortest \
      -r $FPS -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac "$out"
  else
    "$FFMPEG" -y -v error -i "$src" \
      -f lavfi -i anullsrc=r=48000:cl=stereo \
      -map 0:v -map 1:a -shortest \
      -vf "scale=${W}:${H},fps=${FPS},format=yuv420p" \
      -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac "$out"
  fi
  d="$("$FFPROBE" -v error -select_streams v:0 -show_entries format=duration \
        -of csv=p=0 "$out")"
  # Freeze-pad short Act-2 Blender clips (which end on a full frame) to their
  # VO-paced target by cloning the last frame.
  tgt="$(pad_target "$id")"
  if [ -n "$tgt" ] && "$PY" -c "import sys; sys.exit(0 if float('$d')<$tgt else 1)"; then
    padded="$NORM/${id}_pad.mp4"
    "$FFMPEG" -y -v error -i "$out" \
      -vf "tpad=stop_mode=clone:stop_duration=$("$PY" -c "print(max($tgt-float('$d'),0))")" \
      -af "apad" -t "$tgt" \
      -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac "$padded"
    mv "$padded" "$out"
    d="$("$FFPROBE" -v error -show_entries format=duration -of csv=p=0 "$out")"
    echo "   [pad] $id held to ${d}s"
  fi
  PRESENT+=("$id"); DURS+=("$d")
done

if [ "${#PRESENT[@]}" -eq 0 ]; then
  echo "!! no picture clips found in $RENDERS — nothing to assemble."; exit 1
fi
echo "   picture clips present: ${#PRESENT[@]} / ${#BASE[@]}"

# --------------------------------------------------------------------------
# 2. Build the xfade crossfade chain + compute each clip's timeline start.
#    Emits: FILTER (filter_complex), TOTAL (duration), START_<id> per clip.
# --------------------------------------------------------------------------
# per-transition xfade INTO each clip (default XFADE; longer on Act1->D0 handoff)
XF_INTO=()
for id in "${PRESENT[@]}"; do XF_INTO+=("$(xfade_into "$id")"); done

eval "$("$PY" - "$XFADE" "${PRESENT[@]}" -- "${DURS[@]}" -- "${XF_INTO[@]}" <<'PY'
import sys
xf = float(sys.argv[1])
args = sys.argv[2:]
sep = args.index("--")
sep2 = args.index("--", sep + 1)
ids = args[:sep]
durs = [float(x) for x in args[sep+1:sep2]]
xf_into = [float(x) for x in args[sep2+1:]]
n = len(ids)
# timeline start of each clip in an xfaded chain
starts = [0.0]*n
acc = 0.0
for i in range(1, n):
    acc += durs[i-1] - xf
    starts[i] = acc
total = (sum(durs) - xf*(n-1)) if n > 1 else durs[0]

# Per-transition xfade, clamped so a short clip cannot collapse the timeline.
starts = [0.0]*n
offsets = [0.0]*n
xfs = [0.0]*n
chain = durs[0] if n else 0.0
for i in range(1, n):
    xfi = min(xf_into[i], 0.4*min(durs[i-1], durs[i]))
    xfs[i] = xfi
    offsets[i] = chain - xfi
    starts[i] = chain - xfi
    chain = chain + durs[i] - xfi
total = chain

if n == 1:
    print('FILTER="null"')
else:
    parts = []
    prev = "[0:v]"
    for i in range(1, n):
        lbl = f"[vx{i}]" if i < n-1 else "[vout]"
        parts.append(f"{prev}[{i}:v]xfade=transition=dissolve:"
                     f"duration={xfs[i]:.4f}:offset={offsets[i]:.4f}{lbl}")
        prev = lbl
    print('FILTER="' + ";".join(parts) + '"')
print(f'TOTAL={total:.4f}')
for i, cid in enumerate(ids):
    print(f'START_{cid}={starts[i]:.4f}')
PY
)"
echo "   assembled runtime (pre-audio): ${TOTAL}s over ${#PRESENT[@]} clips"

# Assemble the crossfaded picture.
INPUTS=(); for id in "${PRESENT[@]}"; do INPUTS+=(-i "$NORM/$id.mp4"); done
if [ "${#PRESENT[@]}" -eq 1 ]; then
  cp "$NORM/${PRESENT[0]}.mp4" "$TMP/concat.mp4"
else
  "$FFMPEG" -y -v error "${INPUTS[@]}" \
    -filter_complex "${FILTER}" -map "[vout]" \
    -r $FPS -c:v libx264 -crf 16 -pix_fmt yuv420p "$TMP/concat.mp4"
fi

# --------------------------------------------------------------------------
# 3. ONE grade + grain + vignette across the whole timeline (Blender + Manim).
# --------------------------------------------------------------------------
"$FFMPEG" -y -v error -i "$TMP/concat.mp4" \
  -vf "${GRADE},${GRAIN},${VIGNETTE},format=yuv420p" \
  -c:v libx264 -crf 16 -pix_fmt yuv420p "$TMP/graded.mp4"

# --------------------------------------------------------------------------
# 4. Overlay the alpha text .mov's at their timecodes.
# --------------------------------------------------------------------------
CUR="$TMP/graded.mp4"
oi=0
for ov in "${OVERLAYS[@]}"; do
  IFS=: read -r movid clipid offset <<<"$ov"
  mov="$RENDERS/$movid.mov"
  startvar="START_${clipid}"
  base_start="${!startvar:-}"
  if [ ! -f "$mov" ] || [ -z "$base_start" ]; then
    echo "   [skip] overlay $movid (missing mov or base clip $clipid)"; continue
  fi
  at="$("$PY" -c "print(f'{${base_start}+${offset}:.3f}')")"
  nxt="$TMP/ov_$oi.mp4"
  "$FFMPEG" -y -v error -i "$CUR" -i "$mov" \
    -filter_complex "[1:v]format=rgba,setpts=PTS-STARTPTS+${at}/TB[o];[0:v][o]overlay=eof_action=pass,format=yuv420p[v]" \
    -map "[v]" -r $FPS -c:v libx264 -crf 16 -pix_fmt yuv420p "$nxt"
  CUR="$nxt"; oi=$((oi+1))
  echo "   overlaid $movid at ${at}s"
done
cp "$CUR" "$TMP/video.mp4"

# --------------------------------------------------------------------------
# 5. Audio: concat VO in scene order + SFX at cue timecodes, then loudnorm -14.
# --------------------------------------------------------------------------
HAVE_AUDIO=0
AMIX_INPUTS=(); AMIX_FILTERS=(); ai=0

# VO: concatenate any wavs in audio/vo (filename order) across the timeline.
if compgen -G "$AUDIO/vo/*.wav" >/dev/null 2>&1; then
  vo_list="$TMP/vo.txt"; : > "$vo_list"
  for w in "$AUDIO"/vo/*.wav; do echo "file '$w'" >> "$vo_list"; done
  "$FFMPEG" -y -v error -f concat -safe 0 -i "$vo_list" -ar 48000 -ac 2 "$TMP/vo.wav"
  AMIX_INPUTS+=(-i "$TMP/vo.wav"); AMIX_FILTERS+=("[$ai:a]aresample=48000[a$ai]"); ai=$((ai+1))
  HAVE_AUDIO=1
fi

# SFX at their cue timecodes.
for cue in "${SFX[@]}"; do
  IFS=: read -r name clipid offset <<<"$cue"
  f="$AUDIO/sfx/$name.wav"; startvar="START_${clipid}"; bs="${!startvar:-}"
  [ -f "$f" ] || continue; [ -z "$bs" ] && continue
  at_ms="$("$PY" -c "print(int((${bs}+${offset})*1000))")"
  AMIX_INPUTS+=(-i "$f")
  AMIX_FILTERS+=("[$ai:a]adelay=${at_ms}|${at_ms},aresample=48000[a$ai]")
  ai=$((ai+1)); HAVE_AUDIO=1
done

if [ "$HAVE_AUDIO" -eq 1 ]; then
  mixspec=""; for k in $(seq 0 $((ai-1))); do mixspec+="[a$k]"; done
  "$FFMPEG" -y -v error "${AMIX_INPUTS[@]}" \
    -filter_complex "$(IFS=';'; echo "${AMIX_FILTERS[*]}");${mixspec}amix=inputs=${ai}:duration=longest:normalize=0[mix];[mix]loudnorm=I=-14:TP=-1.5:LRA=11[a]" \
    -map "[a]" -t "$TOTAL" -ar 48000 -ac 2 "$TMP/audio.wav"
else
  echo "   [warn] no audio/vo or audio/sfx found — writing a silent track (Phase 4 pending)."
  "$FFMPEG" -y -v error -f lavfi -t "$TOTAL" -i anullsrc=r=48000:cl=stereo "$TMP/audio.wav"
fi

# --------------------------------------------------------------------------
# 6. Final mux -> build/OccluSynth_demo_final.mp4 (H.264 1080p24).
# --------------------------------------------------------------------------
"$FFMPEG" -y -v error -i "$TMP/video.mp4" -i "$TMP/audio.wav" \
  -map 0:v -map 1:a -shortest \
  -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -r $FPS \
  -c:a aac -b:a 192k -movflags +faststart "$FINAL"

dur="$("$FFPROBE" -v error -show_entries format=duration -of csv=p=0 "$FINAL")"
"$PY" -c "d=float('$dur'); print(f'\n== done ==\n  $FINAL\n  runtime: {d:.1f}s ({int(d//60)}m{int(d%60):02d}s)')"
