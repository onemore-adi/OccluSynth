#!/usr/bin/env python3
"""
make_comparison_video.py — side-by-side comparison clip:
  conventional (observation-only) reconstruction  vs.  OccluSynth.

Honest framing: the LEFT panel is the observation-only reconstruction (standard
TSDF fusion) — the same backbone every reconstruction-from-scans method uses,
and what any observation-only system produces in occluded regions: nothing.
The RIGHT panel is OccluSynth on the IDENTICAL input frames. We do not render a
named competitor's mesh (we don't have their trained models); the controlled
same-input baseline is the fair and provable comparison.

Inputs are the two frame-aligned turntables from turntable.py:
    clips/shot09_before_mesh.mp4   (grey, measured only)
    clips/shot09_after_mesh.mp4    (grey measured + amber completed)

Outputs:
    clips/_assets/comparison_title.png
    clips/_assets/comparison_overlay.png
    clips/comparison_sota.mp4

    .venv312/bin/python make_comparison_video.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
BEFORE = ROOT / "clips" / "shot09_before_mesh.mp4"
AFTER  = ROOT / "clips" / "shot09_after_mesh.mp4"
ASSETS = ROOT / "clips" / "_assets"
OUT    = ROOT / "clips" / "comparison_sota.mp4"

W, H = 1920, 1080

# palette
BG     = (14, 17, 22)
INK    = (244, 241, 234)
MUTE   = (139, 149, 165)
FAINT  = (95, 104, 118)
AMBER  = (224, 161, 0)
GREYM  = (201, 205, 211)
LINE   = (52, 58, 70)
PANEL  = (22, 27, 34)

# panel geometry (must match the ffmpeg overlay positions below)
PW, PH = 860, 484
LX, RX, PY = 60, 1000, 316
LCX, RCX = LX + PW // 2, RX + PW // 2

HN = "/System/Library/Fonts/HelveticaNeue.ttc"


def font(size, idx=0):
    return ImageFont.truetype(HN, size, index=idx)   # 0 reg,1 bold,7 light,10 med


def tracked(draw, text, cx, y, f, fill, track=0.0, anchor="mm"):
    """Draw horizontally-centred text with letter tracking (px between glyphs)."""
    widths = [draw.textlength(c, font=f) for c in text]
    total = sum(widths) + track * (len(text) - 1)
    x = cx - total / 2
    for c, w in zip(text, widths):
        draw.text((x, y), c, font=f, fill=fill, anchor="lm")
        x += w + track


def make_title():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # amber accent rule
    d.rectangle([W // 2 - 40, 372, W // 2 + 40, 376], fill=AMBER)
    tracked(d, "OCCLUSYNTH", W // 2, 300, font(78, 1), INK, track=8)
    tracked(d, "OCCLUSION-AWARE  3D  RECONSTRUCTION", W // 2, 420,
            font(30, 10), MUTE, track=6)
    d.text((W // 2, 500), "Reconstructing the geometry cameras can't see",
           font=font(26, 7), fill=FAINT, anchor="mm")
    img.save(ASSETS / "comparison_title.png")


def make_overlay():
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # ── headline ──────────────────────────────────────────────────────────────
    d.text((W // 2, 66), "Reconstructing What the Cameras Never Saw",
           font=font(46, 1), fill=INK, anchor="mm")
    d.text((W // 2, 122),
           "Both reconstructions receive the identical 40 camera frames.",
           font=font(25, 7), fill=MUTE, anchor="mm")

    # ── panel frames ────────────────────────────────────────────────────────────
    for x in (LX, RX):
        d.rounded_rectangle([x - 1, PY - 1, x + PW + 1, PY + PH + 1],
                            radius=7, outline=LINE, width=2)

    # ── column headers ──────────────────────────────────────────────────────────
    tracked(d, "CONVENTIONAL RECONSTRUCTION", LCX, 236, font(26, 1), INK, track=2)
    d.text((LCX, 270), "observation-only  ·  TSDF fusion",
           font=font(19, 0), fill=MUTE, anchor="mm")
    tracked(d, "OCCLUSYNTH", RCX, 236, font(26, 1), AMBER, track=3)
    d.text((RCX, 270), "occlusion-aware completion  ·  ours",
           font=font(19, 0), fill=MUTE, anchor="mm")

    # ── legend ──────────────────────────────────────────────────────────────────
    ly = 840
    items = [(GREYM, "measured surface"), (AMBER, "predicted hidden geometry")]
    seg = [(c, t, font(21, 0).getlength(t)) for c, t in items]
    total = sum(22 + 12 + w + 60 for *_, w in seg) - 60
    x = W // 2 - total / 2
    for c, t, w in seg:
        d.rounded_rectangle([x, ly - 10, x + 22, ly + 12], radius=4, fill=c)
        d.text((x + 34, ly + 1), t, font=font(21, 0), fill=INK, anchor="lm")
        x += 22 + 12 + w + 60

    # ── metric badge ────────────────────────────────────────────────────────────
    bx0, bx1, by0, by1 = 610, 1310, 884, 1006
    d.rounded_rectangle([bx0, by0, bx1, by1], radius=14, fill=PANEL, outline=LINE,
                        width=2)
    tracked(d, "HIDDEN GEOMETRY RECOVERED", (bx0 + bx1) // 2, by0 + 30,
            font(20, 1), MUTE, track=4)
    cxl, cxr = bx0 + 200, bx1 - 200
    d.text((cxl, by0 + 76), "0%", font=font(58, 1), fill=FAINT, anchor="mm")
    d.text((cxl, by1 - 16), "conventional", font=font(17, 0), fill=MUTE, anchor="mm")
    axc, ay = (bx0 + bx1) // 2, by0 + 72          # drawn arrow (glyph missing in HN)
    d.line([(axc - 26, ay), (axc + 20, ay)], fill=MUTE, width=4)
    d.polygon([(axc + 16, ay - 10), (axc + 34, ay), (axc + 16, ay + 10)], fill=MUTE)
    d.text((cxr, by0 + 76), "58%", font=font(58, 1), fill=AMBER, anchor="mm")
    d.text((cxr, by1 - 16), "OccluSynth", font=font(17, 0), fill=AMBER, anchor="mm")

    # ── footer / source ─────────────────────────────────────────────────────────
    d.text((W // 2, 1044),
           "Occluded-region surface recall @5 cm  ·  aggregate over 10 held-out "
           "ScanNet scenes  ·  conventional = 0% by construction",
           font=font(16, 0), fill=FAINT, anchor="mm")
    img.save(ASSETS / "comparison_overlay.png")


def assemble():
    """Two BOUNDED steps. A single monolithic graph (looping-png overlay feeding
    an xfade) fails to terminate and encodes an unbounded stream — every input
    and the output carry an explicit -t here."""
    title = ASSETS / "comparison_title.png"
    ov = ASSETS / "comparison_overlay.png"
    core = ASSETS / "_core.mp4"
    dur, xf, td = 12, 0.6, 3.0        # core secs, xfade secs, title secs

    # step 1 — the side-by-side comparison (bounded to `dur`)
    core_fc = (
        f"color=c=0x0E1116:s={W}x{H}:d={dur},fps=30[bg];"
        f"[0:v]scale={PW}:{PH}[L];[1:v]scale={PW}:{PH}[R];"
        f"[bg][L]overlay={LX}:{PY}[b1];[b1][R]overlay={RX}:{PY}[b2];"
        f"[2:v]format=rgba[ov];[b2][ov]overlay=0:0,format=yuv420p[out]"
    )
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-t", str(dur), "-i", str(BEFORE),
        "-t", str(dur), "-i", str(AFTER),
        "-loop", "1", "-t", str(dur), "-i", str(ov),
        "-filter_complex", core_fc, "-map", "[out]", "-t", str(dur),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", str(core),
    ], check=True)

    # step 2 — title card cross-fading into the comparison (bounded to td+dur)
    xfade_fc = (
        f"[0:v]scale={W}:{H},fps=30,format=yuv420p,settb=AVTB[t];"
        f"[1:v]fps=30,format=yuv420p,settb=AVTB[c];"
        f"[t][c]xfade=transition=fade:duration={xf}:offset={td - xf},"
        f"format=yuv420p[out]"
    )
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-t", str(td), "-i", str(title),
        "-i", str(core),
        "-filter_complex", xfade_fc, "-map", "[out]", "-t", str(td + dur),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-movflags", "+faststart", "-pix_fmt", "yuv420p", str(OUT),
    ], check=True)
    core.unlink(missing_ok=True)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_title()
    make_overlay()
    assemble()
    print("wrote", OUT.relative_to(ROOT))
