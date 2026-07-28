#!/usr/bin/env python3
"""
make_comparison_video3.py — three-way comparison clip:
  conventional (observation-only)  vs.  OccluSynth  vs.  ScanNet ground truth.

Same honest framing as the two-panel version (make_comparison_video.py, kept):
the LEFT panel is observation-only TSDF fusion — what any observation-only system
produces in occluded regions: nothing. The MIDDLE panel is OccluSynth on the
IDENTICAL input frames. The RIGHT panel is the ScanNet ground-truth mesh, which
no method sees at inference — it is the answer key, shown so judges can check the
completion against what is actually in the room.

All three turntables are frame-aligned (same camera, same orbit — see
render_gt_turntable.py for how the GT camera is locked to the same bounding box).

Inputs:
    clips/shot09_before_mesh.mp4   (grey, measured only)
    clips/shot09_after_mesh.mp4    (grey measured + amber completed)
    clips/shot09_gt_mesh.mp4       (neutral grey, ScanNet ground truth)

Outputs:
    clips/_assets/comparison3_title.png
    clips/_assets/comparison3_overlay.png
    clips/comparison_sota3.mp4

    .venv312/bin/python make_comparison_video3.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
BEFORE = ROOT / "clips" / "shot09_before_mesh.mp4"
AFTER = ROOT / "clips" / "shot09_after_mesh.mp4"
GT = ROOT / "clips" / "shot09_gt_mesh.mp4"
ASSETS = ROOT / "clips" / "_assets"
OUT = ROOT / "clips" / "comparison_sota3.mp4"

W, H = 1920, 1080

BG = (14, 17, 22)
INK = (244, 241, 234)
MUTE = (139, 149, 165)
FAINT = (95, 104, 118)
AMBER = (224, 161, 0)
GREYM = (201, 205, 211)
LINE = (52, 58, 70)
PANEL = (22, 27, 34)

# three panels, 16:9, evenly spaced across the frame
PW, PH = 610, 343
PX = [24, 655, 1286]
PY = 300
PCX = [x + PW // 2 for x in PX]

HN = "/System/Library/Fonts/HelveticaNeue.ttc"


def font(size, idx=0):
    return ImageFont.truetype(HN, size, index=idx)   # 0 reg,1 bold,7 light,10 med


def tracked(draw, text, cx, y, f, fill, track=0.0):
    widths = [draw.textlength(c, font=f) for c in text]
    total = sum(widths) + track * (len(text) - 1)
    x = cx - total / 2
    for c, w in zip(text, widths):
        draw.text((x, y), c, font=f, fill=fill, anchor="lm")
        x += w + track


def make_title():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([W // 2 - 40, 372, W // 2 + 40, 376], fill=AMBER)
    tracked(d, "OCCLUSYNTH", W // 2, 300, font(78, 1), INK, track=8)
    tracked(d, "OCCLUSION-AWARE  3D  RECONSTRUCTION", W // 2, 420,
            font(30, 10), MUTE, track=6)
    d.text((W // 2, 500), "Reconstructing the geometry cameras can't see",
           font=font(26, 7), fill=FAINT, anchor="mm")
    img.save(ASSETS / "comparison3_title.png")


def make_overlay():
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.text((W // 2, 66), "Reconstructing What the Cameras Never Saw",
           font=font(46, 1), fill=INK, anchor="mm")
    d.text((W // 2, 122),
           "Both reconstructions receive the identical 40 camera frames. "
           "Neither ever sees the ground truth.",
           font=font(25, 7), fill=MUTE, anchor="mm")

    for x in PX:
        d.rounded_rectangle([x - 1, PY - 1, x + PW + 1, PY + PH + 1],
                            radius=7, outline=LINE, width=2)

    heads = [("CONVENTIONAL RECONSTRUCTION", "observation-only  ·  TSDF fusion", INK, 2),
             ("OCCLUSYNTH", "occlusion-aware completion  ·  ours", AMBER, 3),
             ("GROUND TRUTH", "ScanNet scene0000_00  ·  the answer key", GREYM, 3)]
    for cx, (name, sub, col, tr) in zip(PCX, heads):
        tracked(d, name, cx, 236, font(24, 1), col, track=tr)
        d.text((cx, 270), sub, font=font(18, 0), fill=MUTE, anchor="mm")

    # legend
    ly = 700
    items = [(GREYM, "measured surface"), (AMBER, "predicted hidden geometry")]
    seg = [(c, t, font(21, 0).getlength(t)) for c, t in items]
    total = sum(22 + 12 + w + 60 for *_, w in seg) - 60
    x = W // 2 - total / 2
    for c, t, w in seg:
        d.rounded_rectangle([x, ly - 10, x + 22, ly + 12], radius=4, fill=c)
        d.text((x + 34, ly + 1), t, font=font(21, 0), fill=INK, anchor="lm")
        x += 22 + 12 + w + 60

    # metric badge
    bx0, bx1, by0, by1 = 610, 1310, 754, 876
    d.rounded_rectangle([bx0, by0, bx1, by1], radius=14, fill=PANEL, outline=LINE,
                        width=2)
    tracked(d, "HIDDEN GEOMETRY RECOVERED", (bx0 + bx1) // 2, by0 + 30,
            font(20, 1), MUTE, track=4)
    cxl, cxr = bx0 + 200, bx1 - 200
    d.text((cxl, by0 + 76), "0%", font=font(58, 1), fill=FAINT, anchor="mm")
    d.text((cxl, by1 - 16), "conventional", font=font(17, 0), fill=MUTE, anchor="mm")
    axc, ay = (bx0 + bx1) // 2, by0 + 72
    d.line([(axc - 26, ay), (axc + 20, ay)], fill=MUTE, width=4)
    d.polygon([(axc + 16, ay - 10), (axc + 34, ay), (axc + 16, ay + 10)], fill=MUTE)
    d.text((cxr, by0 + 76), "57.6%", font=font(58, 1), fill=AMBER, anchor="mm")
    d.text((cxr, by1 - 16), "OccluSynth", font=font(17, 0), fill=AMBER, anchor="mm")

    d.text((W // 2, 940),
           "Occluded-region surface recall @5 cm  ·  aggregate over 10 held-out "
           "ScanNet scenes  ·  conventional = 0% by construction",
           font=font(16, 0), fill=FAINT, anchor="mm")
    img.save(ASSETS / "comparison3_overlay.png")


def assemble():
    title = ASSETS / "comparison3_title.png"
    ov = ASSETS / "comparison3_overlay.png"
    core = ASSETS / "_core3.mp4"
    dur, xf, td = 12, 0.6, 3.0

    core_fc = (
        f"color=c=0x0E1116:s={W}x{H}:d={dur},fps=30[bg];"
        f"[0:v]scale={PW}:{PH}[L];[1:v]scale={PW}:{PH}[M];[2:v]scale={PW}:{PH}[R];"
        f"[bg][L]overlay={PX[0]}:{PY}[b1];"
        f"[b1][M]overlay={PX[1]}:{PY}[b2];"
        f"[b2][R]overlay={PX[2]}:{PY}[b3];"
        f"[3:v]format=rgba[ov];[b3][ov]overlay=0:0,format=yuv420p[out]"
    )
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-t", str(dur), "-i", str(BEFORE),
        "-t", str(dur), "-i", str(AFTER),
        "-t", str(dur), "-i", str(GT),
        "-loop", "1", "-t", str(dur), "-i", str(ov),
        "-filter_complex", core_fc, "-map", "[out]", "-t", str(dur),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", str(core),
    ], check=True)

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
