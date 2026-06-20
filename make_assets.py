#!/usr/bin/env python3
"""Generate video_assets/ images: title.png, close.png, lt1-lt8.png."""
from PIL import Image, ImageDraw, ImageFont
import os, sys

OUT = "video_assets"
os.makedirs(OUT, exist_ok=True)

BG = (14, 17, 22)          # #0E1116
WHITE = (255, 255, 255)
DIM = (170, 178, 190)
ACCENT = (0, 210, 140)     # teal-green accent

W, H = 1920, 1080

def font(size):
    # Try system fonts; fall back to default
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

def make_title(path, title, subtitle, byline=""):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # accent bar
    d.rectangle([(W//2 - 200, H//2 - 110), (W//2 + 200, H//2 - 106)], fill=ACCENT)
    # title
    f_title = font(96)
    bbox = d.textbbox((0, 0), title, font=f_title)
    tw = bbox[2] - bbox[0]
    d.text(((W - tw) // 2, H//2 - 90), title, font=f_title, fill=WHITE)
    # subtitle
    f_sub = font(42)
    bbox = d.textbbox((0, 0), subtitle, font=f_sub)
    sw = bbox[2] - bbox[0]
    d.text(((W - sw) // 2, H//2 + 30), subtitle, font=f_sub, fill=DIM)
    # byline
    if byline:
        f_by = font(30)
        bbox = d.textbbox((0, 0), byline, font=f_by)
        bw = bbox[2] - bbox[0]
        d.text(((W - bw) // 2, H//2 + 90), byline, font=f_by, fill=(100, 110, 120))
    img.save(path)
    print("wrote", path)

def make_close(path):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([(W//2 - 180, H//2 - 110), (W//2 + 180, H//2 - 106)], fill=ACCENT)
    f = font(80)
    t = "OccluSynth"
    bbox = d.textbbox((0, 0), t, font=f)
    d.text(((W - (bbox[2]-bbox[0])) // 2, H//2 - 90), t, font=f, fill=WHITE)
    f2 = font(36)
    for i, line in enumerate([
        "Occlusion-Aware 3D Reconstruction",
        "github.com/onemore-adi/OccluSynth",
    ]):
        bbox = d.textbbox((0, 0), line, font=f2)
        d.text(((W - (bbox[2]-bbox[0])) // 2, H//2 + 30 + i*54), line, font=f2, fill=DIM)
    img.save(path)
    print("wrote", path)

# Lower-third design: 1920 x 130 RGBA, semi-transparent dark bar, title + subtitle
CAPTIONS = [
    ("THE PROBLEM",         "Standard reconstruction is blind to occlusion"),
    ("THE THIRD STATE",     "Visible  ·  Occluded  ·  Unknown"),
    ("DEPTH PERCEPTION",    "VGGT predictions with scale calibration"),
    ("VISIBILITY FUSION",   "OccluSynth voxel grid — the key contribution"),
    ("THE COMPLETER",       "Geometry-aware scene completion"),
    ("UNCERTAINTY",         "Monte Carlo dropout calibration"),
    ("THE PLANNER",         "Risk-aware path planning  ·  λ-risk 0 → 4"),
    ("RESULTS",             "OccluSynth outperforms baselines across scenes"),
]

def make_lt(path, title, subtitle):
    LW, LH = 1920, 130
    img = Image.new("RGBA", (LW, LH), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # semi-transparent bar
    bar = Image.new("RGBA", (LW, LH), (14, 17, 22, 200))
    img.alpha_composite(bar)
    d = ImageDraw.Draw(img)
    # left accent stripe
    d.rectangle([(0, 0), (5, LH)], fill=(*ACCENT, 255))
    # title text
    f_t = font(46)
    d.text((28, 14), title, font=f_t, fill=(*WHITE, 255))
    # subtitle text
    f_s = font(30)
    d.text((28, 74), subtitle, font=f_s, fill=(*DIM, 220))
    img.save(path)
    print("wrote", path)

# Generate
make_title(f"{OUT}/title.png",
           "OccluSynth",
           "Occlusion-Aware 3D Scene Reconstruction",
           "NIT Rourkela  ·  ScanNet v2  ·  VGGT-Omega")
make_close(f"{OUT}/close.png")
for i, (t, s) in enumerate(CAPTIONS, 1):
    make_lt(f"{OUT}/lt{i}.png", t, s)

print("\nAll assets written to", OUT)
