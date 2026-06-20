#!/usr/bin/env python3
"""Generate geometry_table.png from demo_outputs/geometry_eval/results.json."""
import json
from PIL import Image, ImageDraw, ImageFont
import os

BG        = (14, 17, 22)        # #0E1116
HEADER_BG = (22, 27, 36)
ROW_ALT   = (18, 22, 30)
HL_BG     = (14, 40, 34)        # highlighted row tint (teal-dark)
WHITE     = (255, 255, 255)
DIM       = (160, 170, 185)
ACCENT    = (0, 210, 140)       # #00D28C
MUTED     = (100, 110, 125)
SEP       = (35, 42, 55)        # column separator

W, H = 1920, 1080

def font(size):
    for path in ["/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

with open("demo_outputs/geometry_eval/results.json") as f:
    data = json.load(f)

agg  = data["aggregate"]
pub  = data["published"]

# --- table data ---
# columns: Method | Chamfer L1↓ | F@5cm↑ | Compl%↑ || Occluded F@5cm↑ | Occluded Compl%↑
COLS = [
    ("Method",               220),
    ("Chamfer L1 (cm)",      185),
    ("F-score@5cm",          180),
    ("Completion",           165),
    ("│",                    24),    # divider
    ("Occluded\nF-score",    195),
    ("Occluded\nCompletion", 210),
]

def fmt_ch(v):  return f"{v:.2f} cm" if v else "—"
def fmt_f(v):   return f"{v*100:.1f}%"
def fmt_c(v):   return f"{v*100:.1f}%" if v else "—"

ROWS = [
    # (label,  superscript, ch,   fscore, compl,   occ_f,  occ_c,  highlight)
    ("Atlas",          "†",   6.50,  0.396,  None,   None,   None,  False),
    ("NeuralRecon",    "†",   5.20,  0.434,  None,   None,   None,  False),
    ("TSDF-only",      "",
     agg["surface"]["tsdf_only"]["chamfer_l1_cm"],
     agg["surface"]["tsdf_only"]["fscore_5cm"],
     agg["surface"]["tsdf_only"]["completion_ratio"],
     agg["occluded"]["tsdf_only"]["fscore_5cm"],
     agg["occluded"]["tsdf_only"]["completion_ratio"],
     False),
    ("OccluSynth",     " (ours)",
     agg["surface"]["completer"]["chamfer_l1_cm"],
     agg["surface"]["completer"]["fscore_5cm"],
     agg["surface"]["completer"]["completion_ratio"],
     agg["occluded"]["completer"]["fscore_5cm"],
     agg["occluded"]["completer"]["completion_ratio"],
     True),
]

# ---- layout -----------------------------------------------------------------
TITLE_H    = 100
SUBTITLE_H = 50
NOTE_H     = 70
COL_HDR_H  = 80
ROW_H      = 78
TABLE_H    = COL_HDR_H + len(ROWS) * ROW_H
SIDE_PAD   = 60

total_col_w = sum(w for _, w in COLS)
table_w     = total_col_w + SIDE_PAD * 2
table_top   = (H - TABLE_H - TITLE_H - SUBTITLE_H - NOTE_H) // 2 + 20

img = Image.new("RGB", (W, H), BG)
d   = ImageDraw.Draw(img)

f_title  = font(58)
f_sub    = font(28)
f_hdr    = font(26)
f_cell   = font(30)
f_ours   = font(32)
f_note   = font(22)
f_sup    = font(20)

# Title
title = "Geometry Evaluation"
tb = d.textbbox((0,0), title, font=f_title)
tx = (W - (tb[2]-tb[0])) // 2
ty = table_top - TITLE_H - SUBTITLE_H + 10
d.text((tx, ty), title, font=f_title, fill=WHITE)

# Accent underline
d.rectangle([(tx, ty + (tb[3]-tb[1]) + 8),
             (tx + (tb[2]-tb[0]), ty + (tb[3]-tb[1]) + 12)], fill=ACCENT)

# Subtitle
sub = "scene0000_00  ·  6 frames  ·  GT poses"
sb  = d.textbbox((0,0), sub, font=f_sub)
d.text(((W-(sb[2]-sb[0]))//2, ty+(tb[3]-tb[1])+22), sub, font=f_sub, fill=MUTED)

# Section labels above table
tbl_left  = (W - total_col_w) // 2
surf_cols = [COLS[0], COLS[1], COLS[2], COLS[3]]   # method + 3 surface cols
surf_w    = sum(w for _, w in surf_cols[1:3+1])  # width of the 3 data cols
surf_x    = tbl_left + COLS[0][1]
sec_y     = table_top - 36

d.text((surf_x + surf_w//2 - 60, sec_y), "Surface region", font=f_note, fill=DIM)

occ_x = tbl_left + sum(w for _, w in COLS[:5])
occ_w = COLS[5][1] + COLS[6][1]
d.text((occ_x + occ_w//2 - 70, sec_y), "Occluded region *", font=f_note, fill=ACCENT)

# Column header row
x = tbl_left
d.rectangle([(x, table_top), (x + total_col_w, table_top + COL_HDR_H)], fill=HEADER_BG)
cx = x
for col_name, cw in COLS:
    if col_name == "│":
        d.rectangle([(cx + cw//2 - 1, table_top+6), (cx + cw//2 + 1, table_top+COL_HDR_H-6)],
                    fill=SEP)
        cx += cw; continue
    lines = col_name.split("\n")
    line_h = 28
    ty_hdr = table_top + (COL_HDR_H - len(lines)*line_h) // 2
    for li, line in enumerate(lines):
        lb = d.textbbox((0,0), line, font=f_hdr)
        lx = cx + (cw - (lb[2]-lb[0])) // 2
        d.text((lx, ty_hdr + li*line_h), line, font=f_hdr, fill=DIM)
    cx += cw

# Data rows
for ri, (name, sup, ch, fs, co, occ_f, occ_c, hl) in enumerate(ROWS):
    ry = table_top + COL_HDR_H + ri * ROW_H
    row_bg = HL_BG if hl else (ROW_ALT if ri % 2 else BG)
    d.rectangle([(tbl_left, ry), (tbl_left + total_col_w, ry + ROW_H)], fill=row_bg)

    # subtle bottom border
    d.rectangle([(tbl_left, ry+ROW_H-1), (tbl_left+total_col_w, ry+ROW_H)], fill=SEP)

    cell_y  = ry + (ROW_H - 32) // 2
    name_f  = f_ours if hl else f_cell
    name_c  = ACCENT if hl else WHITE
    vals    = [
        fmt_ch(ch),
        fmt_f(fs),
        fmt_c(co),
        "│",
        fmt_f(occ_f) if occ_f is not None else "—",
        fmt_c(occ_c) if occ_c is not None else "—",
    ]

    cx = tbl_left
    # Method cell
    cw = COLS[0][1]
    mb = d.textbbox((0,0), name, font=name_f)
    d.text((cx + 16, cell_y), name, font=name_f, fill=name_c)
    if sup:
        sup_x = cx + 16 + (mb[2]-mb[0]) + 2
        d.text((sup_x, cell_y), sup, font=f_sup,
               fill=MUTED if not hl else (120,220,180))
    cx += cw

    for vi, (col_name, cw) in enumerate(COLS[1:]):
        v = vals[vi]
        if v == "│":
            d.rectangle([(cx+cw//2-1, ry+6),(cx+cw//2+1, ry+ROW_H-6)], fill=SEP)
            cx += cw; continue
        is_zero = (v == "0.0%")
        is_best = hl and v not in ("—",)
        vc = (220, 80, 80) if is_zero else (ACCENT if is_best else WHITE)
        vb = d.textbbox((0,0), v, font=f_cell)
        vx = cx + (cw - (vb[2]-vb[0])) // 2
        d.text((vx, cell_y), v, font=f_cell, fill=vc)
        cx += cw

# Note
note = "† Atlas / NeuralRecon numbers from published papers — full scene, dense video input, different protocol."
nb = d.textbbox((0,0), note, font=f_note)
d.text(((W-(nb[2]-nb[0]))//2,
        table_top + TABLE_H + 20), note, font=f_note, fill=MUTED)

note2 = "* TSDF-only scores 0 on occluded by construction — it cannot hallucinate behind walls."
nb2 = d.textbbox((0,0), note2, font=f_note)
d.text(((W-(nb2[2]-nb2[0]))//2,
        table_top + TABLE_H + 48), note2, font=f_note, fill=DIM)

out = "video_assets/geometry_table.png"
img.save(out)
print("wrote", out)
