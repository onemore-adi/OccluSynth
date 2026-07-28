#!/usr/bin/env python3
"""Assemble the 2-minute pure-demo cut for the 7-minute presentation format.

Slides carry all explanation now, so this cut is footage only — no section or
explainer cards. Reuses the conformed clips from the 5-min build
(renders/hb5/conformed/): run conform_hb.sh first if they are missing.

Order: RGB input → VGGT depth → fusion rays → amber voxels → completion growth
→ mesh before/after → sofa close-up → SOTA comparison → confidence cloud →
naive collide vs risk-aware detour → end card.

Usage:  python3 assemble_hb2.py [--dry]
"""
import subprocess, sys, os, json

C = "renders/hb5/conformed"
OUT = "build/OccluSynth_hb2_2min.mp4"
XF = 0.8

# (clip, trim_start, trim_dur) — None = full clip
SEQ = [
    ("c12_filmstrip", None, None),   # 40 RGB frames scrolling
    ("c14_depth",     None, None),   # VGGT raw / calibrated / GT depth
    ("c17_fusion",    None, None),   # ray-cast fusion animation
    ("c07_amber",     None, None),   # amber voxel reveal
    ("c21_growth",    None, None),   # completion growing vs ghost GT
    ("c23_before",    None, None),   # TSDF-only mesh turntable
    ("c24_fade",      None, None),   # completion fading in
    ("c25_sofa",      None, None),   # behind-the-sofa close-up
    ("c26_compare",   None, None),   # side-by-side 0% -> 58%
    ("c28_uncert",    None, None),   # per-voxel confidence cloud
    ("c30_collide",   None, None),   # naive path hits hidden hazard
    ("c31_detour",    None, None),   # risk-aware detour
    ("c37_end",       0, 6.0),       # end card build, trimmed
]


def dur(f):
    return float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", f]).decode().strip())


def main():
    dry = "--dry" in sys.argv
    durs = []
    for name, ss, d in SEQ:
        f = os.path.join(C, name + ".mp4")
        assert os.path.exists(f), f"missing {f} — run conform_hb.sh"
        durs.append(d if d else dur(f) - (ss or 0))

    total = sum(durs) - XF * (len(SEQ) - 1)
    start = 0.0
    print(f"clips={len(SEQ)}  total={total:.1f}s = {int(total//60)}:{total % 60:04.1f}")
    for (name, ss, d), dd in zip(SEQ, durs):
        print(f"  {int(start//60)}:{start % 60:04.1f}  {name:15s} {dd:5.2f}")
        start += dd - XF
    if dry:
        return

    lines = []
    for i, ((name, ss, d), dd) in enumerate(zip(SEQ, durs)):
        trim = ""
        if ss or d:
            trim = f",trim=start={ss or 0}:duration={(ss or 0) + dd},setpts=PTS-STARTPTS"
        lines.append(f"[{i}:v]settb=AVTB,fps=24{trim}[v{i}]")
    cur, off = "v0", durs[0]
    for i in range(1, len(SEQ)):
        off -= XF
        lines.append(f"[{cur}][v{i}]xfade=transition=fade:duration={XF}:offset={off:.4f}[x{i}]")
        cur = f"x{i}"
        off += durs[i]
    lines.append(f"[{cur}]fade=t=in:st=0:d=0.8,"
                 f"fade=t=out:st={total-1.2:.2f}:d=1.2,format=yuv420p[vout]")
    os.makedirs("renders/hb5", exist_ok=True)
    with open("renders/hb5/xfade_hb2.txt", "w") as fh:
        fh.write(";\n".join(lines))

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for name, _, _ in SEQ:
        cmd += ["-i", os.path.join(C, name + ".mp4")]
    cmd += ["-f", "lavfi", "-t", f"{total:.3f}",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-filter_complex_script", "renders/hb5/xfade_hb2.txt",
            "-map", "[vout]", "-map", f"{len(SEQ)}:a",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "aac", "-b:a", "128k", "-shortest",
            "-movflags", "+faststart", OUT]
    subprocess.check_call(cmd)
    print(f"done -> {OUT}  ({dur(OUT):.1f}s)")


if __name__ == "__main__":
    main()
