#!/usr/bin/env python3
"""Assemble the 5-min hackathon cut from renders/hb5/conformed/c*.mp4.

Slow dissolves throughout (0.8s), longer (1.2s) into the three chapter pivots,
fade from/to black at the ends, silent stereo track for player compatibility.

Usage:  python3 assemble_hb.py [--dry]   (from demo_video/)
"""
import subprocess, sys, os, glob, json

O = "renders/hb5/conformed"
OUT = "build/OccluSynth_hb5_5min.mp4"
DEFAULT_XF = 0.8
# boundaries that get a longer, slower dissolve: clip name whose ENTRY is slow
SLOW_IN = {"c09_pipeline": 1.2, "c32_why": 1.2, "c37_end": 1.4}
# optional per-clip max duration cap (tuning knob to hit 5:00)
CAP = {}


def dur(f):
    return float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", f]).decode().strip())


def main():
    dry = "--dry" in sys.argv
    files = sorted(glob.glob(os.path.join(O, "c*.mp4")))
    assert files, "no conformed clips found"
    names = [os.path.basename(f)[:-4] for f in files]
    durs = []
    for f, n in zip(files, names):
        d = dur(f)
        if n in CAP:
            d = min(d, CAP[n])
        durs.append(d)

    xfs = []
    for i in range(1, len(files)):
        xfs.append(SLOW_IN.get(names[i], DEFAULT_XF))

    total = sum(durs) - sum(xfs)
    print(f"clips={len(files)}  sum={sum(durs):.1f}s  total after xfades={total:.1f}s"
          f"  = {int(total//60)}:{total % 60:04.1f}")
    for n, d in zip(names, durs):
        print(f"  {n:18s} {d:6.2f}")
    if dry:
        return

    # build filter graph
    lines = []
    for i, (f, n) in enumerate(zip(files, names)):
        cap = f",trim=duration={CAP[n]}" if n in CAP else ""
        lines.append(f"[{i}:v]settb=AVTB,fps=24{cap}[v{i}]")
    cur = "v0"
    off = durs[0]
    for i in range(1, len(files)):
        xf = xfs[i - 1]
        off -= xf
        nxt = f"x{i}"
        lines.append(f"[{cur}][v{i}]xfade=transition=fade:duration={xf}:offset={off:.4f}[{nxt}]")
        cur = nxt
        off += durs[i]
    # ends: fade from and to black
    lines.append(f"[{cur}]fade=t=in:st=0:d=1.0,"
                 f"fade=t=out:st={total-1.6:.2f}:d=1.6,format=yuv420p[vout]")
    script = ";\n".join(lines)
    with open("renders/hb5/xfade_graph.txt", "w") as fh:
        fh.write(script)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stats_period", "10", "-stats"]
    for f in files:
        cmd += ["-i", f]
    cmd += ["-f", "lavfi", "-t", f"{total:.3f}", "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000"]
    cmd += ["-filter_complex_script", "renders/hb5/xfade_graph.txt",
            "-map", "[vout]", "-map", f"{len(files)}:a",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "aac", "-b:a", "128k", "-shortest",
            "-movflags", "+faststart", OUT]
    print("encoding ...")
    subprocess.check_call(cmd)
    print(f"done -> {OUT}  ({dur(OUT):.1f}s)")


if __name__ == "__main__":
    main()
