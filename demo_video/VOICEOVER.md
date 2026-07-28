# OccluSynth 2-min Demo — Talk-over Script

For `build/OccluSynth_hb2_2min.mp4` (**1:49**, silent) — the demo segment of the
7-minute presentation. It plays MID-TALK: slides 1–5 first (problem → solution →
innovation → the reconstruction stills), then this video, then the KPI slide
to close. Full talk:
[`../STAGE_TALK.md`](../STAGE_TALK.md).

**This cut has no explainer cards** — every stage was already explained on your
Solution slide. So the talk-over is *pointing, not teaching*: name what's on
screen, attach it to the stage number from the slide, and let the footage
breathe. If you fall behind, skip a line — never talk over the sofa reveal or
the three-way comparison.

The previous 5-minute version (with its own VO) is preserved:
`OccluSynth_hb5_5min.mp4` + `VOICEOVER_5min.md`.

---

## The script

`[BEAT]` = one second of silence. **Bold** = the word to lean on. Timecodes match
the cut exactly; re-sync at 0:59 (sofa) and 1:11 (comparison) if you drift.

**0:00 – 0:12 · RGB filmstrip scrolling**

> This is the entire input. Forty ordinary photographs — one handheld sweep of a
> real apartment. No depth sensor anywhere.

**0:12 – 0:20 · VGGT depth: raw / calibrated / ground truth**

> Stage one — VGGT turns every frame into dense depth, and our anchors pin it to
> **metric** scale.

**0:20 – 0:27 · Fusion rays carving the grid**

> Stage two — every pixel becomes a ray, carving free space and marking surfaces.

**0:27 – 0:33 · Amber voxel reveal**

> And everything in amber — that's the space **no camera ever saw**. [BEAT]

**0:33 – 0:40 · Completion growing vs ghost ground truth**

> Stage three — the completer imagines the hidden geometry. The white ghost is the
> ground truth it was never shown.

**0:40 – 0:48 · Grey "BEFORE" turntable**

> Without it, this is the map: accurate — and **hollow**.

**0:48 – 0:59 · Completion fade "AFTER"**

> Watch the same room as completion switches on. [BEAT] The floor, the walls, the
> backs of the furniture — closed.

**0:59 – 1:11 · Sofa close-up**

> *(quieter, slower)* Behind this sofa: the floor it sits on, the wall it hides.
> **No camera was ever back there.** [BEAT]

**1:11 – 1:25 · Three-way comparison, 0% → 57.6%**

> The same three panels from the slide — now turning. Conventional, ours, and
> the ScanNet ground truth the model never sees.
> *(silence — let the 57.6% land)*

*(These are the identical stills from slide 5, so say "the same three panels"
and let them re-anchor. All three turntables share one camera.)*

**1:25 – 1:33 · Flickering confidence cloud**

> Stage four — every guess carries a confidence. Solid is sure; flicker is honest
> doubt.

**1:33 – 1:38 · Naive path hits hidden hazard**

> A planner without this drives the straight line —

**1:38 – 1:43 · Risk-aware detour**

> — ours detours **before** it can see why.

**1:43 – 1:49 · End card**

> *(silence to black — advance to the KPI slide as it fades)*

---

~150 words of speech in 1:49. The three protected silences: the amber beat
(0:31), the sofa reveal (1:05), and the three-way comparison (1:18).

---

## Recording / rehearsal notes

- You'll most likely deliver this **live** over the video — rehearse with the
  file playing muted; the blocks have slack, and the two re-sync points are the
  sofa (0:59) and the comparison (1:11).
- If you'd rather pre-record it, the guidance from the 5-min version applies
  unchanged (closet room tone, mic off-axis, one sitting, calibrate pace) — see
  `VOICEOVER_5min.md`. Mux command:

```bash
ffmpeg -i build/OccluSynth_hb2_2min.mp4 -i vo_2min.wav \
  -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -af "loudnorm=I=-16:TP=-1.5" \
  -shortest build/OccluSynth_hb2_2min_vo.mp4
```
