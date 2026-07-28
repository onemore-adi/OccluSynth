# OccluSynth 5-min Video — Voiceover Script & Recording Guide

For `build/OccluSynth_hb5_5min.mp4` (5:03, silent). Timecodes below match the final
cut exactly. **Read this first:**

## How this fits your stage flow

You walk on → intro yourself → play this video (with this VO) → 15-min stage talk
(`STAGE_TALK.md`). The two are deliberately split:

| | The video VO (this file) | The stage talk |
|---|---|---|
| Job | **HOW it works** — calm, technical, describes what's on screen | **WHY it matters** — problem, market, moat, ask |
| Register | Documentary narrator. Low, even, unhurried. | Founder. Energy, beats, eye contact. |
| Numbers | Pipeline numbers (40 frames, 1.9 % depth error, 14.7 M params, 58 % recall) | Business numbers (warehouse wedge, $250K, 12 months) |

**Lines reserved for the stage — do NOT say them in the VO** (the audience must hear
them for the first time live): "the world **ends** at the front of that sofa" ·
the dramatic quiet "*Zero.*" · "the child behind the parked car" (it appears
**on screen** in the video at 4:39 — on stage it becomes a callback) · "Our map has
a memory. Theirs doesn't." · "Object permanence for machines" (also on-screen only)
· "We built it."

The overlap that remains (structural zero, 58 %, occluded-vs-unobservable) is
intentional: the video shows it, the talk *lands* it. Slide 2 (holey mesh), slide 3
(the Reveal clip) and slide 5 (how it works) can now be brisker on stage — say
"you saw this in the film" and go deeper instead of re-explaining. Since the video
already plays `comparison_sota.mp4` at 3:25, consider a **static frame** on Slide 3
instead of replaying the clip.

---

## The script

Target pace **~135 wpm** (calm, measured). `[BEAT]` = one full second of silence.
`(silence)` = say nothing; the screen is talking. **Bold** = the word to lean on.
Each block starts on the timecode — if you finish a block early, hold silence;
never rush to fill.

---

**0:00 – 0:10 · Title card**

> *(silence until 0:04)*
> This is a five-minute look at how we taught a machine to see **past** its own eyes.

**0:10 – 0:15 · Robot drives into the room**

> A robot enters a room it has never seen. Cameras only. No lidar.

**0:15 – 0:20 · Reveal — "should it drive anyway?" text**

> Almost immediately, it has a problem. [BEAT]

**0:20 – 0:27 · Card: "A robot's world ends at the first surface it sees."**

> Every camera, every depth sensor ever built shares one limit — light stops at the
> first thing it hits.

**0:27 – 0:34 · Grey mesh full of holes ("TODAY" pill)**

> So this is what today's best reconstruction actually produces. Every black gap is
> space **no camera reached**.

**0:34 – 0:41 · The 0 % card**

> And of the geometry hidden behind those surfaces, observation recovers exactly
> **none** — by construction.

**0:41 – 0:48 · Amber voxel reveal ("HIDDEN" pill)**

> Everything in amber is that hidden space. [BEAT] Amber is what this system is **for**.

**0:48 – 0:56 · Thesis card**

> OccluSynth reconstructs it anyway — and reports, voxel by voxel, how much of the
> map is **imagined**, and how much to trust it.

**0:56 – 1:02 · Chapter card: The pipeline**

> Here's the entire system — six stages, one pass, running on real data.

**1:02 – 1:14 · System-overview diagram (five boxes)**

> Perception turns pixels into depth. Fusion turns depth into evidence. Completion
> imagines what's hidden. Uncertainty grades every guess — and the planner acts on
> all of it.

**1:14 – 1:20 · Card: 01 Capture**

> Stage one. The input. [BEAT]

**1:20 – 1:32 · RGB filmstrip scrolling**

> Forty ordinary photographs — one handheld sweep of a real apartment. This is
> **everything** the system will ever see. No depth sensor anywhere in this pipeline.

**1:32 – 1:38 · Card: 02 Perception**

> Stage two turns those pictures into geometry.

**1:38 – 1:46 · Depth maps: raw / calibrated / ground truth**

> A frozen foundation model — VGGT — predicts dense depth for every frame in a
> single forward pass.

**1:46 – 2:00 · Anchor scatter + RANSAC fit animation**

> But its depth comes out **relative** — beautiful shape, wrong size. A few hundred
> sparse anchor points and one robust fit per frame pin it to metric scale. [BEAT]
> Final depth error: about **two percent**.

**2:00 – 2:06 · Card: 03 Fusion**

> Stage three — every pixel becomes evidence.

**2:06 – 2:14 · Ray-casting fusion animation**

> Each calibrated depth ray is cast into a shared voxel grid — carving out the empty
> space it crossed, marking the surface it stopped at.

**2:14 – 2:21 · Four-state card**

> That gives every voxel one of four states. The split that matters is the last two:
> occluded space is hidden but **inferable**. Unobserved space is not — so we never
> touch it.

**2:21 – 2:28 · Card: 04 Completion**

> Stage four is where the imagination happens.

**2:28 – 2:42 · 3D U-Net architecture build**

> A three-D U-Net — fifteen million parameters, trained on how real rooms are built.
> The loss is masked: it's graded **only** on measured and occluded space, so it's
> never rewarded for guessing outside its rights.

**2:42 – 2:48 · Amber mesh growing in**

> Here it is filling the room in — the white ghost is the ground truth it never saw.

**2:48 – 2:54 · Card: 05 The mesh**

> Stage five turns voxels into solid geometry.

**2:54 – 3:02 · Grey "BEFORE" turntable**

> Marching cubes, on fusion alone — accurate, and **hollow**. This mesh simply ends
> where the cameras did.

**3:02 – 3:14 · Completion fade "AFTER"**

> Now watch the same room as completion switches on. [BEAT] Amber closes the floor,
> the walls, the backs of the furniture.

**3:14 – 3:26 · Sofa close-up**

> Behind this sofa, there is real geometry in the model — the floor it sits on, the
> wall it hides. **No camera was ever back there.**

**3:26 – 3:40 · Side-by-side comparison (0 % → 58 %)**

> Same forty frames. Conventional on the left — OccluSynth on the right.
> *(silence — let the 58 % panel land)*
> Fifty-eight percent of the hidden surfaces, recovered.

**3:40 – 3:45 · Card: 06 Trust & plan**

> The last stage is the one that makes this **usable**.

**3:45 – 3:53 · Flickering uncertainty cloud**

> Sixteen stochastic passes through the network give every predicted voxel a
> confidence. Solid means sure — flicker means honest doubt.

**3:53 – 4:08 · Planner cost-grid animation**

> The planner prices that in. Measured surfaces are impassable. Amber costs whatever
> its confidence says it should. [BEAT] So caution is no longer a blanket rule — it's
> a **number**, per voxel.

**4:08 – 4:14 · Naive path hits hidden hazard**

> A planner without this drives the straight line — through space it merely couldn't see.

**4:14 – 4:20 · Risk-aware detour**

> Ours detours **before** it can see why. [BEAT]

**4:20 – 4:25 · Chapter card: Why it matters**

> *(silence)*

**4:25 – 4:30 · Hazard-dots benchmark**

> On our safety benchmark, hazard awareness goes from zero — to one in five,
> anticipated before first sight.

**4:30 – 4:35 · Metrics card**

> Reconstruction error nearly halves. [BEAT]

**4:35 – 4:48 · Three lines + "Object permanence, for machines."**

> *(silence — the screen carries these lines; do not read them)*

**4:48 – 4:53 · Room callback**

> The same room as minute one. [BEAT] Nothing about it is a mystery anymore.

**4:53 – 5:03 · End card**

> OccluSynth. [BEAT] *(silence to black — you walk to center as it fades)*

---

Word count ≈ 480 → ~3:35 of speech inside 5:03. The silences are part of the design;
protect them.

---

## Recording guide

**Sound**
- Room: the quietest, deadest space you have — a closet full of clothes beats an
  empty room. Kill fans/AC. Phone on aeroplane mode.
- Mic 15–20 cm away, slightly **off-axis** (talk past it, not into it) to avoid pops.
  A phone in a quiet closet is genuinely fine; hold it steady, don't touch it while
  recording.
- Record everything at one sitting so the voice matches; 48 kHz if the app lets you choose.
- Leave 2 s of silence at the start and end of every take (room tone — needed for cleanup).

**Delivery**
- Calibrate pace first: the block at 1:02 ("Perception turns pixels…") should take
  **~11 seconds**. Time yourself until it does — that's your speed for the whole read.
- Pitch it like a nature documentary, not a pitch. The excitement lives in the stage
  talk; the VO is the calm expert. If a line feels flat, go **slower and quieter**,
  never louder.
- Stress only the bolded words. One stress per sentence, maximum.
- Consonant endings matter at this pace — don't swallow "reached", "none", "sure".
- The three killers: rushing after a mistake, smiling-voice on technical lines, and
  filling the marked silences. When in doubt: silence.

**Workflow**
- Easiest: play the video muted on one screen, record in one continuous take while
  watching the timecodes — the blocks are written with slack, so drifting a second
  is fine; just re-sync at each chapter card (0:56, 2:21, 3:40, 4:20).
- Safer: record each block as its own take (say the timecode before each, clap once,
  then read). I can assemble and align them to the cut.
- Do a full throwaway take first. Take two is usually the keeper.

**When you have the audio** — drop the file(s) anywhere in the repo and I'll clean,
level (-16 LUFS), align and mux them. Or DIY for a single aligned take:

```bash
ffmpeg -i build/OccluSynth_hb5_5min.mp4 -i vo_take2.wav \
  -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -af "loudnorm=I=-16:TP=-1.5" \
  -shortest build/OccluSynth_hb5_5min_vo.mp4
```
