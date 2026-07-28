# OccluSynth — Final-Round Talk (7 min) — SLIDES → DEMO → KPI

*(Previous 8-min founder-pitch version is in git history: `git show 7999d63:STAGE_TALK.md`.)*

**Format:** 5 minutes of slides + 2 minutes of video, in a sandwich:
slides 1–5 (problem → solution → innovation → the reconstruction stills) →
**2-min demo video** → slide 6 (KPI + impact) for the close.

**Materials:**
- Deck: `OccluSynth_7min.pptx` (own laptop) / `OccluSynth_7min.pdf` (venue machine)
- Video: `demo_video/build/OccluSynth_hb2_2min.mp4` (1:49, silent — you narrate)
- Video talk-over: [`demo_video/VOICEOVER.md`](demo_video/VOICEOVER.md)
- Q&A appendix: the full 12-slide `OccluSynth_Final_Round_Samsung.pptx`

**The one failure mode is going too fast.** Start every section ~20% slower than
feels natural. Hold the marked **[BEAT]**s — they feel 3× longer to you than to
the room. The video is your pace anchor: it cannot be rushed, so match it.

## Timing map

| # | Segment | On screen | Target | Running |
|---|---------|-----------|--------|---------|
| 1 | Walk-on + title | Slide 1 | 0:25 | 0:25 |
| 2 | The problem | Slide 2 | 1:10 | 1:35 |
| 3 | Our solution | Slide 3 | 1:15 | 2:50 |
| 4 | Innovation | Slide 4 | 0:55 | 3:45 |
| 5 | The demo — stills | Slide 5 | 0:30 | 4:15 |
| 6 | **DEMO VIDEO** | video, 1:49 | 1:50 | 6:05 |
| 7 | KPI → impact → close | Slide 6 | 0:55 | 7:00 |

Legend: **[BEAT]** = full 1-sec silence · *(slow)* / *(quieter)* = pace/volume ·
**HIT** = punch this word.

---

## Script

### 1 — Walk-on [SLIDE 1 — title] (0:25)

*(Walk to center. Beat. Look up before the first word.)*

Good [morning/afternoon]. I'm Aditya Agarwal, from NIT Rourkela — this is
**OccluSynth**, problem statement nine: reconstructing 3D scenes when most of
the scene is something your cameras **cannot see**.

### 2 — The problem [SLIDE 2] (1:20)

A robot walks into a room. It sees a sofa. And to that robot, the world **ENDS**
at the front of that sofa. Behind it — nothing. Underneath it — nothing.
**[BEAT]**

*(point at 61%)* In a typical cluttered room, sixty-one percent of the
observable volume is occluded — hidden behind furniture. And of that hidden
geometry, how much does a state-of-the-art reconstruction recover?
*(quieter, slow)* **Zero.** Not a low score — **structural**. No sensor ever
returns a measurement from behind a solid surface.

So today every fusion system makes one of two dishonest choices. *(gesture,
top card)* Call the unseen space free — and the planner drives straight through
the hidden chair leg. A **silent collision**. *(bottom card)* Or call it
blocked — and every hidden voxel becomes a wall. The robot **freezes**.

Unsafe, or useless. *(slow)* Warehouse robots, home robots, inspection bots —
safety-critical autonomy cannot ship on "assume it's empty." **[BEAT]**

### 3 — Our solution [SLIDE 3] (1:25)

Our answer is an honest **third state**. *(legend row)* Every voxel is labelled
with what the sensor actually knew: seen empty. Measured solid. **Occluded** —
hidden, but inside the camera's view, so it can be **inferred**. And
unobservable — outside the view entirely — which we **leave alone**. *(slow)*
We never guess where we have no right to. That restraint is what makes the
output trustworthy.

*(pipeline row, left to right)* Five stages, one pass. A frozen 3D foundation
model — VGGT — turns plain RGB into dense depth, and sparse anchors pin it to
metric scale. Fusion casts every pixel as a ray into a voxel grid. A 3D U-Net —
trained on how real rooms are built — writes geometry into the occluded region
**only**: it continues the floor under the table, closes the back of the couch.
Monte-Carlo dropout attaches a **confidence** to every voxel it imagined. And
the planner prices that risk — instead of trusting a guess. **[BEAT]** A map
that knows what it's unsure of is a map you can plan around.

### 4 — Innovation [SLIDE 4] (1:05)

What's actually new here? Against the closest published systems — Atlas,
NeuralRecon, VGGT itself, the 3D diffusion completers — OccluSynth is the first
to do all five of these. *(pick two, don't read the list)* The first to
reconstruct occluded geometry and attach a per-voxel confidence to it. And the
first to **measure** it as a safety problem — we built the benchmark, because
none existed.

The key idea is the **visibility mask as conditioning** — generation is fenced
to the recoverable region, so the model is never rewarded for hallucinating.
And we run **recall-first**, on purpose: a phantom obstacle costs a slowdown; a
**missed** one costs a collision. That asymmetry is priced, per voxel.

Enough diagrams. Let me show you the real thing.

### 5 — The demo, in stills [SLIDE 5] (0:30)

*(let them look for a beat before speaking)* One apartment. Forty ordinary
photographs, no depth sensor. **[BEAT]**

*(left)* This is what conventional fusion returns — accurate, and **hollow**.
*(middle)* This is OccluSynth on the identical frames; every amber patch is
geometry **no camera in that room ever measured**. *(right)* And that is the
ScanNet ground truth — the answer key, which the model never sees at any point.
*(slow)* Compare the middle to the right. That is the whole claim.

Keep this picture in mind — now watch it being built, in two minutes.

### 6 — DEMO VIDEO [PLAY `OccluSynth_hb2_2min.mp4` — 1:49]

Narrate over it with [`demo_video/VOICEOVER.md`](demo_video/VOICEOVER.md) —
pointing, not teaching. The three protected silences: the **amber reveal**
(0:31), the **sofa close-up** (1:05), and the **three-way comparison** (1:11 —
the same three panels they just saw as stills, now turning). *(Stop talking,
point at the screen, let them land.)* As the end card fades, advance to slide 6
and walk back to center.

### 7 — KPI + close [SLIDE 6] (0:55)

*(top row, numbers first — they just saw it work)* Measured, not promised — ten
held-out scenes the model had never seen. **Fifty-seven point six** percent of
the hidden geometry recovered, against a structural **zero**. Occluded F-score
zero to **thirty-seven**. Twenty-one percent of hidden hazards **anticipated
before first sight**. Reconstruction error down to two-point-two centimetres.
**[BEAT]**

*(middle row — trace it left to right with your hand; this is the "so what")*
And this is what that buys the robot. Fill the hole, and the obstacle behind the
sofa **enters the map at all**. Once it's in the map, its confidence becomes a
**cost** — not an infinite wall, a price. And once risk has a price, the robot
can hedge **before** it has line of sight. No circling. No second pass.
*(honest, don't skip)* The map and the confidence are what ship today —
path-level avoidance is early, and we report that openly.

*(bottom row)* Which is why this lands first where looking twice is expensive.
Warehouse fleets that burn minutes circling every pallet. Home robots where
furniture hides most of the floor. *(slower)* Inspection and rescue — behind
rubble, inside a duct, under a vehicle — where looking again isn't slow, it's
**impossible**. *(quieter, slowest line of the talk)* The child behind the
parked car. **[BEAT]**

Machines have always been blind to what they cannot directly see. We treat that
as normal. It isn't. *(land it)* The future of mapping isn't seeing more — it's
knowing what you **cannot** see. **[BEAT]** We built it. Thank you.

---

*Spoken words: ~700 across ~5:10 of slide time at a measured pace + 1:49 video
with light narration → lands at 7:00 ± 15s. If the clock is short, the clean
cut is §4's "recall-first" paragraph (move it to Q&A).*

## Delivery cues

- §2 **"Zero."** — quieter and slower, not louder. Pause before "Not a low score."
- §3 "We never guess **where we have no right to**" — hard stop after it.
- §5 → §6 handoff: say "in two minutes", click play, step to the side, half-turn
  to the screen. Don't talk until the filmstrip is rolling.
- §5 is your safety net: if the video ever fails to play, that slide alone
  carries the demo — talk to it and move on.
- Video: match the timecodes, and when in doubt — silence. The footage carries it.
- §7 has three rows — numbers, then the robot chain, then applications. Trace
  the chain left to right with your hand; it is the bridge from metric to meaning.
- §7 "the child behind the parked car" — slowest, quietest line of the talk,
  with a beat after. (It's not on any slide or in the video this time — it's yours.)
- Last two words with eye contact. Don't trail off.

## Q&A prep (appendix deck: `OccluSynth_Final_Round_Samsung.pptx`)

**Q: "Occluded precision is under 40% — isn't a hallucinated obstacle as
dangerous as a missed one?"**
> "It's exactly why every voxel ships with a confidence, not a hard label. A
> false obstacle at low confidence costs a slowdown; a missed real one costs a
> collision. We tune toward recall and let the planner reason over uncertainty —
> and we report both numbers openly, which is the point of building the
> benchmark." *(Appendix slide 8 has the full tables.)*

**Q: "Where do the camera poses come from?"**
> "Today, known poses — our one lab assumption, and I'm not hiding it. Dropping
> it is a SLAM/VIO front-end; the completion and confidence — the hard part —
> don't change." *(Appendix slide 10, path to production.)*

**Q: "How is this different from semantic scene completion / Occ3D?"**
> "Two things they don't do: we separate genuinely-inferable occluded space from
> out-of-view space and predict only the first; and we attach per-voxel
> confidence and score the occluded region as a safety problem, not a
> leaderboard number."

**Q: "Does the planner actually avoid hidden obstacles?"** *(don't overclaim)*
> "Honestly — not yet, at scale. On one of the ten benchmark scenes the
> risk-graded planner avoided 15.5% of the hidden hazards; on the other nine the
> path was unchanged, so the aggregate is zero. The deliverable today is the map
> and the confidence — turning that into closed-loop avoidance is the next step,
> and we report the zero rather than the one good scene."

**Q: "Isn't the 0% baseline a strawman?"**
> "It's by construction — no observation-only method, however good, can measure
> behind a surface. That's exactly why completion is a separate problem worth
> solving."

**Q: "What's the compute story?"**
> "Everything you saw trained on a 16 GB MacBook — the full 96³ A100 run is
> scripted and about a hundred GPU-hours away. It's a compute gap, not missing
> work." *(Appendix slide 11.)*

## Rehearsal checklist

- [ ] 3 full run-throughs with the video actually playing; write split times at
      §3 end (2:50), video start (4:15), video end (6:05).
- [ ] Practice the §4→video handoff until the click is invisible.
- [ ] Confirm the venue machine plays the mp4 BEFORE you go on (and keep the
      PDF deck as the fallback for fonts).
- [ ] Record yourself once; check you're not swallowing "Zero" and the final
      "Thank you."
