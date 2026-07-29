# OccluSynth — Final-Round Talk (7 min) — story-driven

*(Previous versions in git history: `git show 7999d63:STAGE_TALK.md`.)*

**Format:** 5 minutes of slides + 2 minutes of video, in a sandwich:
slides 1–5 (problem → insight → what we built → stills) →
**2-min demo video** → slide 6 (proof + impact) for the close.

**Materials:**
- Deck: `OccluSynth_7min.pptx` (own laptop) / PDF export (venue machine)
- Video: `demo_video/build/OccluSynth_hb2_2min.mp4` (1:49, silent — you narrate)
- Video talk-over: [`demo_video/VOICEOVER.md`](demo_video/VOICEOVER.md)
- Q&A appendix: `OccluSynth_Final_Round_Samsung.pptx`
- Numbers: [`docs/NUMBERS_CHEATSHEET.md`](docs/NUMBERS_CHEATSHEET.md)
- Deep dive if pressed: [`docs/COMPLETER_DEEPDIVE.md`](docs/COMPLETER_DEEPDIVE.md)

**The story spine — one sentence you are always inside of:**
> *A robot walks into a room, and the world ends at the front of the sofa. We spent
> the project teaching it that the world doesn't.*

**The one failure mode is going too fast.** Start ~20% slower than feels natural.
Hold the **[BEAT]**s — they feel 3× longer to you than to the room.

## Timing map

| # | Segment | On screen | Target | Running |
|---|---|---|---|---|
| 1 | Cold open | Slide 1 | 0:25 | 0:25 |
| 2 | The blind spot | Slide 2 | 1:05 | 1:30 |
| 3 | The third state | Slide 3 | 1:10 | 2:40 |
| 4 | What we built + what we learned | Slide 4 | 1:05 | 3:45 |
| 5 | The demo, in stills | Slide 5 | 0:30 | 4:15 |
| 6 | **DEMO VIDEO** | video, 1:49 | 1:50 | 6:05 |
| 7 | Proof → impact → close | Slide 6 | 0:55 | 7:00 |

Legend: **[BEAT]** = full 1-sec silence · *(slow)* / *(quieter)* = pace/volume ·
**HIT** = punch this word.

---

## Script

### 1 — Cold open [SLIDE 1 — title] (0:25)

*(Walk to centre. Beat. Look up before the first word.)*

Good [morning/afternoon]. I'm Aditya Agarwal, from NIT Rourkela.

*(slow)* I want to start with something you did this morning without noticing.
You walked past a sofa — and you knew the floor kept going underneath it. You've
never seen that floor. You just **knew**. **[BEAT]**

This is **OccluSynth**, problem statement nine. It's about the fact that robots
don't.

> *Why this open:* it makes the audience the expert for one sentence. Everything
> after lands as "obvious once said" rather than "technical claim."

### 2 — The blind spot [SLIDE 2] (1:05)

A robot walks into that same room. It sees a sofa. And to that robot, the world
**ENDS** at the front of it. Behind — nothing. Underneath — nothing. **[BEAT]**

*(point at 61%)* In a cluttered room, sixty-one percent of the observable volume
is hidden behind something. And of that hidden geometry, how much does a
state-of-the-art reconstruction recover? *(quieter, slow)* **Zero.** Not a low
score — **structural**. No sensor ever returns a measurement from behind a solid
surface.

So every fusion system today makes one of two dishonest choices. *(top card)*
Call the unseen space free — the planner drives straight through the hidden chair
leg. A **silent collision**. *(bottom card)* Or call it blocked — every hidden
voxel becomes a wall, and the robot **freezes**.

*(slow)* Unsafe, or useless. Safety-critical autonomy cannot ship on "assume it's
empty." **[BEAT]**

### 3 — The third state [SLIDE 3] (1:10)

The fix started with a question that sounds like semantics and isn't: *what does
the robot actually know about that space?*

Not "occupied or empty." *(legend row)* **Four** states. Seen empty. Measured
solid. **Occluded** — hidden, but inside the camera's view, so physics constrains
it and it can be **inferred**. And unobservable — outside the view entirely —
which we **leave alone**. *(slow)* We never guess where we have no right to.
That restraint is what makes the rest trustworthy. **[BEAT]**

*(pipeline row, left to right)* Then five stages, one pass. A frozen foundation
model — VGGT — turns plain RGB into depth. Sparse anchors pin it to metric scale.
Fusion casts every pixel as a ray and labels every voxel with those four states.
Then a 3D U-Net — trained on how real rooms are built — writes geometry into the
occluded region **only**: continues the floor under the table, closes the back of
the couch. And the planner prices what's imagined as **risk**, not fact.

A map that knows what it's unsure of is a map you can plan around.

### 4 — What we built, and what it taught us [SLIDE 4] (1:05)

*(This is the credibility beat. Deliver it calmly — it is the most senior-sounding
thing in the talk.)*

The completer is fourteen million parameters, trained on a **laptop**. No cloud
GPU. And the honest story of this project is what happened when it wasn't good
enough.

Feedback said the reconstructions didn't look convincing. So we tried to fix the
model. **Four times.** Reweighted the loss. Changed the architecture. Tripled the
training data. Retrained it stably when it diverged. **[BEAT]**

*(slow)* All four landed on **exactly the same** precision–recall curve. One of
them even improved its own validation loss — and still lost when we scored it
against ground truth.

*(quieter)* That's not four failures. That's a **finding**. It told us the ceiling
isn't tuning — it's resolution and the fact that hidden space is genuinely
ambiguous. What actually fixed the renders was calibration and honest filtering,
not a bigger model. **[BEAT]**

And we run **recall-first**, on purpose: a phantom obstacle costs a slowdown; a
**missed** one costs a collision. That asymmetry is priced, per voxel.

Enough diagrams. Let me show you the real thing.

> *If the clock is tight, cut from "And we run recall-first" — move it to Q&A.*

### 5 — The demo, in stills [SLIDE 5] (0:30)

*(let them look for a beat before speaking)* One apartment. Forty ordinary
photographs, no depth sensor. **[BEAT]**

*(left)* Conventional fusion — accurate, and **hollow**. *(middle)* OccluSynth on
the identical frames; every amber patch is geometry **no camera in that room ever
measured**. *(right)* ScanNet ground truth — the answer key, which the model never
sees at any point.

*(slow)* Compare the middle to the right. That is the whole claim.

Keep this picture in mind — now watch it being built.

### 6 — DEMO VIDEO [PLAY — 1:49]

Narrate with [`demo_video/VOICEOVER.md`](demo_video/VOICEOVER.md) — pointing, not
teaching. Three protected silences: the **amber reveal** (0:31), the **sofa
close-up** (1:05), the **three-way comparison** (1:11). *(Stop talking, point, let
them land.)* As the end card fades, advance to slide 6 and walk back to centre.

### 7 — Proof → impact → close [SLIDE 6] (0:55)

*(numbers first — they just saw it work)* Measured, not promised. Ten held-out
scenes the model had never seen. **Fifty-seven point six** percent of hidden
geometry recovered — against a structural **zero**. Occluded F-score, zero to
**thirty-seven**. Twenty-one percent of hidden hazards anticipated **before first
sight**. And reconstruction error *improved* where cameras **did** see — two-point-two
centimetres. **[BEAT]**

*(middle row — trace left to right; this is the "so what")* Here's what that buys
the robot. Fill the hole, and the obstacle behind the sofa **enters the map at
all**. Once it's in the map, its confidence becomes a **cost** — not an infinite
wall, a price. And once risk has a price, the robot hedges **before** it has line
of sight. *(honest, don't skip)* The map and the confidence ship today —
path-level avoidance is early, and we report that openly.

*(bottom row)* Which is why this lands first where looking twice is expensive.
Warehouse fleets burning minutes circling every pallet. Home robots where furniture
hides most of the floor. *(slower)* Inspection and rescue — behind rubble, inside
a duct, under a vehicle — where looking again isn't slow, it's **impossible**.
*(quieter, slowest line of the talk)* The child behind the parked car. **[BEAT]**

*(land it)* We taught a machine the thing you knew about that floor this morning
without thinking. The future of mapping isn't seeing more — it's knowing what you
**cannot** see. **[BEAT]** Thank you.

---

*~700 spoken words across ~5:10 of slide time + 1:49 video → 7:00 ± 15s.*

## Delivery cues

- §1 "You just **knew**" — warm, almost conversational. This is the only friendly
  moment before the problem gets serious. Use it.
- §2 **"Zero."** — quieter and slower, not louder. Pause before "Not a low score."
- §3 "We never guess **where we have no right to**" — hard stop after it.
- §4 **"That's a finding."** — the most important delivery in the talk. Do not
  rush it, do not sound apologetic. You are reporting a result, not conceding.
- §5 — say nothing for the first full beat. Let three panels do the work.
- §7 "The child behind the parked car" — slowest line. Then stop completely.

## If the clock runs long

1. Cut §4's "recall-first" paragraph → Q&A.
2. Compress §3's pipeline to three stages (VGGT → fusion → completer).
3. **Never** cut the §4 "four interventions" beat or the §7 middle row — the first
   is your scientific credibility, the second is your "so what".

## Q&A prep — likely questions

**"Why not diffusion / Stable Diffusion?"**
> Image models can't help — the hidden region is in *no* image, so there's nothing
> to inpaint; it has to happen in 3D. And a sampler gives a different plausible
> room every run: for a safety planner, "plausible" is the failure mode. Regression
> gives the *average* of plausible rooms — accurate but soft. We chose accurate.

**"Only 27% precision?"**
> Deliberate. Recall-first: a missed obstacle is a collision, a phantom one is a
> slowdown. Asymmetric costs, asymmetric operating point. Full curve is in the
> appendix.

**"Did more data help?"**
> No — tested twice, including once where it improved validation loss and still
> lost on the frontier and against ground truth. Four interventions, one frontier.
> That points at resolution, not tuning.

**"Is the 96³ run done?"**
> No — scripted and data-prepared, never executed, no compute access. It's the most
> credible remaining lever *precisely because* four 64³ interventions all plateaued.

**"How is this different from Atlas / NeuralRecon?"**
> They reconstruct what cameras saw, and they're good at it. Neither predicts behind
> surfaces — both score a structural zero there. Careful: their published numbers use
> a different protocol (full-scene, dense video), so I don't claim a direct
> comparison. The defensible claim is the occluded region: 0% versus 37%.

**"How do you know it isn't memorising?"**
> Validation scenes never appear in training — verified, not assumed. Unobservable
> space is excluded from the loss entirely, so it can't be rewarded for reproducing
> ground truth it had no evidence for.

## Rehearsal checklist

- [ ] Run once with a timer, no stopping — note where you drift over
- [ ] Run §4 alone three times until "That's a finding" sounds calm, not defensive
- [ ] Practise the video narration *muted* — you must not depend on audio
- [ ] Say "the child behind the parked car" out loud until you can pause after it
- [ ] Have the cheat sheet open on your phone for Q&A numbers
