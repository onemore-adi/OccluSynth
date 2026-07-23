# OccluSynth — Main-Stage Talk (7–8 min) — FINAL

**Format:** founder-style pitch. No live/interactive demo in-talk (that's the booth).
The Reveal slide uses the **pre-rendered** comparison clip (`clips/comparison_sota.mp4`) —
a visual aid, not the live demo. Swap for a static frame if you prefer.

**The one failure mode (per delivery coach): going too fast.** ~930 spoken words +
the beats + the ~14-sec Reveal clip land at **7:30–8:00** at a normal-to-measured pace.
But adrenaline pushes first-timers faster — rush it and you finish near 6:45, sounding
hurried. Start every section ~20% slower than feels natural. Hold the marked **[BEAT]**s —
they feel 3× longer to you than to the room. Aim to land **7:30–7:45** with pauses intact.

## Timing map
| # | Section | Target | Running |
|---|---------|--------|---------|
| 1 | Cold open — the blind spot | 0:50 | 0:50 |
| 2 | The structural zero | 0:45 | 1:35 |
| 3 | Reveal — OccluSynth (show clip) | 0:45 | 2:20 |
| 4 | Why now — newly possible, we're first | 0:30 | 2:50 |
| 5 | How it works (predict vs. leave alone + confidence) | 1:05 | 3:55 |
| 6 | Proof — recall + the recall-first shield | 1:05 | 5:00 |
| 7 | Market — the warehouse wedge | 0:55 | 5:55 |
| 8 | Moat + traction | 0:40 | 6:35 |
| 9 | The ask + close | 0:45 | 7:20 |

Legend: **[BEAT]** = full 1-sec silence · *(slow)* / *(quieter)* = pace/volume · **HIT** = punch this word.

---

## Script

**[SLIDE 1 — Title: OCCLUSYNTH]**
*(Walk to center. Beat. Look up before the first word. Start 20% slower than feels right.)*

A robot walks into a room. It sees a sofa. And to that robot, the world **ENDS** at the front of that sofa. *(step down, slow each phrase)* Behind it — empty. Underneath it — empty. The wall it's pushed against — doesn't exist.

Every machine that perceives space today — every warehouse robot, every delivery bot, every phone scanning a room — is blind to everything it cannot **directly** see. We treat that as normal. It isn't. **[BEAT]** It's the biggest blind spot in spatial AI.

**[SLIDE 2 — Conventional reconstruction: the holey mesh]**

We measured exactly how blind. We took a state-of-the-art reconstruction system, gave it a set of camera views, and asked: of the geometry hidden behind surfaces — how much can it recover? **[BEAT]** *(quieter, slow)* Zero.

And here's the thing — that's not a low score. It's **structural**. No sensor ever returns a measurement from behind a solid surface. It's not a benchmark result; it's a law of the problem. And it's the gap we fill.

Today we paper over that gap by making machines **move more** — take another photo, circle around, look again. That works until it doesn't. And it's expensive every single time.

That zero is why a warehouse robot circles a pallet three times to be sure. It's why a rescue drone can't tell you what's behind the rubble without flying behind it. *(slow, quieter, eye contact with one judge)* It's the child behind the parked car. **[BEAT]**

**[SLIDE 3 — Reveal: play `comparison_sota.mp4` — grey measured + amber predicted]**

We're OccluSynth. We give machines something they've never had — the ability to reconstruct what they **can't** see.

*(Let the clip breathe — stop talking for a beat, point at the screen.)* Same room. Grey is what the cameras measured. *(beat)* Amber is what OccluSynth reconstructed — the back of the sofa, the floor beneath it, the wall behind it. Geometry no camera ever captured — from the same handful of frames, in a single pass. **[BEAT]**

**[SLIDE 4 — Why now]**

And this is newly possible. Two years ago, getting metric 3D from a few images took minutes and a rig. 3D foundation models changed that overnight — single-pass geometry is now cheap. Every serious robotics and AR company is building on that new geometry layer right now — and every one of them inherits the same blind spot. We're the layer on top that closes it — that makes the geometry **complete**, and tells you how much to trust it.

**[SLIDE 5 — How it works: predict vs. leave alone + confidence]**

How? The key idea is restraint. We split the hidden world in two. There's space that's **occluded** — behind a surface, but still inside the camera's view. And space that's **unobservable** — outside the view completely. We predict the first. We leave the second alone. *(clean, slower)* We never guess where we have no right to — and that's what makes the output trustworthy.

A 3D neural network fills those occluded regions — it continues the floor under the table, closes the back of the couch, using how real rooms are built. And every voxel it predicts carries a **confidence**. The robot doesn't just imagine geometry — it knows how much to trust each guess. A map that knows what it's unsure of is a map you can plan around.

**[SLIDE 6 — Numbers]**

Does it work? On ten scenes the model had never seen, OccluSynth recovers **fifty-eight percent** of the hidden geometry — fifty-eight percent surface recall at five centimeters — against that structural zero.

Now — we run **recall-first**, on purpose. *(this is the shield — say it before anyone asks)* Because the two mistakes aren't equal. A phantom obstacle costs a robot a slowdown. A **missed** one costs a collision. So we deliberately over-flag — and the per-voxel confidence lets the planner weigh each guess instead of trusting a hard map. Precision is the knob we climb with the larger model. And we built the first open benchmark that scores the occluded region as a **safety** problem — with confidence, not just an accuracy number.

**[SLIDE 7 — Market: the warehouse wedge]**

Where does this land first? Warehouse robots. Fleets optimized to the second, where circling every shelf to peek behind it destroys throughput. Think about what occlusion costs a fleet today: a robot that stops for obstacles that aren't there burns minutes; one that clips an obstacle that **is** there burns a repair and a safety review. Single-pass correctness attacks both. A robot that acts correctly on one pass is a fleet-utilization multiplier — a buyer with a budget and a number. From there, the market is every system that has to **act before it can look**.

And the obvious objection — why not just move the camera and look? Sometimes you can't. Often it's too expensive. And in a changing room, the view you took thirty seconds ago is already stale. *(land it)* Our map has a memory. Theirs doesn't.

**[SLIDE 8 — Moat + traction]**

What's defensible here isn't the neural network — it's the **representation** and the **confidence**. We're the only system that can tell you how much of a map is imagined, and how sure we are, voxel by voxel. That audit trail — what's measured, what's inferred, and how sure — doesn't exist in any reconstruction stack today. And it's exactly what a safety board — or an insurance underwriter — needs to sign off. Today this is a working system, an open model, and the first public occlusion-safety benchmark — fully reproducible.

**[SLIDE 9 — The ask + close]**

We're raising **$250K pre-seed** to do three things in twelve months: train at full scale, ship the front-end that drops our last lab assumption — known camera poses — so it runs on any robot, and land a paid warehouse pilot with a design partner.

*(slow, calm — calm reads as confidence)* Object permanence for machines — the ability to reason about the unseen — is the missing layer of spatial intelligence. **[BEAT]** We built it. **[BEAT]** Come see it live at our booth. *(chin up, hold eye contact through the last two words — don't trail off)* Thank you.

---
*Spoken words: ~930. At ~135 wpm + beats + the Reveal clip → ~7:30–7:45.*

## If you hit 7:45 and must land under 8:00 (rehearse these as clean exits)
1. **§7** — cut the whole "why not just move and look?" block; keep only **"Our map has a memory. Theirs doesn't."** (~18 sec, safest cut)
2. **§4** — cut the "two years ago… minutes and a rig" sentence; keep "We're the layer that makes it complete."
3. **§6** — cut "And we built the first open benchmark…" (move to Q&A)

## Delivery cues (mark these on your printout)
- §1 after **"It isn't."** — hold a full second before the blind-spot line. Don't fill it.
- §2 **"Zero."** — quieter and slower, not louder. Then pause before "that's not a low score."
- §2 **"the child behind the parked car"** — slowest, quietest line in the talk; beat of silence after.
- §3 the clip — **stop talking, let it play**, point, then narrate grey→amber.
- §9 **"We built it."** — isolate it with beats on both sides. Last word as strong as the first.

## Hardest lines — say them this way (mouth-friendly)
- "Occluded — behind a surface, but still inside the camera's view." *(hard stop)* "Unobservable — outside the view completely." — short sentences = built-in breaths.
- "We never guess **where we have no right to**." (not "…what we have no business guessing about")
- "…needs to **sign off**." (confident landing, avoids the "safety-certification / insurance-underwriter" pileup)

---

## Q&A prep — the questions that decide the round

**Q (the hardest, from a technical judge): "Your occluded precision is under 40% — so more than half the geometry you draw isn't real. Isn't a confident hallucinated obstacle as dangerous as a missed one?"**
> "Fair — and it's exactly why every voxel ships with a confidence, not a hard label. A false obstacle at low confidence costs a slowdown; a missed real one costs a collision. That asymmetry is the design: we tune toward recall and let the planner reason over uncertainty, not a binary map. Precision is what the 96³ model and confidence calibration are for — and we report both numbers openly, which is the point of building the benchmark."

**Q: "Where do the camera poses come from?"**
> "Today we use known poses — that's our one lab assumption, and I'm not hiding it. Dropping it is a solved-in-principle SLAM/pose front-end; it's the first thing the raise funds. The completion and confidence — the hard part — don't change."

**Q: "How is this different from semantic scene completion / occupancy prediction work (SSC, Occ3D)?"**
> "Those predict a full occupancy grid and score it with one accuracy number. We do two things they don't: we separate genuinely-inferable occluded space from out-of-view space and only predict the first, and we attach per-voxel confidence and score the occluded region as a safety problem. It's completion built for a robot that has to act, not for a leaderboard."

**Q: "Does the planner actually avoid the hidden obstacles yet?"** *(be honest — do NOT overclaim)*
> "Path-level avoidance is early — it helps on a minority of scenes today. The deliverable right now is the map and the confidence; turning that into closed-loop avoidance is downstream work, and it's on the roadmap."

**Q: "Isn't the 0% baseline a strawman?"**
> "It's the opposite of a strawman — it's by construction. No observation-only method, however good, can measure behind a surface. That's the whole reason completion is a separate problem worth solving."

## Rehearsal checklist
- [ ] 3 full run-throughs with a stopwatch; write your split times at §3, §6, §9.
- [ ] Practice the 3 cuts as clean exits so a long clock doesn't cause a panic-skip.
- [ ] Record yourself once; check you're not swallowing "Zero" and the last "Thank you."
- [ ] Confirm the Reveal clip is cued and plays on the venue machine before you go on.
