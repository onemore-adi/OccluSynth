# OccluSynth — 3-Minute Booth Pitch

**Audience:** curious visitors, not judges. **Goal:** they leave understanding the
*problem* and *why it matters* — not the architecture.
**Rule:** problem and applications over inner workings. Warm, light, no jargon.
If they want depth, that is what [`docs/COMPLETER_DEEPDIVE.md`](docs/COMPLETER_DEEPDIVE.md) is for.

---

## Timing map

| Beat | Time | Purpose |
|---|---|---|
| 1 — The hook | 0:00–0:30 | A question they can answer |
| 2 — The problem | 0:30–1:10 | Make the gap obvious and visual |
| 3 — What we do | 1:10–1:50 | One idea, no architecture |
| 4 — Why it matters | 1:50–2:35 | Applications — the part they remember |
| 5 — Close | 2:35–3:00 | One number + invitation |

---

## 1 — The hook (0:00–0:30)

> **[Point at the screen, before explaining anything]**
>
> "Quick question — look at this 3D scan of a room. Where's the floor under that
> sofa?"
>
> *[let them look — they'll notice it's missing]*
>
> "Right — there isn't one. The scan has a hole there. And here's the thing: the
> computer doesn't think there's a hole. It thinks that space is **empty**. As in,
> drive-right-through-it empty."

**Delivery:** genuinely wait for the answer. The moment they spot it themselves,
you have them for the next three minutes.

---

## 2 — The problem (0:30–1:10)

> "Every camera has the same limitation your eyes do — it only sees the front of
> things. Look at a sofa, you see the front of a sofa. You've never in your life
> seen the back of your own sofa while looking at the front of it."
>
> *[beat]*
>
> "You know it's there. You'd never walk into it. But that's because you've got
> years of experience with sofas. A robot with a camera has none of that. It sees a
> hole in its map, and a hole means 'go ahead'."
>
> "So every 3D scanner today makes the same mistake: it confuses **'I didn't see
> it'** with **'there's nothing there.'** For a video game that's fine. For a robot
> in your kitchen at 2am, that's how you end up buying a new robot."

**If they laugh, ride it — that is the joke landing.**

---

## 3 — What we do (1:10–1:50)

> "So we taught a system to do the thing you do without thinking: **imagine the
> parts it can't see.**"
>
> **[gesture at the before/after]**
>
> "Grey is what the cameras actually measured. Orange is what our model *predicted*
> was hidden back there — the floor continuing under the sofa, the wall carrying on
> behind the cabinet, the back of the furniture."
>
> "And crucially, it *knows the difference*. It never pretends the orange is
> measured. It keeps three separate ideas apart: 'I saw this and it's solid', 'I
> saw this and it's empty', and 'this is my blind spot — here's my best guess, treat
> it with caution.'"
>
> "That third one basically doesn't exist in normal 3D scanning. That's the whole
> project."

**Do not mention:** U-Nets, SDFs, voxels, losses. If asked "how?" — *"a small neural
network trained on a few hundred real room scans, learning what rooms usually do —
floors continue, walls don't just stop."* Then move on.

---

## 4 — Why it matters (1:50–2:35) ← *the part they remember*

> "Where this actually matters:"
>
> **Home and warehouse robots.** "A robot vacuum that knows a chair leg continues to
> the floor instead of rediscovering it by collision. Warehouse robots planning
> around a pallet they can only see one side of."
>
> **Self-driving.** "The child behind the parked van. The car can't see them — but
> it *can* know that a van-shaped blind spot exists and slow down for it. Right now
> most systems treat what they can't see as empty road."
>
> **Search and rescue.** "Drone in a collapsed building. Knowing which voids might
> contain a survivor — versus which are solid rubble — changes where you dig."
>
> **AR/VR.** "Virtual objects that sit behind your real furniture properly, instead
> of floating through it."

> *[if energy is good]*
>
> "The honest version of the pitch is: we built object permanence. The thing babies
> figure out at eight months — that stuff doesn't stop existing when you can't see
> it. Turns out robots are still working on it."

---

## 5 — Close (2:35–3:00)

> "One number to take away. In the hidden regions — behind things — every
> conventional method scores **exactly zero**. Not 'poor'. Zero. There's nothing
> there to score, because no sensor measures through a sofa."
>
> "We're at **37%**. It's not perfect, and we're upfront that it's a hard problem
> where nobody gets a perfect answer. But it's the difference between a robot
> knowing it has a blind spot, and a robot that doesn't know what it doesn't know."
>
> *[beat, warmer]*
>
> "That's the pitch. Want to see the model actually working, or shall I show you the
> bit where it gets things wrong? The failures are honestly more interesting."

**That last line does real work:** it is disarming, it invites a follow-up, and it
signals scientific honesty — which lands well with anyone technical who wanders past.

---

## Booth survival kit

**If they ask "does it actually work?"**
> "In the regions cameras never saw, we recover about 58% of the real surface within
> 5 cm. Conventional methods recover 0% there — they don't attempt it."

**If they ask "is this AI hallucinating?"**
> "Great question, and yes — genuinely. About 40% of what it invents is wrong. So we
> never present it as measured, we keep it visually separate, and the path planner
> treats it as *risk* rather than fact. It's a system that knows it's guessing."

**If they ask "why not ChatGPT / Stable Diffusion for this?"**
> "Those are image models — this hidden space isn't in *any* image, so there's
> nothing for them to fill in. It has to happen in 3D. And an image generator gives
> you a different plausible answer every time — lovely for art, alarming for a robot
> deciding whether to drive somewhere."

**If they ask what's next:**
> "Higher resolution. Everything here trained on a laptop — we're at the point where
> resolution is the honest bottleneck, not clever tricks. We tried four clever
> tricks. They all hit the same ceiling." *(said with a smile — it is a good line)*

**If a child asks:**
> "You know how when you hide a toy under a blanket, you still know it's there? The
> robot didn't know that. We taught it."

---

## Delivery notes

- **Do not rush.** 3 minutes is generous. The hook pause is worth more than any extra sentence.
- **Point at the screen constantly.** Grey vs orange does the explaining.
- **Never say:** voxel, SDF, U-Net, TSDF, checkpoint, frontier, precision/recall.
- **Say instead:** blind spot, best guess, measured vs imagined, what it can't see.
- **Reset between visitors.** Have the before/after still on screen, not mid-video.
- **If they're clearly technical**, jump straight to §"why not Stable Diffusion" and
  the 0% vs 37% framing — that is what earns respect fastest.
