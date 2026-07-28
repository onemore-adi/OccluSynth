# Samsung-styled cards for the 5-min hackathon cut.
# Render (from demo_video/):  ./.venv-manim/bin/manim -q h --fps 24 --format mp4 -o <id> manim/hb_cards.py <Scene>
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from manim import *
from config.tokens_hb import (BG, WHITE, GREY, GREY_DIM, BLUE, AMBER,
                              FREE, SURFACE, UNOBS, FONT_HEAD, FONT_BOLD, FONT_BODY)

config.background_color = BG

EASE = rate_functions.ease_out_cubic


def spaced(s, gap=" "):
    """Letter-spaced kicker text (Pango has no tracking in manim Text)."""
    return gap.join(list(s))


def kicker(text, color=GREY_DIM, size=21):
    return Text(spaced(text.upper()), font=FONT_BODY, weight=SEMIBOLD,
                font_size=size, color=color)


def pill(text, fill=BLUE, fg=WHITE, size=23, pad_x=0.34, pad_y=0.16):
    t = Text(text, font=FONT_BOLD, font_size=size, color=fg)
    box = RoundedRectangle(corner_radius=(t.height + 2 * pad_y) / 2,
                           width=t.width + 2 * pad_x, height=t.height + 2 * pad_y,
                           fill_color=fill, fill_opacity=1.0, stroke_width=0)
    box.move_to(t)
    return VGroup(box, t)


def rise(m, dy=0.28, t=1.1):
    """Samsung-style entrance: fade + gentle rise."""
    m.shift(DOWN * dy)
    return AnimationGroup(m.animate(rate_func=EASE, run_time=t).shift(UP * dy),
                          FadeIn(m, rate_func=rate_functions.ease_out_sine, run_time=t))


class _Card(Scene):
    HOLD = 3.0

    def hold(self, extra=0.0):
        self.wait(self.HOLD + extra)


# ----------------------------------------------------------------------------- ACT 1
class HBTitle(_Card):
    def construct(self):
        word = Text(spaced("OCCLUSYNTH", " "), font=FONT_HEAD, font_size=108, color=WHITE)
        rule = Rectangle(width=1.5, height=0.055, fill_color=AMBER, fill_opacity=1, stroke_width=0)
        sub = kicker("Occlusion-aware 3D reconstruction", color=GREY, size=24)
        tag = Text("Reconstructing what the cameras never saw.",
                   font=FONT_BODY, font_size=27, color=GREY_DIM)
        word.move_to(UP * 0.55)
        rule.next_to(word, DOWN, buff=0.5)
        sub.next_to(rule, DOWN, buff=0.5)
        tag.next_to(sub, DOWN, buff=0.65)

        self.play(rise(word, t=1.4))
        self.play(rule.animate(rate_func=EASE, run_time=0.9).stretch_to_fit_width(4.6),
                  FadeIn(rule, run_time=0.5))
        self.play(rise(sub, dy=0.18, t=0.9))
        self.wait(0.4)
        self.play(rise(tag, dy=0.18, t=0.9))
        self.hold(2.6)


class HBBlind(_Card):
    def construct(self):
        k = kicker("The problem")
        h = Text("A robot's world ends at the\nfirst surface it sees.",
                 font=FONT_HEAD, font_size=60, color=WHITE,
                 line_spacing=1.05, should_center=True)
        s = Text("Behind the sofa. Under the table. Past the doorway — nothing.",
                 font=FONT_BODY, font_size=27, color=GREY)
        k.move_to(UP * 1.85); h.move_to(UP * 0.35); s.move_to(DOWN * 1.35)
        self.play(FadeIn(k, run_time=0.7))
        self.play(rise(h, t=1.2))
        self.wait(0.3)
        self.play(rise(s, dy=0.18, t=0.9))
        self.hold(1.9)


class HBZero(_Card):
    def construct(self):
        n = Text("0%", font=FONT_HEAD, font_size=190, color=WHITE)
        s1 = Text("of hidden geometry is recovered by conventional reconstruction.",
                  font=FONT_BODY, font_size=28, color=GREY)
        s2 = Text("Not a low score — a structural zero. No sensor measures behind a surface.",
                  font=FONT_BODY, font_size=24, color=GREY_DIM)
        n.move_to(UP * 0.85); s1.move_to(DOWN * 1.1); s2.move_to(DOWN * 1.75)
        self.play(rise(n, dy=0.35, t=1.3))
        self.play(rise(s1, dy=0.18, t=0.9))
        self.play(rise(s2, dy=0.14, t=0.8))
        self.hold(2.0)


class HBThesis(_Card):
    def construct(self):
        k = kicker("Occlusynth")
        h1 = Text("We reconstruct the geometry", font=FONT_HEAD, font_size=56, color=WHITE)
        h2 = Text("no camera ever captured.", font=FONT_HEAD, font_size=56, color=AMBER)
        s = Text("And we tell you exactly how much of the map is imagined — and how sure we are.",
                 font=FONT_BODY, font_size=25, color=GREY)
        k.move_to(UP * 1.9); h1.move_to(UP * 0.62); h2.next_to(h1, DOWN, buff=0.18)
        s.move_to(DOWN * 1.5)
        self.play(FadeIn(k, run_time=0.7))
        self.play(rise(h1, t=1.1))
        self.play(rise(h2, t=1.1))
        self.wait(0.2)
        self.play(rise(s, dy=0.16, t=0.9))
        self.hold(1.6)


# ----------------------------------------------------------------------------- ACT 2
class HBPipeline(_Card):
    def construct(self):
        k = kicker("The pipeline")
        h = Text("From a handful of photos to a map\nthat knows what it can't see.",
                 font=FONT_HEAD, font_size=54, color=WHITE, line_spacing=1.08, should_center=True)
        s = Text("Six stages. One pass. No depth sensor.", font=FONT_BODY, font_size=27, color=GREY)
        k.move_to(UP * 1.95); h.move_to(UP * 0.3); s.move_to(DOWN * 1.45)
        self.play(FadeIn(k, run_time=0.7))
        self.play(rise(h, t=1.2))
        self.play(rise(s, dy=0.16, t=0.9))
        self.hold(1.4)


def section_scene(name, num, title, sub_lines):
    class _S(_Card):
        def construct(self):
            p = pill(num)
            h = Text(title, font=FONT_HEAD, font_size=64, color=WHITE)
            s = Text(sub_lines, font=FONT_BODY, font_size=26, color=GREY,
                     line_spacing=1.25, should_center=True)
            p.move_to(UP * 1.7); h.move_to(UP * 0.45); s.move_to(DOWN * 1.0)
            self.play(FadeIn(p, scale=0.9, run_time=0.7))
            self.play(rise(h, t=1.1))
            self.play(rise(s, dy=0.16, t=0.9))
            self.hold(0.9)
    _S.__name__ = name
    _S.__qualname__ = name
    return _S


HBSec01 = section_scene("HBSec01", "01 · Capture", "Forty photographs.",
                        "A single handheld RGB sweep of a real apartment — ScanNet scene0000.\nNo lidar. No depth sensor. Just colour images and camera poses.")
HBSec02 = section_scene("HBSec02", "02 · Perception", "Geometry from pixels.",
                        "VGGT — a frozen 3D foundation model — predicts dense depth for every frame\nin one forward pass. Sparse anchors pin its relative depth to metric scale.")
HBSec03 = section_scene("HBSec03", "03 · Fusion", "Every pixel becomes evidence.",
                        "Calibrated depth is ray-cast into a shared voxel grid —\ncarving out free space, marking measured surfaces.")
HBSec04 = section_scene("HBSec04", "04 · Completion", "A network imagines the rest.",
                        "A 3D U-Net, trained on how real rooms are built, continues the floor\nunder the table and closes the back of the couch — in amber.")
HBSec05 = section_scene("HBSec05", "05 · The mesh", "From voxels to solid geometry.",
                        "Marching cubes turns the completed grid into a single watertight mesh —\ngrey where cameras measured, amber where OccluSynth imagined.")
HBSec06 = section_scene("HBSec06", "06 · Trust & plan", "A map that knows how sure it is.",
                        "Every predicted voxel carries a confidence. The planner prices risk\ninstead of trusting a guess.")


class HBStates(_Card):
    def construct(self):
        k = kicker("One grid · four kinds of space")
        k.move_to(UP * 2.6)
        cols = [
            (FREE,    "FREE",         "seen empty"),
            (SURFACE, "SURFACE",      "measured solid"),
            (AMBER,   "OCCLUDED",     "hidden, in view"),
            (UNOBS,   "UNOBSERVED",   "outside all views"),
        ]
        groups = VGroup()
        for c, name, desc in cols:
            chip = RoundedRectangle(corner_radius=0.16, width=0.9, height=0.9,
                                    fill_color=c, fill_opacity=1, stroke_width=0)
            nm = Text(name, font=FONT_BOLD, font_size=25, color=WHITE)
            de = Text(desc, font=FONT_BODY, font_size=20, color=GREY)
            g = VGroup(chip, nm, de).arrange(DOWN, buff=0.28)
            groups.add(g)
        groups.arrange(RIGHT, buff=1.15).move_to(UP * 0.55)

        ring = RoundedRectangle(corner_radius=0.24, width=groups[2].width + 0.55,
                                height=groups[2].height + 0.55,
                                stroke_color=AMBER, stroke_width=3, fill_opacity=0)
        ring.move_to(groups[2])
        cap1 = Text("We predict only the occluded space —", font=FONT_BODY, font_size=27, color=WHITE)
        cap2 = Text("and never guess where we have no right to.", font=FONT_BODY, font_size=27, color=GREY)
        cap1.move_to(DOWN * 1.75); cap2.next_to(cap1, DOWN, buff=0.18)

        self.play(FadeIn(k, run_time=0.7))
        self.play(LaggedStart(*[rise(g, dy=0.2, t=0.8) for g in groups], lag_ratio=0.18))
        self.wait(0.4)
        self.play(Create(ring, run_time=1.0))
        self.play(rise(cap1, dy=0.14, t=0.8), rise(cap2, dy=0.14, t=0.8))
        self.hold(1.6)


# ----------------------------------------------------------------------------- ACT 3
class HBWhy(_Card):
    def construct(self):
        k = kicker("Why it matters")
        h = Text("Machines that act\nbefore they can look.", font=FONT_HEAD, font_size=62,
                 color=WHITE, line_spacing=1.05, should_center=True)
        k.move_to(UP * 1.9); h.move_to(DOWN * 0.05)
        self.play(FadeIn(k, run_time=0.7))
        self.play(rise(h, t=1.2))
        self.hold(1.3)


class HBMatters(_Card):
    def construct(self):
        lines = [
            Text("A warehouse robot that doesn't circle the shelf.",
                 font=FONT_BOLD, font_size=40, color=WHITE),
            Text("A rescue drone that maps behind the rubble.",
                 font=FONT_BOLD, font_size=40, color=WHITE),
            Text("The child behind the parked car.",
                 font=FONT_BOLD, font_size=40, color=AMBER),
        ]
        lines[0].move_to(UP * 1.25)
        lines[1].move_to(UP * 0.0)
        lines[2].move_to(DOWN * 1.25)
        self.play(rise(lines[0], t=1.0)); self.wait(1.0)
        self.play(rise(lines[1], t=1.0)); self.wait(1.0)
        self.play(rise(lines[2], t=1.2)); self.wait(1.8)
        hero = Text("Object permanence, for machines.", font=FONT_HEAD, font_size=58, color=WHITE)
        self.play(FadeOut(VGroup(*lines), run_time=0.8))
        self.play(rise(hero, t=1.2))
        self.hold(1.4)


class HBEnd(_Card):
    def construct(self):
        word = Text(spaced("OCCLUSYNTH", " "), font=FONT_HEAD, font_size=84, color=WHITE)
        rule = Rectangle(width=3.8, height=0.05, fill_color=AMBER, fill_opacity=1, stroke_width=0)
        l1 = Text("The future of mapping isn't seeing more —", font=FONT_BODY, font_size=28, color=GREY)
        l2 = Text("it's knowing what you cannot see.", font=FONT_BOLD, font_size=32, color=AMBER)
        word.move_to(UP * 0.8); rule.next_to(word, DOWN, buff=0.45)
        l1.next_to(rule, DOWN, buff=0.55); l2.next_to(l1, DOWN, buff=0.2)
        self.play(rise(word, t=1.3))
        self.play(FadeIn(rule, run_time=0.7))
        self.play(rise(l1, dy=0.16, t=0.9))
        self.play(rise(l2, dy=0.16, t=1.0))
        self.hold(4.2)
