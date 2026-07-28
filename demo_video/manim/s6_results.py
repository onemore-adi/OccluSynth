"""Scene 6 — results + benchmark (plan.md §4).

  S6A_Dots    -> renders/s6a_dots.mp4     10x10 grid, 21 light green one-by-one, 0->21%
  S6B_Numbers -> renders/s6b_numbers.mp4  Chamfer 3.1->1.8, F 0->32%, 55+ tests, cut to black

6A reads the real 21/100 hazard flags exported in Phase 1 (benchmark_dots.npz).
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, Text, VGroup, Dot, ValueTracker,
    FadeIn, FadeOut, Flash, LaggedStart, UP, DOWN, LEFT, RIGHT, ORIGIN,
)

from _manim_common import FONT, WARMWHITE, FREE, UNOBS, OCCLUDED, DATA, value_text


def _txt(text, size, color=WARMWHITE, weight="NORMAL"):
    return Text(text, font=FONT, font_size=size, color=color, weight=weight)


class S6A_Dots(Scene):
    def construct(self):
        data = np.load(DATA / "benchmark_dots.npz")
        flags = data["flags"]              # (100,) bool, 21 True
        order = data["order"]              # (100,) reveal order
        awareness = float(data["awareness_completer"]) * 100.0   # 21.32

        title = _txt("Hazards the robot now anticipates", 38).to_edge(UP, buff=0.6)

        dots = []
        for idx in range(100):
            r, c = divmod(idx, 10)
            p = np.array([c - 4.5, 4.5 - r, 0.0]) * 0.62
            dots.append(Dot(p, radius=0.14, color=UNOBS).set_opacity(0.5))
        grid = VGroup(*dots).move_to(ORIGIN + DOWN * 0.2)

        tracker = ValueTracker(0.0)
        counter_at = RIGHT * 4.6 + UP * 0.4
        counter = value_text(tracker, lambda v: f"{v:.0f}%", font_size=64,
                             color=FREE, at=counter_at)
        c_cap = _txt("hazard awareness", 24, UNOBS).move_to(counter_at + DOWN * 0.55)
        baseline = _txt("baseline  0%", 26, UNOBS).move_to(counter_at + DOWN * 1.4)

        self.play(FadeIn(title, shift=DOWN * 0.2), FadeIn(grid), run_time=0.8)
        self.add(counter, c_cap, baseline)

        # Light up the 21 flagged dots one-by-one, in the exported order.
        flagged = [i for i in order if flags[i]]
        anims = []
        for i in flagged:
            anims.append(dots[i].animate.set_color(FREE).set_opacity(1.0))
        self.play(
            LaggedStart(*anims, lag_ratio=0.6),
            tracker.animate.set_value(round(awareness)),
            run_time=3.0,
        )
        counter.clear_updaters()
        self.play(Flash(counter, color=FREE, line_length=0.25), run_time=0.6)
        self.wait(1.2)
        self.play(FadeOut(VGroup(title, grid, counter, c_cap, baseline)), run_time=0.7)


class S6B_Numbers(Scene):
    def construct(self):
        chamfer = ValueTracker(3.1)
        fscore = ValueTracker(0.0)

        ch_at = LEFT * 3.2 + UP * 0.8
        f_at = RIGHT * 3.2 + UP * 0.8
        ch_num = value_text(chamfer, lambda v: f"{v:.1f}", font_size=72,
                            color=OCCLUDED, at=ch_at)
        f_num = value_text(fscore, lambda v: f"{v:.0f}%", font_size=72,
                           color=FREE, at=f_at)
        ch_cap = _txt("Chamfer distance (cm)", 26, UNOBS).move_to(ch_at + DOWN * 0.7)
        f_cap = _txt("F-score gain", 26, UNOBS).move_to(f_at + DOWN * 0.7)
        tests = _txt("55+ tests", 60, WARMWHITE, weight="BOLD").move_to(DOWN * 1.9)

        self.add(ch_num, f_num, ch_cap, f_cap)
        self.play(FadeIn(ch_cap), FadeIn(f_cap), run_time=0.6)
        self.play(chamfer.animate.set_value(1.8), fscore.animate.set_value(32.0),
                  run_time=1.6)
        ch_num.clear_updaters()
        f_num.clear_updaters()
        self.wait(0.6)
        self.play(FadeIn(tests, scale=1.1), run_time=0.8)
        self.wait(1.4)
        # Cut to black (plan.md §4).
        self.play(FadeOut(VGroup(ch_num, f_num, ch_cap, f_cap, tests)), run_time=0.5)
        self.wait(0.4)
