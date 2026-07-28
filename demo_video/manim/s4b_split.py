"""Scene 4B — split vs baseline (plan.md §4). renders/s4b_split.mp4.

Synchronized TSDF-hole (left) vs OccluSynth-filled (right), with the number
countdown 45.3 -> 27.1 and completeness 0 -> 32%. Rendered abstractly in Manim
(no Blender needed for this beat): the occluded gap in the TSDF surface is filled
with amber, then resolves to solid red on the OccluSynth side.
"""
from __future__ import annotations

from manim import (
    Scene, Text, VGroup, Rectangle, ValueTracker,
    FadeIn, FadeOut, UP, DOWN, LEFT, RIGHT,
)

from _manim_common import FONT, WARMWHITE, SURFACE, OCCLUDED, UNOBS, VOID, value_text


def _label(text, size=30, color=WARMWHITE, weight="BOLD"):
    return Text(text, font=FONT, font_size=size, color=color, weight=weight)


def _surface(with_fill: bool):
    """A stylised measured surface (red) with an occluded gap. If with_fill, the
    gap is completed (amber); otherwise it is left as a void hole."""
    seg_w, seg_h, gap_w = 0.9, 0.9, 0.9
    left = Rectangle(width=seg_w * 1.6, height=seg_h, color=SURFACE,
                     fill_color=SURFACE, fill_opacity=1.0, stroke_width=0)
    right = Rectangle(width=seg_w * 1.6, height=seg_h, color=SURFACE,
                      fill_color=SURFACE, fill_opacity=1.0, stroke_width=0)
    gap = Rectangle(width=gap_w, height=seg_h, stroke_width=0,
                    fill_color=(OCCLUDED if with_fill else VOID),
                    fill_opacity=(1.0 if with_fill else 0.0))
    row = VGroup(left, gap, right).arrange(RIGHT, buff=0.0)
    return row, gap


class S4B_Split(Scene):
    def construct(self):
        title = _label("Completion fills what was occluded", 40).to_edge(UP, buff=0.55)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.6)

        left_surface, left_gap = _surface(with_fill=False)
        right_surface, right_gap = _surface(with_fill=True)
        left_surface.move_to(LEFT * 3.4 + UP * 0.6)
        right_surface.move_to(RIGHT * 3.4 + UP * 0.6)

        left_lab = _label("TSDF only", 28, UNOBS).next_to(left_surface, UP, buff=0.35)
        right_lab = _label("OccluSynth", 28, OCCLUDED).next_to(right_surface, UP, buff=0.35)

        self.play(
            FadeIn(left_surface), FadeIn(left_lab),
            FadeIn(right_surface), FadeIn(right_lab),
            run_time=0.8,
        )
        # The occluded gap fills in (amber) on the OccluSynth side.
        right_gap.set_opacity(0.0)
        self.play(right_gap.animate.set_opacity(1.0), run_time=0.9)

        # Number countdowns (plan.md §4).
        err = ValueTracker(45.3)
        comp = ValueTracker(0.0)
        err_at = LEFT * 3.0 + DOWN * 2.4
        comp_at = RIGHT * 3.0 + DOWN * 2.4
        err_num = value_text(err, lambda v: f"{v:.1f}", font_size=56,
                             color=SURFACE, at=err_at)
        comp_num = value_text(comp, lambda v: f"{v:.0f}%", font_size=56,
                              color=OCCLUDED, at=comp_at)
        err_cap = Text("surface error (mm)", font=FONT, font_size=24,
                       color=UNOBS).move_to(err_at + DOWN * 0.55)
        comp_cap = Text("occluded completeness", font=FONT, font_size=24,
                        color=UNOBS).move_to(comp_at + DOWN * 0.55)
        self.add(err_num, comp_num, err_cap, comp_cap)

        self.play(err.animate.set_value(27.1), comp.animate.set_value(32.0), run_time=1.6)
        err_num.clear_updaters()
        comp_num.clear_updaters()
        self.wait(1.4)
        self.play(FadeOut(VGroup(title, left_surface, right_surface, left_lab,
                                 right_lab, err_num, comp_num, err_cap, comp_cap)),
                  run_time=0.7)
