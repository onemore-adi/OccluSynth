"""D0 bridge + D1 system overview (plan.md §4 Act 2).

  D0_Bridge   -> renders/d0_bridge.mp4    "open the hood" handoff
  D1_Overview -> renders/d1_overview.mp4   5-stage architecture, legend docks
"""
from __future__ import annotations

from manim import (
    Scene, FadeIn, FadeOut, Write, Create, Indicate, VGroup,
    UP, DOWN, LEFT, RIGHT,
)

from _manim_common import OCCLUDED, WARMWHITE, UNOBS
from _act2_common import (
    txt, pipeline_diagram, io_rail, state_legend, section_tag, STAGES, hold,
)


class D0_Bridge(Scene):
    def construct(self):
        l1 = txt("That's the demo.", 60, WARMWHITE, "BOLD")
        l2 = txt("Here's how every stage actually works.", 44, OCCLUDED)
        g = VGroup(l1, l2).arrange(DOWN, buff=0.5)
        self.play(FadeIn(l1, shift=UP * 0.2), run_time=1.0)
        self.wait(0.4)
        self.play(FadeIn(l2, shift=UP * 0.1), run_time=1.0)
        self.wait(1.4)
        # Resolve into the pipeline diagram.
        diagram, _ = pipeline_diagram(scale=0.62)
        diagram.move_to(DOWN * 0.2)
        self.play(g.animate.scale(0.5).to_edge(UP, buff=0.6), run_time=0.8)
        self.play(Create(diagram), run_time=1.6)
        self.wait(1.4)
        hold(self)
        self.play(FadeOut(VGroup(g, diagram)), run_time=0.7)


class D1_Overview(Scene):
    def construct(self):
        tag = section_tag(1, "System overview")
        diagram, boxes = pipeline_diagram(scale=0.9)
        diagram.move_to(UP * 0.3)
        rail = io_rail(diagram)

        self.play(FadeIn(tag), run_time=0.5)
        self.play(Create(diagram), run_time=1.5)
        self.play(FadeIn(rail), run_time=0.8)

        # Light each stage left -> right.
        for st in STAGES:
            box = boxes[st][0]
            self.play(
                box.animate.set_fill(OCCLUDED, opacity=0.35).set_stroke(OCCLUDED),
                Indicate(boxes[st][1], color=OCCLUDED, scale_factor=1.15),
                run_time=0.7,
            )
            self.wait(0.25)

        # Dock the 4-state legend bottom-left (stays for the rest of Act 2).
        legend = state_legend()
        legend_title = txt("4-state map (throughout)", 16, UNOBS)
        legend_title.next_to(legend, UP, buff=0.12).align_to(legend, LEFT)
        self.play(FadeIn(legend, shift=RIGHT * 0.2), FadeIn(legend_title), run_time=1.0)
        self.wait(2.0)
        hold(self)
        self.play(FadeOut(VGroup(tag, diagram, rail, legend, legend_title)), run_time=0.8)
