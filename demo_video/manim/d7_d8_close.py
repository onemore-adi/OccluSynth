"""D7 validation tables + D8 honesty/roadmap columns (plan.md §4 Act 2).

  D7_Validation -> renders/d7_validation.mp4   (geometry.json + results.json)
  D8_Close      -> renders/d8_close.mp4         (built / gaps / phase-2 columns)

D8 dissolves to the reused 7A callback + 7B thesis card in the final timeline.
"""
from __future__ import annotations

import json

from manim import (
    Scene, VGroup, Rectangle, FadeIn, FadeOut, Write, Create, Line,
    UP, DOWN, LEFT, RIGHT,
)

from _manim_common import FONT, WARMWHITE, OCCLUDED, FREE, SURFACE, UNOBS, NAVY, DATA
from _act2_common import txt, state_legend, section_tag, hold


class D7_Validation(Scene):
    def construct(self):
        g = json.loads((DATA / "geometry.json").read_text())
        tag = section_tag(7, "Validation & benchmark")
        legend = state_legend(0.85)
        self.add(legend)
        self.play(FadeIn(tag), run_time=0.5)

        # Benchmark definition.
        defn = VGroup(
            txt("ScanNet-native safety benchmark  (no simulator)", 26, WARMWHITE, "BOLD"),
            txt("hidden hazard  =  OCCLUDED  ∧  GT-occupied", 22, OCCLUDED),
            txt("Metric 1: hazard awareness      Metric 2: collision-avoidance", 20, UNOBS),
        ).arrange(DOWN, buff=0.28).to_edge(UP, buff=1.1)
        self.play(FadeIn(defn, shift=DOWN * 0.1), run_time=1.0)
        self.wait(1.0)
        self.play(defn.animate.scale(0.7).to_edge(UP, buff=0.7), run_time=0.6)

        # Geometry table: surface + occluded, TSDF-only vs OccluSynth.
        s = g["surface"]; o = g["occluded"]
        rows = [
            ("surface  Chamfer-L1 (cm)", s["tsdf_only"]["chamfer_l1_cm"], s["completer"]["chamfer_l1_cm"]),
            ("surface  F@5cm", s["tsdf_only"]["fscore_5cm"], s["completer"]["fscore_5cm"]),
            ("occluded  F@5cm", o["tsdf_only"]["fscore_5cm"], o["completer"]["fscore_5cm"]),
            ("occluded  Chamfer-L1 (cm)", None, o["completer"]["chamfer_l1_cm"]),
        ]
        hdr = VGroup(txt("", 20), txt("TSDF-only", 22, UNOBS, "BOLD"),
                     txt("OccluSynth", 22, OCCLUDED, "BOLD"))
        for k, c in enumerate(hdr):
            c.move_to([-1.0 + k * 2.8, 1.2, 0])
        table = VGroup(hdr)
        for ri, (name, a, b) in enumerate(rows):
            y = 0.5 - ri * 0.7
            nm = txt(name, 20, WARMWHITE).move_to([-4.6, y, 0]).align_to([-4.6, y, 0], LEFT)
            av = txt("—" if a is None else f"{a:.2f}" if a > 1 else f"{a:.3f}", 20, UNOBS).move_to([-1.0 + 2.8, y, 0])
            bv = txt(f"{b:.2f}" if b > 1 else f"{b:.3f}", 22, OCCLUDED, "BOLD").move_to([-1.0 + 2 * 2.8, y, 0])
            table.add(nm, av, bv)
        self.play(FadeIn(table), run_time=1.2)
        stamp = VGroup(
            txt("Occl-F  0 → 32%", 22, FREE, "BOLD"),
            txt("55+ unit tests · 18 planner tests", 20, UNOBS),
            txt("7-Scenes cross-dataset: fusion runs unchanged", 20, UNOBS),
        ).arrange(DOWN, buff=0.18).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(stamp, shift=UP * 0.1), run_time=0.9)
        self.wait(1.8)
        hold(self)
        self.play(FadeOut(VGroup(defn, table, stamp, tag, legend)), run_time=0.7)


class D8_Close(Scene):
    def construct(self):
        tag = section_tag(8, "Honesty & roadmap")
        self.play(FadeIn(tag), run_time=0.5)

        cols = [
            ("Built & tested", FREE, [
                "4-state fusion + obliquity fix",
                "RANSAC depth (ARE 0.024)",
                "14.7M completer (MAE 27cm)",
                "risk-graded A* planner",
                "ScanNet safety benchmark",
            ]),
            ("Partial gaps", OCCLUDED, [
                "interim 64³ checkpoint",
                "ECE 0.42 (pre-calibration)",
                "HF publish pending",
            ]),
            ("Phase 2", UNOBS, [
                "96³ A100 training run",
                "diffusion completer",
                "semantics + active perception",
            ]),
        ]
        colmobs = VGroup()
        xs = [-4.6, 0.0, 4.6]
        for (title, col, items), x in zip(cols, xs):
            head = txt(title, 26, col, "BOLD")
            bar = Line(LEFT * 1.6, RIGHT * 1.6, color=col, stroke_width=3)
            lines = VGroup(*[txt("• " + it, 18, WARMWHITE) for it in items]).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
            block = VGroup(head, bar, lines).arrange(DOWN, buff=0.25)
            block.move_to([x, 0.2, 0])
            colmobs.add(block)
        for block in colmobs:
            self.play(FadeIn(block, shift=UP * 0.15), run_time=0.8)
            self.wait(0.3)
        self.wait(1.6)
        hold(self)
        self.play(FadeOut(VGroup(tag, colmobs)), run_time=0.8)
