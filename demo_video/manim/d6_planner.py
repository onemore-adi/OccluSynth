"""D6 — Planner in depth (plan.md §4 Act 2). renders/d6_planner.mp4.

From costmap.npz: build the 2D cost map (colour columns by cost), overlay the A*
frontier expansion, then the 13.56 m / 244-cell risk-graded path detouring the
amber hazard field. 15.5% collision-avoidance headline.
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, Square, VGroup, Dot, FadeIn, FadeOut, Create, Line,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
)

from _manim_common import FONT, WARMWHITE, OCCLUDED, FREE, SURFACE, UNOBS, DATA
from _act2_common import txt, state_legend, section_tag, hold

_UNOBS, _FREE, _SURFACE, _OCCLUDED = 0, 1, 2, 3
STATE_COLOR = {0: UNOBS, 1: FREE, 2: SURFACE, 3: OCCLUDED}


class D6_Planner(Scene):
    def construct(self):
        d = np.load(DATA / "costmap.npz")
        state2d = d["state_2d"]; nx, ny = state2d.shape
        start = tuple(int(v) for v in d["start"]); goal = tuple(int(v) for v in d["goal"])
        occ_path = d["occ_path"]; naive = d["naive_path"]; expansion = d["expansion"]

        tag = section_tag(6, "Planner — risk-graded A*")
        legend = state_legend(0.85)
        self.add(legend)
        self.play(FadeIn(tag), run_time=0.5)

        # Map the (nx,ny) grid to screen. Downsample cells to keep mobject count sane.
        stride = 2
        cell = 6.5 / (nx / stride)
        ox, oy = -3.0, -3.2

        def cpos(i, j):
            return np.array([ox + (i / stride) * cell, oy + (j / stride) * cell, 0.0])

        head = txt("collapse z-band → colour columns by cost", 24, WARMWHITE, "BOLD").to_edge(UP, buff=1.2)
        self.play(FadeIn(head), run_time=0.5)

        # Cost-map tiles, revealed by state.
        tiles = {}
        tg = VGroup()
        for i in range(0, nx, stride):
            for j in range(0, ny, stride):
                st = int(state2d[i, j])
                sq = Square(side_length=cell * 0.95, fill_color=STATE_COLOR[st],
                            fill_opacity=(0.85 if st in (2, 3) else 0.5), stroke_width=0)
                sq.move_to(cpos(i, j))
                tiles[(i, j)] = sq; tg.add(sq)
        self.play(FadeIn(tg, lag_ratio=0.001), run_time=1.6)
        costnote = VGroup(
            txt("SURFACE → ∞", 18, SURFACE),
            txt("OCCLUDED → 1 + λ·p_occ", 18, OCCLUDED),
            txt("FREE → 1     UNOBS → 6", 18, UNOBS),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT).to_edge(RIGHT, buff=0.5)
        self.play(FadeIn(costnote), run_time=0.7)
        self.wait(0.8)

        # A* frontier expansion (subsampled).
        self.play(FadeOut(head), run_time=0.3)
        head2 = txt("8-connected A* — Euclidean heuristic", 24, WARMWHITE, "BOLD").to_edge(UP, buff=1.2)
        self.play(FadeIn(head2), run_time=0.4)
        exp = expansion[::7]
        frontier = VGroup(*[Dot(cpos(int(i), int(j)), radius=cell * 0.28, color=WARMWHITE).set_opacity(0.5)
                            for i, j in exp])
        self.play(FadeIn(frontier, lag_ratio=0.01), run_time=1.8)
        self.wait(0.4)
        self.play(FadeOut(frontier), run_time=0.4)

        # Naive straight path (red) vs risk-graded detour (green).
        npath = VGroup(*[Line(cpos(*naive[k]), cpos(*naive[k + 1]), color=SURFACE, stroke_width=4)
                         for k in range(len(naive) - 1)])
        self.play(Create(npath), run_time=1.0)
        nlab = txt("naive straight line", 18, SURFACE).next_to(costnote, DOWN, buff=0.5)
        self.play(FadeIn(nlab), run_time=0.3)
        self.wait(0.4)
        opath = VGroup(*[Line(cpos(*occ_path[k]), cpos(*occ_path[k + 1]), color=FREE, stroke_width=5)
                         for k in range(len(occ_path) - 1)])
        self.play(Create(opath), run_time=1.6)
        s_dot = Dot(cpos(*start), radius=cell * 0.5, color=FREE)
        g_dot = Dot(cpos(*goal), radius=cell * 0.5, color=OCCLUDED)
        self.play(FadeIn(s_dot), FadeIn(g_dot), run_time=0.4)
        stats = VGroup(
            txt("13.56 m · 244 cells", 26, FREE, "BOLD"),
            txt("15.5% collision-avoidance", 22, OCCLUDED),
        ).arrange(DOWN, buff=0.2).next_to(nlab, DOWN, buff=0.5)
        self.play(FadeIn(stats, shift=UP * 0.15), run_time=0.8)
        self.wait(1.6)
        hold(self)
        self.play(FadeOut(VGroup(head2, tg, costnote, nlab, npath, opath, s_dot, g_dot, stats, tag, legend)), run_time=0.7)
