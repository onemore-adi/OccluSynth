"""D2 — Perception in depth (plan.md §4 Act 2). renders/d2_perception.mp4.

From scale_fit.npz (real cached VGGT depth + fits): the raw-vs-GT scale gap, the
live RANSAC affine fit on 500 anchors (inliers green / outliers red), the 4-method
ARE bar chart, the 10-scene strip, and the noise-robustness curve.
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, Axes, Dot, Line, Rectangle, VGroup, FadeIn, FadeOut, Create, Write,
    UP, DOWN, LEFT, RIGHT,
)

from _manim_common import FONT, WARMWHITE, OCCLUDED, FREE, SURFACE, UNOBS, NAVY, DATA
from _act2_common import txt, state_legend, section_tag, hold


def _axes(x_range, y_range, w, h):
    return Axes(x_range=x_range, y_range=y_range, x_length=w, y_length=h,
                axis_config={"include_numbers": False, "include_tip": True,
                             "color": UNOBS, "stroke_width": 2})


class D2_Perception(Scene):
    def construct(self):
        d = np.load(DATA / "scale_fit.npz", allow_pickle=True)
        tag = section_tag(2, "Perception — metric grounding")
        legend = state_legend(0.85)
        self.add(legend)
        self.play(FadeIn(tag), run_time=0.5)

        # --- Beat 1: frozen VGGT + scale gap histogram ---
        frozen = txt("VGGT-Omega  (frozen, no fine-tuning)", 26, WARMWHITE, "BOLD").to_edge(UP, buff=1.3)
        self.play(FadeIn(frozen, shift=DOWN * 0.1), run_time=0.6)
        ax = _axes([0, 3.5, 1], [0, 1.05, 1], 8.5, 3.2).move_to(DOWN * 0.3)
        edges = d["hist_edges"]; ph = d["pred_hist"] / d["pred_hist"].max(); gh = d["gt_hist"] / d["gt_hist"].max()
        xlab = txt("depth (m)", 20, UNOBS).next_to(ax, DOWN, buff=0.2)
        self.play(Create(ax), FadeIn(xlab), run_time=0.8)

        def bars(hist, color, op):
            g = VGroup()
            for k in range(len(hist)):
                x0, x1 = edges[k], edges[k + 1]
                hgt = float(hist[k])
                if hgt <= 0:
                    continue
                p0 = ax.c2p(x0, 0); p1 = ax.c2p(x1, hgt)
                r = Rectangle(width=abs(p1[0] - p0[0]), height=abs(p1[1] - p0[1]),
                              fill_color=color, fill_opacity=op, stroke_width=0)
                r.move_to((p0 + p1) / 2)
                g.add(r)
            return g
        raw = bars(ph, SURFACE, 0.65); gt = bars(gh, FREE, 0.55)
        rlab = txt("raw VGGT ≈ 0.2", 20, SURFACE).next_to(ax, UP, buff=0.05).shift(LEFT * 3)
        glab = txt("GT 1.3–2.9 m", 20, FREE).next_to(ax, UP, buff=0.05).shift(RIGHT * 2)
        self.play(FadeIn(raw), FadeIn(rlab), run_time=0.7)
        self.play(FadeIn(gt), FadeIn(glab), run_time=0.7)
        self.wait(1.2)
        self.play(FadeOut(VGroup(ax, xlab, raw, gt, rlab, glab, frozen)), run_time=0.6)

        # --- Beat 2: RANSAC scatter fit ---
        head = txt("500 stratified anchors → per-frame affine  d = a·d_pred + b", 26,
                   WARMWHITE, "BOLD").to_edge(UP, buff=1.3)
        self.play(FadeIn(head), run_time=0.5)
        dp = d["d_pred"]; dg = d["d_gt"]; inl = d["inliers"]; a = float(d["fit_a"]); b = float(d["fit_b"])
        xmax = float(dp.max()) * 1.1; ymax = float(dg.max()) * 1.1
        sax = _axes([0, xmax, xmax / 4], [0, ymax, ymax / 4], 6.6, 4.6).move_to(DOWN * 0.4 + LEFT * 1.2)
        xl = txt("d_pred", 20, UNOBS).next_to(sax, DOWN, buff=0.15)
        yl = txt("d_gt (m)", 20, UNOBS).rotate(np.pi / 2).next_to(sax, LEFT, buff=0.15)
        self.play(Create(sax), FadeIn(xl), FadeIn(yl), run_time=0.8)
        dots = VGroup()
        for x, y, ok in zip(dp, dg, inl):
            dots.add(Dot(sax.c2p(float(x), float(y)), radius=0.03,
                         color=(FREE if ok else SURFACE)).set_opacity(0.0))
        self.add(dots)
        self.play(*[dd.animate.set_opacity(0.8) for dd in dots], run_time=1.0)
        # RANSAC line fits in (rotate from flat mean line to the fitted line).
        line = Line(sax.c2p(0, a * 0 + b), sax.c2p(xmax, a * xmax + b),
                    color=OCCLUDED, stroke_width=5)
        flat = Line(sax.c2p(0, dg.mean()), sax.c2p(xmax, dg.mean()), color=OCCLUDED, stroke_width=5)
        self.play(Create(flat), run_time=0.5)
        self.play(flat.animate.become(line), run_time=1.3)
        info = VGroup(
            txt(f"a = {a:.2f}", 26, OCCLUDED, "BOLD"),
            txt(f"b = {b:.3f} m", 24, OCCLUDED),
            txt(f"{int(inl.sum())}/{len(inl)} inliers", 22, FREE),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT).next_to(sax, RIGHT, buff=0.6)
        self.play(FadeIn(info, shift=LEFT * 0.2), run_time=0.8)
        self.wait(1.4)
        self.play(FadeOut(VGroup(head, sax, xl, yl, dots, flat, info)), run_time=0.6)

        # --- Beat 3: 4-method ARE bar chart ---
        head2 = txt("4 fits benchmarked — RANSAC wins", 26, WARMWHITE, "BOLD").to_edge(UP, buff=1.3)
        self.play(FadeIn(head2), run_time=0.5)
        methods = [str(m) for m in d["methods"]]; are = d["method_are"]
        bax = _axes([0, 4, 1], [0, 0.08, 0.02], 8.5, 3.6).move_to(DOWN * 0.4)
        yl2 = txt("mean ARE ↓", 20, UNOBS).rotate(np.pi / 2).next_to(bax, LEFT, buff=0.1)
        self.play(Create(bax), FadeIn(yl2), run_time=0.6)
        barg = VGroup(); labs = VGroup()
        for i, (m, v) in enumerate(zip(methods, are)):
            best = (i == len(methods) - 1)
            col = OCCLUDED if best else NAVY
            top = bax.c2p(i + 0.5, float(v)); base = bax.c2p(i + 0.5, 0)
            r = Rectangle(width=1.1, height=0.01, fill_color=col, fill_opacity=0.85, stroke_width=0)
            r.move_to(base); barg.add(r)
            labs.add(txt(m.replace("PerFrame", "PF"), 16, WARMWHITE).next_to(base, DOWN, buff=0.12))
            labs.add(txt(f"{v:.3f}", 18, col, "BOLD").move_to(top + UP * 0.2))
        self.add(labs)
        anims = []
        for i, (r, v) in enumerate(zip(barg, are)):
            hgt = abs(bax.c2p(0, float(v))[1] - bax.c2p(0, 0)[1])
            target = r.copy().stretch_to_fit_height(hgt).move_to(bax.c2p(i + 0.5, float(v) / 2))
            anims.append(r.animate.become(target))
        self.play(FadeIn(labs), run_time=0.4)
        self.play(*anims, run_time=1.4)
        self.wait(1.2)
        self.play(FadeOut(VGroup(head2, bax, yl2, barg, labs)), run_time=0.6)

        # --- Beat 4: multi-scene strip + noise curve ---
        head3 = txt("10 scenes · robust to sensor noise", 26, WARMWHITE, "BOLD").to_edge(UP, buff=1.3)
        self.play(FadeIn(head3), run_time=0.5)
        ms = d["multiscene_are"]
        strip = VGroup()
        for i, v in enumerate(ms):
            sq = Rectangle(width=0.5, height=0.5, fill_color=FREE, fill_opacity=0.85, stroke_width=0)
            strip.add(VGroup(sq, txt(f"{v:.3f}", 15, WARMWHITE).move_to(sq)))
        strip.arrange(RIGHT, buff=0.12).move_to(UP * 1.2)
        striplab = txt(f"per-scene ARE (mean {ms.mean():.3f}, 0/10 degrade)", 20, UNOBS).next_to(strip, UP, buff=0.2)
        self.play(FadeIn(striplab), run_time=0.4)
        self.play(FadeIn(strip, lag_ratio=0.1), run_time=1.2)

        sig = d["noise_sigmas"]; nare = d["noise_are"]
        nax = _axes([0, 0.26, 0.05], [0, 0.06, 0.02], 7.5, 2.4).move_to(DOWN * 1.5)
        nxl = txt("depth noise σ (m)", 18, UNOBS).next_to(nax, DOWN, buff=0.12)
        pts = [nax.c2p(float(s), float(v)) for s, v in zip(sig, nare)]
        curve = VGroup(*[Line(pts[k], pts[k + 1], color=OCCLUDED, stroke_width=4) for k in range(len(pts) - 1)])
        cdots = VGroup(*[Dot(p, radius=0.05, color=OCCLUDED) for p in pts])
        self.play(Create(nax), FadeIn(nxl), run_time=0.6)
        self.play(Create(curve), FadeIn(cdots), run_time=1.2)
        self.wait(1.4)
        hold(self)
        self.play(FadeOut(VGroup(head3, strip, striplab, nax, nxl, curve, cdots, tag, legend)), run_time=0.7)
