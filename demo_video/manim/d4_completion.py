"""D4 — Completion in depth, Manim part (plan.md §4 Act 2). renders/d4_completion_arch.mp4.

From completer_meta.json: the 3D U-Net block diagram (14.7M), the 3 input
channels, the masked-L1 region, the metrics table (completer vs baselines), and
the training curve. (Blender renders the input-channel volumes separately.)
"""
from __future__ import annotations

import json

import numpy as np
from manim import (
    Scene, Axes, Rectangle, Line, Dot, VGroup, FadeIn, FadeOut, Create, Write,
    Arrow, UP, DOWN, LEFT, RIGHT,
)

from _manim_common import FONT, WARMWHITE, OCCLUDED, FREE, SURFACE, UNOBS, NAVY, DATA
from _act2_common import txt, state_legend, section_tag, hold


class D4_CompletionArch(Scene):
    def construct(self):
        meta = json.loads((DATA / "completer_meta.json").read_text())
        arch = meta["arch"]; curve = meta["training_curve"]; m = meta["metrics"]
        tag = section_tag(4, "Completion — 3D U-Net")
        legend = state_legend(0.85)
        self.add(legend)
        self.play(FadeIn(tag), run_time=0.5)

        # --- U-Net block diagram ---
        enc = arch["encoder_channels"]           # [32,64,128,256]
        boxes = VGroup()
        heights = [2.6, 2.0, 1.5, 1.1]
        xs = np.linspace(-5.2, -1.2, 4)
        enc_boxes = []
        for x, h, c in zip(xs, heights, enc):
            r = Rectangle(width=0.55, height=h, fill_color=NAVY, fill_opacity=0.3,
                          stroke_color=WARMWHITE, stroke_width=2).move_to([x, 0.6, 0])
            lab = txt(str(c), 18, WARMWHITE).next_to(r, DOWN, buff=0.1)
            enc_boxes.append(r); boxes.add(VGroup(r, lab))
        bott = Rectangle(width=0.55, height=0.9, fill_color=OCCLUDED, fill_opacity=0.4,
                         stroke_color=OCCLUDED, stroke_width=2).move_to([0, 0.6, 0])
        boxes.add(VGroup(bott, txt("256", 18, OCCLUDED).next_to(bott, DOWN, buff=0.1)))
        dec_boxes = []
        for x, h, c in zip(-xs[::-1], heights[::-1], enc[::-1]):
            r = Rectangle(width=0.55, height=h, fill_color=NAVY, fill_opacity=0.3,
                          stroke_color=WARMWHITE, stroke_width=2).move_to([x, 0.6, 0])
            lab = txt(str(c), 18, WARMWHITE).next_to(r, DOWN, buff=0.1)
            dec_boxes.append(r); boxes.add(VGroup(r, lab))
        title = txt(f"encoder → bottleneck → decoder   ·   {arch['param_count_m']}M params",
                    24, WARMWHITE, "BOLD").to_edge(UP, buff=1.2)
        self.play(FadeIn(title), run_time=0.5)
        self.play(FadeIn(boxes, lag_ratio=0.15), run_time=1.8)
        # skip connections
        skips = VGroup(*[Line(e.get_top(), dd.get_top(), color=FREE, stroke_width=2)
                         .set_opacity(0.6) for e, dd in zip(enc_boxes, dec_boxes[::-1])])
        skiplab = txt("skip connections", 18, FREE).next_to(boxes, UP, buff=0.05).shift(RIGHT * 0.2)
        self.play(Create(skips), FadeIn(skiplab), run_time=1.0)

        # 3 input channels feeding in
        chans = VGroup(*[Rectangle(width=0.35, height=0.35, fill_color=c, fill_opacity=0.8,
                                   stroke_width=1, stroke_color=WARMWHITE)
                         for c in (SURFACE, UNOBS, OCCLUDED)]).arrange(DOWN, buff=0.08)
        chans.next_to(enc_boxes[0], LEFT, buff=0.6)
        clab = txt("sdf · weight · p_obs", 16, UNOBS).next_to(chans, UP, buff=0.12)
        self.play(FadeIn(chans), FadeIn(clab), run_time=0.7)
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, boxes, skips, skiplab, chans, clab)), run_time=0.6)

        # --- masked-L1 + metrics table ---
        loss = txt("Loss: masked L1 over  SURFACE ∪ OCCLUDED   (UNOBSERVABLE excluded)",
                   24, WARMWHITE, "BOLD").to_edge(UP, buff=1.2)
        self.play(FadeIn(loss), run_time=0.6)

        occ = m["occluded"]
        rows = [
            ("occluded MAE (cm)", occ["no_completion"]["mae_cm"], occ["occluded_as_free"]["mae_cm"], occ["completer"]["mae_cm"], True),
            ("sign accuracy", occ["no_completion"]["sign_acc"], occ["occluded_as_free"]["sign_acc"], occ["completer"]["sign_acc"], False),
            ("completion <5cm", occ["no_completion"]["completion_ratio"], occ["occluded_as_free"]["completion_ratio"], occ["completer"]["completion_ratio"], False),
        ]
        headers = ["metric", "no-fill", "occ=free", "OccluSynth"]
        table = VGroup()
        hrow = VGroup(*[txt(h, 22, (OCCLUDED if h == "OccluSynth" else UNOBS), "BOLD") for h in headers])
        for k, cell in enumerate(hrow):
            cell.move_to([-4.5 + k * 3.0, 1.4, 0])
        table.add(hrow)
        metric_mobs = []
        for ri, (name, v0, v1, v2, is_mae) in enumerate(rows):
            y = 0.6 - ri * 0.8
            nm = txt(name, 20, WARMWHITE).move_to([-4.5, y, 0])
            fmt = (lambda v: f"{v:.1f}") if is_mae else (lambda v: f"{v:.2f}")
            c0 = txt(fmt(v0), 20, UNOBS).move_to([-1.5, y, 0])
            c1 = txt(fmt(v1), 20, UNOBS).move_to([1.5, y, 0])
            c2 = txt(fmt(v2), 22, OCCLUDED, "BOLD").move_to([4.5, y, 0])
            table.add(nm, c0, c1, c2); metric_mobs.append(c2)
        self.play(FadeIn(hrow), run_time=0.5)
        self.play(FadeIn(table[1:]), run_time=1.0)
        self.play(*[m_.animate.scale(1.15) for m_ in metric_mobs], run_time=0.5)
        self.play(*[m_.animate.scale(1 / 1.15) for m_ in metric_mobs], run_time=0.3)
        self.wait(1.2)
        self.play(FadeOut(VGroup(loss, table)), run_time=0.6)

        # --- training curve ---
        tc = txt(f"trained to val_loss {meta['best_val_loss']:.3f} @ epoch {meta['best_epoch']}",
                 24, WARMWHITE, "BOLD").to_edge(UP, buff=1.2)
        self.play(FadeIn(tc), run_time=0.5)
        ep = [c[0] for c in curve]; vl = [c[1] for c in curve]
        ax = Axes(x_range=[0, 35, 10], y_range=[0.18, 0.20, 0.01], x_length=8, y_length=3.6,
                  axis_config={"include_numbers": False, "include_tip": True, "color": UNOBS, "stroke_width": 2}).move_to(DOWN * 0.5)
        xl = txt("epoch", 18, UNOBS).next_to(ax, DOWN, buff=0.15)
        pts = [ax.c2p(e, v) for e, v in zip(ep, vl)]
        cv = VGroup(*[Line(pts[k], pts[k + 1], color=OCCLUDED, stroke_width=4) for k in range(len(pts) - 1)])
        cd = VGroup(*[Dot(p, radius=0.06, color=OCCLUDED) for p in pts])
        interim = txt("interim 64³ checkpoint — full 96³ A100 run scripted", 20, UNOBS)
        interim.next_to(ax, DOWN, buff=0.7)
        self.play(Create(ax), FadeIn(xl), run_time=0.6)
        self.play(Create(cv), FadeIn(cd), run_time=1.2)
        self.play(FadeIn(interim), run_time=0.6)
        self.wait(1.4)
        hold(self)
        self.play(FadeOut(VGroup(tc, ax, xl, cv, cd, interim, tag, legend)), run_time=0.7)
