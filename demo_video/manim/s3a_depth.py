"""Scene 3A — depth compare (plan.md §4). renders/s3a_depth.mp4.

Three depth PNGs (exported in Phase 1) wipe in raw -> calibrated -> GT, ~200
anchor dots ripple over the calibrated panel (the 500 stratified anchors that
drive the per-frame fit), and a counter ticks up to 2.4% mean abs. rel. error.
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, ImageMobject, Text, VGroup, Group, Dot,
    FadeIn, FadeOut, ValueTracker, LaggedStart, Indicate,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
)

from _manim_common import FONT, WARMWHITE, OCCLUDED, FREE, UNOBS, DATA, value_text


def _label(text, color=WARMWHITE):
    return Text(text, font=FONT, font_size=30, color=color, weight="BOLD")


class S3A_Depth(Scene):
    def construct(self):
        panels = [
            ("VGGT raw", DATA / "depth_vggt_raw.png", UNOBS),
            ("Calibrated", DATA / "depth_vggt_calibrated.png", OCCLUDED),
            ("Ground truth", DATA / "depth_gt.png", FREE),
        ]
        imgs, labels = [], []
        for name, path, col in panels:
            im = ImageMobject(str(path))
            im.height = 3.6
            lab = _label(name, col)
            imgs.append(im)
            labels.append(lab)

        row = Group(*imgs).arrange(RIGHT, buff=0.5).move_to(DOWN * 0.4)
        for im, lab in zip(imgs, labels):
            lab.next_to(im, UP, buff=0.25)

        title = _label("Metric-grounded depth", WARMWHITE).scale(1.15).to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.6)

        # Wipe the three panels in sequentially.
        for im, lab in zip(imgs, labels):
            im.shift(LEFT * 0.6)
            self.play(FadeIn(im, shift=RIGHT * 0.6), FadeIn(lab), run_time=0.7)

        # Ripple anchor dots over the calibrated (middle) panel.
        mid = imgs[1]
        rng = np.random.default_rng(3)
        dots = VGroup()
        w, h = mid.width * 0.9, mid.height * 0.9
        for _ in range(200):
            p = mid.get_center() + np.array([
                (rng.random() - 0.5) * w, (rng.random() - 0.5) * h, 0.0])
            dots.add(Dot(p, radius=0.018, color=WARMWHITE).set_opacity(0.0))
        self.add(dots)
        self.play(
            LaggedStart(*[d.animate.set_opacity(0.9) for d in dots],
                        lag_ratio=0.004),
            run_time=1.2,
        )
        self.play(LaggedStart(*[Indicate(d, scale_factor=2.2, color=OCCLUDED) for d in dots],
                              lag_ratio=0.003), run_time=1.2)

        # Error counter ticks up to 2.4% (plan.md §4).
        tracker = ValueTracker(0.0)
        anchor = DOWN * 3.0
        num = value_text(tracker, lambda v: f"{v:.1f}%", font_size=64,
                         color=OCCLUDED, at=anchor)
        caption = Text("mean abs. rel. error", font=FONT, font_size=26, color=UNOBS)
        caption.move_to(anchor + DOWN * 0.5)
        self.add(num, caption)
        self.play(tracker.animate.set_value(2.4), run_time=1.4)
        num.clear_updaters()
        self.wait(1.2)
        self.play(FadeOut(Group(title, row, *labels, dots, num, caption)), run_time=0.7)
