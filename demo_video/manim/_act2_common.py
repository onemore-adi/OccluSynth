"""Shared Act-2 components (plan.md §4 Act 2): the 5-stage pipeline diagram and the
persistent 4-state legend. Reused across D0–D8 so colours/structure never need
re-explaining. All colours from tokens via _manim_common.
"""
from __future__ import annotations

from manim import (
    VGroup, Rectangle, RoundedRectangle, Text, Arrow, Square, Line,
    RIGHT, LEFT, UP, DOWN, ORIGIN,
)

from _manim_common import FONT, WARMWHITE, NAVY, FREE, SURFACE, OCCLUDED, UNOBS, VOID

STAGES = ["Perception", "Fusion", "Completion", "Uncertainty", "Planner"]

STATE_LEGEND = [
    (FREE, "FREE", "seen empty"),
    (SURFACE, "SURFACE", "measured solid"),
    (OCCLUDED, "OCCLUDED", "imagined"),
    (UNOBS, "UNOBS", "no evidence"),
]


def txt(s, size, color=WARMWHITE, weight="NORMAL"):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight)


def pipeline_diagram(scale=1.0, active=None):
    """5 stage boxes left->right with arrows; returns (VGroup, {stage: box}).
    `active` optionally highlights one stage in amber."""
    boxes = {}
    row = VGroup()
    for st in STAGES:
        col = OCCLUDED if st == active else NAVY
        box = RoundedRectangle(width=2.15, height=1.0, corner_radius=0.12,
                               stroke_color=col, stroke_width=3,
                               fill_color=col, fill_opacity=0.18)
        label = txt(st, 24, WARMWHITE, "BOLD").move_to(box)
        g = VGroup(box, label)
        boxes[st] = g
        row.add(g)
    row.arrange(RIGHT, buff=0.55)

    arrows = VGroup()
    for a, b in zip(STAGES[:-1], STAGES[1:]):
        arrows.add(Arrow(boxes[a].get_right(), boxes[b].get_left(),
                         buff=0.08, stroke_width=3, color=WARMWHITE,
                         max_tip_length_to_length_ratio=0.25))
    diagram = VGroup(row, arrows).scale(scale)
    return diagram, boxes


def io_rail(diagram):
    """INPUT ... OUTPUT captions under the pipeline."""
    inp = txt("INPUT: RGB + sparse anchors", 22, UNOBS)
    out = txt("OUTPUT: occlusion-aware SDF + collision-safe path", 22, UNOBS)
    inp.next_to(diagram, DOWN, buff=0.5).align_to(diagram, LEFT)
    out.next_to(diagram, DOWN, buff=0.5).align_to(diagram, RIGHT)
    return VGroup(inp, out)


def state_legend(scale=1.0):
    """The 4-state legend, docked bottom-left for all of Act 2."""
    rows = VGroup()
    for hexc, name, desc in STATE_LEGEND:
        sw = Square(side_length=0.22, fill_color=hexc, fill_opacity=1.0,
                    stroke_width=0)
        lab = txt(f"{name}", 18, WARMWHITE, "BOLD")
        d = txt(desc, 15, UNOBS)
        r = VGroup(sw, lab, d).arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
        rows.add(r)
    rows.arrange(DOWN, buff=0.14, aligned_edge=LEFT).scale(scale)
    rows.to_corner(DOWN + LEFT, buff=0.4)
    return rows


def hold(scene, default=0.0):
    """Trailing hold on the finished diagram, controlled by env OCCLU_HOLD.

    Lets each Act-2 segment be rendered to its VO-paced target length without
    re-authoring the animation — the completed visualization stays on screen
    (as it will under narration) before the segment's final fade.
    """
    import os
    scene.wait(float(os.environ.get("OCCLU_HOLD", default)))


def section_tag(idx, title):
    """Small 'D{idx} · TITLE' tag, top-left."""
    tag = txt(f"D{idx}", 26, OCCLUDED, "BOLD")
    ttl = txt(title, 32, WARMWHITE, "BOLD")
    g = VGroup(tag, ttl).arrange(RIGHT, buff=0.35)
    g.to_corner(UP + LEFT, buff=0.5)
    return g
