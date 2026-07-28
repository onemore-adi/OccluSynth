"""Text / title cards and alpha overlays (plan.md §4).

Scenes here render with a transparent background for overlaying on Blender
footage — render with `--format mov -t`:

  S1D_Freeze  -> renders/s1d_text.mov   (Scene 1D freeze overlay)
  S2A_Novelty -> renders/s2a_title.mov  (Scene 2A novelty title)
  S5B_Text    -> renders/s5b_text.mov   (Scene 5B hero-line overlay)

S7B_End is a full title card (opaque VOID) -> renders/s7b_end.mp4.

VO/on-screen text is taken from plan.md §4. The companion
OccluSynth_Demo_Video_Script.md was not present in the repo; the 2A novelty-card
wording below is composed from the project thesis and should be reconciled with
the script if/when it is available.
"""
from __future__ import annotations

from manim import (
    Scene, Text, VGroup, FadeIn, FadeOut, Write, Create, Line,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
)

from _manim_common import FONT, WARMWHITE, OCCLUDED, UNOBS, NAVY


_MAX_W = 12.5   # keep text inside the 14.22-unit-wide 16:9 frame with margins


def _t(text, size, color=WARMWHITE, weight="NORMAL", max_width=_MAX_W):
    m = Text(text, font=FONT, font_size=size, color=color, weight=weight)
    if m.width > max_width:
        m.scale_to_fit_width(max_width)
    return m


class S1D_Freeze(Scene):
    """Scene 1D — freeze + question. Alpha overlay on the frozen reveal frame."""

    def construct(self):
        line1 = _t("The robot never saw this obstacle.", 54)
        line2 = _t("Should it drive anyway?", 60, color=OCCLUDED, weight="BOLD")
        group = VGroup(line1, line2).arrange(DOWN, buff=0.55)

        self.play(FadeIn(line1, shift=UP * 0.2), run_time=1.0)
        self.wait(0.6)
        self.play(FadeIn(line2, scale=1.05), run_time=0.9)
        self.wait(1.6)
        self.play(FadeOut(group), run_time=0.6)


class S2A_Novelty(Scene):
    """Scene 2A — two-line novelty title. Alpha overlay.

    NOTE: wording composed from the project thesis (occlusion-aware mapping);
    reconcile with OccluSynth_Demo_Video_Script.md when available.
    """

    def construct(self):
        kicker = _t("THE IDEA", 30, color=OCCLUDED, weight="BOLD")
        kicker.set_opacity(0.9)
        line1 = _t("A map that labels the space it cannot see —", 50)
        line2 = _t("and plans as if the hidden hazards are already there.", 50)
        rule = Line(ORIGIN, ORIGIN, color=OCCLUDED)

        body = VGroup(line1, line2).arrange(DOWN, buff=0.4)
        stack = VGroup(kicker, body).arrange(DOWN, buff=0.6)

        self.play(FadeIn(kicker, shift=DOWN * 0.15), run_time=0.7)
        self.play(Write(line1), run_time=1.1)
        self.play(Write(line2), run_time=1.2)
        rule_y = body.get_bottom() + DOWN * 0.35
        half = min(body.width, 12.0) / 2.0
        rule.put_start_and_end_on(
            rule_y + LEFT * half, rule_y + RIGHT * half
        )
        self.play(Create(rule), run_time=0.8)
        self.wait(1.8)
        self.play(FadeOut(VGroup(stack, rule)), run_time=0.7)


class S5B_Text(Scene):
    """Scene 5B — hero line + path length. Alpha overlay on the detour render."""

    def construct(self):
        dist = _t("13.6 m", 96, color=OCCLUDED, weight="BOLD")
        hero = _t(
            "It avoided an obstacle that never appeared\nin any camera frame.",
            52,
        )
        hero.set_opacity(0.0)
        group = VGroup(dist, hero).arrange(DOWN, buff=0.7)

        self.play(FadeIn(dist, scale=1.1), run_time=0.9)
        self.wait(0.5)
        self.play(hero.animate.set_opacity(1.0), run_time=1.2)
        # Hero reveal holds >= 3 s (QA checklist).
        self.wait(3.2)
        self.play(FadeOut(group), run_time=0.8)


class S7B_End(Scene):
    """Scene 7B — closing thesis card. Opaque (VOID). renders/s7b_end.mp4."""

    def construct(self):
        thesis = VGroup(
            _t("The future of mapping isn't seeing more —", 52),
            _t("it's knowing what you cannot see.", 56, color=OCCLUDED, weight="BOLD"),
        ).arrange(DOWN, buff=0.5)

        credits = VGroup(
            _t("OccluSynth", 40, weight="BOLD"),
            _t("github.com/onemore-adi/OccluSynth", 30, color=UNOBS),
            _t("NIT Rourkela", 28, color=UNOBS),
        ).arrange(DOWN, buff=0.28)

        thesis.move_to(UP * 1.1)
        credits.next_to(thesis, DOWN, buff=1.2)

        self.play(Write(thesis[0]), run_time=1.4)
        self.play(Write(thesis[1]), run_time=1.4)
        # Held long enough to read twice (QA checklist).
        self.wait(3.0)
        self.play(FadeIn(credits, shift=UP * 0.2), run_time=1.0)
        self.wait(2.6)
        self.play(FadeOut(VGroup(thesis, credits)), run_time=1.0)
