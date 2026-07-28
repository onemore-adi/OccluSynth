"""Shared setup for all OccluSynth demo-video Manim scenes.

Every scene imports colours / fps / resolution / font from config/tokens.py — the
single source of truth (plan.md §1). No scene hardcodes a colour or fps.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `config.tokens` importable regardless of CWD.
_ROOT = Path(__file__).resolve().parents[1]        # demo_video/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.tokens import (  # noqa: E402
    NAVY, NAVY_SHADOW, FREE, SURFACE, OCCLUDED, UNOBS, VOID, WARMWHITE,
    FPS, RES, FONT,
)
from manim import config, Text, ORIGIN  # noqa: E402

# Lock every render to the plan's spec (CLI flags may repeat these harmlessly).
config.frame_rate = FPS
config.pixel_width, config.pixel_height = RES
config.background_color = VOID

# Where the exporters (Phase 1) wrote the animation-ready bridge files.
DATA = _ROOT / "renders" / "data"


def value_text(tracker, fmt, *, font_size, color, weight="BOLD", at=ORIGIN):
    """A Pango-Text counter driven by a ValueTracker — avoids Manim's LaTeX-based
    DecimalNumber (no TeX dependency) while keeping the tokens FONT. Call
    ``.clear_updaters()`` to freeze it at the end of an animation."""
    def _make():
        return Text(fmt(tracker.get_value()), font=FONT, font_size=font_size,
                    color=color, weight=weight).move_to(at)
    t = _make()
    t.add_updater(lambda m: m.become(_make()))
    return t


__all__ = [
    "NAVY", "NAVY_SHADOW", "FREE", "SURFACE", "OCCLUDED", "UNOBS", "VOID",
    "WARMWHITE", "FPS", "RES", "FONT", "DATA", "value_text",
]
