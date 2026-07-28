# Static lower-third labels (Samsung pill style) for the 5-min hackathon cut.
# Rendered as transparent PNGs:
#   ./.venv-manim/bin/manim -q h -s -t -o <id> manim/hb_lowerthirds.py <Scene>
# Composited over footage in ffmpeg with alpha fades.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from manim import *
from config.tokens_hb import BG, WHITE, GREY, BLUE, AMBER, FONT_BOLD, FONT_BODY

config.background_color = BG


def lower_third(scene, chip_text, line_text, chip_fill=BLUE):
    chip = Text(chip_text, font=FONT_BOLD, font_size=23, color=WHITE)
    box = RoundedRectangle(corner_radius=(chip.height + 0.32) / 2,
                           width=chip.width + 0.68, height=chip.height + 0.32,
                           fill_color=chip_fill, fill_opacity=1.0, stroke_width=0)
    box.move_to(chip)
    pill = VGroup(box, chip)
    line = Text(line_text, font=FONT_BODY, font_size=26, color=WHITE)
    grp = VGroup(pill, line).arrange(RIGHT, buff=0.35)
    grp.to_edge(DOWN, buff=0.55)
    bg = RoundedRectangle(corner_radius=0.18, width=grp.width + 0.8,
                          height=grp.height + 0.42, fill_color=BG,
                          fill_opacity=0.82, stroke_width=0)
    bg.move_to(grp)
    scene.add(bg, grp)


def lt_scene(name, chip, line, fill=BLUE):
    class _L(Scene):
        def construct(self):
            lower_third(self, chip, line, fill)
    _L.__name__ = name
    _L.__qualname__ = name
    return _L


LT_Input    = lt_scene("LT_Input",   "INPUT",      "40 RGB frames · one handheld sweep · no depth sensor")
LT_Holes    = lt_scene("LT_Holes",   "TODAY",      "State-of-the-art reconstruction — holes wherever no camera saw")
LT_Amber    = lt_scene("LT_Amber",   "HIDDEN",     "Amber — the space the cameras never saw", AMBER)
LT_Fusion   = lt_scene("LT_Fusion",  "FUSION",     "Each depth ray carves free space and marks surfaces")
LT_Growth   = lt_scene("LT_Growth",  "COMPLETION", "The predicted room grows in — ghost-white is ground truth")
LT_Before   = lt_scene("LT_Before",  "BEFORE",     "Conventional mesh — only what the cameras measured")
LT_After    = lt_scene("LT_After",   "AFTER",      "OccluSynth mesh — amber is predicted hidden geometry", AMBER)
LT_Sofa     = lt_scene("LT_Sofa",    "CLOSE-UP",   "Behind the sofa — geometry no camera ever captured", AMBER)
LT_Uncert   = lt_scene("LT_Uncert",  "CONFIDENCE", "16 stochastic passes → per-voxel certainty")
LT_Collide  = lt_scene("LT_Collide", "NAIVE",      "Planning on the measured map alone — straight through hidden space")
LT_Detour   = lt_scene("LT_Detour",  "RISK-AWARE", "The planner prices amber risk — and detours before it can see")
LT_Bench    = lt_scene("LT_Bench",   "BENCHMARK",  "Held-out ScanNet scenes — occluded region scored as a safety problem")
