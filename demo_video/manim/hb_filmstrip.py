# RGB-input filmstrip for the 5-min hackathon cut: two slow counter-scrolling
# rows of the real ScanNet scene0000_00 colour frames (the pipeline's only input).
# Render: ./.venv-manim/bin/manim -q h --fps 24 --format mp4 -o hb_filmstrip manim/hb_filmstrip.py HBFilmstrip
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from manim import *
from config.tokens_hb import BG, WHITE, GREY, BLUE, FONT_HEAD, FONT_BOLD, FONT_BODY

config.background_color = BG

FRAME_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                         "data/scannet/tasks/scannet_frames_25k/scene0000_00/color")


class HBFilmstrip(Scene):
    def construct(self):
        jpgs = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))
        top_paths = jpgs[0:40:2]      # 20 frames
        bot_paths = jpgs[1:40:2]      # 20 frames

        def row(paths, y):
            imgs = Group(*[ImageMobject(p) for p in paths])
            for im in imgs:
                im.height = 2.55
            imgs.arrange(RIGHT, buff=0.18)
            imgs.move_to(UP * y)
            return imgs

        top = row(top_paths, 1.42)
        bot = row(bot_paths, -1.42)
        # top row: left edge pinned near screen left, scrolls leftward (frames advance)
        top.shift(RIGHT * (-7.6 - top.get_left()[0]))
        # bottom row: right edge pinned near screen right, scrolls rightward
        bot.shift(RIGHT * (7.6 - bot.get_right()[0]))

        self.add(top, bot)
        # slow counter-scroll
        drift = 7.5
        self.play(top.animate(rate_func=linear, run_time=12.5).shift(LEFT * drift),
                  bot.animate(rate_func=linear, run_time=12.5).shift(RIGHT * drift))
        self.wait(0.2)


class HBFilmstripLabel(Scene):
    """Transparent overlay label for the filmstrip (render with --format mov -t)."""
    def construct(self):
        chip = Text("INPUT", font=FONT_BOLD, font_size=23, color=WHITE)
        box = RoundedRectangle(corner_radius=(chip.height + 0.32) / 2,
                               width=chip.width + 0.68, height=chip.height + 0.32,
                               fill_color=BLUE, fill_opacity=1.0, stroke_width=0)
        box.move_to(chip)
        pill = VGroup(box, chip)
        line = Text("40 RGB frames · one handheld sweep · no depth sensor",
                    font=FONT_BODY, font_size=26, color=WHITE)
        grp = VGroup(pill, line).arrange(RIGHT, buff=0.35)
        grp.to_edge(DOWN, buff=0.55)
        bg = RoundedRectangle(corner_radius=0.18, width=grp.width + 0.8,
                              height=grp.height + 0.42, fill_color="#0B0E14",
                              fill_opacity=0.82, stroke_width=0)
        bg.move_to(grp)
        panel = VGroup(bg, grp)
        panel.shift(DOWN * 0.2)
        self.play(FadeIn(panel, run_time=0.9),
                  panel.animate(rate_func=rate_functions.ease_out_cubic, run_time=0.9).shift(UP * 0.2))
        self.wait(10.5)
        self.play(FadeOut(panel, run_time=0.8))
