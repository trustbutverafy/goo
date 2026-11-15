"""Estuary, evening: a watercolour on paper, 100 x 75 mm, painted in the studio's vocabulary in four sittings.

  sky     the sun reserved; a graded wash from ultramarine through violet and rose to a warm glow; clouds wetted
          again, violet dropped into their undersides and clean water run back into them as they go damp
  hills   a pale blue wash on dry paper, a warmer stroke run along one end
  water   the sun's road reserved; the water graded from warm under the horizon to deep blue; glitter lifted out
          while it is dry but re-wettable; the hills reflected in a glaze; ripples
  bank    dark earth, reeds, two posts and their reflections, two birds

Writes out/estuary/estuary.png, its raking-light picture and its journal. Re-running repaints only from the
first sitting whose code changed.
"""
import os
import numpy as np
from goo.classic.studio import Studio, band, disc, rect, polygon, below, above

OUT = os.environ.get("OUT", "out/estuary")
W, H, HOR = 400, 300, 172
st = Studio.new(W, H, ["ultramarine", "rose", "ochre", "sienna", "carbon", "yellow"], seed=5, workdir=OUT)
xs = np.arange(0, W + 8, 8)
ridge = [(x, min(HOR - 5, HOR - 9 - 6 * np.sin(x / 61 + 0.6) - 3 * np.sin(x / 23) - 13 * np.exp(-((x - 96) / 52) ** 2)
                 - 7 * np.exp(-((x - 372) / 40) ** 2))) for x in xs]
sun, sky, water = disc(288, 122, 15), band(0, HOR + 3), band(HOR, H)
hills = below(ridge) & band(0, HOR + 1)
road = polygon([(285, HOR), (291, HOR), (296, H), (280, H)])                                  # the sun's road on the water
glitter = [rect(288 - 1.8 * h * u, y, 288 + 1.8 * h * v, y + 2) for y, h, u, v in                           # ... and its broken glitter
           ((y, 7 + (y - HOR) * 0.16, *np.random.default_rng(y).uniform(0.6, 1.1, 2)) for y in range(HOR + 6, H - 4, 7))]
reflection = above([(x, HOR + 1 + 0.8 * (HOR - y)) for x, y in ridge]) & band(HOR + 1, H)   # the hills, upside down and shorter
bank = below([(x, H - 26 - 18 * np.exp(-((x - 40) / 70) ** 2) - 3 * np.sin(x / 9)) for x in xs]) & polygon([(0, 240), (225, 240), (215, H), (0, H)])

CLOUDS = [(70, 62, 46), (118, 54, 30), (190, 84, 54), (236, 74, 30), (46, 104, 34), (330, 60, 40)]


def palette(s):
    s.mix("zenith", {"watercolour": "ultramarine", "strength": 0.05})
    s.mix("violet", {"watercolour": "2 ultramarine + 1.2 rose", "strength": 0.034})
    s.mix("rose", {"watercolour": "1.5 rose + 1.6 ochre", "strength": 0.028})
    s.mix("glow", {"watercolour": "2 ochre + 1.2 yellow + 0.5 rose", "strength": 0.034})
    s.mix("cloud shadow", {"watercolour": "2 ultramarine + 1.2 rose + 0.5 sienna", "strength": 0.04})
    s.mix("far hills", {"watercolour": "3 ultramarine + 0.6 rose + 0.2 carbon", "strength": 0.07})
    s.mix("warm hills", {"watercolour": "2 sienna + 0.6 rose + 0.5 ultramarine", "strength": 0.09})
    s.mix("near water", {"watercolour": "1.6 ochre + 1.2 rose + 0.5 ultramarine", "strength": 0.034})
    s.mix("mid water", {"watercolour": "1.6 ultramarine + 1.1 rose", "strength": 0.036})
    s.mix("deep water", {"watercolour": "4 ultramarine + 0.3 carbon", "strength": 0.058})
    s.mix("dark", {"watercolour": "1.4 sienna + 1.5 ultramarine + carbon", "strength": 0.2})


def paint_sky(s):
    s.reserve(sun)
    s.graded_wash(sky, "zenith", "glow", via=["violet", "rose"])
    for x, y, w in CLOUDS:                                         # one cloud at a time, while it is wet
        cloud = rect(x, y - 6, x + w, y + 8)
        s.wet(cloud, "damp")
        s.drop_in(rect(x + 4, y + 6, x + w - 4, y + 10), "cloud shadow")
        s.dry("damp", region=cloud)
        s.backrun(rect(x, y - 4, x + w, y + 4))
    s.dry("fixed")


def paint_hills(s):
    s.wash(hills, "far hills", strokes="across")
    s.line([(x, y + 9) for x, y in ridge if 30 <= x <= 150], "warm hills", width=7, taper=False, wetness=0.6)
    s.dry("fixed")


def paint_water(s):
    s.reserve(road & ~band(HOR, HOR + 3))
    s.graded_wash(water, "near water", "deep water", via=["mid water"], wetness=0.5, wobble=0.8)
    s.dry("touch")                                                 # dry, but still re-wettable: the glitter lifts out
    for dash in glitter: s.lift(dash, scrubs=1)
    s.glaze(reflection & ~road, "far hills", wetness=0.35)
    rng = np.random.default_rng(11)
    for _ in range(15):
        y = HOR + 30 + rng.uniform(0, 96); x = rng.uniform(10, W - 60); L = rng.uniform(14, 40) * (0.5 + (y - HOR) / 130)
        s.line([(x, y), (x + L / 2, y + rng.normal(0, 0.6)), (x + L, y)], "deep water", width=3.5, wetness=0.45)
    s.dry("fixed")
    s.release()


def paint_bank(s):
    s.wash(bank, "dark", strokes=5, wetness=0.6)
    rng = np.random.default_rng(12)
    for _ in range(46):                                            # reeds, some with a seed head
        x0 = rng.uniform(6, 200); y0 = H - 26 - 18 * np.exp(-((x0 - 40) / 70) ** 2) + rng.uniform(2, 10)
        h = rng.uniform(26, 70) * (1.15 - x0 / 420); lean = rng.normal(0.12, 0.16)
        reed = [(x0 + lean * h * t + 3 * lean * np.sin(3 * t), y0 - h * t) for t in np.linspace(0, 1, 7)]
        s.line(reed, "dark", width=2.5)
        if rng.random() < 0.3: s.line([(reed[-2][0] - 1, reed[-2][1] + 4), (reed[-2][0] + 0.5, reed[-2][1] - 4)], "dark", width=2.5, taper=False, wetness=0.8)
    for x, top, foot in [(318, HOR + 22, HOR + 74), (344, HOR + 38, HOR + 82)]:   # posts, and their broken reflections
        s.line([(x + 1.5, top), (x, foot)], "dark", width=5, taper=False, wetness=0.6)
        s.line([(x + 2.5 * np.sin(k * 1.3), foot + 3 + 4.5 * k) for k in range(9)], "deep water", width=3.5, wetness=0.4)
    for x, y, w in [(152, 44, 7), (176, 34, 5)]:                   # two birds
        s.line([(x - w, y - w / 2), (x, y), (x + w, y - 0.6 * w)], "dark", width=2.5, wetness=0.4)
    s.dry("fixed")


for name, fn in [("palette", palette), ("sky", paint_sky), ("hills", paint_hills), ("water", paint_water), ("bank", paint_bank)]:
    print(name, st.sitting(name, fn), "tick", st.clock, flush=True)
st.save(f"{OUT}/estuary.png")
st.save(f"{OUT}/estuary-raking.png", raking=dict(azimuth=20, elevation=18))
