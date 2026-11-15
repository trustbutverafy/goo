"""The optics study: what v2 shows that classic cannot, each effect beside classic, in one labelled figure.

    python studies/optics.py                  (PEAR=out/pear.canvas.pkl: run paintings/pear.py first)

Every scene is painted once, with tools or Canvas.add, and photographed through both models: optics never
changes the World, so the two columns are the same paint seen two ways. Colours are linear sRGB under D65
unless a panel says otherwise, displayed with the sRGB curve. Numbers printed and written to
out/optics-study.json; the figure is out/optics-study.png.

Panels
  1. a watercolour wash (0.40 water + 0.03 ultramarine, no gum, as goo's watercolour is) photographed
     as it dries, on paper and on gesso; and its mean L* against time
  2. an acrylic stroke (polymer + water + ultramarine) wet and dry
  3. ultramarine as an oil film and as a dried lean wash, over white and over a black (carbon) stripe
  4. black: glossy (fat oil), satin (oil just filling the bed) and lean (underbound) carbon
  5. ultramarine + yellow in oil, mixed in nine steps
  6. the pear, under daylight and under tungsten (as a daylight-balanced camera sees it, and as an eye adapts)
  7. an ultramarine + white film crossing its critical pigment volume concentration (PVC 0.40 .. 0.90)
"""
import json, os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from goo.engine import Canvas, PIGMENTS
from goo.brush import Brush
from goo.palette import lab, linear_to_srgb
from goo import optics as O

OUT = os.environ.get("OUT", "out"); os.makedirs(OUT, exist_ok=True)
PEAR = os.environ.get("PEAR", os.path.join(OUT, "pear.canvas.pkl"))
CLASSIC, V2 = O.ClassicOptics(), O.V2Optics()
PIGMENTS.setdefault("acrylic", dict(K=[0.0] * 4, S=[0.0] * 4))   # classic has no binders: it sees the polymer as clear
show = lambda rgb: linear_to_srgb(np.clip(rgb, 0, 1))
rgb2 = lambda cv: CLASSIC.reflectance(cv)[..., :3]               # classic's colour (its bands are sRGB)
rgbv = lambda cv, light="D65", adapt=False: V2.reflectance(cv, light=light, adapt=adapt)
Lab = lambda rgb: lab(np.clip(rgb, 0, 1))
dE = lambda a, b: float(np.linalg.norm(Lab(a) - Lab(b), axis=-1).mean())
num = {}

# ------------------------------------------------------------------ 1. a wash drying, on paper and on gesso
def stroke_mask(H, W, y, half, x0, x1, soft=2.5):
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    edge = np.minimum(np.minimum(xx - x0, x1 - xx), half - np.abs(yy - y - 1.5 * np.sin(xx / 13)))
    return np.clip(edge / soft, 0, 1)

WASH_T = (0, 20, 40, 60, 70, 80, 90, 100, 110, 120, 140, 200, 300)
def wash(ground, gum=0.0, frames_at=WASH_T):
    """A wash as a brush leaves it (0.40 units of water carrying 0.03 of ultramarine, phi 0.07; `gum` units of
    gum arabic if asked), photographed as it dries at the canvas's own rates, then dried out and fixed."""
    pig = ["ultramarine"] + (["gum"] if gum else [])
    cv = Canvas(110, 44, pig, seed=5, ground=ground, optics="v2")
    m = stroke_mask(44, 110, 22, 9, 6, 104)
    cv.add("water", 0.40 * m); cv.add("ultramarine", 0.03 * m)
    if gum: cv.add("gum", gum * m)
    frames, t = [], 0
    for ft in frames_at:
        while t < ft: cv.wait(5); t += 5
        frames.append((t, rgbv(cv), rgb2(cv), float(cv.q[0].max()), float(cv.store.max())))
    while cv.q[0].max() > 1e-6: cv.wait(50); t += 50
    while cv.store.max() > 0: cv.wait(50); t += 50
    cv.fix()
    frames.append((t, rgbv(cv), rgb2(cv), 0.0, 0.0))
    return cv, frames

PIGMENTS.setdefault("gum", dict(K=[0.0] * 4, S=[0.0] * 4))         # likewise the gum
WASH = {g: wash(g) for g in ("paper", "gesso")}
core = (slice(18, 26), slice(30, 80))                      # the middle of the stroke
for g, (cv, fr) in WASH.items():
    wet, dry = fr[2], fr[-1]                               # t = 40 (wet, ground soaked) vs dried and fixed
    num[f"wash on {g}: dE wet (t={wet[0]}) vs dry, v2"] = dE(wet[1][core], dry[1][core])
    num[f"wash on {g}: dE wet vs dry, classic"] = dE(wet[2][core], dry[2][core])
    num[f"wash on {g}: L* wet -> dry, v2"] = [float(Lab(wet[1][core])[..., 0].mean()), float(Lab(dry[1][core])[..., 0].mean())]
    num[f"wash on {g}: L* wet -> dry, classic"] = [float(Lab(wet[2][core])[..., 0].mean()), float(Lab(dry[2][core])[..., 0].mean())]
cvg, frg = wash("paper", gum=0.03, frames_at=(40,))
num["gum-bound wash (gum 0.03) on paper: dE wet vs dry, v2"] = dE(frg[0][1][core], frg[-1][1][core])
num["gum-bound wash on paper: L* wet -> dry, v2"] = [float(Lab(frg[0][1][core])[..., 0].mean()), float(Lab(frg[-1][1][core])[..., 0].mean())]
for g in ("paper", "gesso"):
    seen_dry = V2.preview(["water", "oil"], np.zeros((2, 1)), ground=g)[0]
    seen_wet = V2.preview(["water", "oil"], np.zeros((2, 1)), ground=g, store=0.08)[0]
    num[f"bare {g}: R(550 nm) dry -> soaked (store full)"] = [float(seen_dry[7]), float(seen_wet[7])]
    num[f"bare {g}: dE dry vs soaked"] = dE(V2.colour(seen_dry), V2.colour(seen_wet))

# ------------------------------------------------------------------ 2. acrylic wet and dry
def acrylic():
    """A fluid acrylic as laid: 0.30 units, half water, a third polymer, the rest ultramarine."""
    cv = Canvas(110, 40, ["ultramarine", "acrylic"], seed=6, optics="v2")
    m = stroke_mask(40, 110, 20, 8, 6, 104)
    cv.add("water", 0.15 * m); cv.add("acrylic", 0.10 * m); cv.add("ultramarine", 0.05 * m)
    wet = (rgbv(cv), rgb2(cv), cv.finish())
    while cv.q[0].max() > 1e-6: cv.wait(50)
    cv.wait(100); cv.fix()
    return cv, wet, (rgbv(cv), rgb2(cv), cv.finish())
ACR = acrylic()
acore = (slice(16, 24), slice(30, 80))
num["acrylic: dE wet vs dry, v2"] = dE(ACR[1][0][acore], ACR[2][0][acore])
num["acrylic: dE wet vs dry, classic"] = dE(ACR[1][1][acore], ACR[2][1][acore])
num["acrylic: L* wet -> dry, v2"] = [float(Lab(ACR[1][0][acore])[..., 0].mean()), float(Lab(ACR[2][0][acore])[..., 0].mean())]
num["acrylic: gloss wet -> dry, v2"] = [float(ACR[1][2][acore].mean()), float(ACR[2][2][acore].mean())]

# ------------------------------------------------------------------ 0. how well v2's spectra reduce to classic: masstones in oil
sp = ["water", "oil"]; mass = {}
for p_ in [k for k in PIGMENTS if k not in ("acrylic", "gum")]:
    f = np.zeros((3, 1)); f[1] = 1.0; f[2] = 1.0
    c = CLASSIC.preview(sp + [p_], f, np.broadcast_to(np.zeros(4), (1, 4)))[0, :3]
    nosurf = O.V2Optics(surface=False)
    a = nosurf.colour(nosurf.preview(sp + [p_], f, under=np.zeros(16)))[0]
    b_ = V2.colour(V2.preview(sp + [p_], f, under=np.zeros(16)))[0]
    mass[p_] = [dE(c, a), dE(c, b_)]
num["masstone in oil, dE to classic: [15 bands, no surface; v2]"] = {k: [round(x, 1) for x in v] for k, v in mass.items()}

# ------------------------------------------------------------------ 3. ultramarine in oil and as a lean wash, over a black stripe
def glaze():
    cv = Canvas(100, 50, ["carbon", "ultramarine"], seed=7, optics="v2", laws=dict(evaporation=1.0))
    yy, xx = np.mgrid[0:50, 0:100]
    cv.add("carbon", 0.06 * (np.abs(yy - 25) < 5)); cv.add("oil", 0.002 * (np.abs(yy - 25) < 5)); cv.fix()   # a black stripe
    cv.add("water", 0.3 * (xx >= 50)); cv.add("ultramarine", 0.03 * (xx >= 50))          # right: a wash
    while cv.q[0].max() > 1e-6: cv.wait(50, evap_boost=8.0)
    cv.wait(200, evap_boost=8.0); cv.fix()
    cv.add("oil", 0.06 * (xx < 50)); cv.add("ultramarine", 0.03 * (xx < 50))             # left: the same pigment in oil
    return cv
GL = glaze(); gv, gc = rgbv(GL), rgb2(GL)
for name, xs in (("oil film", slice(10, 40)), ("dried lean wash", slice(60, 90))):
    for lab_, img in (("v2", gv), ("classic", gc)):
        over_white = Lab(img[5:15, xs])[..., 0].mean(); over_black = Lab(img[22:28, xs])[..., 0].mean()
        num[f"ultramarine {name}: L* over white / over black stripe, {lab_}"] = [float(over_white), float(over_black)]

# ------------------------------------------------------------------ 4. black: glossy, satin, lean
def blacks():
    cv = Canvas(90, 30, ["carbon"], seed=8, optics="v2")
    xx = np.mgrid[0:30, 0:90][1]
    for i, pvc in enumerate([0.30, 0.62, 0.90]):            # 0.62: oil just fills the bed (CPVC 0.64)
        m = (xx >= 30 * i + 2) & (xx < 30 * i + 28)
        cv.add("carbon", 0.1 * m); cv.add("oil", 0.1 * (1 - pvc) / pvc * m)
    return cv
BK = blacks(); bv, bc, bg = rgbv(BK), rgb2(BK), BK.finish()
for i, name in enumerate(["glossy (PVC 0.30)", "satin (PVC 0.62)", "lean (PVC 0.90)"]):
    xs = slice(30 * i + 6, 30 * i + 24)
    num[f"black {name}: L* v2 / classic, gloss v2"] = [float(Lab(bv[8:22, xs])[..., 0].mean()), float(Lab(bc[8:22, xs])[..., 0].mean()), float(bg[8:22, xs].mean())]

# ------------------------------------------------------------------ 5. blue and yellow
def mixes():
    cv = Canvas(90, 20, ["ultramarine", "yellow"], seed=9, optics="v2")
    xx = np.mgrid[0:20, 0:90][1]
    for i, f in enumerate(np.linspace(0, 1, 9)):            # f = ultramarine share of the pigment
        m = (xx >= 10 * i) & (xx < 10 * i + 10)
        for k, v in Brush.paint("oil", 0.5, 0.4, {"ultramarine": f + 1e-12, "yellow": 1 - f + 1e-12}).items(): cv.add(k, v * m)
    return cv
MX = mixes(); mv, mc = rgbv(MX), rgb2(MX)
hue = lambda rgb: float(np.degrees(np.arctan2(Lab(rgb)[..., 2].mean(), Lab(rgb)[..., 1].mean())) % 360)
num["blue+yellow 50/50: CIELAB hue v2 / classic (green ~ 135-180)"] = [hue(mv[5:15, 42:48]), hue(mc[5:15, 42:48])]
num["blue+yellow 50/50: chroma v2 / classic"] = [float(np.hypot(*Lab(mv[5:15, 42:48])[..., 1:].mean((0, 1)))), float(np.hypot(*Lab(mc[5:15, 42:48])[..., 1:].mean((0, 1))))]

# ------------------------------------------------------------------ 6. the pear under daylight and tungsten
PEARS = None
if os.path.exists(PEAR):
    pc = pickle.load(open(PEAR, "rb"))
    PEARS = dict(classic=rgb2(pc), day=rgbv(pc), tung=rgbv(pc, "A"), tung_eye=rgbv(pc, "A", True))
    num["pear: dE v2 daylight vs classic"] = dE(PEARS["day"], PEARS["classic"])
    num["pear: dE v2 tungsten (adapted eye) vs daylight"] = dE(PEARS["tung_eye"], PEARS["day"])
else:
    print(f"no pear at {PEAR}: run paintings/pear.py (it writes out/pear.canvas.pkl) for panel 6")

# ------------------------------------------------------------------ 7. a lean film crossing its CPVC
def cpvc():
    cv = Canvas(100, 24, ["ultramarine", "white"], seed=10, optics="v2", laws=dict(cure_rate=0.002))
    xx = np.mgrid[0:24, 0:100][1]; pvcs = np.linspace(0.40, 0.90, 10)
    for i, pvc in enumerate(pvcs):
        m = (xx >= 10 * i) & (xx < 10 * i + 10); solid = 0.08
        cv.add("ultramarine", 0.8 * solid * m); cv.add("white", 0.2 * solid * m); cv.add("oil", solid * (1 - pvc) / pvc * m)
    cv.wait(300, cure_boost=12.0); cv.fix()                 # cured and fixed
    return cv, pvcs
CP, PVCS = cpvc(); cv_, cc_ = rgbv(CP), rgb2(CP); cg_ = CP.finish()
cp_L = [[float(Lab(img[6:18, 10 * i + 2:10 * i + 8])[..., 0].mean()) for i in range(10)] for img in (cv_, cc_)]
cp_C = [[float(np.hypot(*Lab(img[6:18, 10 * i + 2:10 * i + 8])[..., 1:].mean((0, 1)))) for i in range(10)] for img in (cv_, cc_)]
num["cpvc: PVC"] = [float(p) for p in PVCS]
num["cpvc: L* v2"], num["cpvc: L* classic"] = cp_L
num["cpvc: chroma v2"], num["cpvc: chroma classic"] = cp_C
num["cpvc: gloss v2"] = [float(cg_[6:18, 10 * i + 2:10 * i + 8].mean()) for i in range(10)]

# ------------------------------------------------------------------ the figure
fig = plt.figure(figsize=(17, 22))
gs = fig.add_gridspec(9, 8, hspace=0.55, wspace=0.12)
def im(ax, rgb, title, fs=8):
    ax.imshow(show(rgb), interpolation="nearest"); ax.set_title(title, fontsize=fs); ax.set_xticks([]); ax.set_yticks([])
for r, g in enumerate(("paper", "gesso")):
    cv, fr = WASH[g]
    for j, k in enumerate([WASH_T.index(t_) for t_ in (0, 20, 110, 120, 140)] + [len(fr) - 1]):
        t, v, c, w, s = fr[k]
        im(fig.add_subplot(gs[2 * r, j]), v, f"v2, {g}, t={t}" + (" (fixed)" if k == len(fr) - 1 else f"\nwater {w:.2f}, ground {s / 0.08:.0%}"))
        im(fig.add_subplot(gs[2 * r + 1, j]), c, f"classic, {g}, t={t}")
    ax = fig.add_subplot(gs[2 * r:2 * r + 2, 6:8])
    ts = [f[0] for f in fr]
    ax.plot(ts, [Lab(f[1][core])[..., 0].mean() for f in fr], "o-", label="v2")
    ax.plot(ts, [Lab(f[2][core])[..., 0].mean() for f in fr], "s--", label="classic")
    ax.set_xscale("symlog", linthresh=40); ax.set_xlabel("ticks", fontsize=8); ax.set_ylabel("L* of the stroke", fontsize=8)
    ax.set_title(f"1. ultramarine wash (0.03 in 0.40 water, no gum) drying on {g}", fontsize=9); ax.legend(fontsize=8); ax.tick_params(labelsize=7)
row = 4
im(fig.add_subplot(gs[row, 0:2]), ACR[1][0], f"2. acrylic WET, v2 (L* {num['acrylic: L* wet -> dry, v2'][0]:.0f}, gloss {num['acrylic: gloss wet -> dry, v2'][0]:.2f})")
im(fig.add_subplot(gs[row, 2:4]), ACR[2][0], f"acrylic DRY, v2 (L* {num['acrylic: L* wet -> dry, v2'][1]:.0f}, gloss {num['acrylic: gloss wet -> dry, v2'][1]:.2f})")
im(fig.add_subplot(gs[row, 4:6]), ACR[1][1], "acrylic WET, classic")
im(fig.add_subplot(gs[row, 6:8]), ACR[2][1], "acrylic DRY, classic")
row = 5
im(fig.add_subplot(gs[row, 0:2]), gv, "3. ultramarine: oil film (left) | dried lean wash (right)\nover a carbon stripe, v2")
im(fig.add_subplot(gs[row, 2:4]), gc, "the same, classic")
im(fig.add_subplot(gs[row, 4:6]), bv, "4. carbon: glossy | satin | lean (PVC .30 .62 .90), v2\n" + " / ".join(f"L*{num[k][0]:.0f} g{num[k][2]:.1f}" for k in num if k.startswith("black")))
im(fig.add_subplot(gs[row, 6:8]), bc, "the same, classic\n" + " / ".join(f"L*{num[k][1]:.0f}" for k in num if k.startswith("black")))
row = 6
im(fig.add_subplot(gs[row, 0:4]), mv, f"5. ultramarine -> yellow in oil, 9 steps, v2 (50/50 hue {num['blue+yellow 50/50: CIELAB hue v2 / classic (green ~ 135-180)'][0]:.0f} deg)")
im(fig.add_subplot(gs[row, 4:8]), mc, f"the same, classic (50/50 hue {num['blue+yellow 50/50: CIELAB hue v2 / classic (green ~ 135-180)'][1]:.0f} deg)")
row = 7
if PEARS:
    for j, (k, t) in enumerate([("classic", "6. pear, classic"), ("day", "pear, v2, daylight (D65)"),
                                ("tung", "v2, tungsten (A), daylight camera"), ("tung_eye", "v2, tungsten, adapted eye")]):
        im(fig.add_subplot(gs[row, 2 * j:2 * j + 2]), PEARS[k], t)
row = 8
im(fig.add_subplot(gs[row, 0:3]), cv_, "7. ultramarine + white, PVC 0.40 -> 0.90 (CPVC 0.64), cured, v2")
im(fig.add_subplot(gs[row, 3:6]), cc_, "the same, classic")
ax = fig.add_subplot(gs[row, 6:8])
ax.plot(PVCS, cp_L[0], "o-", label="L* v2"); ax.plot(PVCS, cp_L[1], "s--", label="L* classic")
ax.plot(PVCS, cp_C[0], "o-", label="chroma v2"); ax.plot(PVCS, cp_C[1], "s--", label="chroma classic")
ax.axvline(O.PHI_PACK, color="k", lw=0.6); ax.set_xlabel("PVC", fontsize=8); ax.legend(fontsize=7); ax.tick_params(labelsize=7)
ax.set_title("7. L* and chroma vs PVC", fontsize=9)
fig.suptitle("goo optics v2 beside classic: one index rule, one surface, 15 visible bands + 1.5 um (D65 unless noted)", fontsize=12)
fig.savefig(os.path.join(OUT, "optics-study.png"), dpi=72, bbox_inches="tight")
json.dump(num, open(os.path.join(OUT, "optics-study.json"), "w"), indent=1)
for k, v in num.items(): print(f"{k:70s} {v if isinstance(v, dict) else np.round(v, 3)}")
