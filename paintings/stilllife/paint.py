"""Still life with jug, fruit and knife: painted in sittings, the way a painter would, on a 16 x 12 cm panel.

  1. charcoal: the contours of the set-up (and a first, abandoned placing of the apple)        -> fixative
  2. imprimatura: a thin burnt-sienna watercolour wash, dried                                    -> sealed
  3. dead colour: big brushes, long lean paint, the masses and values                            -> dries (days)
  4. second sitting: middle brushes, alla prima, the forms turned                                -> dries
  5. glazes: thin transparent paint where the shadows are still too light
  6. third sitting: small brushes, edges and features, wet into wet
  7. impasto: short loaded paint on the highlights
  8. a signature scratched through to the imprimatura; then the panel cures

Everything is done with brushes and a stylus; the painter sees only colour. SCALE=0.5 paints a 320 x 240 test.
"""
import os, json, time, pickle
import numpy as np
from scipy import ndimage
from skimage import measure
from goo import Canvas, Brush, Stylus
from goo.painter import Painter
from goo.palette import Palette

SCALE = float(os.environ.get("SCALE", "1.0"))
OUT = os.environ.get("OUT", "out/stilllife" + ("" if SCALE == 1 else f"-{SCALE:g}"))
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
ref = np.load("out/still-life-reference.npz")
def down(a, order=1):
    if SCALE == 1: return a
    if a.ndim == 3: return np.stack([ndimage.zoom(a[..., c], SCALE, order=order) for c in range(a.shape[2])], -1)
    return ndimage.zoom(a, SCALE, order=order)
motif, ids, hl = down(ref["rgb"]), down(ref["ids"], 0), down(ref["highlight"])
H, W = ids.shape
PAL = ["white", "carbon", "ultramarine", "yellow", "red", "rose", "ochre", "sienna", "green"]
cv = Canvas(W, H, PAL, seed=11, laws=dict(lazy=90, cure_rate=0.0, flow_sub=0.1, diff_sub=0.1))  # leveling capped at 1 substep per 4 ticks: very runny paint levels over
                                                                     # tens of ticks, not a fraction of one (stable either way: the face cap)
PLAN = {0: np.radians(62), 1: 0.0, 2: 0.0, 3: "around", 4: "around", 5: "around", 6: "around", 7: np.radians(-20), 8: np.radians(-20)}
painter = Painter(cv, motif, ids, Palette(PAL, steps=12), seed=5, plan=PLAN)
rng = np.random.default_rng(3)
s = lambda v: v * SCALE                                          # lengths in full-size cells
frames, log = [], []

class Snap:
    def __init__(self, every): self.every, self.last = every, -10 ** 9
    def __call__(self, force=False):
        if force or cv.t - self.last >= self.every:
            self.last = cv.t; frames.append((cv.t, (np.clip(cv.reflectance()[..., :3], 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)))
snap = Snap(int(400 * SCALE ** 2) + 50)
_step = cv.step
def stepped(): _step(); snap()
cv.step = stepped

def stage(name):
    cv.flush()
    led = cv.ledger(); worst = max(abs(v["residual"]) / max(1, v["loaded"]) for v in led["species"].values())
    e = painter.error(); e4 = painter.error(blur=s(4))
    log.append(dict(stage=name, t=cv.t, substeps=cv.substeps, capped=cv.capped, strata=len(cv.strata), residual=worst,
                    min_q=float(cv.q.min()), max_h=float(cv.h.max()), dE=e, dE_blur1mm=e4, wall=round(time.time() - T0)))
    print(f"[{time.time() - T0:6.0f}s] {name:34s} t={cv.t:6d} strata={len(cv.strata)} residual={worst:.0e} max_h={cv.h.max():.2f} "
          f"dE={e:5.1f} (at 1 mm: {e4:5.1f})", flush=True)
    snap(force=True)
    import matplotlib.pyplot as plt
    plt.imsave(f"{OUT}/stage-{len(log):02d}.png", frames[-1][1])

_pass = painter.pass_
def pass_(*a, **k):
    t0 = time.time(); n = _pass(*a, **k)
    print(f"         {k.get('label', ''):28s} {n:5d} strokes in {time.time() - t0:5.0f}s, swatch {painter.log[-1]['swatch']:.3f}", flush=True); return n
painter.pass_ = pass_

def cure_and_fix(ticks=400):
    cv.L["cure_rate"] = 0.002; cv.wait(int(ticks), cure_boost=12.0); cv.L["cure_rate"] = 0.0
    left = cv.fix(); return left

oil = lambda phi: (lambda mix, vol: Brush.paint("oil", phi, vol, mix))

# ------------------------------------------------------------------ 1. charcoal
coal = Brush(cv, s(2.6) + 0.6, bristles=1, exchange=0.6, tooth_only=True, drag=0, plough=0, load={"carbon": 6.0})
def draw(path, target=0.045, pressure=1.0):
    jit = [(x + rng.normal(0, 0.35), y + rng.normal(0, 0.35)) for x, y in path]
    coal.stroke(jit, speed=1.8, pressure=pressure, target=target)
    if coal.load.sum() < 2: coal.charge({"carbon": 6.0})
def contours(mask, min_len=30):
    out = []
    for c in measure.find_contours(ndimage.binary_opening(mask, iterations=1).astype(float), 0.5):
        if len(c) > s(min_len): out.append([(p[1], p[0]) for p in c[::max(2, int(s(5)))]])
    return out
apple = ids == 5
for c in contours(np.roll(apple, -int(s(46)), axis=1)):                        # the first placing, abandoned
    for k in range(0, len(c) - 6, 6): draw(c[k:k + 8], target=0.03, pressure=0.8)
for oid in (3, 4, 5, 6, 7, 8):
    for c in contours(ids == oid):
        for k in range(0, len(c) - 6, 6): draw(c[k:k + 8], target=0.045)
table_edge = [(x, y) for y, x in zip(*np.nonzero(np.diff((ids == 1).astype(int), axis=0) == -1))]   # top/front boundary
if table_edge:
    ys = np.array([p[1] for p in table_edge]); yy = int(np.median(ys))
    draw([(2, yy), (W - 2, yy)], target=0.04)
by = int(np.median(np.nonzero((ids == 0)[:, W // 2])[0].max()))                 # wall / table line
draw([(2, by), (W - 2, by)], target=0.035)
stage("charcoal"); cv.fix(); stage("fixative")

# ------------------------------------------------------------------ 2. imprimatura
wash = lambda: Brush.paint("water", 0.09, 12000 * SCALE ** 2 * 4, {"sienna": 3, "ochre": 1})
mop = Brush(cv, s(60), bristles=14, exchange=0.45, plough=0.2, load=wash())
for row, y in enumerate(np.arange(s(8), H + s(20), s(36))):
    xs = (-6, W + 6) if row % 2 == 0 else (W + 6, -6)
    mop.charge(wash())
    mop.stroke([(xs[0], y + rng.normal(0, 2)), ((xs[0] + xs[1]) / 2, y + rng.normal(0, 3)), (xs[1], y + rng.normal(0, 2))], speed=s(12), pressure=0.9, target=0.3)
stage("imprimatura laid")
cv.wait(300, evap_boost=8.0)
while cv.q[0].max() > 1e-6: cv.wait(60, evap_boost=8.0)
cv.fix(); stage("imprimatura sealed")

# ------------------------------------------------------------------ 3. dead colour
painter.pass_(s(28), oil(0.42), target=0.4, threshold=0.0, blur=1.0, curvature=0.5, max_len=6, colour_tol=18, phi=0.42, label="dead colour")
painter.pass_(s(20), oil(0.42), target=0.4, threshold=8.0, blur=0.8, curvature=0.5, max_len=7, colour_tol=16, phi=0.42, label="dead colour 2")
stage("dead colour"); left = cure_and_fix(); stage(f"dead colour dry")

# ------------------------------------------------------------------ 4. second sitting
painter.pass_(s(14), oil(0.48), target=0.5, threshold=5.0, blur=0.7, curvature=0.6, max_len=9, phi=0.48, drag=0.7, label="second sitting")
painter.pass_(s(9), oil(0.48), target=0.5, threshold=6.0, blur=0.6, curvature=0.7, max_len=12, phi=0.48, drag=0.6, label="second sitting, smaller", max_strokes=int(1500 * SCALE ** 2))
stage("second sitting"); cure_and_fix(); stage("second sitting dry")

# ------------------------------------------------------------------ 5. third sitting: features and edges, a loaded brush and a light touch
edges = ndimage.binary_dilation(ndimage.morphological_gradient(ids, size=3) > 0, iterations=int(s(4)) + 1)
painter.pass_(s(6), oil(0.5), target=0.55, threshold=7.0, blur=0.5, curvature=0.7, max_len=10, phi=0.5, drag=0.35, plough=0.3, width_scale=1.4, label="features", max_strokes=int(1800 * SCALE ** 2))
painter.pass_(s(4), oil(0.5), target=0.5, threshold=6.0, blur=0.4, curvature=0.75, max_len=8, phi=0.5, region=edges, drag=0.3, plough=0.3, width_scale=1.2, label="edges", max_strokes=int(1200 * SCALE ** 2))
plain = np.isin(ids, [0, 1, 2]) & ~ndimage.binary_dilation(~np.isin(ids, [0, 1, 2]), iterations=int(s(5)))
painter.pass_(s(5), oil(0.5), target=0.5, threshold=9.0, blur=0.8, curvature=0.5, max_len=6, phi=0.5, region=plain, drag=0.35, plough=0.3,
              cell_stat="max", label="tidying", max_strokes=int(900 * SCALE ** 2))
stage("third sitting"); cure_and_fix(); stage("third sitting dry")

# ------------------------------------------------------------------ 6. glazes over dry paint, where the shadows are still too light or the colour too dull
from goo.palette import lab
seen = painter.see(); Ls_, Lr_ = lab(seen), lab(motif)
too_light = (Ls_[..., 0] - Lr_[..., 0] > 4) & (Lr_[..., 0] < 45)
too_dull = (np.hypot(Lr_[..., 1], Lr_[..., 2]) - np.hypot(Ls_[..., 1], Ls_[..., 2]) > 6) & np.isin(ids, [4, 5, 6])
inside = np.zeros_like(ids, bool)
for oid in (1, 2, 3, 4, 5, 6):                                       # glazes inside things, never along their edges
    inside |= ndimage.binary_erosion(ids == oid, iterations=int(s(6)))
glaze_region = (ndimage.gaussian_filter((too_light | too_dull).astype(float), 2) > 0.5) & inside
opaque = painter.palette
painter.palette = Palette(["ultramarine", "rose", "sienna", "green", "yellow", "red"], steps=8)   # transparent colours, no black
painter.pass_(s(10), oil(0.1), target=0.25, threshold=3.0, blur=0.6, region=glaze_region, curvature=0.6, max_len=8, phi=0.1, drag=0.3, plough=0.1, opacity_bias=0.0, label="glazes", max_strokes=int(900 * SCALE ** 2))
painter.palette = opaque
stage("glazes"); cure_and_fix(); stage("glazes dry")

# ------------------------------------------------------------------ 7. impasto on the highlights
lbl, n = ndimage.label(hl > 0.22)
hb = Brush(cv, max(3, s(7)), bristles=3, exchange=0.5)
for i in range(1, n + 1):
    m = lbl == i
    if m.sum() < s(3) ** 2: continue
    yx = np.argwhere(m); c = yx.mean(0); cov = np.cov(yx.T) if len(yx) > 2 else np.eye(2)
    w_, v_ = np.linalg.eigh(cov); ax = v_[:, -1]; L = max(s(4), 2.2 * np.sqrt(w_[-1]))
    y0, x0 = c - ax * L / 2; y1, x1 = c + ax * L / 2
    col = motif[tuple(yx[np.argmax(hl[m])])]
    mix, _, _ = painter.palette.match(col, painter.see()[int(c[0]), int(c[1])], 0.58 * 4.0 * 0.8)
    hb.charge(Brush.paint("oil", 0.58, 900 * SCALE ** 2 * 4, mix))
    hb.stroke([(x0, y0, 1.0), (0.5 * (x0 + x1), 0.5 * (y0 + y1), 1.0), (x1, y1, 0.5)], speed=0.8, target=5.0)
stage("impasto")

# ------------------------------------------------------------------ 8. signature, and time
sty = Stylus(cv, max(2.0, s(2.6)))
fy = int(np.median(np.nonzero(ids[:, int(W * 0.9)] == 2)[0])) if (ids[:, int(W * 0.9)] == 2).any() else int(H * 0.85)
x0 = W * 0.86
for path in [[(x0, fy - s(8)), (x0 + s(5), fy + s(6)), (x0 + s(10), fy - s(8))]]: sty.stroke(path, speed=0.7)
stage("signed")
cv.L["cure_rate"] = 0.002; cv.wait(500, cure_boost=12.0); cv.L["cure_rate"] = 0.0
stage("cured")

del cv.step
pickle.dump(cv, open(f"{OUT}/canvas.pkl", "wb"))
json.dump(dict(log=log, passes=painter.log, ledger=cv.ledger()), open(f"{OUT}/receipts.json", "w"), indent=1, default=float)
np.savez_compressed(f"{OUT}/frames.npz", t=np.array([f[0] for f in frames]), rgb=np.stack([f[1] for f in frames]))
print(f"done in {time.time() - T0:.0f}s, {len(frames)} frames")
