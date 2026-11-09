"""A pear on a ledge, painted the old way on an 80 x 60 mm gesso panel (320 x 240 cells, 0.25 mm each).

  1. charcoal underdrawing, with a pentimento: a first pear, further left, abandoned and redrawn
  2. fixative (the drawing becomes a stratum)
  3. a sienna watercolour imprimatura, laid in broad wet strokes, dried, sealed
  4. oil: a long blue-grey background painted around the pear; the ledge; the pear in form-following strokes;
     wet-into-wet shadows; short-paint impasto highlights; a sienna stem
  5. sgraffito: a stylus through the wet ledge to the imprimatura (a signature, hatching in the cast shadow)
  6. time: the oil levels where it is long, holds where it is short, and cures
"""
import json, time, pickle
import numpy as np
from skimage import measure
from goo.engine import Canvas
from goo.brush import Brush, Stylus

W, H = 320, 240
OUT = "out"
rng = np.random.default_rng(7)
cv = Canvas(W, H, ["carbon", "white", "ultramarine", "ochre", "sienna", "green"], seed=3,
            laws=dict(cure_rate=0.0))
frames, log = [], []
T0 = time.time()

class Snapper:
    def __init__(self, every): self.every = every; self.last = -999
    def __call__(self):
        if cv.t - self.last >= self.every:
            self.last = cv.t; frames.append((cv.t, (np.clip(cv.reflectance()[..., :3], 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)))
snapper = Snapper(30)
_step = cv.step
def step_and_snap():
    _step(); snapper()
cv.step = step_and_snap

def stage(name):
    led = cv.ledger(); worst = max(abs(v["residual"]) / max(1, v["loaded"]) for v in led["species"].values())
    log.append(dict(stage=name, t=cv.t, substeps=cv.substeps, capped=cv.capped, strata=len(cv.strata),
                    worst_residual=worst, min_q=float(cv.q.min()), max_h=float(cv.h.max()), wall=round(time.time() - T0, 1)))
    print(f"[{time.time() - T0:6.0f}s] {name:34s} t={cv.t:5d} sub={cv.substeps:6d} strata={len(cv.strata)} residual={worst:.1e} max_h={cv.h.max():.2f}", flush=True)
    frames.append((cv.t, (np.clip(cv.reflectance()[..., :3], 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)))

# ---------------------------------------------------------------- shapes
yy, xx = np.mgrid[0:H, 0:W] + 0.5
def pear_field(cx, cy, s=1.0, tilt=0.0):
    """signed-ish field: > 0 inside. A smooth union of a bulb and a shoulder."""
    c, sn = np.cos(tilt), np.sin(tilt)
    X = (xx - cx) * c + (yy - cy) * sn; Y = -(xx - cx) * sn + (yy - cy) * c
    d1 = 1 - np.hypot(X / (44 * s), (Y - 14 * s) / (42 * s))
    d2 = 1 - np.hypot((X + 5 * s) / (25 * s), (Y + 34 * s) / (27 * s))
    k = 0.35; hh = np.clip(0.5 + 0.5 * (d1 - d2) / k, 0, 1)
    return d1 * hh + d2 * (1 - hh) + k * hh * (1 - hh)        # smooth max
PEAR = dict(cx=178, cy=112)
pf = pear_field(**PEAR); pear = pf > 0
first = pear_field(118, 120, 0.82, -0.25)                     # the abandoned first idea
LEDGE = 168

def outline(field, level=0.0):
    c = max(measure.find_contours(field, level), key=len)
    return [(p[1], p[0]) for p in c[::3]]

def recharge(b, paint_fn, below=0.25):
    if b.load.sum() < below * b.full: b.charge(paint_fn())

# ---------------------------------------------------------------- 1. charcoal underdrawing
print("charcoal underdrawing")
coal = Brush(cv, 2.6, bristles=1, exchange=0.6, tooth_only=True, drag=0, plough=0, load={"carbon": 4.0})
def draw(path, target=0.045, pressure=1.0, speed=1.6):
    jit = [(x + rng.normal(0, 0.35), y + rng.normal(0, 0.35)) for x, y in path]
    coal.stroke(jit, speed=speed, pressure=pressure, target=target)
    if coal.load.sum() < 1.0: coal.charge({"carbon": 4.0})
# the first idea: a lighter, searching contour, then abandoned
op = outline(first)
for k in range(0, len(op) - 12, 10): draw(op[k:k + 14], target=0.035, pressure=0.8)
draw([(84, 150), (150, 154)], target=0.03)                    # its shadow, a single stroke
# the drawing as kept: contour twice (a searching line and a firmer one), stem, ledge, hatching
ol = outline(pf)
for rep in range(2):
    for k in range(0, len(ol) - 10, 9): draw(ol[k:k + 12], target=0.04 + 0.02 * rep)
draw([(170, 66), (172, 52), (180, 40)], target=0.06)
draw([(4, LEDGE), (W - 4, LEDGE + 2)], target=0.05); draw([(4, LEDGE + 1), (W - 4, LEDGE + 3)], target=0.03)
for i in range(16):                                            # hatching on the shadow side of the pear
    x0 = 192 + i * 3.2; y0 = 72 + i * 4.5
    seg = [(x0 + t * 14, y0 + t * 30) for t in np.linspace(0, 1, 6)]
    seg = [p for p in seg if pf[int(min(H - 1, p[1])), int(min(W - 1, p[0]))] > 0.06]
    if len(seg) > 1: draw(seg, target=0.04, pressure=0.8)
for i in range(14):                                            # the cast shadow on the ledge
    x0 = 205 + i * 5; draw([(x0, LEDGE + 6), (x0 + 12, LEDGE + 26)], target=0.035, pressure=0.8)
stage("charcoal underdrawing")
cv.fix(); stage("fixative")

# ---------------------------------------------------------------- 2. imprimatura
print("imprimatura")
wash = lambda: Brush.paint("water", 0.06, 12000, {"sienna": 3, "ochre": 1})   # ~0.19 mL
mop = Brush(cv, 44, bristles=12, exchange=0.45, plough=0.2, load=wash()); mop.full = 12000
for row, y in enumerate(np.arange(16, H, 30)):
    xs = (6, W - 6) if row % 2 == 0 else (W - 6, 6)
    recharge(mop, wash, 0.5)
    mop.stroke([(xs[0], y + rng.normal(0, 2)), ((xs[0] + xs[1]) / 2, y + rng.normal(0, 3)), (xs[1], y + rng.normal(0, 2))],
               speed=4.0, pressure=0.9, target=0.55)
stage("imprimatura laid")
cv.wait(260, evap_boost=8.0)
while cv.q[0].max() > 1e-6: cv.wait(40, evap_boost=8.0)
stage("imprimatura dry"); cv.fix(); stage("imprimatura sealed")

# ---------------------------------------------------------------- 3. oil
def oil(phi, mix): return lambda: Brush.paint("oil", phi, 6000, mix)          # ~0.09 mL on the brush

def lay(b, paint_fn, path, target, pressure=1.0, speed=2.6, avoid=None, below=0.3):
    """A stroke that lifts off where it would enter `avoid` (the painter works around a shape)."""
    run = []
    for p in path:
        inside = avoid is not None and avoid[int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1))]
        if inside:
            if len(run) > 1: recharge(b, paint_fn, below); b.stroke(run, speed=speed, pressure=pressure, target=target)
            run = []
        else: run.append(p)
    if len(run) > 1: recharge(b, paint_fn, below); b.stroke(run, speed=speed, pressure=pressure, target=target)

def look_and_fill(region, b, paint_fn, target, angle=0.0, frac=0.35, length=24, max_strokes=80, label="", avoid=None):
    """The painter looks: where the region is still bare, put a stroke through the biggest bare patch."""
    n = 0
    for n in range(max_strokes):
        bare = region & (cv.h < 0.06)
        if bare.sum() < 0.005 * region.sum(): break
        lab, k = ndimage.label(bare)
        sizes = ndimage.sum(bare, lab, range(1, k + 1)); j = int(np.argmax(sizes)) + 1
        y, x = ndimage.center_of_mass(lab == j)
        a = angle + rng.normal(0, 0.25); dx, dy = np.cos(a) * length / 2, np.sin(a) * length / 2
        lay(b, paint_fn, [(x - dx, y - dy), (x, y), (x + dx, y + dy)], target, avoid=avoid)
    print(f"   looked and filled {label}: {n} strokes, bare now {(region & (cv.h < 0.06)).mean() / max(region.mean(), 1e-9):.2%}", flush=True)

def wobble(p0, p1, n=8, amp=1.5):
    t = np.linspace(0, 1, n); x = p0[0] + (p1[0] - p0[0]) * t; y = p0[1] + (p1[1] - p0[1]) * t
    return [(a + rng.normal(0, amp), b + rng.normal(0, amp)) for a, b in zip(x, y)]

print("background")
from scipy import ndimage
keepout = ndimage.binary_erosion(pear, iterations=3)              # paint up to (a little over) the contour
bg = oil(0.46, {"white": 5, "ultramarine": 2, "sienna": 0.6, "carbon": 0.1})
bgb = Brush(cv, 22, bristles=9, load=bg()); bgb.full = 6000
for k, y in enumerate(np.arange(3, LEDGE - 2, 12)):              # horizontal passes, alternating direction
    a, b2 = ((-8, y), (W + 8, y + rng.normal(0, 4))) if k % 2 == 0 else ((W + 8, y), (-8, y + rng.normal(0, 4)))
    lay(bgb, bg, wobble(a, b2, 14), 0.85, avoid=keepout, below=0.6)
halo = outline(pear_field(**PEAR) + 0.10)                          # cutting in round the pear with a smaller brush
cut = Brush(cv, 10, bristles=5, load=bg()); cut.full = 6000
for k in range(0, len(halo) - 8, 7):
    seg = [p for p in halo[k:k + 9] if p[1] < LEDGE - 2]
    if len(seg) > 2: lay(cut, bg, seg, 0.85, pressure=0.9, speed=2.0, avoid=keepout, below=0.6)
bgl = oil(0.46, {"white": 6, "ultramarine": 1.6, "sienna": 0.8})     # a lighter pass top-left: the light comes from there
for k in range(12):
    x0 = 20 + k * 13; lay(bgb, bgl, wobble((x0, 4), (x0 - 50, LEDGE - 50), 4, 1.0), 0.8, pressure=0.7, avoid=keepout)
bg_region = ~pear & (yy < LEDGE - 1)
look_and_fill(bg_region, cut, bg, 0.85, angle=0.0, label="background", avoid=keepout)
stage("background")

print("ledge")
lt = oil(0.5, {"ochre": 3, "white": 2, "sienna": 1})
dk = oil(0.5, {"sienna": 3, "ultramarine": 1, "ochre": 1})
lb = Brush(cv, 18, bristles=8, load=lt()); lb.full = 6000
for y in np.arange(LEDGE + 4, 206, 9): lay(lb, lt, wobble((-6, y), (W + 6, y + rng.normal(0, 1)), 12, 0.8), 0.85, avoid=keepout, below=0.6)
db = Brush(cv, 18, bristles=8, load=dk()); db.full = 6000
for y in np.arange(208, H + 2, 9): lay(db, dk, wobble((W + 6, y), (-6, y), 12, 0.8), 0.85, below=0.6)
look_and_fill(~pear & (yy > LEDGE + 1) & (yy < 206), lb, lt, 0.85, angle=0.0, label="ledge top")
look_and_fill((yy >= 206), db, dk, 0.85, angle=np.pi, label="ledge front")
stage("ledge")
print("cast shadow, wet into wet")
sh = oil(0.48, {"sienna": 2, "ultramarine": 2, "ochre": 1, "green": 0.5})
sb = Brush(cv, 14, bristles=6, load=sh()); sb.full = 6000
for i in range(10):                                               # an ellipse, darkest at the pear's foot
    e = (i + 0.5) / 10; half = np.sqrt(max(0, 1 - (2 * e - 1) ** 2))
    y = LEDGE + 3 + e * 26; xc = 232 + 6 * e
    lay(sb, sh, wobble((xc - 40 * half, y), (xc + 44 * half, y + 1), 6, 0.5), 0.75, pressure=0.6 + 0.4 * half, avoid=keepout)
stage("cast shadow")

print("pear")
body = oil(0.5, {"green": 2, "ochre": 3, "white": 1})
shade = oil(0.5, {"green": 2, "ultramarine": 2, "sienna": 2, "carbon": 0.15})
light = oil(0.52, {"ochre": 3, "white": 3, "green": 0.5})
pb = Brush(cv, 13, bristles=6, load=body()); pb.full = 6000
cx, cy = PEAR["cx"], PEAR["cy"]
outside = ~ndimage.binary_dilation(pear, iterations=1)
def around(rad_frac, a0, a1, n=12):
    """a stroke following the form: an arc at a fraction of the way from the centre to the contour"""
    pts = []
    for a in np.linspace(a0, a1, n):
        # walk out from the bulb centre to the contour along angle a
        for r in np.arange(0, 70, 0.5):
            x, y = cx + r * np.cos(a), cy + 12 + r * np.sin(a) * 1.15
            if not pear[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))]: break
        pts.append((cx + rad_frac * r * np.cos(a), cy + 12 + rad_frac * r * 1.15 * np.sin(a)))
    return pts
for a0 in np.arange(0, np.pi, np.pi / 5):                        # the heart of the bulb first, criss-crossed
    lay(pb, body, [(cx + 22 * np.cos(a0), cy + 14 + 22 * np.sin(a0)), (cx, cy + 14), (cx - 22 * np.cos(a0), cy + 14 - 22 * np.sin(a0))], 0.9, avoid=outside)
for k in range(26):                                                # the body, round the form, never twice the same
    f = rng.uniform(0.2, 0.95); a0 = rng.uniform(-np.pi, np.pi); L = rng.uniform(0.7, 1.5)
    lay(pb, body, around(f, a0, a0 + L, 8), 0.9, avoid=outside)
for y in np.arange(64, 96, 6): lay(pb, body, wobble((150, y), (196, y - 3), 6, 0.6), 0.9, avoid=outside)   # the shoulder
inner = ndimage.binary_erosion(pear, iterations=2)
pb.width = 9
look_and_fill(inner, pb, body, 0.9, angle=np.pi / 2, length=16, label="pear", avoid=outside)
pb.width = 13
stage("pear body")
pb.charge(shade()); pb.width = 11
for k in range(12):                                                # the turning edge, wet into wet
    f = rng.uniform(0.6, 1.0); a0 = rng.uniform(-0.9, 1.0); L = rng.uniform(0.5, 1.0)
    lay(pb, shade, around(f, a0, a0 + L, 7), 0.85, pressure=0.9, avoid=outside)
lay(pb, shade, wobble((190, 66), (200, 98), 6, 0.6), 0.8, avoid=outside)
pb.charge(light()); pb.width = 10
for k in range(8):
    f = rng.uniform(0.25, 0.65); a0 = rng.uniform(3.0, 4.2); lay(pb, light, around(f, a0, a0 + rng.uniform(0.4, 0.9), 6), 1.0, avoid=outside)
stage("pear modelled")

print("impasto highlights (short paint)")
hi = lambda: Brush.paint("oil", 0.58, 600, {"white": 5, "ochre": 1})
hb = Brush(cv, 7, bristles=3, load=hi(), exchange=0.5); hb.full = 600
for (x, y, dx, dy) in [(150, 104, 7, -6), (153, 112, 6, -7), (148, 118, 5, -5), (158, 98, 6, -4), (160, 74, 4, -5), (163, 70, 4, -3)]:
    recharge(hb, hi, 0.5)
    hb.stroke([(x, y, 1.0), (x + dx * 0.5, y + dy * 0.5, 1.0), (x + dx, y + dy, 0.5)], speed=0.8, target=6.0)
hb.charge(Brush.paint("oil", 0.58, 400, {"white": 6, "ochre": 0.5}))
for (x, y) in [(152, 106), (161, 73)]: hb.stroke([(x, y, 1.0), (x + 2, y - 2, 0.4)], speed=0.5, target=8.0)   # the brightest touch
stage("impasto")

print("stem")
st = Brush(cv, 4, bristles=2, load=Brush.paint("oil", 0.5, 300, {"sienna": 3, "carbon": 1}))
st.stroke([(171, 70, 1.0), (173, 56, 0.9), (181, 42, 0.6)], speed=1.0, target=1.0)
stage("stem")

# ---------------------------------------------------------------- 4. sgraffito through the wet ledge
print("sgraffito")
sty = Stylus(cv, 2.6)
def V_sig(x, y, s=1.0):
    return [[(x, y), (x + 5 * s, y + 13 * s), (x + 10 * s, y)]]
for path in V_sig(286, 214): sty.stroke(path, speed=0.7)
for i in range(7):                                            # scratched hatching in the cast shadow
    x0 = 228 + i * 6; sty.stroke([(x0, LEDGE + 8), (x0 + 6, LEDGE + 20)], speed=0.8)
stage("sgraffito")

# ---------------------------------------------------------------- 5. time
print("drying and curing")
snapper.every = 40
cv.L["cure_rate"] = 0.002
cv.wait(400, cure_boost=10.0)
stage("cured")
cv.L["cure_rate"] = 0.0

del cv.step
pickle.dump(cv, open(f"{OUT}/pear.canvas.pkl", "wb"))
json.dump(dict(log=log, ledger=cv.ledger()), open(f"{OUT}/pear-receipts.json", "w"), indent=1, default=float)
np.savez_compressed(f"{OUT}/pear-frames.npz", t=np.array([f[0] for f in frames]), rgb=np.stack([f[1] for f in frames]))
print(f"done in {time.time() - T0:.0f}s, {len(frames)} frames")
