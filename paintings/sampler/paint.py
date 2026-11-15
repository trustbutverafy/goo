"""A sampler of fine features: the 64 x 64 pixel-art plate of motif.py painted in oil at 4 cells per texel
(256 x 256 cells, a 6.4 cm panel) by goo.detail.DetailPainter:

  ground and body colour   the broad motif (every feature two texels wide or less replaced by the colour
                           around it), region by region, each in its own colour along its planned direction
  second coats             wherever the painter still sees a region unlike the motif once dry
  features                 the lines, glyphs, face, mortar and dither, traced as strokes along their chains with
                           brushes as wide as they are, one colour at a time, each dried and fixed before the next
  raised work              the yellow lines, last, in short loaded paint

Writes out/sampler/judge.png (the painting under an overhead light beside the motif, both enlarged by whole
pixels) and prints the colour difference (CIEDE2000, per texel) overall and on the features, the ticks and the
ledger's relative residual. About 3 minutes on two cores.
"""
import os, sys, time, pickle, json
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goo import Canvas
from goo.detail import DetailPainter
from goo.palette import Palette, srgb_to_linear, linear_to_srgb
from motif import plate, C

OUT = os.environ.get("OUT", "out/sampler"); os.makedirs(OUT, exist_ok=True)
F, T0 = 4, time.time()
img, ids = plate()
up = lambda a: np.repeat(np.repeat(a, F, 0), F, 1)
motif = up(srgb_to_linear(img / 255.0)); n = img.shape[0]
PAL = ["white", "carbon", "ultramarine", "yellow", "red", "rose", "ochre", "sienna", "green"]
# the World's climate: oil that would take weeks to cure at painting pace; the painter's lamp makes it hours
cv = Canvas(n * F, n * F, PAL, seed=11, laws=dict(lazy=90, cure_rate=1e-5, flow_sub=0.1, diff_sub=0.1))
PLAN = {0: np.pi / 2, 1: np.pi / 2, 2: 0.0, 3: np.pi / 4, 4: np.pi / 2, 5: np.pi / 2, 6: 0.0}
raised = up(np.all(img == C["yellow"], -1))
painter = DetailPainter(cv, motif, up(ids), Palette(PAL, steps=12), width=2 * F, raised=raised, plan=PLAN, seed=5)
print(f"{len(painter.regions)} regions, {len(painter.chains)} chains, {len(painter.raised)} raised", flush=True)
painter.paint()
for e in painter.log: print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()})
wall = time.time() - T0

# ---- the judge: colour per texel, CIEDE2000 against the motif
seen = cv.reflectance()[..., :3]
texel = seen.reshape(n, F, n, F, 3).mean(axis=(1, 3))
dE = deltaE_ciede2000(rgb2lab(img / 255.0), rgb2lab(linear_to_srgb(texel)))
fine = painter.fine.reshape(n, F, n, F).mean(axis=(1, 3)) > 0.5
led = cv.ledger(); res = max(abs(v["residual"]) / v["loaded"] for v in led["species"].values() if v["loaded"] > 0)
# under an overhead light: the reflectance, shaded by the paint's relief (light from the top, 60 degrees up)
z = cv.surface(); gy, gx = np.gradient(z); nz = 1 / np.sqrt(1 + gx ** 2 + gy ** 2)
l = np.array([0.0, -np.cos(np.radians(60)), np.sin(np.radians(60))])
shade = np.clip((-gx * l[0] - gy * l[1] + l[2]) * nz / l[2], 0, 2)        # 1 where the paint lies flat
lit = (linear_to_srgb(np.clip(seen * shade[..., None], 0, 1)) * 255 + 0.5).astype(np.uint8)
k = 2
pic = np.concatenate([np.repeat(np.repeat(lit, k, 0), k, 1), np.full((n * F * k, 8, 3), 255, np.uint8), np.repeat(np.repeat(img, F * k, 0), F * k, 1)], 1)
Image.fromarray(pic).save(f"{OUT}/judge.png")
Image.fromarray(np.repeat(np.repeat(lit[:n * F // 2, :n * F // 2], 4, 0), 4, 1)).save(f"{OUT}/judge-detail.png")
pickle.dump(cv, open(f"{OUT}/canvas.pkl", "wb"))
stats = dict(dE2000=float(dE.mean()), dE2000_features=float(dE[fine].mean()), dE2000_masses=float(dE[~fine].mean()),
             ticks=cv.t, strata=len(cv.strata), residual=res, capped=cv.capped, wall_s=round(wall))
json.dump(dict(stats=stats, log=painter.log), open(f"{OUT}/receipts.json", "w"), indent=1, default=str)
print(f"colour difference (CIEDE2000, per texel): {stats['dE2000']:.2f} overall, {stats['dE2000_features']:.2f} on the features, "
      f"{stats['dE2000_masses']:.2f} on the masses; {cv.t} ticks, {len(cv.strata)} strata, ledger residual {res:.1e}, "
      f"capped {cv.capped}; {wall:.0f} s")
