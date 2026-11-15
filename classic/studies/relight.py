"""Photograph a finished painting with the visible-light sensor: an overhead light and two raking lights.

    CANVAS=out/stilllife-0.5/canvas.pkl OUT=out NAME=stilllife python classic/studies/relight.py

The sensor sees only `Surface.from_canvas(cv)`. Each panel is shown at exposure 1 / (lz + ambient), so a flat
white matte cell displays as 1 under every light; anything brighter (a glint) is clipped, and the clipped
fraction is printed in the panel's title. Display: clip(exposure * radiance, 0, 1) ** (1 / 2.2).
"""
import json, os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from goo.classic.surface import Surface
from goo.classic.sensors import relight

CANVAS = os.environ.get("CANVAS", "out/stilllife-0.5/canvas.pkl")
OUT = os.environ.get("OUT", "out"); NAME = os.environ.get("NAME", "stilllife")
AMBIENT = 0.10
LIGHTS = {   # name: (azimuth deg, elevation deg); azimuth 0 = from the left, 90 = from the top
    "overhead, 70 deg up, from the upper left": (45, 70),
    "raking from the left, 15 deg up": (0, 15),
    "raking from the top, 15 deg up": (90, 15),
}

cv = pickle.load(open(CANVAS, "rb"))
surf = Surface.from_canvas(cv)
prep = relight.prepare(surf)
srgb = lambda a: np.clip(a, 0, 1) ** (1 / 2.2)

shots, stats = {}, {}
for name, (az, el) in LIGHTS.items():
    L = relight.light_dir(az, el)
    img, parts = prep.render(L, intensity=1.0, ambient=AMBIENT, return_parts=True)
    k = 1 / (L[2] + AMBIENT)
    shots[name] = srgb(k * img)
    lit = parts["V"] > 0.5
    stats[name] = dict(L=[float(x) for x in L], exposure=float(k), clipped=float((k * img > 1).any(-1).mean()),
                       shadowed=float((~lit).mean()), max_display_linear=float((k * img).max()))
# where the glint is: the overhead shot minus the same shot with gloss off (same surface, matte)
matte = Surface(surf.Z_mm, reflectance=surf.reflectance, gloss=np.zeros(surf.shape), rough_um=surf.rough_um, cell_mm=surf.cell_mm)
L0 = relight.light_dir(*LIGHTS[next(iter(LIGHTS))])
spec = (prep.render(L0, 1.0, AMBIENT) - relight.prepare(matte).render(L0, 1.0, AMBIENT)).max(-1) / (L0[2] + AMBIENT)
stats["gloss"] = dict(mean=float(surf.gloss.mean()), glossy_fraction=float((surf.gloss > 0.5).mean()),
                      overhead_glint_gt_0p1_fraction=float((spec > 0.1).mean()))
print(json.dumps(stats, indent=1))

# a detail: the window around the thickest paint
h = (surf.Z_mm - np.median(surf.Z_mm))
cy, cx = np.unravel_index(np.argmax(h), h.shape); H, W = surf.shape
dh, dw = min(H, 120), min(W, 160); y0 = int(np.clip(cy - dh // 2, 0, H - dh)); x0 = int(np.clip(cx - dw // 2, 0, W - dw))
crop = (slice(y0, y0 + dh), slice(x0, x0 + dw))

fig, axes = plt.subplots(3, 3, figsize=(16, 13.5), facecolor="#111")
for c, (name, img) in enumerate(shots.items()):
    ax = axes[0, c]; ax.imshow(img); ax.axis("off")
    ax.set_title(f"{name}\nexposure {stats[name]['exposure']:.2f}, clipped {stats[name]['clipped']:.2%}", color="#ddd", fontsize=10, loc="left")
    ax.add_patch(plt.Rectangle((x0, y0), dw, dh, fill=False, ec="#f3c26b", lw=0.8))
    ax = axes[1, c]; ax.imshow(img[crop], interpolation="nearest"); ax.axis("off")
    ax.set_title(f"detail ({dw * surf.cell_mm:.0f} x {dh * surf.cell_mm:.0f} mm)", color="#ddd", fontsize=10, loc="left")
ax = axes[2, 0]; ax.imshow(surf.gloss, cmap="gray", vmin=0, vmax=1); ax.axis("off")
ax.set_title("Surface.gloss (0 black .. 1 white): finish of the top film", color="#ddd", fontsize=10, loc="left")
ax = axes[2, 1]; ax.imshow(np.clip(spec, 0, 1), cmap="magma", vmin=0, vmax=1); ax.axis("off")
ax.set_title("overhead: glossy minus matte render (display units)", color="#ddd", fontsize=10, loc="left")
ax = axes[2, 2]; ax.imshow(spec[crop], cmap="magma", vmin=0, vmax=1, interpolation="nearest"); ax.axis("off")
ax.set_title("the same, detail", color="#ddd", fontsize=10, loc="left")
fig.suptitle(f"{NAME}: goo.classic.sensors.relight (viewer overhead, light intensity 1, white sky {AMBIENT})", color="#eee", fontsize=13, x=0.01, ha="left")
fig.savefig(f"{OUT}/{NAME}-relight.png", dpi=100, facecolor="#111", bbox_inches="tight")
json.dump(stats, open(f"{OUT}/{NAME}-relight-stats.json", "w"), indent=1)
print("saved", f"{OUT}/{NAME}-relight.png")
