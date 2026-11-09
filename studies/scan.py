"""Scan the finished pear with the lidar, then look at what came back."""
import json, pickle, sys
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from goo.engine import Canvas
from goo.sensors.lidar import Scanner, grid, write_ply, CELL, bilinear

import os
CANVAS = os.environ.get("CANVAS", "out/pear.canvas.pkl")
OUT = os.environ.get("OUT", "out"); NAME = os.environ.get("NAME", "pear")
cv = pickle.load(open(CANVAS, "rb"))
H, W = cv.H, cv.W; Wm, Hm = W * CELL, H * CELL

# ---- what the World presents to a sensor: a surface and its optics (never the species)
Z = cv.surface() * CELL                                   # mm
R = cv.reflectance()                                      # (H, W, 4): R, G, B, 1550 nm
nir = R[..., 3]; rgb = R[..., :3]
oilfrac = cv.q[1] / np.maximum(cv.h, 1e-9)
gloss = np.clip(cv.h / 0.3, 0, 1) * np.clip(oilfrac / 0.3, 0, 1)   # a wet or cured oil film is glossy; dried watercolour is matte

# ---- two stations, 150 mm out, 40 degrees off the normal, either side
stations = {}
for name, pos in {"left": (-0.64 * 150 + Wm / 2, Hm / 2, 0.77 * 150), "right": (0.64 * 150 + Wm / 2, Hm / 2, 0.77 * 150)}.items():
    sc = Scanner(pos, spacing=0.14, spot=0.12, sub=4, sigma_range=0.008, seed=hash(name) % 1000)
    stations[name] = sc.scan(Z, nir, gloss, rgb)
    s = stations[name]; print(f"{name}: {s['n_hits']}/{s['n_rays']} returns", flush=True)
# a third, grazing station (78 deg off the normal, from the left), kept out of the merged cloud: a raking scan,
# where impasto does cast lidar shadows
graze = Scanner((Wm / 2 - 150 * np.sin(np.radians(78)), Hm / 2, 150 * np.cos(np.radians(78))), spacing=0.14, spot=0.12, sub=4, seed=5)
gs_ = graze.scan(Z, nir, gloss, rgb); print(f"grazing: {gs_['n_hits']}/{gs_['n_rays']} returns", flush=True)
xyz = np.concatenate([s["xyz"] for s in stations.values()]); I = np.concatenate([s["intensity"] for s in stations.values()])
col = np.concatenate([s["rgb"] for s in stations.values()]); cos = np.concatenate([s["cos"] for s in stations.values()])
# PLY in a right-handed frame (x right, y up the panel, z out of it)
P = xyz.copy(); P[:, 1] = Hm - P[:, 1]
write_ply(f"{OUT}/{NAME}-scan.ply", P, I, col)

# ---- back onto a grid: height, coverage per station, intensity normalised for incidence
zg, cnt = grid(xyz, xyz[:, 2], (H, W))
ig, _ = grid(xyz, I / np.maximum(cos, 0.2), (H, W))
cov = {k: grid(s["xyz"], s["xyz"][:, 2], (H, W))[1] > 0 for k, s in stations.items()}
cov_g = grid(gs_["xyz"], gs_["xyz"][:, 2], (H, W))[1] > 0
holes = np.isnan(zg)
zf = zg.copy(); zf[holes] = ndimage.generic_filter(np.nan_to_num(zg, nan=np.nanmean(zg)), np.mean, 3)[holes]
err = (zg - Z)[~holes]
inner = (slice(4, -4), slice(4, -4))
stats = dict(points=int(len(xyz)), per_station={k: dict(rays=s["n_rays"], returns=s["n_hits"]) for k, s in stations.items()},
             height_rmse_um=float(np.sqrt(np.mean(err ** 2)) * 1000), height_p99_um=float(np.percentile(np.abs(err), 99) * 1000),
             shadowed_left=float(1 - cov["left"][inner].mean()), shadowed_right=float(1 - cov["right"][inner].mean()),
             unseen_merged=float(holes[inner].mean()), shadowed_grazing_78deg=float(1 - cov_g[inner].mean()),
             paint_max_mm=float((cv.surface() - cv.g).max() * CELL), relief_range_mm=float(Z.max() - Z.min()))
print(json.dumps(stats, indent=1))

# ---- renders
def raking(Zmm, albedo, light=(-0.85, -0.35, 0.40), gl=None):
    gy, gx = np.gradient(Zmm, CELL)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.array(light) / np.linalg.norm(light)
    diff = np.clip((n * L).sum(-1), 0, 1)
    sh = 0.35 + 0.65 * diff / max(L[2], 1e-3) * 0.55
    img = albedo * sh[..., None]
    if gl is not None:
        Hh = L + np.array([0, 0, 1.0]); Hh /= np.linalg.norm(Hh)
        img = img + 0.35 * gl[..., None] * np.clip((n * Hh).sum(-1), 0, 1)[..., None] ** 60
    return np.clip(img, 0, 1)

srgb = lambda a: np.clip(a, 0, 1) ** (1 / 2.2)
photo = srgb(rgb)
rake = srgb(raking(Z, rgb, gl=gloss))
plt.imsave(f"{OUT}/{NAME}-photo.png", photo); plt.imsave(f"{OUT}/{NAME}-raking.png", rake)

# peel the layers back (the painting's own stratigraphy, as the World holds it, for comparison)
peel = [srgb(cv.reflectance(layers=k)[..., :3]) for k in (1, 2, 3)]

# oblique view of the point cloud: perspective splats, height exaggerated
def oblique(P, colors, exag=6.0, size=(900, 620), elev=28, azim=-60):
    p = P.copy(); p[:, 2] = p[:, 2] * exag
    c = p.mean(0); p -= c
    a, e = np.radians(azim), np.radians(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    q = p @ Rz.T @ Rx.T
    # camera looks along +y in rotated frame; screen x = q0, screen y = -q2; depth = q1
    dist = 140.0; f = 1600.0
    zc = q[:, 1] + dist; sx = f * q[:, 0] / zc + size[0] / 2; sy = -f * q[:, 2] / zc + size[1] / 2
    order = np.argsort(-zc)
    img = np.ones((size[1], size[0], 3)) * 0.06
    xi, yi = sx[order].astype(int), sy[order].astype(int); cc = colors[order]
    for dx in (0, 1):
        for dy in (0, 1):
            ok = (xi + dx >= 0) & (xi + dx < size[0]) & (yi + dy >= 0) & (yi + dy < size[1])
            img[yi[ok] + dy, xi[ok] + dx] = cc[ok]
    return img
sel = np.random.default_rng(0).choice(len(P), min(len(P), 900000), replace=False)
Pv = xyz[sel].copy(); Pv[:, 1] = Hm - Pv[:, 1]
shade = np.clip(I[sel] / np.percentile(I, 99), 0, 1)
cloud_rgb = oblique(Pv, srgb(col[sel]) * (0.55 + 0.45 * shade[:, None]))
cloud_nir = oblique(Pv, np.repeat(shade[:, None], 3, 1))
plt.imsave(f"{OUT}/{NAME}-cloud-rgb.png", cloud_rgb); plt.imsave(f"{OUT}/{NAME}-cloud-nir.png", cloud_nir)

# ---- the contact sheet
fig = plt.figure(figsize=(16, 21.5), facecolor="#111")
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.95], hspace=0.16, wspace=0.05)
def panel(r, c, img, title, **kw):
    ax = fig.add_subplot(gs[r, c]); ax.imshow(img, **kw); ax.set_title(title, color="#ddd", fontsize=11, loc="left"); ax.axis("off"); return ax
panel(0, 0, photo, "1  the painting (diffuse light)")
panel(0, 1, rake, "2  raking light from the left: the brushwork")
ax = panel(0, 2, np.where(holes, np.nan, ig), "3  lidar intensity at 1550 nm: underdrawing + pentimento", cmap="gray",
           vmin=np.nanpercentile(ig, 1), vmax=np.nanpercentile(ig, 99.5))
hs = raking(zf, np.ones((H, W, 3)) * 0.85)[..., 0]
panel(1, 0, hs, "4  lidar height, hillshaded (merged stations)", cmap="gray")
ax = panel(1, 1, (zf - ndimage.gaussian_filter(zf, 18)) * 1000, "5  lidar height, detrended (um): impasto, burrs, gesso", cmap="magma",
           vmin=-60, vmax=np.nanpercentile((zf - ndimage.gaussian_filter(zf, 18)) * 1000, 99.7))
blind = ~cov_g; blind_img = np.repeat((0.25 + 0.6 * (hs - hs.min()) / (np.ptp(hs) + 1e-9))[..., None], 3, 2)
blind_img[blind] = [0.95, 0.25, 0.2]
panel(1, 2, blind_img, f"6  grazing scan (78 deg, from left): no return in red ({stats['shadowed_grazing_78deg']:.1%}: shadows + dropouts)")
panel(2, 0, peel[0], "7  stratum 1: the charcoal (and the first pear)")
panel(2, 1, peel[1], "8  + stratum 2: the imprimatura")
ax = fig.add_subplot(gs[2, 2]); ax.imshow(cloud_rgb); ax.set_title("9  point cloud, oblique, z x6", color="#ddd", fontsize=11, loc="left"); ax.axis("off")
# cross-sections through the brightest impasto and through the signature
ax = fig.add_subplot(gs[3, 0:2]); ax.set_facecolor("#181818")
row = int(np.argmax(cv.h.max(1)))
rows = {"impasto row": row, "signature row": 220}
for (lab, r), c in zip(rows.items(), ["#f3c26b", "#8ecae6"]):
    x = (np.arange(W) + 0.5) * CELL
    base = cv.g[r] * CELL; ax.fill_between(x, -0.05, base, color="#555", lw=0)
    acc = cv.g[r].copy()
    for k, (s, cc) in enumerate(zip(cv.strata, ["#222", "#b5653a", "#999"])):
        top = acc + s.sum(0)[r]; ax.fill_between(x, acc * CELL, top * CELL, color=cc, lw=0, alpha=0.9); acc = top
    ax.plot(x, Z[r], color=c, lw=1.2, label=f"{lab} (y = {r * CELL:.1f} mm): true surface")
    m = np.abs(xyz[:, 1] - (r + 0.5) * CELL) < 0.07
    ax.plot(xyz[m, 0], xyz[m, 2], ".", ms=1.6, color=c, alpha=0.6, label="lidar returns within 0.07 mm of the row")
ax.set_xlabel("x (mm)", color="#bbb"); ax.set_ylabel("height (mm)", color="#bbb"); ax.tick_params(colors="#999")
ax.legend(fontsize=8, facecolor="#222", labelcolor="#ddd", loc="upper left"); ax.set_title("10  cross-sections: lidar vs truth (layers under the curve: ground, charcoal, imprimatura)", color="#ddd", fontsize=11, loc="left")
ax = fig.add_subplot(gs[3, 2]); ax.axis("off")
txt = (f"lidar  1550 nm FMCW, 2 stations at 150 mm, 40 deg\n"
       f"       spot 0.12 mm, pitch 0.14 mm, sigma 8 um\n\n"
       f"points             {stats['points']:,}\n"
       f"height RMSE        {stats['height_rmse_um']:.1f} um (p99 {stats['height_p99_um']:.0f} um)\n"
       f"blind, left        {stats['shadowed_left']:.2%}\n"
       f"blind, right       {stats['shadowed_right']:.2%}\n"
       f"blind, both        {stats['unseen_merged']:.2%}\n"
       f"no return, 78 deg  {stats['shadowed_grazing_78deg']:.2%}\n"
       f"thickest paint     {stats['paint_max_mm']:.2f} mm\n"
       f"relief range       {stats['relief_range_mm']:.2f} mm")
ax.text(0, 0.95, txt, color="#ddd", family="monospace", fontsize=10.5, va="top")
fig.savefig(f"{OUT}/{NAME}-contact-sheet.png", dpi=110, facecolor="#111", bbox_inches="tight")
json.dump(stats, open(f"{OUT}/{NAME}-scan-stats.json", "w"), indent=1)
print("saved")
