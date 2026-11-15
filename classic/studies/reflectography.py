"""Can a coherent lidar's intensity channel do reflectography? Contrast-to-noise of the abandoned pear's line."""
import pickle, json
import numpy as np
from scipy import ndimage
from goo.classic.engine import Canvas
from goo.classic.sensors.lidar import CELL, grid
from goo.classic.sensors.coherent import Surface, Geometry
cv = pickle.load(open("out/pear.canvas.pkl", "rb")); H, W = cv.H, cv.W; Wm, Hm = W * CELL, H * CELL
surf = Surface.from_canvas(cv); Z, nir = surf.Z_mm, surf.nir   # what the World presents to a sensor (SPEC §6)
box = (18, 42, 26, 46)                                       # mm: the first pear's contour, under the background
g = Geometry(surf, (-0.64 * 150 + Wm / 2, Hm / 2, 0.77 * 150), spacing=0.14, spot=0.12, aim_box=box, seed=2)
coal = cv.strata[0][cv.idx("carbon")]
line = ndimage.binary_dilation(coal > 0.01, iterations=1)
ring = ndimage.binary_dilation(line, iterations=5) & ~ndimage.binary_dilation(line, iterations=2)
j0, j1, i0, i1 = [int(v / CELL) for v in box]; roi = np.zeros((H, W), bool); roi[i1 - (i1 - i0):i1, j0:j1] = True
roi[:] = False; roi[int(box[2] / CELL) + 4:int(box[3] / CELL) - 4, int(box[0] / CELL) + 4:int(box[1] / CELL) - 4] = True
L, Rg = line & roi, ring & roi
def cnr(img):
    return float((np.nanmean(img[Rg]) - np.nanmean(img[L])) / np.nanstd(img[Rg]))
P = g.points(g.t_plane * 0 + np.nan_to_num(g.t_true)); ok = g.valid
out = {}; imgs = {}
truth = nir.copy(); out["true 1550 nm reflectance (no sensor)"] = cnr(truth); imgs["truth"] = truth
for name, kw in [("averaged spot", dict(mode="incoherent")),
                 ("1 coherent look", dict()),
                 ("4 looks (1530-1575 nm)", dict(lams=tuple(np.linspace(1530e-9, 1575e-9, 4)), combine="weighted")),
                 ("16 looks (1500-1600 nm)", dict(lams=tuple(np.linspace(1500e-9, 1600e-9, 16)), combine="weighted")),
                 ("64 looks (1500-1600 nm)", dict(lams=tuple(np.linspace(1500e-9, 1600e-9, 64)), combine="weighted"))]:
    m = g.measure(**kw); Pm = g.points(m["R"])
    img = grid(Pm[ok], m["I"][ok], (H, W))[0]
    out[name] = cnr(img); imgs[name] = img
    sm = ndimage.gaussian_filter(np.nan_to_num(img, nan=np.nanmean(img)), 1.0)
    out[name + ", smoothed 0.25 mm"] = cnr(sm)
    print(f"{name:28s} CNR {out[name]:5.2f}   smoothed {out[name + ', smoothed 0.25 mm']:5.2f}", flush=True)
print("truth CNR", out["true 1550 nm reflectance (no sensor)"])
json.dump(out, open("out/irr-cnr.json", "w"), indent=1)
pickle.dump(dict(imgs=imgs, box=box), open("out/irr-imgs.pkl", "wb"))
