"""The spot, measured: what a coherent FMCW scanner really returns from the pear, against ground truth."""
import json, pickle, time
import numpy as np
from scipy import ndimage
from goo.engine import Canvas
from goo.sensors.lidar import CELL, grid, write_ply
from goo.sensors.coherent import Surface, Geometry
from studies.spot_check import flat, run

T0 = time.time()
L4 = (1530e-9, 1545e-9, 1560e-9, 1575e-9)
CONFIGS = {                                   # name: measure() kwargs
    "averaged spot (my first scanner)": dict(mode="incoherent"),
    "coherent, 100 GHz, 1 look":        dict(bandwidth=100e9),
    "coherent, 100 GHz, 4 looks":       dict(bandwidth=100e9, lams=L4, combine="weighted"),
    "coherent, 1 THz, 4 looks":         dict(bandwidth=1e12, lams=L4, combine="weighted"),
}
res = dict(flat={}, painting={}, spot={})

# ---------------------------------------------------------------- 1. flat white gesso, incidence sweep
angles = list(range(0, 80, 5))
for name, kw in CONFIGS.items(): res["flat"][name] = []
res["flat"]["coherent, 1 THz, 4 looks, 0.06 mm spot"] = []
for th in angles:
    g = run(flat(), th)
    for name, kw in CONFIGS.items():
        e = g.errors(g.measure(**kw)["R"])["e_beam"][g.valid] * 1000
        res["flat"][name].append(dict(theta=th, rms=float(np.sqrt(np.mean(e ** 2))), p99=float(np.percentile(np.abs(e), 99)), spread=float(np.nanmedian(g.spread) * 1000)))
    thr = np.radians(th)
    g2 = Geometry(flat(), (10 - 150 * np.sin(thr), 10, 150 * np.cos(thr)), spacing=0.1, spot=0.06, aim_box=(6, 14, 6, 14), seed=1)
    e = g2.errors(g2.measure(**CONFIGS["coherent, 1 THz, 4 looks"])["R"])["e_beam"][g2.valid] * 1000
    res["flat"]["coherent, 1 THz, 4 looks, 0.06 mm spot"].append(dict(theta=th, rms=float(np.sqrt(np.mean(e ** 2))), p99=float(np.percentile(np.abs(e), 99)), spread=float(np.nanmedian(g2.spread) * 1000)))
print(f"[{time.time()-T0:5.0f}s] flat sweep done", flush=True)

# ---------------------------------------------------------------- 2. the painting
cv = pickle.load(open("out/pear.canvas.pkl", "rb"))
H, W = cv.H, cv.W; Wm, Hm = W * CELL, H * CELL
Z = cv.surface() * CELL; R4 = cv.reflectance(); nir = R4[..., 3]
oilfrac = cv.q[1] / np.maximum(cv.h, 1e-9)
gloss = np.clip(cv.h / 0.3, 0, 1) * np.clip(oilfrac / 0.3, 0, 1)
rough = 1.5 * (1 - gloss) + 0.3 * gloss                         # um: matte gesso / dry watercolour vs an oil film
surf = Surface(Z, nir, rough)
stations = {"left": (-0.64 * 150 + Wm / 2, Hm / 2, 0.77 * 150), "right": (0.64 * 150 + Wm / 2, Hm / 2, 0.77 * 150)}
geo = {k: Geometry(surf, p, spacing=0.14, spot=0.12, seed=i + 3) for i, (k, p) in enumerate(stations.items())}
print(f"[{time.time()-T0:5.0f}s] geometry: {sum(g.B for g in geo.values())} beams x {geo['left'].N} scatterers", flush=True)

def merge(clouds, how="mean"):
    """clouds: list of (P, I). Per-cell merge of both stations' points onto the grid."""
    P = np.concatenate([c[0] for c in clouds]); I = np.concatenate([c[1] for c in clouds])
    j = np.clip((P[:, 0] / CELL).astype(int), 0, W - 1); i = np.clip((P[:, 1] / CELL).astype(int), 0, H - 1); k = i * W + j
    if how == "mean":
        s = np.bincount(k, P[:, 2], H * W); c = np.bincount(k, None, H * W)
        return np.where(c > 0, s / np.maximum(c, 1), np.nan).reshape(H, W)
    if how == "weighted":
        s = np.bincount(k, P[:, 2] * I, H * W); c = np.bincount(k, I, H * W)
        return np.where(c > 0, s / np.maximum(c, 1e-12), np.nan).reshape(H, W)
    if how == "median":
        o = np.argsort(k, kind="stable"); ks, zs = k[o], P[o, 2]
        out = np.full(H * W, np.nan); starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]; ends = np.r_[starts[1:], len(ks)]
        for a, b in zip(starts, ends): out[ks[a]] = np.median(zs[a:b])
        return out.reshape(H, W)

truth_cell = ndimage.uniform_filter(Z, 1)                       # truth at cell resolution
inner = (slice(6, -6), slice(6, -6))
meas = {}
for name, kw in CONFIGS.items():
    st = {}
    for k, g in geo.items():
        m = g.measure(**kw); e = g.errors(m["R"]); st[k] = dict(P=e["P"], ez=e["e_z"], I=m["I"], spread=g.spread, valid=g.valid)
    meas[name] = st
    ez = np.concatenate([s["ez"][s["valid"]] for s in st.values()]) * 1000
    sp = np.concatenate([s["spread"][s["valid"]] for s in st.values()]) * 1000
    cells = {}
    for how in ("mean", "weighted", "median"):
        zg = merge([(s["P"][s["valid"]], s["I"][s["valid"]]) for s in st.values()], how)
        d = (zg - truth_cell)[inner] * 1000; d = d[np.isfinite(d)]
        cells[how] = dict(rms=float(np.sqrt(np.mean(d ** 2))), p99=float(np.percentile(np.abs(d), 99)))
    bins = [0, 20, 40, 60, 90, 130, 200, 300, 600]
    by_spread = []
    for a, b in zip(bins, bins[1:]):
        m_ = (sp >= a) & (sp < b)
        if m_.sum() > 50: by_spread.append(dict(lo=a, hi=b, n=int(m_.sum()), rms=float(np.sqrt(np.mean(ez[m_] ** 2))), p99=float(np.percentile(np.abs(ez[m_]), 99))))
    res["painting"][name] = dict(points_rms=float(np.sqrt(np.mean(ez ** 2))), points_p99=float(np.percentile(np.abs(ez), 99)),
                                 flying_frac_50um=float((np.abs(ez) > 50).mean()), cells=cells, by_spread=by_spread,
                                 mixed_frac=float((sp > 130).mean()))
    print(f"[{time.time()-T0:5.0f}s] {name:34s} points rms {res['painting'][name]['points_rms']:6.1f} p99 {res['painting'][name]['points_p99']:6.0f} "
          f">50um {res['painting'][name]['flying_frac_50um']:.1%}  cells mean/weighted/median {cells['mean']['rms']:.1f}/{cells['weighted']['rms']:.1f}/{cells['median']['rms']:.1f}", flush=True)

# ---------------------------------------------------------------- 3. a smaller spot, over the pear (edges, impasto)
box = (30, 58, 8, 44)
for spot in (0.12, 0.06):
    gs = [Geometry(surf, p, spacing=0.07, spot=spot, aim_box=box, seed=7) for p in stations.values()]
    out = {}
    for name in ("coherent, 100 GHz, 4 looks", "coherent, 1 THz, 4 looks"):
        ez = np.concatenate([g.errors(g.measure(**CONFIGS[name])["R"])["e_z"][g.valid] for g in gs]) * 1000
        sp = np.concatenate([g.spread[g.valid] for g in gs]) * 1000
        out[name] = dict(rms=float(np.sqrt(np.mean(ez ** 2))), p99=float(np.percentile(np.abs(ez), 99)), flying=float((np.abs(ez) > 50).mean()),
                         rms_mixed=float(np.sqrt(np.mean(ez[sp > 130] ** 2))) if (sp > 130).any() else None)
    res["spot"][f"{spot:.2f} mm"] = out
    print(f"[{time.time()-T0:5.0f}s] spot {spot} mm over the pear: {json.dumps(out)}", flush=True)

json.dump(res, open("out/spot-study.json", "w"), indent=1)
pickle.dump(dict(meas={n: {k: dict(P=s["P"], ez=s["ez"], I=s["I"], valid=s["valid"], spread=s["spread"]) for k, s in st.items()} for n, st in meas.items()},
                 Z=Z, nir=nir), open("out/spot-study.pkl", "wb"))
print(f"done {time.time()-T0:.0f}s")
