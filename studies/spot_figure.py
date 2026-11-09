import json, pickle
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from goo.sensors.lidar import CELL, grid

res = json.load(open("out/spot-study.json")); D = pickle.load(open("out/spot-study.pkl", "rb"))
Z, meas = D["Z"], D["meas"]; H, W = Z.shape
names = list(meas.keys())
col = {names[0]: "#9aa0a6", names[1]: "#e76f51", names[2]: "#f4a261", names[3]: "#2a9d8f"}
plt.rcParams.update({"axes.facecolor": "#181818", "figure.facecolor": "#111", "text.color": "#ddd", "axes.labelcolor": "#bbb",
                     "xtick.color": "#999", "ytick.color": "#999", "axes.edgecolor": "#444", "font.size": 10})
fig = plt.figure(figsize=(17, 22))
gs = fig.add_gridspec(4, 3, hspace=0.32, wspace=0.18, height_ratios=[1, 1, 1, 0.9])

# A: flat sweep
ax = fig.add_subplot(gs[0, 0])
for n, rows in res["flat"].items():
    th = [r["theta"] for r in rows]; ax.plot(th, [r["rms"] for r in rows], "-o", ms=3, color=col.get(n, "#8ecae6"), label=n)
ax.plot(th, [r["spread"] for r in res["flat"][names[0]]], "--", color="#666", label="depth spread inside the 0.12 mm spot")
ax.set_yscale("log"); ax.set_xlabel("incidence (deg)"); ax.set_ylabel("RMS range error (um)")
ax.set_title("A  flat white gesso: error vs incidence", loc="left"); ax.legend(fontsize=7.5, facecolor="#222", labelcolor="#ddd")
# B: error vs spread on the painting
ax = fig.add_subplot(gs[0, 1])
for n in names:
    b = res["painting"][n]["by_spread"]; x = [0.5 * (r["lo"] + r["hi"]) for r in b]
    ax.plot(x, [r["rms"] for r in b], "-o", ms=3, color=col[n], label=n)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("true depth spread inside the spot (um)"); ax.set_ylabel("RMS height error of the point (um)")
ax.set_title("B  the painting: error grows with what the spot straddles", loc="left"); ax.legend(fontsize=7.5, facecolor="#222", labelcolor="#ddd")
# C: histograms
ax = fig.add_subplot(gs[0, 2])
bins = np.linspace(0, 600, 121)
for n in names:
    e = np.concatenate([np.abs(s["ez"][s["valid"]]) for s in meas[n].values()]) * 1000
    ax.hist(e, bins, histtype="step", color=col[n], lw=1.4, density=True, label=n)
ax.set_yscale("log"); ax.set_xlabel("|height error| of a point (um)"); ax.set_title("C  the tails: speckle outliers", loc="left")
ax.legend(fontsize=7.5, facecolor="#222", labelcolor="#ddd")

# D-F: merged error maps
truth = Z
def merged(n):
    P = np.concatenate([s["P"][s["valid"]] for s in meas[n].values()]); I = np.concatenate([s["I"][s["valid"]] for s in meas[n].values()])
    s_, c_ = grid(P, P[:, 2] * I, (H, W))[0], None
    j = np.clip((P[:, 0] / CELL).astype(int), 0, W - 1); i = np.clip((P[:, 1] / CELL).astype(int), 0, H - 1); k = i * W + j
    num = np.bincount(k, P[:, 2] * I, H * W); den = np.bincount(k, I, H * W)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan).reshape(H, W)
for c, n in enumerate([names[0], names[1], names[2]]):
    ax = fig.add_subplot(gs[1, c]); err = (merged(n) - truth) * 1000
    im = ax.imshow(err, cmap="RdBu_r", vmin=-80, vmax=80); ax.axis("off")
    cr = res["painting"][n]["cells"]["weighted"]
    ax.set_title(f"{'DEF'[c]}  {n}\n    merged height - truth: RMS {cr['rms']:.1f} um, p99 {cr['p99']:.0f} um", loc="left", fontsize=10)
cb = fig.colorbar(im, ax=fig.axes[-3:], fraction=0.015, pad=0.01); cb.set_label("um", color="#bbb"); cb.ax.tick_params(colors="#999")

# G: cross-section through the impasto
ax = fig.add_subplot(gs[2, 0:2])
row = 70; y = (row + 0.5) * CELL; x = (np.arange(W) + 0.5) * CELL
ax.plot(x, Z[row], color="#fff", lw=1.0, label="true surface", zorder=5)
for n, ms in [(names[1], 2.2), (names[3], 1.6)]:
    for s in meas[n].values():
        P = s["P"][s["valid"]]; m = np.abs(P[:, 1] - y) < 0.06
        ax.plot(P[m, 0], P[m, 2], ".", ms=ms, color=col[n], alpha=0.7)
    ax.plot([], [], ".", color=col[n], label=f"returns: {n}")
ax.set_xlim(32, 50); ax.set_ylim(Z[row, 128:200].min() - 0.15, Z[row].max() + 0.25)
ax.set_xlabel("x (mm)"); ax.set_ylabel("height (mm)"); ax.legend(fontsize=8, facecolor="#222", labelcolor="#ddd", loc="upper right")
ax.set_title(f"G  a row through the highlights (y = {y:.1f} mm): flying pixels at the edges, speckle on the flats", loc="left")
# H: what a single look does to reflectography
ax = fig.add_subplot(gs[2, 2]); crop = (slice(70, 150), slice(70, 150))
P = np.concatenate([s["P"][s["valid"]] for s in meas[names[1]].values()]); I = np.concatenate([s["I"][s["valid"]] for s in meas[names[1]].values()])
P0 = np.concatenate([s["P"][s["valid"]] for s in meas[names[0]].values()]); I0 = np.concatenate([s["I"][s["valid"]] for s in meas[names[0]].values()])
P4 = np.concatenate([s["P"][s["valid"]] for s in meas[names[2]].values()]); I4 = np.concatenate([s["I"][s["valid"]] for s in meas[names[2]].values()])
ims = [grid(p, i, (H, W))[0][crop] for p, i in [(P0, I0), (P, I), (P4, I4)]]
ims = [np.nan_to_num(m / np.nanpercentile(m, 99), nan=0) for m in ims]
ax.imshow(np.concatenate([ims[0], np.ones((80, 3)), ims[1], np.ones((80, 3)), ims[2]], 1), cmap="gray", vmin=0, vmax=1.1); ax.axis("off")
ax.set_title("H  1550 nm intensity over the first pear:\n    averaged spot | 1 coherent look | 4 looks", loc="left", fontsize=10)

# K: table
ax = fig.add_subplot(gs[3, :]); ax.axis("off")
hdr = f"{'':36s}{'point RMS':>11s}{'point p99':>11s}{'points >50um':>14s}{'grid RMS (mean)':>17s}{'grid RMS (I-wtd)':>18s}{'grid RMS (median)':>19s}"
lines = [hdr, "-" * len(hdr)]
for n in names:
    r = res["painting"][n]; c = r["cells"]
    lines.append(f"{n:36s}{r['points_rms']:9.1f}um{r['points_p99']:9.0f}um{r['flying_frac_50um']:13.1%}{c['mean']['rms']:15.1f}um{c['weighted']['rms']:16.1f}um{c['median']['rms']:17.1f}um")
lines += ["", "over the pear, pitch 0.07 mm (points):"]
for spot, o in res["spot"].items():
    for n, v in o.items():
        lines.append(f"  spot {spot:8s} {n:30s} RMS {v['rms']:6.1f} um   p99 {v['p99']:5.0f} um   >50um {v['flying']:6.1%}   RMS where the spot straddles >130 um: {v['rms_mixed'] or float('nan'):6.1f} um")
ax.text(0, 1, "\n".join(lines), family="monospace", fontsize=9.6, va="top", color="#ddd")
fig.suptitle("The spot, measured: a coherent 1550 nm FMCW scan of the pear against ground truth", color="#eee", fontsize=14, x=0.07, ha="left", y=0.995)
fig.savefig("out/spot-study.png", dpi=105, bbox_inches="tight")
print("ok")
