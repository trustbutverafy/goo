"""goo.matter.classic: goo's classic pigments as points in the description space.

For each pigment in goo.engine.PIGMENTS, a description (a dielectric function: eps_inf, two causal Gaussian
bands and one Lorentz oscillator, all of non-negative strength; a log-normal size distribution of spheres)
whose derived K and S in oil (n_s = 1.48) come as close as this family allows to the table's eight numbers.
The descriptions are *fits*: their provenance is the table, not a material, so they are `possible`, never
`real`. The residual is reported per pigment as the rms and the worst factor between derived and tabled
values (log scale, with a floor of FLOOR so that 0.01 vs 0.02 is not a factor-2 failure).

`FITS` holds the fitted parameters (written by `fit_all`, run from studies/matter.py; re-fitting is slow).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from goo.engine import PIGMENTS
from goo.matter.dielectric import Oscillators, Gauss, Lorentz, HC
from goo.matter.morphology import LogNormal, Sphere
from goo.matter.substance import Substance, Provenance
from goo.matter import response

N_OIL = 1.48
FLOOR = 0.01
# parameter vector: eps_inf, (log10 A, E0, s) x 2 causal Gaussians, (log10 f, E0, gamma) Lorentz, log10 d_um, sigma_g
LO = np.array([1.0, -6, 0.5, 0.04, -6, 0.5, 0.04, -6, 0.3, 0.05, np.log10(0.02), 1.15])
HI = np.array([12.0, 1.2, 6.5, 2.5, 1.2, 6.5, 2.5, 1.0, 9.0, 8.0, np.log10(50.0), 2.5])
DENSITY = dict(carbon=1.8, white=4.2, ultramarine=2.35, ochre=3.5, sienna=3.6, rose=1.6, green=2.0, yellow=4.8,
               red=8.1, ivory_black=2.6, dioxazine=1.5)       # g/cm^3, of the real pigments these stand for (order)
BAND_EV = HC / np.array([650.0, 550.0, 450.0, 1550.0])


def build(p, name="fit"):
    p = np.asarray(p, float)
    diel = Oscillators(float(p[0]), (Gauss(float(10 ** p[1]), float(p[2]), float(p[3])),
                                     Gauss(float(10 ** p[4]), float(p[5]), float(p[6])),
                                     Lorentz(float(10 ** p[7]), float(p[8]), float(p[9]))))
    return diel, Sphere(LogNormal(float(10 ** p[10]), float(p[11])))


def target(name):
    e = PIGMENTS[name]; return np.array(e["K"], float), np.array(e["S"], float)


def _logres(K, S, Kt, St):
    return np.r_[np.log10(K + FLOOR) - np.log10(Kt + FLOOR), np.log10(S + FLOOR) - np.log10(St + FLOOR)]


def _resid(p, Kt, St, coarse=True, q=7):
    diel, morph = build(p)
    K, S = response.km_bands(diel, morph, [N_OIL], coarse=coarse, q=q)
    return _logres(K[0], S[0], Kt, St)


# stage A: clear spheres on a grid of (n, d, sigma_g), scattering only; shared by every pigment
GRID_N = np.linspace(1.52, 3.0, 15)
GRID_D = np.geomspace(0.03, 40.0, 24)
GRID_SG = np.array([1.25, 1.6, 2.1])
_GRID_S = None


def scattering_grid():
    global _GRID_S
    if _GRID_S is None:
        out = np.zeros((len(GRID_N), len(GRID_D), len(GRID_SG), 4))
        for j, d in enumerate(GRID_D):
            for k, sg in enumerate(GRID_SG):
                for i, n in enumerate(GRID_N):
                    out[i, j, k] = response.km_bands(Oscillators(n * n, ()), Sphere(LogNormal(d, sg)), [N_OIL], coarse=True, q=5)[1][0]
        _GRID_S = out
    return _GRID_S


def _start_from(n, d, sg, Kt):
    """A start with the particle fixed: Gaussians at the two most absorbing visible bands and a NIR one if the
    table absorbs there, strengths scaled from a probe so each band's K is about right."""
    order = np.argsort(-Kt[:3])
    E1, E2 = BAND_EV[order[0]], BAND_EV[order[1]]
    p = np.array([n * n, -3, E1, 0.35, -3, E2, 0.35, -5, 6.0, 1.0, np.log10(d), sg])
    for slot, b in ((1, order[0]), (4, order[1])):
        diel, morph = build(p)
        K = response.km_bands(diel, morph, [N_OIL], coarse=True, q=5)[0][0]
        p[slot] = np.clip(p[slot] + np.log10(max(Kt[b], 1e-3) / max(K[b], 1e-9)), LO[slot] + 1e-3, HI[slot] - 1e-3)
    return np.clip(p, LO + 1e-6, HI - 1e-6)


def fit_pigment(name, n_starts=4, max_nfev=150):
    Kt, St = target(name)
    G = scattering_grid()
    err = np.sqrt(np.mean((np.log10(G + FLOOR) - np.log10(St + FLOOR)) ** 2, -1))
    idx = np.dstack(np.unravel_index(np.argsort(err, axis=None), err.shape))[0]
    starts, seen = [], set()
    for i, j, k in idx:                                       # the best distinct particles (by size)
        if (j // 3, k) in seen: continue
        seen.add((j // 3, k)); starts.append((GRID_N[i], GRID_D[j], GRID_SG[k]))
        if len(starts) == n_starts: break
    best = None
    for n, d, sg in starts:
        p0 = _start_from(n, d, sg, Kt)
        r = least_squares(_resid, p0, bounds=(LO, HI), args=(Kt, St), max_nfev=max_nfev, diff_step=1e-3)
        if best is None or r.cost < best.cost: best = r
    r = least_squares(_resid, best.x, bounds=(LO, HI), args=(Kt, St, False, 9), max_nfev=40, diff_step=1e-3)
    return r.x


def substance(name):
    """The classic pigment `name` as a description (from FITS)."""
    diel, morph = build(FITS[name], name)
    return Substance(name, diel, morph, density=DENSITY.get(name, 2.0),
                     provenance=Provenance("fit", f"fitted to goo.engine.PIGMENTS[{name!r}] in oil (n_s = {N_OIL})"),
                     note="classic fit")


def residual(name, sub=None):
    """Per pigment: derived K, S in oil, the table's, and the log10 residuals (rms and worst, with FLOOR)."""
    sub = substance(name) if sub is None else sub
    K, S = sub.ks(N_OIL, exact=True); Kt, St = target(name)
    r = _logres(K, S, Kt, St)
    return dict(K=K, S=S, Kt=Kt, St=St, r=r, rms=float(np.sqrt(np.mean(r * r))), worst=float(np.abs(r).max()),
                worst_entry=(["K", "S"][int(np.argmax(np.abs(r))) // 4] + "RGBN"[int(np.argmax(np.abs(r))) % 4]))


# fitted by fit_pigment(name) (studies/matter.py --fit; ~20 s to ~18 min a pigment on a shared core); see docs/MATTER.md
FITS = {
    "carbon": [3.1903781408, 0.456059013, 2.5391111225, 0.8953759228, -5.0052597853, 1.5051165752, 0.4207063396, -0.9885959413, 8.4244116123, 1.9772501334, 1.3391604428, 1.1640950143],
    "white": [2.7004760909, -4.2515130999, 2.6629820909, 0.2818842731, -4.0425711419, 1.1796773576, 1.0046558155, -4.0872012997, 6.0459130002, 1.6786758006, -0.5851735284, 1.2315507514],
    "ultramarine": [2.3326791791, 0.3126284913, 1.0453929582, 0.0807225885, -1.2534038028, 2.0609568974, 0.04, -1.6623329506, 4.4265075927, 0.05, -0.9940672551, 1.15],
    "ochre": [2.5766208597, -2.2289315906, 3.06322905, 0.375997353, -2.9932770804, 2.3184380262, 0.3398692298, -3.2156446163, 2.0478118308, 3.0738125488, 0.937280134, 1.4827456494],
    "sienna": [2.5925632704, -2.0927511378, 2.5708470805, 0.2263256959, -2.8127413834, 2.7464383024, 0.3417472067, -2.8547004004, 3.8726200401, 6.8329753577, 1.1354636869, 1.2235161938],
    "rose": [2.3593115089, -1.8170632088, 2.3786884864, 0.207995478, -4.0321862206, 0.7922797416, 0.3282021454, -4.4162126238, 1.027505719, 0.5570977437, 1.2483147263, 1.4213316051],
    "green": [2.4556217431, -2.0066798171, 2.8314428159, 0.216954383, -1.837228516, 1.5416593292, 0.3671514554, -3.1040928291, 1.5408940122, 4.0379501763, 1.2784674284, 1.3452941913],
    "yellow": [2.7053321996, -1.9281552766, 3.0080101994, 0.3135089554, -3.8204253295, 1.250441372, 2.3642067817, -3.5034686349, 6.0171760906, 1.8042388616, 0.8042438542, 1.3084374005],
    "red": [2.7071808429, -1.5451261136, 2.4757759673, 0.1702576634, -3.6553642718, 1.1545159846, 1.0102018576, -3.6642932578, 5.7801309031, 1.8951721667, 0.9483511218, 1.5481402177],
    "ivory_black": [1.9621277174, -1.081509676, 3.1391508163, 0.7566365856, -1.7471065973, 2.1051810575, 0.1893039245, 0.5485184803, 0.5032065358, 7.9999838198, 1.2219105612, 1.9019905285],
    "dioxazine": [2.3724898757, -1.7968988129, 2.2362044465, 0.2419524479, -3.4651954555, 1.7540936136, 1.228537799, -2.6682830334, 5.9788480174, 0.5762000835, 1.1235875284, 1.5212109655],
}
# rms and worst log10 residual of the eight numbers (K, S in R, G, B, N), at full resolution, rounded up
REPORTED = {
    "carbon": dict(rms=0.0107, worst=0.0177, entry="KN"),
    "white": dict(rms=0.0459, worst=0.0977, entry="SB"),
    "ultramarine": dict(rms=0.0578, worst=0.1246, entry="SG"),
    "ochre": dict(rms=0.1399, worst=0.3104, entry="SN"),
    "sienna": dict(rms=0.094, worst=0.1841, entry="SR"),
    "rose": dict(rms=0.0462, worst=0.1072, entry="SB"),
    "green": dict(rms=0.0397, worst=0.0707, entry="SN"),
    "yellow": dict(rms=0.1238, worst=0.1821, entry="SN"),
    "red": dict(rms=0.1155, worst=0.2844, entry="SR"),
    "ivory_black": dict(rms=0.0001, worst=0.0001, entry="KR"),
    "dioxazine": dict(rms=0.0413, worst=0.091, entry="SB"),
}
