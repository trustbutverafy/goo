"""goo.classic.optics: what the World looks like. A versioned optics model with its own contract (classic/SPEC.md §6).

An optics model is an object the canvas holds (`Canvas(..., optics="classic" | "v2")`, `cv.optics`). It has
its bands, its species table, and pure functions from the canvas state to what is seen:

    state read     the species volumes of the wet layer and of each stratum, the wet layer's cure (kappa),
                   the water in the ground (store, out of store_cap), the ground type
    reflectance    (H, W, nb) in the model's bands: diffuse illumination, viewed along the normal, specular
                   excluded (a d/0 or 0/d "specular excluded" geometry; by reciprocity the same number)
    finish         (H, W) gloss 0..1: the smooth fraction of the top surface (the specular part reflectance()
                   leaves out)
    colour         (..., nb) band reflectance -> linear sRGB under a named light ("D65" daylight, "A" tungsten)
    preview        the look of a film of given composition over a given body (the palette's question), through
                   the same functions as reflectance(): one code path

None of them flushes, steps or writes anything. A model never reads the canvas's laws beyond `store_cap`.

`classic` is goo's original optics, bit for bit: Kubelka-Munk on pigment volumes alone in R, G, B and 1.5 um
over a fixed ground, no surface. `v2` adds what surrounds the particles, the surface, a porous ground, and
spectral bands with an observer; see V2Optics. The species table SPECIES and the ground table GROUNDS are data:
measured values can replace the plausible ones without touching code (give a species "spectrum" entries, see
`_band_values`).
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from goo.classic import engine as _engine

# ======================================================================================== bands, light, observer
#
# A band set: the visible boxes (nm) whose means the model computes in, plus the 1.5 um band (1550 nm) the
# lidar and reflectography use. Light outside 400-700 nm (380-400, 700-780) is folded into the end boxes for
# the observer, as ASTM E308 folds the tails of its 20 nm tables.

class Bands:
    def __init__(self, name, boxes, labels):
        self.name = name
        self.boxes = [tuple(map(float, b)) for b in boxes]            # visible (lo, hi) nm, in array order
        self.labels = list(labels) + ["N"]
        self.nir = len(self.boxes)                                     # index of the 1.5 um band
        self.n = len(self.boxes) + 1
        self.centres = np.array([0.5 * (lo + hi) for lo, hi in self.boxes] + [1550.0])

    def __repr__(self): return f"Bands({self.name}: {self.n} bands, N at index {self.nir})"
    def __reduce__(self): return (_bands_by_name, (self.name,))      # pickles by name


CLASSIC_BANDS = Bands("classic", [(600, 700), (500, 600), (400, 500)], ["R", "G", "B"])
SPECTRAL15 = Bands("spectral15", [(400 + 20 * i, 420 + 20 * i) for i in range(15)], [f"{410 + 20 * i}" for i in range(15)])
_BANDS = {b.name: b for b in (CLASSIC_BANDS, SPECTRAL15)}
def _bands_by_name(name): return _BANDS[name]

# fine wavelength grid for every band integral: 1 nm midpoints, 380.5 .. 779.5
_LAM = np.arange(380, 780) + 0.5


def _cie_cmf(lam):
    """CIE 1931 2-degree colour matching functions, the multi-lobe analytic fit of Wyman, Sloan & Shirley (2013),
    'Simple analytic approximations to the CIE XYZ colour matching functions', JCGT 2(2). Error vs the table:
    well under 1% of peak, adequate for 20 nm bands."""
    def g(x, mu, s1, s2): s = np.where(x < mu, s1, s2); return np.exp(-0.5 * ((x - mu) / s) ** 2)
    x = 1.056 * g(lam, 599.8, 37.9, 31.0) + 0.362 * g(lam, 442.0, 16.0, 26.7) - 0.065 * g(lam, 501.1, 20.4, 26.2)
    y = 0.821 * g(lam, 568.8, 46.9, 40.5) + 0.286 * g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * g(lam, 437.0, 11.8, 36.0) + 0.681 * g(lam, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


# CIE standard illuminant D65, relative SPD at 10 nm, 380..780 (CIE 015:2018, Table T.1)
_D65_10 = np.array([49.9755, 54.6482, 82.7549, 91.486, 93.4318, 86.6823, 104.865, 117.008, 117.812, 114.861,
                    115.923, 108.811, 109.354, 107.802, 104.790, 107.689, 104.405, 104.046, 100.000, 96.3342,
                    95.788, 88.6856, 90.0062, 89.5991, 87.6987, 83.2886, 83.6992, 80.0268, 80.2146, 82.2778,
                    78.2842, 69.7213, 71.6091, 74.349, 61.604, 69.8856, 75.087, 63.5927, 46.4182, 66.8054, 63.3828])


def _planck(lam_nm, T):
    c2 = 1.4388e7                                                    # nm K
    return lam_nm ** -5.0 / np.expm1(c2 / (lam_nm * T))


LIGHTS = {
    "D65": lambda lam: np.interp(lam, np.arange(380, 790, 10), _D65_10),       # daylight (CIE D65, tabulated)
    "A": lambda lam: _planck(lam, 2856.0),                                      # tungsten (CIE A: Planck, 2856 K)
}
LIGHTS["daylight"] = LIGHTS["D65"]; LIGHTS["tungsten"] = LIGHTS["A"]

_XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])  # IEC 61966-2-1
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def _light_spd(light):
    if callable(light): return light
    if isinstance(light, str): return LIGHTS[light]
    raise TypeError(f"light must be a name in goo.classic.optics.LIGHTS ({sorted(LIGHTS)}) or a function of nm")


@lru_cache(maxsize=64)
def _observer_matrix(bands_name, light_key, adapt):
    """(3, nb) matrix: band reflectance -> linear sRGB, under `light`, normalised so that a perfect white under
    D65 is exactly (1, 1, 1) and every light has the same luminance for white. The 1.5 um band gets weight 0.
    adapt: von Kries adaptation (Bradford) of the light's white to D65, as an eye adapts; otherwise the camera
    is balanced for daylight and tungsten looks orange."""
    b = _BANDS[bands_name]; spd = _light_spd(_LIGHT_KEYS[light_key])
    cmf = _cie_cmf(_LAM)
    E = spd(_LAM)
    W = np.zeros((3, b.n))
    for j, (lo, hi) in enumerate(b.boxes):
        lo_ = 380.0 if j == 0 else lo; hi_ = 780.0 if j == len(b.boxes) - 1 else hi   # fold the tails
        m = (_LAM >= lo_) & (_LAM < hi_)
        W[:, j] = (cmf[m] * E[m, None]).sum(0)
    W /= W[1].sum()                                                  # Y of a perfect white = 1
    if adapt:
        Ed = LIGHTS["D65"](_LAM); wd = (cmf * Ed[:, None]).sum(0); wd /= wd[1]
        ws = W.sum(1)
        M = np.linalg.inv(_BRADFORD) @ np.diag((_BRADFORD @ wd) / (_BRADFORD @ ws)) @ _BRADFORD
        W = M @ W
    rgb = _XYZ_TO_SRGB @ W
    if bands_name != "classic":
        ref = _XYZ_TO_SRGB @ _observer_xyz_d65(bands_name)
        rgb = rgb / ref.sum(1, keepdims=True)                        # D65 white -> exactly (1, 1, 1)
    return rgb


def _observer_xyz_d65(bands_name):
    b = _BANDS[bands_name]; cmf = _cie_cmf(_LAM); E = LIGHTS["D65"](_LAM); W = np.zeros((3, b.n))
    for j, (lo, hi) in enumerate(b.boxes):
        lo_ = 380.0 if j == 0 else lo; hi_ = 780.0 if j == len(b.boxes) - 1 else hi
        m = (_LAM >= lo_) & (_LAM < hi_); W[:, j] = (cmf[m] * E[m, None]).sum(0)
    return W / W[1].sum()


_LIGHT_KEYS = {}
def _light_key(light):
    k = light if isinstance(light, str) else f"fn{id(light)}"
    _LIGHT_KEYS[k] = light
    return k


def colour(R, bands, light="D65", adapt=False):
    """Band reflectance (..., nb) -> linear sRGB (..., 3) under `light`, unclipped.

    Spectral bands: the CIE 1931 observer (analytic fit) and the light, integrated over each band; white under
    D65 is (1, 1, 1). Classic bands are already R, G, B reflectances: under D65 they are returned as they are
    (bit for bit); under another light each channel is scaled by that light's white in sRGB (a camera-RGB
    approximation: classic has no spectrum to do better), and `adapt` undoes the scaling."""
    if isinstance(bands, str): bands = _BANDS[bands]
    R = np.asarray(R, float)
    if R.shape[-1] != bands.n: raise ValueError(f"reflectance has {R.shape[-1]} bands, the model {bands!r} has {bands.n}")
    if bands.name == "classic":
        if (isinstance(light, str) and LIGHTS.get(light) is LIGHTS["D65"]) or adapt: return R[..., :3]
        w = white(light, "spectral15")
        return R[..., :3] * w
    M = _observer_matrix(bands.name, _light_key(light), bool(adapt))
    return R @ M.T


def white(light="D65", bands="spectral15", adapt=False):
    """Linear sRGB of a perfect white under `light` (equal luminance; (1, 1, 1) for D65)."""
    b = _BANDS[bands] if isinstance(bands, str) else bands
    return colour(np.ones(b.n), b, light, adapt)


# ======================================================================================== interfaces and films

def km_layer(K, S):
    """Kubelka-Munk reflectance and transmittance of a layer of total absorption K and scattering S (both times
    thickness). The same formula as Canvas.km_layer (classic), with exact limits: R = 0, T = 1 where K = S = 0."""
    R, T = _engine.Canvas.km_layer(K, S)
    empty = (K == 0) & (S == 0)
    if np.any(empty): R = np.where(empty, 0.0, R); T = np.where(empty, 1.0, T)
    return R, T


def km_rinf(k):
    """R-infinity of a semi-infinite KM medium from k = K/S: 1 + k - sqrt(k^2 + 2k)."""
    return 1 + k - np.sqrt(k * k + 2 * k)


def km_k(Rinf):
    """K/S of a semi-infinite medium of reflectance Rinf (inverse of km_rinf)."""
    return (1 - Rinf) ** 2 / (2 * Rinf)


def over(top, below):
    """Compose layer `top` over `below`. A layer is (Rf, Rb, Tf, Tb): reflectance from above and from below,
    transmittance downward and upward (diffuse light). The scattering-matrix composition of Kubelka (1954),
    associative: over(a, over(b, c)) == over(over(a, b), c) up to round-off."""
    Rf1, Rb1, Tf1, Tb1 = top; Rf2, Rb2, Tf2, Tb2 = below
    d = 1 / (1 - Rb1 * Rf2)
    return (Rf1 + Tf1 * Tb1 * Rf2 * d, Rb2 + Tb2 * Tf2 * Rb1 * d, Tf1 * Tf2 * d, Tb2 * Tb1 * d)


def fresnel_normal(n):
    """Reflectance at normal incidence from outside (air) onto index n."""
    return ((n - 1) / (n + 1)) ** 2


def _fresnel_unpolarised(n, cos_i):
    """Fresnel reflectance from air into index n (>= 1) at incidence cosine cos_i."""
    sin_t = np.sqrt(np.maximum(1 - cos_i ** 2, 0)) / n; cos_t = np.sqrt(np.maximum(1 - sin_t ** 2, 0))
    rs = (cos_i - n * cos_t) / (cos_i + n * cos_t); rp = (n * cos_i - cos_t) / (n * cos_i + cos_t)
    return 0.5 * (rs * rs + rp * rp)


_NGRID = np.linspace(1.0, 3.0, 2001)
def _diffuse_external(ngrid):
    th = (np.arange(4000) + 0.5) * (np.pi / 2) / 4000; c = np.cos(th); w = 2 * c * np.sin(th) * (np.pi / 2) / 4000
    return np.array([(_fresnel_unpolarised(n, c) * w).sum() for n in ngrid])
_RE_GRID = _diffuse_external(_NGRID)


def interface(n, smooth):
    """The top surface as a layer (Rf, Rb, Tf, Tb) for the reflectance() geometry, index n (>= 1), smooth 0..1.

    Light enters along the normal (reciprocity: the same as diffuse light viewed along the normal). It reflects
    F0 = ((n - 1)/(n + 1))^2 at the surface: the smooth part of that is specular and leaves the measurement
    (finish() reports it), the rough part (1 - smooth) is diffuse and counted. Diffuse light inside meets the
    surface from below with the internal diffuse reflectance r_i = 1 - (1 - r_e)/n^2 (r_e the hemispherical
    external reflectance, Fresnel integrated over a cosine-weighted hemisphere; Stokes' relation). Composed over
    a body of reflectance Rb this is the Saunderson (1942) correction:
        R = (1 - smooth) F0 + (1 - F0)(1 - r_i) Rb / (1 - r_i Rb).
    n = 1: no surface, R = Rb exactly."""
    F0 = fresnel_normal(n)
    re = np.interp(n, _NGRID, _RE_GRID)
    ri = 1 - (1 - re) / (n * n)
    return ((1 - smooth) * F0, ri, 1 - F0, 1 - ri)


# ---- scattering by index contrast

def contrast_scattering(n_p, n_s):
    """Transport scattering of a particle of index n_p in a surrounding of index n_s, up to a constant: the
    Lorentz-Lorenz contrast factor ((m^2 - 1)/(m^2 + 2))^2, m = n_p / n_s. It is the index dependence of
    Rayleigh scattering (Bohren & Huffman 1983, eq. 5.8) and, for m near 1, the (m - 1)^2 law that governs the
    transport (diffuse) scattering of particles of any size, whose scattering is otherwise forward; the
    coatings rule that hiding goes as (n_p - n_b)^2 is the same law. Zero when n_p = n_s; it saturates
    (towards 1) at high contrast. It carries no size or wavelength dependence: those are in the table's S."""
    m2 = (np.asarray(n_p, float) / np.asarray(n_s, float)) ** 2
    return ((m2 - 1) / (m2 + 2)) ** 2


# ======================================================================================== the species table
#
# role: "volatile" (a carrier that leaves, and what the ground soaks up: water), "binder" (stays and becomes the film's continuous phase: oil,
# gum, acrylic polymer), "pigment" (a dispersed solid), "void" (air). A binder may be "dispersed_in" a carrier
# (an emulsion: droplets in that carrier until they pack and coalesce). n: refractive index (sodium D line).
# K, S: per 0.025 mm of the pure species in the reference surrounding (oil), in classic bands (R, G, B, N);
# pigments take theirs from goo.classic.engine.PIGMENTS. "src" says what is measured and what is plausible.
# A species may carry "spectrum": {"K": (nm, values), "S": (nm, values)}, measured visible spectra in the
# reference surrounding: they then replace the parametric curves, and no code changes. "yellows": the
# species' K grows with cure as K (1 + 2 min(kappa / KAPPA_FULL, 1)) (linseed; fully yellowed in a stratum).

REFERENCE = "oil"        # the reference surrounding of the table's K and S (the entry marked reference=True)
PHI_PACK = 0.64          # packing fraction of a pigment bed: its void capacity is bed (1 - PHI_PACK)/PHI_PACK.
                         # Random close packing of spheres (Scott & Kilgour 1969), goo's own phi_m; it makes the
                         # critical pigment volume concentration 0.64 (real paints: 0.5-0.65)
AIR_N = 1.0

SPECIES = {
    "air":     dict(role="void", n=1.000, src="measured"),
    "water":   dict(role="volatile", n=1.333, K=[0.0, 0.0, 0.0, 0.04], S=[0, 0, 0, 0],
                    src="n measured; K at 1.5 um from absorption ~8 /cm (Hale & Querry 1973), K ~ 2 alpha; visible K 0"),
    "oil":     dict(role="binder", n=1.48, K=[0.02, 0.03, 0.06, 0.01], S=[0, 0, 0, 0], yellows=True, reference=True,
                    src="n measured (linseed 1.477-1.48); K is goo's classic yellowing, plausible"),
    "gum":     dict(role="binder", n=1.47, K=[0.002, 0.003, 0.006, 0.002], S=[0, 0, 0, 0],
                    src="gum arabic, dry; n plausible (1.45-1.48 reported); K plausible"),
    "acrylic": dict(role="binder", dispersed_in="water", n=1.49, K=[0.001, 0.001, 0.002, 0.001],
                    S=[0.0015, 0.002, 0.003, 0.0003],
                    src="acrylic polymer (latex); n 1.49 (PMMA, measured order of magnitude); S in oil plausible"),
    # pigments: K, S from goo.classic.engine.PIGMENTS (classic, in oil); n plausible unless said
    "white":       dict(role="pigment", n=2.50, src="titanium/lead-white-like; plausible (rutile 2.7, lead white 2.0)"),
    "carbon":      dict(role="pigment", n=2.40, src="effective index of an absorbing particle, plausible (carbon ~2.0 + 1.0i)"),
    "ultramarine": dict(role="pigment", n=1.58, src="effective, plausible: textbook 1.50, raised for the inhomogeneity "
                                                    "that makes it scatter in oil at all"),
    "ochre":       dict(role="pigment", n=1.95, src="goethite + clay, plausible"),
    "sienna":      dict(role="pigment", n=1.90, src="plausible"),
    "rose":        dict(role="pigment", n=1.57, src="madder lake on alumina hydrate, plausible"),
    "green":       dict(role="pigment", n=1.65, src="transparent green, plausible"),
    "yellow":      dict(role="pigment", n=2.40, src="cadmium-like (CdS ~2.5), plausible"),
    "red":         dict(role="pigment", n=2.80, src="vermilion-like (HgS ~2.9), plausible"),
    "ivory_black": dict(role="pigment", n=1.65, src="hydroxyapatite + carbon, plausible"),
    "dioxazine":   dict(role="pigment", n=1.70, src="organic, plausible"),
}

# Grounds: porous scatterers. R: reflectance of the dry ground as seen (with its own surface), classic bands;
# n_solid: index of what scatters (cellulose; chalk and white); bound: fraction of the pore space filled for
# good with a binder of index n_bound (size, glue). Water in the ground (store / store_cap) fills the rest.
GROUNDS = {
    "gesso": dict(R=list(_engine.GROUND_RGBN), n_solid=2.0, bound=0.55, n_bound=1.50,
                  src="R is goo's classic ground; chalk + white in glue: n_solid, bound plausible"),
    "paper": dict(R=list(_engine.GROUND_RGBN), n_solid=1.53, bound=0.05, n_bound=1.50,
                  src="R as classic; cellulose n 1.53 (measured order); lightly sized: bound plausible"),
}

KAPPA_FULL = 0.8        # cure at which linseed's yellowing is complete (fix()'s default cure_min): wet and fixed
                        # films of the same cure look the same (continuity at fix)
H_COVER = 0.02          # film volume (units, 5 um) over which a film covers what is below it at the surface; plausible
E_SMOOTH = 0.10         # clear liquid or binder above the pigment bed (units, 25 um) that makes a surface fully
                        # smooth (levels the ground's tooth and the brush's texture); plausible


def _entry(table, name):
    if name not in table:
        raise KeyError(f"species {name!r} has no optical constants in goo.classic.optics.SPECIES (role, n, d_um; K, S "
                       f"for a pigment come from goo.classic.engine.PIGMENTS): add them to use it with optics='v2'")
    e = dict(table[name])
    if e["role"] == "pigment" and not ("K" in e and "S" in e):
        p = _engine.PIGMENTS[name]; e.setdefault("K", p["K"]); e.setdefault("S", p["S"])
    return e


# ---- smooth spectra from the classic band values: exp(c0 + c1 x + c2 x^2), x = (nm - 550)/100, fitted so its
# means over the classic boxes (B 400-500, G 500-600, R 600-700 nm) equal the table's R, G, B exactly. The
# log-quadratic is the positive analogue of Jakob & Hanika's (2019) sigmoid-quadratic spectral upsampling.

_VIS = np.arange(400, 700) + 0.5
_X = (_VIS - 550.0) / 100.0
_BOXIDX = [(_VIS >= 600) & (_VIS < 700), (_VIS >= 500) & (_VIS < 600), (_VIS >= 400) & (_VIS < 500)]   # R, G, B


@lru_cache(maxsize=4096)
def fit_curve(rgb):
    """Coefficients c (3,) of exp(c0 + c1 x + c2 x^2) whose classic-box means are rgb (R, G, B), by Newton."""
    v = np.maximum(np.array(rgb, float), 1e-9)
    xc = np.array([1.0, 0.0, -1.0])                                   # box centres in x: R, G, B
    c = np.linalg.solve(np.stack([np.ones(3), xc, xc ** 2], 1), np.log(v))
    P = np.stack([np.ones_like(_X), _X, _X ** 2], 1)
    for _ in range(60):
        f = np.exp(P @ c)
        m = np.array([f[b].mean() for b in _BOXIDX])
        J = np.array([(f[b, None] * P[b]).mean(0) for b in _BOXIDX])
        r = np.log(m) - np.log(v)
        if np.abs(r).max() < 1e-14: break
        c = c - np.linalg.solve(J / m[:, None], r)
    return tuple(c)


def curve(rgb, lam=_VIS):
    c = fit_curve(tuple(float(x) for x in rgb)); x = (np.asarray(lam) - 550.0) / 100.0
    return np.exp(c[0] + c[1] * x + c[2] * x * x)


def fit_reflectance(rgb):
    """A smooth reflectance spectrum (on _VIS) whose classic-box means are rgb: the sigmoid-quadratic of Jakob &
    Hanika (2019), s(z) = 1/2 + z / (2 sqrt(1 + z^2)), z quadratic in x; fitted by Newton."""
    v = np.clip(np.array(rgb, float), 1e-6, 1 - 1e-6)
    zt = (2 * v - 1) / (2 * np.sqrt(v * (1 - v)))                    # s^-1(v)
    xc = np.array([1.0, 0.0, -1.0]); P = np.stack([np.ones_like(_X), _X, _X ** 2], 1)
    c = np.linalg.solve(np.stack([np.ones(3), xc, xc ** 2], 1), zt)
    for _ in range(80):
        z = P @ c; s = 0.5 + z / (2 * np.sqrt(1 + z * z)); ds = 0.5 * (1 + z * z) ** -1.5
        m = np.array([s[b].mean() for b in _BOXIDX]); J = np.array([(ds[b, None] * P[b]).mean(0) for b in _BOXIDX])
        r = m - v
        if np.abs(r).max() < 1e-15: break
        c = c - np.linalg.solve(J, r)
    z = P @ c
    return 0.5 + z / (2 * np.sqrt(1 + z * z))


def _band_means(spec_vis, bands):
    """Means of a spectrum sampled on _VIS over each visible box of `bands` (exact partitions of the classic boxes)."""
    return np.array([spec_vis[(_VIS >= lo) & (_VIS < hi)].mean() for lo, hi in bands.boxes])


def _band_values(e, key, bands):
    """K or S of a table entry in `bands` (nb,). Classic bands: the table's numbers. Spectral bands: a measured
    spectrum if the entry has one, else the smooth curve through the table's R, G, B; the N band from the table."""
    v = list(map(float, e[key]))
    if bands.name == "classic": return np.array(v)
    spec = e.get("spectrum", {}).get(key)
    if spec is not None: vis = np.interp(_VIS, np.asarray(spec[0], float), np.asarray(spec[1], float))
    elif max(v[:3]) == 0: vis = np.zeros_like(_VIS)
    else: vis = curve(v[:3])
    return np.r_[_band_means(vis, bands), v[3]]


# ======================================================================================== the models

def state(cv):
    """What an optics model may read of a canvas, without touching the watched public arrays (no rescan, no
    change to the canvas at all): species, size, the wet layer, the strata (compact), the wet layer's cure,
    the water in the ground and its capacity, the ground type."""
    return dict(species=cv.species, nc=cv.nc, H=cv.H, W=cv.W, wet=cv._q, strata=cv.strata, kappa=cv._kappa,
                store=cv._store, store_cap=cv.L["store_cap"], ground=getattr(cv, "ground", "gesso"))


class ClassicOptics:
    """goo's original optics, bit for bit (classic/SPEC.md §6 as it was): Kubelka-Munk per layer on pigment volumes in R,
    G, B, N, composed bottom-up over GROUND_RGBN with R = R1 + T1^2 Rb / (1 - R1 Rb); linseed yellows with cure;
    no surface, no surroundings; finish from the oil fraction of the top film."""
    name = "classic"
    bands = CLASSIC_BANDS

    def __eq__(self, other): return type(other) is type(self)
    def __hash__(self): return hash(self.name)

    @staticmethod
    def _KS(species, nc, vol, kappa):
        K = np.zeros(vol.shape[1:] + (4,)); S = np.zeros(vol.shape[1:] + (4,))
        for i, s in enumerate(species[nc:]):
            p = _engine.PIGMENTS[s]; v = vol[nc + i][..., None]
            K += v * np.array(p["K"]); S += v * np.array(p["S"])
        K += vol[1][..., None] * np.array([0.02, 0.03, 0.06, 0.01]) * (1 + 2 * kappa)    # linseed yellows a touch
        return K, S

    def reflectance(self, cv, gain=10.0, layers=None, light=None, adapt=False):
        st = state(cv); H, W = st["H"], st["W"]; sp, nc = st["species"], st["nc"]
        Rb = np.broadcast_to(_engine.GROUND_RGBN, (H, W, 4)).copy()
        stack = list(range(len(st["strata"]))) + [None]
        if layers is not None: stack = stack[:layers]
        for i in stack:
            if i is None: K, S = self._KS(sp, nc, st["wet"] * gain, st["kappa"][..., None])
            else:
                y0, y1, x0, x1, blk = st["strata"].block(i)
                Kb, Sb = self._KS(sp, nc, blk * gain, 1.0)
                K = np.zeros((H, W, 4)); S = np.zeros((H, W, 4))
                K[y0:y1, x0:x1] = Kb; S[y0:y1, x0:x1] = Sb
            R, T = _engine.Canvas.km_layer(K, S)
            Rb = R + T * T * Rb / (1 - R * Rb)
        R = np.clip(Rb, 0, 1)
        return R if light is None else colour(R, self.bands, light, adapt)

    def finish(self, cv):
        st = state(cv); oil = st["species"].index("oil")
        gloss = np.zeros((st["H"], st["W"]))
        for i in range(len(st["strata"])):
            y0, y1, x0, x1, blk = st["strata"].block(i)
            gloss[y0:y1, x0:x1] = np.where(blk.sum(0) > _engine.HAS_PAINT, _engine.film_gloss(blk, oil), gloss[y0:y1, x0:x1])
        return np.where(st["wet"].sum(0) > _engine.HAS_PAINT, _engine.film_gloss(st["wet"], oil), gloss)

    def colour(self, R, light="D65", adapt=False): return colour(R, self.bands, light, adapt)

    def preview(self, species, film, under, gain=10.0, wet=True, kappa=0.0):
        """Classic look of a film (species, ...) over a body `under` (..., 4): the palette's question."""
        K, S = self._KS(list(species), len(_engine.CARRIERS), np.asarray(film, float) * gain, kappa if wet else 1.0)
        R, T = _engine.Canvas.km_layer(K, S)
        return np.clip(R + T * T * under / (1 - R * under), 0, 1)


class V2Optics:
    """Optics v2: what surrounds the particles, one surface, a porous ground, 15 visible bands + 1.5 um.

    Per film (the wet layer, each stratum), from its composition alone:
      continuous phase   the carriers and binders (water, oil, gum, and a dispersed binder's coalesced part),
                         index n_c = volume-weighted mean of their indices
      bed                the pigments and a dispersed binder's droplets; void capacity V = bed (1 - P)/P,
                         P = PHI_PACK. The continuous phase fills the voids up to V; whatever is left is air:
                         air fraction a = 1 - min(L, V)/V
      emulsion           a binder dispersed in a carrier coalesces as it packs: its coalesced fraction is
                         clip((b/(b + w) - P)/(1 - P), 0, 1) of polymer b in carrier w; the rest is droplets
      scattering         S_j = v_j S_ref_j [(1 - a) C(n_j, n_c) + a C(n_j, 1)] / C(n_j, n_oil): the table's S
                         (in oil) scaled by the index contrast with what is around the particle, C the
                         Lorentz-Lorenz contrast factor ((m^2 - 1)/(m^2 + 2))^2, m = n_j / n_surround
                         (contrast_scattering). It has no size or wavelength dependence: those stay in the
                         table's S. Zero when indices match; 1 in oil.
      absorption         K_j = v_j K_ref_j (pigments, water at 1.5 um, linseed yellowing with cure)
      surface            closure c = clip(L / V, 0, 1); index n = c n_c + (1 - c) n_void, n_void = 1 (the open
                         bed has no interface of its own where underbound); smooth = clip(excess / E_SMOOTH, 0, 1),
                         excess = (L - V)+ the clear phase above the bed. Each film covers what is below it by
                         clip(h / H_COVER, 0, 1), blending n and smooth with the surface beneath.
    Ground: a semi-infinite KM scatterer whose dry body is the table's R seen through its own (rough) surface,
    inverted with Saunderson; its scattering scales with the contrast between its solid and its pore filler
    (bound binder, air, and water as store / store_cap fills it).
    Composition: KM layers bottom-up over the ground, then the surface (Saunderson); reflectance() excludes the
    specular part, finish() is the surface's smooth fraction.

    Constructor: bands ("spectral15" or "classic"), table / grounds (species and ground tables, defaults SPECIES,
    GROUNDS), surface (False: no interface, n = 1 everywhere).
    """
    name = "v2"

    def __init__(self, bands="spectral15", table=None, grounds=None, surface=True):
        self.bands = _BANDS[bands] if isinstance(bands, str) else bands
        self.table = SPECIES if table is None else table
        self.grounds = GROUNDS if grounds is None else grounds
        self.surface = bool(surface)

    # ---- per-species optical data in this model's bands (cached at module level: the object holds no state)
    def _key(self): return (self.bands.name, id(self.table), id(self.grounds))

    def species_data(self, species):
        """Per species, in this model's bands: role, n, K (nb,), S (nb,), contrast tables. Cached by species list."""
        return _species_data(self.bands.name, _freeze(self.table), tuple(species))

    def ground_body(self, ground, f_water):
        """Ground body reflectance (..., nb) with its free pores a fraction f_water (...) filled with water, and
        its surface index (...). The table's R is the dry ground as seen: its body is that R with its own
        (rough) surface taken off by inverting Saunderson. Scattering of the solid scales with the contrast
        to what fills its pores (bound binder; air or water); the surface follows the same closure rule as a
        film: index = c n_fill + (1 - c) n_air, c the filled fraction of the pores."""
        g = self.grounds[ground]; b = self.bands
        Rtab = np.asarray(g["R"], float)
        R_vis = _band_means(fit_reflectance(Rtab[:3]), b) if b.name != "classic" else Rtab[:3]
        Rseen = np.r_[R_vis, Rtab[3]]
        n_air, n_w, n_b, bound = self._n_role("void"), self._n_role("volatile"), float(g["n_bound"]), float(g["bound"])
        f = np.asarray(f_water, float)
        C = lambda ns: contrast_scattering(g["n_solid"], ns)
        s_dry = bound * C(n_b) + (1 - bound) * C(n_air)
        s_wet = bound * C(n_b) + (1 - bound) * ((1 - f) * C(n_air) + f * C(n_w))
        def n_surf(f):                                       # c n_fill + (1 - c) n_air, c the filled fraction
            f = np.asarray(f, float)
            return bound * n_b + (1 - bound) * f * n_w + (1 - bound) * (1 - f) * n_air if self.surface else np.ones_like(f)
        Rf, Ri, Tf, Ti = interface(np.float64(n_surf(0.0)), 0.0)
        body_dry = (Rseen - Rf) / (Tf * Ti + Ri * (Rseen - Rf))
        ratio = np.where(s_wet == s_dry, 1.0, s_dry / np.maximum(s_wet, 1e-300))
        return km_rinf(km_k(body_dry) * np.asarray(ratio)[..., None]), n_surf(f)

    def _n_role(self, role):
        """Index of the table's species of this role (void: air; volatile: what the ground soaks up)."""
        for e in self.table.values():
            if e.get("role") == role: return float(e["n"])
        if role == "void": return AIR_N
        raise KeyError(f"the species table has no {role!r} species")

    # ---- one film: K, S (..., nb) per gain, and its surface (n, smooth, cover)
    def film(self, species, vol, kappa, stratum, gain=10.0):
        """Optics of one film of composition vol (species, ...): K, S (..., nb) as optical thickness (x gain), and
        its surface: n (...), smooth (...), cover (...). kappa: the film's cure (strata: fully yellowed)."""
        D = self.species_data(species); nb = self.bands.n
        vol = np.asarray(vol, float); shp = vol.shape[1:]
        P = PHI_PACK
        Lq = np.zeros(shp); nL = np.zeros(shp); bed = np.zeros(shp); nbed = np.zeros(shp)
        drops = {}
        for i, d in enumerate(D):
            v = vol[i]
            if d["role"] == "pigment": bed = bed + v; nbed = nbed + d["n"] * v
            elif d.get("carrier_index") is not None:                    # an emulsion binder: droplets until packed
                w = vol[d["carrier_index"]]; tot = v + w
                coal = np.clip((v / np.where(tot > 0, tot, 1.0) - P) / (1 - P), 0, 1)
                coal = np.where(tot > 0, coal, 1.0)
                Lq = Lq + coal * v; nL = nL + d["n"] * coal * v
                dv = (1 - coal) * v; drops[i] = dv; bed = bed + dv; nbed = nbed + d["n"] * dv
            else: Lq = Lq + v; nL = nL + d["n"] * v
        n_c = np.where(Lq > 0, nL / np.where(Lq > 0, Lq, 1.0), AIR_N)
        Vc = bed * (1 - P) / P
        fill = np.minimum(Lq, Vc)
        a = np.where(Vc > 0, 1 - fill / np.where(Vc > 0, Vc, 1.0), 0.0)
        K = np.zeros(shp + (nb,)); S = np.zeros(shp + (nb,))
        for i, d in enumerate(D):
            v = drops.get(i, vol[i])
            if not np.any(v): continue
            Kc = d["K"]
            if d["yellows"]: Kc = Kc * (1 + 2 * (np.minimum(kappa / KAPPA_FULL, 1.0)[..., None] if not stratum else 1.0))
            K = K + (v * gain)[..., None] * Kc
            if d["role"] == "pigment" or i in drops:
                ratio = (1 - a) * d["ratio"](n_c) + a * d["ratio_air"]
                S = S + (v * gain * ratio)[..., None] * d["S"]
        h = vol.sum(0)
        c = np.where(Vc > 0, np.clip(Lq / np.where(Vc > 0, Vc, 1.0), 0, 1), 1.0)
        n = c * n_c + (1 - c) * self._n_role("void") if self.surface else np.ones(shp)   # an open bed: no interface
        smooth = np.clip(np.maximum(Lq - Vc, 0) / E_SMOOTH, 0, 1)
        cover = np.clip(h / H_COVER, 0, 1)
        return K, S, n, smooth, cover

    # ---- the stack
    def _layers(self, st, layers):
        stack = list(range(len(st["strata"]))) + [None]
        return stack if layers is None else stack[:layers]

    def _compose(self, st, gain, layers):
        """Bottom-up: (body reflectance (H, W, nb), surface n (H, W), smooth (H, W))."""
        H, W = st["H"], st["W"]; sp = st["species"]
        f = np.clip(st["store"] / st["store_cap"], 0, 1)
        Rb, n = self.ground_body(st["ground"], f)
        Rb = np.array(np.broadcast_to(Rb, (H, W, self.bands.n))); n = np.array(np.broadcast_to(n, (H, W)), float)
        smooth = np.zeros((H, W))
        for i in self._layers(st, layers):
            if i is None: y0, y1, x0, x1 = 0, H, 0, W; blk = st["wet"]; kap = st["kappa"]; strat = False
            else: y0, y1, x0, x1, blk = st["strata"].block(i); kap = None; strat = True
            m = blk.sum(0) > 0
            if not m.any(): continue
            v = blk[:, m]
            K, S, nf, sf, cov = self.film(sp, v, None if strat else kap[y0:y1, x0:x1][m], strat, gain)
            R, T = km_layer(K, S)
            sub = Rb[y0:y1, x0:x1]; rb = sub[m]
            sub[m] = R + T * T * rb / (1 - R * rb)
            ns = n[y0:y1, x0:x1]; ns[m] = cov * nf + (1 - cov) * ns[m]
            ss = smooth[y0:y1, x0:x1]; ss[m] = cov * sf + (1 - cov) * ss[m]
        return Rb, n, smooth

    def reflectance(self, cv, gain=10.0, layers=None, light=None, adapt=False):
        st = state(cv)
        Rb, n, smooth = self._compose(st, gain, layers)
        R = self._seen(Rb, n, smooth)
        return R if light is None else colour(R, self.bands, light, adapt)

    def _seen(self, Rb, n, smooth):
        if not self.surface: return np.clip(Rb, 0, 1)
        Rf, Ri, Tf, Ti = interface(n, smooth)
        R = Rf[..., None] + Tf[..., None] * Ti[..., None] * Rb / (1 - Ri[..., None] * Rb)
        return np.clip(R, 0, 1)

    def finish(self, cv):
        """Gloss 0..1: the smooth fraction of the top surface (the same surface reflectance() reads)."""
        st = state(cv); H, W = st["H"], st["W"]; sp = st["species"]
        smooth = np.zeros((H, W))
        for i in self._layers(st, None):
            if i is None: y0, y1, x0, x1 = 0, H, 0, W; blk = st["wet"]
            else: y0, y1, x0, x1, blk = st["strata"].block(i)
            m = blk.sum(0) > 0
            if not m.any(): continue
            _, _, _, sf, cov = self.film(sp, blk[:, m], np.zeros(int(m.sum())), i is not None)
            ss = smooth[y0:y1, x0:x1]; ss[m] = cov * sf + (1 - cov) * ss[m]
        return smooth

    def surface_index(self, cv):
        """The top surface's refractive index (H, W) (F0 = ((n - 1)/(n + 1))^2), for sensors that want it."""
        return self._compose(state(cv), 10.0, None)[1]

    def colour(self, R, light="D65", adapt=False): return colour(R, self.bands, light, adapt)

    def preview(self, species, film, under=None, ground=None, gain=10.0, wet=True, kappa=0.0, store=0.0, store_cap=0.08):
        """The look (..., nb) of one film (species, ...) laid on a ground (dry unless `store`), or over a body
        `under` (..., nb) taken as a ground of no surface of its own. The same film(), KM, composition and
        surface functions as reflectance(): a palette that previews with this agrees with the canvas."""
        film = np.asarray(film, float); shp = film.shape[1:]
        if under is None:
            ground = next(iter(self.grounds)) if ground is None else ground
            Rb, n0 = self.ground_body(ground, np.full(shp, np.clip(store / store_cap, 0, 1)))
            Rb = np.broadcast_to(Rb, shp + (self.bands.n,)); n0 = np.broadcast_to(n0, shp)
        else:
            Rb = np.broadcast_to(np.asarray(under, float), shp + (self.bands.n,)); n0 = np.ones(shp)
        m = film.sum(0) > 0
        K, S, nf, sf, cov = self.film(species, film, np.broadcast_to(np.asarray(kappa, float), shp), not wet, gain)
        R, T = km_layer(K, S)
        Rn = np.where(m[..., None], R + T * T * Rb / (1 - R * Rb), Rb)
        n = np.where(m, cov * nf + (1 - cov) * n0, n0); smooth = np.where(m, cov * sf, 0.0)
        return self._seen(Rn, n, smooth)


def _freeze(table):
    """A hashable snapshot of a species table (for the cache): the table is data and may be replaced."""
    def fz(v):
        if isinstance(v, dict): return tuple(sorted((k, fz(x)) for k, x in v.items()))
        if isinstance(v, (list, tuple, np.ndarray)): return tuple(fz(x) for x in v)
        return v
    return _Frozen(fz(table), table)


class _Frozen:
    def __init__(self, key, table): self.key, self.table = key, table
    def __hash__(self): return hash(self.key)
    def __eq__(self, o): return self.key == o.key


@lru_cache(maxsize=256)
def _species_data(bands_name, ftable, species):
    b = _BANDS[bands_name]; table = ftable.table
    n_ref = next(float(e["n"]) for e in table.values() if e.get("reference"))
    n_air = next((float(e["n"]) for e in table.values() if e.get("role") == "void"), AIR_N)
    out = []
    for s in species:
        e = _entry(table, s)
        d = dict(name=s, role=e["role"], n=float(e["n"]), K=_band_values(e, "K", b), S=_band_values(e, "S", b),
                 yellows=bool(e.get("yellows", False)))
        d["carrier_index"] = species.index(e["dispersed_in"]) if e.get("dispersed_in") in species else None
        if e["role"] == "pigment" or e.get("dispersed_in"):
            d["ratio"] = _Ratio(d["n"], n_ref)
            d["ratio_air"] = d["ratio"](np.float64(n_air))
        out.append(d)
    return tuple(out)


class _Ratio:
    """contrast_scattering(n_p, n) / contrast_scattering(n_p, n_ref): 1 in the reference surrounding, exactly."""
    def __init__(self, n_p, n_ref): self.n_p, self.n_ref = n_p, n_ref; self.ref = contrast_scattering(n_p, n_ref)
    def __call__(self, n):
        n = np.asarray(n, float)
        return np.where(n == self.n_ref, 1.0, contrast_scattering(self.n_p, n) / self.ref)


# ======================================================================================== the canvas's model

MODELS = {"classic": ClassicOptics, "v2": V2Optics}


def make_optics(spec="classic"):
    """An optics model from a name ("classic", "v2") or a model object (returned as it is)."""
    if isinstance(spec, str):
        if spec not in MODELS: raise ValueError(f"unknown optics model {spec!r}: one of {sorted(MODELS)}")
        return MODELS[spec]()
    if hasattr(spec, "reflectance") and hasattr(spec, "finish") and hasattr(spec, "bands"): return spec
    raise TypeError("optics must be a model name or an optics model object")


def optics_of(cv):
    """The canvas's optics model (classic for a canvas pickled before models existed)."""
    m = cv.__dict__.get("optics")
    return m if m is not None else _CLASSIC


_CLASSIC = ClassicOptics()
