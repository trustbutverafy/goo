"""goo.classic.matter.phenotype: a substance's data sheet, made by making paint in the engine and looking, and the
inverse-design searches.

The engine is used as it stands, through its extension point (classic/SPEC.md section 9): the substance's derived K
and S at a chosen surrounding index are registered in goo.classic.engine.PIGMENTS (under a temporary name, removed
afterwards), films are laid on a small canvas with `Canvas.add` (booked, as a test setup), a black stratum
is fixed under half of them, and `reflectance()` is read. Nothing is computed beside the engine's optics.

Colour: goo's R, G, B band reflectances are read as linear sRGB, then CIELAB (D65) by goo.classic.palette.lab, as
goo's painters see them.
"""
from __future__ import annotations
from contextlib import contextmanager
import itertools
import numpy as np
from goo.classic.engine import PIGMENTS, Canvas, GROUND_RGBN
from goo.classic.palette import lab

_counter = itertools.count()
OIL_K = np.array([0.02, 0.03, 0.06, 0.01])     # the engine's linseed term (Canvas._KS), wet, uncured
GAIN = 10.0


@contextmanager
def registered(K, S, name=None):
    """Register K, S (R, G, B, N) as a pigment for the duration of the block; yields its name."""
    name = name or f"matter:{next(_counter)}"
    if name in PIGMENTS: raise KeyError(name)
    PIGMENTS[name] = dict(K=[float(x) for x in K], S=[float(x) for x in S])
    try: yield name
    finally: PIGMENTS.pop(name, None)


def lab_to_linear(L):
    """CIELAB (D65) -> linear sRGB, the inverse of goo.classic.palette.lab."""
    L = np.asarray(L, float)
    fy = (L[..., 0] + 16) / 116; fx = fy + L[..., 1] / 500; fz = fy - L[..., 2] / 200
    finv = lambda f: np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27))
    xyz = np.stack([finv(fx), finv(fy), finv(fz)], -1) * np.array([0.95047, 1.0, 1.08883])
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    return xyz @ np.linalg.inv(M).T


def lch(L):
    L = np.asarray(L); return np.stack([L[..., 0], np.hypot(L[..., 1], L[..., 2]), np.degrees(np.arctan2(L[..., 2], L[..., 1])) % 360], -1)


THICK = np.geomspace(0.004, 4.0, 13)            # film thickness, volume units (1 um .. 1 mm)
PHI = 0.4                                       # pigment volume fraction of the paint (the rest is oil)


def sheet_from_ks(K, S, label="", phi=PHI, white_ratio=10.0, carrier="oil"):
    """Data sheet of a pigment with K, S (goo units, R G B N), as paint at volume fraction phi in `carrier`
    ('oil', or 'water': optically nothing in the classic engine, so the particles' K, S stand alone)."""
    nT = len(THICK)
    with registered(K, S) as p:
        cv = Canvas(nT, 4, [p, "white", "carbon"], seed=0)
        black = np.zeros((4, nT)); black[1] = 1.0                    # row 1: over a fixed black stratum
        cv.add("carbon", black); cv.fix()
        film = np.zeros((4, nT)); film[0] = THICK; film[1] = THICK
        cv.add(p, film * phi); cv.add(carrier, film * (1 - phi))
        mass = np.zeros((4, nT)); mass[3] = 4.0
        cv.add(p, mass * phi); cv.add(carrier, mass * (1 - phi))
        tint = np.zeros((4, nT)); tint[2] = 4.0
        cv.add(p, tint * phi / (1 + white_ratio)); cv.add("white", tint * phi * white_ratio / (1 + white_ratio))
        cv.add(carrier, tint * (1 - phi))
        R = cv.reflectance()
    Lw, Lb = lab(R[0, :, :3]), lab(R[1, :, :3])
    dE = np.linalg.norm(Lw - Lb, axis=1)
    hid = np.nonzero(dE < 1.0)[0]
    if hid.size == 0: hiding = np.inf
    elif hid[0] == 0: hiding = THICK[0]
    else:                                                            # log-interpolate the dE = 1 crossing
        i = hid[0]; a, b = np.log(dE[i - 1]), np.log(dE[i])
        hiding = float(np.exp(np.log(THICK[i - 1]) + (np.log(THICK[i]) - np.log(THICK[i - 1])) * (a - 0) / (a - b)))
    j50 = int(np.argmin(np.abs(THICK - 0.2)))                          # a 50 um film
    return dict(label=label, carrier=carrier, K=np.asarray(K), S=np.asarray(S),
                masstone_rgb=R[3, 0, :3], masstone_lab=lab(R[3, 0, :3]),
                undertone_rgb=R[0, 4, :3], undertone_lab=Lw[4], undertone_um=THICK[4] * 250,
                tint_rgb=R[2, 0, :3], tint_lab=lab(R[2, 0, :3]),
                tint_strength=float(100 - lab(R[2, 0, :3])[0]),            # lightness lost by 1:10 in white
                hiding_um=hiding * 250 * (1 if np.isfinite(hiding) else 1),
                nir_masstone=float(R[3, 0, 3]), nir_contrast_50um=float(R[0, j50, 3] - R[1, j50, 3]),
                drawdown_white=R[0], drawdown_black=R[1])


def datasheet(sub, n_s=1.48, **kw):
    """Data sheet of a substance as oil paint, its K and S taken at surrounding index n_s."""
    K, S = sub.ks(n_s)
    d = sheet_from_ks(K, S, label=sub.name, **kw)
    d["n_s"] = n_s; d["klass"] = sub.klass
    return d


def format_sheet(d):
    f = lambda v: "(" + ", ".join(f"{x:.1f}" for x in v) + ")"
    return (f"{d['label']} [{d.get('klass', '?')}, in {d.get('carrier', 'oil')}]  masstone Lab {f(d['masstone_lab'])}  undertone ({d['undertone_um']:.0f} um over "
            f"gesso) {f(d['undertone_lab'])}  tint 1:10 white {f(d['tint_lab'])} (strength {d['tint_strength']:.1f})  "
            f"hiding {d['hiding_um']:.0f} um  NIR masstone R {d['nir_masstone']:.2f}, 50 um contrast {d['nir_contrast_50um']:.2f}")


# ------------------------------------------------------------------------------- the engine's optics, inline
def film_over(K, S, pigment_vol, oil_vol, Rb, extra=None):
    """Reflectance (4,) of one wet film over a body Rb (4,), by the engine's own layer formula (Canvas.km_layer)
    and composition rule. `extra`: a list of (K, S, volume) of other pigments in the same film."""
    Kt = np.asarray(K) * pigment_vol * GAIN + oil_vol * GAIN * OIL_K
    St = np.asarray(S) * pigment_vol * GAIN
    for (k2, s2, v2) in (extra or []):
        Kt = Kt + np.asarray(k2) * v2 * GAIN; St = St + np.asarray(s2) * v2 * GAIN
    R, T = Canvas.km_layer(Kt, St)
    return R + T * T * Rb / (1 - R * Rb)
