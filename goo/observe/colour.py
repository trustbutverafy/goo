"""Colorimetry: spectral band reflectance and an illuminant -> CIE XYZ -> CIELAB, sRGB; CIEDE2000. docs/OBSERVE.md.

Every table, fit and constant here has a Provenance (`PROV`). None of it is a table copied from a restricted source:

  observer     CIE 1931 2-degree colour matching functions as the multi-lobe analytic fit of Wyman, Sloan & Shirley
               (2013), promoted by copying from goo/classic/optics.py (`_cie_cmf`). `assumed`: its error against the
               CIE table was NOT measured in this repository (the CIE table is not CC0 and was not read here).
  A            Planck's law at 2856 K with c2 = h c / k_B from the exact SI constants: `exact` arithmetic. That CIE
               illuminant A is this radiator is recalled (assumed until read).
  daylight     a Planckian radiator at 6504 K, standing in for CIE D65 (D65 is a table, not shipped): `assumed`.
               It is not D65: D65 has less violet and more structure; the difference was not measured.
  sRGB         XYZ -> linear sRGB from the primaries' and white's chromaticities (IEC 61966-2-1, recalled:
               `assumed`), and its transfer function.
  Bradford     the chromatic-adaptation matrix (Lam 1985, recalled: `assumed`).
  CIELAB       CIE 15's definition (recalled), relative to the illuminant's own white.
  CIEDE2000    Sharma, Wu & Dalal (2005)'s statement of the formula, implemented here and compared in the tests with
               scikit-image's independent implementation.

Band integration: a band's reflectance is taken constant across the band, and beyond the bands' coverage the
reflectance of the nearest band is held (assumed). Bands outside 360-830 nm get zero weight.
"""
from dataclasses import dataclass
import numpy as np
from goo.evidence import Provenance, weakest

H_PLANCK = 6.62607015e-34      # J s, exact (SI 2019)
C_0 = 299792458.0              # m/s, exact
K_BOLTZ = 1.380649e-23         # J/K, exact
C2 = H_PLANCK * C_0 / K_BOLTZ  # m K: second radiation constant, exact arithmetic (1.438777e-2)

LAM_LO, LAM_HI = 360e-9, 830e-9
FINE = np.arange(360.5, 830.0, 1.0) * 1e-9          # 1 nm midpoints, m

PROV = {
    "observer": Provenance("assumed", "C. Wyman, P.-P. Sloan, P. Shirley, 'Simple analytic approximations to the CIE XYZ "
                           "colour matching functions', J. Computer Graphics Techniques 2(2), 2013; coefficients promoted "
                           "from goo/classic/optics.py", note="multi-lobe fit to the CIE 1931 2-degree observer. Its error "
                           "against the CIE 1931 table was not measured here (the table is CIE's, not CC0, and was not "
                           "read in this repository); classic's docstring says 'well under 1% of peak' (recalled, "
                           "unverified). Evidence stays assumed until a check against the table is made"),
    "A": Provenance("exact", "Planck's law, c2 = h c / k_B from the exact SI constants, T = 2856 K",
                    note="the arithmetic is exact; that CIE illuminant A is a Planckian radiator at 2856 K (with "
                         "c2 = 1.4388e-2 m K) is recalled, not read: the identification is assumed"),
    "daylight": Provenance("assumed", "Planck's law at 6504 K (D65's nominal correlated colour temperature, recalled)",
                           note="NOT CIE D65: the D65 table is not shipped (CIE data, not CC0). A Planckian radiator "
                                "has more violet and a smoother blue than D65; the colour difference this makes was "
                                "not measured"),
    "srgb": Provenance("assumed", "IEC 61966-2-1 (sRGB): primaries (0.64, 0.33), (0.30, 0.60), (0.15, 0.06), white "
                       "(0.3127, 0.3290), transfer 12.92 / 0.0031308 / 1.055 / 2.4", note="recalled, not read; the "
                       "matrix is derived from these chromaticities here, not typed"),
    "bradford": Provenance("assumed", "Bradford chromatic adaptation (K. M. Lam 1985; as given by Hunt, Measuring Colour)",
                           note="matrix recalled, not read"),
    "cielab": Provenance("assumed", "CIE 15 (CIELAB: f(t) = t^(1/3) above (6/29)^3, linear below)",
                         note="definition recalled; constants are exact rationals"),
    "ciede2000": Provenance("assumed", "G. Sharma, W. Wu, E. N. Dalal, Color Res. Appl. 30, 21-30 (2005)",
                            note="implemented from the published statement (recalled); tested against scikit-image's "
                                 "deltaE_ciede2000 and Sharma's first test pair"),
    "bands": Provenance("assumed", note="reflectance constant within each band; held at the nearest band beyond the "
                                        "bands' coverage of 360-830 nm"),
}


def planck(lam, T):
    """Spectral radiance shape of a black body at T (K), relative (lam in m): lam^-5 / (exp(c2 / (lam T)) - 1)."""
    lam = np.asarray(lam, float)
    return lam ** -5.0 / np.expm1(C2 / (lam * T))


def cmf_1931(lam):
    """CIE 1931 2-degree colour matching functions [..., 3] at lam (m), by Wyman, Sloan & Shirley's multi-lobe fit.
    Copied from goo/classic/optics.py `_cie_cmf` (wavelengths there in nm); see PROV['observer']."""
    x_nm = np.asarray(lam, float) * 1e9

    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)

    x = 1.056 * g(x_nm, 599.8, 37.9, 31.0) + 0.362 * g(x_nm, 442.0, 16.0, 26.7) - 0.065 * g(x_nm, 501.1, 20.4, 26.2)
    y = 0.821 * g(x_nm, 568.8, 46.9, 40.5) + 0.286 * g(x_nm, 530.9, 16.3, 31.1)
    z = 1.217 * g(x_nm, 437.0, 11.8, 36.0) + 0.681 * g(x_nm, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


@dataclass(frozen=True)
class Illuminant:
    """A named spectral power distribution: spd(lam in m) -> relative power. `prov` says what it is."""
    name: str
    T: float
    prov: Provenance

    def spd(self, lam):
        return planck(lam, self.T)

    def band_power(self, bands):
        """Mean relative power in each band (for goo.film.Light); bands with no overlap of 360-830 nm get the power
        at their centre."""
        out = []
        for c, w in zip(bands.centres, bands.widths):
            lam = np.linspace(c - w / 2, c + w / 2, 21)
            out.append(float(self.spd(lam).mean()))
        p = np.array(out)
        return p / p.max()


A = Illuminant("A (Planck 2856 K)", 2856.0, PROV["A"])
DAYLIGHT = Illuminant("daylight (Planck 6504 K, not D65)", 6504.0, PROV["daylight"])
ILLUMINANTS = {"daylight": DAYLIGHT, "A": A}


@dataclass(frozen=True)
class Observer:
    """A colorimetric observer: colour matching functions and their provenance."""
    name: str = "CIE 1931 2-degree (Wyman-Sloan-Shirley fit)"

    def cmf(self, lam):
        return cmf_1931(lam)

    @property
    def prov(self):
        return PROV["observer"]


CIE1931 = Observer()


def band_of_fine(bands):
    """Index of the band each FINE wavelength is assigned to: the band containing it, else the nearest band (by distance
    to its interval) among bands that overlap 360-830 nm; -1 if no band overlaps."""
    c, w = np.asarray(bands.centres), np.asarray(bands.widths)
    lo, hi = c - w / 2, c + w / 2
    vis = (hi > LAM_LO) & (lo < LAM_HI)
    if not vis.any():
        return np.full(FINE.size, -1)
    dist = np.maximum(lo[None] - FINE[:, None], 0) + np.maximum(FINE[:, None] - hi[None], 0)
    dist = np.where(vis[None], dist, np.inf)
    return np.argmin(dist, 1)


def weights(bands, illuminant, observer=CIE1931):
    """W [3, B]: XYZ = W @ R for band reflectance R, normalised so that a perfect diffuser (R = 1) has Y = 1."""
    idx = band_of_fine(bands)
    E = illuminant.spd(FINE)
    cm = observer.cmf(FINE) * E[:, None]
    W = np.zeros((3, bands.B))
    for b in range(bands.B):
        m = idx == b
        if m.any():
            W[:, b] = cm[m].sum(0)
    return W / W[1].sum()


def white_xyz(bands, illuminant, observer=CIE1931):
    return weights(bands, illuminant, observer).sum(1)


def xyz(R, bands, illuminant, observer=CIE1931):
    """CIE XYZ [..., 3] of band reflectance R [..., B] under the illuminant (Y of a perfect diffuser = 1)."""
    R = np.asarray(R, float)
    if R.shape[-1] != bands.B:
        raise ValueError(f"reflectance has {R.shape[-1]} bands, the bands {bands.B}")
    return R @ weights(bands, illuminant, observer).T


# ------------------------------------------------------------------------------------------------ CIELAB, sRGB
_D = 6.0 / 29.0


def lab(XYZ, white):
    """CIELAB [..., 3] of XYZ relative to the white XYZ (the illuminant's perfect diffuser)."""
    t = np.asarray(XYZ, float) / np.asarray(white, float)
    f = np.where(t > _D ** 3, np.cbrt(np.maximum(t, 0)), t / (3 * _D * _D) + 4.0 / 29.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_xyz(L, white):
    L = np.asarray(L, float)
    fy = (L[..., 0] + 16) / 116
    fx, fz = fy + L[..., 1] / 500, fy - L[..., 2] / 200
    inv = lambda f: np.where(f > _D, f ** 3, 3 * _D * _D * (f - 4.0 / 29.0))
    return np.stack([inv(fx), inv(fy), inv(fz)], -1) * np.asarray(white, float)


def _xy_to_XYZ(x, y):
    return np.array([x / y, 1.0, (1 - x - y) / y])


SRGB_PRIMARIES = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))
SRGB_WHITE_XY = (0.3127, 0.3290)


def _rgb_to_xyz_matrix():
    P = np.stack([_xy_to_XYZ(*p) for p in SRGB_PRIMARIES], 1)
    S = np.linalg.solve(P, _xy_to_XYZ(*SRGB_WHITE_XY))
    return P * S[None]


M_RGB_TO_XYZ = _rgb_to_xyz_matrix()
M_XYZ_TO_RGB = np.linalg.inv(M_RGB_TO_XYZ)
SRGB_WHITE = M_RGB_TO_XYZ.sum(1)                     # XYZ of sRGB white, Y = 1

BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def adapt(XYZ, white_src, white_dst=SRGB_WHITE):
    """Bradford (von Kries in sharpened cone space) adaptation of XYZ from one white to another."""
    s, d = BRADFORD @ np.asarray(white_src, float), BRADFORD @ np.asarray(white_dst, float)
    M = np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD
    return np.asarray(XYZ, float) @ M.T


def srgb_linear(XYZ, white=None):
    """Linear sRGB [..., 3] of XYZ; with `white` (the illuminant's XYZ) adapted first so that white maps to (1, 1, 1):
    an observer adapted to the light. Unclipped."""
    X = np.asarray(XYZ, float)
    if white is not None:
        X = adapt(X, np.asarray(white, float) / np.asarray(white, float)[1])
    return X @ M_XYZ_TO_RGB.T


def encode(rgb_linear):
    """sRGB transfer function on [0, 1] (input clipped)."""
    c = np.clip(np.asarray(rgb_linear, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def srgb8(rgb_linear):
    return np.round(encode(rgb_linear) * 255).astype(np.uint8)


def out_of_gamut(rgb_linear, tol=1e-6):
    c = np.asarray(rgb_linear, float)
    return np.any((c < -tol) | (c > 1 + tol), -1)


# --------------------------------------------------------------------------------------------------- CIEDE2000
def ciede2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0):
    """CIEDE2000 colour difference between CIELAB arrays [..., 3] (broadcast). Sharma, Wu & Dalal (2005)."""
    a, b = np.broadcast_arrays(np.asarray(lab1, float), np.asarray(lab2, float))
    L1, a1, b1 = a[..., 0], a[..., 1], a[..., 2]
    L2, a2, b2 = b[..., 0], b[..., 1], b[..., 2]
    C1, C2_ = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb7 = ((C1 + C2_) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cb7 / (Cb7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    h1p = np.where((b1 == 0) & (a1p == 0), 0.0, h1p)
    h2p = np.where((b2 == 0) & (a2p == 0), 0.0, h2p)
    dLp = L2 - L1
    dCp = C2p - C1p
    zero = C1p * C2p == 0
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(zero, 0.0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = np.where(zero, hs, np.where(np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Cbp7 = Cbp ** 7
    RC = 2 * np.sqrt(Cbp7 / (Cbp7 + 25.0 ** 7))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2 * dtheta)) * RC
    tL, tC, tH = dLp / (kL * SL), dCp / (kC * SC), dHp / (kH * SH)
    return np.sqrt(tL ** 2 + tC ** 2 + tH ** 2 + RT * tC * tH)


def evidence():
    """The weakest evidence of the colorimetric chain (observer, daylight, sRGB, CIELAB, band integration)."""
    return weakest(*(p.evidence for p in PROV.values()))
