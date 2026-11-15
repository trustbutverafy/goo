"""goo.matter.response: from a particle description to the numbers goo's optics consumes.

Map (each step stated, each with its source):

1. Single particle. Mie efficiencies Q_abs, Q_sca and asymmetry g of each particle in a non-absorbing medium
   of index n_s: relative index m = (n + ik) / n_s, size parameter x = pi d n_s / lambda (goo.matter.mie).
2. Population, per unit particle volume. A particle of diameter d has volume pi d^3 / 6 and cross-section
   Q pi d^2 / 4, so cross-section per volume is 3 Q / (2 d). Averaged over the volume distribution:
       mu_a = < 3 Q_abs / (2 d) >_v ,  mu_s = < 3 Q_sca / (2 d) >_v ,  g = < g Q_sca / d >_v / < Q_sca / d >_v
   (1/um per unit volume fraction). This is the independent-scattering approximation: particles scatter as if
   alone (no crowding, no near-field coupling, no multiple scattering inside a particle's neighbourhood). It
   holds for volume fractions of a few percent; at paint loadings (PVC 10-50%) dependent scattering lowers
   the scattering of titania by up to ~2x (Ross 1971, J. Paint Technol. 43, 50; Auger & Stout 2012), and it is
   NOT modelled: the response is linear in volume, which is exactly what goo's optics assumes.
3. Two-flux coefficients, from the diffusion limit of radiative transfer for diffuse light (van Gemert & Star
   1987, Lasers Life Sci. 1, 287; Star, Marijnissen & van Gemert 1988, Phys. Med. Biol. 33, 437):
       K = 2 mu_a ,   S = (3/4) mu_s (1 - g)
   The factor 2 is the mean path of diffuse light per unit thickness; S uses the transport (reduced)
   scattering, so forward-peaked scattering by large particles counts for little. Their full relation has
   S = (3/4) mu_s (1 - g) - mu_a / 4; the correction is dropped (it can make S negative for strong absorbers,
   where the diffusion limit it comes from fails anyway). Valid when scattering dominates and the film is
   many transport lengths thick; for strong absorbers K is right to the factor 2, S is indicative.
4. goo's units (SPEC.md section 2): K, S in `PIGMENTS` are per 0.025 mm of pure pigment volume, so
       K_goo = 25 um * K ,  S_goo = 25 um * S      (K, S in 1/um per unit volume fraction)
   and the engine multiplies by gain = 10 per volume unit of 0.25 mm.
5. Bands: goo's classic bands are boxes, B 400-500, G 500-600, R 600-700 nm (goo.optics.CLASSIC_BANDS), plus
   N around 1.5 um (1500-1600 nm). A band value is the mean of the spectral K (and S) over its box, sampled
   every 10 nm (20 nm in `coarse` mode) at box midpoints. Order (R, G, B, N), as in `PIGMENTS`.
6. Surroundings. The response is a function of the index n_s around the particle. It is precomputed on
   n_s = 1.00, 1.05, ..., 1.60 and interpolated by a cubic spline per band (`ResponseTable`), so the engine
   can ask cheaply; `exact=True` computes at any n_s.
"""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from goo.matter import mie
from goo.matter.morphology import NotDerivable

UNIT_UM = 25.0                                   # goo's K, S are per 0.025 mm of pure pigment
BAND_BOXES = [(600, 700), (500, 600), (400, 500), (1500, 1600)]  # R, G, B, N
BANDS = ["R", "G", "B", "N"]
NS_GRID = np.round(np.arange(1.0, 1.6001, 0.05), 4)


def band_grid(coarse=False):
    """(lam_nm, band_index): sample wavelengths at box midpoints."""
    step = 20.0 if coarse else 10.0
    lam, idx = [], []
    for j, (lo, hi) in enumerate(BAND_BOXES):
        st = 50.0 if j == 3 else step
        pts = np.arange(lo + st / 2, hi, st)
        lam.extend(pts); idx.extend([j] * len(pts))
    return np.array(lam), np.array(idx)


def band_means(spec, idx):
    """(..., n_lam) -> (..., 4) means per band."""
    return np.stack([spec[..., idx == j].mean(-1) for j in range(4)], -1)


def cross_sections(dielectric, morph, lam_nm, n_s, q=9):
    """mu_a, mu_s (1/um per unit particle volume fraction) and g, shape (len(n_s), len(lam)), for particles of
    `dielectric` with morphology `morph` in non-absorbing media of indices n_s (scalar or 1-D)."""
    if morph is None: raise NotDerivable("a continuous phase has no particles: no particle cross-sections")
    kind = morph.kind
    if kind not in ("sphere", "coated", "hollow"):
        raise NotDerivable(f"optics of particle kind {getattr(morph, 'name', kind)!r} is not derived by goo "
                           "(only spheres, coated and hollow spheres)")
    lam = np.asarray(lam_nm, float); ns = np.atleast_1d(np.asarray(n_s, float))
    d, w = morph.size.nodes(q)
    m_p = dielectric.nk(lam)                                         # (L,)
    NS, D, LAM = np.meshgrid(ns, d, lam, indexing="ij")              # (N, Q, L)
    M = m_p[None, None, :] / NS
    X = np.pi * D * NS / (LAM / 1000.0)                              # d in um, lambda in nm
    if kind == "sphere":
        qe, qs, qa, g = mie.sphere(M, X)
    elif kind == "coated":
        m_sh = morph.shell.nk(lam)[None, None, :] / NS
        qe, qs, qa, g = mie.coated(M, m_sh, morph.ratio * X, X)
    else:                                                            # hollow: void core, substance shell
        qe, qs, qa, g = mie.coated(1.0 / NS + 0j, M, morph.ratio * X, X)
    W = (w / d)[None, :, None] * 1.5                                 # 3/(2d) volume weight
    mu_a = (W * qa).sum(1); mu_s = (W * qs).sum(1)
    gs = np.where(mu_s > 0, (W * qs * g).sum(1) / np.maximum(mu_s, 1e-300), 0.0)
    return mu_a, mu_s, gs


def km_spectral(dielectric, morph, lam_nm, n_s, q=9):
    """K, S per unit particle volume in goo units (per 0.025 mm of pure pigment), (len(n_s), len(lam))."""
    mu_a, mu_s, g = cross_sections(dielectric, morph, lam_nm, n_s, q)
    return 2 * mu_a * UNIT_UM, 0.75 * mu_s * (1 - g) * UNIT_UM


def km_spectral_dispersive(dielectric, morph, lam_nm, medium, q=9):
    """K, S (len(lam),) in a dispersive non-absorbing medium (a Dielectric, e.g. WATER): Mie at n_s(lambda)
    for each wavelength."""
    lam = np.asarray(lam_nm, float); ns = medium.nk(lam).real
    K = np.empty(lam.shape); S = np.empty(lam.shape)
    for i, (l, n) in enumerate(zip(lam, ns)):
        k_, s_ = km_spectral(dielectric, morph, [l], [n], q); K[i], S[i] = k_[0, 0], s_[0, 0]
    return K, S


def km_bands(dielectric, morph, n_s, coarse=False, q=9):
    """K, S in goo's bands (R, G, B, N): arrays (len(n_s), 4)."""
    lam, idx = band_grid(coarse)
    K, S = km_spectral(dielectric, morph, lam, n_s, q)
    return band_means(K, idx), band_means(S, idx)


# --------------------------------------------------------------------------------------- the response table
_CACHE = {}


def _cache_dir():
    d = os.environ.get("GOO_MATTER_CACHE")
    if d: return Path(d)
    return Path(__file__).resolve().parents[2] / "out" / "matter-cache"


def digest(obj):
    def enc(o):
        if isinstance(o, np.ndarray): return np.round(o, 12).tolist()
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, tuple): return list(o)
        return str(o)
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=enc).encode()).hexdigest()[:16]


class ResponseTable:
    """K(n_s), S(n_s) in goo's bands for one particle description, on NS_GRID, spline-interpolated.
    Cached in memory and on disk (out/matter-cache, keyed by a digest of the description)."""

    def __init__(self, dielectric, morph, q=9, disk=True):
        self.dielectric, self.morph, self.q = dielectric, morph, q
        key = digest(dict(d=dielectric.describe(), m=morph.describe(), q=q, grid=NS_GRID, v=1))
        self.key = key
        if key in _CACHE: self.K, self.S = _CACHE[key]
        else:
            path = _cache_dir() / f"{key}.npz"
            if disk and path.exists():
                z = np.load(path); self.K, self.S = z["K"], z["S"]
            else:
                self.K, self.S = km_bands(dielectric, morph, NS_GRID, q=q)
                if disk:
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True); np.savez(path, K=self.K, S=self.S)
                    except OSError: pass
            _CACHE[key] = (self.K, self.S)
        self._sK = CubicSpline(NS_GRID, self.K, axis=0); self._sS = CubicSpline(NS_GRID, self.S, axis=0)

    def __call__(self, n_s):
        """K, S (4,) at surrounding index n_s: a scalar, or 4 values (one per band, for a dispersive medium)."""
        ns = np.asarray(n_s, float)
        if ns.min() < NS_GRID[0] - 1e-9 or ns.max() > NS_GRID[-1] + 1e-9:
            raise ValueError(f"surrounding index {ns} outside the table's {NS_GRID[0]}-{NS_GRID[-1]}")
        if ns.ndim == 0: K, S = self._sK(ns), self._sS(ns)
        else:
            K = np.array([self._sK(ns[j])[j] for j in range(4)]); S = np.array([self._sS(ns[j])[j] for j in range(4)])
        # spline overshoot must not invent gain or negative scattering; a table that holds gain keeps its sign
        K = np.where((self.K >= 0).all(0), np.maximum(K, 0), K)
        return K, np.maximum(S, 0)
