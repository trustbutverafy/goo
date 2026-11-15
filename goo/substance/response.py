"""From a particle description to two-flux coefficients per unit particle volume, as functions of the host index;
and the precomputed table the optical map interpolates. Promoted from goo/classic/matter/response.py, rewritten
in SI with goo's arbitrary bands and without classic's 0.025 mm unit.

1. Single particle: Mie efficiencies Q_abs, Q_sca and g of a sphere (or coated or hollow sphere) in a
   non-absorbing host of real index n_h: m = (n + ik) / n_h, x = pi d n_h / lambda (goo.substance.mie). Proved
   (Maxwell's equations for a sphere), to the series truncation of Wiscombe (1980).
2. Population, per unit particle volume (1/m): mu_a = <3 Q_abs / (2 d)>_v, mu_s = <3 Q_sca / (2 d)>_v,
   g = <g Q_sca / d>_v / <Q_sca / d>_v over the volume distribution (Gauss-Hermite in ln d, `Q_NODES` nodes).
   Anisotropic crystals: cross-sections averaged over the parts of eps (2/3 o + 1/3 e, assumed). Independent
   scattering: linear in particle volume.
3. Two-flux (van Gemert & Star 1987, Lasers Life Sci. 1, 287; the diffusion limit for diffuse light):
   K = 2 mu_a, S = (3/4) mu_s (1 - g), the -mu_a/4 term of S dropped (it makes S negative for strong absorbers,
   where the diffusion limit fails anyway). Valid for scattering-dominated, optically thick films.
4. Bands: a band's value is the mean over `samples` wavelengths at the midpoints of equal sub-intervals of its box
   (centre +- width / 2), one per SAMPLE_STEP (10 nm), at most MAX_SAMPLES. A flat spectrum within the band is
   assumed.
5. Host index: precomputed on HOST_GRID (1.00 to 1.80 by 0.01) per band and interpolated by a cubic spline in the
   host index (`OpticalTable`). Its error against direct Mie is measured in tests/substance/test_optics.py and stated
   in docs/SUBSTANCES.md. Outside the grid the value is clipped to its end and the cell is out of domain.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from goo.substance import mie
from goo.substance.morphology import NotDerivable, SPHERICAL

Q_NODES = 16
HOST_GRID = np.round(np.arange(1.0, 1.8 + 1e-9, 0.01), 10)
SAMPLE_STEP = 10e-9
MAX_SAMPLES = 8


def band_samples(bands):
    """(lam, owner): sample wavelengths (m) and the band each belongs to."""
    lam, own = [], []
    for b, (c, w) in enumerate(zip(bands.centres, bands.widths)):
        n = int(np.clip(round(w / SAMPLE_STEP), 1, MAX_SAMPLES))
        lam.extend(c - w / 2 + w * (np.arange(n) + 0.5) / n); own.extend([b] * n)
    return np.asarray(lam), np.asarray(own)


def band_mean(spec, own, B):
    """(..., L) -> (..., B)."""
    return np.stack([spec[..., own == b].mean(-1) for b in range(B)], -1)


def cross_sections(eps, morph, lam, n_host, q=Q_NODES):
    """mu_a, mu_s (1/m per unit particle volume fraction) and g, shape (len(n_host), len(lam)), for particles of
    dielectric `eps` and morphology `morph` in non-absorbing hosts of real indices n_host."""
    if morph is None:
        raise NotDerivable("a continuous phase has no particles: no particle cross-sections")
    if morph.kind not in SPHERICAL:
        raise NotDerivable(f"optics of particle kind {getattr(morph, 'name', morph.kind)!r} is not derived by goo")
    lam = np.atleast_1d(np.asarray(lam, float)); ns = np.atleast_1d(np.asarray(n_host, float))
    d, w = morph.size.nodes(q)
    NS, D, LAM = np.meshgrid(ns, d, lam, indexing="ij")             # (N, Q, L)
    X = np.pi * D * NS / LAM
    Wt = (1.5 * w / d)[None, :, None]
    mu_a = np.zeros((ns.size, lam.size)); mu_s = np.zeros_like(mu_a); gs = np.zeros_like(mu_a)
    for pw, m_p in eps.parts(lam):
        M = m_p[None, None, :] / NS
        if morph.kind == "sphere":
            qe, qs, qa, g = mie.efficiencies(M, X)
        elif morph.kind == "coated":
            shell = [m for _, m in morph.shell.parts(lam)]
            msh = sum(m for m in shell) / len(shell)
            qe, qs, qa, g = mie.coated_efficiencies(M, msh[None, None, :] / NS, morph.ratio * X, X)
        else:
            qe, qs, qa, g = mie.coated_efficiencies(1.0 / NS + 0j, M, morph.ratio * X, X)
        a = (Wt * qa).sum(1); sc = (Wt * qs).sum(1)
        mu_a += pw * a; mu_s += pw * sc; gs += pw * (Wt * qs * g).sum(1)
    g = np.divide(gs, mu_s, out=np.zeros_like(gs), where=mu_s > 0)
    return mu_a, mu_s, g


def two_flux(eps, morph, lam, n_host, q=Q_NODES):
    """K = 2 mu_a, S = (3/4) mu_s (1 - g), 1/m per unit particle volume fraction, (len(n_host), len(lam))."""
    mu_a, mu_s, g = cross_sections(eps, morph, lam, n_host, q)
    return 2 * mu_a, 0.75 * mu_s * (1 - g)


def bands_key(bands):
    return (tuple(np.round(bands.centres, 15)), tuple(np.round(bands.widths, 15)))


class OpticalTable:
    """K(n_h), S(n_h) per band for one particle substance, on HOST_GRID, cubic-spline interpolated per band.
    `in_range[b]`: every sample of band b lies where the substance's eps is described."""

    def __init__(self, sub, bands, q=Q_NODES):
        lam, own = band_samples(bands)
        K, S = two_flux(sub.eps, sub.morph, lam, HOST_GRID, q)
        # a passive particle (Im m >= 0 at every sample) absorbs nothing negative: Q_abs = Q_ext - Q_sca leaves round-off
        # of either sign where k = 0, and K >= 0 is then a theorem, so it is imposed; a particle with gain keeps its sign
        ds = [sub.eps] + ([sub.morph.shell] if sub.morph.kind == "coated" else [])
        self.passive = all(np.all(m.imag >= 0) for d in ds for _, m in d.parts(lam))
        if self.passive:
            K = np.maximum(K, 0.0)
        S = np.maximum(S, 0.0)                                   # mu_s >= 0 and g <= 1: S >= 0 is a theorem
        self.K = band_mean(K, own, bands.B); self.S = band_mean(S, own, bands.B)     # (N, B)
        inr = np.ones(lam.shape, bool)
        for d in ([sub.eps] + ([sub.morph.shell] if sub.morph.kind == "coated" else [])):
            inr &= d.in_range(lam)
        self.in_range = np.array([inr[own == b].all() for b in range(bands.B)])
        self._c = [CubicSpline(HOST_GRID, T, axis=0).c for T in (self.K, self.S)]   # (4, N-1, B)

    def __call__(self, n_h):
        """K, S at host indices n_h [B, ...] (one plane per band). Returns (K, S, inside_grid)."""
        x = np.asarray(n_h, float)
        B = x.shape[0]
        inside = (x >= HOST_GRID[0] - 1e-12) & (x <= HOST_GRID[-1] + 1e-12)
        xc = np.clip(x, HOST_GRID[0], HOST_GRID[-1])
        i = np.clip(np.searchsorted(HOST_GRID, xc, side="right") - 1, 0, HOST_GRID.size - 2)
        t = xc - HOST_GRID[i]
        b = np.arange(B).reshape((B,) + (1,) * (x.ndim - 1))
        out = []
        for c in self._c:
            out.append(((c[0][i, b] * t + c[1][i, b]) * t + c[2][i, b]) * t + c[3][i, b])
        return out[0], out[1], inside


_TABLES = {}


def table(sub, bands):
    key = (sub, bands_key(bands))
    if key not in _TABLES:
        _TABLES[key] = OpticalTable(sub, bands)
    return _TABLES[key]
