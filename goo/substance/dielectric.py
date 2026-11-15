"""How a substance meets light: its dielectric function eps(lambda), lambda in m (SPEC §4.1).

Convention: time dependence exp(-i omega t), so a passive (absorbing) medium has Im eps >= 0, and the complex index
n + ik = sqrt(eps) on the branch n >= 0 has k >= 0. Photon energy E = h c / lambda, in J (write it with `eV`).

Three kinds of description, one interface:

  eps(lam)        complex dielectric function at wavelengths lam (m); for an anisotropic crystal, the orientation
                  average 1/3 tr eps (weighted parts)
  parts(lam)      ((weight, n + ik), ...): the principal components whose cross-sections are averaged for randomly
                  oriented grains (one part for an isotropic medium; 2/3 ordinary + 1/3 extraordinary for a uniaxial
                  crystal: an approximation, `assumed`)
  in_range(lam)   where the description says anything; outside it a value is still returned (a formula evaluated
                  or a table held at its end) and every map using it marks the cell out of domain

  Oscillators  eps(E) = eps_inf + sum of Lorentz, Drude and causal-Gaussian terms (Bohren & Huffman 1983 ch. 9;
               the causal Gaussian: De Sousa Meneses, Malki & Echegut 2006, J. Non-Cryst. Solids 352, 769). Each term
               is analytic in the upper half frequency plane, so the family is causal by construction; it is passive
               (Herglotz) when every strength and width is >= 0 and eps_inf >= 1. Defined at every wavelength.
  Tabulated    measured n and k on a grid, with a citation: the description of a real material.
  Free         arbitrary n and k with no claim: how impossible responses are written.

Promoted from goo/classic/matter/dielectric.py with wavelengths in m and energies in J; the classic water and linseed
descriptions were left behind (goo.substance.library holds the kernel's).
"""
from dataclasses import dataclass, field
from functools import cached_property
import numpy as np
from scipy.special import dawsn
from goo.units import c_0

eV = 1.602176634e-19                 # J, exact (SI 2019)
h_planck = 6.62607015e-34            # J s, exact (SI 2019)
HC = h_planck * c_0                  # J m


def energy(lam):
    """Photon energy (J) at wavelength lam (m)."""
    return HC / np.asarray(lam, np.float64)


def nk_from_eps(eps):
    """n + ik = sqrt(eps) on the branch n >= 0 (so k >= 0 exactly when Im eps >= 0)."""
    m = np.sqrt(np.asarray(eps, np.complex128))
    return np.where(m.real < 0, -m, m)


# ----------------------------------------------------------------------------------------------- oscillator terms
@dataclass(frozen=True)
class Lorentz:
    """f E0^2 / (E0^2 - E^2 - i gamma E): a bound oscillator at E0 (J), strength f (the static contribution to eps,
    dimensionless), damping gamma (J). gamma = 0 is the lossless (Sellmeier) limit."""
    f: float
    E0: float
    gamma: float = 0.0

    def eps(self, E): return self.f * self.E0 ** 2 / (self.E0 ** 2 - E ** 2 - 1j * self.gamma * E)
    def signs(self): return dict(strength=self.f, width=self.gamma)


@dataclass(frozen=True)
class Drude:
    """-Ep^2 / (E^2 + i gamma E): free carriers, plasma energy Ep (J), damping gamma (J)."""
    Ep: float
    gamma: float

    def eps(self, E): return -self.Ep ** 2 / (E ** 2 + 1j * self.gamma * E)
    def signs(self): return dict(width=self.gamma)


@dataclass(frozen=True)
class Gauss:
    """Causal Gaussian band: eps2 = A [exp(-((E-E0)/s)^2) - exp(-((E+E0)/s)^2)], eps1 its Kramers-Kronig transform
    (2A/sqrt(pi)) [F((E+E0)/s) - F((E-E0)/s)], F the Dawson function. A >= 0 absorbs; A < 0 is gain (still causal).
    E0 and s (1/e half-width) in J."""
    A: float
    E0: float
    s: float

    def eps(self, E):
        u, v = (E - self.E0) / self.s, (E + self.E0) / self.s
        return 2 * self.A / np.sqrt(np.pi) * (dawsn(v) - dawsn(u)) + 1j * self.A * (np.exp(-u * u) - np.exp(-v * v))

    def signs(self): return dict(strength=self.A, width=self.s)


class Dielectric:
    kind = "abstract"

    def eps(self, lam): raise NotImplementedError
    def parts(self, lam): return ((1.0, nk_from_eps(self.eps(lam))),)
    def nk(self, lam): return nk_from_eps(self.eps(lam))
    def in_range(self, lam): return np.ones(np.shape(lam), bool)

    def n_at(self, lam=589.3e-9):
        """Real index at one wavelength (default the sodium D line)."""
        return float(np.real(self.nk(np.array([lam]))[0]))


@dataclass(frozen=True)
class Oscillators(Dielectric):
    """eps_inf + sum(terms). Causal by construction; passive when every strength and width is >= 0 and eps_inf >= 1
    (goo.substance.laws.herglotz checks it rather than trusts it)."""
    eps_inf: float = 1.0
    terms: tuple = ()
    kind = "oscillators"

    def eps(self, lam):
        E = energy(lam)
        out = np.full(E.shape, self.eps_inf, np.complex128)
        for t in self.terms:
            out = out + t.eps(E)
        return out


def _table(a):
    return np.asarray(a, np.float64)


@dataclass(frozen=True)
class Tabulated(Dielectric):
    """Measured constants. lam: grid (m, ascending); parts: ((weight, n, k), ...) on the grid, weights summing to 1;
    lo, hi: the range (m) where the source's data hold; citation: the measurement. Linear interpolation on the grid,
    held at the grid's ends beyond it; outside [lo, hi] `in_range` is false."""
    lam: tuple
    parts_nk: tuple
    lo: float
    hi: float
    citation: str = ""
    kind = "tabulated"

    def __post_init__(self):
        lam = np.asarray(self.lam, float)
        if lam.ndim != 1 or np.any(np.diff(lam) <= 0):
            raise ValueError("Tabulated: lam must be ascending")
        w = [p[0] for p in self.parts_nk]
        if abs(sum(w) - 1.0) > 1e-12 or min(w) <= 0:
            raise ValueError("Tabulated: part weights must be positive and sum to 1")
        for _, n, k in self.parts_nk:
            if len(n) != lam.size or len(k) != lam.size:
                raise ValueError("Tabulated: n and k must be on the grid")
        if not self.lo < self.hi:
            raise ValueError("Tabulated: lo < hi")

    @cached_property
    def _arrays(self):
        return _table(self.lam), [(w, _table(n) + 1j * _table(k)) for w, n, k in self.parts_nk]

    def parts(self, lam):
        g, ps = self._arrays
        x = np.asarray(lam, float)
        return tuple((w, np.interp(x, g, m.real) + 1j * np.interp(x, g, m.imag)) for w, m in ps)

    def eps(self, lam):
        return sum(w * m * m for w, m in self.parts(lam))

    def nk(self, lam):
        return nk_from_eps(self.eps(lam))

    def in_range(self, lam):
        x = np.asarray(lam, float)
        return (x >= self.lo * (1 - 1e-9)) & (x <= self.hi * (1 + 1e-9))


@dataclass(frozen=True)
class Free(Tabulated):
    """Arbitrary n and k on a grid, no claim. Its range is its grid."""
    kind = "free"

    @classmethod
    def of(cls, lam, n, k, note=""):
        lam = tuple(float(x) for x in lam)
        n = tuple(np.broadcast_to(np.asarray(n, float), (len(lam),)).tolist())
        k = tuple(np.broadcast_to(np.asarray(k, float), (len(lam),)).tolist())
        return cls(lam, ((1.0, n, k),), lam[0], lam[-1], note)


def sellmeier(B, C_m2, eps_inf=1.0):
    """n^2 - eps_inf = sum B_i lam^2 / (lam^2 - C_i) (C_i in m^2) as lossless Lorentz terms at E_i = hc / sqrt(C_i)."""
    return Oscillators(eps_inf, tuple(Lorentz(b, HC / np.sqrt(c), 0.0) for b, c in zip(B, C_m2)))


def from_source(src, k_zero_note=True):
    """A Tabulated from one or more entries of goo.substance._nk_data.SOURCES: ((entry, weight), ...)."""
    from goo.substance import _nk_data as D
    parts, lo, hi, cites = [], 0.0, np.inf, []
    for e, w in src:
        parts.append((w, e["n"], e["kk"]))
        lo, hi = max(lo, e["range_m"][0]), min(hi, e["range_m"][1])
        cites.append(f"{e['reference']} [{D.DATABASE}: {e['file']}, commit {D.COMMIT[:12]}]")
    return Tabulated(D.GRID_M, tuple(parts), lo, hi, "; ".join(dict.fromkeys(cites)))
