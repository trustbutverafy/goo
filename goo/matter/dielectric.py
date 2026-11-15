"""goo.matter.dielectric: how a substance bends and absorbs light, as a dielectric function eps(lambda).

Convention: time dependence exp(-i omega t), so a passive (absorbing) medium has Im eps >= 0 and the complex
index n + ik = sqrt(eps) (principal branch) has k >= 0. Photon energy E (eV) = HC / lambda (nm).

Three kinds of description, one interface (`eps(lam_nm)`, `nk(lam_nm)`, `describe()`):

  Oscillators  eps(E) = eps_inf + sum of Lorentz, Drude and causal-Gaussian terms. Each term is the response of
               a damped harmonic oscillator (or a superposition of them), so eps - eps_inf is analytic in the
               upper half of the complex frequency plane and vanishes at infinity: it is Kramers-Kronig
               consistent by construction, and passive (Im eps >= 0 at every real frequency) when every
               strength and width is non-negative. eps_inf >= 1 stands for oscillators above the range, which
               are passive only if eps_inf >= 1. Sources: Lorentz/Drude: Bohren & Huffman (1983) ch. 9;
               the causal Gaussian: De Sousa Meneses, Malki & Echegut (2006), J. Non-Cryst. Solids 352, 769
               ("Structure and lattice dynamics of binary lead silicate glasses ...", the causal Gaussian
               profile with the Dawson function), also Brendel & Bormann (1992).
  Tabulated    measured n and k on a wavelength grid, with a citation (refractiveindex.info entries, a CC0
               measurement): the description of a real material. Interpolated linearly inside its range;
               outside it the description says nothing (asking raises, unless `extrapolate="flat"`).
  Free         arbitrary n and k arrays with no claim at all. Free is how impossible substances are written: it
               can violate passivity (k < 0) or causality (n and k that belong to no single response).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.special import dawsn

HC = 1239.84193                # eV nm


def energy(lam_nm): return HC / np.asarray(lam_nm, float)


def nk_from_eps(eps):
    """n + ik = sqrt(eps), the branch with n >= 0 (so k >= 0 exactly when Im eps >= 0)."""
    m = np.sqrt(np.asarray(eps, complex))
    return np.where(m.real < 0, -m, m)


# ------------------------------------------------------------------------------------------------- terms
@dataclass(frozen=True)
class Lorentz:
    """f E0^2 / (E0^2 - E^2 - i gamma E): a bound oscillator at E0 (eV), strength f (dimensionless, the
    static contribution to eps), damping gamma (eV). gamma = 0 is the lossless (Sellmeier) limit."""
    f: float
    E0: float
    gamma: float = 0.0

    def eps(self, E): return self.f * self.E0 ** 2 / (self.E0 ** 2 - E ** 2 - 1j * self.gamma * E)
    def strength(self): return self.f * self.E0 ** 2          # = (2/pi) int E eps2 dE  (the f-sum weight, eV^2)
    def signs(self): return dict(strength=self.f, width=self.gamma)


@dataclass(frozen=True)
class Drude:
    """-Ep^2 / (E^2 + i gamma E): free carriers, plasma energy Ep (eV), damping gamma (eV)."""
    Ep: float
    gamma: float

    def eps(self, E): return -self.Ep ** 2 / (E ** 2 + 1j * self.gamma * E)
    def strength(self): return np.sign(self.Ep) * self.Ep ** 2
    def signs(self): return dict(width=self.gamma)


@dataclass(frozen=True)
class Gauss:
    """Causal Gaussian absorption band: eps2 = A [exp(-((E-E0)/s)^2) - exp(-((E+E0)/s)^2)] (odd in E, as an
    absorption must be), eps1 = (2A/sqrt(pi)) [F((E+E0)/s) - F((E-E0)/s)] with F the Dawson function: the
    Kramers-Kronig transform of eps2 in closed form. A >= 0 passive; A < 0 is gain. s: 1/e half-width (eV)."""
    A: float
    E0: float
    s: float

    def eps(self, E):
        u, v = (E - self.E0) / self.s, (E + self.E0) / self.s
        e2 = self.A * (np.exp(-u * u) - np.exp(-v * v))
        e1 = 2 * self.A / np.sqrt(np.pi) * (dawsn(v) - dawsn(u))
        return e1 + 1j * e2

    def strength(self):                                       # (2/pi) int_0^inf E eps2 dE = 2 A s E0 / sqrt(pi)
        return 2 * self.A * self.s * self.E0 / np.sqrt(np.pi)
    def signs(self): return dict(strength=self.A, width=self.s)


TERMS = {"lorentz": Lorentz, "drude": Drude, "gauss": Gauss}


# ------------------------------------------------------------------------------------------- descriptions
class Dielectric:
    kind = "abstract"
    range_nm = (0.0, np.inf)              # where the description says anything
    citation = None

    def eps(self, lam_nm): raise NotImplementedError
    def nk(self, lam_nm): return nk_from_eps(self.eps(lam_nm))
    def n_at(self, lam_nm=589.3): return float(self.nk(np.array([lam_nm]))[0].real)
    def describe(self): return dict(kind=self.kind)


@dataclass
class Oscillators(Dielectric):
    """eps_inf + sum(terms). Passive and causal by construction when every strength and width is >= 0 and
    eps_inf >= 1; the laws (goo.matter.laws) check it rather than trust it."""
    eps_inf: float = 1.0
    terms: tuple = ()
    kind = "oscillators"

    def eps(self, lam_nm):
        E = energy(lam_nm)
        out = np.full(E.shape, self.eps_inf, complex)
        for t in self.terms: out = out + t.eps(E)
        return out

    def strength(self):
        """Total f-sum weight sum_j (2/pi) int E eps2_j dE = Ep^2 of the electrons the terms stand for (eV^2)."""
        return float(sum(t.strength() for t in self.terms))

    def describe(self):
        return dict(kind=self.kind, eps_inf=self.eps_inf,
                    terms=[dict(type=type(t).__name__.lower(), **t.__dict__) for t in self.terms])

    @staticmethod
    def from_dict(d):
        return Oscillators(d["eps_inf"], tuple(TERMS[t["type"]](**{k: v for k, v in t.items() if k != "type"})
                                               for t in d["terms"]))


@dataclass
class Tabulated(Dielectric):
    """Measured n, k at lam_nm (ascending), with a citation. Linear interpolation inside the range; the end values
    are held up to MARGIN (5%) beyond it, e.g. a table ending at 1550 nm covers goo's N box (1500-1600)."""
    lam_nm: np.ndarray
    n: np.ndarray
    k: np.ndarray
    citation: str | None = None
    extrapolate: str = "raise"           # or "flat": hold the end values (and say so in describe())
    kind = "tabulated"
    MARGIN = 0.05                        # held flat up to 5% beyond the ends (a band box around a last point)

    def __post_init__(self):
        o = np.argsort(self.lam_nm)
        self.lam_nm = np.asarray(self.lam_nm, float)[o]
        self.n = np.asarray(self.n, float)[o]; self.k = np.asarray(self.k, float)[o]
        self.range_nm = (float(self.lam_nm[0]), float(self.lam_nm[-1]))

    def nk(self, lam_nm):
        lam = np.asarray(lam_nm, float)
        m = self.MARGIN
        if self.extrapolate == "raise" and (lam.min() < self.range_nm[0] * (1 - m) or lam.max() > self.range_nm[1] * (1 + m)):
            raise ValueError(f"{self.kind} description covers {self.range_nm[0]:.0f}-{self.range_nm[1]:.0f} nm only; "
                             f"asked {lam.min():.0f}-{lam.max():.0f} nm")
        return np.interp(lam, self.lam_nm, self.n) + 1j * np.interp(lam, self.lam_nm, self.k)

    def eps(self, lam_nm): return self.nk(lam_nm) ** 2

    def describe(self):
        return dict(kind=self.kind, range_nm=self.range_nm, points=len(self.lam_nm), citation=self.citation,
                    extrapolate=self.extrapolate)


class Free(Tabulated):
    """Arbitrary n and k on a grid. No claim: this is where impossible responses are written."""
    kind = "free"

    def __init__(self, lam_nm, n, k, note="", extrapolate="flat"):
        super().__init__(lam_nm, np.broadcast_to(n, np.shape(lam_nm)), np.broadcast_to(k, np.shape(lam_nm)),
                         None, extrapolate)
        self.note = note


# ---------------------------------------------------------------------------------------- common liquids
def sellmeier(B, C_um2, eps_inf=1.0):
    """n^2 - 1 = sum B_i lam^2 / (lam^2 - C_i) as lossless Lorentz terms at E_i = HC/(1000 sqrt(C_i))."""
    return Oscillators(eps_inf, tuple(Lorentz(b, HC / (1000.0 * np.sqrt(c)), 0.0) for b, c in zip(B, C_um2)))


# water at 20 C, 0.18-1.13 um: Daimon & Masumura (2007), Appl. Opt. 46, 3811, four-term Sellmeier
WATER = sellmeier([5.684027565e-1, 1.726177391e-1, 2.086189578e-2, 1.130748688e-1],
                  [5.101829712e-3, 1.821153936e-2, 2.620722293e-2, 1.069792721e1])
# linseed oil: one UV oscillator at 10 eV, strength set for n_D = 1.479 (measured order, 1.477-1.482);
# its dispersion (n_F - n_C ~ 0.009) is plausible, not measured
LINSEED = Oscillators(1.0, (Lorentz(1.1348, 10.0, 0.0),))
AIR = Oscillators(1.0, ())
