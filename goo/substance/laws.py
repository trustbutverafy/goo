"""Admissibility (SPEC §4.2): five named predicates over a description, each returning a Verdict with a measured
shortfall and its unit, and `classify`.

  herglotz        eps is the boundary value of a Herglotz function: passive (Im eps >= 0 at every real frequency) and
                  causal (Kramers-Kronig). For the oscillator family with every strength and width >= 0 and
                  eps_inf >= 1 this is a theorem (each term is the response of a damped oscillator; eps_inf >= 1 is
                  what passive oscillators above the described range can contribute), and the verdict says `proved`.
                  Otherwise it is measured: the gain is max(0, -min Im eps) over the described range, and causality
                  is classic's finite-range Kramers-Kronig test (below), whose remainder must be <= KK_TOL.
                  Shortfall: the larger of the gain and the remainder's excess over KK_TOL; unit 1 (eps is
                  dimensionless; the remainder is relative to max |eps - 1|).
  stable          its terms keep F bounded below: rho > 0 (gravity), gamma >= 0 (capillarity), cohesion >= 0
                  (a real solubility parameter; for a volatile liquid dH_vap >= R T_ref), its polar and hydrogen-
                  bonding parts >= 0 and no larger than it (a dispersive part >= 0), 0 <= gamma_d <= gamma, M > 0
                  and particle size > 0 (a positive mixing volume), p_sat >= 0, a particle population's excluded-
                  volume term convex (its osmotic pressure rises with loading: packing fraction phi_m > 2/15 for
                  goo's hard-sphere form, whose x^2 psi' has derivative x (2 + 3 a x) > 0 up to phi_m iff
                  7.5 phi_m > 1), a reaction's free energy finite. Shortfall in the unit of the worst term.
  dissipative     its terms in Psi* are convex, non-negative and zero at zero force: eta > 0, separation > 0,
                  0 <= phi_percolation < 1, 0 <= glass < 1 (the viscosity diverges at a solvent fraction in the
                  range, convex in the force at every composition), D_vapour >= 0, reaction rates >= 0, and a
                  reaction that runs (rate > 0) runs downhill at its reference state (dG < 0: Psi* = r m A_+ with
                  A = -dG there), coupling's symmetric part positive semi-definite. Shortfall in the unit of the
                  worst term.
  reciprocal      linear cross-effects are symmetric (Onsager): coupling equals its transpose. An isotropic scalar
                  eps and the Hildebrand interaction (a function of a symmetric difference) are reciprocal by
                  construction. Shortfall max |L - L^T| / 2, in the coupling's unit.
  stoichiometric  every reaction conserves mass: |sum of coefficients| (kg per kg of extent), and the substance is a
                  reactant of its own reactions.

The Kramers-Kronig test (promoted from goo/classic/matter/laws.py). On a finite range [E1, E2] the in-range eps2
determines, by the principal-value integral (Maclaurin's alternate-point rule, Ohta & Ishida 1988, Appl. Spectrosc.
42, 952), the in-range part of eps1 - 1. Any passive absorption outside the range adds a non-negative mixture of
c / (W^2 - E^2) (W > E2), -c / (E^2 - w^2) (w < E1) and a constant: those span every passive extension. The test
fits the residual by non-negative least squares on a dense set of them and reports the rms of what is left,
relative to max |eps - 1|. A large remainder shows (to the discretisation) that no passive material has these n and
k on this range; a small one says only that the data are consistent with some passive extension.

classify(s): "impossible" if any law fails; "real" if possible, `material` names a material, and every field it
gives has evidence `computed` or stronger, or `checked`; else "possible". Fields left None make no claim and do not
count against "real".
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import nnls
from goo.units import R_gas, T_room
from goo.substance.dielectric import Oscillators, Lorentz, Drude, Gauss, eV, HC
from goo.substance._freeenergy import population_packing, SPHERICAL

KK_TOL = 0.02
OSC_RANGE = (200e-9, 3000e-9)        # where an oscillator description's passivity is measured when not proved
STRONG = ("proved", "exact", "computed", "checked")
LAWS = ("herglotz", "stable", "dissipative", "reciprocal", "stoichiometric")


@dataclass(frozen=True)
class Verdict:
    law: str
    ok: bool
    shortfall: float
    unit: str
    detail: str
    parts: tuple = ()                 # ((term, shortfall, unit), ...) every term measured
    method: str = "measured"          # or "proved"

    def __str__(self):
        return f"{'pass' if self.ok else 'FAIL'} {self.law} (shortfall {self.shortfall:.3g} {self.unit}): {self.detail}"


def _verdict(law, terms, ok_detail, method="measured"):
    """terms: [(term, shortfall >= 0, unit, sentence[, failed])] measured; a term fails if `failed` (default:
    shortfall > 0). A strict inequality (eta > 0) fails at equality with shortfall 0."""
    terms = [t if len(t) == 5 else t + (t[1] > 0,) for t in terms]
    bad = [t for t in terms if t[4]]
    parts = tuple((t[0], float(t[1]), t[2]) for t in terms)
    if bad:
        w = max(bad, key=lambda t: t[1])
        return Verdict(law, False, float(w[1]), w[2], "; ".join(t[3] for t in bad), parts, "measured")
    return Verdict(law, True, 0.0, terms[0][2] if terms else "1", ok_detail, parts, method)


def _positive(term, x, unit, sentence):
    """A term that must be > 0: shortfall max(0, -x), failing at x <= 0."""
    return (term, max(0.0, -x), unit, sentence, not x > 0)


# --------------------------------------------------------------------------------------------------- herglotz
def _dielectrics(s):
    out = [("", s.eps)]
    if s.morph is not None and getattr(s.morph, "kind", "") == "coated":
        out.append(("shell ", s.morph.shell))
    return out


def _osc_proved(d):
    """True when the oscillator family's theorem applies: eps_inf >= 1 and every term passive."""
    if not isinstance(d, Oscillators) or d.eps_inf < 1:
        return False
    for t in d.terms:
        if isinstance(t, Lorentz) and not (t.f >= 0 and t.gamma >= 0 and t.E0 > 0):
            return False
        if isinstance(t, Drude) and not t.gamma >= 0:
            return False
        if isinstance(t, Gauss) and not (t.A >= 0 and t.s > 0 and t.E0 > 0):
            return False
    return True


def _range(d):
    if isinstance(d, Oscillators):
        return OSC_RANGE
    return max(d.lo, d.lam[0]), min(d.hi, d.lam[-1])


def kk_residual(d, npts=None):
    """(relative rms remainder, E (eV), eps1 - 1, in-range KK part, fitted passive extension)."""
    lo, hi = _range(d)
    E1, E2 = HC / hi / eV, HC / lo / eV
    if npts is None:
        npts = 1201 if isinstance(d, Oscillators) else int(np.clip(4 * len(d.lam), 801, 6001))
        if isinstance(d, Oscillators):
            w = [t.gamma for t in d.terms if isinstance(t, (Lorentz, Drude)) and t.gamma > 0]
            w += [abs(t.s) for t in d.terms if isinstance(t, Gauss)]
            if w:
                npts = int(np.clip((E2 - E1) / (min(w) / eV / 6), 801, 6001))
    npts += (npts + 1) % 2
    E = np.linspace(E1, E2, npts); h = E[1] - E[0]
    eps = d.eps(HC / (E * eV)); e1, e2 = eps.real - 1, eps.imag
    kk = np.zeros(npts); j = np.arange(npts)
    for par in (0, 1):
        ii = j[j % 2 == par]; jj = j[j % 2 != par]
        f = E[jj] * e2[jj]
        kk[ii] = (4 * h / np.pi) * (f[None, :] / (E[jj][None, :] ** 2 - E[ii][:, None] ** 2)).sum(1)
    r = e1 - kk
    W_up = E2 * np.array([1.003, 1.01, 1.03, 1.06, 1.1, 1.2, 1.35, 1.6, 2.0, 3.0, 5.0])
    w_dn = E1 * np.array([0.0, 0.3, 0.55, 0.7, 0.8, 0.88, 0.94, 0.97, 0.99, 0.997])
    cols = [np.ones_like(E)] + [1 / (W ** 2 - E ** 2) for W in W_up] + [-1 / (E ** 2 - w ** 2) for w in w_dn]
    A = np.stack(cols, 1); sc = np.abs(A).max(0); A = A / sc
    c, _ = nnls(A, r, maxiter=4000)
    ext = A @ c
    scale = max(np.abs(e1).max(), np.abs(e2).max(), 1e-6)
    return float(np.sqrt(np.mean((r - ext) ** 2)) / scale), E, e1, kk, ext


def herglotz(s):
    terms, proved = [], True
    for label, d in _dielectrics(s):
        if _osc_proved(d):
            terms.append((f"{label}passive", 0.0, "1", ""))
            terms.append((f"{label}causal", 0.0, "1", ""))
            continue
        proved = False
        lo, hi = _range(d)
        if isinstance(d, Oscillators):
            lam = np.geomspace(lo, hi, 4000)
        else:
            g = np.asarray(d.lam)
            lam = g[(g >= lo * (1 - 1e-9)) & (g <= hi * (1 + 1e-9))]
        gain, where = 0.0, None
        for w, m in d.parts(lam):
            e2 = (m * m).imag
            i = int(np.argmin(e2))
            if -e2[i] > gain:
                gain, where = float(-e2[i]), lam[i]
        terms.append((f"{label}passive", gain, "1",
                      f"{label}Im eps = {-gain:.3g} at {where * 1e9:.0f} nm: gain" if gain > 0 else ""))
        if isinstance(d, Oscillators):
            ex = max(0.0, 1.0 - d.eps_inf)
            terms.append((f"{label}eps_inf", ex, "1", f"{label}eps_inf = {d.eps_inf:.3g} < 1" if ex else ""))
            terms.append((f"{label}causal", 0.0, "1", ""))     # every term is analytic in the upper half plane
        else:
            rel = max(kk_residual(_Part(d, p))[0] for p in range(len(d.parts_nk)))
            ex = max(0.0, rel - KK_TOL)
            terms.append((f"{label}KK remainder", rel, "1", "", False))
            terms.append((f"{label}causal", ex, "1",
                          f"{label}Kramers-Kronig remainder {rel:.1%} of |eps-1| > {KK_TOL:.0%}: no single passive "
                          "response has this n and k" if ex else ""))
    return _verdict("herglotz", terms, "passive and causal" + (" (oscillators of non-negative strength: proved)"
                                                                if proved else " on the described range"),
                    "proved" if proved else "measured")


@dataclass(frozen=True)
class _Part:
    """One principal component of a Tabulated, as a description (for the KK test)."""
    d: object
    i: int

    @property
    def lam(self): return self.d.lam
    @property
    def lo(self): return self.d.lo
    @property
    def hi(self): return self.d.hi
    @property
    def parts_nk(self): return (self.d.parts_nk[self.i],)

    def eps(self, lam):
        m = self.d.parts(lam)[self.i][1]
        return m * m


# ------------------------------------------------------------------------------------------- stable, dissipative
def stable(s):
    t = s.thermo
    T = [_positive("rho", s.rho, "kg/m^3", f"rho = {s.rho:.3g} kg/m^3 <= 0: gravity's term is unbounded below")]
    if t.gamma is not None:
        T.append(("gamma", max(0.0, -t.gamma), "J/m^2", f"gamma = {t.gamma:.3g} J/m^2 < 0: capillarity unbounded below"))
    if t.gamma_d is not None:
        top = t.gamma if t.gamma is not None else np.inf
        T.append(("gamma_d", max(0.0, -t.gamma_d, t.gamma_d - top), "J/m^2",
                  f"gamma_d = {t.gamma_d:.3g} J/m^2 outside [0, gamma]: a component of a surface energy"))
    c = s.cohesion
    if c is not None:
        T.append(("cohesion", max(0.0, -c), "J/m^3", f"cohesive energy density {c:.3g} J/m^3 < 0"))
    for nm, v in (("cohesion_p", t.cohesion_p), ("cohesion_h", t.cohesion_h)):
        if v is not None:
            T.append((nm, max(0.0, -v), "J/m^3", f"{nm} = {v:.3g} J/m^3 < 0"))
    parts = s.cohesion_parts
    if parts is not None:
        T.append(("cohesion_d", max(0.0, -parts[0]), "J/m^3", f"the dispersive part of the cohesive energy "
                                                              f"(cohesion - p - h) is {parts[0]:.3g} J/m^3 < 0"))
    if t.M is not None:
        T.append(_positive("M", t.M, "kg/mol", f"M = {t.M:.3g} kg/mol <= 0"))
    if t.p_sat is not None:
        T.append(("p_sat", max(0.0, -t.p_sat), "Pa", f"p_sat = {t.p_sat:.3g} Pa < 0"))
    if s.morph is not None:
        z = s.morph.size
        T.append(_positive("size", z.d, "m", f"particle diameter {z.d:.3g} m <= 0"))
        T.append(("sigma_g", max(0.0, 1.0 - z.sigma_g), "1", f"sigma_g {z.sigma_g} < 1"))
        if z.d > 0 and z.sigma_g >= 1:
            pm = population_packing(s) if s.morph.kind in SPHERICAL else 0.634
            T.append(("excluded volume", max(0.0, 2 / 15 - pm), "1", f"packing fraction {pm:.3g} <= 2/15: the "
                      "hard-sphere osmotic pressure would fall with loading (not convex)", not pm > 2 / 15))
    for i, rx in enumerate(s.reactions):
        if rx.dG is not None:
            T.append((f"reaction[{i}].dG", 0.0 if np.isfinite(rx.dG) else np.inf, "J/kg",
                      f"reaction {i}'s free energy {rx.dG} is not finite", not np.isfinite(rx.dG)))
        for n, M in rx.M_reservoir:
            T.append(_positive(f"reaction[{i}].M[{n}]", M, "kg/mol", f"reservoir {n}'s molar mass {M:.3g} <= 0"))
    return _verdict("stable", T, "F bounded below by its terms")


def dissipative(s):
    r = s.rheo; t = s.thermo
    T = []
    if r.eta is not None:
        T.append(_positive("eta", r.eta, "Pa s", f"eta = {r.eta:.3g} Pa s <= 0: Psi* is not convex (negative friction)"))
    if r.separation is not None:
        T.append(_positive("separation", r.separation, "m", f"separation {r.separation:.3g} m <= 0"))
    if r.phi_percolation is not None:
        p = r.phi_percolation
        T.append(("phi_percolation", max(0.0, -p, p - 1.0), "1", f"phi_percolation {p} outside [0, 1)",
                  not 0.0 <= p < 1.0))
    if r.glass is not None:
        g = r.glass
        T.append(("glass", max(0.0, -g, g - 1.0), "1", f"glass composition {g} outside [0, 1)", not 0.0 <= g < 1.0))
    if t.D_vapour is not None:
        T.append(("D_vapour", max(0.0, -t.D_vapour), "m^2/s", f"D_vapour {t.D_vapour:.3g} m^2/s < 0"))
    for i, rx in enumerate(s.reactions):
        T.append((f"reaction[{i}].rate", max(0.0, -rx.rate), "1/s", f"reaction {i} runs at rate {rx.rate:.3g} 1/s < 0"))
        if rx.dG is not None and rx.rate > 0:
            T.append((f"reaction[{i}].affinity", max(0.0, rx.dG), "J/kg", f"reaction {i} runs (rate {rx.rate:.3g} "
                      f"1/s) against its free energy (dG = {rx.dG:.3g} J/kg >= 0): it would make free energy",
                      not rx.dG < 0))
    if r.coupling is not None:
        L = np.asarray(r.coupling, float)
        ev = np.linalg.eigvalsh(0.5 * (L + L.T))
        T.append(("coupling", max(0.0, -ev.min()), "as declared",
                  f"coupling's symmetric part has eigenvalue {ev.min():.3g} < 0"))
    if not T:
        return Verdict("dissipative", True, 0.0, "1", "no dissipative terms declared", (), "proved")
    return _verdict("dissipative", T, "every term of Psi* convex, non-negative, zero at zero force")


def reciprocal(s):
    r = s.rheo
    if r.coupling is None:
        return Verdict("reciprocal", True, 0.0, "1",
                       "no cross-effects declared; isotropic eps and the Hildebrand interaction are reciprocal by "
                       "construction", (), "proved")
    L = np.asarray(r.coupling, float)
    a = float(np.abs(L - L.T).max() / 2)
    return _verdict("reciprocal", [("coupling", a, "as declared",
                                    f"coupling is not symmetric (max |L - L^T| / 2 = {a:.3g})")], "coupling symmetric")


def stoichiometric(s):
    T = []
    for i, rx in enumerate(s.reactions):
        net = sum(v for _, v in rx.stoich)
        T.append((f"reaction[{i}]", abs(net), "kg/kg", f"reaction {i} creates {net:+.3g} kg per kg of extent",
                  abs(net) > 1e-12))
        own = rx.coefficient(s.name)
        T.append((f"reaction[{i}].reactant", max(0.0, own), "kg/kg",
                  f"reaction {i} does not consume {s.name} (coefficient {own:+.3g})", not own < 0))
    if not T:
        return Verdict("stoichiometric", True, 0.0, "kg/kg", "no reactions declared", (), "proved")
    return _verdict("stoichiometric", T, "every reaction conserves mass")


PREDICATES = dict(herglotz=herglotz, stable=stable, dissipative=dissipative, reciprocal=reciprocal,
                  stoichiometric=stoichiometric)


@dataclass(frozen=True)
class Classification:
    klass: str                         # "real" | "possible" | "impossible"
    passed: tuple
    failed: tuple
    verdicts: tuple
    blocking: tuple = ()               # ((key, evidence), ...): what keeps a possible substance from "real"

    def __str__(self):
        return f"{self.klass}: passes {list(self.passed)}, fails {list(self.failed)}"


_CACHE = {}


def verdicts(s):
    if s not in _CACHE:
        _CACHE[s] = tuple(PREDICATES[k](s) for k in LAWS)
    return _CACHE[s]


def classify(s):
    v = verdicts(s)
    passed = tuple(x.law for x in v if x.ok); failed = tuple(x.law for x in v if not x.ok)
    if failed:
        return Classification("impossible", passed, failed, v)
    block = tuple((k, p.evidence) for k, p in s.prov if p.evidence not in STRONG)
    if not s.material:
        block = (("material", "none named"),) + block
    if s.reservoir != 0:
        block = (("reservoir", f"{s.reservoir} J/m^3 on an admissible substance"),) + block
    return Classification("possible" if block else "real", passed, failed, v, block)
