"""goo.matter.laws: the admissibility laws of the matter constitution, each a named predicate over a
description, returning a Verdict (pass or fail, a number saying how badly, a sentence).

  passivity       Im eps >= 0 at every wavelength the description covers (it absorbs or passes, never
                  amplifies). amount: the most negative Im eps (0 if it passes).
  causality       n and k are two faces of one response: a numerical Kramers-Kronig test over the covered range
                  (below). amount: the rms of what no passive extension can explain, relative to |eps - 1|.
  boundedness     eps(inf) = eps_inf >= 1; the f-sum weight (2/pi) int E eps2 dE <= EP_MAX^2 (a plasma energy of
                  30 eV: the valence electron density of the densest common solids, diamond ~31 eV); particle
                  sizes 2 nm .. 500 um, sigma_g in [1, 4], shell / void ratios in [0, 1), densities > 0.
  thermodynamic   viscosity, vapour pressure, surface tension, change rates >= 0; an irreversible change has no
                  reverse rate (time runs one way).
  conservation    every change maps volume to volume: product + sinks - sources = 1 per unit consumed.
  reciprocity     an isotropic scalar eps is reciprocal by construction; a claim of non-reciprocity needs a
                  declared bias (a magnetic field): without one it fails.

The Kramers-Kronig test. On a finite range [E1, E2] the measured eps2 determines, by the principal-value
integral eps1(E) - 1 = (2/pi) P int E' eps2(E') / (E'^2 - E^2) dE', only the in-range part of eps1. Absorption
outside the range adds, inside it, a sum of terms c / (W^2 - E^2) with c >= 0: rising with E for W > E2,
falling (-c / (E^2 - w^2)) for w < E1, a constant for W -> inf. Those terms, with non-negative weights, are
*every* passive extension (any passive eps2 outside the range is a non-negative mixture of such poles). So
the test fits the residual eps1 - 1 - KK_in by non-negative least squares on a dense set of out-of-range poles
and reports what is left. A large remainder proves (to the discretisation's accuracy) that no passive material
has this n and k on this range. A small remainder does not prove the converse: it says the data are consistent
with some passive extension, not that one exists in nature, and features narrower than the grid, noise, and
an absorbing tail exactly at the edges limit what any finite table can show. The in-range integral uses
Maclaurin's alternate-point rule on a uniform energy grid (Ohta & Ishida 1988, Appl. Spectrosc. 42, 952).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import nnls
from goo.matter.dielectric import HC, Oscillators, Lorentz, Gauss, Drude

EP_MAX = 30.0                     # eV
D_MIN, D_MAX = 0.002, 500.0       # um
OSC_RANGE_NM = (250.0, 2500.0)    # where an oscillator description is tested
KK_TOL = 0.02


@dataclass(frozen=True)
class Verdict:
    law: str
    ok: bool
    amount: float
    sentence: str
    def __str__(self): return f"{'pass' if self.ok else 'FAIL'} {self.law} ({self.amount:.3g}): {self.sentence}"


def _range(diel):
    lo, hi = diel.range_nm
    if not np.isfinite(hi) or lo <= 0: lo, hi = OSC_RANGE_NM
    return lo, hi


def _dielectrics(sub):
    out = [("", sub.dielectric)]
    m = sub.morphology
    if m is not None and getattr(m, "kind", "") == "coated": out.append(("shell ", m.shell))
    return out


def _min_width(diel):
    if isinstance(diel, Oscillators):
        w = [t.gamma for t in diel.terms if isinstance(t, (Lorentz, Drude)) and t.gamma > 0]
        w += [abs(t.s) for t in diel.terms if isinstance(t, Gauss)]
        return min(w) if w else 1.0
    return None


# ------------------------------------------------------------------------------------------------- laws
def passivity(sub):
    worst, where = 0.0, None
    for label, d in _dielectrics(sub):
        lo, hi = _range(d)
        lam = np.geomspace(lo, hi, 2000)
        e2 = d.eps(lam).imag
        i = int(np.argmin(e2))
        if e2[i] < worst: worst, where = float(e2[i]), (label, lam[i])
    if worst < 0:
        return Verdict("passivity", False, -worst, f"{where[0]}Im eps = {worst:.3g} at {where[1]:.0f} nm: it amplifies light (gain)")
    return Verdict("passivity", True, 0.0, "Im eps >= 0 everywhere it is described")


def kk_residual(diel, npts=None):
    """(relative rms remainder, E grid, eps1 - 1, in-range KK part, fitted extension)."""
    lo, hi = _range(diel)
    E1, E2 = HC / hi, HC / lo
    w = _min_width(diel)
    if npts is None:
        npts = 801 if w is None else int(np.clip((E2 - E1) / (w / 6), 801, 6001))
        if w is None and hasattr(diel, "lam_nm"): npts = int(np.clip(4 * len(diel.lam_nm), 801, 6001))
    E = np.linspace(E1, E2, npts); h = E[1] - E[0]
    eps = diel.eps(HC / E); e1, e2 = eps.real - 1, eps.imag
    kk = np.zeros(npts)
    j = np.arange(npts)
    for par in (0, 1):                                      # Maclaurin: sum over j with j - i odd
        ii = j[j % 2 == par]; jj = j[j % 2 != par]
        f = E[jj] * e2[jj]
        kk[ii] = (4 * h / np.pi) * (f[None, :] / (E[jj][None, :] ** 2 - E[ii][:, None] ** 2)).sum(1)
    r = e1 - kk
    W_up = E2 * np.array([1.003, 1.01, 1.03, 1.06, 1.1, 1.2, 1.35, 1.6, 2.0, 3.0, 5.0])
    w_dn = E1 * np.array([0.0, 0.3, 0.55, 0.7, 0.8, 0.88, 0.94, 0.97, 0.99, 0.997])
    cols = [np.ones_like(E)] + [1 / (Wu ** 2 - E ** 2) for Wu in W_up] + [-1 / (E ** 2 - wd ** 2) for wd in w_dn]
    A = np.stack(cols, 1); sc = np.abs(A).max(0); A = A / sc
    c, _ = nnls(A, r, maxiter=2000)
    ext = A @ c
    scale = max(np.abs(e1).max(), np.abs(e2).max(), 1e-6)
    rel = float(np.sqrt(np.mean((r - ext) ** 2)) / scale)
    return rel, E, e1, kk, ext


def causality(sub):
    worst, lab = 0.0, ""
    for label, d in _dielectrics(sub):
        rel = kk_residual(d)[0]
        if rel > worst: worst, lab = rel, label
    if worst > KK_TOL:
        return Verdict("causality", False, worst, f"{lab}refraction and absorption belong to no single passive response "
                       f"(Kramers-Kronig remainder {worst:.1%} of |eps-1|, tolerance {KK_TOL:.0%})")
    return Verdict("causality", True, worst, f"Kramers-Kronig consistent over the described range (remainder {worst:.2%})")


def boundedness(sub):
    bad = []
    for label, d in _dielectrics(sub):
        if isinstance(d, Oscillators):
            if d.eps_inf < 1: bad.append((1 - d.eps_inf, f"{label}eps_inf = {d.eps_inf:.3g} < 1"))
            f = sum(max(t.strength(), 0) for t in d.terms)
            if f > EP_MAX ** 2: bad.append((f / EP_MAX ** 2 - 1, f"{label}f-sum weight {np.sqrt(f):.1f} eV plasma energy > {EP_MAX:.0f} eV"))
        else:
            lo, hi = _range(d); E = np.linspace(HC / hi, HC / lo, 1000)
            f = 2 / np.pi * np.trapezoid(E * np.maximum(d.eps(HC / E).imag, 0), E)
            if f > EP_MAX ** 2: bad.append((f / EP_MAX ** 2 - 1, f"{label}partial f-sum {np.sqrt(f):.1f} eV > {EP_MAX:.0f} eV"))
    m = sub.morphology
    if m is not None:
        d = m.size.d_um
        if not (D_MIN <= d <= D_MAX): bad.append((abs(np.log10(d / np.clip(d, D_MIN, D_MAX))), f"diameter {d:.3g} um outside {D_MIN}-{D_MAX}"))
        if not (1.0 <= m.size.sigma_g <= 4.0): bad.append((abs(m.size.sigma_g - np.clip(m.size.sigma_g, 1, 4)), f"sigma_g {m.size.sigma_g:.3g} outside [1, 4]"))
        r = getattr(m, "ratio", None)
        if r is not None and not (0 <= r < 1): bad.append((abs(r - np.clip(r, 0, 0.999)), f"radius ratio {r:.3g} outside [0, 1)"))
    if sub.density <= 0: bad.append((-sub.density + 1e-12, f"density {sub.density} <= 0"))
    if bad:
        return Verdict("boundedness", False, max(b[0] for b in bad), "; ".join(b[1] for b in bad))
    return Verdict("boundedness", True, 0.0, "within sum-rule and geometric bounds")


def thermodynamic(sub):
    bad = []
    for k in ("viscosity", "p_sat", "surface_tension"):
        v = getattr(sub, k, None)
        if v is not None and v < 0: bad.append((-v, f"{k} = {v:.3g} < 0"))
    for c in sub.changes:
        if c.rate < 0: bad.append((-c.rate, f"change to {c.product} has rate {c.rate:.3g} < 0"))
        if c.reverse_rate < 0: bad.append((-c.reverse_rate, f"reverse rate {c.reverse_rate:.3g} < 0"))
        if not c.reversible and c.reverse_rate > 0:
            bad.append((c.reverse_rate, f"irreversible change to {c.product} runs backwards at {c.reverse_rate:.3g}"))
    if bad: return Verdict("thermodynamic", False, max(b[0] for b in bad), "; ".join(b[1] for b in bad))
    return Verdict("thermodynamic", True, 0.0, "every rate and transport coefficient is non-negative")


def conservation(sub):
    bad = []
    for c in sub.changes:
        tot = c.yield_volume + sum(v for _, v in c.sinks) - sum(v for _, v in c.sources)
        if abs(tot - 1) > 1e-9 or c.yield_volume < 0:
            bad.append((abs(tot - 1), f"change to {c.product}: volume out / in = {tot:.4g}"))
    if bad: return Verdict("conservation", False, max(b[0] for b in bad), "; ".join(b[1] for b in bad))
    return Verdict("conservation", True, 0.0, "every change maps volume to volume" if sub.changes else "no changes declared")


def reciprocity(sub):
    if sub.claims.get("nonreciprocal"):
        if sub.claims.get("bias_field"):
            return Verdict("reciprocity", True, 0.0, "non-reciprocal under a declared bias field (goo's two-flux optics cannot show it)")
        return Verdict("reciprocity", False, 1.0, "claims non-reciprocity with no bias: a passive linear isotropic medium is reciprocal")
    return Verdict("reciprocity", True, 0.0, "isotropic scalar eps: reciprocal by construction")


LAWS = (passivity, causality, boundedness, thermodynamic, conservation, reciprocity)


def verdicts(sub): return [law(sub) for law in LAWS]


def classify(sub, v=None):
    """'real' (provenance to a measured material, and passes), 'possible' (passes, no provenance),
    'impossible' (fails something)."""
    v = verdicts(sub) if v is None else v
    if not all(x.ok for x in v): return "impossible"
    p = sub.provenance
    return "real" if (p is not None and p.ties_to_material) else "possible"


def report(sub):
    v = verdicts(sub); k = classify(sub, v)
    return dict(name=sub.name, klass=k, passes=[x.law for x in v if x.ok], fails=[x.law for x in v if not x.ok],
                verdicts=[str(x) for x in v])


# ------------------------------------------------------------------------------------- gain, and its debt
def km_any(K, S):
    """Kubelka-Munk layer (R, T) for any sign of K (K, S already times thickness), continuing the classic
    formulas analytically: for -2S < K < 0, b is imaginary and sinh becomes sin. Where the denominator reaches 0
    the layer has crossed its lasing threshold: R, T are returned as inf."""
    K = np.asarray(K, float); S = np.maximum(np.asarray(S, float), 1e-9)
    a = 1 + K / S; b = np.sqrt((a * a - 1).astype(complex))
    b = np.where(np.abs(b) < 1e-7, 1e-7, b)                    # a = 1 (no absorption): the engine's own floor
    bS = b * S
    bS = np.where(bS.real > 30, 30 + 1j * bS.imag, bS)
    sh, ch = np.sinh(bS), np.cosh(bS)
    D = a * sh + b * ch
    R = (sh / D).real; T = (b / D).real
    # the denominator starts at b > 0 for a thin film and must not cross zero as the film thickens:
    #   -1 < a < 1 (b = i beta):  threshold at beta S = atan2(beta, -a)
    #   a < -1 (b real):          threshold at b S = atanh(b / |a|)
    beta = np.abs(b.imag); br = np.abs(b.real)
    with np.errstate(invalid="ignore", divide="ignore"):
        thr = np.where(np.abs(a) < 1, beta * S >= np.arctan2(beta, -a),
                       np.where(a <= -1, br * S >= np.arctanh(np.minimum(br / np.abs(a), 1 - 1e-16)), False))
    R = np.where(thr, np.inf, R); T = np.where(thr, np.inf, T)
    return R, T


def gain_debt(sub, n_s=1.48, pigment_units=0.1):
    """What a reservoir must supply for a substance with gain. Per unit particle volume and unit irradiance:
    the spectral power created, -mu_a (1/um), at its worst; per goo band, for a film holding `pigment_units`
    of pure pigment (in 0.025 mm units, i.e. K and S are multiplied by it) over a black ground, the light out
    per unit light in beyond what came in: R + T - 1 (> 0 is debt). A passive film has R + T <= 1."""
    from goo.matter import response
    lam = np.linspace(400, 700, 61)
    mu_a, mu_s, g = response.cross_sections(sub.dielectric, sub.morphology, lam, [n_s])
    K, S = sub.ks(n_s, exact=True)
    R, T = km_any(K * pigment_units, S * pigment_units)
    return dict(spectral_debt_per_um=float(np.maximum(-mu_a[0], 0).max()),
                worst_nm=float(lam[np.argmin(mu_a[0])]),
                K=K, S=S, R=R, T=T, debt=np.maximum(R + T - 1, 0))


# What an energy ledger in the engine would have to record, per cell, per band, per tick, for this debt to
# have somewhere to go (the engine does not keep it yet; SPEC.md I1 is the volume ledger only):
ENERGY_LEDGER = dict(
    incident="irradiance arriving at the top surface (per band; the light is a property of the optics model)",
    reflected="R x incident, leaving the top surface",
    transmitted_to_ground="what reaches the ground and is absorbed or returned (part of R)",
    absorbed="incident - reflected, summed over layers; negative where a layer has gain",
    drawn="max(0, -absorbed) per species with gain, booked against that species' declared reservoir",
    reservoir="a finite capacity per species (energy per unit volume of the substance), declared with it",
    exhausted="when a reservoir is empty the species' K is taken as max(K, 0) from then on: the law that "
              "the debt cannot exceed the reservoir, and the receipt that it was reached",
    invariant="incident = reflected + absorbed_passive - drawn, per band, to round-off (an optics I1)",
)
