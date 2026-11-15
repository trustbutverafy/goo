"""search: inverse design (SPEC §10, docs/LAB.md §5). Given a target, the nearest POSSIBLE substance in a stated
parametric family, and separately the IMPOSSIBLE substance that hits the target, with the laws it breaks.

Family (every possible candidate): spheres, log-normal sizes (sigma_g = 1.5), of an oscillator dielectric function
eps_inf + one causal Gaussian band Gauss(A >= 0, E0, s) (or a lossless UV Lorentz for clear particles), at a loading in
a medium (a recipe of goo.lab.recipes). Each candidate is a Substance with `guessed` provenance (a searched point is a
guess until made and measured), made into films by composition, dried by goo.lab.dry, and looked at through the maps,
exactly as a drawdown is. The optimiser is scipy's differential evolution, seeded (on 40 nm bands for (b) and (c), on the lab's 20 nm bands for
(a), since 40 nm bands misjudge a narrow absorption band), then Nelder-Mead on the lab's 20 nm bands.

The impossible complement keeps the found substance's real index n(lambda) and its size, and solves, band by band,
the k that makes the film's reflectance hit a target spectrum: a `Free` dielectric (n and k with no claim). k < 0
where more light must come out than goes in (optical gain); a k that no causal response pairs with that n shows up
as a Kramers-Kronig remainder. `goo.substance.classify` then names the laws it fails and their shortfalls.

Demonstrations (`demo_a`, `demo_b`, `demo_c`):
  (a) a colour the library's pigments cannot mix (in oil, up to three pigments with the white)
  (b) a paint dark when wet and white when dry (particles in water, over the black stripe)
  (c) colourless in the visible, absorbing at 1.5 um (in oil, over white and black)
"""
from dataclasses import dataclass, field, replace
from itertools import combinations
import warnings
import numpy as np
from scipy.optimize import differential_evolution, minimize
from goo.evidence import Provenance
from goo.film import Film
from goo.light import look
from goo.substance import (Substance, Oscillators, Gauss, Lorentz, Free, Sphere, LogNormal, eV, maps, classify,
                           library as lib)
from goo.substance import _nk_data as D, response
from goo.substance.dielectric import HC
from goo import observe as O
from goo.lab import recipes as R
from goo.lab.dry import dry

SEARCH_BANDS = R.bands(step=40e-9)
SIGMA_G = 1.5
RHO = 3000.0
PALETTE = (lib.RUTILE, lib.SOOT, lib.HEMATITE, lib.CDS, lib.OCHRE, lib.ULTRAMARINE, lib.ROSE, lib.GREEN, lib.DIOXAZINE)


def candidate(name, eps, d, note):
    """A searched substance: every field guessed (it is a point found by an optimiser, not a material)."""
    return Substance(name, eps, Sphere(LogNormal(d, SIGMA_G)), RHO,
                     prov=dict(eps=Provenance("guessed", note=f"found by goo.lab.search: {note}"),
                               morph=Provenance("guessed", note=f"found by search: d50 {d * 1e6:.3g} um, sigma_g {SIGMA_G}"),
                               rho=Provenance("guessed", note="a placeholder density; optics do not read it"),
                               reservoir=Provenance("exact", note="no reservoir")))


def _see(stack, bands, light=None):
    light = light or O.diffuse_light(bands, O.DAYLIGHT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return look(stack, bands, light, maps)


class FastOptical:
    """goo.light's Optical protocol for a film holding one searched candidate, without building the candidate's Mie
    table on 81 host indices. Exact rearrangement of `maps.optical` where it applies: in a film with no voids the host
    is the liquids' own index and K, S are sums over substances of phi_s K_s(n_host) (independent scattering), so
    K = (h'/h) K' + phi_c K_c(n_host), with K' the maps' answer for the film without the candidate (thickness h') and
    K_c the candidate's two-flux coefficients by Mie at that one host index per band (response.two_flux, band
    means; no spline). A film with voids, or without the candidate, goes to the maps unchanged. Domain flags and
    evidence are not tracked: the winner of a search is always re-measured through the maps."""

    def __init__(self, candidate):
        self.c = candidate
        self._kc = {}

    def _cand(self, bands, n_h):
        key = (response.bands_key(bands), tuple(np.round(n_h, 12)))
        if key not in self._kc:
            lam, own = response.band_samples(bands)
            K = np.zeros(bands.B); S = np.zeros(bands.B)
            for b in range(bands.B):
                Kb, Sb = response.two_flux(self.c.eps, self.c.morph, lam[own == b], np.array([n_h[b]]))
                K[b], S[b] = Kb.mean(), Sb.mean()
            self._kc[key] = (K, S)
        return self._kc[key]

    def optical(self, film, bands):
        names = [x.name for x in film.substances]
        if self.c.name not in names:
            return maps.optical(film, bands)
        i = names.index(self.c.name)
        phim = np.asarray(maps.packing(film).value)
        part = np.array([x.morph is not None for x in film.substances])
        P, L = film.q[part].sum(0), film.q[~part].sum(0)
        if np.any(P * (1 / phim - 1) - L > 0) or film.q.shape[1:] != (1, 1):
            return maps.optical(film, bands)
        keep = [j for j in range(len(names)) if j != i]
        rest = Film(tuple(film.substances[j] for j in keep), film.q[keep])
        K1, S1, n1 = (np.asarray(x.value if hasattr(x, "value") else x) for x in maps.optical(rest, bands))
        h, h1 = film.h, rest.h
        Kc, Sc = self._cand(bands, n1[:, 0, 0])
        phi = film.q[i] / h
        K = K1 * (h1 / h)[None] + phi[None] * Kc[:, None, None]
        S = S1 * (h1 / h)[None] + phi[None] * Sc[:, None, None]
        return (K, S, n1)

    def surface(self, film, bands):
        return maps.surface(film, bands)

    def ground_optical(self, ground, bands):
        return maps.ground_optical(ground, bands)

    def ground_surface(self, ground, bands):
        return maps.ground_surface(ground, bands)


def fast_measure(film, ground, bands, candidate, illuminant=O.DAYLIGHT):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = look(R.stack(film, ground), bands, O.diffuse_light(bands, illuminant), FastOptical(candidate))
    return L, O.colour_of(L, illuminant)


def film_of(sub, binder, pvc, h=R.WET_THICKNESS, white_share=0.0, state="dry", env=R.ROOM):
    paint = R.Paint(sub.name, ((sub, 1.0),), binder, pvc)
    wet = R.wet_film(paint, binder, white_share, h)
    return wet if state == "wet" else dry(wet, env).film


def measure(film, ground, bands, illuminant=O.DAYLIGHT):
    L = _see(R.stack(film, ground), bands)
    c = O.colour_of(L, illuminant)
    return L, c


def _de(f, bounds, seed, maxiter, popsize, polish=None, polish_iter=80):
    """Differential evolution, then (if `polish` is given: the loss on the final bands) Nelder-Mead from the best
    point, clipped to the bounds. Returns (x, evaluations)."""
    r = differential_evolution(f, bounds, seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-8, polish=False,
                               init="sobol", updating="immediate")
    x, n = r.x, int(r.nfev)
    if polish is not None and polish_iter:
        lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
        m = minimize(lambda z: polish(np.clip(z, lo, hi)), x, method="Nelder-Mead",
                     options=dict(maxfev=polish_iter, xatol=1e-4, fatol=1e-4))
        if m.fun <= polish(x):
            x = np.clip(m.x, lo, hi)
        n += int(m.nfev) + 1
    return x, n


# ---------------------------------------------------------------------------------------------- (a) a gap colour
def library_gamut(binder=R.OIL, palette=PALETTE, steps=6, bands=SEARCH_BANDS, h=R.WET_THICKNESS):
    """CIELAB (daylight, dry, over white) of every mixture of up to three palette pigments on a simplex grid of
    `steps`, at the binder's loading. Returns (lab [N, 3], mixtures [(names, weights)])."""
    mixes = []
    n = len(palette)
    for k in (1, 2, 3):
        for combo in combinations(range(n), k):
            for ws in _simplex(k, steps):
                if np.all(ws > 0):
                    mixes.append((combo, ws))
    subs = tuple(palette) + tuple(s for s, _ in binder.parts)
    N = len(mixes)
    q = np.zeros((len(subs), 1, N))
    P = binder.pvc * h
    for j, (combo, ws) in enumerate(mixes):
        for i, w in zip(combo, ws):
            q[i, 0, j] = P * w
        for k, (s, f) in enumerate(binder.parts):
            q[n + k, 0, j] = (1 - binder.pvc) * h * f
    film = dry(Film(subs, q), R.ROOM).film
    L = _see(R.stack(film, "white"), bands)
    lab_ = O.colour_of(L, O.DAYLIGHT).lab[0]
    return lab_, [(tuple(palette[i].name for i in c), tuple(float(x) for x in w)) for c, w in mixes]


def _simplex(k, steps):
    if k == 1:
        return [np.array([1.0])]
    out = []
    def rec(prefix, left, depth):
        if depth == k - 1:
            out.append(np.array(prefix + [left]) / steps); return
        for i in range(left + 1):
            rec(prefix + [i], left - i, depth + 1)
    rec([], steps, 0)
    return out


def gap(target_lab, gamut):
    lab_, mixes = gamut
    dE = O.ciede2000(lab_, np.asarray(target_lab)[None])
    i = int(np.argmin(dE))
    return float(dE[i]), mixes[i], lab_[i]


A_BOUNDS = [(1.8, 6.0), (0.0, 6.0), (1.55, 2.6), (0.06, 0.8), (np.log10(0.03), np.log10(3.0)), (0.0, 0.95)]


def a_family(p):
    eps = Oscillators(p[0], (Gauss(p[1], p[2] * eV, p[3] * eV),))
    return eps, 10 ** p[4] * 1e-6, p[5]


def search_colour(target_lab, binder=R.OIL, seed=0, maxiter=10, popsize=8, bands=R.LAB_BANDS, polish_iter=80):
    """Nearest possible substance (masstone or a tint with the white, dry, over the white ground) to a CIELAB target.
    On the lab's 20 nm bands: 40 nm bands misjudge a narrow absorption band by up to ~10 dE00."""
    target_lab = np.asarray(target_lab, float)

    def loss_on(b):
        def loss(p):
            eps, d, w = a_family(p)
            s = candidate("searched", eps, d, "")
            _, c = (fast_measure if FAST else lambda f, g, bb, _s: measure(f, g, bb))(
                film_of(s, binder, binder.pvc, white_share=w), "white", b, s)
            return float(O.ciede2000(c.lab[0, 0], target_lab))
        return loss

    x, nfev = _de(loss_on(bands), A_BOUNDS, seed, maxiter, popsize, loss_on(R.LAB_BANDS), polish_iter)
    eps, d, w = a_family(x)
    note = (f"eps_inf {x[0]:.3g}, Gauss A {x[1]:.3g} at {x[2]:.3g} eV width {x[3]:.3g} eV; nearest possible to "
            f"Lab {np.round(target_lab, 1).tolist()} in {binder.name}, white share {w:.2f}")
    s = candidate("search_a", eps, d, note)
    _, c = measure(film_of(s, binder, binder.pvc, white_share=w), "white", R.LAB_BANDS)
    return dict(substance=s, params=x.tolist(), white_share=w, lab=c.lab[0, 0].tolist(),
                dE00=float(O.ciede2000(c.lab[0, 0], target_lab)), evaluations=nfev)


def target_spectrum(R_p, bands, target_lab, illuminant=O.DAYLIGHT, iters=60):
    """A spectrum near R_p with exactly the target colour: R_p exp(W^T c / max W), with c (3 numbers) found by Newton's
    method on XYZ. Multiplicative, so it stays positive and keeps R_p's shape where the observer does not look."""
    W = O.weights(bands, illuminant)
    Wn = W / W.max()
    X_t = O.lab_to_xyz(np.asarray(target_lab, float), O.white_xyz(bands, illuminant))
    R_p = np.maximum(np.asarray(R_p, float), 1e-6)
    c = np.zeros(3)
    for _ in range(iters):
        R_ = R_p * np.exp(Wn.T @ c)
        r = W @ R_ - X_t
        if np.max(np.abs(r)) < 1e-13:
            break
        J = W @ (R_[:, None] * Wn.T)
        c = c - np.linalg.solve(J, r)
    return R_p * np.exp(Wn.T @ c)


def solve_k(sub, make_stack, R_target, bands, which, k_max=1.0, k_min=-1.0, iters=40):
    """A Free substance with sub's real index n(lambda) and morphology, and k solved per band (constant across each band,
    on the library's wavelength grid) so that the diffuse reflectance of make_stack(substance) hits R_target in the
    bands `which` (bool [B]); elsewhere k = 0. Returns (substance, achieved R). |k| <= 1: beyond it a particle turns
    reflective (metal-like) and the reflectance is no longer monotone in k, so bisection could take the wrong branch."""
    lam = np.asarray(D.GRID_M, float)
    n = np.real(sub.eps.nk(lam))
    c = np.asarray(bands.centres)
    owner = np.argmin(np.abs(lam[:, None] - c[None]), 1)
    lo = np.full(bands.B, k_min); hi = np.full(bands.B, k_max)

    def make(kb):
        k = np.where(which[owner], kb[owner], 0.0)
        eps = Free.of(lam, n, k, "solved band by band to hit a target spectrum")
        return replace(sub, name=sub.name + "_impossible", eps=eps,
                       prov=dict(sub.prov) | {"eps": Provenance("guessed", note="n of the possible substance; k solved "
                                                                "band by band to hit the target (goo.lab.search.solve_k)")})

    def f(kb):
        L = _see(make_stack(make(kb)), bands)
        Rr = L.diffuse[0, 0]
        return np.where(L.diverged[0, 0], np.inf, Rr), Rr

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val, _ = f(mid)
        up = val > R_target                 # too bright: more absorption
        lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
    kb = np.where(which, 0.5 * (lo + hi), 0.0)
    s = make(kb)
    _, Rr = f(kb)
    return s, Rr


def _laws(s):
    c = classify(s)
    return dict(klass=c.klass, failed=[dict(law=v.law, shortfall=v.shortfall, unit=v.unit, detail=v.detail)
                                       for v in c.verdicts if not v.ok],
                herglotz=next(dict(ok=v.ok, shortfall=v.shortfall, detail=v.detail) for v in c.verdicts
                              if v.law == "herglotz"))


def impossible_colour(found, target_lab, binder=R.OIL, bands=R.LAB_BANDS):
    s, w = found["substance"], found["white_share"]
    mk = lambda sub: R.stack(film_of(sub, binder, binder.pvc, white_share=w), "white")
    L = _see(mk(s), bands)
    R_t = target_spectrum(L.diffuse[0, 0], bands, target_lab)
    vis = np.asarray(bands.centres) < 780e-9
    imp, Rr = solve_k(s, mk, R_t, bands, vis)
    _, c = measure(film_of(imp, binder, binder.pvc, white_share=w), "white", bands)
    return dict(substance=imp, lab=c.lab[0, 0].tolist(), dE00=float(O.ciede2000(c.lab[0, 0], target_lab)),
                k_by_band=_k_bands(imp, bands), laws=_laws(imp), R_target=R_t.tolist(), R=Rr.tolist())


def _k_bands(s, bands):
    return [float(np.mean(np.imag(s.eps.nk(np.array([c]))))) for c in bands.centres]


# ------------------------------------------------------------------------------------- (b) dark wet, white dry
B_BOUNDS = [(1.0, 3.0), (0.0, 2.0), (6.0, 20.0), (np.log10(0.05), np.log10(5.0)), (0.05, 0.5)]


def b_family(p):
    return Oscillators(p[0], (Lorentz(p[1], p[2] * eV, 0.0),)), 10 ** p[3] * 1e-6, p[4]


def wet_dry_L(s, binder, pvc, bands, h=R.WET_THICKNESS, fast=False):
    """L* over black, wet and dry. fast: the wet film (overbound) through FastOptical; the dry film has voids and
    always goes through the maps."""
    wet = film_of(s, binder, pvc, h, state="wet")
    _, cw = fast_measure(wet, "black", bands, s) if fast else measure(wet, "black", bands)
    _, cd = measure(film_of(s, binder, pvc, h, state="dry"), "black", bands)
    return float(cw.lab[0, 0, 0]), float(cd.lab[0, 0, 0])


B_BANDS = R.bands(ir=(), step=60e-9)      # L* needs no infrared, and the dry film's table is the cost: coarse first


def search_wet_dark(binder=R.WATER, seed=0, maxiter=12, popsize=8, bands=B_BANDS, L_wet=10.0, L_dry=90.0,
                    polish_iter=60):
    """Clear particles in a medium, over the black stripe: L* wet <= L_wet and L* dry >= L_dry, as near as possible."""
    def loss_on(b):
        def loss(p):
            eps, d, pvc = b_family(p)
            Lw, Ld = wet_dry_L(candidate("searched", eps, d, ""), binder, pvc, b, fast=FAST)
            return max(0.0, Lw - L_wet) + max(0.0, L_dry - Ld) + 0.01 * (Lw - Ld)
        return loss

    x, nfev = _de(loss_on(bands), B_BOUNDS, seed, maxiter, popsize, loss_on(R.LAB_BANDS), polish_iter)
    eps, d, pvc = b_family(x)
    s = candidate("search_b", eps, d, f"clear spheres, eps_inf {x[0]:.3g}, UV Lorentz f {x[1]:.3g} at {x[2]:.3g} eV "
                                      f"(n_D {eps.n_at():.4f}); in {binder.name} at pvc {pvc:.3f}: dark wet, white dry")
    Lw, Ld = wet_dry_L(s, binder, pvc, R.LAB_BANDS)
    return dict(substance=s, params=x.tolist(), pvc=pvc, L_wet=Lw, L_dry=Ld, n_D=eps.n_at(), d=d,
                evaluations=nfev)


def impossible_wet_dark(found, binder=R.WATER, bands=R.LAB_BANDS):
    """Keep n and size; solve k per band so that the DRY film over black reflects everything (L* = 100)."""
    s, pvc = found["substance"], found["pvc"]
    mk = lambda sub: R.stack(film_of(sub, binder, pvc, state="dry"), "black")
    vis = np.asarray(bands.centres) < 780e-9
    imp, Rr = solve_k(s, mk, np.ones(bands.B), bands, vis)
    Lw, Ld = wet_dry_L(imp, binder, pvc, bands)
    return dict(substance=imp, L_wet=Lw, L_dry=Ld, k_by_band=_k_bands(imp, bands), laws=_laws(imp), R=Rr.tolist())


# ----------------------------------------------------------------------- (c) clear in the visible, black at 1.5 um
C_BOUNDS = [(1.5, 3.0), (0.0, 8.0), (0.70, 1.0), (0.02, 0.3), (np.log10(0.02), np.log10(1.0)), (0.02, 0.3)]
IR_TARGET = 0.2


def c_family(p):
    return Oscillators(p[0], (Gauss(p[1], p[2] * eV, p[3] * eV),)), 10 ** p[4] * 1e-6, p[5]


FAST = True          # candidates through FastOptical (exact rearrangement, no 81-host table); False: the full maps


def ir_ink(s, pvc, binder, bands, h=R.WET_THICKNESS, fast=False):
    """(dE00 against the clear binder film, worst of over white and over black; reflectance at the band nearest
    1.5 um over white)."""
    dEs = []
    clear = dry(Film.uniform({x: h * f for x, f in binder.parts}), R.ROOM).film          # the binder alone
    for g in ("white", "black"):
        _, c0 = measure(clear, g, bands)
        Lg, c1 = (fast_measure(film_of(s, binder, pvc, h), g, bands, s) if fast else
                  measure(film_of(s, binder, pvc, h), g, bands))
        dEs.append(float(O.ciede2000(c0.lab[0, 0], c1.lab[0, 0])))
        if g == "white":
            j = int(np.argmin(np.abs(np.asarray(bands.centres) - 1.5e-6)))
            r15 = float(Lg.diffuse[0, 0, j])
    return max(dEs), r15


def search_ir(binder=R.OIL, seed=0, maxiter=12, popsize=8, bands=SEARCH_BANDS, polish_iter=40):
    def loss_on(b):
        def loss(p):
            eps, d, pvc = c_family(p)
            dE, r = ir_ink(candidate("searched", eps, d, ""), pvc, binder, b, fast=FAST)
            return dE + 30.0 * max(0.0, r - IR_TARGET)
        return loss

    x, nfev = _de(loss_on(bands), C_BOUNDS, seed, maxiter, popsize, loss_on(R.LAB_BANDS), polish_iter)
    eps, d, pvc = c_family(x)
    s = candidate("search_c", eps, d, f"eps_inf {x[0]:.3g}, Gauss A {x[1]:.3g} at {x[2]:.3g} eV "
                                      f"({HC / (x[2] * eV) * 1e9:.0f} nm) width {x[3]:.3g} eV; in {binder.name} at pvc "
                                      f"{pvc:.3f}: clear in the visible, absorbing at 1.5 um")
    dE, r15 = ir_ink(s, pvc, binder, R.LAB_BANDS)
    return dict(substance=s, params=x.tolist(), pvc=pvc, dE00_visible=dE, R_1500_over_white=r15, evaluations=nfev)


def impossible_ir(found, binder=R.OIL, bands=R.LAB_BANDS, R_ir=0.02):
    """n equal to the dry film's continuous phase's n(lambda) at every wavelength (so no scattering, no colour), k = 0 in
    the visible, and k at the 1.5 um band solved so the film over white reflects R_ir there."""
    s, pvc = found["substance"], found["pvc"]
    clear = dry(Film.uniform({x: R.WET_THICKNESS * f for x, f in binder.parts}), R.ROOM).film
    li = [i for i, x in enumerate(clear.substances) if x.morph is None]      # the dry film's continuous phases
    host = clear.substances[max(li, key=lambda i: clear.q[i].sum())]
    base = replace(s, eps=host.eps)
    mk = lambda sub: R.stack(film_of(sub, binder, pvc), "white")
    which = np.abs(np.asarray(bands.centres) - 1.5e-6) < 1e-9
    imp, Rr = solve_k(base, mk, np.full(bands.B, R_ir), bands, which)
    imp = replace(imp, name="search_c_impossible")
    dE, r15 = ir_ink(imp, pvc, binder, bands)
    return dict(substance=imp, dE00_visible=dE, R_1500_over_white=r15, k_by_band=_k_bands(imp, bands), laws=_laws(imp))


# ----------------------------------------------------------------------------------------------- demonstrations
TARGET_A = (62.0, -38.0, -22.0)      # a light turquoise (CIELAB, daylight): chosen, then checked against the gamut


def demo_a(seed=0, maxiter=10, popsize=8, steps=6):
    gam = library_gamut(steps=steps)
    g, mix, lab_ = gap(TARGET_A, gam)
    found = search_colour(TARGET_A, seed=seed, maxiter=maxiter, popsize=popsize)
    imp = impossible_colour(found, TARGET_A)
    return dict(target_lab=TARGET_A, library_best=dict(dE00=g, mixture=mix, lab=lab_.tolist()), possible=found,
                impossible=imp)


def demo_b(seed=0, maxiter=12, popsize=8):
    found = search_wet_dark(seed=seed, maxiter=maxiter, popsize=popsize)
    return dict(target="L* wet <= 10 and L* dry >= 90 over the black stripe, in water", possible=found,
                impossible=impossible_wet_dark(found))


def demo_c(seed=0, maxiter=12, popsize=8):
    found = search_ir(seed=seed, maxiter=maxiter, popsize=popsize)
    return dict(target=f"dE00 ~ 0 against the clear oil film over white and black; R(1.5 um) over white <= {IR_TARGET}",
                possible=found, impossible=impossible_ir(found))


def summary(result):
    """JSON-ready copy of a demo result (substances written in the contribution format)."""
    from goo.lab.contribute import to_json

    def conv(o):
        if isinstance(o, Substance):
            return to_json(o)
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    return conv(result)
