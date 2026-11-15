"""goo.classic.matter.search: inverse design. Given a phenotype wanted, find the nearest *possible* description, and
separately what *impossible* description would hit the target exactly and which laws it breaks.

Each search evaluates candidates with the engine's own layer formula (phenotype.film_over -> Canvas.km_layer,
the same composition rule as Canvas.reflectance), and K, S derived by Mie (response.km_bands, coarse 20 nm
sampling and a 5-point size rule during the search; the winner is re-derived at full resolution and its
data sheet made in the engine). The optimiser is scipy's differential evolution, seeded, with small budgets.

  (a) cyan:  a pigment for the light cyan-to-blue region the stock palette misses
  (b) wet-dark: a paint dark when its particles sit in water and white when they sit in air
  (c) nir ink: colourless in the visible, strongly absorbing at 1.5 um
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import differential_evolution, brentq
from goo.classic.engine import PIGMENTS, GROUND_RGBN
from goo.classic.palette import Palette, lab
from goo.classic.matter.dielectric import Oscillators, Gauss, Lorentz, Free, WATER, LINSEED, HC
from goo.classic.matter.morphology import LogNormal, Sphere
from goo.classic.matter.substance import Substance
from goo.classic.matter import response, laws, phenotype
from goo.classic.matter.phenotype import film_over, lab_to_linear

N_OIL = 1.48
PAINT = dict(pigment_vol=0.3, oil_vol=0.45)            # a 0.75-unit (190 um) opaque film at phi = 0.4
WHITE_SHARES = np.array([0.0, 0.25, 0.5, 0.7, 0.85, 0.93, 0.97])
STOCK = [p for p in ("carbon", "white", "ultramarine", "ochre", "sienna", "rose", "green", "yellow", "red",
                     "ivory_black", "dioxazine") if p in PIGMENTS]


def _de(f, bounds, seed=0, maxiter=40, popsize=10):
    return differential_evolution(f, bounds, seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-6, polish=False,
                                  init="sobol", updating="immediate")


# ------------------------------------------------------------------------------------------- (a) cyan
def stock_gap(targets_lab, palette=None):
    """Best stock mix (up to three pigments, goo.classic.palette) for each target, as oil paint over gesso."""
    pal = palette or Palette(STOCK, steps=8, max_pigments=3)
    out = []
    for t in np.atleast_2d(targets_lab):
        m, pred, dE = pal.match(lab_to_linear(t), GROUND_RGBN[:3], PAINT["pigment_vol"], PAINT["oil_vol"])
        out.append((m, pred, dE))
    return out


def _tints(K, S):
    """Lab of the candidate tinted with stock white at WHITE_SHARES, as PAINT over gesso."""
    Kw, Sw = np.array(PIGMENTS["white"]["K"]), np.array(PIGMENTS["white"]["S"])
    v = PAINT["pigment_vol"]
    R = np.array([film_over(K, S, v * (1 - w), PAINT["oil_vol"], GROUND_RGBN, extra=[(Kw, Sw, v * w)]) for w in WHITE_SHARES])
    return lab(R[:, :3])


def cyan_family(p):
    """eps_inf, a red-orange absorption band (A, E0, s), a UV band (A, E0), log10 d."""
    diel = Oscillators(p[0], (Gauss(p[1], p[2], p[3]), Gauss(p[4], p[5], 0.6)))
    return diel, Sphere(LogNormal(10 ** p[6], 1.5))


CYAN_BOUNDS = [(1.8, 8.0), (0.0, 8.0), (1.55, 2.4), (0.08, 0.8), (0.0, 6.0), (3.0, 6.0), (np.log10(0.04), np.log10(3.0))]


def search_cyan(target_lab, seed=0, maxiter=30):
    target_lab = np.asarray(target_lab, float)

    def loss(p):
        K, S = response.km_bands(*cyan_family(p), [N_OIL], coarse=True, q=5)
        return float(np.linalg.norm(_tints(K[0], S[0]) - target_lab, axis=1).min())

    r = _de(loss, CYAN_BOUNDS, seed=seed, maxiter=maxiter)
    diel, morph = cyan_family(r.x)
    sub = Substance("cyan (searched)", diel, morph, density=3.0)
    K, S = sub.ks(N_OIL, exact=True)
    L = _tints(K, S); dE = np.linalg.norm(L - target_lab, axis=1); i = int(np.argmin(dE))
    out = dict(sub=sub, p=r.x, dE=float(dE[i]), white_share=float(WHITE_SHARES[i]), lab=L[i], K=K, S=S)
    out["impossible"] = exact_band_hit(target_lab, K, S, i)
    return out


def exact_band_hit(target_lab, K, S, i):
    """The impossible complement of a search: keep the found S and solve each visible band's K so the colour
    is exact. If every K comes out >= 0 and the change is small the target needed no law broken beyond what the
    possible substance already gives; K < 0 would be gain (passivity); a K that no causal n, k gives is
    reported as the dispersion-free step it requires."""
    Kw, Sw = np.array(PIGMENTS["white"]["K"]), np.array(PIGMENTS["white"]["S"])
    v = PAINT["pigment_vol"]; w = WHITE_SHARES[i]
    Rt = lab_to_linear(target_lab)
    from goo.classic.matter.laws import km_any
    Kx = np.array(K, float); f_p = v * (1 - w) * 10
    St = S * f_p + Sw * v * w * 10
    for b in range(3):
        def f(k):
            Kt = k * f_p + PAINT["oil_vol"] * 10 * phenotype.OIL_K[b] + Kw[b] * v * w * 10
            R, T = km_any(np.array([Kt]), np.array([St[b]]))
            return float((R + T * T * GROUND_RGBN[b] / (1 - R * GROUND_RGBN[b]))[0] - Rt[b])
        grid = np.r_[-np.geomspace(2 * St[b] / f_p, 1e-6, 60), 0.0, np.geomspace(1e-6, 1e4, 80)]
        vals = np.array([f(k) for k in grid]); ok = np.isfinite(vals)
        g, vv = grid[ok], vals[ok]
        ch = np.nonzero(np.sign(vv[:-1]) != np.sign(vv[1:]))[0]
        if ch.size == 0: Kx[b] = np.nan; continue
        c = ch[np.argmin(np.abs(g[ch] - K[b]))]                       # the root nearest the found K
        Kx[b] = brentq(f, g[c], g[c + 1])
    return dict(K=Kx, dK=Kx - np.asarray(K), gain=bool(np.nanmin(Kx[:3]) < 0))


# --------------------------------------------------------------------------------------- (b) wet-dark
WETDARK_FILM = 0.1          # pigment volume (units) in a 0.2-unit (50 um) film at phi = 0.5, over black
WATER_N = WATER.nk(np.array([650.0, 550.0, 450.0, 1550.0])).real


def wet_dry(diel, morph, coarse=True, q=5, Rb=0.0):
    """(L in water, L in air) of the film over a black ground (R = Rb), the particles' surroundings water or air."""
    lam, idx = response.band_grid(coarse)
    Kw, Sw = response.km_spectral_dispersive(diel, morph, lam, WATER, q)          # water with its dispersion
    Ka, Sa = response.km_spectral(diel, morph, lam, 1.0, q)
    out = []
    for K, S in ((Kw, Sw), (Ka, Sa)):
        K, S = np.atleast_2d(K), np.atleast_2d(S)
        Kb, Sb = response.band_means(K, idx)[0], response.band_means(S, idx)[0]
        from goo.classic.engine import Canvas
        R, T = Canvas.km_layer(Kb * WETDARK_FILM * 10, Sb * WETDARK_FILM * 10)
        out.append((lab((R + T * T * Rb / (1 - R * Rb))[:3])[0], Kb, Sb))
    return out


def wetdark_family(p):
    """Clear spheres: eps_inf and one UV oscillator (f, E0) for the dispersion; log10 d; sigma_g."""
    return Oscillators(p[0], (Lorentz(p[1], p[2], 0.0),)), Sphere(LogNormal(10 ** p[3], p[4]))


WETDARK_BOUNDS = [(1.0, 1.6), (0.0, 2.0), (6.0, 20.0), (np.log10(0.1), np.log10(5.0)), (1.1, 2.0)]


def search_wetdark(seed=0, maxiter=25, L_wet=10.0, L_dry=90.0):
    def loss(p):
        (Lw, _, _), (La, _, _) = wet_dry(*wetdark_family(p))
        return max(0.0, Lw - L_wet) + max(0.0, L_dry - La) + 0.01 * (Lw - La)
    r = _de(loss, WETDARK_BOUNDS, seed=seed, maxiter=maxiter)
    diel, morph = wetdark_family(r.x)
    sub = Substance("wet-dark (searched)", diel, morph, density=2.9)
    (Lw, Kw, Sw), (La, Ka, Sa) = wet_dry(diel, morph, coarse=False, q=9)
    out = dict(sub=sub, p=r.x, L_wet=Lw, L_dry=La, K_wet=Kw, S_wet=Sw, K_dry=Ka, S_dry=Sa,
               n_D=diel.n_at(589.3))
    # impossible complement: white (R = 1) over black at this film needs light out = light in with T lost to
    # the black: only gain can do it. Solve the K < 0 that makes R = 1 in each band with the dry S.
    from goo.classic.matter.laws import km_any
    Kneed = []
    for b in range(3):
        s = Sa[b] * WETDARK_FILM * 10
        try: Kneed.append(brentq(lambda k: km_any(k, s)[0] - 1.0, -2 * s * 0.999, 0.0) / (WETDARK_FILM * 10))
        except ValueError: Kneed.append(np.nan)
    out["impossible"] = dict(K_for_R1=np.array(Kneed), note="R_dry = 1 over black at finite S needs K < 0 (gain); "
                             "R_wet = 0 needs no law broken (index-matched to water, as the possible one nearly is)")
    return out


# ---------------------------------------------------------------------------------------- (c) nir ink
NIR_FILM = dict(pigment_vol=0.03, oil_vol=0.07)    # a 25 um glaze at phi = 0.3, over gesso
K_N_TARGET = 3.0                                   # as absorbing at 1.5 um as goo's carbon


BLACK = np.full(4, 0.03)


def nir_eval(K, S):
    """Colour difference the glaze makes against clear oil, the worse of over gesso and over black (so a
    hazy scatterer is not colourless), and its band reflectances over gesso."""
    out = []
    for Rb in (GROUND_RGBN, BLACK):
        base = film_over(np.zeros(4), np.zeros(4), 0.0, NIR_FILM["oil_vol"], Rb)
        R = film_over(K, S, NIR_FILM["pigment_vol"], NIR_FILM["oil_vol"], Rb)
        out.append((float(np.linalg.norm(lab(R[:3]) - lab(base[:3]))), R, base))
    return max(out[0][0], out[1][0]), out[0][1], out[0][2]


def nir_family(p):
    """eps_inf, a NIR band (A, E0, s), a UV band strength (at 6 eV), log10 d."""
    return Oscillators(p[0], (Gauss(p[1], p[2], p[3]), Lorentz(p[4], 6.0, 0.5))), Sphere(LogNormal(10 ** p[5], 1.4))


NIR_BOUNDS = [(1.0, 3.0), (0.0, 12.0), (0.6, 1.3), (0.03, 0.5), (0.0, 1.5), (np.log10(0.01), np.log10(1.0))]


def search_nir(seed=0, maxiter=30):
    def loss(p):
        K, S = response.km_bands(*nir_family(p), [N_OIL], coarse=True, q=5)
        dE, _, _ = nir_eval(K[0], S[0])
        return dE + 10 * max(0.0, np.log(K_N_TARGET / max(K[0][3], 1e-9)))
    r = _de(loss, NIR_BOUNDS, seed=seed, maxiter=maxiter)
    diel, morph = nir_family(r.x)
    sub = Substance("nir ink (searched)", diel, morph, density=2.0)
    K, S = sub.ks(N_OIL, exact=True)
    dE, R, base = nir_eval(K, S)
    out = dict(sub=sub, p=r.x, K=K, S=S, dE=dE, R=R, base=base)
    # impossible complement: n equal to oil's at every wavelength, k = 0 in the visible, the found NIR band's
    # k at 1.5 um, with no dispersion: colourless exactly, absorbing exactly; not one response.
    lam = np.linspace(250, 2500, 451)
    k_nir = diel.nk(np.array([1550.0]))[0].imag
    k = k_nir * np.exp(-((lam - 1550) / 120.0) ** 2)
    free = Substance("nir ink (impossible)", Free(lam, N_OIL, k, note="index matched to oil, no dispersion from the NIR band"),
                     morph, density=2.0)
    Ki, Si = free.ks(N_OIL, exact=True)
    out["impossible"] = dict(sub=free, K=Ki, S=Si, dE=nir_eval(Ki, Si)[0], report=laws.report(free))
    return out
