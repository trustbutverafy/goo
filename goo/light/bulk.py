"""The operator of a homogeneous film from its two-flux coefficients (docs/LIGHT.md §1).

Kubelka-Munk: in a film of absorption K and scattering S (both in 1/m, the two-flux coefficients, S with the
asymmetry already taken; SPEC §4.3) the diffuse fluxes i (down) and j (up) obey

    di/dz = -(K + S) i + S j          dj/dz = (K + S) j - S i            (z downward)

With x^2 = z2 = K (K + 2S) d^2 the solution for a film of thickness d is, for every sign of K,

    D = cosh x + (K + S) d shc(x)          shc(x) = sinh(x) / x
    T = 1 / D                              R = S d shc(x) / D

cosh x and shc x are entire functions of x^2, so the same expressions hold for x^2 < 0 (cos, sin/y), which
occurs only with gain (K < 0 < K + 2S). For K >= 0 they are evaluated divided through by cosh x, which keeps
every limit finite: S -> 0 gives T = exp(-K d) (Beer-Lambert), K -> 0 gives R = S d / (1 + S d), d -> 0 the
identity exactly, d -> infinity R = R_inf = S / (K + S + sqrt(K (K + 2S))) and T = 0.

With gain (K < 0) a film has a lasing threshold: the first thickness at which D reaches 0. Below it the
operator is finite and non-negative but not passive; at and beyond it there is no steady state.
"""
import numpy as np


def _sech(x):
    e = np.exp(-x)
    return 2.0 * e / (1.0 + e * e)


def _thc(x):
    """tanh(x) / x for x >= 0, with its series near 0."""
    small = x < 1e-4
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0 - x * x / 3.0, np.tanh(xs) / xs)


def homogeneous(K, S, d):
    """(R, T, below_threshold) of a homogeneous film, symmetric (R' = R, T' = T), shapes broadcast from K, S, d.

    `d` may be +inf (a semi-infinite body). `below_threshold` is True where a steady state exists: everywhere
    for K >= 0; for gain, where the film is thinner than its lasing threshold. Where it is False, R and T are
    set to 0 (a defined value; the caller books and flags it, see look()). No value is clipped.

    The passive finite case is computed on every cell; the branches only gain or infinite depth need are
    computed on those cells alone (the same expressions, see _general)."""
    K, S, d = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (K, S, d)))
    gain = K < 0
    inf = np.isinf(d)
    special = gain | inf
    if not special.any():
        alpha = np.sqrt(K * (K + 2.0 * S))
        x = alpha * d
        u = d * _thc(x)
        den = 1.0 + (K + S) * u
        R = S * u / den
        T = _sech(x) / den
        return R, T, np.ones(K.shape, bool)
    R = np.empty(K.shape)
    T = np.empty(K.shape)
    ok = np.ones(K.shape, bool)
    plain = ~special
    if plain.any():
        R[plain], T[plain], _ = homogeneous(K[plain], S[plain], d[plain])
    R[special], T[special], ok[special] = _general(K[special], S[special], d[special])
    return R, T, ok


def _general(K, S, d):
    """homogeneous() on every branch at once: passive, gain (both signs of x^2), semi-infinite."""
    K, S, d = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (K, S, d)))
    inf = np.isinf(d)
    df = np.where(inf, 0.0, d)
    z = K * (K + 2.0 * S)                      # x^2 / d^2, in 1/m^2
    KS = K + S
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # real branch, z >= 0
        alpha = np.sqrt(np.maximum(z, 0.0))
        x = alpha * df
        u = df * _thc(x)                       # d tanh(x)/x  (m)
        den = 1.0 + KS * u
        R_r = S * u / den
        T_r = _sech(x) / den
        # gain with x^2 > 0 (K < 0, K + S < 0): 1 + K tanh/alpha cancels for large x. With c = (K+S)/alpha,
        # 1 + c = -S^2 / (alpha (alpha - K - S)) exactly, so D = cosh x + c sinh x is formed without cancellation:
        # D = (e^-x (alpha - K - S) - e^x S^2 / (alpha - K - S)) / (2 alpha)
        g = (K < 0) & (z > 0) & (x > 1.0)
        am = np.where(g, alpha - KS, 1.0)
        ex, emx = np.exp(np.where(g, x, 0.0)), np.exp(-np.where(g, x, 0.0))
        Dg = emx * am - ex * S * S / am
        T_g = 2.0 * alpha / Dg
        R_g = S * (ex - emx) / Dg
        R_r = np.where(g, R_g, R_r)
        T_r = np.where(g, T_g, T_r)
        # oscillating branch, z < 0 (gain only)
        beta = np.sqrt(np.maximum(-z, 0.0))
        y = beta * df
        sinc = np.sinc(y / np.pi)              # sin(y)/y
        Dosc = np.cos(y) + KS * df * sinc
        R_o = S * df * sinc / Dosc
        T_o = 1.0 / Dosc
        # semi-infinite
        Rinf = np.where(alpha + KS > 0, S / (KS + alpha), 0.0)
        Rinf = np.where((K == 0) & (S > 0), 1.0, Rinf)
        Tinf = np.where((K == 0) & (S == 0), 1.0, 0.0)

    osc = z < 0
    R = np.where(osc, R_o, R_r)
    T = np.where(osc, T_o, T_r)
    R = np.where(inf, Rinf, R)
    T = np.where(inf, Tinf, T)

    # lasing threshold (docs/LIGHT.md §6)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = np.arctan2(beta, -KS)                          # first zero of cos y + (K+S)/beta sin y, in (0, pi)
        osc_ok = y < t_star
        ratio = np.where(KS < 0, alpha / np.abs(np.where(KS == 0, 1.0, KS)), 0.0)
        x_star = np.where(ratio < 1.0, np.arctanh(np.minimum(ratio, 1.0 - 1e-300)), np.inf)
        real_ok = np.where(KS < 0, np.where(z > 0, x < x_star, 1.0 + KS * df > 0), True)
    ok = np.where(osc, osc_ok, real_ok)
    ok = np.where(K >= 0, True, ok)
    ok = np.where(inf & (K < 0), False, ok)                     # a semi-infinite gain medium never settles
    ok = ok & np.isfinite(R) & np.isfinite(T)
    R = np.where(ok, R, 0.0)
    T = np.where(ok, T, 0.0)
    return R, T, ok
