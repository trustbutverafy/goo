"""Fresnel's equations, their hemispherical integrals, and surfaces as boundary operators (docs/LIGHT.md §3-§4).

Everything here is a function of the relative index m = n_beyond / n_here of the side the light leaves, and of
the incidence cosine mu. Nothing is tabulated: the diffuse reflectance is the integral of Fresnel's
reflectance over a Lambertian (cosine-weighted) hemisphere, evaluated by its closed form (Walsh 1926; Stern
1964) away from m = 1 and by its series near m = 1 (both derived and tested in docs/LIGHT.md and
tests/light/test_fresnel.py against direct quadrature).
"""
import numpy as np
from goo.light.operator import Layer

LN2 = np.log(2.0)


def fresnel(m, mu):
    """Unpolarised Fresnel reflectance for light in a medium of index n1 meeting one of n2 = m n1, at incidence
    cosine mu (0..1). Total internal reflection (m < 1 past the critical angle) gives 1."""
    m, mu = np.broadcast_arrays(np.asarray(m, np.float64), np.asarray(mu, np.float64))
    st2 = (1.0 - mu * mu) / (m * m)                       # sin^2 of the refracted angle
    tir = st2 >= 1.0
    mt = np.sqrt(np.where(tir, 0.0, 1.0 - st2))
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = (mu - m * mt) / (mu + m * mt)
        rp = (m * mu - mt) / (m * mu + mt)
        F = 0.5 * (rs * rs + rp * rp)
    F = np.where(mu + mt == 0, 1.0, F)                    # grazing: everything is reflected
    return np.where(tir, 1.0, F)


# Series of r_e(1 + d) in d = n - 1, from the closed form (docs/LIGHT.md §3): sum_k (P_k + Q_k ln d) d^k.
# The closed form loses digits to cancellation as d -> 0 (terms ~ 1/d^2); the series is used below d = 0.1,
# where both agree to 1e-16 and the truncation (order d^19) is below 1e-18.
_P = np.array([
    0,
    1/3,
    19/24 - LN2/2,
    -79/60,
    37/160 + 5*LN2/8,
    275/224 - 5*LN2/8,
    -341/210 + LN2/16,
    LN2/2 + 1489/2016,
    22189/40320 - 5*LN2/8,
    -555019/443520 + 5*LN2/16,
    LN2/8 + 2083877/2027520,
    -3*LN2/8 - 11581223/46126080,
    -22405057/49201152 + 43*LN2/128,
    513194771/738017280 - 15*LN2/128,
    -12758917/26836992 - 25*LN2/256,
    328663/4752384 + 3*LN2/16,
    82416451/350945280 - 37*LN2/256,
    -291462776963/953518325760 + 5*LN2/128,
    25*LN2/512 + 5756646774427/30512586424320])
_Q = np.array([
    0,
    0,
    1/2,
    0,
    -5/8,
    5/8,
    -1/16,
    -1/2,
    5/8,
    -5/16,
    -1/8,
    3/8,
    -43/128,
    15/128,
    25/256,
    -3/16,
    37/256,
    -5/128,
    -25/512], dtype=float)
_SERIES_BELOW = 0.1


def _re_closed(n):
    """Hemispherical external reflectance, n > 1, closed form."""
    n2 = n * n
    n4 = n2 * n2
    return (0.5 + (n - 1.0) * (3.0 * n + 1.0) / (6.0 * (n + 1.0) ** 2)
            + n2 * (n2 - 1.0) ** 2 / (n2 + 1.0) ** 3 * np.log((n - 1.0) / (n + 1.0))
            - 2.0 * n * n2 * (n2 + 2.0 * n - 1.0) / ((n2 + 1.0) * (n4 - 1.0))
            + 8.0 * n4 * (n4 + 1.0) / ((n2 + 1.0) * (n4 - 1.0) ** 2) * np.log(n))


def _re_series(d):
    ld = np.log(d)
    out = np.zeros_like(d)
    for k in range(len(_P) - 1, 0, -1):                   # Horner over the fixed series
        out = (out + _P[k] + _Q[k] * ld) * d
    return out


def external_diffuse(n):
    """r_e(n): reflectance of Lambertian flux in the lower-index medium meeting a higher one, relative index
    n >= 1: r_e = integral_0^1 F(n, mu) 2 mu dmu. r_e(1) = 0 exactly."""
    n = np.asarray(n, np.float64)
    if np.any(n < 1.0):
        raise ValueError("external_diffuse: n >= 1")
    d = n - 1.0
    out = np.zeros(n.shape)
    far = d >= _SERIES_BELOW
    near = (d > 0) & ~far
    if far.any():
        out[far] = _re_closed(n[far])
    if near.any():
        out[near] = _re_series(d[near])
    return out


def _both_sides(m):
    """(r(m), r(1/m)) for Lambertian flux meeting relative index m from one side and 1/m from the other, from
    one evaluation of r_e at max(m, 1/m)."""
    m = np.asarray(m, np.float64)
    if np.any(m <= 0):
        raise ValueError("diffuse_reflectance: relative index must be positive")
    up = m >= 1.0
    M = np.where(up, m, 1.0 / m)
    re = external_diffuse(M)
    internal = 1.0 - (1.0 - re) / (M * M)
    return np.where(up, re, internal), np.where(up, internal, re)


def diffuse_reflectance(m):
    """Reflectance of Lambertian flux at a boundary of relative index m (beyond / here), either side.

    m >= 1: r_e(m). m < 1 (leaving the denser side): radiance divided by n^2 is invariant across a lossless
    interface, so n_here^2 t(here->beyond) = n_beyond^2 t(beyond->here), which gives
    r = 1 - m^2 (1 - r_e(1/m)): the internal reflectance with total internal reflection included."""
    return _both_sides(m)[0]


def boundary(n_above, n_below):
    """A smooth or rough interface as a layer for diffuse flux on both sides: lossless, so T = 1 - R and
    T' = 1 - R'. Arrays of the leading shape; returns a one-channel Layer. Equal indices give the identity
    exactly. Roughness does not enter: on the two-flux rung both sides carry diffuse flux, so it only divides
    the power reflected from the incident beam at the top surface (see entry())."""
    n_above, n_below = np.broadcast_arrays(np.asarray(n_above, np.float64), np.asarray(n_below, np.float64))
    R, Rp = _both_sides(n_below / n_above)
    c = lambda a: a[..., None, None]
    return Layer(c(R), c(1.0 - R), c(Rp), c(1.0 - Rp))


# Gauss-Legendre nodes on [0, 1] for the rough-surface integrals
_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)
_GL_X, _GL_W = 0.5 * (_GL_X + 1.0), 0.5 * _GL_W


def tis_exponent(sigma, wavelength, n_incident=1.0):
    """a in the specular share g(mu) = exp(-a mu^2) of power reflected by a surface of rms height sigma
    (Davies 1954; Bennett & Porteus 1961): a = (4 pi n sigma / lambda)^2, lambda in vacuum."""
    return (4.0 * np.pi * n_incident * np.asarray(sigma, np.float64) / np.asarray(wavelength, np.float64)) ** 2


def specular_diffuse(m, a):
    """(specular, diffuse) parts of the hemispherical reflectance r(m) of Lambertian incidence on a surface
    with TIS exponent a (shapes broadcast). specular = int F(m, mu) exp(-a mu^2) 2mu dmu; diffuse = r - specular,
    each formed from a non-negative integrand: for a <= 1 the diffuse part is integrated (1 - exp(-a mu^2)) on
    [0, 1]; for a > 1 the specular part on [0, min(1, 6/sqrt(a))], beyond which exp(-a mu^2) < 3e-16.
    a = 0 gives specular = r exactly."""
    m, a = np.broadcast_arrays(np.asarray(m, np.float64), np.asarray(a, np.float64))
    r = diffuse_reflectance(m)
    small = a <= 1.0
    # below mu_c = sqrt(1 - m^2) (m < 1) F = 1 and the integrals are elementary; quadrature runs on [mu_c, U]
    L = np.sqrt(np.maximum(1.0 - m * m, 0.0))
    U = np.where(small, 1.0, np.minimum(1.0, 6.0 / np.sqrt(np.where(small, 1.0, a))))
    L = np.minimum(L, U)
    aL2 = a * L * L
    with np.errstate(invalid="ignore", divide="ignore"):
        e_int = np.where(a > 0, -np.expm1(-aL2) / np.where(a > 0, a, 1.0), L * L)  # int_0^L exp(-a mu^2) 2mu
    acc = np.where(small, L * L - e_int, e_int)
    if np.any(a > 0):
        # Fresnel's equations inlined, in place: on [mu_c, U] the refracted cosine is real, so no branch is needed
        inv_m2 = 1.0 / (m * m)
        span = U - L
        sign = np.where(small, -1.0, 1.0)
        shift = np.where(small, 0.0, 1.0)
        mu, mu2, mt, t1, t2, F = (np.empty(m.shape) for _ in range(6))
        for x, w in zip(_GL_X, _GL_W):                     # a loop over quadrature nodes, not over cells or bands
            np.multiply(span, x * x, out=mu); mu += L      # mu - mu_c = (U - L) t^2 takes out the sqrt at mu_c
            np.multiply(mu, mu, out=mu2)
            np.subtract(1.0, mu2, out=mt); mt *= inv_m2; np.subtract(1.0, mt, out=mt)
            np.maximum(mt, 0.0, out=mt); np.sqrt(mt, out=mt)                       # cos of the refracted angle
            np.multiply(m, mt, out=t1)                                              # rs = (mu - m mt)/(mu + m mt)
            np.subtract(mu, t1, out=F); t1 += mu; F /= t1; F *= F
            np.multiply(m, mu, out=t1)                                              # rp = (m mu - mt)/(m mu + mt)
            np.subtract(t1, mt, out=t2); t1 += mt; t2 /= t1; t2 *= t2
            F += t2                                                                 # 2 F
            np.multiply(a, mu2, out=t1); np.negative(t1, out=t1); np.expm1(t1, out=t1)
            t1 *= sign; t1 += shift                        # 1 - exp(-a mu^2) for a <= 1, exp(-a mu^2) otherwise
            F *= t1; F *= mu; F *= span
            acc += (2.0 * w * x) * F                       # 2 mu dmu = 2 mu (2 t span dt); 2F/2 = F
    spec = np.where(small, r - acc, acc)
    spec = np.where(a == 0, r, spec)
    return spec, r - spec


def entry(n_ambient, n_surface, sigma, wavelength, geometry):
    """How an illuminant's light crosses the top surface: (R, T, specular) of the leading shape.

    geometry 'diffuse': R = r(m), specular part from specular_diffuse. ('collimated', theta): R = F(m, cos theta),
    specular part exp(-a cos^2 theta) R. T = 1 - R (the surface is lossless). On the first rung the light that
    enters is diffuse inside (Saunderson's assumption); docs/LIGHT.md §4."""
    m = np.asarray(n_surface, np.float64) / np.asarray(n_ambient, np.float64)
    a = tis_exponent(sigma, wavelength, n_ambient)
    if geometry == "diffuse":
        spec, diff = specular_diffuse(m, a)
        R = spec + diff
    elif isinstance(geometry, tuple) and len(geometry) == 2 and geometry[0] == "collimated":
        mu = np.cos(float(geometry[1]))
        if not 0.0 < mu <= 1.0:
            raise ValueError("collimated light must arrive at an angle in [0, pi/2)")
        R = fresnel(m, mu)
        spec = np.exp(-a * mu * mu) * R
    else:
        raise ValueError(f"unknown light geometry: {geometry!r}")
    R, spec = np.broadcast_arrays(R, spec)
    return R, 1.0 - R, spec
