"""Fresnel's equations, their hemispherical integrals and boundaries, checked against direct quadrature."""
import numpy as np, pytest
from scipy.integrate import quad
from goo.light import (fresnel, external_diffuse, diffuse_reflectance, boundary, specular_diffuse, entry,
                       tis_exponent, passivity_excess)
from goo.light.interface import _re_closed, _re_series


def F_ref(m, mu):
    """Fresnel's equations written independently: amplitude coefficients from Snell's law in angles."""
    th = np.arccos(mu)
    s = np.sin(th) / m
    if s >= 1.0:
        return 1.0
    tt = np.arcsin(s)
    if th == 0.0:
        return ((1 - m) / (1 + m)) ** 2
    rs = -np.sin(th - tt) / np.sin(th + tt)
    rp = np.tan(th - tt) / np.tan(th + tt)
    return 0.5 * (rs * rs + rp * rp)


def hemi(m, weight=lambda mu: 1.0):
    pts = [np.sqrt(1 - m * m)] if m < 1 else None
    return quad(lambda mu: F_ref(m, mu) * weight(mu) * 2 * mu, 0, 1, points=pts, epsabs=1e-14, epsrel=1e-13,
                limit=400)[0]


def test_fresnel_matches_snell_form_and_normal_incidence():
    for m in [0.5, 0.8, 1.0, 1.2, 1.5, 2.4]:
        for mu in [1.0, 0.9, 0.5, 0.2, 0.01]:
            assert abs(fresnel(m, mu) - F_ref(m, mu)) < 1e-14
        assert abs(fresnel(m, 1.0) - ((m - 1) / (m + 1)) ** 2) < 1e-16
    assert fresnel(1.0, 0.3) < 1e-30 and fresnel(1.5, 0.0) == 1.0 and fresnel(0.6, 0.1) == 1.0


@pytest.mark.parametrize("m", [1.001, 1.005, 1.0101, 1.0999, 1.1001, 1.05, 1.333, 1.5, 2.0, 3.0, 10.0])
def test_external_diffuse_is_the_hemispherical_integral(m):
    assert abs(external_diffuse(m) - hemi(m)) < 2e-13


@pytest.mark.parametrize("m", [1 / 1.001, 1 / 1.05, 1 / 1.333, 1 / 1.5, 1 / 2.4])
def test_internal_diffuse_with_total_internal_reflection_is_the_integral(m):
    assert abs(diffuse_reflectance(m) - hemi(m)) < 2e-13


def test_series_and_closed_form_agree_where_they_meet_and_r_e_of_1_is_0():
    d = 0.1
    assert abs(_re_series(np.array(d)) - _re_closed(np.array(1 + d))) < 2e-16
    assert external_diffuse(1.0) == 0.0 and diffuse_reflectance(1.0) == 0.0
    # near index match r_e = d/3 + d^2 ln(d)/2 + ...: continuous and monotone from 0
    d = (1 + np.logspace(-12, 0, 400)) - 1
    r = external_diffuse(1 + d)
    assert np.all(np.diff(r) > 0) and abs(r[0] / d[0] - 1 / 3) < 1e-9
    with pytest.raises(ValueError): external_diffuse(0.9)


def test_reciprocity_n2_t_is_symmetric():
    for na, nb in [(1.0, 1.5), (1.33, 1.6), (2.0, 1.1), (1.5, 1.5)]:
        t_ab = 1 - diffuse_reflectance(nb / na)
        t_ba = 1 - diffuse_reflectance(na / nb)
        assert abs(na ** 2 * t_ab - nb ** 2 * t_ba) < 1e-14


def test_boundary_is_lossless_and_equal_indices_are_the_identity():
    na, nb = np.array([1.0, 1.4, 2.0, 1.5]), np.array([1.5, 1.4, 1.2, 1.5 + 1e-9])
    b = boundary(na, nb)
    assert np.all(np.abs(b.R + b.T - 1) < 1e-16) and np.all(np.abs(b.Rp + b.Tp - 1) < 1e-16)
    assert np.all(passivity_excess(b) <= 1e-16)
    assert b.R[1, 0, 0] == 0.0 and b.T[1, 0, 0] == 1.0 and b.Rp[1, 0, 0] == 0.0 and b.Tp[1, 0, 0] == 1.0


@pytest.mark.parametrize("m", [1.0, 1.02, 1.5, 2.2, 1 / 1.5])
@pytest.mark.parametrize("a", [0.0, 0.3, 1.0, 4.0, 50.0, 1e4])
def test_roughness_divides_the_reflected_power(m, a):
    spec, diff = specular_diffuse(m, a)
    r = diffuse_reflectance(m)
    ref = hemi(m, lambda mu: np.exp(-a * mu * mu))
    assert abs(spec - ref) < 1e-10 and abs(spec + diff - r) < 1e-15
    assert diff >= -1e-15 and spec >= -1e-15
    if a == 0:
        assert spec == r and diff == 0.0


def test_tis_and_collimated_entry():
    lam, sig = 550e-9, 20e-9
    a = tis_exponent(sig, lam)
    assert np.isclose(a, (4 * np.pi * sig / lam) ** 2)
    R, T, spec = entry(1.0, np.array([1.5]), np.array([sig]), lam, ("collimated", 0.0))
    assert np.isclose(R[0], 0.04) and R[0] + T[0] == 1 and np.isclose(spec[0], 0.04 * np.exp(-a))
    R, T, spec = entry(1.0, np.array([1.5]), np.array([0.0]), lam, ("collimated", 1.2))
    assert abs(R[0] - F_ref(1.5, np.cos(1.2))) < 1e-14 and spec[0] == R[0]
    R, T, spec = entry(1.0, np.array([1.5]), np.array([0.0]), lam, "diffuse")
    assert R[0] == external_diffuse(1.5) and spec[0] == R[0]
    with pytest.raises(ValueError): entry(1.0, 1.5, 0.0, lam, "sideways")
