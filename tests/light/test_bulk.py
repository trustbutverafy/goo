"""The homogeneous film's operator: Kubelka-Munk (G4), its limits, the thick limit (G5), and gain."""
import warnings
import numpy as np, pytest
from goo.light import homogeneous, scalar, symmetric, star, passivity_excess
from goo.units import um


def km_hyperbolic(K, S, d):
    """Kubelka's closed form (1948), written in a, b, independently of goo.light: a = 1 + K/S, b = sqrt(a^2 - 1),
    R = sinh(bSd) / (a sinh(bSd) + b cosh(bSd)), T = b / (a sinh(bSd) + b cosh(bSd)). For K, S > 0."""
    a = 1 + K / S
    b = np.sqrt(a * a - 1)
    sh, ch = np.sinh(b * S * d), np.cosh(b * S * d)
    return sh / (a * sh + b * ch), b / (a * sh + b * ch)


def kubelka_over_ground(K, S, d, Rg):
    """Kubelka's layer over a ground of reflectance Rg: R = (1 - Rg (a - b coth bSd)) / (a - Rg + b coth bSd)."""
    a = 1 + K / S
    b = np.sqrt(a * a - 1)
    cth = 1 / np.tanh(b * S * d)
    return (1 - Rg * (a - b * cth)) / (a - Rg + b * cth)


def test_G4_scalar_case_reproduces_the_kubelka_munk_closed_form():
    rng = np.random.default_rng(4)
    K = 10 ** rng.uniform(1, 5, 500)
    S = 10 ** rng.uniform(1, 5, 500)
    d = 10 ** rng.uniform(-7, -4, 500)
    R, T, ok = homogeneous(K, S, d)
    Rr, Tr = km_hyperbolic(K, S, d)
    assert ok.all()
    assert np.max(np.abs(R - Rr)) < 1e-13 and np.max(np.abs(T - Tr)) < 1e-13
    Rg = rng.uniform(0, 1, 500)
    ground = scalar(Rg, np.zeros(500))
    over = star(symmetric(R[:, None, None], T[:, None, None]), ground).R[:, 0, 0]
    assert np.max(np.abs(over - kubelka_over_ground(K, S, d, Rg))) < 1e-13


def test_limits_are_exact_or_stable():
    K, S = np.array([0.0, 3e4, 0.0, 7e5]), np.array([0.0, 0.0, 2e4, 1e3])
    R, T, ok = homogeneous(K, S, 0.0)                                   # thickness -> 0: the identity, exactly
    assert np.all(R == 0.0) and np.all(T == 1.0) and ok.all()
    R, T, _ = homogeneous(0.0, 0.0, 50 * um)                            # nothing in it
    assert R == 0.0 and T == 1.0
    d = np.logspace(-9, -2, 50)
    R, T, _ = homogeneous(4e4, 0.0, d)                                  # S -> 0: Beer-Lambert
    assert np.all(R == 0.0) and np.allclose(T, np.exp(-4e4 * d), rtol=1e-13, atol=0)
    R, T, _ = homogeneous(1e-12, 2e4, d)                                # K -> 0: R + T -> 1, R = Sd/(1 + Sd)
    assert np.allclose(R, 2e4 * d / (1 + 2e4 * d), rtol=1e-9) and np.all(np.abs(R + T - 1) < 1e-9)
    R0, T0, _ = homogeneous(0.0, 2e4, d)
    assert np.allclose(R0, 2e4 * d / (1 + 2e4 * d), rtol=1e-14, atol=0) and np.allclose(R0 + T0, 1, atol=1e-15)
    # continuity across the series switch (x = 1e-4) and in K at fixed S
    x = np.array([1e-4 * (1 - 1e-9), 1e-4 * (1 + 1e-9)])
    R, T, _ = homogeneous(x / 1e-6, 0.0, 1e-6)
    assert abs(T[0] - T[1]) < 1e-12


def test_no_warnings_and_finite_over_extremes():
    K = np.array([0.0, 1e-30, 1.0, 1e3, 1e6, 1e9, 1e12])
    S = K[:, None]
    d = np.array([0.0, 1e-12, 1e-6, 1e-3, 1.0, np.inf])[:, None, None]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        R, T, ok = homogeneous(K, S, d)
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(T)) and ok.all()
    L = symmetric(R[..., None, None], T[..., None, None])
    assert np.max(passivity_excess(L)) <= 4e-16


def test_G5_thick_limit_shows_its_own_reflectance_whatever_lies_below():
    K, S = np.array([1e3, 2e4, 5e5]), np.array([5e4, 1e5, 2e5])
    Rinf = 1 + K / S - np.sqrt((K / S) ** 2 + 2 * K / S)
    R, T, _ = homogeneous(K, S, np.inf)
    assert np.allclose(R, Rinf, rtol=1e-14, atol=0) and np.all(T == 0)
    R, T, _ = homogeneous(K, S, 0.05)                                   # 5 cm: optically thick, finite
    assert np.allclose(R, Rinf, rtol=1e-14, atol=0) and np.all(T < 1e-200)
    film = symmetric(R[:, None, None], T[:, None, None])
    for Rg in [0.0, 0.5, 1.0]:
        assert np.max(np.abs(star(film, scalar(np.full(3, Rg), np.zeros(3))).R[:, 0, 0] - Rinf)) < 1e-14
    R, T, _ = homogeneous(0.0, 1e5, np.inf)                             # lossless and infinite: everything back
    assert R == 1.0 and T == 0.0


def gain_reference(K, S, d):
    """Solve the two-flux equations with gain directly: i(0) = 1, j(d) = 0, as a linear boundary value problem
    for the 2x2 transfer matrix exp(M d) (scipy.linalg.expm), independently of goo.light's closed forms."""
    from scipy.linalg import expm
    M = np.array([[-(K + S), S], [-S, K + S]])
    P = expm(M * d)                                     # [i(d), j(d)] = P [i(0), j(0)]
    j0 = -P[1, 0] / P[1, 1]                             # j(d) = 0
    return j0, P[0, 0] + P[0, 1] * j0                   # R = j(0), T = i(d)


@pytest.mark.parametrize("K,S,d", [(-1e3, 1e4, 30 * um), (-1e4, 1e4, 20 * um), (-3e4, 1e4, 5 * um),
                                   (-2e4, 1e4, 30 * um), (-5e4, 1e3, 60 * um), (-1e3, 0.0, 1e-3)])
def test_gain_below_threshold_solves_the_two_flux_equations(K, S, d):
    R, T, ok = homogeneous(K, S, d)
    Rr, Tr = gain_reference(K, S, d)
    assert ok and abs(R - Rr) < 1e-10 * max(1, abs(Rr)) and abs(T - Tr) < 1e-10 * max(1, abs(Tr))
    assert R + T > 1                                    # not passive
    assert passivity_excess(symmetric(np.array(R)[None, None], np.array(T)[None, None])) > 0


def test_pure_gain_amplifies_without_threshold():
    R, T, ok = homogeneous(-2e3, 0.0, np.array([1e-4, 1e-3, 5e-3]))
    assert ok.all() and np.all(R == 0) and np.allclose(T, np.exp(2e3 * np.array([1e-4, 1e-3, 5e-3])), rtol=1e-12)


def test_lasing_threshold_is_where_the_two_flux_denominator_first_vanishes():
    for K, S in [(-1e4, 1e4), (-3e4, 1e4), (-2e4, 1e4), (-1.9e4, 1e4)]:
        d = np.linspace(1e-9, 2e-3, 20001)
        z = K * (K + 2 * S) + 0j
        x = np.sqrt(z) * d
        D = np.real(np.cosh(x) + (K + S) * d * np.where(np.abs(x) > 0, np.sinh(x) / np.where(x == 0, 1, x), 1))
        first = np.argmax(D <= 0) if np.any(D <= 0) else d.size
        _, _, ok = homogeneous(K, S, d)
        assert np.all(ok[:first]) and not np.any(ok[first:]), (K, S)
        R, T, _ = homogeneous(K, S, d)
        assert np.all(np.isfinite(R)) and np.all(R[first:] == 0) and np.all(T[first:] == 0)
    _, _, ok = homogeneous(-1e3, 1e4, np.inf)
    assert not ok
