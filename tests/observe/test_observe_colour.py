"""goo.observe.colour: the colorimetric chain. Owner: lab and observers."""
import numpy as np, pytest
from goo.film import Bands
from goo.observe import colour as C
from goo import observe as O

BANDS = Bands.uniform(380e-9, 740e-9, 20e-9).with_band(1500e-9, 100e-9)


def test_every_table_and_fit_has_a_provenance_and_none_claims_more_than_it_has():
    for k, p in C.PROV.items():
        assert p.evidence in ("exact", "assumed"), k            # nothing here was read and checked
    assert C.PROV["observer"].evidence == "assumed" and "not measured" in C.PROV["observer"].note
    assert C.PROV["daylight"].evidence == "assumed" and "NOT CIE D65" in C.PROV["daylight"].note
    assert C.evidence() == "assumed"


def test_planck_A_uses_the_exact_second_radiation_constant():
    assert C.C2 == pytest.approx(1.438776877e-2, rel=1e-9)
    lam = np.array([400e-9, 560e-9, 700e-9])
    s = C.A.spd(lam)
    ref = lam ** -5 / np.expm1(C.H_PLANCK * C.C_0 / (C.K_BOLTZ * lam * 2856.0))
    assert np.allclose(s, ref, rtol=1e-14, atol=0)              # exact to rounding


def test_a_perfect_diffuser_has_Y_one_L_100_and_maps_to_srgb_white_under_both_lights():
    for il in (C.DAYLIGHT, C.A):
        w = C.white_xyz(BANDS, il)
        assert w[1] == pytest.approx(1.0, abs=1e-15)
        L = C.lab(w, w)
        assert np.allclose(L, [100, 0, 0], atol=1e-12)
        assert np.allclose(C.srgb_linear(w, w), 1.0, atol=1e-12)


def test_infrared_bands_carry_no_colour_weight():
    W = C.weights(BANDS, C.DAYLIGHT)
    assert np.all(W[:, -1] == 0)
    assert np.all(W[:, :-1].sum(0) >= 0)


def test_srgb_matrix_is_derived_from_its_chromaticities():
    M = C.M_RGB_TO_XYZ
    for j, (x, y) in enumerate(C.SRGB_PRIMARIES):
        X, Y, Z = M[:, j]
        assert X / (X + Y + Z) == pytest.approx(x, abs=1e-12) and Y / (X + Y + Z) == pytest.approx(y, abs=1e-12)
    assert C.SRGB_WHITE[1] == pytest.approx(1.0, abs=1e-12)


def test_lab_round_trips_through_xyz():
    w = C.white_xyz(BANDS, C.A)
    rng = np.random.default_rng(1)
    X = rng.uniform(0.001, 1.0, (200, 3)) * w
    assert np.allclose(C.lab_to_xyz(C.lab(X, w), w), X, rtol=1e-12, atol=1e-14)


def test_ciede2000_agrees_with_scikit_image_and_sharmas_first_pair():
    from skimage.color import deltaE_ciede2000
    rng = np.random.default_rng(0)
    a = rng.uniform([0, -100, -100], [100, 100, 100], (5000, 3))
    b = a + rng.normal(0, 8, a.shape)
    assert np.max(np.abs(C.ciede2000(a, b) - deltaE_ciede2000(a, b))) < 1e-10
    # Sharma, Wu & Dalal (2005), test pair 1 (recalled: 2.0425)
    assert C.ciede2000([50, 2.6772, -79.7751], [50, 0, -82.7485]) == pytest.approx(2.0425, abs=1e-4)
    assert C.ciede2000([50, 10, 10], [50, 10, 10]) == 0.0


def test_grey_reflectance_is_neutral_and_monotone_in_lightness():
    for il in (C.DAYLIGHT, C.A):
        w = C.white_xyz(BANDS, il)
        Ls = [C.lab(C.xyz(np.full(BANDS.B, r), BANDS, il), w) for r in (0.05, 0.2, 0.5, 0.9)]
        assert all(abs(L[1]) < 1e-9 and abs(L[2]) < 1e-9 for L in Ls)
        assert all(Ls[i][0] < Ls[i + 1][0] for i in range(3))


def test_the_same_spectrum_under_two_lights_differs_only_through_the_light():
    R = np.linspace(0.1, 0.8, BANDS.B)
    a = C.lab(C.xyz(R, BANDS, C.DAYLIGHT), C.white_xyz(BANDS, C.DAYLIGHT))
    b = C.lab(C.xyz(R, BANDS, C.A), C.white_xyz(BANDS, C.A))
    assert C.ciede2000(a, b) > 0.1          # a sloped spectrum is not colour-constant under von Kries-free CIELAB
