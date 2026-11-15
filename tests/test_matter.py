"""goo.matter: the Mie synthesis against an independent implementation and its limits; the admissibility laws
hold by construction where they must and catch violations where they must; response functions behave as the
physics says; classic fits reproduce their table within the residual they report; class labels."""
import numpy as np
import pytest
from goo.engine import PIGMENTS, Canvas
from goo.matter import (mie, laws, mechanics, Oscillators, Gauss, Lorentz, Drude, Tabulated, Free, WATER, LINSEED,
                        LogNormal, Sphere, Coated, Hollow, Shape, NotDerivable, Substance, Liquid, Provenance, Change,
                        km_bands, cross_sections, ResponseTable, sellmeier)


# ------------------------------------------------------------------------------------------------ Mie
def _cases(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = 10 ** rng.uniform(-2, 2.2, n)
    m = rng.uniform(1.0, 3.0, n) + 1j * 10 ** rng.uniform(-4, 0.5, n)
    m[:40] = m[:40].real
    return m, x


def test_mie_against_miepython():
    mp = pytest.importorskip("miepython")
    m, x = _cases()
    qe, qs, qa, g = mie.sphere(m, x)
    ref = np.array([mp.efficiencies_mx(np.conj(mi), xi)[:4] for mi, xi in zip(m, x)])  # miepython: m = n - ik
    assert np.abs(qe / ref[:, 0] - 1).max() < 1e-3
    assert np.abs(qs / ref[:, 1] - 1).max() < 1e-3
    assert np.abs(g - ref[:, 3]).max() < 1e-3


def test_mie_rayleigh_limit():
    for m in (1.5 + 0j, 1.5 + 0.1j, 2.7 + 0.01j):
        x = np.array([0.005, 0.01])
        _, qs, qa, _ = mie.sphere(m, x)
        rs, ra = mie.rayleigh(m, x)
        assert np.allclose(qs, rs, rtol=1e-3)
        if m.imag: assert np.allclose(qa, ra, rtol=1e-3)


def test_mie_geometric_limit():
    """Extinction tends to 2 (the extinction paradox) and an absorbing sphere absorbs about its cross-section."""
    qe, qs, qa, g = mie.sphere(1.5 + 0.5j, np.array([50.0, 200.0, 1000.0]))
    assert np.all(np.abs(np.diff(np.abs(qe - 2))) <= 0) or np.abs(qe[-1] - 2) < 0.02
    assert abs(qe[-1] - 2) < 0.02 and 0.8 < qa[-1] < 1.0 and g[-1] > 0.85


def test_coated_reduces_to_sphere():
    m, x = _cases(150, 1)
    x = np.minimum(x, 40)
    ref = mie.sphere(m, x)
    for f in (0.3, 0.8):
        c = mie.coated(m, m, f * x, x)
        assert np.abs(c[1] / ref[1] - 1).max() < 1e-4 and np.abs(c[0] / ref[0] - 1).max() < 1e-4
    c = mie.coated(1.0 + 0j, m, 1e-9 * x, x)                    # vanishing core
    assert np.abs(c[1] / ref[1] - 1).max() < 1e-4


# ------------------------------------------------------------------------------------- admissibility laws
def _random_oscillators(rng):
    terms = [Gauss(rng.uniform(0, 4), rng.uniform(0.8, 5.5), rng.uniform(0.04, 0.8)) for _ in range(2)]
    terms += [Lorentz(rng.uniform(0, 2), rng.uniform(0.5, 9), rng.uniform(0.03, 2)), Drude(rng.uniform(0, 3), rng.uniform(0.1, 3))]
    return Oscillators(rng.uniform(1, 6), tuple(terms))


def test_oscillators_are_passive_and_causal_by_construction():
    rng = np.random.default_rng(7)
    for _ in range(12):
        s = Substance("t", _random_oscillators(rng), Sphere())
        v = {x.law: x for x in s.laws()}
        assert v["passivity"].ok and v["causality"].ok, (v["passivity"], v["causality"])
        assert v["causality"].amount < laws.KK_TOL / 2


def test_violations_are_detected():
    lam = np.linspace(300, 2000, 400)
    gain = Free(lam, 1.6, -0.05 * np.exp(-((lam - 530) / 30) ** 2))
    v = {x.law: x for x in Substance("g", gain, Sphere()).laws()}
    assert not v["passivity"].ok and v["passivity"].amount > 0.1
    nodisp = Free(lam, 1.5, 0.5 * np.exp(-((lam - 550) / 40) ** 2))      # absorption with no dispersion
    v = {x.law: x for x in Substance("d", nodisp, Sphere()).laws()}
    assert v["passivity"].ok and not v["causality"].ok and v["causality"].amount > 5 * laws.KK_TOL
    # gain written causally: fails passivity only (the laws are independent)
    osc_gain = Oscillators(3.0, (Gauss(-0.4, 2.3, 0.12), Gauss(2.0, 4.5, 0.5)))
    v = {x.law: x for x in Substance("og", osc_gain, Sphere()).laws()}
    assert not v["passivity"].ok and v["causality"].ok
    # a transparent constant index is consistent (a pole far above); measured water too
    assert laws.causality(Substance("c", Free(lam, 1.5, 0 * lam), Sphere())).ok
    assert laws.causality(Substance("w", WATER, None)).ok


def test_boundedness_thermo_conservation_reciprocity():
    sub = Substance("b", Oscillators(0.5, (Gauss(500.0, 3, 1.0),)), Sphere(LogNormal(2000.0, 5.0)))
    v = laws.boundedness(sub); assert not v.ok and "eps_inf" in v.sentence and "f-sum" in v.sentence
    liq = Liquid("l", WATER, None, density=1.0, viscosity=-1e-3, changes=(Change("x", 1e-3, 1.0, reversible=False, reverse_rate=1e-4),))
    assert not laws.thermodynamic(liq).ok
    leak = Liquid("w", WATER, None, density=1.0, changes=(Change("vapour", 1e-4, 0.0, sinks=(("air", 0.9),)),))
    assert not laws.conservation(leak).ok
    ok = Liquid("w", WATER, None, density=1.0, changes=(Change("vapour", 1e-4, 0.0, sinks=(("air", 1.0),)),))
    assert laws.conservation(ok).ok
    assert not laws.reciprocity(Substance("r", WATER, Sphere(), claims=dict(nonreciprocal=True))).ok
    assert laws.reciprocity(Substance("r", WATER, Sphere())).ok


def test_class_labels():
    lam = np.linspace(300, 2000, 200)
    silica_n = sellmeier([0.6961663, 0.4079426, 0.8974794], [0.0684043 ** 2, 0.1162414 ** 2, 9.896161 ** 2])
    real = Substance("silica", silica_n, Sphere(LogNormal(2.0)), density=2.2,
                     provenance=Provenance("measured", "Malitson 1965, JOSA 55, 1205 (fused silica Sellmeier)", "fused silica"))
    assert real.klass == "real"
    assert Substance("p", silica_n, Sphere()).klass == "possible"
    fit = Substance("f", silica_n, Sphere(), provenance=Provenance("fit", "a table"))
    assert fit.klass == "possible"                                    # a fit is not a material
    imp = Substance("i", Oscillators(3.0, (Gauss(-0.3, 2.3, 0.1),)), Sphere(),
                    provenance=Provenance("measured", "a claimed measurement"))
    assert imp.klass == "impossible"                                  # provenance does not excuse a violation
    tab = Tabulated(lam, silica_n.nk(lam).real, 0 * lam, citation="Malitson 1965")
    assert Substance("t", tab, Sphere(), provenance=Provenance("measured", "Malitson 1965")).klass == "real"
    r = laws.report(imp); assert r["fails"] == ["passivity"]


# ------------------------------------------------------------------------------------- response functions
def test_response_vanishes_without_contrast():
    clear = Oscillators(1.3 ** 2, ())                                 # dispersionless n = 1.3
    K, S = km_bands(clear, Sphere(LogNormal(0.5, 1.5)), [1.3])
    assert np.abs(K).max() < 1e-12 and np.abs(S).max() < 1e-12
    # a weak absorber at matched n: S -> 0 quadratically in k, K stays and is linear in k
    Ss, Ks = [], []
    for k in (1e-3, 1e-4):
        eps = (1.3 + 1j * k) ** 2
        d = Free(np.linspace(300, 2000, 50), 1.3, k)
        K, S = km_bands(d, Sphere(LogNormal(0.5, 1.5)), [1.3]); Ss.append(S[0, 1]); Ks.append(K[0, 1])
    assert Ss[1] / Ss[0] < 0.02 and 0.09 < Ks[1] / Ks[0] < 0.11


def test_more_contrast_more_scattering_where_that_is_true():
    """Small particles (Rayleigh-like, x < ~1): S rises monotonically with |n_p - n_s|. Large monodisperse
    spheres: Mie resonances make Q_sca non-monotone in m, so more contrast can scatter less."""
    small = Sphere(LogNormal(0.08, 1.3))
    S = np.array([km_bands(Oscillators(n ** 2, ()), small, [1.48])[1][0] for n in (1.55, 1.7, 1.9, 2.2, 2.6)])
    assert np.all(np.diff(S[:, :3], axis=0) > 0)
    # at fixed particle index, raising n_s lowers the contrast but also shortens the wavelength in the medium
    # (x = pi d n_s / lambda): for small particles S ~ n_s^4 |(m^2-1)/(m^2+2)|^2, so from air the first step
    # *raises* S. Contrast alone is not the law; Mie is.
    K, S2 = km_bands(Oscillators(2.0 ** 2, ()), small, [1.0, 1.05, 1.5, 1.6])
    assert S2[1, 2] > S2[0, 2] and S2[3, 2] < S2[2, 2]
    m = np.linspace(1.05, 1.6, 200)
    _, qs, _, _ = mie.sphere(m, 25.0)
    assert np.any(np.diff(qs) < 0)                                    # resonances: not monotone
    _, qs_small, _, _ = mie.sphere(m, 0.3)
    assert np.all(np.diff(qs_small) > 0)


def test_response_table_interpolates():
    sub = Substance("t", Oscillators(2.4 ** 2, (Gauss(1.0, 2.0, 0.3),)), Sphere(LogNormal(0.3, 1.5)))
    tab = ResponseTable(sub.dielectric, sub.morphology, disk=False)
    for n in (1.12, 1.33, 1.48, 1.57):
        K, S = tab(n); Ke, Se = sub.ks(n, exact=True)
        assert np.allclose(K, Ke, rtol=0.02, atol=1e-4) and np.allclose(S, Se, rtol=0.02, atol=1e-4)


def test_hollow_sphere_scatters_by_its_void():
    shell = Oscillators(1.55 ** 2, ())                                # a styrene-acrylic shell, index-matched-ish to oil
    hol = km_bands(shell, Hollow(LogNormal(0.4, 1.3), 0.55), [1.5])[1][0]
    solid = km_bands(shell, Sphere(LogNormal(0.4, 1.3)), [1.5])[1][0]
    assert hol[1] > 10 * solid[1]


def test_shapes_not_derived_are_said_so():
    s = Substance("flake", Oscillators(2.0, ()), Shape("platelet", params=(("aspect", 30.0),)))
    with pytest.raises(NotDerivable): s.ks(1.48, exact=True)
    with pytest.raises(NotDerivable): mechanics.phi_max(s.morphology)


# ---------------------------------------------------------------------------------------------- mechanics
def test_mechanics_maps():
    assert abs(mechanics.phi_max(Sphere(LogNormal(1.0, 1.0))) - 0.634) < 1e-9
    assert mechanics.phi_max(Sphere(LogNormal(1.0, 1.6))) > 0.66        # polydisperse packs denser
    assert abs(mechanics.krieger_exponent(Sphere(LogNormal(1.0, 1.0))) - 2.5 * 0.634) < 1e-9
    water = Liquid("water", WATER, None, density=0.997, viscosity=0.89e-3, p_sat=3169.0, molar_mass=0.018015)
    D = mechanics.stokes_einstein(1.0, 0.89e-3)
    assert 4e-13 < D < 6e-13                                          # ~0.49 um^2/s for 1 um in water
    v = mechanics.stokes_settling(1.0, 2.65, 0.997, 0.89e-3)
    assert 0.9e-6 < v < 1.1e-6                                        # quartz 1 um: ~1 um/s
    m = water.mechanics()
    assert 0.5 < m["evap_units_per_tick"] / mechanics.EVAP_UNITS < 2   # goo's tick read from its own evap
    p = Substance("u", Oscillators(2.0), Sphere(LogNormal(5.0, 1.3)), density=2.35).mechanics(water)
    assert p["v_settle_m_s"] > 1e-5 and p["D_cells2_tick"] < 1e-4 * 0.15   # goo's D0 = 0.15 is not Brownian


# -------------------------------------------------------------------------------------------------- gain
def test_km_any_agrees_with_engine_when_passive_and_shows_gain():
    K = np.array([0.0, 0.1, 1.0, 5.0]); S = np.array([0.5, 1.0, 2.0, 0.3])
    R, T = laws.km_any(K, S); Re, Te = Canvas.km_layer(K, S)
    assert np.allclose(R, Re, atol=1e-6) and np.allclose(T, Te, atol=1e-6)
    R, T = laws.km_any(np.array([-0.2]), np.array([1.0]))
    assert R[0] + T[0] > 1                                            # light out with less light in
    R, T = laws.km_any(np.array([-1.5]), np.array([10.0]))            # past the lasing threshold
    assert np.isinf(R[0])


def test_gain_debt_is_a_number():
    sub = Substance("g", Oscillators(4.0, (Gauss(-0.3, 2.25, 0.12),)), Sphere(LogNormal(0.3, 1.4)))
    d = laws.gain_debt(sub, pigment_units=0.01)
    assert d["spectral_debt_per_um"] > 0 and 500 < d["worst_nm"] < 600 and d["debt"][1] > 0 and d["debt"][3] == 0


# --------------------------------------------------------------------------------------- classic and sheet
def test_classic_fits_reproduce_their_table_within_residual():
    from goo.matter import classic
    assert set(classic.FITS) == set(PIGMENTS) or set(classic.FITS) <= set(PIGMENTS)
    for name in classic.FITS:
        r = classic.residual(name)
        assert r["rms"] <= classic.REPORTED[name]["rms"] + 1e-6, (name, r["rms"])
        assert classic.substance(name).klass == "possible"


def test_datasheet_uses_the_engine_and_leaves_it_as_it_was():
    from goo.matter import phenotype
    before = dict(PIGMENTS)
    sub = Substance("t", Oscillators(2.4 ** 2, (Gauss(1.5, 2.0, 0.3),)), Sphere(LogNormal(0.3, 1.5)))
    d = phenotype.datasheet(sub)
    assert PIGMENTS == before
    assert d["masstone_lab"][0] < d["tint_lab"][0] and np.isfinite(d["hiding_um"])
    # an absorber in the red band: its masstone is not red
    assert d["masstone_rgb"][0] < d["masstone_rgb"][2]


def test_measured_tables_are_held_as_real_points(tmp_path):
    """A table in goo.optics-data form (as goo/data/optics.json will be): cited constants are `real`; a
    species with guessed constants is not; the loader needs no file to exist."""
    import json
    from goo.matter import tables
    assert tables.load(tmp_path / "absent.json") == {}
    lam = np.linspace(400, 800, 41).tolist() + [1550.0]
    silica = sellmeier([0.6961663, 0.4079426, 0.8974794], [0.0684043 ** 2, 0.1162414 ** 2, 9.896161 ** 2])
    n = silica.nk(np.array(lam)).real.tolist(); z = [0.0] * len(lam)
    e = lambda **kw: dict(role="pigment", n=n, k=z, size=dict(d50_um=1.0, sigma_g=1.4, provenance="assumed"),
                          K=dict(oil=z), S=dict(oil=z), **kw)
    d = dict(schema="goo.optics-data/1", wavelength_nm=lam, species=dict(
        silica=e(sources=[dict(reference="I. H. Malitson, JOSA 55, 1205 (1965)")]),
        guess=e(sources=[dict(reference="somebody")], provenance=dict(by_wavelength=dict(n=["guessed"] + ["computed"] * (len(lam) - 1)))),
        bare=e(sources=[])))
    p = tmp_path / "optics.json"; p.write_text(json.dumps(d))
    subs = tables.load(p)
    assert subs["silica"].klass == "real" and subs["guess"].klass == "possible" and subs["bare"].klass == "possible"
    K, S = subs["silica"].ks(1.48, exact=True)                      # silica (1.46) in oil (1.48): nearly invisible
    assert K.max() < 1e-9 and S.max() < 0.05
