"""SPEC §4.3: the maps other than `optical` (tests/substance/test_optics.py): shapes and units of the table, sanity
anchors, exact limits, evidence propagation and in_domain flags, and that no map reads a name or a position."""
import dataclasses
import numpy as np
import pytest
from goo.evidence import Derived, Bounded
from goo.film import Film, Bands
from goo.substance import maps, library as L, Sphere, LogNormal
from goo.units import um, nm, mPa_s, k_B, T_room, g_0, celsius, R_gas, kPa
from conftest import liquid, particle

ALLMAPS = ["density", "viscosity", "packing", "surface_tension"]


def film(amounts, H=2, W=3):
    return Film.uniform(amounts, H, W)


def test_shapes_and_types(small_bands):
    f = film({L.WATER: 30 * um, L.GUM_ARABIC: 10 * um, L.RUTILE: 2 * um, L.SOOT: 0.1 * um})
    S, (H, W) = 4, f.shape
    for name in ALLMAPS:
        r = getattr(maps, name)(f)
        assert isinstance(r, Derived) and r.value.shape == (H, W), name
    for name in ("diffusivity", "settling", "chemical_potential"):
        r = getattr(maps, name)(f)
        assert isinstance(r, Derived) and r.value.shape == (S, H, W), name
    y = maps.yield_stress(f)
    assert isinstance(y, Bounded) and y.lo.shape == (H, W) and np.all(y.lo <= y.hi)
    s = maps.surface(f, small_bands)
    assert np.shape(s.index.value) == (3, H, W) and s.roughness.value.shape == (H, W)
    ex = maps.evaporation(f, maps.Atmosphere(activity={"water": 0.5}))
    assert set(ex) == {"water"} and ex["water"].rate.value.shape == (H, W)


def test_water_viscosity_is_about_one_mPa_s():
    eta = maps.viscosity(film({L.WATER: 50 * um}))
    assert np.allclose(eta.value, 1.0016 * mPa_s) and eta.everywhere_in_domain and eta.evidence == "checked"
    warm = Film.uniform({L.WATER: 50 * um}); warm.state["temperature"] = np.full((1, 1), celsius(25.0))
    assert np.allclose(maps.viscosity(warm).value, 0.890022 * mPa_s, rtol=1e-12)


def test_dilute_limit_of_viscosity_is_einsteins():
    for phi in (1e-4, 1e-5):
        f = film({L.WATER: (1 - phi) * 100 * um, L.RUTILE: phi * 100 * um})
        r = maps.viscosity(f).value[0, 0] / (1.0016 * mPa_s) - 1
        assert r == pytest.approx(2.5 * phi, rel=10 * phi)


def test_phi_to_zero_limits_are_finite_and_continuous(small_bands):
    base = {L.WATER: 50 * um}
    for name in ("viscosity", "packing", "diffusivity", "settling", "density", "surface_tension"):
        v0 = getattr(maps, name)(film({**base, L.HEMATITE: 0.0})).value
        v1 = getattr(maps, name)(film({**base, L.HEMATITE: 1e-12 * um})).value
        assert np.all(np.isfinite(v0)) and np.all(np.isfinite(v1)), name
        if name != "packing":                       # packing jumps from the empty convention to the population's own
            assert np.allclose(v0, v1, rtol=1e-6, atol=1e-30), name
    y0 = maps.yield_stress(film({**base, L.HEMATITE: 0.0}))
    y1 = maps.yield_stress(film({**base, L.HEMATITE: 1e-12 * um}))
    assert np.all(y0.hi == 0) and np.all(y1.hi == 0)


def test_jammed_and_dry_cells_are_out_of_domain():
    jam = maps.viscosity(film({L.WATER: 1 * um, L.RUTILE: 10 * um}))
    assert np.all(np.isinf(jam.value)) and not np.any(jam.in_domain) and jam.evidence == "assumed"
    q = np.zeros((2, 1, 2)); q[1, 0, 0] = 5 * um                  # cell 0: powder; cell 1: empty
    dry = maps.viscosity(Film((L.WATER, L.RUTILE), q))
    assert np.all(np.isinf(dry.value)) and list(np.ravel(dry.in_domain)) == [False, True]


def test_density_volumes_add():
    f = film({L.WATER: 30 * um, L.RUTILE: 10 * um})
    rho = maps.density(f)
    assert np.allclose(rho.value, (998.207 * 30 + 4230 * 10) / 40) and rho.evidence == "assumed"
    assert np.allclose(rho.value * f.h * 1e-6, f.mass(1e-3) .sum() / 6)     # mass per area, six cells of 1 mm


def test_packing_by_desmond_and_weeks():
    mono = particle("m", sg=1.0)
    assert np.allclose(maps.packing(film({L.WATER: 10 * um, mono: 1 * um})).value, 0.634)
    s = np.log(1.5); delta = np.sqrt(np.expm1(s * s)); skew = (np.exp(s * s) + 2) * delta
    poly = particle("p", sg=1.5)
    want = 0.634 + 0.0658 * delta + 0.0398 * skew * delta ** 2
    assert np.allclose(maps.packing(film({L.WATER: 10 * um, poly: 1 * um})).value, want, rtol=1e-12)
    # a mixture of two monodisperse sizes packs denser than either
    a, b = particle("a", d=0.2 * um), particle("b", d=1.0 * um)
    assert np.all(maps.packing(film({L.WATER: 10 * um, a: 1 * um, b: 1 * um})).value > 0.634)


def test_diffusivity_is_stokes_einstein():
    p = particle("p", d=0.2 * um)
    f = film({L.WATER: 100 * um, p: 1e-6 * um})
    eta = maps.viscosity(f).value[0, 0]
    D = maps.diffusivity(f)
    assert D.value[1, 0, 0] == pytest.approx(k_B * T_room / (3 * np.pi * eta * 0.2 * um), rel=1e-12)
    # water's own self-diffusion by molecular Stokes-Einstein: of order 1e-9 m^2/s, and flagged
    assert 2e-10 < D.value[0, 0, 0] < 5e-9 and not np.all(D.in_domain[0])


def test_settling_is_stokes():
    p = particle("p", d=1.0 * um, rho=4000.0)
    f = film({L.WATER: 100 * um, p: 1e-6 * um})
    v = maps.settling(f)
    want = (4000.0 - 998.207) * g_0 * (1 * um) ** 2 / (18 * 1.0016 * mPa_s)
    assert v.value[1, 0, 0] == pytest.approx(want, rel=1e-8) and v.value[0, 0, 0] == 0.0
    buoyant = particle("b", d=1.0 * um, rho=500.0)
    assert maps.settling(film({L.WATER: 100 * um, buoyant: 1e-6 * um})).value[1, 0, 0] < 0       # creaming


def test_chemical_potential_limits_and_identity():
    w = maps.chemical_potential(film({L.WATER: 10 * um}))
    assert np.allclose(w.value, 0.0)                                       # the pure substance
    a, b = liquid("a", M=0.05, rho=1000.0), liquid("b", M=0.2, rho=1000.0)
    f = film({a: 30 * um, b: 10 * um})
    mu = maps.chemical_potential(f).value[:, 0, 0]
    va, vb = a.molecular_volume, b.molecular_volume
    kT = k_B * T_room
    want_a = kT / va * (np.log(0.75) + 1 - va * (0.75 / va + 0.25 / vb))
    assert mu[0] == pytest.approx(want_a, rel=1e-12)
    # Euler: f = sum phi mu (h f is homogeneous of degree one in q); an interacting mixture
    c, d = liquid("c", M=0.05, cohesion=4e8), liquid("d", M=0.1, cohesion=1e8)
    g = film({c: 20 * um, d: 20 * um})
    mu = maps.chemical_potential(g).value[:, 0, 0]
    vc, vd = c.molecular_volume, d.molecular_volume
    Bcd = (np.sqrt(4e8) - np.sqrt(1e8)) ** 2
    fmix = kT * 0.5 * np.log(0.5) * (1 / vc + 1 / vd) + Bcd * 0.25
    assert 0.5 * (mu[0] + mu[1]) == pytest.approx(fmix, rel=1e-10)
    # absent: -inf, and exp(mu v / kT) = 0
    e = maps.chemical_potential(film({L.WATER: 10 * um, L.ETHANOL: 0.0}))
    assert np.all(np.isneginf(e.value[1])) and np.all(np.isfinite(e.value[0]))


def test_surface_tension():
    assert np.allclose(maps.surface_tension(film({L.WATER: 10 * um})).value, 72.74e-3)
    m = maps.surface_tension(film({L.WATER: 10 * um, L.ETHANOL: 10 * um}))
    assert np.allclose(m.value, 0.5 * (72.74e-3 + 22.39e-3)) and m.evidence == "assumed"


def test_exchange_of_water():
    f = film({L.WATER: 10 * um})
    env = maps.Atmosphere(activity={"water": 0.5}, boundary_layer=5e-3)
    e = maps.evaporation(f, env)["water"]
    assert np.allclose(e.activity.value, 1.0) and e.activity_env == 0.5
    want = 2.42e-5 * 18.0153e-3 * 2339.3 / (R_gas * T_room * 5e-3)
    assert np.allclose(e.rate.value, want, rtol=1e-12)
    # recession of a pure water surface at 50 % RH: of order 0.1 um/s
    assert 2e-8 < want * 0.5 / 998.207 < 3e-7
    warm = Film.uniform({L.WATER: 10 * um}); warm.state["temperature"] = np.full((1, 1), celsius(25.0))
    assert maps.evaporation(warm, env)["water"].p_sat.value[0, 0] == pytest.approx(3.17 * kPa, rel=0.01)   # vs 3.169


@pytest.mark.xfail(reason="Hildebrand's regular solution has no hydrogen bonding: it predicts water and ethanol "
                          "separate (B = 4.8e8 J/m^3 against a critical 1.6e8); they mix in every proportion",
                   strict=True)
def test_water_and_ethanol_mix():
    e = maps.evaporation(film({L.WATER: 10 * um, L.ETHANOL: 10 * um}), maps.Atmosphere())
    assert np.all(e["ethanol"].activity.value <= 1.0) and np.all(e["water"].activity.value <= 1.0)


def test_yield_stress_bounds_and_limits():
    tau = lambda q: maps.yield_stress(film({L.WATER: 10 * um, L.HEMATITE: q}))
    below = tau(1 * um)                                                    # phi = 0.09 < phi_0
    assert np.all(below.hi == 0)
    mid, more = tau(5 * um), tau(8 * um)                                   # phi = 0.33, 0.44
    assert np.all(mid.lo > 0) and np.all(mid.hi > mid.lo) and np.all(more.lo > mid.lo)
    assert 1.0 < mid.lo[0, 0] < 1e5                                        # Pa: a flocculated slurry
    assert np.all(np.isinf(tau(30 * um).lo))                               # jammed
    # an index-matched particle has no dispersion attraction: only the zero-frequency bound remains
    m = particle("m", n=float(np.sqrt(maps._host_eps(maps._F.of(film({L.WATER: 1 * um})), maps._OneBand())[0][0, 0, 0])))
    m = dataclasses.replace(m, rheo=dataclasses.replace(m.rheo, separation=1e-9, phi_percolation=0.1),
                            prov=dict(m.prov) | {"rheo.separation": m.provenance("eps"),
                                                 "rheo.phi_percolation": m.provenance("eps")})
    t = maps.yield_stress(film({L.WATER: 10 * um, m: 6 * um}))
    assert np.allclose(t.lo, 0.0, atol=1e-12) and np.all(t.hi > 0)
    assert np.all(maps.yield_stress_domain(film({L.WATER: 10 * um, L.HEMATITE: 5 * um})))


def test_surface_index_and_roughness(small_bands):
    over = maps.surface(film({L.WATER: 50 * um, L.CALCITE: 10 * um}), small_bands)
    n_w = maps._liquid_bands(L.WATER, small_bands)[0]
    assert np.allclose(over.index.value, n_w[:, None, None]) and np.all(over.roughness.value == 0)
    under = maps.surface(film({L.WATER: 0.1 * um, L.CALCITE: 10 * um}), small_bands)
    assert isinstance(under.index, Bounded) and np.all(under.index.lo >= 1.0) and np.all(under.index.width > 0)
    d32 = L.CALCITE.morph.size.sauter()
    assert np.all(under.roughness.value > 0.05 * d32) and np.all(under.roughness.value < 0.5 * d32)
    assert under.roughness.evidence == "assumed"


def test_reactions_cure_with_oxygen():
    f = film({L.LINSEED: 10 * um})
    r = maps.reactions(f, maps.Atmosphere(activity={"oxygen": 1.0}))
    assert len(r) == 1 and np.allclose(r[0].rate.value, 5.8e-6) and not r[0].rate.everywhere_in_domain   # linoxyn absent
    f2 = film({L.LINSEED: 10 * um, L.LINOXYN: 0.0})
    assert np.all(maps.reactions(f2, maps.Atmosphere(activity={"oxygen": 1.0}))[0].rate.in_domain)
    assert np.all(maps.reactions(f2, maps.Atmosphere())[0].rate.value == 0)                              # no oxygen


def test_evidence_propagates():
    assert maps.viscosity(film({L.WATER: 10 * um})).evidence == "checked"
    assert maps.viscosity(film({L.WATER: 10 * um, L.LINSEED: 0.0})).evidence == "checked"     # absent: not read
    assert maps.viscosity(film({L.LINSEED: 10 * um})).evidence == "guessed"
    assert maps.surface_tension(film({L.WATER: 10 * um})).evidence == "checked"


def test_no_map_reads_a_name_or_a_position(small_bands):
    a, b, c = L.WATER, L.GUM_ARABIC, L.HEMATITE
    rng = np.random.default_rng(3)
    q = rng.uniform(0, 10 * um, (3, 2, 2))
    f = Film((a, b, c), q)
    perm = [2, 0, 1]
    g = Film(tuple(f.substances[i] for i in perm), q[perm])
    ren = Film(tuple(dataclasses.replace(s, name=f"x{i}") for i, s in enumerate(f.substances)), q)
    for name in ("density", "viscosity", "packing", "surface_tension"):
        v = getattr(maps, name)(f).value
        assert np.allclose(getattr(maps, name)(g).value, v) and np.array_equal(getattr(maps, name)(ren).value, v), name
    for name in ("diffusivity", "settling", "chemical_potential"):
        v = getattr(maps, name)(f).value
        assert np.allclose(getattr(maps, name)(g).value, v[perm]) and np.array_equal(getattr(maps, name)(ren).value, v)
    o, og, orn = maps.optical(f, small_bands), maps.optical(g, small_bands), maps.optical(ren, small_bands)
    val = lambda d: d.lo if isinstance(d, Bounded) else d.value
    assert np.allclose(val(og.S), val(o.S)) and np.array_equal(val(orn.K), val(o.K))


def test_maps_are_fast_on_256_squared():
    import time
    rng = np.random.default_rng(0)
    q = np.stack([40 * um * rng.uniform(0.5, 1, (256, 256)), 5 * um * rng.uniform(0, 1, (256, 256)),
                  2 * um * rng.uniform(0, 1, (256, 256))])
    f = Film((L.WATER, L.GUM_ARABIC, L.HEMATITE), q)
    for name in ("density", "viscosity", "packing", "diffusivity", "settling", "chemical_potential",
                 "surface_tension", "yield_stress"):
        t = time.perf_counter(); getattr(maps, name)(f); dt = time.perf_counter() - t
        assert dt < 0.5, name
