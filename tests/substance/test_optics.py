"""The optical response: Mie against miepython (dev-only check), the rutile anchor, exact limits, the precomputed
table's interpolation error, and the `optical` map's shapes, units, evidence and domain."""
import numpy as np
import pytest
from goo.film import Film, Bands
from goo.substance import mie, response as R, library as L, maps, Sphere, Coated, Hollow, LogNormal, Oscillators
from goo.units import um, nm
from conftest import liquid, particle


def test_mie_against_miepython():
    miepython = pytest.importorskip("miepython")
    rng = np.random.default_rng(1)
    m = rng.uniform(1.05, 3.0, 150) + 1j * rng.uniform(0, 1.2, 150)
    x = np.concatenate([rng.uniform(0.01, 2, 50), rng.uniform(2, 100, 100)])
    qe, qs, qa, g = mie.efficiencies(m, x)
    for i in range(m.size):
        e, s, _, gg = miepython.efficiencies_mx(complex(m[i].real, -m[i].imag), x[i])   # miepython: k < 0 absorbs
        assert qe[i] == pytest.approx(e, rel=1e-7) and qs[i] == pytest.approx(s, rel=1e-7)
        assert g[i] == pytest.approx(gg, rel=1e-6, abs=1e-9)


def test_mie_small_particle_limit_is_rayleigh():
    m = np.array([1.5 + 0.1j, 2.0 + 0.0j]); x = np.array([1e-3, 1e-3])
    qe, qs, qa, g = mie.efficiencies(m, x)
    rs, ra = mie.rayleigh(m, x)
    assert np.allclose(qs, rs, rtol=1e-4) and np.allclose(qa[0], ra[0], rtol=1e-4)


def test_coated_and_hollow_reduce_to_spheres():
    m, x = 1.8 + 0.05j, 3.0
    s = mie.efficiencies(np.array([m]), np.array([x]))
    c = mie.coated_efficiencies(np.array([m]), np.array([m]), np.array([0.6 * x]), np.array([x]))   # core = shell
    assert np.allclose(np.ravel(s), np.ravel(c), rtol=1e-6)
    c0 = mie.coated_efficiencies(np.array([1.0 + 0j]), np.array([m]), np.array([0.0]), np.array([x]))  # no core
    assert np.allclose(np.ravel(s), np.ravel(c0), rtol=1e-9)


def test_rutile_scatters_most_near_0p2_um_at_order_10_per_um():
    """Anchor (classic/docs/OPTICS-DATA.md check 2): monodisperse rutile in n = 1.5 at 550 nm, mu_s (1 - g) per unit
    volume peaks near d = 0.2 um at ~12.5 /um (the coatings literature: ~12 /um at 0.22 um)."""
    ds = np.linspace(0.1, 0.4, 31) * um
    v = [(lambda a: a[1][0, 0] * (1 - a[2][0, 0]))(R.cross_sections(L.RUTILE.eps, Sphere(LogNormal(d)), [550 * nm], [1.5]))
         for d in ds]
    i = int(np.argmax(v))
    assert 0.17 * um <= ds[i] <= 0.23 * um
    assert 5 / um < v[i] < 20 / um


def test_a_particle_in_a_host_of_its_own_index_scatters_nothing(small_bands):
    host = liquid("host", n=1.5)
    p = particle("p", n=1.5, d=0.4 * um, sg=1.3)
    f = Film.uniform({host: 30 * um, p: 3 * um}, 2, 2)
    o = maps.optical_exact(f, small_bands)
    assert np.all(o.S.value <= 1e-18) and np.all(o.K.value <= 1e-18)          # 1/m: zero to rounding
    off = maps.optical_exact(Film.uniform({liquid("h2", n=1.33): 30 * um, p: 3 * um}), small_bands)
    assert np.all(off.S.value > 1e3)                                     # the same particles in water do scatter
    t = maps.optical(f, small_bands)                                     # 1.5 is a node of the table's grid
    assert np.all(t.S.value <= 1e-9 * off.S.value.max())


def test_no_particles_gives_S_zero_and_K_from_the_liquid(small_bands):
    f = Film.uniform({L.WATER: 50 * um}, 3, 4)
    o = maps.optical(f, small_bands)
    assert np.all(o.S.value == 0.0)
    lam = small_bands.centres
    n, k, _, _ = maps._liquid_bands(L.WATER, small_bands)
    assert np.allclose(o.K.value[:, 0, 0], 2 * 4 * np.pi * k / lam, rtol=1e-12)
    assert np.allclose(o.host.value[:, 0, 0], n, rtol=1e-12)


def test_optical_shapes_units_and_linearity(bands):
    f = Film.uniform({L.LINSEED: 40 * um, L.RUTILE: 1 * um, L.SOOT: 0.2 * um}, 3, 5)
    o = maps.optical(f, bands)
    assert o.K.value.shape == o.S.value.shape == o.host.value.shape == (bands.B, 3, 5)
    assert np.all(np.asarray(o.K.in_domain).shape == (bands.B, 3, 5))
    # per unit volume of rutile in oil at 550 nm: S of order 1e6-1e7 /m (several per um)
    f1 = Film.uniform({L.LINSEED: 40 * um, L.RUTILE: 1 * um})
    f2 = Film.uniform({L.LINSEED: 40 * um, L.RUTILE: 2 * um})
    o1, o2 = maps.optical(f1, bands), maps.optical(f2, bands)
    b550 = int(np.argmin(np.abs(bands.centres - 550 * nm)))
    S_per_volume = o1.S.value[b550, 0, 0] * f1.h[0, 0] / (1 * um)
    assert 3e6 < S_per_volume < 1.5e7
    # optical depth is linear in the particles' amount (independent scattering): S h doubles with q
    assert np.allclose(o2.S.value * f2.h, 2 * o1.S.value * f1.h, rtol=1e-12)


def test_optical_evidence_and_domain(bands):
    lean = Film.uniform({L.LINSEED: 100 * um, L.HEMATITE: 2 * um})       # dilute: in domain in hematite's range
    o = maps.optical(lean, bands)
    dom = np.asarray(o.K.in_domain)
    assert dom.all() and o.K.evidence == "assumed"                          # linseed's eps is assumed
    rich = Film.uniform({L.LINSEED: 30 * um, L.HEMATITE: 20 * um})        # 40 % pigment: dependent scattering
    assert not np.asarray(maps.optical(rich, bands).K.in_domain).any()
    w = Film.uniform({L.WATER: 100 * um, L.SOOT: 1 * um})                   # soot is described from 436 nm
    d = np.asarray(maps.optical(w, bands).S.in_domain)[:, 0, 0]
    assert not d[bands.centres < 436 * nm].any() and d[(bands.centres > 450 * nm) & (bands.centres < 1500 * nm)].all()


def test_voids_give_bounds(bands):
    """Too little binder to fill the packed particles' interstices: the host is liquid and air in an unknown
    microstructure, and host, K and S come as Bounded."""
    f = Film.uniform({L.WATER: 1 * um, L.CALCITE: 20 * um})
    o = maps.optical(f, bands)
    assert type(o.host).__name__ == "Bounded" and np.all(o.host.lo >= 1.0 - 1e-12)
    assert np.all(o.host.hi <= maps._liquid_bands(L.WATER, bands)[0][:, None, None] + 1e-12)
    assert np.all(o.host.width > 0) and np.all(o.S.width >= 0) and o.S.evidence == "assumed"
    # dry: no liquid at all, the host is air exactly
    dry = maps.optical(Film.uniform({L.CALCITE: 20 * um}), bands)
    assert np.allclose(dry.host.value, 1.0)


def test_empty_cells(small_bands):
    q = np.zeros((2, 2, 2)); q[:, 0, 0] = [10 * um, 1 * um]
    f = Film((L.WATER, L.RUTILE), q)
    o = maps.optical(f, small_bands)
    assert np.all(o.K.value[:, 1, 1] == 0) and np.all(o.S.value[:, 1, 1] == 0) and np.all(o.host.value[:, 1, 1] == 1)


# the table's error against direct Mie at the mid-points of its host-index grid, as a fraction of the band's largest
# S (and K against the larger of the two), stated in docs/SUBSTANCES.md
STATED = {"rutile": 5e-3, "soot": 1e-6, "hematite": 1e-5, "cadmium_sulfide": 2e-3, "pmma_latex": 1e-5,
          "ochre": 1e-3, "ultramarine": 5e-3, "rose": 1e-5, "green": 1e-4, "dioxazine": 1e-5, "gain_green": 1e-5,
          "calcite": 3e-2, "cellulose": 5e-2}


@pytest.mark.parametrize("name", sorted(STATED))
def test_interpolation_error_of_the_precomputed_optics(name, bands):
    s = L.get(name)
    T = R.table(s, bands)
    mid = 0.5 * (R.HOST_GRID[:-1] + R.HOST_GRID[1:])
    lam, own = R.band_samples(bands)
    Ke, Se = R.two_flux(s.eps, s.morph, lam, mid)
    Ke, Se = R.band_mean(Ke, own, bands.B), R.band_mean(Se, own, bands.B)
    if T.passive:
        Ke = np.maximum(Ke, 0)
    Ki, Si, inside = T(np.repeat(mid[None], bands.B, 0))
    assert inside.all()
    scale = np.maximum(np.abs(Se).max(0), np.abs(Ke).max(0))
    assert np.max(np.abs(Si.T - Se) / np.abs(Se).max(0)) <= STATED[name]
    assert np.max(np.abs(Ki.T - Ke) / scale) <= STATED[name]


def test_optical_is_fast_enough_for_a_time_step(bands):
    """256 x 256 cells, 19 bands, three substances: under a second once the tables exist (typically ~0.2 s)."""
    import time
    rng = np.random.default_rng(0)
    q = np.stack([40 * um * rng.uniform(0.5, 1, (256, 256)), 2 * um * rng.uniform(0, 1, (256, 256)),
                  0.5 * um * rng.uniform(0, 1, (256, 256))])
    f = Film((L.LINSEED, L.RUTILE, L.HEMATITE), q)
    maps.optical(f, bands)
    t = time.perf_counter(); maps.optical(f, bands); dt = time.perf_counter() - t
    assert dt < 1.0


def test_impossible_substances_still_answer(small_bands):
    """Gain gives negative K in its band (the light ledger's draw on the reservoir); an acausal eps still has optics
    (its flag travels with its classification, not with these numbers)."""
    o = maps.optical(Film.uniform({L.LINSEED: 100 * um, L.GAIN_GREEN: 2 * um}), small_bands)
    K = o.K.value[:, 0, 0]
    assert K[1] < 0 and K[0] > K[1]                                         # 550 nm band
    a = maps.optical(Film.uniform({L.LINSEED: 100 * um, L.ACAUSAL_BLUE: 2 * um}), small_bands)
    assert np.all(np.isfinite(a.K.value)) and np.all(a.K.value >= 0)
