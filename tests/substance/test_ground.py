"""Grounds: the same optical physics as a film, on a porous solid whose pores hold matter and air; and the named-tuple
form of the multi-quantity maps."""
import dataclasses
import numpy as np
import pytest
from goo.evidence import Derived, Bounded
from goo.film import Film, Ground, Bands
from goo.substance import maps, library as L
from goo.units import um, nm
from conftest import liquid, particle


def ground(solid, held=None, substances=(), porosity=0.5, depth=100 * um, shape=(2, 2)):
    g = Ground(np.zeros(shape), solid, porosity, 2 * um, depth, substances)
    if held is not None:
        g.held = np.broadcast_to(np.asarray(held, float).reshape((-1, 1, 1)), (len(substances),) + shape).copy()
    return g


def value(x, end="hi"):
    return getattr(x, end) if isinstance(x, Bounded) else x.value


def R_over_black(S, depth):
    """Two-flux reflectance of a non-absorbing layer over black: S X / (1 + S X)."""
    return S * depth / (1 + S * depth)


def test_maps_unpack_by_position(small_bands):
    f = Film.uniform({L.LINSEED: 40 * um, L.RUTILE: 2 * um})
    K, S, n_host = maps.optical(f, small_bands)
    assert isinstance(maps.optical(f, small_bands), tuple) and K.value.shape == (3, 1, 1)
    n_top, sigma = maps.surface(f, small_bands)
    assert sigma.value.shape == (1, 1)
    K, S, n_host = maps.ground_optical(ground(L.CELLULOSE), small_bands)
    n_top, sigma = maps.ground_surface(ground(L.CELLULOSE), small_bands)
    a, a_env, rate, p_sat = maps.evaporation(Film.uniform({L.WATER: 1 * um}), maps.Atmosphere())["water"]


def test_a_soaked_cellulose_ground_reflects_less_than_a_dry_one(small_bands):
    dry = maps.ground_optical(ground(L.CELLULOSE), small_bands)
    wet = maps.ground_optical(ground(L.CELLULOSE, [50 * um], (L.WATER,)), small_bands)        # pores full
    half = maps.ground_optical(ground(L.CELLULOSE, [25 * um], (L.WATER,)), small_bands)
    assert isinstance(dry.host, Derived) and np.allclose(dry.host.value, 1.0)
    assert isinstance(wet.host, Derived) and np.allclose(wet.host.value, maps._liquid_bands(L.WATER, small_bands)[0][:, None, None])
    Rd, Rw = R_over_black(dry.S.value, 100 * um), R_over_black(wet.S.value, 100 * um)
    assert np.all(Rw < Rd) and np.all(wet.S.value < 0.5 * dry.S.value)
    assert isinstance(half.S, Bounded)                          # liquid and air: microstructure unknown
    assert np.all(half.S.lo >= wet.S.value * (1 - 1e-9)) and np.all(half.S.hi <= dry.S.value * (1 + 1e-9))
    assert dry.K.evidence == "assumed"                          # a ground is far above independent scattering


def test_pores_filled_with_the_solids_index_scatter_nothing(small_bands):
    solid, fill = particle("grain", n=1.5, d=2 * um, sg=1.3), liquid("match", n=1.5)
    full = maps.ground_optical(ground(solid, [50 * um], (fill,)), small_bands)
    dry = maps.ground_optical(ground(solid), small_bands)
    assert np.all(full.S.value <= 1e-9 * dry.S.value.max()) and np.all(dry.S.value > 0)


def test_grounds_differ_only_through_their_descriptions(small_bands):
    paper = dataclasses.replace(L.CELLULOSE, name="paper")
    a = maps.ground_optical(ground(L.CELLULOSE, [10 * um], (L.WATER,)), small_bands)
    b = maps.ground_optical(ground(paper, [10 * um], (L.WATER,)), small_bands)
    c = maps.ground_optical(ground(L.CALCITE, [10 * um], (L.WATER,)), small_bands)
    assert np.array_equal(a.S.lo, b.S.lo) and np.array_equal(a.S.hi, b.S.hi)
    assert not np.allclose(a.S.lo, c.S.lo)


def test_ground_surface(small_bands):
    dry = maps.ground_surface(ground(L.CALCITE), small_bands)
    full = maps.ground_surface(ground(L.CALCITE, [50 * um], (L.WATER,)), small_bands)
    d32 = L.CALCITE.morph.size.sauter()
    assert np.all(dry.roughness.value > 0.05 * d32) and np.all(full.roughness.value == 0)
    assert isinstance(full.index, Bounded) and np.all(full.index.lo > 1.3)
    assert np.all(full.index.hi <= np.sqrt(np.real(L.CALCITE.eps.eps(np.array([550e-9])))).max() + 1e-9)


def test_overfull_pores_and_a_solid_without_grains_are_named(small_bands):
    o = maps.ground_optical(ground(L.CALCITE, [80 * um], (L.WATER,)), small_bands)
    assert "its pores hold more than their volume" in o.K.map
    g = maps.ground_optical(ground(L.PMMA), small_bands)
    assert "has no morphology" in g.S.map
