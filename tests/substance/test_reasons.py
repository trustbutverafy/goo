"""Every out-of-domain flag says why: one test of each kind of reason, and the reasons account for every flagged
cell."""
import dataclasses
import numpy as np
import pytest
from goo.evidence import Bounded
from goo.film import Film, Bands
from goo.substance import maps, library as L, Shape, LogNormal, Substance, Thermo, Rheo, Oscillators
from goo.units import um, nm, mPa_s, celsius
from conftest import liquid, particle, G, NONE

B = Bands.uniform()


def f(amounts, T=None):
    film = Film.uniform(amounts)
    if T is not None:
        film.state["temperature"] = np.full((1, 1), T)
    return film


def says(d, phrase):
    assert phrase in d.map, d.map
    if not isinstance(d, Bounded):
        assert not np.all(d.in_domain)


def test_source_range_exceeded():
    K = maps.optical(f({L.LINSEED: 98 * um, L.RUTILE: 2 * um}), B).K
    says(K, "eps of rutile not described across bands 390, 410, 430 nm (described 430-1530 nm)")
    d = np.asarray(K.in_domain)[:, 0, 0]
    assert not d[:3].any() and d[3:].all()                    # only those bands (420-440 nm reaches below 430)


def test_independent_scattering_bound():
    says(maps.optical(f({L.LINSEED: 30 * um, L.HEMATITE: 20 * um}), B).S, "independent scattering")


def test_host_outside_the_table():
    says(maps.optical(f({liquid("dense", n=1.9): 90 * um, L.HEMATITE: 1 * um}), B).S, "host index outside")


def test_particle_shape_without_a_law():
    plate = Substance("mica", Oscillators(2.4), Shape("platelet", LogNormal(10 * um)), 2800.0,
                      prov=dict(eps=G, morph=G, rho=G, reservoir=NONE))
    says(maps.optical(f({L.WATER: 90 * um, plate: 1 * um}), B).S, "of shape 'shape'")
    says(maps.viscosity(f({L.WATER: 90 * um, plate: 1 * um})), "of shape 'shape'")


def test_immiscible_liquids():
    says(maps.viscosity(f({L.WATER: 10 * um, L.LINSEED: 10 * um})), "water and linseed_oil separate")


def test_jammed_powder_and_temperature():
    says(maps.viscosity(f({L.WATER: 1 * um, L.RUTILE: 10 * um})), "jammed")
    says(maps.viscosity(f({L.RUTILE: 10 * um})), "a powder does not flow")
    says(maps.viscosity(f({L.LINOXYN: 10 * um}, T=celsius(30))), "linoxyn has no E_eta")


def test_polydispersity_beyond_the_packing_fit():
    wide = particle("wide", sg=3.0)
    says(maps.packing(f({L.WATER: 10 * um, wide: 1 * um})), "polydispersity above 0.6")


def test_reynolds_number_and_molecules():
    big = particle("boulder", d=200 * um, rho=8000.0)
    says(maps.settling(f({L.WATER: 900 * um, big: 1 * um})), "Reynolds number of boulder")
    says(maps.diffusivity(f({L.WATER: 10 * um})), "Stokes-Einstein for the molecules of water")


def test_missing_parameters():
    says(maps.surface_tension(f({L.GUM_ARABIC: 10 * um})), "gum_arabic has no surface tension")
    vol = liquid("vol", p_sat=1000.0, M=0.1)                   # volatile without dH_vap or D_vapour
    e = maps.evaporation(f({vol: 10 * um}, T=celsius(30)), maps.Atmosphere())["vol"]
    says(e.p_sat, "no dH_vap"); says(e.rate, "no vapour diffusivity")


def test_reaction_product_absent():
    r = maps.reactions(f({L.LINSEED: 10 * um}), maps.Atmosphere(activity={"oxygen": 1.0}))[0].rate
    says(r, "its product linoxyn is not a substance of the film")


def test_yield_reasons_and_domain():
    y = maps.yield_stress(f({L.WATER: 0.5 * um, L.HEMATITE: 10 * um}))
    assert "voids in the film" in y.map or "jammed" in y.map
    assert not maps.yield_stress_domain(f({L.WATER: 0.5 * um, L.HEMATITE: 10 * um})).any()


def test_the_reasons_account_for_every_flag():
    rng = np.random.default_rng(5)
    q = rng.uniform(0, 10 * um, (3, 4, 4)); q[:, 0, 0] = 0
    film = Film((L.WATER, L.LINSEED, L.HEMATITE), q)
    for name in ("viscosity", "packing", "surface_tension"):
        d = getattr(maps, name)(film)
        flagged = int((~np.asarray(d.in_domain, bool)).sum())
        if flagged:
            counts = [int(s.split("[")[1].split(" of")[0]) for s in d.map.split("; out of domain: ")[1].split("; ")]
            assert max(counts) <= flagged <= sum(counts), (name, d.map)
        else:
            assert "out of domain" not in d.map
        assert np.asarray(d.in_domain, bool).reshape(-1)[0]      # the empty cell is in domain
