"""goo.lab.dry: the dry film is the composition in equilibrium with the atmosphere (docs/LAB.md §2). Owner: lab."""
from dataclasses import replace
import warnings
import numpy as np, pytest
from goo.units import um
from goo.film import Film, Environment
from goo.substance import maps, library as lib
from goo.lab.dry import dry
from goo.lab import recipes as R

ROOM = Environment(activity={"water": 0.5, "oxygen": 0.21})


def wc(h=50 * um, cells=((0.1, 0.1),)):
    """A watercolour row: (pigment, gum) volume fractions per cell, the rest water."""
    q = np.array([[[h * p for p, g in cells]], [[h * g for p, g in cells]], [[h * (1 - p - g) for p, g in cells]]])
    return Film((lib.HEMATITE, lib.GUM_ARABIC, lib.WATER), q)


def test_nothing_volatile_and_nothing_reacting_dries_to_itself_exactly():
    f = Film.uniform({lib.PMMA: 20 * um, lib.SOOT: 1 * um})
    d = dry(f, ROOM)
    assert np.array_equal(d.film.q, f.q) and d.film.substances == f.substances


def test_every_volatile_ends_at_the_environments_activity():
    f = wc(cells=((0.1, 0.1), (0.2, 0.05), (0.02, 0.3)))
    for rh in (0.2, 0.5, 0.8):
        d = dry(f, ROOM.with_(activity={"water": rh}))
        a = maps.evaporation(d.film, ROOM.with_(activity={"water": rh}))["water"].activity.value
        assert np.allclose(a, rh, atol=1e-10)
        assert np.array_equal(d.film.q[:2], f.q[:2])            # non-volatile matter stays, bit for bit


def test_more_humid_air_leaves_more_water():
    f = wc()
    w = [dry(f, ROOM.with_(activity={"water": rh})).film.q[2].item() for rh in (0.1, 0.5, 0.9)]
    assert w[0] < w[1] < w[2]


def test_a_pure_volatile_in_unsaturated_air_leaves_entirely_and_with_no_reservoir_stays():
    f = Film.uniform({lib.WATER: 30 * um})
    assert dry(f, ROOM).film.h.item() < 1e-20
    assert dry(f, ROOM.with_(activity={"water": 1e-6})).film.h.item() < 1e-20
    kept = dry(f, ROOM.with_(activity={"oxygen": 0.21}))             # the air names no water: no reservoir to take it
    assert np.array_equal(kept.film.q, f.q)


def test_volatility_comes_from_the_description_never_the_name():
    renamed = replace(lib.WATER, name="not_water")
    q = wc().q
    a = dry(Film((lib.HEMATITE, lib.GUM_ARABIC, lib.WATER), q), ROOM)
    b = dry(Film((lib.HEMATITE, lib.GUM_ARABIC, renamed), q), Environment(activity={"not_water": 0.5}))
    assert np.array_equal(a.film.q, b.film.q)
    still = replace(lib.WATER, name="still_water", thermo=replace(lib.WATER.thermo, p_sat=None, dH_vap=None, D_vapour=None),
                    prov={k: v for k, v in lib.WATER.prov if k not in ("thermo.p_sat", "thermo.dH_vap", "thermo.D_vapour")})
    c = dry(Film((lib.HEMATITE, lib.GUM_ARABIC, still), q), ROOM)
    assert np.array_equal(c.film.q, q)


def test_reactions_run_to_completion_only_where_the_environment_feeds_them():
    f = Film.uniform({lib.LINSEED: 40 * um, lib.HEMATITE: 10 * um})
    d = dry(f, ROOM)
    names = [s.name for s in d.film.substances]
    assert d.film.q[names.index("linseed_oil")].item() == 0.0
    ox = lib.LINSEED.reactions[0].coefficient("oxygen")
    m_oil = lib.LINSEED.rho * 40 * um
    assert d.film.q[names.index("linoxyn")].item() * lib.LINOXYN.rho == pytest.approx(m_oil * (1 - ox), rel=1e-12)
    assert d.drawn["oxygen"].item() == pytest.approx(-ox * m_oil, rel=1e-12)
    no_air = dry(f, ROOM.with_(activity={"water": 0.5}))
    assert np.array_equal(no_air.film.q[:2], f.q)


def test_the_matter_ledger_of_drying_closes():
    for f in (wc(cells=((0.1, 0.1), (0.3, 0.02))), Film.uniform({lib.LINSEED: 40 * um, lib.HEMATITE: 10 * um})):
        d = dry(f, ROOM)
        assert d.receipt.residual["matter"] <= 1e-12


def test_the_dry_state_is_a_fixed_point():
    for f in (wc(), R.wet_film(R.Paint.of(lib.ULTRAMARINE), R.ACRYLIC), R.wet_film(R.Paint.of(lib.CDS), R.OIL)):
        once = dry(f, ROOM).film
        twice = dry(once, ROOM).film
        assert twice.substances == once.substances
        assert np.allclose(twice.q, once.q, rtol=1e-9, atol=1e-18)


def test_an_underbound_dry_film_holds_voids_the_maps_see_as_air():
    f = R.wet_film(R.Paint.of(lib.RUTILE), R.WATER)               # particles and water only
    d = dry(f, ROOM).film
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = maps.optical(d, R.LAB_BANDS)
    # every drop of water leaves (next test), so the dry film is powder in air: an exact host of index 1, with no
    # liquid left to make its microstructure unknown
    host = getattr(o.host, "value", None)
    assert host is not None and np.allclose(host, 1.0)


def test_particles_in_water_dry_to_a_powder_in_ordinary_air():
    d = dry(R.wet_film(R.Paint.of(lib.RUTILE), R.WATER), ROOM).film
    water = d.q[[s.name for s in d.substances].index("water")].item()
    assert water <= 1e-20 * d.h.item()                          # particles hold no water: only a binder does
