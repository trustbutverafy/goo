"""Integration across owners: substances plugged straight into light (SPEC §4.4). Owner: the lead."""
import warnings
import numpy as np, pytest
from goo.units import um, nm
from goo.film import Film, Ground, Stack, Bands, Light
from goo.substance import maps, library as lib
from goo.light import look, Optical

BANDS = Bands.uniform()
LIGHT = Light(BANDS, np.ones(BANDS.B))
Z = np.zeros((1, 1))


def paper(held=0.0):
    return Ground(Z, lib.CELLULOSE, 0.5, 5 * um, 200 * um, (lib.WATER,), np.full((1, 1, 1), held))


def gesso():
    return Ground(Z, lib.CALCITE, 0.3, 1 * um, 300 * um)


def see(ground, films):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return look(Stack(ground, films, dx=1e-3), BANDS, LIGHT, maps)


def test_the_substance_maps_are_lights_optical_protocol():
    assert isinstance(maps, Optical)


def test_a_look_at_real_substances_closes_and_says_what_it_assumed():
    L = see(gesso(), [Film.uniform({lib.LINSEED: 45 * um, lib.HEMATITE: 5 * um})])
    assert L.receipt.residual["light"] <= 1e-12
    total = L.reflectance + L.absorbed.sum(0) + L.ground + L.transmitted + L.unresolved
    assert np.allclose(total, 1.0, atol=1e-12)
    assert {f.kind for f in L.receipt.flags} <= {"bounds_collapsed", "out_of_domain"}


def test_wet_paper_is_darker_and_passes_more_light_from_descriptions_alone():
    dry, wet = see(paper(0.0), []), see(paper(100 * um), [])
    assert wet.reflectance.mean() < 0.7 * dry.reflectance.mean()
    assert wet.transmitted.mean() > dry.transmitted.mean()


def test_order_of_films_matters():
    body = Film.uniform({lib.LINSEED: 9.5 * um, lib.RUTILE: 0.5 * um})
    glaze = Film.uniform({lib.LINSEED: 45 * um, lib.HEMATITE: 5 * um})
    over, under = see(gesso(), [glaze, body]), see(gesso(), [body, glaze])
    assert over.reflectance.mean() > 3 * under.reflectance.mean()


def test_an_impossible_substance_is_flagged_and_its_debt_booked():
    L = see(gesso(), [Film.uniform({lib.LINSEED: 2.7 * um, lib.GAIN_GREEN: 0.3 * um})])
    assert L.reflectance.max() > 1.0 and L.debt.max() > 0.0
    assert any(f.kind == "impossible" for f in L.receipt.flags)
    assert L.receipt.residual["light"] <= 1e-12
