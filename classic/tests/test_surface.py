"""Surface.from_canvas (SPEC §1, Optics): the one presentation of the World to sensors. Pure; finish from the top film."""
import pickle
import numpy as np
import pytest
from goo.classic.engine import Canvas
from goo.classic.surface import Surface, film_gloss, roughness_from_gloss
from goo.classic.sensors.lidar import CELL

BOX = (slice(10, 20), slice(12, 28))           # where the paint goes
INNER = (slice(13, 17), slice(15, 25))         # away from its edges


def oil_patch(cv, vol=0.6, oil=0.5):
    m = np.zeros((cv.H, cv.W)); m[BOX] = 1
    cv.add("oil", m * vol * oil); cv.add("white", m * vol * (1 - oil))


def cure_and_fix(cv, ticks=300):
    cv.L["cure_rate"] = 0.002; cv.wait(ticks, cure_boost=12.0); cv.L["cure_rate"] = 0.0
    return cv.fix()


def dry_and_fix(cv):
    cv.wait(100, evap_boost=8.0)
    while cv.q[0].max() > 1e-6: cv.wait(40, evap_boost=8.0)
    return cv.fix()


def canvas(): return Canvas(40, 30, ["white", "ultramarine"], seed=3)


def test_wet_oil_is_glossy_and_stays_glossy_when_cured_and_fixed():
    cv = canvas(); oil_patch(cv)
    wet = Surface.from_canvas(cv)
    assert wet.gloss[INNER].min() > 0.99                      # 0.6 units, half oil: a full skin
    assert wet.gloss[0:5].max() == 0 and wet.gloss[:, 0:5].max() == 0   # bare ground is matte
    assert cure_and_fix(cv) > 0.99
    assert len(cv.strata) == 1 and cv.q[:, INNER[0], INNER[1]].sum() == 0   # the paint is now a stratum, wet layer empty
    cured = Surface.from_canvas(cv)
    assert cured.gloss[INNER].min() > 0.99                   # the cured film keeps its gloss (it used to be 0)
    assert np.allclose(cured.rough_um[INNER], 0.3) and np.allclose(cured.rough_um[0:5], 1.5)


def test_lean_paint_and_dried_watercolour_are_matte():
    cv = canvas(); m = np.zeros((30, 40)); m[BOX] = 1
    cv.add("water", m * 0.4); cv.add("ultramarine", m * 0.04)
    assert Surface.from_canvas(cv).gloss.max() == 0            # wet watercolour: no oil
    dry_and_fix(cv)
    assert len(cv.strata) == 1 and cv.strata[0].sum() > 0
    s = Surface.from_canvas(cv)
    assert s.gloss.max() == 0 and np.all(s.rough_um == 1.5)
    cv2 = canvas(); oil_patch(cv2, vol=0.6, oil=0.03)           # lean: 3% oil
    assert Surface.from_canvas(cv2).gloss[INNER].max() == pytest.approx(0.1)


def test_the_top_film_decides():
    cv = canvas(); oil_patch(cv); cure_and_fix(cv)
    m = np.zeros((30, 40)); m[:, :20] = 1                      # a watercolour wash over the left half
    cv.add("water", m * 0.4); cv.add("ultramarine", m * 0.04)
    s = Surface.from_canvas(cv)
    assert s.gloss[INNER[0], 15:18].max() == 0                 # wet wash on top: matte
    assert s.gloss[INNER[0], 23:25].min() > 0.99               # the cured oil beside it still glossy
    dry_and_fix(cv)
    s = Surface.from_canvas(cv)
    assert len(cv.strata) == 2
    assert s.gloss[INNER[0], 15:18].max() == 0                 # the dried wash is the topmost stratum there
    assert s.gloss[INNER[0], 23:25].min() > 0.99


def test_from_canvas_changes_nothing():
    cv = canvas(); oil_patch(cv); cure_and_fix(cv); oil_patch(cv, vol=0.2)
    cv.step()                                                  # leave slow-law ticks pending: no flush may happen
    before = pickle.dumps(cv.__dict__)
    s = Surface.from_canvas(cv)
    assert pickle.dumps(cv.__dict__) == before
    assert np.array_equal(s.Z_mm, cv.surface() * CELL) and np.array_equal(s.reflectance, cv.reflectance())
    with pytest.raises(ValueError): s.Z_mm[0, 0] = 1.0         # and what it hands out is read-only


def test_rule_and_compatibility():
    f = np.zeros((3, 1, 4)); f[1] = [0, 0.075, 0.3, 0.03]; f[2] = [0, 0.075, 0.7, 0.27]   # (species, H, W): water, oil, pigment
    assert np.allclose(film_gloss(f, 1)[0], [0, 0.5 * 1.0, 1.0 * 1.0, 1.0 * (0.1 / 0.3)])
    assert roughness_from_gloss(np.array(0.0)) == 1.5 and roughness_from_gloss(np.array(1.0)) == pytest.approx(0.3)
    from goo.classic.sensors.coherent import Surface as S2
    assert S2 is Surface
    old = Surface(np.zeros((4, 5)), np.full((4, 5), 0.9), np.full((4, 5), 1.5))
    assert old.nir.shape == (4, 5) and np.isnan(old.rgb).all() and old.gloss.max() == 0 and old.rough[0, 0] == 1.5
