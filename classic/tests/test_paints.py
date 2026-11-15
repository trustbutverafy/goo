"""Paints and the palette (goo.classic.paints): recipes, preview against a real canvas, mix_for, the swatch card."""
import time
import numpy as np
import pytest
from goo.classic import paints as P
from goo.classic.engine import Canvas, PIGMENTS
from goo.classic.brush import Brush
from goo.classic.palette import lab, srgb_to_linear, delta_e2000


def dE(a_lin, b_lin): return float(delta_e2000(lab(np.asarray(a_lin)[:3]), lab(np.asarray(b_lin)[:3])))


def test_recipes_and_loads():
    w = P.watercolour({"ultramarine": 2, "rose": 2}, strength=0.05)
    assert w.mix == {"ultramarine": 0.5, "rose": 0.5} and not w.permanent and w.medium == "watercolour"
    assert w.parts["water"] == pytest.approx(0.95) and w.solids == pytest.approx(0.05)
    ld = w.load(1000.0)
    assert sum(ld.values()) == pytest.approx(1000.0) and ld["rose"] == pytest.approx(25.0)
    assert sum(w.load().values()) == pytest.approx(w.brushful)
    a = P.acrylic({"red": 1})
    assert a.permanent and a.parts[P.BINDER] == pytest.approx(0.35) and a.parts["water"] == pytest.approx(0.5)
    assert a.dry_thickness(0.3) == pytest.approx(0.15) and "acrylic" in a.pigments
    o = P.oil({"white": 1}, fat=0.4)
    assert o.permanent and o.parts["oil"] == pytest.approx(0.4) and o.dry_thickness(0.2) == pytest.approx(0.2)
    g = P.gouache({"ultramarine": 1})
    assert not g.permanent and g.parts["white"] > 0 and g.solids == pytest.approx(0.35)
    assert P.water().load(10) == {"water": 10.0} and P.water().pigments == []
    with pytest.raises(KeyError): P.oil({"unobtainium": 1})
    # a brush takes the load and the books stay closed
    cv = Canvas(30, 20, a.pigments); b = Brush(cv, 6, load=a.load(500))
    assert b.load.sum() == pytest.approx(500)


@pytest.mark.parametrize("paint", [P.watercolour({"ultramarine": 1}), P.gouache({"sienna": 1}),
                                   P.acrylic({"red": 0.6, "white": 0.4}), P.oil({"ochre": 0.7, "green": 0.3})])
def test_preview_is_the_canvas_optics_exactly(paint):
    """Laid uniformly with Canvas.add (no leveling, so it stays uniform): wet = the wet layer's reflectance;
    dried and fixed = the stratum's, to round-off. Over a coloured underlayer too."""
    t = 0.3
    cv = Canvas(12, 12, paint.pigments, laws=dict(flow=0, cure_rate=0.01))
    for s, v in paint.parts.items(): cv.add(s, v * t)
    assert np.abs(cv.reflectance()[3:9, 3:9] - P.preview(paint, t)["wet_R"]).max() < 1e-12
    for _ in range(60):
        cv.wait(200, evap_boost=10, cure_boost=10)
        if cv.fix() >= 1: break
    assert cv.h.max() < 1e-9 and len(cv.strata) >= 1
    # 1e-6, not round-off: the canvas still composes its empty wet layer, and Canvas.km_layer floors S at 1e-7,
    # so an empty layer reflects ~1e-7 (the World's arithmetic, not a difference between the two)
    assert np.abs(cv.reflectance()[3:9, 3:9] - P.preview(paint, t)["dry_R"]).max() < 1e-6
    under = cv.reflectance()[5, 5]
    pr = P.preview(paint, t, over=np.broadcast_to(under, (3, 3, 4)))
    assert np.allclose(pr["dry_R"], P.preview(paint, t, over=under)["dry_R"])


def test_wet_and_dry():
    """Water carries no K or S in goo's optics, so a water-borne film looks the same wet and dry; only oil
    changes (its linseed yellows: blue falls)."""
    w = P.preview(P.watercolour({"rose": 1}), 0.5)
    assert np.allclose(w["wet"], w["dry"]) and w["dry_thickness"] == pytest.approx(0.02)
    o = P.preview(P.oil({"white": 1}), 0.25)
    assert o["dry_R"][2] < o["wet_R"][2] and o["wet_R"][2] - o["dry_R"][2] > 1.5 * (o["wet_R"][0] - o["dry_R"][0])


def _scrap(paint, target, cure):
    laws = dict(lazy=1, cure_rate=0.002 if cure else 0.0)
    cv = Canvas(90, 40, paint.pigments, laws=laws); b = Brush(cv, 16, exchange=0.85)
    b.charge(paint.load(3 * 70 * 16 * target))
    b.stroke([(10, 20), (45, 20), (80, 20)], speed=2.0, target=target)
    for _ in range(40):
        cv.wait(300, evap_boost=5, cure_boost=10)
        if cv.fix() >= 1: break
    assert cv.h.max() < 1e-9
    sl = (slice(15, 25), slice(25, 65))                              # the stroke's interior
    solid = sum(cv.strata[i] for i in range(len(cv.strata)))
    ref = next(s for s in paint.dried() if s != "oil")
    teq = (solid[cv.idx(ref)] / paint.parts[ref])[sl].ravel()      # wet thickness laid, from the dry solids
    return cv.reflectance()[sl].reshape(-1, 4), teq


@pytest.mark.parametrize("paint,cure,tol,uneven", [
    (P.watercolour({"ultramarine": 1}), False, 2.5, 0.3),           # measured 1.7 and 0.01: a wash levels
    (P.acrylic({"red": 1}), False, 2.5, 1.2),                       # measured 1.6 and 0.85: bristle ridges
    (P.oil({"ochre": 1}), True, 2.0, 1.2)])                         # measured 0.8 and 0.75: bristle ridges
def test_preview_matches_a_brushed_scrap(paint, cure, tol, uneven):
    """A real brush on a scrap canvas, dried and fixed. One pass lays ~45% of the brush's `target` (Brush.touch
    aims at target * pressure * (0.55 + 0.45 bristle) and exchanges 85%), so the painter calibrates on a
    first scrap, as Painter.swatch does, then lays the film it wants. Three checks:
    1. per cell, preview at the thickness the brush actually laid there = the canvas, to 1e-6;
    2. the canvas's mean vs preview at the mean laid thickness: the cost of an uneven film (bristle
       ribbons), since reflectance is not linear in thickness;
    3. the canvas's mean vs preview at the thickness asked for: calibration plus unevenness."""
    t = paint.thickness
    _, teq0 = _scrap(paint, t, cure)
    k = teq0.mean() / t                                             # what one pass lays, per unit target
    assert 0.3 < k < 0.7
    R, teq = _scrap(paint, t / k, cure)
    per = np.array([P.preview(paint, x)["dry_R"] for x in teq])
    assert np.abs(per - R).max() < 1e-6                             # 1. the same optics, cell by cell
    assert abs(teq.mean() / t - 1) < 0.12                           # laid "about that thickness"
    assert dE(R.mean(0), P.preview(paint, teq.mean())["dry_R"]) < uneven   # 2.
    assert dE(R.mean(0), P.preview(paint, t)["dry_R"]) < tol                 # 3.


def test_mix_for_finds_a_reachable_colour_and_is_honest():
    for medium, make in (("oil", lambda: P.oil({"ultramarine": 0.3, "yellow": 0.5, "white": 0.2})),
                         ("acrylic", lambda: P.acrylic({"rose": 0.7, "ochre": 0.3})),
                         ("watercolour", lambda: P.watercolour({"sienna": 0.6, "ultramarine": 0.4}, strength=0.02))):
        want = P.preview(make(), None)["dry"]
        paint, pred, de = P.mix_for(want, medium)
        assert paint.medium == medium and de < 1.5, (medium, paint, de)
        assert np.allclose(pred, P.preview(paint)["dry"])           # it predicts what preview shows
        assert de == pytest.approx(dE(srgb_to_linear(pred), srgb_to_linear(want)), abs=1e-9)
    # outside the gamut it says so
    paint, pred, de = P.mix_for((0.0, 1.0, 1.0), "gouache")
    assert de > 8


def test_mix_for_watercolour_dilutes_and_sees_what_is_under():
    pale, _, de = P.mix_for((0.85, 0.88, 0.97), "watercolour")
    assert "white" not in pale.mix and pale.params["strength"] < 0.03 and de < 3
    blue = P.preview(P.watercolour({"ultramarine": 1}, 0.08))["dry_R"]
    a, _, _ = P.mix_for((0.35, 0.55, 0.35), "watercolour")
    b, pred, _ = P.mix_for((0.35, 0.55, 0.35), "watercolour", over=blue)   # glazing over a blue wash
    assert a.mix != b.mix
    assert np.allclose(pred, P.preview(b, over=blue)["dry"])


def test_mix_for_is_fast():
    P.mix_for((0.5, 0.2, 0.2), "watercolour"); P.mix_for((0.5, 0.2, 0.2), "oil")   # build the caches
    t0 = time.perf_counter()
    for c in ((0.2, 0.5, 0.7), (0.9, 0.6, 0.1), (0.3, 0.3, 0.3)):
        P.mix_for(c, "watercolour"); P.mix_for(c, "oil", over=np.full((20, 20, 4), 0.4))
    assert (time.perf_counter() - t0) / 6 < 0.5


def test_swatch_card(tmp_path):
    img = P.swatch_card([P.watercolour({"rose": 1}), P.oil({"white": 1})], tmp_path / "card.png", title="t")
    assert (tmp_path / "card.png").exists() and img.ndim == 3 and img.shape[2] == 3
    assert "acrylic" in PIGMENTS


# ---------------------------------------------------------------- blending and the journal's form
def test_blend_mixes_by_volume_and_dilutes_toward_water():
    a, b = P.watercolour({"ultramarine": 1}, 0.04), P.watercolour({"ochre": 1}, 0.08)
    assert a.blend(b, 0).parts == pytest.approx(a.parts) and a.blend(b, 1).parts == pytest.approx(b.parts)
    h = a.blend(b, 0.5)
    assert h.medium == "watercolour" and not h.permanent
    assert h.parts["ultramarine"] == pytest.approx(0.02) and h.parts["ochre"] == pytest.approx(0.04)
    assert h.mix == pytest.approx({"ultramarine": 1 / 3, "ochre": 2 / 3})
    ld = h.load(1000.0); la, lb = a.load(1000.0), b.load(1000.0)
    assert all(ld[k] == pytest.approx(0.5 * la.get(k, 0) + 0.5 * lb.get(k, 0)) for k in ld)
    pale = a.blend(P.water(), 0.75)
    assert pale.medium == "watercolour" and pale.solids == pytest.approx(0.01) and pale.mix == pytest.approx({"ultramarine": 1.0})
    with pytest.raises(ValueError): a.blend(P.oil({"ochre": 1}), 0.5)
    with pytest.raises(ValueError): a.blend(b, 1.5)


def test_a_paint_survives_json():
    import json
    for p in (P.watercolour({"rose": 2, "ochre": 1}, 0.03), P.gouache({"ultramarine": 1}), P.acrylic({"red": 1}, body=0.6),
              P.oil({"white": 1, "yellow": 1}, fat=0.4), P.water(), P.watercolour({"rose": 1}).blend(P.water(), 0.3)):
        q = P.Paint.from_json(json.loads(json.dumps(p.to_json())))
        assert q.to_json() == p.to_json() and q.load(777.0) == p.load(777.0) and repr(q) == repr(p)
        assert (q.medium, q.permanent, q.thickness, q.brushful) == (p.medium, p.permanent, p.thickness, p.brushful)


def test_one_acrylic_binder_in_the_engine():
    import goo.classic.paints, inspect
    assert "setdefault" not in inspect.getsource(goo.classic.paints)    # the engine's PIGMENTS["acrylic"] is the only one
    assert P.BINDER in PIGMENTS and P.acrylic({"red": 1}).parts[P.BINDER] > 0
