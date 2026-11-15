"""The World's contract (classic/SPEC.md §3): the books close, nothing goes negative, each law switches off exactly,
and the media behave the way their physics says they should."""
import numpy as np
import pytest
from goo.classic import Canvas, Brush, Stylus


def ridges(cv, phi, kind="oil", n=8, vol=0.7):
    """Lay a bristled ribbon by hand (no brush), so a test isolates the medium."""
    yy, xx = np.mgrid[0:cv.H, 0:cv.W]
    band = (np.abs(yy - cv.H / 2) < 10) & (xx > 10) & (xx < cv.W - 10)
    h = vol * band * (1 + 0.6 * np.cos(2 * np.pi * yy / n))
    cv.add(kind, h * (1 - phi)); cv.add("ultramarine", h * phi)
    return band


def relief(cv, band): return float(cv.h[band].std())


def worst_residual(cv):
    led = cv.ledger()
    return max(abs(v["residual"]) / max(1.0, v["loaded"]) for v in led["species"].values())


@pytest.fixture(scope="module")
def messy():
    """Wet and oily strokes, a stylus, drying and curing: the kind of session that breaks books."""
    cv = Canvas(120, 90, ["ultramarine", "ochre", "white"], laws=dict(cure_rate=0.004))
    b = Brush(cv, 14, load=Brush.paint("water", 0.08, 1400, {"ultramarine": 1}))
    b.stroke([(10, 20), (110, 30)]); b.stroke([(10, 40), (110, 28)])
    o = Brush(cv, 10, load=Brush.paint("oil", 0.5, 1600, {"ochre": 2, "white": 1}))
    o.stroke([(15, 70), (105, 60)], target=0.9); o.charge(Brush.paint("oil", 0.55, 1200, {"white": 1})); o.stroke([(60, 85), (60, 10)], target=0.9)
    Stylus(cv, 3).stroke([(5, 50), (115, 50)], speed=1.5)
    for _ in range(150): cv.step()
    return cv


def test_conservation(messy):
    assert worst_residual(messy) < 1e-9


def test_positivity(messy):
    assert messy.q.min() >= 0 and messy.store.min() >= 0


def test_no_capped_steps(messy):
    assert messy.capped == 0


def test_all_laws_off_is_exactly_still():
    cv = Canvas(80, 60, ["ultramarine", "ochre"], laws=dict(flow=0, drag=0, diffusion=0, evaporation=0, absorption=0, curing=0))
    ridges(cv, 0.05, "water"); cv.q[cv.idx("ochre")][::7] += 0.1
    before = cv.q.copy()
    for _ in range(50): cv.step()
    assert np.array_equal(before, cv.q)


def test_diffusion_mixes_colour_not_thickness():
    cv = Canvas(80, 60, ["ultramarine", "ochre"], laws=dict(flow=0, drag=0, evaporation=0, absorption=0))
    ridges(cv, 0.05, "water")
    cv.q[cv.idx("ochre")][:, 40:] += cv.q[cv.idx("ultramarine")][:, 40:]; cv.q[cv.idx("ultramarine")][:, 40:] = 0
    h0 = cv.h.copy(); x0 = cv.q[cv.idx("ochre")] / np.maximum(cv.h, 1e-12)
    for _ in range(60): cv.step()
    x1 = cv.q[cv.idx("ochre")] / np.maximum(cv.h, 1e-12)
    assert np.allclose(h0, cv.h, atol=1e-12)
    assert x1[cv.h > 0].var() < x0[cv.h > 0].var()


@pytest.mark.parametrize("phi,kind,kept_lo,kept_hi", [(0.06, "water", 0.0, 0.15), (0.56, "oil", 0.85, 1.01)])
def test_lean_levels_short_holds(phi, kind, kept_lo, kept_hi):
    cv = Canvas(80, 60, ["ultramarine"], laws=dict(evaporation=0, absorption=0))
    band = ridges(cv, phi, kind); r0 = relief(cv, band)
    for _ in range(300): cv.step()
    assert kept_lo <= relief(cv, band) / r0 <= kept_hi


def test_curing_stiffens():
    out = []
    for rate in (0.05, 0.0):
        cv = Canvas(80, 60, ["ultramarine"], laws=dict(cure_rate=rate, evaporation=0, absorption=0))
        band = ridges(cv, 0.15, "oil")
        for _ in range(400): cv.step()
        out.append(relief(cv, band))
    assert out[0] > 1.5 * out[1]


def test_drag_conserves_and_moves():
    cv = Canvas(100, 60, ["ultramarine", "ochre"], laws=dict(flow=0, diffusion=0, evaporation=0, absorption=0))
    yy, xx = np.mgrid[0:60, 0:100]
    blob = 0.5 * np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / 60)
    cv.add("oil", blob * 0.5); cv.add("ochre", blob * 0.5)
    cx0 = (cv.h * xx).sum() / cv.h.sum(); tot0 = cv.h.sum()
    st = Stylus(cv, 40)
    for i in range(20): st.touch(30 + i, 30, 1.0, 0.0, 1.0, 0); cv.step()
    assert abs(cv.h.sum() - tot0) < 1e-9 * tot0
    assert (cv.h * xx).sum() / cv.h.sum() - cx0 > 5


def test_runny_spike_spreads_and_never_grows():
    cv = Canvas(60, 60, ["ultramarine"], laws=dict(eta0=0.01))
    spike = np.zeros((60, 60)); spike[28:32, 28:32] = 40.0; cv.add("water", spike)
    for _ in range(30): cv.step()
    assert np.isfinite(cv.q).all() and cv.q.min() >= 0 and worst_residual(cv) < 1e-9
    assert cv.h.max() < 10 and (cv.h > 0.05).sum() >= 64 and cv.capped == 0


def test_thin_wash_is_pinned():
    cv = Canvas(60, 60, ["ultramarine"], laws=dict(evaporation=0, absorption=0))
    cv.q[0, 20:40, 20:40] = 0.06; cv.q[2, 20:40, 20:40] = 0.004
    for _ in range(100): cv.step()
    assert (cv.h > 1e-6).sum() == 400


def test_strata_sgraffito():
    cv = Canvas(80, 60, ["sienna", "ultramarine", "white"], laws=dict(evaporation=0, absorption=0))
    cv.add("sienna", np.full((60, 80), 0.05))
    assert cv.fix() == 1.0
    under = cv.strata[0].copy()
    yy, xx = np.mgrid[0:60, 0:80]; band = (np.abs(yy - 30) < 18)
    for k, v in Brush.paint("oil", 0.55, 1.0, {"ultramarine": 1, "white": 2}).items(): cv.add(k, v * 0.8 * band)
    R0 = cv.reflectance()
    st = Stylus(cv, 3)
    for i in range(70): st.touch(5 + i, 30, 1.0, 0.0, 1.0, 0); cv.step()
    R1 = cv.reflectance()
    assert worst_residual(cv) < 1e-9
    assert np.array_equal(under, cv.strata[0])                                   # a dried layer can't be moved
    assert cv.h[30, 20:60].mean() < 0.5 * cv.h[24:27, 20:60].mean()              # the groove is cut
    warm = lambda R: (R[30, 20:60, 0] - R[30, 20:60, 2]).mean()
    assert warm(R1) > warm(R0) + 0.05                                            # and shows the sienna beneath


def test_reflectography():
    cv = Canvas(60, 40, ["carbon", "white", "ochre"])
    line = np.zeros((40, 60)); line[:, 28:32] = 0.03; cv.add("carbon", line); cv.fix()
    for k, v in Brush.paint("oil", 0.5, 0.6, {"white": 2, "ochre": 1}).items(): cv.add(k, np.full((40, 60), v))
    R = cv.reflectance()
    assert 1 - R[20, 30, 1] / R[20, 10, 1] < 0.08                               # hidden in green
    assert 1 - R[20, 30, 3] / R[20, 10, 3] > 0.25                               # plain at 1.5 um


def test_determinism():
    def session():
        c = Canvas(60, 40, ["ultramarine"]); Brush(c, 8, load=Brush.paint("water", 0.1, 600, {"ultramarine": 1})).stroke([(5, 20), (55, 22)])
        for _ in range(20): c.step()
        return c.q
    assert np.array_equal(session(), session())


def test_lazy_clock_conserves_and_agrees():
    """Slow laws batched every 25 ticks: same books, and the paint ends up close to the every-tick result."""
    def session(lazy):
        cv = Canvas(100, 70, ["ultramarine", "white"], laws=dict(lazy=lazy, cure_rate=0.002))
        b = Brush(cv, 12, load=Brush.paint("oil", 0.48, 3000, {"ultramarine": 1, "white": 2}))
        b.stroke([(10, 20), (90, 25)]); b.stroke([(10, 45), (90, 35)]); b.stroke([(50, 5), (50, 65)])
        for _ in range(100): cv.step()
        cv.flush(); return cv
    a, b = session(1), session(25)
    assert worst_residual(b) < 1e-9 and b.q.min() >= 0
    assert a.t == b.t
    assert np.abs(a.h - b.h).mean() < 0.05 * a.h[a.h > 0.01].mean()


# ---- windowed bookkeeping and slow laws (L["windowed"]) must give exactly the whole-panel result

def _edge_session(windowed):
    """Tools, charcoal, a wash that soaks the ground and is fixed while the ground still holds water, watercolour
    far from it, paint at the panel's border and corner, a stylus, wait() with boosts, overlapping strata,
    lazy > 1, the drag law off for a while, and paint written straight into q (a test-setup habit)."""
    cv = Canvas(110, 80, ["ultramarine", "sienna", "white", "carbon"], seed=4, laws=dict(windowed=windowed, store_evap=2e-5))
    coal = Brush(cv, 2.6, bristles=1, exchange=0.6, tooth_only=True, drag=0, plough=0, load={"carbon": 4.0})
    coal.stroke([(10, 10), (50, 15), (70, 40)], speed=1.6, target=0.045); coal.stroke([(0.5, 75), (25, 79.5)], speed=1.6, target=0.05)
    cv.fix()
    mop = Brush(cv, 20, bristles=8, exchange=0.45, plough=0.2, load=Brush.paint("water", 0.06, 1500, {"sienna": 3, "ultramarine": 1}))
    mop.stroke([(5, 15), (35, 18), (55, 12)], speed=4.0, pressure=0.9, target=0.5)
    cv.wait(100, evap_boost=8.0)
    while cv._q[0].max() > 1e-6: cv.wait(20, evap_boost=8.0)
    cv.fix(); assert cv._store.max() > 0                                   # the ground still holds water
    b = Brush(cv, 10, bristles=4, load=Brush.paint("water", 0.08, 800, {"ultramarine": 1}))
    b.stroke([(85, 55), (109.5, 65), (109.5, 79.5)], target=0.5)             # far from it, into the corner
    for _ in range(15): cv.step()
    cv.q[2, 40:44, 0:3] += 0.05; cv.led["loaded"][2] += 0.05 * 12           # written straight in at the border
    o = Brush(cv, 9, load=Brush.paint("oil", 0.5, 900, {"white": 2, "sienna": 1}))
    o.stroke([(30, 0.5), (80, 2), (100, 0.2)], target=0.8)                   # along the top border
    Stylus(cv, 3).stroke([(35, 4), (85, 1)], speed=1.5)
    cv.L["cure_rate"] = 0.003; cv.L["lazy"] = 6
    cv.L["drag"] = 0; o.stroke([(60, 30), (70, 50)], target=0.7); cv.L["drag"] = 1.0
    cv.wait(60, evap_boost=3.0, cure_boost=10.0); cv.fix(cure_min=0.3)
    o.charge(Brush.paint("oil", 0.5, 700, {"ultramarine": 1, "white": 1})); o.stroke([(55, 5), (60, 35), (50, 60)], target=0.7)
    cv.wait(40, cure_boost=12.0); cv.fix(cure_min=0.2)
    for _ in range(10): cv.step()
    cv.flush(); return cv


def test_windowed_equals_whole_panel():
    a, b = _edge_session(1), _edge_session(0)
    assert len(a.strata) == len(b.strata) >= 4
    for name in ("_q", "_kappa", "_store", "zb", "_U", "_V"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert all(np.array_equal(x, y) for x, y in zip(a.strata, b.strata))
    assert (a.t, a.substeps, a.capped) == (b.t, b.substeps, b.capped)
    assert a.ledger() == b.ledger()
    assert worst_residual(a) < 1e-9


def test_windowed_equals_whole_panel_capped():
    def run(windowed):
        cv = Canvas(60, 60, ["ultramarine"], laws=dict(eta0=0.01, max_sub=3, windowed=windowed))
        spike = np.zeros((60, 60)); spike[28:32, 28:32] = 40.0; cv.add("water", spike)
        for _ in range(12): cv.step()
        return cv
    a, b = run(1), run(0)
    assert a.capped == b.capped > 0 and a.substeps == b.substeps
    assert np.array_equal(a._q, b._q) and np.array_equal(a._store, b._store) and a.ledger() == b.ledger()


def _dense_fix(q, kappa, cure_min):
    """What the dense implementation's fix() would append, from copies of the state."""
    h = q.sum(0)
    ok = (h > 0) & (q[0] <= 1e-6) & ((q[1] <= 0.05 * np.maximum(h, 1e-12)) | (kappa >= cure_min))
    layer = q * ok; layer[0] = 0
    return layer


def test_compact_strata_round_trip():
    cv = Canvas(90, 70, ["sienna", "ultramarine", "white"], laws=dict(evaporation=0, absorption=0, cure_rate=0.01))
    expect = []
    for k, (y, x) in enumerate([(10, 10), (30, 40), (20, 25), (60, 85)]):     # partly overlapping patches, one at the border
        yy, xx = np.mgrid[0:70, 0:90]; blob = 0.2 * (np.hypot(yy - y, xx - x) < 9)
        cv.add("oil", blob * 0.5); cv.add(["sienna", "ultramarine", "white"][k % 3], blob * 0.5)
        cv.wait(30, cure_boost=20.0); cv.flush()
        layer = _dense_fix(cv._q.copy(), cv._kappa.copy(), 0.5)
        cv.fix(cure_min=0.5)
        if layer.any(): expect.append(layer)
    assert len(cv.strata) == len(expect) >= 3
    for i, d in enumerate(expect):
        s = cv.strata[i]
        assert s.shape == d.shape and np.array_equal(s, d)
        assert np.array_equal(cv.strata.totals(i), d.sum(axis=(1, 2)))      # the ledger's view, bit for bit
        with pytest.raises(ValueError): s[0, 0, 0] = 1.0                     # read-only (I5)
        y0, y1, x0, x1, blk = cv.strata.block(i)
        assert blk.shape == (len(cv.species), y1 - y0, x1 - x0) and not blk.flags.writeable
    assert [x.shape for x in cv.strata] == [d.shape for d in expect]
    assert all(np.array_equal(x, d) for x, d in zip(cv.strata, expect)) and len(cv.strata[1:]) == len(expect) - 1
    assert np.array_equal(cv.strata[-1], expect[-1]) and np.array_equal(cv.strata[0].copy(), expect[0])
    assert np.array_equal(cv.zb, cv.g + sum(d.sum(0) for d in expect))
    assert cv.strata.nbytes < 0.5 * sum(d.nbytes for d in expect)
    # the optics, composed from dense layers the way the dense implementation did
    Rb = np.broadcast_to(np.array([0.93, 0.92, 0.88, 0.90]), (cv.H, cv.W, 4)).copy()
    for vol, wet in [(d, False) for d in expect] + [(cv._q, True)]:
        R, T = cv.km_layer(*cv._KS(vol * 10.0, wet)); Rb = R + T * T * Rb / (1 - R * Rb)
    assert np.array_equal(np.clip(Rb, 0, 1), cv.reflectance())
    assert worst_residual(cv) < 1e-9


def test_ivory_black_is_darker_than_carbon():
    out = {}
    for pig in ("ivory_black", "carbon"):
        cv = Canvas(30, 20, [pig])
        for k, v in Brush.paint("oil", 0.5, 1.0, {pig: 1}).items(): cv.add(k, np.full((20, 30), v))    # 0.25 mm of paint
        out[pig] = cv.reflectance()[10, 15, :3]
    assert out["ivory_black"].max() < 0.004
    assert (out["ivory_black"] < out["carbon"]).all() and 0.01 < out["carbon"].mean() < 0.02


def test_windowed_equals_whole_panel_when_the_window_moves(monkeypatch):
    """A runny puddle spreads across many cells within one flush; with a 3-cell margin the window must move
    again and again. Same arrays, books and substeps as the whole panel (a margin of _NEAR = 1 would differ)."""
    monkeypatch.setattr(Canvas, "_MARGIN", 3)
    def run(windowed):
        cv = Canvas(90, 70, ["ultramarine", "white"], laws=dict(eta0=0.01, windowed=windowed, lazy=3))
        spike = np.zeros((70, 90)); spike[30:34, 40:44] = 30.0; cv.add("water", spike); cv.add("ultramarine", spike * 0.05)
        b = Brush(cv, 6, load=Brush.paint("water", 0.05, 400, {"white": 1})); b.stroke([(5, 5), (30, 8)], target=1.5)
        for _ in range(40): cv.step()
        cv.flush(); return cv
    a, b = run(1), run(0)
    assert (a.h > 1e-9).sum() > 300
    for name in ("_q", "_kappa", "_store"): assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert (a.substeps, a.capped) == (b.substeps, b.capped) and a.ledger() == b.ledger()
    a, b = _edge_session(1), _edge_session(0)
    for name in ("_q", "_kappa", "_store", "zb"): assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert all(np.array_equal(x, y) for x, y in zip(a.strata, b.strata)) and a.ledger() == b.ledger()


def _masked_session(windowed):
    """The frisket in the windowed books: tape over the panel's border and corner, over a fixed stratum, over dry
    unfixed pigment and over a ground still holding water; washes, oil, a stylus and a flood against it; lazy > 1,
    boosts, partial unmasking, taping again, fixing."""
    cv = Canvas(110, 80, ["ultramarine", "sienna", "white"], seed=5, laws=dict(windowed=windowed, store_evap=2e-5, cure_rate=0.003))
    yy, xx = np.mgrid[0:80, 0:110] + 0.5
    cv.add("sienna", 0.03 * (np.hypot(yy - 40, xx - 55) < 12)); cv.fix()            # a stratum under the tape
    mop = Brush(cv, 18, bristles=6, load=Brush.paint("water", 0.06, 1500, {"ultramarine": 1}))
    mop.stroke([(5, 70), (40, 72)], speed=4.0, target=0.5)
    cv.wait(100, evap_boost=8.0)
    while cv._q[0].max() > 1e-6: cv.wait(20, evap_boost=8.0)
    assert cv._store.max() > 0                                                       # the ground still holds water
    cv.add("ultramarine", 0.01 * ((yy < 6) & (xx > 80)))                            # dry pigment at the top border
    tape = (np.hypot(yy - 40, xx - 55) < 8) | ((yy > 66) & (xx < 30)) | ((yy < 4) & (xx > 85)) | ((xx > 104) & (yy > 70))
    cv.mask(tape)
    b = Brush(cv, 10, bristles=4, load=Brush.paint("water", 0.08, 900, {"ultramarine": 1, "sienna": 1}))
    b.stroke([(30, 40), (80, 42)], target=0.5); b.stroke([(85, 2), (109.5, 6), (109.5, 79.5)], target=0.5)
    o = Brush(cv, 9, load=Brush.paint("oil", 0.5, 900, {"white": 2, "sienna": 1}))
    o.stroke([(10, 60), (60, 75), (100, 70)], target=0.8)
    cv.L["lazy"] = 5
    Stylus(cv, 3).stroke([(5, 74), (100, 66)], speed=1.5)
    cv.wait(60, evap_boost=3.0, cure_boost=10.0); cv.fix(cure_min=0.3)
    cv.unmask((yy > 66) & (xx < 30))
    for _ in range(30): cv.step()
    cv.wait(40, evap_boost=4.0, cure_boost=10.0); cv.fix(cure_min=0.3)
    cv.mask(cv.touch_dry() & (xx < 20))
    o.stroke([(2, 50), (30, 55)], target=0.8)
    cv.wait(30, cure_boost=12.0); cv.fix(cure_min=0.2)
    cv.add("water", 12.0 * ((np.abs(yy - 40) < 3) & (np.abs(xx - 44) < 3)))         # a flood against the tape
    for _ in range(9): cv.step()
    cv.flush(); return cv


def test_windowed_equals_whole_panel_with_mask():
    a, b = _masked_session(1), _masked_session(0)
    assert a._mask is not None and np.array_equal(a._mask, b._mask)
    assert len(a.strata) == len(b.strata) >= 3
    for name in ("_q", "_kappa", "_store", "zb", "_U", "_V"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert all(np.array_equal(x, y) for x, y in zip(a.strata, b.strata))
    assert (a.t, a.substeps, a.capped) == (b.t, b.substeps, b.capped)
    assert a.ledger() == b.ledger()
    assert worst_residual(a) < 1e-9 and a._q.min() >= 0 and a._store.min() >= 0
