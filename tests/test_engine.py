"""The World's contract (SPEC.md §3): the books close, nothing goes negative, each law switches off exactly,
and the media behave the way their physics says they should."""
import numpy as np
import pytest
from goo import Canvas, Brush, Stylus


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
