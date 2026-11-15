"""Tools (classic/SPEC.md §5, §9): stroke receipts, rinsing, the flat brush, the knife, the blotter, spatter, and tape
(cv._open). Each tool exchanges only through its load, keeps the books closed (I1), never touches a stratum
(I5), honours the canvas's tape, and gives the same arrays windowed and whole-panel (I8)."""
import sys
import numpy as np
import pytest
import goo.classic.tools
from goo.classic import Canvas, Brush, Stylus
from goo.classic.brush import Receipt
from goo.classic.tools import Knife, Blotter, spatter

NOLAWS = dict(evaporation=0, absorption=0)


def worst_residual(cv):
    led = cv.ledger()
    return max(abs(v["residual"]) / max(1.0, v["loaded"]) for v in led["species"].values())


def fwhm(profile):
    return int((profile > 0.5 * profile.max()).sum())


def oil(volume, phi=0.5, pig="ultramarine"):
    return Brush.paint("oil", phi, volume, {pig: 1})


# ---------------------------------------------------------------- receipts and rinsing

def test_receipt_books_what_the_stroke_did():
    cv = Canvas(100, 60, ["ultramarine", "white"], laws=NOLAWS)
    b = Brush(cv, 10, load=oil(3000))
    load0 = b.load.copy()
    r = b.stroke([(10, 20), (90, 20)], target=0.8)
    assert np.allclose(r.given - r.taken, load0 - b.load, rtol=0, atol=1e-9)
    assert np.isclose(r.given.volume, cv.h.sum() + r.taken.volume, rtol=1e-12)    # the canvas was bare
    assert r.length == pytest.approx(80) and r.ticks == 32 and 0.8 * 80 * 9 < r.cells < 82 * 15
    r2 = b.stroke([(10, 40), (90, 45)], target=0.8, steps_after=5)
    s = r + r2
    assert s.ticks == r.ticks + r2.ticks == cv.t and s.length == pytest.approx(r.length + r2.length)
    assert np.allclose(s.given, r.given + r2.given) and r.cells < s.cells <= r.cells + r2.cells
    assert sum([r, r2]).cells == s.cells
    over = r + b.stroke([(10, 20), (90, 20)], target=0.8)                          # the same cells again
    assert over.cells == r.cells
    assert worst_residual(cv) < 1e-9


def test_a_dry_brush_lays_nothing_and_says_so():
    cv = Canvas(60, 40, ["ultramarine"], laws=NOLAWS)
    r = Brush(cv, 8).stroke([(5, 20), (55, 20)])
    assert r.given.sum() == 0 and r.taken.sum() == 0 and r.cells > 0 and r.given.volume == 0.0
    st = Stylus(cv, 3).stroke([(5, 10), (55, 10)])
    assert st.given.sum() == 0 and st.ticks > 0


def test_rinse_books_the_load_as_rinsed():
    cv = Canvas(60, 40, ["ultramarine"], laws=NOLAWS)
    b = Brush(cv, 8, load=oil(500)); b.stroke([(5, 20), (55, 20)])
    left = b.load.copy()
    out = b.rinse()
    assert np.array_equal(out, left) and b.load.sum() == 0
    assert np.allclose(cv.led["rinsed"], left)
    assert worst_residual(cv) < 1e-9
    assert b.stroke([(5, 30), (55, 30)]).given.sum() == 0                           # nothing left to lay


# ---------------------------------------------------------------- the flat brush

def _flat_mark(heading, path, width=12, aspect=3.0):
    cv = Canvas(120, 120, ["ultramarine"], seed=3, laws=NOLAWS)
    b = Brush(cv, width, shape="flat", aspect=aspect, load=oil(6000))
    r = b.stroke(path, target=0.8, heading=heading)
    return cv, r


def test_flat_brush_face_and_edge():
    W, A = 12, 3.0
    face, rf = _flat_mark(np.pi / 2, [(20, 60), (100, 60)])                         # long axis across the travel
    edge, re = _flat_mark(0.0, [(20, 60), (100, 60)])                               # long axis along it
    wf, we = fwhm(face.h[:, 60]), fwhm(edge.h[:, 60])
    assert 0.75 * W <= wf <= 1.1 * W                                                # a band as wide as the brush
    assert 0.6 * W / A <= we <= 1.35 * W / A                                        # a line width / aspect wide
    assert rf.cells > 2 * re.cells and rf.given.volume > 2 * re.given.volume
    # heading is in the image frame whatever the travel: the same face mark going down the panel
    down, _ = _flat_mark(0.0, [(60, 20), (60, 100)])
    assert fwhm(down.h[60, :]) == wf
    for cv in (face, edge, down): assert worst_residual(cv) < 1e-9


def test_flat_brush_goes_thick_and_thin_round_a_turn():
    """A circle drawn with the heading fixed (long axis horizontal): thin where the travel is horizontal (top and
    bottom), broad where it is vertical (left and right), like a calligraphy pen. (On a curve of radius 35 the thin
    stroke is a little wider than on a straight line: the 12-cell edge spans a sagitta of the arc.)"""
    cx, cy, R = 60, 60, 35
    path = [(cx + R * np.cos(t), cy + R * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 73)]
    cv, r = _flat_mark(0.0, path)
    h = cv.h
    laid = lambda p: int((p > 0.05).sum())
    side = [laid(h[cy, cx - R - 12:cx - R + 13]), laid(h[cy, cx + R - 12:cx + R + 13])]       # travel vertical: face
    topb = [laid(h[cy - R - 12:cy - R + 13, cx]), laid(h[cy + R - 12:cy + R + 13, cx])]       # travel horizontal: edge
    assert min(side) >= 9 and max(topb) <= 7 and min(side) >= 1.5 * max(topb)
    assert worst_residual(cv) < 1e-9


def test_round_brush_has_no_heading():
    cv = Canvas(40, 30, ["ultramarine"])
    with pytest.raises(ValueError): Brush(cv, 6).stroke([(5, 5), (30, 5)], heading=0.3)
    with pytest.raises(ValueError): Brush(cv, 6, shape="square")


# ---------------------------------------------------------------- the knife

def _thick_uneven(cv, phi=0.58):
    yy, xx = np.mgrid[0:cv.H, 0:cv.W]; rng = np.random.default_rng(0)
    h = (1.5 + 0.6 * np.sin(xx / 5.0) * np.cos(yy / 7.0) + 0.3 * rng.random((cv.H, cv.W))) * ((xx > 5) & (xx < cv.W - 5) & (yy > 5) & (yy < cv.H - 5))
    cv.add("oil", (1 - phi) * h); cv.add("white", phi * h)
    return h


def test_knife_scrapes_flat_with_a_ridge_each_side():
    cv = Canvas(120, 90, ["ultramarine", "white"], seed=3, laws=NOLAWS)
    h0 = _thick_uneven(cv)
    k = Knife(cv, 20)
    r = k.stroke([(10, 45), (110, 45)], speed=1.5, target=0.3)
    h = cv.h; floor = h[39:52, 30:90]                                               # the blade's middle 13 of 20 cells
    assert abs(floor.mean() - 0.3) < 0.03 and floor.std() < 0.03 and np.abs(floor - 0.3).max() < 0.1
    gain = (h - h0)[:, 30:90].mean(1)
    assert gain[28:36].max() > 0.3 and gain[55:63].max() > 0.3                      # a ridge thrown up at each side
    assert r.taken.volume > 0.9 * (h0[39:52, 12:108] - 0.3).sum() and k.load.sum() == pytest.approx(r.taken.volume - r.given.volume)
    assert worst_residual(cv) < 1e-9 and cv.capped == 0


def test_knife_takes_all_the_excess_a_brush_a_quarter():
    out = {}
    for name in ("knife", "brush"):
        cv = Canvas(60, 40, ["ultramarine", "white"], laws=dict(NOLAWS, flow=0, drag=0, diffusion=0))
        cv.add("oil", np.full((40, 60), 0.5)); cv.add("white", np.full((40, 60), 0.5))
        t = Knife(cv, 12) if name == "knife" else Brush(cv, 12, bristles=1, exchange=1.0, drag=0, plough=0)
        t.touch(30.5, 20.5, 0.0, 0.0, 1.0, 0.2)                                    # centred on cell (20, 30)
        out[name] = cv.h[20, 30]
    assert out["knife"] == pytest.approx(0.2, abs=1e-12)
    assert out["brush"] > 0.2 + 0.7 * (1.0 - 0.2 * (0.55 + 0.45))


def test_loaded_knife_lays_an_even_slab():
    cv = Canvas(120, 90, ["ultramarine", "white"], seed=3, laws=NOLAWS)
    k = Knife(cv, 20, load=Brush.paint("oil", 0.58, 4000, {"white": 1}))
    r = k.stroke([(10, 45), (110, 45)], speed=1.5, target=1.0)
    slab = cv.h[39:52, 20:100]
    assert abs(slab.mean() - 1.0) < 0.05 and slab.std() < 0.02                     # no bristle ribbons
    assert r.given.volume > 0 and worst_residual(cv) < 1e-9


def test_knife_scrapes_back_to_the_stratum_and_no_further():
    cv = Canvas(80, 60, ["sienna", "ultramarine", "white"], laws=NOLAWS)
    cv.add("sienna", np.full((60, 80), 0.05)); assert cv.fix() == 1.0
    under = cv.strata[0].copy()
    for s, v in Brush.paint("oil", 0.55, 1.0, {"ultramarine": 1, "white": 2}).items(): cv.add(s, np.full((60, 80), v))
    R0 = cv.reflectance()
    Knife(cv, 16).stroke([(5, 30), (75, 30)], speed=1.5, target=0.0)
    assert np.array_equal(under, cv.strata[0])                                      # strata cannot be moved (I5)
    assert cv.h[26:35, 15:65].max() < 1e-9                                          # scraped back to the stratum
    R1 = cv.reflectance()
    assert (R1[30, 20:60, 0] - R1[30, 20:60, 2]).mean() > (R0[30, 20:60, 0] - R0[30, 20:60, 2]).mean() + 0.05
    assert worst_residual(cv) < 1e-9


# ---------------------------------------------------------------- the blotter

def _wash(cv):
    yy, xx = np.mgrid[0:cv.H, 0:cv.W]; band = ((xx > 10) & (xx < cv.W - 10) & (yy > 10) & (yy < cv.H - 10)).astype(float)
    cv.add("water", 0.15 * band); cv.add("ultramarine", 0.012 * band)


def test_blotting_a_wet_wash_lightens_it():
    cv = Canvas(80, 60, ["ultramarine"], seed=3)
    _wash(cv)
    for _ in range(5): cv.step()
    R0 = cv.reflectance()
    b = Blotter(cv, 16)
    under = cv._q[:, 22:39, 32:49].sum(axis=(1, 2))                                 # what lies under the sheet
    U0 = cv._U.copy(); r = b.press(40, 30, ticks=3)
    R1 = cv.reflectance()
    assert R1[30, 40, :3].mean() > R0[30, 40, :3].mean() + 0.1                      # lifted: lighter where pressed
    assert np.allclose(R1[30, 18], R0[30, 18], atol=0.01)                           # not beside it
    assert r.given.sum() == 0 and r.taken.volume > 0 and r.ticks == 3 and r.length == 0
    w, u = r.taken[0], r.taken[2]
    assert w / u > under[0] / under[2]                                              # water a little preferred
    assert np.array_equal(cv._U, U0) and not cv._V.any()                            # no velocity written
    assert worst_residual(cv) < 1e-9


def test_blotting_dry_paint_or_strata_does_nothing():
    cv = Canvas(80, 60, ["ultramarine", "carbon"], seed=3)
    _wash(cv); cv.wait(400, evap_boost=10.0)
    assert cv._q[0].max() == 0                                                      # dry, not yet fixed
    q0 = cv._q.copy(); b = Blotter(cv, 16)
    assert b.press(40, 30, ticks=3).taken.sum() == 0 and np.array_equal(cv._q, q0)
    cv.fix(); n = len(cv.strata); under = cv.strata[-1].copy()
    assert b.press(40, 30, ticks=3).taken.sum() == 0
    assert len(cv.strata) == n and np.array_equal(cv.strata[-1], under)
    c = Brush(cv, 2.6, bristles=1, exchange=0.6, tooth_only=True, drag=0, plough=0, load={"carbon": 4.0})
    c.stroke([(20, 30), (60, 30)], speed=1.6, target=0.05)                         # charcoal: dry pigment, no carrier
    q1 = cv._q.copy()
    assert b.press(40, 30).taken.sum() == 0 and np.array_equal(cv._q, q1)
    assert worst_residual(cv) < 1e-9


def test_blotter_fills_and_is_wrung():
    cv = Canvas(80, 60, ["ultramarine"], seed=3, laws=NOLAWS)
    _wash(cv)
    b = Blotter(cv, 16, capacity=5.0)
    r = b.press(40, 30, ticks=4)
    assert r.taken.volume == pytest.approx(5.0) and b.room == 0
    assert b.press(30, 30).taken.sum() == 0                                         # full: it stops taking
    held = b.load.copy(); out = b.wring()
    assert np.array_equal(out, held) and b.load.sum() == 0 and np.allclose(cv.led["rinsed"], held)
    assert b.press(30, 30).taken.volume > 0                                         # wrung, it takes again
    assert worst_residual(cv) < 1e-9


def test_thirsty_brush_lifts_along_a_path():
    cv = Canvas(80, 60, ["ultramarine"], seed=3, laws=NOLAWS)
    _wash(cv)
    h0 = cv.h.copy()
    r = Blotter(cv, 6, capacity=1e9).stroke([(15, 30), (65, 30)])
    assert r.taken.volume > 0 and cv.h[30, 25:55].mean() < 0.6 * h0[30, 25:55].mean() and np.abs(cv.h[15] - h0[15]).max() < 0.02
    assert worst_residual(cv) < 1e-9


# ---------------------------------------------------------------- spatter

def test_spatter_is_droplets_from_the_load():
    def run(seed, n=60, vol=2000):
        cv = Canvas(100, 80, ["ultramarine", "sienna"], seed=3, laws=NOLAWS)
        b = Brush(cv, 10, load=Brush.paint("water", 0.08, vol, {"sienna": 1}))
        load0 = b.load.copy()
        r = spatter(b, (50, 40), n, 15.0, 0.6, np.random.default_rng(seed))
        return cv, b, load0, r
    cv, b, load0, r = run(4)
    assert np.allclose(r.given, load0 - b.load, atol=1e-12) and r.taken.sum() == 0 and r.ticks == 1
    assert 0 < r.given.volume <= load0.sum()
    from scipy import ndimage
    _, ndrops = ndimage.label(cv.h > 1e-6)
    assert 20 <= ndrops <= 60                                                       # separate droplets (some overlap)
    assert not cv._U.any() and not cv._V.any()                                      # no drag
    cv2, _, _, r2 = run(4)
    assert np.array_equal(cv._q, cv2._q) and np.array_equal(r.given, r2.given)     # deterministic for an rng
    cv3, _, _, _ = run(5)
    assert not np.array_equal(cv._q, cv3._q)
    cv4, b4, l4, r4 = run(4, n=200, vol=3.0)                                        # more asked than it holds
    assert r4.given.volume <= l4.sum() + 1e-12 and b4.load.sum() < 1e-9
    for c in (cv, cv4): assert worst_residual(c) < 1e-9


# ---------------------------------------------------------------- tape: cv._open

class Taped(Canvas):
    """A canvas with tape over x < W/2 (a stand-in for the frisket of goo.classic.engine, which this copy lacks)."""
    def _open(self, y0, y1, x0, x1):
        o = np.ones((y1 - y0, x1 - x0)); o[:, : max(0, self.W // 2 - x0)] = 0.0
        return o


def test_no_tool_acts_under_tape():
    """Tools give, take and write velocity only where the tape leaves the canvas open. (Keeping the engine's own
    laws out from under tape is the canvas's business: here they are off, so only the tools act.)"""
    cv = Taped(100, 60, ["ultramarine", "white"], seed=3, laws=dict(NOLAWS, flow=0, drag=0, diffusion=0))
    yy, xx = np.mgrid[0:60, 0:100]
    pre = 0.4 * ((xx < 45) & (yy > 40) & (yy < 55))                                 # wet paint under the tape
    cv.add("oil", pre); cv.add("white", pre)
    left = lambda: cv._q[:, :, :50].copy()
    q0 = left()
    b = Brush(cv, 10, load=oil(3000))
    b.touch(50, 20, 2.0, 0.0, 1.0, 0.8)
    assert not cv._U[:, :, :50].any() and not cv._V[:, :, :50].any() and cv._U[:, :, 50:].any()
    cv.step()
    rs = [b.stroke([(10, 20), (90, 20)], target=0.8),
          Brush(cv, 12, shape="flat", load=oil(3000)).stroke([(10, 30), (90, 35)], target=0.8, heading=1.2),
          Knife(cv, 16).stroke([(10, 47), (90, 47)], target=0.0),
          Knife(cv, 16, load=oil(2000, 0.58)).stroke([(10, 10), (90, 10)], target=0.8),
          Blotter(cv, 14).press(52, 20, ticks=3),
          Blotter(cv, 8).stroke([(10, 22), (90, 22)]),
          spatter(b, (50, 30), 80, 20.0, 0.6, np.random.default_rng(1))]
    assert np.array_equal(left(), q0)                                               # nothing entered or left
    assert all(r.given.sum() + r.taken.sum() > 0 for r in rs)                       # and each did act where open
    assert cv.h[20, 60:90].mean() > 0.3
    assert worst_residual(cv) < 1e-9


def test_no_tape_means_no_change():
    """A canvas whose _open returns None (nothing masked) gives the same arrays as one with no _open at all."""
    class Untaped(Canvas):
        def _open(self, y0, y1, x0, x1): return None
    out = []
    for C in (Canvas, Untaped):
        cv = C(80, 60, ["ultramarine", "white"], seed=2)
        Brush(cv, 10, load=oil(2000)).stroke([(5, 20), (75, 25)], target=0.8)
        Brush(cv, 10, shape="flat", load=oil(2000)).stroke([(5, 40), (75, 35)], target=0.8, heading=0.4)
        Knife(cv, 12).stroke([(40, 5), (40, 55)], target=0.2)
        out.append(cv._q.copy())
    assert np.array_equal(*out)


# ---------------------------------------------------------------- the engine's contract for the new tools

def _tool_session(windowed):
    cv = Canvas(110, 80, ["ultramarine", "sienna", "white"], seed=4, laws=dict(windowed=windowed, lazy=4, cure_rate=0.002))
    f = Brush(cv, 12, shape="flat", aspect=4.0, load=oil(3000))
    f.stroke([(10, 10), (60, 30), (100, 12)], target=0.9, heading=0.3)
    f.stroke([(0.5, 60), (40, 79.5)], target=0.9)                                   # into the corner, heading follows
    Knife(cv, 14).stroke([(5, 20), (100, 22)], speed=1.5, target=0.2)
    Knife(cv, 10, load=oil(2000, 0.58, "white")).stroke([(60, 40), (109.5, 70)], target=1.0)
    w = Brush(cv, 14, load=Brush.paint("water", 0.08, 1500, {"sienna": 1})); w.stroke([(20, 50), (70, 55)], target=0.5)
    spatter(w, (50, 50), 30, 12.0, 0.5, np.random.default_rng(3))
    bl = Blotter(cv, 10); bl.press(45, 52, ticks=4); bl.stroke([(30, 55), (60, 50)]); bl.wring()
    cv.wait(30, cure_boost=5.0); cv.fix(cure_min=0.3)
    Knife(cv, 12).stroke([(20, 70), (90, 70)], target=0.0)
    for _ in range(8): cv.step()
    cv.flush(); return cv


def test_new_tools_windowed_equals_whole_panel():
    a, b = _tool_session(1), _tool_session(0)
    for name in ("_q", "_kappa", "_store", "zb", "_U", "_V"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert all(np.array_equal(x, y) for x, y in zip(a.strata, b.strata))
    assert (a.t, a.substeps, a.capped) == (b.t, b.substeps, b.capped) and a.ledger() == b.ledger()
    assert worst_residual(a) < 1e-9 and a._q.min() >= 0


TOOL_MAY = {"q", "U", "V", "surface", "_q", "_U", "_V", "_mark", "_mark_tool", "_surface_window",
            "species", "idx", "led", "brushes", "step", "W", "H", "_open"}


class Glass:
    """A canvas behind glass for goo.classic.tools (as classic/tests/test_detail.py's Watched is for goo.classic.brush): code in tools.py
    may read only TOOL_MAY (the tool channel and tape); it may set nothing."""
    def __init__(self, cv): object.__setattr__(self, "_cv", cv); object.__setattr__(self, "read", set())

    def __getattribute__(self, name):
        if sys._getframe(1).f_code.co_filename == goo.classic.tools.__file__:
            object.__getattribute__(self, "read").add(name)
            assert name in TOOL_MAY, f"tools.py read canvas.{name}"
        return getattr(object.__getattribute__(self, "_cv"), name)

    def __setattr__(self, name, value): raise AssertionError(f"canvas.{name} set by a tool")


def test_tools_stay_in_the_tool_channel():
    cv = Taped(80, 60, ["ultramarine"], laws=NOLAWS); g = Glass(cv)
    _wash(cv)
    Knife(g, 12).stroke([(5, 20), (75, 20)], target=0.0)
    Knife(g, 12, load=oil(500)).stroke([(5, 40), (75, 40)], target=0.5)
    bl = Blotter(g, 10); bl.press(60, 30); bl.wring()
    spatter(Brush(g, 6, load=oil(300)), (60, 30), 10, 5.0, 0.5, np.random.default_rng(0))
    read = object.__getattribute__(g, "read")
    assert read <= TOOL_MAY and {"_q", "_open", "_mark"} <= read
    assert worst_residual(cv) < 1e-9


# ---------------------------------------------------------------- one interface: twins, presses, receipts' cells
def test_twin_on_a_fork_continues_exactly():
    cv = Canvas(60, 40, ["ultramarine"])
    b = Brush(cv, 8, load={"water": 400.0, "ultramarine": 20.0}); b.stroke([(5, 10), (40, 12)])
    f = cv.fork(); t = b.twin(f)
    assert t.cv is f and t is f.brushes[0] and np.array_equal(t.profile, b.profile) and np.array_equal(t.load, b.load)
    ra, rb = b.stroke([(10, 30), (50, 25)]), t.stroke([(10, 30), (50, 25)])
    assert np.array_equal(cv.q, f.q) and np.array_equal(ra.given, rb.given) and ra.cells == rb.cells
    with pytest.raises(ValueError):
        Brush(Canvas(60, 40, ["ultramarine"]), 4).twin(f)       # not a fork of its canvas: no twin there


def test_receipt_says_which_cells_it_touched():
    cv = Canvas(60, 40, ["ochre"])
    b = Brush(cv, 6, load={"water": 300.0, "ochre": 30.0})
    r = b.stroke([(10, 20), (30, 20)])
    m = r.mask(cv.H, cv.W)
    assert m.sum() == r.cells > 0
    y0, y1, x0, x1 = r.bbox
    assert m[y0:y1, x0:x1].sum() == r.cells and m[y0, :].any() and m[y1 - 1, :].any() and m[:, x0].any() and m[:, x1 - 1].any()
    assert Receipt(cv.species).bbox is None and not Receipt(cv.species).mask(4, 4).any()
    s = r + b.stroke([(40, 5), (40, 30)])
    assert s.cells == (m | s.mask(cv.H, cv.W)).sum() >= r.cells


@pytest.mark.parametrize("make", [lambda cv: Brush(cv, 6), lambda cv: Knife(cv, 8), lambda cv: Blotter(cv, 8)],
                         ids=["brush", "knife", "blotter"])
def test_every_tool_strokes_presses_and_rinses_alike(make):
    cv = Canvas(50, 40, ["ochre"], laws=NOLAWS)
    cv.add("water", 0.4); cv.add("ochre", 0.04)
    t = make(cv)
    if not isinstance(t, Blotter): t.charge({"water": 200.0, "ochre": 20.0})
    r1 = t.stroke([(5, 10), (40, 10)], speed=2.5, pressure=1.0, steps_after=0)
    r2 = t.press(25, 28, pressure=1.0, ticks=2)
    for r in (r1, r2): assert isinstance(r, Receipt) and r.cells > 0 and r.ticks > 0
    assert r2.ticks == 2 and r2.length == 0
    assert (r1.taken.volume + r2.taken.volume) > 0               # each takes up wet paint above its target
    out = t.rinse()
    assert t.load.sum() == 0 and worst_residual(cv) < 1e-9 and out.volume >= 0
    if isinstance(t, Blotter): assert t.wring().volume == 0
    if isinstance(t, Knife): assert t.wipe().volume == 0
