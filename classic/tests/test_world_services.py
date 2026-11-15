"""World services and the optics of wetness (classic/SPEC.md §4.9, §6): fork, sheen, touch_dry, the frisket, acrylic."""
import pickle
import tracemalloc
import numpy as np
import pytest
from goo.classic import Canvas, Brush, Stylus, PIGMENTS


def worst_residual(cv):
    led = cv.ledger()
    return max(abs(v["residual"]) / max(1.0, v["loaded"]) for v in led["species"].values())


class Dabber:
    """A minimal tool that honours the frisket (`cv._open`): a round footprint that drags and lays paint from its
    load toward a target thickness. It uses the tool channel and books what it carries, like a Brush."""

    def __init__(self, cv, r, load):
        self.cv, self.r = cv, r
        self.load = np.zeros(len(cv.species))
        for k, v in load.items(): self.load[cv.idx(k)] += v
        cv.led["loaded"] += self.load; cv.brushes.append(self)

    def touch(self, x, y, dx, dy, target):
        cv, r = self.cv, self.r
        y0, y1, x0, x1 = max(0, int(y - r - 1)), min(cv.H, int(y + r + 2)), max(0, int(x - r - 1)), min(cv.W, int(x + r + 2))
        yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
        foot = (np.hypot(yy - y, xx - x) < r).astype(float)
        op = cv._open(y0, y1, x0, x1)
        if op is not None: foot = foot * op
        cv._U[0, y0:y1, x0:x1] += foot * dx; cv._U[1, y0:y1, x0:x1] += foot * dy; cv._mark_tool(y0, y1, x0, x1)
        q = cv._q[:, y0:y1, x0:x1]
        give = np.clip(target - q.sum(0), 0, None) * foot * 0.5; L = self.load.sum()
        if L <= 1e-12 or give.sum() <= 0: return
        add = (self.load / L)[:, None, None] * (give * min(1.0, 0.25 * L / give.sum()))[None]
        q += add; self.load -= add.sum(axis=(1, 2)); np.maximum(self.load, 0, out=self.load); cv._mark(y0, y1, x0, x1)

    def stroke(self, path, speed=2.0, target=0.5):
        for (xa, ya), (xb, yb) in zip(path, path[1:]):
            n = max(1, int(np.ceil(np.hypot(xb - xa, yb - ya) / speed)))
            for i in range(n):
                t = (i + 1) / n
                self.touch(xa + (xb - xa) * t, ya + (yb - ya) * t, (xb - xa) / n, (yb - ya) / n, target); self.cv.step()


def disc(cv, y, x, r):
    yy, xx = np.mgrid[0:cv.H, 0:cv.W] + 0.5
    return np.hypot(yy - y, xx - x) < r


STATE = ("_q", "_kappa", "_store", "zb", "g", "_U", "_V")


def assert_same(a, b):
    for name in STATE: assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert len(a.strata) == len(b.strata) and all(np.array_equal(x, y) for x, y in zip(a.strata, b.strata))
    assert (a.t, a.substeps, a.capped, a.pending, a._box, a._tw) == (b.t, b.substeps, b.capped, b.pending, b._box, b._tw)
    assert a.ledger() == b.ledger() and a.L == b.L
    assert set(a.led) == set(b.led) and all(np.array_equal(a.led[k], b.led[k]) for k in a.led)
    assert [type(x) for x in a.brushes] == [type(y) for y in b.brushes]
    assert all(np.array_equal(x.load, y.load) for x, y in zip(a.brushes, b.brushes))
    assert all(x.cv is a for x in a.brushes) and all(y.cv is b for y in b.brushes)


# ---- fork

def _first_half():
    cv = Canvas(110, 80, ["ultramarine", "sienna", "white"], seed=3, laws=dict(lazy=4, cure_rate=0.003, store_evap=2e-4))
    w = Brush(cv, 16, bristles=6, load=Brush.paint("water", 0.07, 1500, {"sienna": 2, "ultramarine": 1}))
    w.stroke([(8, 15), (60, 22), (100, 12)], speed=3.0, target=0.5)
    cv.wait(60, evap_boost=6.0); cv.fix()
    o = Brush(cv, 10, load=Brush.paint("oil", 0.5, 1200, {"white": 2, "ultramarine": 1}))
    o.stroke([(10, 50), (100, 60)], target=0.8)
    Stylus(cv, 3).stroke([(10, 55), (60, 58)], speed=1.5)
    for _ in range(3): cv.step()                                          # leaves slow-law ticks pending
    return cv


def _second_half(cv):
    w, o, st = cv.brushes
    o.stroke([(50, 70), (55, 30)], target=0.8); st.stroke([(20, 40), (90, 45)], speed=1.5)
    w.charge(Brush.paint("water", 0.05, 900, {"ultramarine": 1})); w.stroke([(5, 75), (60, 70)], target=0.4)
    cv.wait(50, evap_boost=4.0, cure_boost=8.0); cv.fix(cure_min=0.4)
    for _ in range(7): cv.step()
    return cv


def test_fork_continues_bit_identically():
    ref = _second_half(_first_half())                                     # never forked
    cv = _first_half(); assert cv.pending > 0
    f = cv.fork()
    assert f is not cv and len(f.brushes) == 3 and all(fb is not b for fb, b in zip(f.brushes, cv.brushes))
    assert all(np.array_equal(fb.profile, b.profile) and type(fb) is type(b) for fb, b in zip(f.brushes, cv.brushes))
    assert_same(f, cv)
    _second_half(cv); _second_half(f)
    assert_same(cv, ref); assert_same(f, ref)
    assert worst_residual(f) < 1e-9


def test_fork_diverges_independently():
    ref = _second_half(_first_half())
    cv = _first_half(); f = cv.fork()
    # the fork goes its own way: more paint, a fix, a recharge
    o = f.brushes[1]; o.charge(Brush.paint("oil", 0.5, 800, {"sienna": 1})); o.stroke([(5, 5), (105, 75)], target=1.0)
    f.wait(30, cure_boost=10.0); f.fix(cure_min=0.3)
    f.add("white", 0.01)
    _second_half(cv)
    assert_same(cv, ref)                                                  # the fork did nothing to the original
    assert worst_residual(f) < 1e-9 and worst_residual(cv) < 1e-9
    assert not np.array_equal(f._q, cv._q)
    # nor the original to the fork: continuing the original further leaves a fresh fork's twin untouched
    g = f.fork(); _second_half(f); _second_half(g)
    assert_same(f, g)


def test_fork_of_a_fork_and_lazy_ticks():
    ref = _second_half(_first_half())
    cv = _first_half(); ff = cv.fork().fork()
    assert ff.pending == cv.pending > 0
    _second_half(ff); assert_same(ff, ref)


def test_fork_shares_strata_does_not_copy_them():
    cv = Canvas(120, 120, ["sienna", "ultramarine", "white"], laws=dict(evaporation=0, absorption=0))
    for k in range(30):
        cv.add(["sienna", "ultramarine", "white"][k % 3], 0.01); cv.fix()
    assert len(cv.strata) == 30 and cv.strata.nbytes > 10e6
    tracemalloc.start()
    f = cv.fork()
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    assert peak < 0.1 * cv.strata.nbytes, peak
    assert all(f.strata.block(i)[4] is cv.strata.block(i)[4] for i in range(30))
    assert f.strata is not cv.strata
    f.add("white", 0.02); f.fix()                                          # a new stratum in the fork only
    assert len(f.strata) == 31 and len(cv.strata) == 30
    assert not np.array_equal(f.zb, cv.zb) and worst_residual(f) < 1e-9 and worst_residual(cv) < 1e-9


# ---- optics of wetness: pure

def test_sheen_and_touch_dry_are_pure():
    cv = _first_half(); assert cv.pending > 0
    before = pickle.dumps(cv)
    s = cv.sheen(); td = cv.touch_dry(); td2 = cv.touch_dry(cure_min=0.3); cv.finish(); cv.reflectance()
    assert pickle.dumps(cv) == before
    assert s.shape == td.shape == (cv.H, cv.W) and td.dtype == bool and 0 <= s.min() and s.max() <= 1
    assert (td2 | ~td).all()                                               # a lower cure_min is a weaker test


def test_sheen_wash_glistens_then_satin_then_matte():
    cv = Canvas(60, 50, ["ultramarine"], laws=dict(store_evap=2e-4))
    assert not cv.sheen().any()                                            # bare dry ground: 0
    d = disc(cv, 25, 30, 12)
    for k, v in Brush.paint("water", 0.06, 0.4, {"ultramarine": 1}).items(): cv.add(k, v * d)
    seq = []
    for _ in range(3000):
        seq.append(cv.sheen()[25, 30]); cv.step()
        if seq[-1] == 0: break
    seq = np.array(seq)
    assert seq[0] == 1.0 and seq[-1] == 0.0
    first_satin = np.argmax(seq < 1); first_matte = np.argmax(seq == 0)
    assert 0 < first_satin < first_matte and (seq[first_satin:first_matte] > 0).all() and (seq[first_satin:first_matte] <= 0.5 + 1e-12).sum() > 50
    assert (np.diff(seq) <= 0).all()
    assert not cv.sheen()[~disc(cv, 25, 30, 16)].any()


@pytest.mark.parametrize("absorption", [0.0, 1.0])
def test_sheen_never_rises_as_water_leaves(absorption):
    cv = Canvas(70, 50, ["ultramarine", "white"], laws=dict(flow=0, drag=0, diffusion=0, absorption=absorption, store_evap=3e-4))
    yy, xx = np.mgrid[0:50, 0:70]
    cv.add("water", 0.3 * np.exp(-((xx - 35) ** 2 + (yy - 25) ** 2) / 150)); cv.add("ultramarine", 0.02 * (xx > 30))
    cv.add("white", 0.03 * (yy > 25) * (xx < 50))
    prev = cv.sheen(); seen = set()
    for _ in range(1500):
        cv.step(); s = cv.sheen()
        assert (s <= prev).all()
        seen.update(np.unique(np.digitize(s, [1e-12, 0.5, 1 - 1e-12]))); prev = s
    assert seen >= {0, 1, 3}                                              # dry, satin and glistening all seen


def test_wet_oil_looks_wet_until_cured_to_the_touch():
    cv = Canvas(40, 30, ["white"], laws=dict(cure_rate=0.01, evaporation=0, absorption=0))
    for k, v in Brush.paint("oil", 0.5, 0.3, {"white": 1}).items(): cv.add(k, np.full((30, 40), v))
    assert (cv.sheen() == 1).all() and not cv.touch_dry().any()
    while cv._kappa.min() < 0.8:
        assert (cv.sheen() == 1).all(); cv.wait(20, cure_boost=10.0)
    assert not cv.sheen().any() and cv.touch_dry().all() and (cv.finish() > 0.9).all()   # dry, still glossy


def _check_touch_dry_against_fix(cv, cure_min):
    td = cv.touch_dry(cure_min)
    f = cv.fork(); f.flush(); h0 = f._q.sum(0)
    ret = f.fix(cure_min); h1 = f._q.sum(0)
    fixed = (h0 > 0) & (h1 == 0)
    assert np.array_equal(td & (h0 > 0), fixed) and td[h0 <= 0].all()
    assert ret == float(fixed.sum()) / max(1, (h0 > 0).sum())
    return fixed.sum(), (h0 > 0).sum()


@pytest.mark.parametrize("lazy", [1, 5])
def test_touch_dry_is_what_fix_would_fix(lazy):
    cv = Canvas(100, 70, ["ultramarine", "ochre", "white"], laws=dict(cure_rate=0.004, lazy=lazy))
    Brush(cv, 14, load=Brush.paint("water", 0.08, 1400, {"ultramarine": 1})).stroke([(10, 20), (90, 30)])
    o = Brush(cv, 10, load=Brush.paint("oil", 0.5, 1600, {"ochre": 2, "white": 1})); o.stroke([(15, 55), (85, 50)], target=0.9)
    cv.add("ochre", 0.01 * disc(cv, 60, 20, 5))
    partial = 0
    for k in range(8):
        for _ in range(25): cv.step()
        if k % 2: cv.wait(30, evap_boost=3.0, cure_boost=6.0)
        for cm in (0.8, 0.3):
            n, m = _check_touch_dry_against_fix(cv, cm); partial += 0 < n < m
    assert partial >= 4


# ---- the frisket

def _runny(cv, region, vol=0.4):
    for k, v in Brush.paint("water", 0.05, vol, {"ultramarine": 1}).items(): cv.add(k, v * region)


def test_open_and_mask_rules():
    cv = Canvas(50, 40, ["ultramarine"])
    assert cv._open(0, 40, 0, 50) is None
    d = disc(cv, 20, 25, 6); _runny(cv, disc(cv, 20, 10, 4))
    cv.mask(d)
    op = cv._open(10, 30, 15, 35)
    assert op.dtype == float and op.shape == (20, 20) and np.array_equal(op, (~d[10:30, 15:35]).astype(float))
    with pytest.raises(ValueError, match="touch-dry"): cv.mask(disc(cv, 20, 10, 2))   # wet paint can't be taped
    with pytest.raises(ValueError): cv.mask(np.ones((3, 3), bool))
    cv.unmask(d); assert cv._open(0, 40, 0, 50) is None and cv._mask is None


def test_wash_beside_a_mask_does_not_creep_in():
    """Dry, unfixed pigment beside a wet wash for many ticks: without tape the wash's colour creeps into it
    (diffusion among the solids); under tape not a molecule moves either way."""
    def run(masked):
        cv = Canvas(70, 50, ["ultramarine", "ochre"], laws=dict(evaporation=0, absorption=0))
        tape = np.zeros((50, 70), bool); tape[:, 35:45] = True
        cv.add("ochre", 0.02 * tape)                                       # dry pigment, not fixed
        if masked: cv.mask(tape)
        wash = np.zeros((50, 70), bool); wash[10:40, 20:35] = True; _runny(cv, wash, 0.5)
        q0 = cv._q[:, tape].copy()
        for _ in range(400):
            cv.step(); assert cv._q.min() >= 0
        assert worst_residual(cv) < 1e-9
        return cv, tape, q0
    cv, tape, q0 = run(True)
    assert np.array_equal(cv._q[:, tape], q0)
    free, tape, q0 = run(False)
    assert free._q[2][tape].sum() > 0 and not np.array_equal(free._q[:, tape], q0)


def test_flood_stops_at_the_mask_edge_and_unmask_lets_it_in():
    tape = np.zeros((50, 70), bool); tape[:, 35:45] = True
    def run(masked):
        cv = Canvas(70, 50, ["ultramarine"], laws=dict(eta0=0.01, evaporation=0, absorption=0))
        if masked: cv.mask(tape)
        blk = np.zeros((50, 70), bool); blk[15:35, 25:35] = True; _runny(cv, blk, 30.0)   # a flood against the tape
        for _ in range(150):
            cv.step(); assert cv._q.min() >= 0
        assert worst_residual(cv) < 1e-9 and cv.capped == 0
        return cv
    cv = run(True)
    assert (cv.h > 1e-3).sum() > 500                                       # it spread everywhere else
    assert not cv._q[:, tape].any()                                        # and stopped at the tape's edge
    assert cv.h[15:35, 34].mean() > 5 and cv.h[15:35, 34].mean() >= cv.h[15:35, 30].mean() - 1e-9   # standing against it
    assert run(False)._q[:, tape].sum() > 100                              # without tape it runs in
    cv.unmask(tape)
    for _ in range(150): cv.step()
    assert cv._q[:, tape].sum() > 10 and worst_residual(cv) < 1e-9 and cv._q.min() >= 0   # tape off: it runs in


def test_tools_lay_nothing_under_the_mask():
    cv = Canvas(90, 60, ["ultramarine", "white"], laws=dict(evaporation=0, absorption=0))
    d = disc(cv, 30, 45, 9); cv.mask(d)
    Dabber(cv, 7, Brush.paint("water", 0.1, 600, {"ultramarine": 1})).stroke([(5, 30), (85, 30)], target=0.5)
    for _ in range(10): cv.step()
    assert not cv._q[:, d].any()
    assert cv.h[30, 20] > 0.05 and cv.h[30, 70] > 0.05                     # laid on both sides of the disc
    for k, v in Brush.paint("oil", 0.5, 0.6, {"white": 1}).items(): cv.add(k, v * (np.mgrid[0:60, 0:90][1] < 30))
    Stylus(cv, 6).stroke([(5, 30), (85, 30)], speed=1.5)                   # a tool dragging wet paint over the tape
    for _ in range(10): cv.step()
    assert not cv._q[:, d].any() and cv._q.min() >= 0 and worst_residual(cv) < 1e-9


def test_nothing_changes_under_the_mask():
    """Dry, unfixed paint, water in the ground and cure: evaporation, absorption, curing and colour all leave
    masked cells alone, even with wet paint of another colour against them."""
    cv = Canvas(60, 40, ["ultramarine", "ochre", "white"], laws=dict(cure_rate=0.02, store_evap=1e-4))
    left = np.zeros((40, 60), bool); left[:, :30] = True
    cv.add("ochre", 0.02 * left); cv.add("water", 0.2 * left)
    oil = np.zeros((40, 60), bool); oil[5:15, 40:50] = True
    for k, v in Brush.paint("oil", 0.5, 0.3, {"white": 1}).items(): cv.add(k, v * oil)
    while cv._q[0].max() > 1e-6 or cv._kappa[oil].min() < 0.8: cv.wait(20, evap_boost=4.0, cure_boost=4.0)
    tape = left | oil
    assert cv._store[left].min() > 0 and cv.touch_dry()[tape].all()
    cv.mask(tape)
    q0, s0, k0 = cv._q[:, tape].copy(), cv._store[tape].copy(), cv._kappa[tape].copy()
    _runny(cv, ~tape & ~disc(cv, 10, 45, 9), 0.3)
    for _ in range(200): cv.step()
    cv.wait(50, evap_boost=4.0, cure_boost=4.0)
    assert np.array_equal(cv._q[:, tape], q0) and np.array_equal(cv._store[tape], s0) and np.array_equal(cv._kappa[tape], k0)
    assert worst_residual(cv) < 1e-9
    cv.unmask(); cv.wait(200, evap_boost=4.0)
    assert cv._store[left].max() < s0.max()                               # untaped, the ground dries again


def test_no_mask_is_todays_canvas():
    def run(touch):
        cv = Canvas(60, 40, ["ultramarine"], laws=dict(lazy=3))
        b = Brush(cv, 8, load=Brush.paint("water", 0.1, 600, {"ultramarine": 1}))
        if touch: cv.mask(disc(cv, 20, 50, 4)); cv.unmask(disc(cv, 20, 50, 4))
        b.stroke([(5, 20), (55, 22)])
        for _ in range(20): cv.step()
        return cv
    a, b = run(False), run(True)
    assert b._mask is None
    assert_same(a, b)


def test_acrylic_binder_is_clear():
    p = PIGMENTS["acrylic"]
    assert all(0.004 <= k <= 0.008 for k in p["K"]) and p["S"] == [0.02] * 4
    cv = Canvas(20, 20, ["acrylic"]); cv.add("acrylic", 0.3); cv.add("water", 0.3)
    R = cv.reflectance()[10, 10]
    assert np.all(np.abs(R - np.array([0.93, 0.92, 0.88, 0.90])) < 0.06)


# ---------------------------------------------------------------- fix(region): fixing part of the panel
def test_fix_region_fixes_only_inside_it():
    def panel():
        cv = Canvas(40, 30, ["ochre"], laws=dict(evaporation=0, absorption=0, flow=0, diffusion=0))
        cv.add("ochre", 0.1)                                     # dry, lean paint everywhere
        return cv
    a, b = panel(), panel()
    m = np.zeros((30, 40), bool); m[5:15, 10:30] = True
    assert a.fix(region=m) == 1.0                                # every wet cell of the region fixed
    assert len(a.strata) == 1 and np.array_equal(a.strata[0][2] > 0, m)
    assert np.allclose(a.q[2][m], 0) and np.allclose(a.q[2][~m], 0.1)
    assert worst_residual(a) < 1e-9
    c, d = panel(), panel()                                      # the whole panel as a region ...
    full, whole = c.fix(), d.fix(region=np.ones((30, 40), bool))
    assert full == whole == 1.0 and np.array_equal(c.strata[0], d.strata[0]) and c.ledger() == d.ledger()   # ... is fix()
    with pytest.raises(ValueError):
        panel().fix(region=np.ones((3, 3), bool))


def test_fork_rebinds_brushes_held_through_a_proxy():
    class Proxy:
        def __init__(self, cv): object.__setattr__(self, "_cv", cv)
        def __getattr__(self, n): return getattr(object.__getattribute__(self, "_cv"), n)
    cv = Canvas(30, 20, ["ochre"])
    b = Brush(Proxy(cv), 6, load={"water": 50.0, "ochre": 5.0})
    f = cv.fork()
    assert f.brushes[0].cv is f and b.twin(f) is f.brushes[0] and np.array_equal(f.brushes[0].load, b.load)
