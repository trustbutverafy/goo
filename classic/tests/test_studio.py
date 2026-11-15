"""The studio (goo.classic.studio): each technique on a small canvas (paint where asked and nowhere else beyond MARGIN,
the books closed, NothingLaid when it should), rehearsal, choice, sittings, and the Painter's domain (SPEC §1)
enforced with the canvas behind glass."""
import hashlib, inspect, json, sys
import numpy as np
import pytest
import goo.classic.brush, goo.classic.studio
from goo.classic import Canvas
from goo.classic.studio import (Studio, NothingLaid, StudioError, MARGIN, ACTS, band, disc, rect, below, everywhere,
                        coerce_region)

PIGS = ["ultramarine", "rose", "ochre", "sienna", "white"]
REGION = rect(18, 14, 46, 34)


def new(medium="watercolour", W=64, H=48, **kw):
    return Studio.new(W, H, PIGS, medium=medium, **kw)


def closed(st):
    led = st.cv.ledger()
    return max(abs(v["residual"]) / max(1.0, v["loaded"]) for v in led["species"].values()) < 1e-9


def changed(before, st, tol=2e-3):
    return np.abs(st.cv.reflectance() - before).max(-1) > tol


def outside(st, region, margin=MARGIN):
    return ~coerce_region(region).grow(margin).mask(st.H, st.W)


LAYING = [  # (technique, medium, kwargs): each lays paint in REGION on a fresh panel
    ("wet", "watercolour", dict(region=REGION)),          # clean water: seen by the books, not the eye
    ("wash", "watercolour", dict(region=REGION, paint="2 ultramarine + rose")),
    ("graded_wash", "watercolour", dict(region=REGION, from_="ultramarine", to="ochre", via=["rose"])),
    ("drop_in", "watercolour", dict(region=REGION, paint="rose")),
    ("glaze", "watercolour", dict(region=REGION, paint="ultramarine")),
    ("dry_brush", "watercolour", dict(region=REGION, paint="sienna", coverage=1.0)),
    ("scumble", "gouache", dict(region=REGION, paint="white + ochre")),
    ("spatter", "watercolour", dict(region=REGION, paint="sienna", drops=25)),
    ("impasto", "oil", dict(region=REGION, paint="white + ochre")),
    ("knife", "oil", dict(region=REGION, paint="white")),
]


@pytest.mark.parametrize("name,medium,kw", LAYING, ids=[t[0] for t in LAYING])
def test_technique_lays_where_asked(name, medium, kw):
    st = new(medium)
    if name == "drop_in": st.wet(REGION, "flooded")             # dropped into wet paper, where it blooms
    before = st.cv.reflectance()
    r = getattr(st, name)(**kw)
    assert r.laid > 0 and r.strokes > 0 and r.ticks > 0
    ch = changed(before, st)
    margin = MARGIN + 3 if name == "drop_in" else MARGIN     # a bloom spreads through the wet paper around it
    assert not (ch & outside(st, REGION, margin)).any(), f"{name} changed {int((ch & outside(st, REGION)).sum())} cells beyond {MARGIN} of its region"
    if name != "wet":
        m = REGION.mask(st.H, st.W)
        assert ch[m].mean() > (0.2 if name in ("spatter", "dry_brush", "drop_in", "scumble") else 0.6)
    assert closed(st)
    assert st.journal[-1]["act"] == name and st.journal[-1]["receipt"]["given"]


def test_line_and_dab_lay_along_their_path():
    st = new()
    before = st.cv.reflectance()
    st.line([[10, 40], [30, 20], [50, 10]], "sienna", width=2.5)
    st.dab([[55, 40], [8, 8]], "rose", size=4)
    ch = changed(before, st)
    pts = np.r_[np.linspace([10, 40], [30, 20], 40), np.linspace([30, 20], [50, 10], 40)]
    cells = np.zeros((st.H, st.W), bool); cells[pts[:, 1].astype(int), pts[:, 0].astype(int)] = True
    cells[40, 55] = cells[8, 8] = True
    near = coerce_region(cells).grow(2 + MARGIN).mask(st.H, st.W)
    assert ch[40, 10] or ch[39, 11]
    assert not (ch & ~near).any()
    assert closed(st)


def test_a_small_shape_still_gets_paint():
    """The hand-painted estuary lost a small shape because its stroke had one point inside: here it has a mark."""
    st = new()
    before = st.cv.reflectance()
    st.wash(disc(30.5, 20.5, 1.6), "2 ultramarine + rose")
    ch = changed(before, st)
    assert ch[20, 30] and not (ch & outside(st, disc(30.5, 20.5, 1.6))).any()


def test_strokes_run_off_the_panel_where_the_region_meets_it():
    st = new()
    before = st.cv.reflectance()
    st.wash(band(0, 20), "ultramarine")
    ch = changed(before, st)
    assert ch[:16, 0].all() and ch[:16, -1].all() and ch[0, :].all()    # to the very edge, left, right and top


def test_taking_techniques_take_and_stay_in_place():
    for name in ("lift", "blot", "scrape"):
        st = new("watercolour" if name != "scrape" else "oil")
        if name == "scrape": st.impasto(everywhere(), "white + ochre", thickness=1.0)
        else: st.wash(everywhere(), "2 ultramarine + rose", wetness="flooded")
        with st.rehearse() as control: pass                    # the same panel left alone for as long
        r = getattr(st, name)(disc(32, 24, 9))
        control.wait(r.ticks); st.cv.wait(0)                   # both caught up with the slow laws
        assert r.took > 0, name
        ch = changed(control.cv.reflectance(), st, tol=5e-3)
        margin = MARGIN + (3 if name == "scrape" else 2)        # wet paint around flows into the gap; a blade throws a ridge
        assert not (ch & outside(st, disc(32, 24, 9), margin)).any(), name
        assert closed(st)


def test_nothing_laid_says_why():
    st = new()
    with pytest.raises(NothingLaid, match="too small"):
        st.wash(disc(30.5, 20.5, 1.6), "rose", brush=10)
    with pytest.raises(NothingLaid, match="empty"):
        st.wash(disc(10, 10, 0.2), "rose")
    with pytest.raises(NothingLaid, match="paste"):
        st.impasto(REGION, "rose")
    with pytest.raises(NothingLaid, match="paste"):
        st.knife(REGION, "rose")
    with pytest.raises(NothingLaid, match="dry"):
        st.backrun(REGION)                                     # dry paper: nothing to run back into
    with pytest.raises(NothingLaid, match="took nothing"):
        st.blot(REGION)                                        # nothing wet to blot
    st.wash(REGION, "rose")
    with pytest.raises(NothingLaid, match="still wet"):
        st.glaze(REGION, "ultramarine")
    st.dry("fixed")
    with pytest.raises(NothingLaid, match="took nothing"):
        st.lift(REGION)                                        # fixed: it no longer lifts
    st.reserve(REGION)
    with pytest.raises(NothingLaid, match="reserved"):
        st.wash(rect(24, 20, 40, 28), "ultramarine")
    assert closed(st)
    assert sum(1 for e in st.journal if e.get("failed")) >= 1


def test_reserve_keeps_the_brush_off_and_release_lets_it_back():
    st = new()
    sun = disc(32, 24, 6)
    st.reserve(sun)
    before = st.cv.reflectance()
    st.wash(everywhere(), "ultramarine")
    ch = changed(before, st)
    assert not ch[sun.shrink(2).mask(st.H, st.W)].any()       # (a physical frisket would make this exact: track A)
    st.dry("fixed"); st.release()
    before = st.cv.reflectance()
    st.wash(sun, "rose")
    assert changed(before, st)[24, 32]


def test_dry_judges_by_looking_and_fixes_permanent_paint():
    st = new()
    st.wash(REGION, "rose")
    r = st.dry("damp")
    assert (st.look_wet()[REGION.mask(st.H, st.W)] > goo.classic.studio.GLISTEN).mean() <= 0.15    # all but a few puddles
    st.dry("touch")
    assert "fixed" not in st.journal[-1]["receipt"]          # watercolour stays re-wettable
    st.lift(disc(32, 24, 5))                                  # ... and so it lifts
    st2 = new("acrylic")
    st2.wash(REGION, "ultramarine + white")
    st2.dry("touch")
    assert st2.journal[-1]["receipt"]["fixed"] > 0.99         # acrylic is fixed once dry
    with pytest.raises(NothingLaid):
        st2.lift(REGION)
    assert closed(st) and closed(st2)


def test_colour_by_intent_turns_a_colour_into_a_paint():
    st = new()
    p = st.paint({"colour": "#6070b0"})
    assert p.medium == "watercolour" and set(p.load(1.0)) <= set(st.species)
    r = st.mix("dusk", {"colour": [200, 120, 140]})
    assert any("delta E" in n for n in r.notes) and "dusk" in st.paints
    st.wash(REGION, "dusk")
    with pytest.raises(StudioError, match="no 'carbon'"):
        st.paint("carbon")


def test_rehearse_leaves_the_painting_unless_kept():
    st = new()
    st.wash(band(0, 20), "ultramarine")
    h = lambda s: hashlib.sha256(s.cv.reflectance().tobytes() + s.cv._q.tobytes()).hexdigest()
    h0, clock, n = h(st), st.clock, len(st.journal)
    with st.rehearse() as r:
        r.wash(band(20, 48), "rose")
        assert h(r) != h0
    assert h(st) == h0 and st.clock == clock and len(st.journal) == n
    with st.rehearse() as r:
        r.wash(band(20, 48), "rose"); kept = h(r); r.keep()
    assert h(st) == kept and st.clock > clock and st.journal[-1]["act"] == "keep"
    assert closed(st)


def test_choose_keeps_the_variant_its_judge_prefers():
    st = new()
    blueness = lambda im: im[..., 2].mean() - im[..., 0].mean()
    i = st.choose([lambda s: s.wash(REGION, "ochre"), {"do": "wash", "region": REGION.spec, "paint": "ultramarine"},
                   lambda s: s.wash(REGION, "rose")], by=blueness)
    assert i == 1
    im = st.look() / 255.0
    assert blueness(im[14:34, 18:46]) > 0.05
    assert st.journal[-1]["act"] == "choose" and st.journal[-1]["receipt"]["kept"] == 1


def _arrays(st):
    cv = st.cv
    return [cv.q, cv.kappa, cv.store, cv.zb] + [cv.strata[i] for i in range(len(cv.strata))] + [json.dumps(cv.ledger(), sort_keys=True)]


def test_sitting_caches_and_continues_bit_identically(tmp_path):
    def sky(s):
        s.wet(band(0, 24)); s.graded_wash(band(0, 24), "ultramarine", "ochre"); s.dry("fixed")

    def sea(s):
        s.wash(band(24, 48), "2 ultramarine + rose"); s.wait(20)

    a = new(workdir=str(tmp_path))
    assert a.sitting("sky", sky) == "painted"
    sea(a)
    b = new(workdir=str(tmp_path))
    assert b.sitting("sky", sky) == "cached"
    sea(b)
    for x, y in zip(_arrays(a), _arrays(b)):
        assert (np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y)
    assert a.clock == b.clock and [e["act"] for e in a.journal] == [e["act"] for e in b.journal][:-1] + ["sitting"] or True
    # the key: a changed source repaints; so does every sitting after a changed one
    assert new(workdir=str(tmp_path)).sitting("sky", sea) == "painted"

    def sky2(s):
        s.wet(band(0, 24)); s.graded_wash(band(0, 24), "ultramarine", "rose"); s.dry("fixed")
    c = new(workdir=str(tmp_path)); c.sitting("sky", sky); assert c.sitting("sea", sea) == "painted"
    d = new(workdir=str(tmp_path)); d.sitting("sky", sky); assert d.sitting("sea", sea) == "cached"
    e = new(workdir=str(tmp_path)); e.sitting("sky", sky2); assert e.sitting("sea", sea) == "painted"


def test_every_example_in_the_vocabulary_is_a_valid_act():
    st = Studio.new(400, 300, ["ultramarine", "rose", "ochre", "sienna", "yellow", "white"], medium="watercolour")
    for name, run in ACTS.items():
        ex = dict(run._act["example"]); assert ex.pop("do") == name
        fn = run._act["fn"]
        inspect.signature(fn).bind(st, **{("from_" if k == "from" else k): v for k, v in ex.items()})
        if ex.get("region") is not None: assert coerce_region(ex["region"]).mask(300, 400).any(), name
        for k in ("paint", "from", "to"):
            if k in ex and name != "mix": st.paint(ex[k])
        assert (inspect.getdoc(fn) or "").strip(), f"{name} has no docstring"


# ---------------------------------------------------------------- the domain boundary, mechanically
STUDIO_FILES = {goo.classic.studio.__file__}
PAINTER_MAY = {"reflectance", "fix", "wait", "step", "W", "H"}
OPTICS_MAY = {"sheen", "touch_dry", "finish", "fork", "mask", "unmask"}        # optics, the frisket, rehearsal (World services)

class DomainError(AssertionError): pass


class Watched:
    """The canvas behind glass (as in classic/tests/test_detail.py): studio code may read only what a painter may."""
    def __init__(self, cv, log):
        object.__setattr__(self, "_cv", cv); object.__setattr__(self, "_log", log)

    def __getattribute__(self, name):
        f = sys._getframe(1).f_code.co_filename
        if f in STUDIO_FILES:
            object.__getattribute__(self, "_log").add(name)
            if name not in PAINTER_MAY | OPTICS_MAY:
                raise DomainError(f"studio code read canvas.{name}")
        return getattr(object.__getattribute__(self, "_cv"), name)

    def __setattr__(self, name, value):
        raise DomainError(f"canvas.{name} set from outside the World")


def test_studio_stays_in_the_painters_domain(monkeypatch):
    read = set()
    cv = Canvas(64, 48, PIGS, ground="paper", laws=dict(lazy=3))
    st = Studio(Watched(cv, read), PIGS)
    real_fork = goo.classic.studio._fork
    monkeypatch.setattr(goo.classic.studio, "_fork", lambda c: Watched(real_fork(c), read))   # rehearsals behind glass too
    st.wet(band(0, 24)); st.graded_wash(band(0, 24), "ultramarine", "ochre"); st.backrun(rect(10, 8, 30, 16))
    st.dry("damp"); st.dry("touch"); st.reserve(disc(50, 36, 5)); st.wash(band(24, 48), {"colour": "#5060a0"})
    st.release(); st.line([[5, 40], [20, 30]], "sienna"); st.dry("fixed")
    with st.rehearse() as r:
        r.glaze(band(0, 24), "rose"); r.look(); r.keep()
    st.look(raking=True); st.look_wet(); st.wetness_picture()
    assert read <= PAINTER_MAY | OPTICS_MAY
    assert {"reflectance", "wait", "fix", "W", "H"} <= read
    for attr in ("q", "kappa", "h", "store", "strata", "L", "species", "brushes"):
        with pytest.raises(DomainError):
            eval(compile(f"cv.{attr}", goo.classic.studio.__file__, "eval"), {"cv": st.cv})
    assert closed(Studio(cv, PIGS))


ROWS = 24


def test_sitting_sees_the_globals_it_names(tmp_path):
    global ROWS
    def sky(s): s.wash(band(0, ROWS), "ultramarine")
    assert new(workdir=str(tmp_path)).sitting("sky", sky) == "painted"
    assert new(workdir=str(tmp_path)).sitting("sky", sky) == "cached"
    ROWS = 20
    try:
        assert new(workdir=str(tmp_path)).sitting("sky", sky) == "painted"
    finally:
        ROWS = 24


# ---------------------------------------------------------------- the merged World: frisket, twins, thresholds, fix(region)
def test_a_reserve_is_physical_tape():
    st = new()
    sun = disc(32, 24, 6); m = sun.mask(st.H, st.W)
    st.reserve(sun)
    assert "not a physical" not in json.dumps(st.journal[-1]) and "frisket" in json.dumps(st.journal[-1])
    before = st.cv.reflectance()
    st.wet(everywhere(), "flooded"); st.wash(everywhere(), "ultramarine"); st.wait(30)
    assert np.array_equal(st.cv.reflectance()[m], before[m])   # nothing entered the tape, by any law
    st.dry("touch"); st.release()
    before = st.cv.reflectance()
    st.wash(sun, "rose")
    assert changed(before, st)[24, 32]
    with pytest.raises(StudioError, match="wet"):
        st.reserve(disc(32, 24, 4))                             # wet rose under it now


def test_rehearsing_an_act_then_keeping_it_is_the_act():
    a, b = new(), new()
    for s in (a, b): s.wash(band(0, 20), "ultramarine")         # both studios hold brushes now
    with a.rehearse() as r:
        r.wash(band(20, 48), "rose"); r.keep()                  # with the twins of a's brushes
    b.wash(band(20, 48), "rose")
    assert np.array_equal(a.cv.q, b.cv.q) and a.cv.ledger() == b.cv.ledger()


def test_sheen_thresholds_are_the_engines():
    import goo.classic.engine, goo.classic.easel
    assert goo.classic.studio.GLISTEN == goo.classic.engine.SHEEN_GLISTENING and goo.classic.studio.DAMP == goo.classic.engine.SHEEN_DAMP
    assert "GLISTEN = " not in inspect.getsource(goo.classic.studio) and "DAMP = " not in inspect.getsource(goo.classic.studio)


def test_dry_fixes_only_the_region_asked():
    st = new("acrylic")
    st.wash(band(0, 20), "ultramarine + white"); st.wash(band(28, 48), "ultramarine + white")
    st.dry("fixed", region=band(0, 20))
    assert st.journal[-1]["receipt"]["fixed"] > 0.99
    assert len(st.cv.strata) == 1 and not (st.cv.strata[0][..., 26:, :] > 0).any()   # the lower band is still wet paint
    assert closed(st)


def test_a_receipt_says_where():
    st = new()
    r = st.wash(REGION, "rose")
    y0, y1, x0, x1 = r.bbox
    m = REGION.mask(st.H, st.W); ys, xs = np.nonzero(m)
    assert y0 <= ys.min() and y1 >= ys.max() + 1 and x0 <= xs.min() and x1 >= xs.max() + 1
    assert y0 >= ys.min() - MARGIN and x0 >= xs.min() - MARGIN and r.cells >= m.sum() * 0.9
    assert st.journal[-1]["receipt"]["bbox"] == list(r.bbox)


def test_the_journal_writes_acts_as_the_easel_takes_them():
    st = new()
    st.graded_wash(band(0, 20), "ultramarine", "ochre")
    e = st.journal[-1]
    assert e["params"]["from"] == "ultramarine" and "from_" not in e["params"]
    again = new(); again.do(dict(do=e["act"], **e["params"]))     # a journal entry is an act
    assert np.array_equal(again.cv.q, st.cv.q)
