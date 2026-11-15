"""goo.ports: paints, vessels, sources, tape, environment, clocks, settle (SPEC §8; docs/PORTS.md). Small panels of
real library substances on real grounds. Tests may read the world; painters may not (tests/journal)."""
import ast
import pathlib
import warnings
import numpy as np
import pytest
from goo.ledger import Receipt
from goo.film import Environment
from goo.substance import maps, library as L
from goo.world import World
from goo.units import um, mm, uL, mL
from goo.ports import Bench, Paint, PortError, make_ground, geometry as G, GROUNDS

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = pathlib.Path(__file__).resolve().parents[2]


def bench(W=32, H=24, subs=(L.WATER, L.GUM_ARABIC), ground="sealed", humidity=0.95):
    """A small bench. The default ground is a sealed calcite panel (porosity 0) in damp air, a scene the raw world
    steps reliably; pigments drying on paper need the journal's Hessian guard (docs/PORTS.md §8)."""
    if ground == "sealed":
        g, _ = make_ground("calcite", H, W, porosity=0.0, pore_radius=1 * um, depth=100 * um)
    else:
        g, _ = make_ground(ground, H, W)
    w = World(maps, subs, g, 0.25 * mm, Environment(activity={"water": humidity}))
    b = Bench(w)
    names = [s.name for s in subs]
    rec = {"water": 0.85, "gum_arabic": 0.15} if "gum_arabic" in names else {"water": 1.0}
    b.paint("wash", rec)
    b.fill("cup", "wash", 1 * mL)
    return b


def closes(r):
    assert isinstance(r, Receipt)
    assert r.residual["matter"] <= 1e-12, r.residual
    assert r.residual.get("energy", 0.0) <= 1e-9, r.residual


# ------------------------------------------------------------------------------------------------ paints
def test_a_paint_is_a_validated_recipe_and_mixing_makes_a_new_one():
    subs = (L.WATER, L.GUM_ARABIC, L.HEMATITE)
    a = Paint.of("red", {"water": 0.8, "gum_arabic": 0.15, "hematite": 0.05}, subs)
    assert np.isclose(a.phi(subs).sum(), 1.0) and a.phi(subs)[2] == pytest.approx(0.05)
    with pytest.raises(PortError, match="sum to"):
        Paint.of("bad", {"water": 0.5}, subs)
    with pytest.raises(PortError, match="not a substance of this world"):
        Paint.of("bad", {"water": 0.5, "soot": 0.5}, subs)
    with pytest.raises(PortError, match=">= 0"):
        Paint.of("bad", {"water": 1.5, "hematite": -0.5}, subs)
    b = Paint.of("clear", {"water": 0.8, "gum_arabic": 0.2}, subs)
    m = Paint.mix("pink", {a: 1, b: 3}, subs)
    assert m.phi(subs) == pytest.approx((a.phi(subs) + 3 * b.phi(subs)) / 4)


def test_ground_presets_carry_provenance_and_any_substance_can_be_a_ground():
    for name, spec in GROUNDS.items():
        g, d = make_ground(name, 4, 5)
        assert g.solid.name == spec.solid and d["evidence"] == spec.prov.evidence
    g, d = make_ground("calcite", 4, 5, porosity=0.2, pore_radius=2 * um, depth=100 * um)
    assert g.porosity == 0.2 and d["evidence"] == "assumed"
    with pytest.raises(ValueError):
        make_ground("calcite", 4, 5)


# ------------------------------------------------------------------------------------------------ sources
def test_every_source_closes_the_matter_ledger_and_books_energy():
    b = bench()
    acts = [lambda: b.pour("cup", [[6, 12], [26, 12]], 0.3 * uL, 2.0, 1.5 * mm),
            lambda: b.flood("cup", {"rect": [0, 0, 8, 6]}, 5 * um),
            lambda: b.drop("cup", [20, 4], 0.05 * uL),
            lambda: b.drip("cup", [[4, 20], [28, 20]], 3, 0.03 * uL, 0.5),
            lambda: b.spray("cup", [[4, 8], [28, 8]], 0.1 * uL, 1.0, 1 * mm, 80 * um, seed=7)]
    for act in acts:
        before = b.world.matter.balance("body[cup]").copy()
        r = act()
        closes(r)
        laid = before - b.world.matter.balance("body[cup]")
        assert np.all(laid > 0)
        assert r.moved("matter", src="body[cup]") == pytest.approx(laid, rel=1e-12)
        assert any(e.ledger == "energy" and e.src == "work[body[cup]]" for e in r.entries)
        assert set(r.F) >= {"gravity", "capillary", "mixing"}
    assert b.closure() <= 1e-12


def test_a_pour_lays_its_volume_along_its_path():
    b = bench()
    r = b.pour("cup", [[6, 12], [26, 12]], 0.4 * uL, 0.0, 1.5 * mm)          # duration 0: laid, not yet flowed
    closes(r)
    q = b.world.q
    vol = q.sum() * b.dx ** 2
    assert vol == pytest.approx(0.4 * uL, rel=1e-12)
    x = (q.sum(0) * (np.arange(32) + 0.5)).sum() / q.sum()
    y = (q.sum(0) * (np.arange(24)[:, None] + 0.5)).sum() / q.sum()
    assert x == pytest.approx(16, abs=0.5) and y == pytest.approx(12, abs=0.2)


def test_a_vessel_runs_dry_and_says_so():
    b = bench()
    b.fill("thimble", "wash", 0.2 * uL)
    r = b.pour("thimble", [[6, 12], [26, 12]], 0.5 * uL, 1.0, 1.5 * mm)
    closes(r)
    assert r.report["laid m^3"] == pytest.approx(0.2 * uL, rel=1e-9)
    assert r.report["shortfall m^3"] == pytest.approx(0.3 * uL, rel=1e-9)
    assert b.vessels["thimble"].volume <= 1e-12 * uL
    r = b.drop("thimble", [10, 10], 0.1 * uL)
    assert r.report["shortfall m^3"] == pytest.approx(0.1 * uL) and "laid m^3" not in r.report
    assert not any(e.ledger == "matter" and e.src == "body[thimble]" for e in r.entries)
    with pytest.raises(PortError, match="one paint"):
        b.paint("other", {"water": 1.0}); b.fill("thimble", "other", 1 * uL)


def test_tape_shields_holds_back_flow_and_peels_with_what_it_caught():
    b = bench()
    region = {"rect": [0, 0, 32, 8]}
    mask = b.region(region)
    r = b.tape("edge", region)
    assert r.flagged("privilege")
    r = b.flood("cup", "all", 10 * um)
    closes(r)
    assert not np.any(b.world.q[:, mask])
    caught = b.world.matter.balance("body[edge]")
    assert caught.sum() > 0 and caught == pytest.approx(b.tapes["edge"].mass, rel=1e-14)
    expect = 10 * um * mask.sum() * b.dx ** 2 * b.rho * b.paints["wash"].phi(b.subs)
    assert caught == pytest.approx(expect, rel=1e-12)
    r = b.wait(2.0)
    closes(r)
    assert not np.any(b.world.q[:, mask]), "flow crossed into taped cells"
    (f,) = [f for f in r.flags if f.kind == "privilege" and f.subject == "painter"]
    assert f.amount == pytest.approx(r.report["rate integral"]) and f.amount < 2.0 * 32 * 24 * b.dx ** 2
    assert any("wall" in f.detail for f in r.flags if f.kind == "privilege")
    with pytest.raises(PortError, match="wet"):
        b.tape("over wet", {"rect": [0, 10, 4, 14]})
    r = b.peel("edge")
    assert r.report["peeled kg"] == pytest.approx(float(caught.sum()))
    assert b.rate_field() is None
    assert b.world.matter.balance("body[edge]") == pytest.approx(caught) and b.closure() <= 1e-12


def test_spray_is_reproducible_from_its_seed_and_books_overspray():
    def run(seed):
        b = bench()
        r = b.spray("cup", [[2, 2], [30, 2]], 0.2 * uL, 1.0, 1.5 * mm, 80 * um, seed=seed)
        return b, r
    (b1, r1), (b2, r2), (b3, r3) = run(11), run(11), run(12)
    assert b1.world.digest() == b2.world.digest() and r1.digest() == r2.digest()
    assert b1.world.digest() != b3.world.digest()
    for b, r in ((b1, r1), (b3, r3)):
        closes(r)
        over = b.world.matter.balance("reservoir[overspray]")
        assert over.sum() > 0 and over == pytest.approx(b.reservoirs["overspray"], rel=1e-14)
        assert r.report["droplets off the panel"] > 0
        n = r.report["droplets"]
        assert n == round(0.2 * uL / (np.pi * (80 * um) ** 3 / 6))
        assert over.sum() == pytest.approx(r.moved("matter", src="body[cup]").sum() * r.report["droplets off the panel"] / n,
                                           rel=1e-9)
    with pytest.raises(PortError, match="seed"):
        bench().spray("cup", [[2, 2], [30, 2]], 0.2 * uL, 1.0, 1.5 * mm, 80 * um, seed=None)


def test_tilt_makes_a_pour_run_downhill():
    def centre(b):
        h = b.world.q.sum(0)
        return (h * (np.arange(b.W) + 0.5)).sum() / h.sum(), (h * (np.arange(b.H)[:, None] + 0.5)).sum() / h.sum()
    flat, tilted = bench(W=40, H=24, subs=(L.WATER,), humidity=0.99), bench(W=40, H=24, subs=(L.WATER,), humidity=0.99)
    for b in (flat, tilted):
        b.pour("cup", [20, 12], 0.4 * uL, 0.5, 2 * mm)
    r = tilted.tilt(np.radians(30), 0.0)
    closes(r)
    assert tilted.world.env.gravity[0] > 0 and tilted.world.env.gravity[1] == 0
    for b in (flat, tilted):
        closes(b.wait(1.0, steps=4))
    (x0, y0), (x1, y1) = centre(flat), centre(tilted)
    assert x1 > x0 + 0.2, (x0, x1)
    assert abs(y1 - y0) < 0.05
    closes(tilted.level())
    assert tilted.world.env.gravity[:2] == (0.0, 0.0)


def test_air_is_an_act_with_a_receipt():
    b = bench()
    b.flood("cup", {"disc": [16, 12, 5]}, 10 * um)
    r = b.air({"water": 0.9}, T=295.0)
    closes(r)
    assert b.world.env.activity["water"] == 0.9 and b.world.env.T == 295.0
    assert any(e.src == "work[environment]" for e in r.entries)
    with pytest.raises(PortError, match="activity"):
        b.air({"water": 0.0})


def test_hold_freezes_a_region_exactly_and_is_flagged():
    b = bench()
    b.flood("cup", {"rect": [4, 4, 28, 20]}, 15 * um)
    hold = {"rect": [4, 4, 14, 20]}
    m = b.region(hold)
    r = b.hold(hold)
    assert r.flagged("privilege")
    before = b.world.q.copy(), b.world.ground.held.copy()
    r = b.wait(3.0, steps=3)
    closes(r)
    assert np.array_equal(b.world.q[:, m], before[0][:, m]) and np.array_equal(b.world.ground.held[:, m], before[1][:, m])
    assert not np.array_equal(b.world.q[:, ~m], before[0][:, ~m])
    priv = [f for f in r.flags if f.kind == "privilege" and f.subject == "painter"]
    assert len(priv) == 1 and priv[0].amount == pytest.approx(3.0 * (~m).sum() * b.dx ** 2)
    b.release()
    assert b.rate_field() is None


def test_a_local_rate_is_a_privilege_and_a_negative_rate_is_refused():
    b = bench()
    b.flood("cup", {"rect": [4, 4, 28, 20]}, 15 * um)
    r = b.wait(1.0, rate=0.5, where={"rect": [0, 0, 16, 24]}, steps=2)
    closes(r)
    f = [f for f in r.flags if f.kind == "privilege" and f.subject == "painter"][0]
    assert f.amount == pytest.approx(1.0 * (16 * 24 * 0.5 + 16 * 24) * b.dx ** 2)
    d = b.world.digest()
    with pytest.raises(PortError, match="backwards"):
        b.wait(1.0, rate=-1.0)
    assert b.world.digest() == d


def test_settle_retires_what_cannot_flow_and_force_is_a_privilege():
    b = bench()
    b.flood("cup", {"rect": [4, 4, 28, 20]}, 10 * um)
    r = b.settle()                                           # wet water-gum film can flow: nothing retired
    closes(r)
    assert r.report["cells retired"] == 0 and not r.flagged("privilege")
    r = b.settle(where={"rect": [4, 4, 16, 20]}, force=True)
    closes(r)
    assert r.report["cells retired"] > 0
    assert any(f.kind == "privilege" and "could still flow" in f.detail for f in r.flags)
    assert len(b.world.strata) == 1
    assert np.all(b.world.q[:, b.region({"rect": [4, 4, 16, 20]})] == 0)


def test_a_failing_world_step_is_never_half_taken():
    """A step the world gives up on is taken on a fork and dropped (bench._step), so the bench's world is unchanged."""
    b = bench()
    b.flood("cup", {"rect": [4, 4, 28, 20]}, 10 * um)
    d = b.world.digest()
    orig = type(b.world).step

    def bad(self, dt, **kw):
        orig(self, dt / 2, **kw)                             # commits half, then fails, as World.step can
        raise RuntimeError("step could not be accepted")
    type(b.world).step = bad
    try:
        with pytest.raises(PortError, match="could not take a step"):
            b.wait(1.0, steps=1)
    finally:
        type(b.world).step = orig
    assert b.world.digest() == d


def test_a_wash_dries_on_paper():
    """Gum water (no pigment) on paper at 50 % humidity for three minutes, with the raw maps (no Hessian guard)."""
    b = bench(ground="paper", humidity=0.5)
    b.pour("cup", [[6, 12], [26, 12]], 0.5 * uL, 2.0, 2 * mm)
    for _ in range(3):
        closes(b.wait(60.0))


# ------------------------------------------------------------------------------------------------ geometry
def test_regions_in_cells_and_metres():
    H, W, dx = 10, 12, 0.5 * mm
    r = G.region({"rect": [2, 1, 5, 4]}, H, W, dx)
    assert r.sum() == 9 and r[1, 2] and not r[4, 2]
    rm = G.region({"rect": [1 * mm, 0.5 * mm, 2.5 * mm, 2 * mm], "units": "m"}, H, W, dx)
    assert np.array_equal(r, rm)
    d = G.region({"disc": [6, 5, 2]}, H, W, dx)
    assert d[4, 5] and not d[0, 0]
    band = G.region({"band": {"cols": [3, 6]}}, H, W, dx)
    assert band.sum() == 3 * H
    poly = G.region({"polygon": [[0, 0], [12, 0], [0, 10]]}, H, W, dx)
    assert poly[0, 0] and not poly[9, 11]
    u = G.region({"union": [{"rect": [0, 0, 2, 2]}, {"rect": [10, 8, 12, 10]}]}, H, W, dx)
    assert u.sum() == 8
    diff = G.region({"difference": ["all", {"rect": [0, 0, 6, 10]}]}, H, W, dx)
    assert diff.sum() == 60
    inter = G.region({"intersection": [{"band": {"rows": [0, 5]}}, {"band": {"cols": [0, 4]}}]}, H, W, dx)
    assert inter.sum() == 20
    with pytest.raises(G.GeometryError):
        G.region({"blob": 1}, H, W, dx)


def test_paths_and_footprints():
    p = G.path({"points": [[0, 0], [10, 0], [10, 10]], "smooth": 2}, 1.0)
    assert np.array_equal(p[0], [0, 0]) and np.array_equal(p[-1], [10, 10]) and len(p) > 3
    assert np.array_equal(G.path([3, 4], 1.0), [[3, 4]])
    w, off = G.footprint(np.array([5.0, 5.0]), np.array([15.0, 5.0]), 2.0, 12, 20)
    assert off == 0 and w.sum() == pytest.approx(1.0, rel=1e-14)
    w, off = G.footprint(np.array([0.0, 5.0]), np.array([0.0, 5.0]), 3.0, 12, 20)
    assert 0.3 < off < 0.7 and w.sum() + off == pytest.approx(1.0, rel=1e-14)
    w, off = G.footprint(np.array([3.3, 2.2]), np.array([3.3, 2.2]), 0.1, 12, 20)
    assert w[2, 3] == 1.0 and off == 0


# ------------------------------------------------------------------------------------------------ rules
CODE = [p for d in ("goo/ports", "goo/journal") for p in (ROOT / d).rglob("*.py")]


def test_no_medium_names_and_no_branch_on_a_substance_name():
    names = set(L.BY_NAME)
    for p in CODE:
        src = p.read_text()
        for word in ("watercolour", "watercolor", "gouache", "acrylic", "oil paint"):
            assert word not in src.lower(), (p, word)
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Compare):
                for c in [node.left] + node.comparators:
                    assert not (isinstance(c, ast.Constant) and c.value in names), f"{p}: compares with {c.value!r}"
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                assert not any("classic" in m for m in mods), p
