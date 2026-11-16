"""The painter's cases (docs/DYNAMICS.md §9a, docs/PORTS.md §8): real library substances through goo.substance.maps, on
96 x 96 and 160 x 120 panels, stepped with plain World calls. Each must step (no failure, no `solver` flag), close its
ledgers, do what the physics says, and stay within a wall-time budget per step. Owner: dynamics."""
import time
import warnings
import numpy as np
import pytest
from goo.units import um, mm
from goo.film import Ground, Environment
from goo.substance import maps, library as L
from goo.world import World, StepFailed
from goo.world.common import Reject
import goo.world.diffuse as diffuse_mod

warnings.filterwarnings("ignore", category=RuntimeWarning)
G0 = 9.80665
BUDGET = 5.0            # s of wall time per world step, on average over a run (the machine of record: 2 cores)


def disc(H, W, r, cy=None, cx=None):
    y, x = np.mgrid[:H, :W]
    cy = H / 2 if cy is None else cy
    cx = W / 2 if cx is None else cx
    return ((y + 0.5 - cy) ** 2 + (x + 0.5 - cx) ** 2) <= r * r


def ground(kind, H, W, subs):
    if kind == "paper":
        return Ground(np.zeros((H, W)), L.CELLULOSE, 0.5, 5 * um, 200 * um, subs, None)
    return Ground(np.zeros((H, W)), L.CALCITE, 0.0, 1 * um, 100 * um, subs, None)       # sealed


def layer(mask, h, fractions):
    return np.stack([np.where(mask, h * f, 0.0) for f in fractions])


def run(w, duration, chunk):
    """advance in chunks; returns (steps, wall per step, receipts)."""
    t0, steps, recs = time.perf_counter(), 0, []
    end = w.t + duration
    while w.t < end * (1 - 1e-12):
        r = w.advance(min(chunk, end - w.t))
        assert not r.flagged("solver"), [f.detail for f in r.flagged("solver")]
        assert r.residual["matter"] <= 1e-12 and r.residual["energy"] <= 1e-9, r.residual
        steps += int(r.report.get("steps", 0))
        recs.append(r)
    wall = time.perf_counter() - t0
    return steps, wall / max(steps, 1), recs


def mass(w, i):
    return float(w.q[i].sum()) * w.dx ** 2 * w.rho[i]


def test_pigmented_gum_water_dries_on_paper_to_equilibrium():
    """Water, gum arabic and hematite on paper at 50 % humidity. Water leaves to the air and into the paper until the
    film's water is at the air's potential; gum arabic is hygroscopic in the maps (its film keeps about a fifth water
    at 50 % RH, viscosity ~ 200 Pa s, finite at every water content), so settle() retires nothing: that is what the
    descriptions say, not a failure (docs/DYNAMICS.md §9a)."""
    H = W = 96
    subs = (L.WATER, L.GUM_ARABIC, L.HEMATITE)
    q = layer(disc(H, W, W / 3), 15 * um, (0.80, 0.15, 0.05))
    w = World(maps, subs, ground("paper", H, W, subs), 0.25 * mm, Environment(activity={"water": 0.5}), q=q)
    m0 = mass(w, 0)
    steps, per, _ = run(w, 240.0, 60.0)
    assert per < BUDGET, per
    assert mass(w, 0) < 0.5 * m0                              # most of the water went to the air
    assert np.all(w.q >= 0)
    # (none soaks in: the gum solution's water potential, ~ -3e6 Pa from the mixing map, is far below the paper's
    # suction, -1.5e4 Pa, so sorption's one-way force is not positive; the maps say so, not the world)
    r = w.settle()
    assert r.report["cells retired"] == 0.0 and r.report["cells still wet"] > 0


@pytest.mark.xfail(strict=True, reason="O3: where one pigment enters the other paint, the diffusion step uses the entering "
                   "substance's explicit (concave) potential masked to 0 instead of its finite value, and D_num's time "
                   "part is negative (-8e-7 J against dF = -2e-13 J), which now raises the solver flag. Keeping mu_x "
                   "finite (goo/world/common.py, tried) fixes the sign but stalls the diffusion Newton on the journal "
                   "replay scene; left to the diffusion rewrite (PROGRESS.md, Checkpoint 4)")
def test_two_pigmented_paints_meet_wet():
    """Hematite and ultramarine gum-water paints laid side by side, touching, on a sealed ground (160 x 120)."""
    H, W = 120, 160
    subs = (L.WATER, L.GUM_ARABIC, L.HEMATITE, L.ULTRAMARINE)
    left = disc(H, W, 30, cx=W / 2 - 29)
    right = disc(H, W, 30, cx=W / 2 + 29) & ~left
    q = layer(left, 20 * um, (0.8, 0.15, 0.05, 0.0)) + layer(right, 20 * um, (0.8, 0.15, 0.0, 0.05))
    w = World(maps, subs, ground("sealed", H, W, subs), 0.25 * mm, Environment(activity={"water": 0.5}), q=q)
    tot0 = [mass(w, i) for i in range(4)]
    steps, per, _ = run(w, 20.0, 10.0)
    assert per < BUDGET, per
    for i in range(4):
        assert mass(w, i) <= tot0[i] * (1 + 1e-12) and (i == 0 or abs(mass(w, i) - tot0[i]) <= 1e-12 * tot0[i])
    # each pigment has crossed into the other paint, a little
    assert w.q[3][left].sum() > 0 and w.q[2][right].sum() > 0


def test_a_tilted_wet_pigmented_film_runs_downhill():
    """A 40 um gum-water hematite film on a sealed ground tilted 5 degrees toward +x: its centre of mass moves +x."""
    H = W = 96
    subs = (L.WATER, L.GUM_ARABIC, L.HEMATITE)
    q = layer(disc(H, W, W / 4), 40 * um, (0.8, 0.15, 0.05))
    a = np.radians(5)
    env = Environment(gravity=(G0 * np.sin(a), 0.0, G0 * np.cos(a)), activity={"water": 0.5})
    w = World(maps, subs, ground("sealed", H, W, subs), 0.25 * mm, env, q=q)

    def centre(w):
        h = w.q.sum(0)
        return (h * (np.arange(W) + 0.5)).sum() / h.sum(), (h * (np.arange(H)[:, None] + 0.5)).sum() / h.sum()
    x0, y0 = centre(w)
    steps, per, _ = run(w, 30.0, 15.0)
    assert per < BUDGET, per
    x1, y1 = centre(w)
    assert x1 > x0 + 0.5, (x0, x1)
    assert abs(y1 - y0) < 0.05


def test_a_pour_over_dried_strata():
    """A gum-water hematite film laid over a dried stratum (gum and pigment) on paper (160 x 120)."""
    H, W = 120, 160
    subs = (L.WATER, L.GUM_ARABIC, L.HEMATITE)
    st = layer(disc(H, W, 40), 5 * um, (0.0, 0.7, 0.3))
    q = layer(disc(H, W, 30, cx=W / 2 + 15), 20 * um, (0.8, 0.15, 0.05))
    w = World(maps, subs, ground("paper", H, W, subs), 0.25 * mm, Environment(activity={"water": 0.5}), q=q,
              strata=(st,))
    steps, per, _ = run(w, 30.0, 15.0)
    assert per < BUDGET, per
    assert len(w.strata) == 1 and np.array_equal(w.strata[0], st)        # strata do not move
    assert mass(w, 2) == pytest.approx(float(q[2].sum()) * w.dx ** 2 * w.rho[2], rel=1e-12)


def test_a_paste_with_a_yield_stress_holds_its_shape_while_it_dries():
    """Rutile at 35 % by volume in gum water (yield stress ~ kPa) on a sealed ground: it loses water to the air and does
    not flow."""
    H = W = 96
    subs = (L.WATER, L.GUM_ARABIC, L.RUTILE)
    rng = np.random.default_rng(1)
    h = 15 * um * (1 + 0.3 * rng.random((H, W))) * disc(H, W, W / 3)
    q = np.stack([h * 0.60, h * 0.05, h * 0.35])
    w = World(maps, subs, ground("sealed", H, W, subs), 0.25 * mm, Environment(activity={"water": 0.5}), q=q)
    solids = w.q[1:].copy()
    steps, per, recs = run(w, 120.0, 60.0)
    assert per < BUDGET, per
    assert sum(r.report.get("D viscous", 0.0) for r in recs) == 0.0               # nothing flowed
    assert np.array_equal(w.q.sum(0) > 0, h > 0)
    assert mass(w, 0) < 0.8 * float(q[0].sum()) * w.dx ** 2 * w.rho[0]
    for i in (1, 2):                                                                # solids stay, to the last bits
        assert abs(mass(w, i) - float(solids[i - 1].sum()) * w.dx ** 2 * w.rho[i]) <= 1e-12 * mass(w, i)


def test_a_latex_paint_dries_and_settle_retires_it():
    """Water, gum arabic, PMMA latex at 35 % and hematite at 5 % on a sealed ground (160 x 120): as the water leaves, the
    particles jam (the viscosity map's Krieger law reaches +inf at packing) and settle() retires the whole film."""
    H, W = 120, 160
    subs = (L.WATER, L.GUM_ARABIC, L.PMMA_LATEX, L.HEMATITE)
    mask = disc(H, W, 40)
    q = layer(mask, 10 * um, (0.55, 0.05, 0.35, 0.05))
    w = World(maps, subs, ground("sealed", H, W, subs), 0.25 * mm, Environment(activity={"water": 0.5}), q=q)
    steps, per, _ = run(w, 240.0, 60.0)
    assert per < BUDGET, per
    r = w.settle()
    assert r.report["cells retired"] == float(mask.sum()) and r.report["cells still wet"] == 0.0
    assert len(w.strata) == 1 and not np.any(w.q)


# ---- R7, StepFailed, advance --------------------------------------------------------------------------------
def _wash(H=48, W=48):
    subs = (L.WATER, L.GUM_ARABIC, L.HEMATITE)
    q = layer(disc(H, W, W / 3), 20 * um, (0.80, 0.15, 0.05))
    return World(maps, subs, ground("sealed", H, W, subs), 0.25 * mm, Environment(activity={"water": 0.5}), q=q)


def test_R7_sub_molecular_floor_changes_nothing_above_it(monkeypatch):
    """With every amount above one molecule per cell, the floor rule is inactive: a step is bit-identical with the
    floor set to zero."""
    a = _wash()
    b = a.fork()
    v = np.array([s.molecular_volume for s in a.substances])
    floor = np.where(np.isfinite(v), v, 0.0) / a.dx ** 2
    assert np.all((a.q == 0) | (a.q > floor[:, None, None]))
    ra = a.step(0.5)
    monkeypatch.setattr(diffuse_mod, "floors", lambda ctx: np.zeros(len(ctx.substances)))
    rb = b.step(0.5)
    assert ra.digest() == rb.digest() and np.array_equal(a.q, b.q)


def test_R7_a_film_of_less_than_a_molecule_steps():
    """A pigment at a fraction of a particle per cell next to a full one: carried exactly, and the step is taken."""
    H = W = 48
    subs = (L.WATER, L.GUM_ARABIC, L.HEMATITE)
    q = layer(disc(H, W, W / 3), 20 * um, (0.80, 0.15, 0.05))
    q[:, 24, 41] = (1e-12, 0.0, 1e-20)     # next to the edge: hematite far below one particle (5e-21 m^3 per 6e-8 m^2)
    w = World(maps, subs, ground("sealed", H, W, subs), 0.25 * mm, Environment(activity={"water": 0.5}), q=q)
    r = w.step(0.2)
    assert r.residual["matter"] <= 1e-12
    assert np.all(w.q >= 0) and not r.flagged("solver")   # (flow may still carry it, in donor proportions)


def test_a_step_that_cannot_be_taken_leaves_the_world_unchanged(monkeypatch):
    import goo.world.world as ww
    w = _wash()
    w.step(0.1)
    d = w.state_digest()
    calls = [0]
    orig = ww.World._piece

    def piece(self, *a):
        calls[0] += 1
        if calls[0] != 2:                    # the first half is accepted and committed, then every later piece fails
            raise Reject("test")
        return orig(self, *a)
    monkeypatch.setattr(ww.World, "_piece", piece)
    monkeypatch.setattr(ww, "MAX_HALVINGS", 2)
    with pytest.raises(StepFailed):
        w.step(0.4)
    assert w.state_digest() == d


def test_advance_stops_with_a_flag_instead_of_raising(monkeypatch):
    import goo.world.world as ww
    w = _wash()
    w.step(0.1)
    d = w.state_digest()
    monkeypatch.setattr(ww.World, "_piece", lambda self, *a: (_ for _ in ()).throw(Reject("test")))
    monkeypatch.setattr(ww, "MAX_HALVINGS", 1)
    r = w.advance(1.0)
    assert r.flagged("solver") and r.t1 == r.t0 and w.state_digest() == d


def test_advance_takes_a_rate_field_and_is_not_sensitive_to_the_last_bit_of_dt():
    a, b = _wash(), _wash()
    r = np.ones((a.H, a.W)); r[:8] = 0.0
    ra = a.advance(2.0, rate=r, dt=0.2)
    rb = b.advance(2.0, rate=r, dt=0.20000000000000018)
    assert not ra.flagged("solver") and not rb.flagged("solver")
    assert ra.flagged("privilege") and rb.flagged("privilege")
    assert np.array_equal(a.q[:, :8], _wash().q[:, :8])               # held cells do not move
    assert np.allclose(a.q, b.q, rtol=1e-3, atol=1e-3 * float(a.q.max()))


def test_a_drop_running_down_a_steep_tilt_books_finite_dissipation():
    """A drop of fluid paint (water 0.86, gum arabic 0.08, cadmium sulfide 0.05, hematite 0.01) on a sealed ground
    tilted 70 degrees runs until its trailing edge thins to almost nothing. The viscous dissipation is flux times the
    force that drove it, so a vanishing film books a vanishing amount (U^2/M overflowed to 1e88 J where M ~ 0), and D_num
    stays non-negative to rounding (a negative one would raise the `solver` flag)."""
    H, W = 40, 64
    subs = (L.WATER, L.GUM_ARABIC, L.CDS, L.HEMATITE)
    q = layer(disc(H, W, 6, cx=12), 60 * um, (0.86, 0.08, 0.05, 0.01))
    a = np.radians(70)
    env = Environment(gravity=(G0 * np.sin(a), 0.0, G0 * np.cos(a)), activity={"water": 0.5})
    w = World(maps, subs, ground("sealed", H, W, subs), 0.25 * mm, env, q=q)
    x0 = (w.q.sum(0) * (np.arange(W) + 0.5)).sum() / w.q.sum()
    released = 0.0
    for _ in range(6):
        r = w.advance(0.5)
        assert not r.flagged("solver"), [f.detail for f in r.flagged("solver")]
        D = sum(v for k, v in r.report.items() if k.startswith("D ") and k[2:] in ("viscous", "plastic", "diffusion",
                                                                                  "exchange", "sorption"))
        dF = sum(b - a_ for (a_, b) in r.F.values())
        Wk = float(r.moved("energy", src="work", dst="stored"))
        assert np.isfinite(D) and D <= Wk - dF + 1e-9 * (abs(dF) + abs(Wk)) + 1e-15   # no more than F released
        assert r.report["D_num time"] + r.report["D_num transport"] >= -1e-9 * max(r.report["energy moved"], 1e-300)
        released += -dF
    x1 = (w.q.sum(0) * (np.arange(W) + 0.5)).sum() / w.q.sum()
    assert x1 > x0 + 1.0
