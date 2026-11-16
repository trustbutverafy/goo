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
    film's water is at the air's potential. Gum arabic keeps what its isotherm says (a water fraction of about 0.16 at
    50 % RH) and vitrifies below a solvent fraction of 0.194 (docs/SUBSTANCES.md §3a: viscosity +inf), so the dried
    film is a glass and settle() retires it by law. (Before the substance maps' particle and binder rework the gum
    kept a fifth water at a finite viscosity and nothing settled.)"""
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
    wet = w.q.sum(0) > 0
    glass = wet & np.isposinf(np.asarray(maps.viscosity(w.film).value))
    r = w.settle()
    # after 240 s the rim has vitrified and the centre is still drying (measured: 748 of 3624 cells): settle retires
    # exactly the glassy cells, by law, and nothing else
    assert r.report["cells retired"] == float(glass.sum()) > 0, r.report
    assert r.report["cells still wet"] == float((wet & ~glass).sum())
    assert r.residual["matter"] <= 1e-12 and r.residual["energy"] <= 1e-9


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


# ---- advance: the noise floor and the bound on change per step (DYNAMICS.md §9a) ------------------------------
def _paste(H=48, W=64):
    """A full-panel paste with a yield stress, losing water to the air (bench/scenes.py quiet, quiet.py's scene)."""
    subs = (L.WATER, L.GUM_ARABIC, L.RUTILE)
    g = Ground(np.zeros((H, W)), L.CELLULOSE, 0.0, 5 * um, 400 * um)
    h = 30 * um * (1.0 + 0.3 * np.random.default_rng(0).random((H, W)))
    q = np.stack([h * 0.60, h * 0.05, h * 0.35])
    return World(maps, subs, g, 0.25 * mm, Environment(activity={"water": 0.5}), q=q)


def test_advance_dries_a_paste_to_equilibrium_in_few_steps_and_matches_small_steps():
    """3600 s of drying: before the noise floor, advance held dt at 0.5 s for 2000 s at equilibrium (5516 steps on
    96 x 128, docs/PERFORMANCE.md §1.2); now dt grows as the film stops changing. Measured (substance maps of
    2026-11-15, with gum's glass transition): 158 steps. The controller's tolerance is an absolute one on the state
    (ADVANCE_CHANGE * tol = 1e-4 of a cell's thickness per step), so the comparison with fixed 10 s steps (themselves
    first order) is absolute too: the mean water every 300 s agrees to 8.7e-4 of the initial water (5 % of the little
    left at 600 s, where the film passes its glass transition), the final state to 2.9e-4 of the thickest cell. The
    dried film is a glass and settle() retires all of it."""
    a, b = _paste(), _paste()
    steps, wa, wb = 0, [], []
    for _ in range(12):
        r = a.advance(300.0)
        assert not r.flagged("solver"), [f.detail for f in r.flagged("solver")]
        assert r.residual["matter"] <= 1e-12 and r.residual["energy"] <= 1e-9, r.residual
        steps += int(r.report["steps"])
        wa.append(float(a.q[0].mean()))
    while b.t < 3600.0 - 1e-9:
        b.step(10.0)
        if abs(b.t / 300.0 - round(b.t / 300.0)) < 1e-9:
            wb.append(float(b.q[0].mean()))
    assert steps <= 200, steps
    w0 = float(_paste().q[0].mean())
    assert max(abs(x - y) for x, y in zip(wa, wb)) <= 3e-3 * w0
    assert float(np.max(np.abs(a.q - b.q))) <= 1e-3 * float(b.q.sum(0).max())
    assert a._dt_next >= 256.0                                    # at equilibrium the step has grown
    r = a.settle()
    assert r.report["cells still wet"] == 0.0 and r.report["cells retired"] == float(a.H * a.W)


@pytest.mark.parametrize("kind", ["sealed", "paper"])
def test_a_particle_film_drains_its_last_water_exactly(kind):
    """Rutile with a trace of water (1 nm to 0.5 um) at 50 % RH: a particle holds no water (its potential is bounded below
    by the menisci's -Pi_e, far above the air's), so the thinnest cells give all their water in one step. The drain rule
    of a cell of several substances (DYNAMICS.md §4.3): water exactly 0 there, rutile untouched bit for bit, ledgers
    closed, D_num >= 0. (Before it, the Newton crept toward 0 by its limiter and the step failed after 24 halvings.)"""
    H = W = 8
    subs = (L.WATER, L.RUTILE)
    q = np.zeros((2, H, W)); q[1] = 3 * um; q[0] = np.linspace(1e-9, 0.5 * um, H * W).reshape(H, W)
    w = World(maps, subs, ground(kind, H, W, subs), 0.25 * mm, Environment(activity={"water": 0.5}), q=q)
    r = w.step(2.0)
    assert not r.flagged("solver"), [f.detail for f in r.flagged("solver")]
    assert r.report["drained cells"] > 0
    assert np.sum(w.q[0] == 0.0) >= r.report["drained cells"] and np.all(w.q >= 0)
    assert np.array_equal(w.q[1], q[1])
    assert r.residual["matter"] <= 1e-12 and r.residual["energy"] <= 1e-9, r.residual
    assert r.report["D_num time"] >= 0.0


def test_a_dilute_pigment_interdiffuses_with_water_at_stokes_einstein():
    """1 % rutile in water with a 10 % cosine of its fraction (no gravity, no reservoir): the mode decays at
    D_c k^2, D_c = h Lambda_p phi_p (H_pp - 2 H_pw + H_ww) the maps' collective diffusivity (Stokes-Einstein times the
    hard-sphere factor, 1.08 at 1 %), k^2 the grid's own eigenvalue. Through the mixture-mean Onsager matrix the pair
    moved at water's molecular mobility, about 1e-16 m^2/s (DYNAMICS.md §3: the pair conductance on a face whose liquid
    is one substance). Measured: the rate within 0.1 % of D_c k^2, by the ray (no Newton)."""
    from goo.film import Film
    H, W, dx, h = 4, 8, 50 * um, 20 * um
    subs = (L.WATER, L.RUTILE)
    x = (np.arange(W) + 0.5) / W
    phi = 0.01 * (1 + 0.1 * np.cos(np.pi * x))[None, :] * np.ones((H, 1))
    q = np.stack([h * (1 - phi), h * phi])
    w = World(maps, subs, ground("sealed", H, W, subs), dx, Environment(gravity=(0.0, 0.0, 0.0)), q=q)
    mode = np.cos(np.pi * x)

    def amp():
        return float(((w.q[1].mean(0) / h - 0.01) * mode).sum() / (mode * mode).sum())
    a0 = amp()
    for _ in range(10):
        r = w.step(200.0)
        assert not r.flagged("solver") and r.residual["matter"] <= 1e-12 and r.residual["energy"] <= 1e-9
    rate = -np.log(amp() / a0) / 2000.0
    f = Film(subs, np.stack([np.full((1, 1), h * 0.99), np.full((1, 1), h * 0.01)]))
    Hs = maps.mixing(f).value.hess[:, :, 0, 0]
    lam = maps.mobility(f).value[:, 0, 0]
    Dc = h * lam[1] * 0.01 * (Hs[1, 1] - 2 * Hs[0, 1] + Hs[0, 0])
    k2 = (2 / dx ** 2) * (1 - np.cos(np.pi / W))
    assert abs(rate / (Dc * k2) - 1) <= 0.02, (rate, Dc * k2)
    D_se = maps.diffusivity(f).value[1, 0, 0]
    assert 1.0 <= Dc / D_se <= 1.2                       # Stokes-Einstein, times the dilute hard-sphere factor
