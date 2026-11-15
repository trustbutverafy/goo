"""SPEC §6, obligations O1 to O8 on the world's step, against analytic constitutive laws (SPEC §4.4)."""
import numpy as np
import pytest
from goo.units import um, mm, mPa_s, mN, m, Pa_s, k_B, T_room, g_0
from goo.world import World, Environment
from world_analytic import Sub, Analytic, flat_ground, KT_V_SMALL
from world_scenes import solver, watercolour, check_energy, ledger_identity, SUBS3, WATER, blobs

RHO_W, ETA_W, GAM_W = 998.0, 1 * mPa_s, 72 * mN / m


def _all_amounts(w):
    return [w.q, w.ground.held] + list(w.strata)


# ---- O1 conservation ----------------------------------------------------------------------------------------
def test_O1_conservation_every_substance_every_step():
    w = watercolour()
    total0 = w.matter.balance("wet") + w.matter.balance("ground")
    for k in range(6):
        r = w.step(0.25)
        assert r.residual["matter"] <= 1e-12
        if k == 2:
            r = w.retire(mask=w.q[2] > 1 * um, substances=["ochre", "gum"])
            assert r.residual["matter"] <= 1e-12
            r = w.add({"water": np.where(w.q.sum(0) > 0, 5 * um, 0.0)})
            assert r.residual["matter"] <= 1e-12
            total0 = total0 + r.moved("matter", src="origin")
    held = (w.matter.balance("wet") + w.matter.balance("ground") + w.matter.balance("reservoir[atmosphere]")
            + w.matter.held("stratum"))
    assert np.allclose(held, total0, rtol=1e-12, atol=0)
    assert w.matter.imbalance() <= 1e-15 * float(np.sum(total0))
    assert np.all(w.matter.balance("reservoir[atmosphere]")[1:] == 0)        # only the volatile left


# ---- O2 positivity ------------------------------------------------------------------------------------------
def test_O2_positivity_under_drying_draining_and_steep_fronts():
    H, W, dx = 16, 16, 0.2 * mm
    z = 20 * um * np.sin(np.arange(W) / 2.0)[None, :] * np.ones((H, 1))
    g = flat_ground(H, W, porosity=0.4, depth=30 * um, z=z)
    q = blobs(H, W, [(5, 5), (10, 10)], 3, [(60 * um,), (30 * um,)])
    con = Analytic((WATER,), eta=ETA_W, gamma=GAM_W, k=3e-14, sorb=(3e4, 5e-15), wet=(-5e-3, 1 * um))
    w = World(con, (WATER,), g, dx, Environment(gravity=(2.0, 5.0, 8.0), activity={"water": 0.3}), q=q)
    dried = False
    for _ in range(40):
        w.step(0.5)
        for a in _all_amounts(w):
            assert np.all(a >= 0)
        assert np.all(w.ground.held.sum(0) < w.ground.capacity)
        if not np.any(w.q):
            dried = True
            break
    assert dried, "the film should have dried completely, cells drained to exactly zero"


# ---- O3 energy ----------------------------------------------------------------------------------------------
def test_O3_energy_identity_and_signs_watercolour():
    """dF = W - D_mech - D_num with every D_mech >= 0 and D_num >= 0 (both its parts), step by step."""
    w = watercolour()
    for _ in range(8):
        r = w.step(0.25)
        check_energy(r)
        tol = max(1e-12, 1e3 * r.report.get("solver tolerance", 0.0))
        assert r.report["D_num transport"] >= -tol * r.report["energy moved"]
        assert r.report["D_num time"] >= -tol * r.report["energy moved"]


def test_O3_energy_bingham_wetting_tilt_and_two_densities():
    H, W, dx = 18, 18, 0.25 * mm
    light, heavy = Sub("oil", 930.0), Sub("lead white", 2500.0)
    q = blobs(H, W, [(6, 6), (11, 11)], 4, [(80 * um, 20 * um), (60 * um, 0.0)])
    con = Analytic((light, heavy), eta=5 * Pa_s, tau=2.0, gamma=30 * mN / m, wet=(-2e-3, 1 * um),
                   kT_v={"oil": KT_V_SMALL / 10, "lead white": k_B * T_room / 1e-19},
                   Lam={"oil": 1e-10 / (KT_V_SMALL / 10), "lead white": 2 * (0.3e-6) ** 2 / (9 * 5)})
    w = World(con, (light, heavy), flat_ground(H, W), dx, Environment(gravity=(0.0, 9.0, 3.0)), q=q)
    plastic = 0.0
    for dt in (0.5, 2.0, 8.0, 30.0):
        r = w.step(dt)
        check_energy(r)
        plastic += r.report["D plastic"]
    assert plastic > 0


def test_O3_add_and_environment_are_work():
    w = watercolour()
    w.step(0.2)
    F0 = sum(w.F().values())
    r = w.set_environment(Environment(gravity=w.env.gravity, activity={"water": 0.9}))
    assert np.isclose(float(r.moved("energy", src="work[environment]")), sum(w.F().values()) - F0, rtol=1e-12)
    assert np.isclose(float(w.energy.balance("stored")), sum(w.F().values()), rtol=1e-12)


def test_O3_D_num_is_the_bregman_divergence_for_a_quadratic_F():
    """Newtonian film, one density, no wetting: F is quadratic, so the time part of D_num equals
    1/2 dy.H.dy exactly; checked here as D_num > 0 and D_num/dt^2 constant as dt -> 0 (per step)."""
    w0 = _ripple_world(32, amp=0.5 * um)
    lam = _rate_exact(w0)
    vals = []
    for n in (1, 2, 4, 8):
        w = w0.fork()
        r = w.step(0.02 / lam / n)
        vals.append(r.report["D_num time"] / (0.02 / lam / n) ** 2)
        assert r.report["D_num time"] > 0
    assert np.allclose(vals, vals[-1], rtol=0.05)


# ---- O4 balance ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["flat film", "bare panel", "uniform mixture", "bingham below yield", "held clock"])
def test_O4_rest_states_do_not_move(case):
    H, W, dx = 12, 14, 0.5 * mm
    rate = None
    if case == "flat film":
        con = Analytic((WATER,), eta=ETA_W, gamma=GAM_W)
        w = World(con, (WATER,), flat_ground(H, W), dx, q=np.full((1, H, W), 40 * um))
    elif case == "bare panel":
        con = Analytic((WATER,), k=1e-15)
        w = World(con, (WATER,), flat_ground(H, W, 0.3, 50 * um), dx, Environment(activity={"water": 0.45}))
    elif case == "uniform mixture":
        a, b = Sub("a", 1000.0), Sub("b", 1200.0)
        con = Analytic((a, b), kT_v=KT_V_SMALL, Lam=1e-17)
        w = World(con, (a, b), flat_ground(H, W), dx, q=np.array([np.full((H, W), 30 * um), np.full((H, W), 10 * um)]))
    elif case == "bingham below yield":
        w = _ripple_world(16, amp=1 * um, tau=50.0, eta=10 * Pa_s, gn=g_0, gy=3.0)
    else:
        w = watercolour()
        rate = 0.0
    before = [a.copy() for a in _all_amounts(w)]
    for _ in range(3):
        r = w.step(1.0, rate=rate)
    for a, b in zip(_all_amounts(w), before):
        assert np.array_equal(a, b)
    assert r.report["energy moved"] == 0.0


# ---- O5 locality --------------------------------------------------------------------------------------------
def test_O5_windowed_step_equals_panel_step_bit_for_bit():
    a = watercolour(H=30, W=36)
    b = a.fork()
    win = a._window("auto")
    assert (win[1] - win[0]) * (win[3] - win[2]) < a.H * a.W        # the window is a proper part of the panel
    for dt in (0.1, 0.4, 0.4, 1.0):
        ra, rb = a.step(dt, window="auto"), b.step(dt, window="panel")
        assert ra.digest() == rb.digest()
    assert a.digest() == b.digest()
    for x, y in zip(_all_amounts(a), _all_amounts(b)):
        assert np.array_equal(x, y)


# ---- O6 determinism -----------------------------------------------------------------------------------------
def test_O6_fork_continues_identically_and_replay_reproduces():
    a = watercolour()
    a.step(0.3)
    f = a.fork()
    for dt in (0.2, 0.5):
        assert a.step(dt).digest() == f.step(dt).digest()
    assert a.digest() == f.digest()
    f.step(0.1)
    assert a.digest() != f.digest()              # independent
    replay = watercolour()
    for dt in (0.3, 0.2, 0.5):
        replay.step(dt)
    assert replay.digest() == a.digest()


# ---- O7 refinement ------------------------------------------------------------------------------------------
L_DOMAIN = 8 * mm


def _ripple_world(N, amp, h0=50 * um, tau=None, eta=ETA_W, gn=g_0, gy=0.0, mode=2):
    dx = L_DOMAIN / N
    x = (np.arange(N) + 0.5) * dx
    k = np.pi * mode / L_DOMAIN
    q = (h0 + amp * np.cos(k * x))[None, None, :] * np.ones((1, 2, 1))
    con = Analytic((WATER,), eta=eta, tau=tau, gamma=GAM_W)
    w = World(con, (WATER,), flat_ground(2, N), dx, Environment(gravity=(0.0, gy, gn)), q=q)
    w._k, w._h0 = k, h0
    return w


def _amp(w):
    x = (np.arange(w.W) + 0.5) * w.dx
    hh = w.q[0, 0] - w.q[0, 0].mean()
    return 2.0 / w.W * np.sum(hh * np.cos(w._k * x))


def _rate_exact(w):
    k = w._k
    return w._h0 ** 3 / (3 * ETA_W) * (RHO_W * g_0 * k ** 2 + GAM_W * k ** 4)


def _rate_discrete(w):
    k2 = (2 - 2 * np.cos(w._k * w.dx)) / w.dx ** 2
    return w._h0 ** 3 / (3 * ETA_W) * (RHO_W * g_0 * k2 + GAM_W * k2 * k2)


def test_O7_newtonian_ripple_decays_at_the_lubrication_rate_second_order_in_dx():
    errs = []
    for N in (16, 32, 64, 128):
        w = _ripple_world(N, amp=1e-3 * 50 * um)
        lam = _rate_exact(w)
        dt = 0.5 / lam
        A0 = _amp(w)
        w.step(dt)
        lam_num = (A0 / _amp(w) - 1) / dt            # implicit Euler, inverted: the space error alone
        assert np.isclose(lam_num, _rate_discrete(w), rtol=1e-5)
        errs.append(abs(lam_num - lam) / lam)
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 1.9), (errs, orders)


def test_O7_newtonian_ripple_first_order_in_dt_and_D_num_falls_first_order():
    w0 = _ripple_world(32, amp=1e-3 * 50 * um)
    lam_d = _rate_discrete(w0)
    T = 2.0 / lam_d
    A0 = _amp(w0)
    errs, dnum = [], []
    for n in (32, 64, 128, 256):
        w = w0.fork()
        D = 0.0
        for _ in range(n):
            D += w.step(T / n).report["D_num time"]
        errs.append(abs(_amp(w) / A0 - np.exp(-lam_d * T)))
        dnum.append(D)
    eo = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    do = np.log2(np.array(dnum[:-1]) / np.array(dnum[1:]))
    assert np.all(np.abs(eo - 1) < 0.15), (errs, eo)
    assert np.all(np.abs(do - 1) < 0.15) and np.all(np.array(dnum) > 0), (dnum, do)


def test_O7_bingham_holds_a_ripple_below_yield_and_stops_above_it():
    amp, h0 = 1 * um, 50 * um
    w = _ripple_world(32, amp=amp)
    k = w._k
    stress = h0 * (RHO_W * g_0 * k + GAM_W * k ** 3) * amp         # wall shear stress h |grad p| at the crest's flank
    held = _ripple_world(32, amp=amp, tau=2.0 * stress, eta=10 * Pa_s)
    before = held.q.copy()
    for _ in range(5):
        held.step(10.0)
    assert np.array_equal(held.q, before)
    moving = _ripple_world(32, amp=amp, tau=0.3 * stress, eta=10 * Pa_s)
    A0 = _amp(moving)
    for _ in range(120):
        moving.step(4000.0)
    rest = moving.q.copy()
    moving.step(4000.0)
    assert np.array_equal(moving.q, rest), "a Bingham film comes to rest, exactly"
    assert 0.3 * A0 < _amp(moving) < 0.5 * A0          # at rest with a residual ripple
    s_ = moving.q[0, 0]                                 # flat ground: s = h
    lap = np.diff(np.concatenate([[s_[0]], s_, [s_[-1]]]), 2) / moving.dx ** 2
    p_ = RHO_W * g_0 * s_ - GAM_W * lap
    wall = 0.5 * (s_[1:] + s_[:-1]) * np.abs(np.diff(p_)) / moving.dx
    tau = 0.3 * stress
    assert wall.max() <= tau * (1 + 1e-9) and wall.max() > 0.9 * tau   # held at, and only at, the yield stress


def _mixture_world(N, B=0.01, D=1e-9):
    a, b = Sub("a", 1000.0), Sub("b", 1000.0)
    dx = L_DOMAIN / N
    x = (np.arange(N) + 0.5) * dx
    k = np.pi * 2 / L_DOMAIN
    phi = 0.5 + B * np.cos(k * x)
    h = 40 * um
    q = np.array([phi * h, (1 - phi) * h])[:, None, :] * np.ones((1, 2, 1))
    con = Analytic((a, b), kT_v=KT_V_SMALL, Lam=D / KT_V_SMALL)
    w = World(con, (a, b), flat_ground(2, N), dx, Environment(gravity=(0, 0, g_0)), q=q)
    w._k, w._D = k, D
    return w


def _camp(w):
    x = (np.arange(w.W) + 0.5) * w.dx
    phi = w.q[0, 0] / w.q[:, 0].sum(0)
    return 2.0 / w.W * np.sum((phi - phi.mean()) * np.cos(w._k * x))


def test_O7_ideal_mixture_concentration_ripple_decays_at_D_k2():
    errs = []
    for N in (16, 32, 64):
        w = _mixture_world(N)
        lam = w._D * w._k ** 2
        dt = 0.5 / lam
        A0 = _camp(w)
        w.step(dt)
        lam_num = (A0 / _camp(w) - 1) / dt
        errs.append(abs(lam_num - lam) / lam)
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert errs[-1] < 2e-3 and np.all(orders > 1.8), (errs, orders)


def test_O7_evaporation_into_a_fixed_atmosphere_then_exact_drain():
    H = W = 3
    k_ex, mu_env, h0 = 1e-15, KT_V_SMALL * np.log(0.5), 20 * um
    con = Analytic((WATER,), eta=ETA_W, gamma=GAM_W, k=k_ex)
    env = Environment(gravity=(0, 0, g_0), activity={"water": 0.5})
    w = World(con, (WATER,), flat_ground(H, W), 1 * mm, env, q=np.full((1, H, W), h0))
    a = k_ex * RHO_W * g_0
    h_inf = mu_env / (RHO_W * g_0)
    t_star = np.log((h0 - h_inf) / (0 - h_inf)) / a
    dt = t_star / 7.5
    hn = h0
    for n in range(7):
        w.step(dt)
        hn = (hn + dt * k_ex * mu_env) / (1 + dt * a)              # implicit Euler of dh/dt = -k (rho g h - mu_env)
        assert np.allclose(w.q, hn, rtol=1e-10)
        exact = h_inf + (h0 - h_inf) * np.exp(-a * w.t)
        assert abs(hn - exact) / h0 < 1e-6
    r = w.step(dt)
    assert np.all(w.q == 0.0)
    assert np.isclose(float(w.matter.balance("reservoir[atmosphere]")[0]), RHO_W * h0 * H * W * (1 * mm) ** 2, rtol=1e-12)
    check_energy(r)


# ---- O8 exact limits ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("term,zero,absent", [
    ("yield_stress", dict(tau=0.0), dict(tau=None)),
    ("wetting", dict(wet=(0.0, 1 * um)), dict(wet=None)),
    ("mobility", dict(Lam=0.0), dict(Lam=None)),
    ("exchange", dict(k=0.0), dict(k=None)),
    ("sorption", dict(sorb=(0.0, 1e-15)), dict(sorb=None)),
    ("surface_tension", dict(gamma=0.0), dict(gamma=None)),
])
def test_O8_a_term_at_zero_is_bit_identical_to_the_world_without_it(term, zero, absent):
    H, W, dx = 14, 16, 0.25 * mm
    a, b = Sub("a", 1000.0), Sub("b", 1300.0)
    q = blobs(H, W, [(6, 6), (8, 10)], 3, [(40 * um, 5 * um), (30 * um, 0.0)])
    base = dict(eta=ETA_W, gamma=GAM_W, kT_v=KT_V_SMALL, Lam=1e-17, k=1e-15, wet=(-1e-3, 1 * um), sorb=(1e4, 1e-15))
    env = Environment(gravity=(0, 2.0, g_0), activity={"a": 0.5})
    worlds = []
    for over in (zero, absent):
        kw = dict(base); kw.update(over)
        worlds.append(World(Analytic((a, b), **kw), (a, b), flat_ground(H, W, 0.3, 50 * um), dx, env, q=q))
    for dt in (0.05, 0.2):
        for w in worlds:
            w.step(dt)
    assert worlds[0].state_digest() == worlds[1].state_digest()
