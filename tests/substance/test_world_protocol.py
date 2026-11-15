"""goo.substance.maps against goo.world's protocol (goo/world/protocol.py, docs/DYNAMICS.md): each function's
shapes, units and obligations, element-wise evaluation, finiteness on empty and one-phase cells, and a tiny world of
real substances stepping with closing receipts."""
import numpy as np
import pytest
from goo.evidence import Derived
from goo.film import Film, Ground
from goo.film import Environment as FilmEnvironment
from goo.substance import maps, library as L
from goo.units import um, k_B, T_room

WORLD_FUNCS = ("viscosity", "yield_stress", "surface_tension", "mixing", "mobility", "exchange", "wetting",
               "sorption", "reactions")


def scene(H=3, W=5, seed=0):
    """A film whose cells include empty ones, pure liquid, pure particles, a binary and a ternary mixture."""
    rng = np.random.default_rng(seed)
    subs = (L.WATER, L.GUM_ARABIC, L.HEMATITE)
    q = rng.uniform(0, 10 * um, (3, H, W)) * (rng.uniform(0, 1, (3, H, W)) > 0.4)
    q[:, 0, 0] = 0; q[:, 0, 1] = [10 * um, 0, 0]; q[:, 0, 2] = [0, 0, 5 * um]; q[:, 0, 3] = [5 * um, 2 * um, 0]
    return Film(subs, q)


def ground_for(film, porosity=0.5, depth=100 * um, held=None):
    g = Ground(np.zeros(film.shape), L.CELLULOSE, porosity, 5 * um, depth, film.substances)
    if held is not None:
        g.held = np.asarray(held, float)
    return g


ENV = FilmEnvironment(activity={"water": 0.5, "oxygen": 1.0})


def call(name, film):
    fn = getattr(maps, name)
    if name in ("exchange", "reactions"):
        return fn(film, ENV)
    if name in ("wetting", "sorption"):
        return fn(film, ground_for(film))
    return fn(film)


def arrays(v):
    if isinstance(v, list):
        return [a for r in v for a in arrays(r.rate)]
    if hasattr(v, "lo"):
        return [np.asarray(v.lo), np.asarray(v.hi)]
    val = v.value if isinstance(v, Derived) else v
    if isinstance(val, tuple):
        return [np.asarray(x) for x in val]
    return [np.asarray(val)]


@pytest.mark.parametrize("name", WORLD_FUNCS)
def test_element_wise_and_finite(name):
    """A cell's answer depends on that cell alone (windows and compacted [S, 1, n] films give the same numbers),
    and is finite everywhere, empty and one-phase cells included (mu may be -inf where a substance is absent, as
    the protocol allows; viscosity is inf where nothing flows, as SPEC's convention says)."""
    f = scene()
    full = arrays(call(name, f))
    cells = np.array([0, 3, 7, 11, 14])
    comp = Film(f.substances, f.q.reshape(3, -1)[:, cells][:, None, :])
    part = arrays(call(name, comp))
    win = Film(f.substances, f.q[:, 1:, 2:])
    wpart = arrays(call(name, win))
    for a, b, c in zip(full, part, wpart):
        if a.ndim < 2:
            assert np.array_equal(a, b, equal_nan=True)
            continue
        fa = a.reshape(a.shape[:-2] + (-1,))
        assert np.array_equal(fa[..., cells], b.reshape(b.shape[:-2] + (-1,)), equal_nan=True), name
        assert np.array_equal(a[..., 1:, 2:], c, equal_nan=True), name
        bad = ~np.isfinite(a)
        if name == "mixing":
            bad &= ~np.isneginf(a)
        if name in ("viscosity", "yield_stress"):
            # +inf only where nothing can flow: no liquid (empty or powder) or particles jammed; the viscosity
            # flags each of these out of domain except empty cells (a stated convention, SUBSTANCES.md)
            eta = maps.viscosity(f)
            stuck = np.isposinf(eta.value)
            assert np.all(np.isposinf(a) <= stuck), name
            assert np.all((~np.asarray(eta.in_domain, bool)) | (f.h == 0) | ~stuck)
            bad &= ~np.isposinf(a)
        assert not bad.any(), name


def test_mixing_split_derivatives_and_convexity():
    f = scene(4, 6, seed=2)
    mx = maps.mixing(f).value
    # f + f_x is the free energy: h times chemical_potential's density, by Euler (sum q mu = h f_v)
    mu_tot = maps.chemical_potential(f).value
    h = f.h
    wet = h > 0
    euler = np.sum(np.where(f.q > 0, f.q * mu_tot, 0.0), 0)
    assert np.allclose((mx.f + mx.f_x)[wet], euler[wet], rtol=1e-9, atol=1e-12)
    assert np.all(mx.f[~wet] == 0) and np.all(mx.f_x[~wet] == 0)
    S = f.q.shape[0]
    rng = np.random.default_rng(3)
    for c in zip(*np.nonzero(np.all(f.q > 0, 0))):
        q0 = f.q[:, c[0], c[1]]
        def parts(qv):
            m = maps.mixing(Film(f.substances, qv[:, None, None])).value
            return m.f[0, 0], m.mu[:, 0, 0], m.hess[:, :, 0, 0], m.f_x[0, 0], m.mu_x[:, 0, 0]
        F0, mu0, H0, Fx0, mux0 = parts(q0)
        for i in range(S):                                  # mu = df/dq by central differences, both parts
            d = 1e-6 * q0[i]
            e = np.zeros(S); e[i] = d
            assert mu0[i] == pytest.approx((parts(q0 + e)[0] - parts(q0 - e)[0]) / (2 * d), rel=1e-5, abs=1e-3)
            assert mux0[i] == pytest.approx((parts(q0 + e)[3] - parts(q0 - e)[3]) / (2 * d), rel=1e-5, abs=1e-3)
            assert np.allclose(H0[:, i], (parts(q0 + e)[1] - parts(q0 - e)[1]) / (2 * d), rtol=1e-4,
                               atol=1e-6 * np.abs(H0).max())
        assert np.linalg.eigvalsh(0.5 * (H0 + H0.T)).min() >= -1e-9 * np.abs(H0).max()   # convex part: PSD
        for _ in range(5):                                  # concave part: below its chords
            a, b = q0 * rng.uniform(0.5, 1.5, S), q0 * rng.uniform(0.5, 1.5, S)
            assert parts(0.5 * (a + b))[3] >= 0.5 * (parts(a)[3] + parts(b)[3]) - 1e-9 * abs(Fx0)
    # homogeneous of degree 1
    m2 = maps.mixing(Film(f.substances, 2 * f.q)).value
    assert np.allclose(m2.f + m2.f_x, 2 * (mx.f + mx.f_x), rtol=1e-12, atol=1e-12)


def test_mobility_is_D_v_over_kT():
    f = Film.uniform({L.WATER: 50 * um, L.HEMATITE: 1e-6 * um})
    D = maps.diffusivity(f).value[:, 0, 0]
    Lam = maps.mobility(f).value[:, 0, 0]
    v = np.array([s.molecular_volume for s in f.substances])
    assert np.allclose(Lam, D * v / (k_B * T_room), rtol=1e-12)
    # a sphere in Stokes-Einstein: D v / kT = d^2 / (18 eta) for a monodisperse population
    from conftest import particle
    p = particle("p", d=0.4 * um)
    g = Film.uniform({L.WATER: 50 * um, p: 1e-9 * um})
    eta = maps.viscosity(g).value[0, 0]
    assert maps.mobility(g).value[1, 0, 0] == pytest.approx((0.4 * um) ** 2 / (18 * eta), rel=1e-9)


def test_exchange_is_the_boundary_layer_law_in_potentials():
    f = Film.uniform({L.WATER: 20 * um, L.GUM_ARABIC: 0.0})
    k, mu_env, act = maps.exchange(f, ENV).value
    v = L.WATER.molecular_volume
    assert mu_env[0] == pytest.approx(k_B * T_room / v * np.log(0.5)) and mu_env[1] == 0.0
    assert np.all(k[1] == 0) and np.allclose(act[0], 1.0)
    mu = maps.mixing(f).value
    flux = k[0, 0, 0] * (mu.mu[0, 0, 0] + mu.mu_x[0, 0, 0] - mu_env[0])          # m/s
    e = maps.evaporation(f, ENV)["water"]
    assert flux == pytest.approx(e.rate.value[0, 0] * (1.0 - 0.5) / L.WATER.rho, rel=1e-12)
    # near equilibrium: k -> D_v M^2 p_sat a / (delta rho^2 (R T)^2)
    from goo.units import R_gas
    wet = maps.exchange(f, FilmEnvironment(activity={"water": 1.0})).value.k[0, 0, 0]
    t = L.WATER.thermo
    assert wet == pytest.approx(t.D_vapour * t.M ** 2 * t.p_sat / (5e-3 * L.WATER.rho ** 2 * (R_gas * T_room) ** 2),
                                rel=1e-12)
    # an activity of 0 is refused by the Environment itself (SPEC §8); a substance with no reservoir has k = 0
    with pytest.raises(ValueError):
        FilmEnvironment(activity={"water": 0.0})
    none = maps.exchange(f, FilmEnvironment(activity={}))
    assert np.all(none.value.k == 0)


def test_wetting_and_sorption():
    f = Film.uniform({L.WATER: 20 * um}, 2, 2)
    g = ground_for(f)
    S, ell = maps.wetting(f, g).value
    t = L.WATER.thermo; s = L.CELLULOSE.thermo
    want = 2 * np.sqrt(s.gamma_d * t.gamma_d) + 2 * np.sqrt((s.gamma - s.gamma_d) * (t.gamma - t.gamma_d)) - 2 * t.gamma
    assert np.allclose(S, want) and np.all(S < 0) and np.all(ell > 0)
    P, kap = maps.sorption(f, g).value
    cos = 1 + want / t.gamma
    assert np.allclose(P[0], 2 * t.gamma * cos / 5e-6) and np.all(kap[0] > 0)
    assert np.allclose(kap[0], 0.5 * (5e-6) ** 2 / (8 * 1.0016e-3 * 5e-6))
    full = ground_for(f, held=np.full((1, 2, 2), 50 * um))
    assert np.all(maps.sorption(f, full).value.P == 0)
    empty = Film.empty(f.substances, 2, 2)
    S0, l0 = maps.wetting(empty, g).value
    assert np.all(S0 == 0) and np.all(l0 > 0)


# ----------------------------------------------------------------------------------------------- a tiny world
def _world_inputs(without=()):
    """(constitutive, environment) for goo.world as it stands in this copy. When goo.world reads exchange's
    (k, mu_env, activity) and goo.film.Environment (the dynamics owner's change in progress), these are maps itself
    and goo.film.Environment; until then a two-line shim gives the world exchange's k and the potentials."""
    from goo.world import protocol
    env = FilmEnvironment(activity={"water": 0.5})
    native = "activity" in getattr(protocol.Environment, "__dataclass_fields__", {})

    class Without:
        def __getattr__(self, name):
            return None if name in without else getattr(maps, name)
    if native:
        return (Without() if without else maps), env

    class Shim:
        def __getattr__(self, name):
            return None if name in without else getattr(maps, name)

        def yield_stress(self, film):                      # the world will collapse the bound itself
            return maps.yield_stress(film).collapse("lo")[0]

        def exchange(self, film, e):
            d = maps.exchange(film, e)
            return Derived(d.value.k, d.evidence, d.in_domain, d.map)

    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class Env:
        T: float
        gravity: tuple
        activity: dict
        boundary_layer: float
        potentials: dict

    probe = Film.uniform({L.WATER: 1 * um})
    mu_env = maps.exchange(probe, env).value.mu_env
    return Shim(), Env(env.T, env.gravity, dict(env.activity), env.boundary_layer, {"water": float(mu_env[0])})


def _tiny(without=()):
    from goo.world import World
    con, env = _world_inputs(without)
    H, W, dx = 6, 6, 1e-4
    q = np.zeros((1, H, W)); q[0, 2:4, 2:4] = 20 * um
    g = Ground(np.zeros((H, W)), L.CELLULOSE, 0.5, 5 * um, 100 * um, (L.WATER,))
    w = World(con, (L.WATER,), g, dx, env, q=q)
    return w, q, dx


def test_a_tiny_world_of_water_on_cellulose():
    """Water on a cellulose ground at 50 % humidity, every function of the protocol but wetting (see the xfail
    below): the receipts close, every dissipation is non-negative, water leaves to the air and into the pores."""
    w, q, dx = _tiny(without=("wetting",))
    m0 = L.WATER.rho * q.sum() * dx * dx
    out = pores = 0.0
    for _ in range(4):
        r = w.step(1e-3)
        assert r.residual["matter"] <= 1e-12
        assert r.residual["energy"] <= 1e-9
        for k in ("viscous", "plastic", "diffusion", "exchange", "sorption"):
            assert r.report[k] >= 0.0, k
        assert r.report["num_time"] >= -1e-12 * max(r.report["moved"], 1e-300)
        out += r.moved("matter", dst="reservoir[atmosphere]")
        pores += r.moved("matter", dst="ground")
    assert out > 0 and pores > 0
    assert np.sum(w.ground.held) > 0
    assert L.WATER.rho * w.q.sum() * dx * dx == pytest.approx(m0 - out - pores, rel=1e-10)


@pytest.mark.xfail(reason="goo.world's dF_wet computes exp(-h0/ell) * -expm1(-dh/ell), which is 0 * inf = nan when "
                          "ell is molecular (0.3 nm here, from the Hamaker constant) and a cell drains by more than "
                          "~700 ell; reported to the lead (the world's arithmetic, not this map)", strict=True)
def test_a_tiny_world_with_wetting():
    w, q, dx = _tiny()
    for _ in range(2):
        r = w.step(1e-3)
        assert r.residual["matter"] <= 1e-12 and r.residual["energy"] <= 1e-9
