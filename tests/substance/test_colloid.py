"""Particles are not solutes; binders vitrify; reactions have a free energy (docs/SUBSTANCES.md §3a). Owner: substance.

The collective diffusivity here is the one the maps imply in the solvent's frame: a population's flux relative to
the liquid, j = -(h Lambda_p phi_p) d(mu_p - mu_l)/dx, so D_c = h Lambda_p phi_p (H_pp - 2 H_pl + H_ll)."""
import dataclasses
import time
import numpy as np
import pytest
from goo.film import Film
from goo.substance import maps, library as L, classify, stable, dissipative, Reaction
from goo.substance import _freeenergy as fe
from goo.units import um, k_B, T_room, R_gas

PARTICLES = [s for s in L.ALL if s.morph is not None and classify(s).klass != "impossible"]
DC_MAX = 1e4            # the stated bound on D_c / D_SE: <= 25 up to phi = 0.64; it grows as (phi_m - phi)^-0.36
                        # toward packing (consolidation of an incompressible bed) and is reached only at the capillary
                        # entry point, within 1e-6 (rutile, 700) to 1e-11 (20 um cellulose, 3.4e3) of packing


def se(p, eta=L.WATER.rheo.eta):
    return k_B * T_room * p.morph.size.number_moment(-1) / (3 * np.pi * eta)


def collective(p, phi, liquid=L.WATER, h=30 * um):
    f = Film.uniform({liquid: (1 - phi) * h, p: phi * h})
    Lam = maps.mobility(f).value[1, 0, 0]
    H = maps.mixing(f).value.hess[:, :, 0, 0]          # one liquid: the concave part is identically 0
    return h * phi * Lam * (H[0, 0] - 2 * H[0, 1] + H[1, 1])


@pytest.mark.parametrize("p", PARTICLES, ids=lambda s: s.name)
def test_collective_diffusivity_is_stokes_einstein_times_order_one(p):
    for phi in (0.01, 0.20, 0.45):
        r = collective(p, phi) / se(p)
        assert 0.1 < r < 10, (phi, r)
    assert collective(p, 0.01) / se(p) == pytest.approx(1.0, abs=0.1)          # dilute: Stokes-Einstein
    ms = fe.MixSet.of((L.WATER, p))
    xe = ms.entry(1, np.array([T_room]))[0][0]
    grid = np.concatenate([np.linspace(0.001, 0.64, 200), xe * (1 - np.geomspace(1e-2, 1e-9, 30))])
    grid = grid[grid < xe]
    r = np.array([collective(p, x) for x in grid]) / se(p)
    assert np.all(r > 0) and r.max() < DC_MAX
    assert np.all(r[grid <= 0.64] < 25)
    # exactly zero at and above packing (the viscosity is infinite: jammed)
    pm = maps.packing(Film.uniform({L.WATER: 1 * um, p: 1 * um})).value[0, 0]
    for phi in (pm * (1 + 1e-12), 0.75, 0.95):
        assert collective(p, phi) == 0.0


def test_the_old_artefact_is_gone():
    """Flory-Huggins made a particle a solute: water's potential in a suspension carried k T / v_w ln phi_w, so a
    particle's diagonal stiffness Lambda_p H_pp was ~1e-4 m^2/s. Now the particles' terms are k T / v_p in scale."""
    f = Film.uniform({L.WATER: 18 * um, L.RUTILE: 12 * um})
    H = maps.mixing(f).value.hess[:, :, 0, 0]
    Lam = maps.mobility(f).value[:, 0, 0]
    assert Lam[1] * H[1, 1] * f.h[0, 0] < 1e-9
    # a particle holds no water: water's potential in a dilute suspension is minus a tiny osmotic pressure
    mu_w = maps.chemical_potential(f).value[0, 0, 0]
    assert -1e3 < mu_w < 0


def eq_water(amounts, a_env=0.5, w0=200 * um):
    """The water amount (m) at which the film's water activity equals a_env, by bisection (activity rises with water)."""
    subs = (L.WATER,) + tuple(amounts)

    def act(w):
        q = np.array([w] + [amounts[s] for s in amounts])[:, None, None]
        return maps.evaporation(Film(subs, q), maps.Atmosphere(activity={"water": a_env}))["water"].activity.value[0, 0]
    lo, hi = 0.0, w0
    if act(1e-30) > a_env:
        return 0.0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if act(m) < a_env else (lo, m)
    return 0.5 * (lo + hi)


def test_a_dried_film_holds_what_its_binder_holds_and_particles_hold_none():
    """At 50 % RH gum arabic holds 13.5 g of water per 100 g (Ahmed & Abdelgadir 2014, the datum its description is
    derived from): water volume fraction 0.159 of the gum's solution. A dried film of pigment and gum holds exactly
    that where the gum fills the pores between the particles, and ~4 % more where it does not (the menisci's
    suction, 3.5 MPa between rutile particles, lowers the water's activity by exp(-Pi_e v_w / kT) = 0.974); particles in water alone dry completely (the lab's expectation: < 1 % of the film)."""
    phi_ga = L._GA_PHI
    g = 10 * um
    alone = eq_water({L.GUM_ARABIC: g})
    assert alone / (alone + g) == pytest.approx(phi_ga, rel=1e-6)
    filled = eq_water({L.GUM_ARABIC: g, L.HEMATITE: 10 * um})
    assert filled / g == pytest.approx(alone / g, rel=1e-6)
    under = eq_water({L.GUM_ARABIC: 2 * um, L.RUTILE: 30 * um})          # voids: the gum's water feels the suction
    assert under / (2 * um) == pytest.approx(alone / g, rel=0.06) and under / (2 * um) > alone / g
    assert eq_water({L.RUTILE: 30 * um}) == 0.0                           # particles in water dry to a powder
    assert eq_water({L.PMMA_LATEX: 30 * um}) == 0.0


def test_drying_by_hand_viscosity_is_monotone_and_vitrifies():
    g, p = 10 * um, 10 * um
    sg = L.GUM_ARABIC.rheo.glass
    w = np.linspace(80 * um, 0.0, 400)
    q = np.stack([w, np.full_like(w, g), np.full_like(w, p)])[:, None, :]
    eta = maps.viscosity(Film((L.WATER, L.GUM_ARABIC, L.HEMATITE), q)).value[0]
    s = w / (w + g)
    wet = s > sg
    assert np.all(np.isfinite(eta[wet])) and np.all(np.isposinf(eta[~wet]))
    assert np.all(np.diff(eta[wet]) > 0)                                  # water leaves: viscosity rises
    w_eq = eq_water({L.GUM_ARABIC: g, L.HEMATITE: p})
    dry = maps.viscosity(Film.uniform({L.WATER: w_eq, L.GUM_ARABIC: g, L.HEMATITE: p})).value[0, 0]
    assert w_eq / (w_eq + g) < sg and np.isposinf(dry)                     # it settles by law
    assert maps.mobility(Film.uniform({L.WATER: w_eq, L.GUM_ARABIC: g, L.HEMATITE: p})).value.max() == 0.0
    # the wet calibration is kept: a 30 % gum solution is ~0.1 Pa s
    assert 0.05 < maps.viscosity(Film.uniform({L.WATER: 7 * um, L.GUM_ARABIC: 3 * um})).value[0, 0] < 0.2


def test_the_osmotic_term_is_convex_and_diverges_at_packing():
    for p in (L.RUTILE, L.CALCITE, L.SOOT):
        ms = fe.MixSet.of((L.WATER, p))
        pm, xe = ms.phim[1], ms.entry(1, np.array([T_room]))[0][0]
        x = np.concatenate([np.linspace(0.01, 0.6, 60), xe * (1 - np.geomspace(1e-2, 1e-7, 20))])
        Pi = np.array([-maps.chemical_potential(Film.uniform({L.WATER: (1 - v) * um, p: v * um})).value[0, 0, 0]
                       for v in x])
        assert np.all(np.diff(Pi) > 0)                                    # convex: the pressure rises with loading
        assert Pi[-1] > 1e4 * Pi[x <= 0.5][-1]                            # and diverges toward packing
        # the analytic form diverges as (phi_m - x)^-1 and matches Carnahan-Starling in the fluid range
        eps = np.geomspace(1e-3, 1e-9, 7)
        assert np.allclose(fe.hs_Z(pm * (1 - eps), pm) * eps, fe.hs_Z(pm * (1 - eps[-1]), pm) * eps[-1], rtol=1e-2)
        cs = lambda v: (1 + v + v * v - v ** 3) / (1 - v) ** 3
        v = np.linspace(0.01, 0.5, 50)
        assert np.all(np.abs(fe.hs_Z(v, pm) / cs(v) - 1) < 0.12)
        # beyond the entry point voids open: the liquid's suction is held at the capillary pressure
        voided = -maps.chemical_potential(Film.uniform({L.WATER: 0.2 * um, p: 0.8 * um})).value[0, 0, 0]
        assert voided == pytest.approx(ms.Pi_e[1], rel=1e-5)        # plus the ideal k T n
        assert stable(p).ok
    # convex part positive semi-definite on mixtures of two populations, a solution and a voided cell
    rng = np.random.default_rng(5)
    subs = (L.WATER, L.GUM_ARABIC, L.RUTILE, L.CALCITE, L.ETHANOL)
    q = rng.uniform(0, 10 * um, (5, 1, 300)) * (rng.uniform(0, 1, (5, 1, 300)) > 0.3)
    q[:, 0, 0] = [0.1 * um, 0, 10 * um, 5 * um, 0]
    H = maps.mixing(Film(subs, q)).value.hess[:, :, 0, :]
    for c in range(q.shape[2]):
        Hc = H[:, :, c]
        if np.abs(Hc).max() > 0:
            assert np.linalg.eigvalsh(Hc).min() >= -1e-9 * np.abs(Hc).max()
    for s in L.ALL:
        if classify(s).klass != "impossible":
            assert stable(s).ok and dissipative(s).ok, s.name


def test_a_powder_and_a_pure_liquid_have_no_mixing_energy():
    for amounts in ({L.RUTILE: 5 * um}, {L.RUTILE: 5 * um, L.HEMATITE: 0.0}, {L.GUM_ARABIC: 5 * um}):
        m = maps.mixing(Film.uniform(amounts)).value
        assert np.all(m.f == 0) and np.all(m.f_x == 0)
    m = maps.mixing(Film.uniform({L.RUTILE: 5 * um, L.WATER: 0.0})).value
    assert np.all(m.f == 0) and np.isfinite(m.mu[0]).all() is not None


def test_cheaper_entry_points_agree_with_mixing():
    rng = np.random.default_rng(2)
    subs = (L.WATER, L.ETHANOL, L.GUM_ARABIC, L.RUTILE, L.HEMATITE)
    q = rng.uniform(0, 10 * um, (5, 7, 9)) * (rng.uniform(0, 1, (5, 7, 9)) > 0.4)
    q[:, :2, :2] = q[:, :1, :1]                                   # repeated compositions are grouped
    q[:, 3, 3] = 2 * q[:, 3, 4]                                   # and so are scaled ones
    f = Film(subs, q)
    m, p = maps.mixing(f).value, maps.mixing_potentials(f).value
    for a, b in ((m.f, p.f), (m.mu, p.mu), (m.f_x, p.f_x), (m.mu_x, p.mu_x)):
        assert np.array_equal(a, b)
    assert p.hess is None
    dq = rng.normal(size=q.shape) * 1e-7
    want = np.einsum("ijhw,jhw->ihw", m.hess, dq)
    assert np.allclose(maps.mixing_hvp(f, dq).value, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


def test_mixing_costs_what_the_composition_costs():
    """256 x 256 cells, 11 substances in the world, 4 present: 0.27 s before this change (measured), and the full
    Hessian; now each distinct composition is solved once on its present substances."""
    subs = (L.WATER, L.ETHANOL, L.LINSEED, L.LINOXYN, L.GUM_ARABIC, L.RUTILE, L.SOOT, L.HEMATITE, L.CDS, L.CALCITE,
            L.PMMA_LATEX)
    rng = np.random.default_rng(0)
    q = np.zeros((11, 256, 256))
    q[0] = 30 * um * rng.uniform(0.5, 1, (256, 256)); q[4] = 3 * um * rng.uniform(0.5, 1, (256, 256))
    q[5] = 5 * um * rng.uniform(0.5, 1, (256, 256)); q[7] = 2 * um * rng.uniform(0.5, 1, (256, 256))
    f = Film(subs, q)
    maps.mixing(f)
    best = {}
    for name in ("mixing", "mixing_potentials"):
        ts = []
        for _ in range(3):
            t = time.perf_counter(); getattr(maps, name)(f); ts.append(time.perf_counter() - t)
        best[name] = min(ts)
    assert best["mixing"] < 0.2 and best["mixing_potentials"] < 0.15, best
    # one composition everywhere costs one solve
    u = Film(subs, np.broadcast_to(q[:, :1, :1], q.shape).copy())
    t = time.perf_counter(); maps.mixing_potentials(u); assert time.perf_counter() - t < 0.05


def test_reactions_carry_an_affinity():
    f = Film.uniform({L.LINSEED: 10 * um, L.LINOXYN: 1 * um})
    env = maps.Atmosphere(activity={"oxygen": 0.21})
    (r,) = maps.reactions(f, env)
    rx = L.LINSEED.reactions[0]
    mu = maps.chemical_potential(f).value[:, 0, 0]
    want = (-rx.dG + mu[0] / L.LINSEED.rho - 1.12 * mu[1] / L.LINOXYN.rho
            + 0.12 * R_gas * T_room / 31.998e-3 * np.log(0.21))
    assert r.affinity.value[0, 0] == pytest.approx(want, rel=1e-12) and r.affinity.value[0, 0] > 0
    assert r.affinity.evidence == "guessed" and np.all(r.affinity.in_domain)
    # more oxygen: more affinity; no free energy given: A = 0, out of domain
    (r2,) = maps.reactions(f, maps.Atmosphere(activity={"oxygen": 1.0}))
    assert r2.affinity.value[0, 0] > r.affinity.value[0, 0]
    bare = dataclasses.replace(L.LINSEED, reactions=(dataclasses.replace(rx, dG=None, M_reservoir=()),),
                               prov={k: v for k, v in L.LINSEED.prov if k != "reactions.dG"})
    (r3,) = maps.reactions(Film.uniform({bare: 10 * um, L.LINOXYN: 0.0}), env)
    assert np.all(r3.affinity.value == 0) and not np.any(r3.affinity.in_domain)
    # a reaction that runs uphill is not dissipative
    up = dataclasses.replace(L.LINSEED, reactions=(dataclasses.replace(rx, dG=+2e5),))
    v = dissipative(up)
    assert not v.ok and v.shortfall == pytest.approx(2e5) and classify(up).klass == "impossible"


def test_vitrification_is_a_description_field_with_its_law():
    assert L.GUM_ARABIC.provenance("rheo.glass").evidence == "guessed"
    bad = dataclasses.replace(L.GUM_ARABIC, rheo=dataclasses.replace(L.GUM_ARABIC.rheo, glass=1.2))
    assert not dissipative(bad).ok
    renamed = dataclasses.replace(L.GUM_ARABIC, name="not_gum")
    q = np.array([[[1 * um]], [[10 * um]]])
    assert np.isposinf(maps.viscosity(Film((L.WATER, renamed), q)).value[0, 0])
    no_glass = dataclasses.replace(L.GUM_ARABIC, rheo=dataclasses.replace(L.GUM_ARABIC.rheo, glass=None),
                                   prov={k: v for k, v in L.GUM_ARABIC.prov if k != "rheo.glass"})
    assert np.isfinite(maps.viscosity(Film((L.WATER, no_glass), q)).value[0, 0])


def test_new_maps_are_element_wise():
    rng = np.random.default_rng(4)
    subs = (L.LINSEED, L.LINOXYN, L.HEMATITE, L.WATER)
    q = rng.uniform(0, 10 * um, (4, 3, 5)) * (rng.uniform(0, 1, (4, 3, 5)) > 0.3)
    q[:, 0, 0] = 0
    dq = rng.normal(size=q.shape) * 1e-7
    f = Film(subs, q)
    cells = np.array([0, 3, 7, 11, 14])
    comp = Film(subs, q.reshape(4, -1)[:, cells][:, None, :])
    full = maps.mixing_hvp(f, dq).value.reshape(4, -1)[:, cells]
    part = maps.mixing_hvp(comp, dq.reshape(4, -1)[:, cells][:, None, :]).value.reshape(4, -1)
    assert np.array_equal(full, part)
    env = maps.Atmosphere(activity={"oxygen": 0.21})
    ra, rb = maps.reactions(f, env), maps.reactions(comp, env)
    for a, b in zip(ra, rb):
        assert np.array_equal(a.affinity.value.reshape(-1)[cells], b.affinity.value.reshape(-1))
        assert np.all(np.isfinite(a.affinity.value))
