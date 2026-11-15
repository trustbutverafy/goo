"""The world's acts and services: clocks (SPEC §8), retirement (§6), add, stack, digest, flags; and the second
priority (cure, dissolution) as xfail obligations."""
import numpy as np
import pytest
from goo.evidence import Derived, Bounded
from goo.film import Stack
from goo.units import um, mm, mPa_s, mN, m, g_0
from goo.world import World, Environment, Constitutive
from world_analytic import Sub, Analytic, flat_ground, KT_V_SMALL
from world_scenes import watercolour, check_energy, WATER, blobs


def test_clock_rate_zero_holds_the_world_exactly_and_is_a_privilege():
    w = watercolour()
    before = [w.q.copy(), w.ground.held.copy()]
    r = w.step(2.0, rate=0.0)
    assert np.array_equal(w.q, before[0]) and np.array_equal(w.ground.held, before[1])
    assert w.t == 2.0
    (f,) = r.flagged("privilege")
    assert f.amount == 0.0
    check_energy(r)


def test_clock_rate_field_holds_a_region_and_runs_the_rest():
    w = watercolour()
    rate = np.ones((w.H, w.W)); rate[:, :12] = 0.0
    before = w.q.copy()
    r = w.step(0.5, rate=rate)
    assert np.array_equal(w.q[:, :, :12], before[:, :, :12])
    assert not np.array_equal(w.q[:, :, 12:], before[:, :, 12:])
    (f,) = r.flagged("privilege")
    assert np.isclose(f.amount, 0.5 * rate.sum() * w.dx ** 2) and np.isclose(r.report["rate integral"], f.amount)
    check_energy(r)


def test_clock_scales_dissipation_like_time():
    """rate r for dt matches rate 1 for r dt, to rounding, for a world whose every process is a Psi*."""
    a, b = watercolour(), watercolour()
    a.step(0.4, rate=0.5)
    b.step(0.2)
    assert np.allclose(a.q, b.q, rtol=1e-9, atol=1e-18)


def test_clock_negative_rate_is_refused():
    w = watercolour()
    with pytest.raises(ValueError, match="reverse"):
        w.step(0.1, rate=-1.0)


def test_retire_moves_matter_and_free_energy_without_loss():
    w = watercolour()
    w.step(0.3)
    stored = float(w.energy.balance("stored"))
    mask = w.q[2] > 0
    r = w.retire(mask=mask, substances=["ochre", "gum"])
    assert r.residual["matter"] <= 1e-12 and r.residual["energy"] <= 1e-12
    assert np.all(w.q[1:][:, mask] == 0) and len(w.strata) == 1
    frozen = float(w.energy.balance("frozen"))
    assert np.isclose(stored - frozen, float(w.energy.balance("stored")), rtol=1e-12)
    # the stratum raises the relief under what is left
    assert np.allclose(w.zb, w.ground.z + w.strata[0].sum(0))
    r = w.step(0.3)
    check_energy(r)


def test_add_books_mass_and_free_energy():
    w = watercolour()
    m0 = w.matter.balance("wet").copy()
    r = w.add({"water": np.where(w.q.sum(0) > 0, 3 * um, 0.0)})
    assert np.all(w.matter.balance("wet") - m0 >= 0)
    assert r.moved("energy", src="work[origin]") != 0
    assert r.residual["energy"] <= 1e-12
    with pytest.raises(ValueError):
        w.add({"water": -1.0})


def test_stack_is_read_only_and_does_not_load_the_world():
    w = watercolour()
    w.step(0.2); w.retire(mask=w.q[2] > 0, substances=["ochre"])
    d = w.digest()
    s = w.stack()
    assert isinstance(s, Stack) and len(s.films) == 2 and s.dx == w.dx
    with pytest.raises(ValueError):
        s.films[-1].q[0, 0, 0] = 1.0
    assert np.allclose(s.top, w.zb + w.q.sum(0))
    assert w.digest() == d


def test_slope_flag_and_out_of_domain_flag():
    class Partial(Analytic):
        def viscosity(self, film):
            return Derived(np.full(film.shape, 1e-3), "assumed", False, "viscosity outside its fit")
    H, W = 10, 10
    q = blobs(H, W, [(5, 5)], 2, [(80 * um,)])
    w = World(Partial((WATER,)), (WATER,), flat_ground(H, W), 0.1 * mm, q=q)
    r = w.step(1e-4)
    assert r.flagged("slope") and r.flagged("out_of_domain")


def test_constitutive_protocol_is_runtime_checkable():
    assert isinstance(Analytic((WATER,)), Constitutive)


@pytest.mark.xfail(strict=True, reason="cure does not fit the gradient structure with the maps as they are: "
                   "reactions(film, env) gives a rate of extent (1/s, first order) and no affinity. A term of F needs a "
                   "free energy of reaction (g per kg of each substance, or dG per kg of extent, with the reservoir "
                   "species' potentials); with it Psi* = k m A_+ (convex, zero at 0) reproduces the map's first-order "
                   "rate wherever A > 0 and stops at equilibrium. See docs/DYNAMICS.md §10")
def test_cure_is_a_reaction_with_a_reservoir():
    w = watercolour()
    assert hasattr(w, "react") and w.react(0.1).moved("matter", src="reservoir[oxygen]") is not None


@pytest.mark.xfail(strict=True, reason="dissolution needs a stratum to hold a live potential: under the "
                   "retirement rule its free energy is in `frozen`, outside F, so no force can drive matter out of it. "
                   "It needs the top stratum's mixing free energy kept in F (frozen holds only what cannot move) and "
                   "a rate coefficient map for stratum-to-liquid transfer. See docs/DYNAMICS.md §10")
def test_dissolution_of_a_stratum_into_a_liquid_above():
    w = watercolour()
    w.retire(substances=["gum"])
    w.add({"water": 20 * um})
    r = w.step(1.0)
    assert np.any(r.moved("matter", src="stratum[0]", dst="wet"))


def test_bounded_values_are_collapsed_at_mid_and_flagged():
    """A map may answer with goo.evidence.Bounded (SPEC §2): the world collapses it at mid, keeps the note, and the
    result is the world given the mid value."""
    class Paste(Analytic):
        def yield_stress(self, film):
            return Bounded(np.full(film.shape, 1.0), np.full(film.shape, 5.0), "computed", "yield stress (Hamaker)")
    class Mid(Analytic):
        def yield_stress(self, film):
            return Derived(np.full(film.shape, 1.0) * 0.5 + np.full(film.shape, 5.0) * 0.5, "assumed")
    H, W = 12, 12
    q = blobs(H, W, [(6, 6)], 3, [(60 * um,)])
    a = World(Paste((WATER,), eta=3.0, tau=1.0), (WATER,), flat_ground(H, W), 0.25 * mm, q=q)
    b = World(Mid((WATER,), eta=3.0, tau=1.0), (WATER,), flat_ground(H, W), 0.25 * mm, q=q)
    ra, rb = a.step(5.0), b.step(5.0)
    (f,) = ra.flagged("bounds_collapsed")
    assert f.subject == "yield_stress" and "yield stress (Hamaker)" in f.detail and "collapsed at 0.5" in f.detail
    assert not rb.flagged("bounds_collapsed")
    assert a.state_digest() == b.state_digest()


def test_wetting_term_is_finite_for_a_molecular_length():
    """ell of 0.3 nm against 0.2 mm cells and films that drain by micrometres: F_wet and its increments stay finite."""
    H, W = 10, 10
    q = blobs(H, W, [(5, 5)], 2, [(30 * um,)])
    con = Analytic((WATER,), eta=1 * mPa_s, wet=(-1e-2, 0.3e-9), k=1e-14)
    w = World(con, (WATER,), flat_ground(H, W), 0.2 * mm, Environment(activity={"water": 0.5}), q=q)
    for _ in range(6):
        r = w.step(0.5)
        assert all(np.isfinite(v) for b, a in r.F.values() for v in (b, a))
        check_energy(r)


def test_sorption_is_a_transfer_inside_F():
    """Water soaking into the pores is counted against its reservoir in F too, so its free energy moves with it:
    sorption's own D_mech carries the loss and D_num stays a small part of it."""
    H, W = 8, 8
    q = np.full((1, H, W), 20 * um)
    con = Analytic((WATER,), eta=1 * mPa_s, k=1e-15, sorb=(3e4, 1e-12))
    w = World(con, (WATER,), flat_ground(H, W, porosity=0.5, depth=100 * um), 0.25 * mm,
              Environment(activity={"water": 0.5}), q=q)
    for _ in range(4):
        r = w.step(0.2)
        check_energy(r)
        assert r.report["D sorption"] > 0
        assert abs(r.report["D_num time"]) <= 1e-2 * (r.report["D sorption"] + r.report["D exchange"])
    assert np.sum(w.ground.held) > 0


def test_infinite_viscosity_is_zero_mobility():
    """A cell where nothing flows (viscosity +inf: a powder, a jam) has mobility exactly 0: no flux through its
    faces, nothing NaN, its content unchanged by flow."""
    class Jam(Analytic):
        def viscosity(self, film):
            eta = np.full(film.shape, 1e-3)
            if film.shape == (12, 12):
                eta[:, 6:] = np.inf
            return Derived(eta, "assumed")
    H, W = 12, 12
    q = blobs(H, W, [(6, 5)], 4, [(40 * um,)])
    w = World(Jam((WATER,)), (WATER,), flat_ground(H, W), 0.25 * mm, q=q, margin=H)
    before = w.q.copy()
    r = w.step(0.1, window="panel")
    assert np.all(np.isfinite(w.q))
    assert np.array_equal(w.q[:, :, 7:], before[:, :, 7:])        # behind the jammed column nothing moved
    assert not np.array_equal(w.q[:, :, :6], before[:, :, :6])
    check_energy(r)
