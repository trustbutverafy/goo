"""The world's acts and services: clocks (SPEC §8), retirement (§6), add, stack, digest, flags; and the second
priority (cure, dissolution) as xfail obligations."""
import numpy as np
import pytest
from goo.evidence import Derived
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
    assert np.isclose(f.amount, 0.5 * rate.mean())
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


@pytest.mark.xfail(strict=True, reason="second priority not done: cure as a reaction with an oxygen reservoir "
                   "(a term of F with a reaction affinity and a Psi* in the reaction rate) is not implemented")
def test_cure_is_a_reaction_with_a_reservoir():
    w = watercolour()
    assert hasattr(w, "react") and w.react(0.1).moved("matter", src="reservoir[oxygen]") is not None


@pytest.mark.xfail(strict=True, reason="second priority not done: dissolution of a stratum's substances into a "
                   "liquid above (exchange between stratum[k] and wet, driven by mu) is not implemented")
def test_dissolution_of_a_stratum_into_a_liquid_above():
    w = watercolour()
    w.retire(substances=["gum"])
    w.add({"water": 20 * um})
    r = w.step(1.0)
    assert np.any(r.moved("matter", src="stratum[0]", dst="wet"))
