"""SPEC §4.1-4.2: the description and the five admissibility laws, each on passing and failing examples with its
shortfall checked; classify."""
import numpy as np
import pytest
from goo.evidence import Provenance as P
from goo.substance import (Substance, Thermo, Rheo, Reaction, StateVar, Oscillators, Lorentz, Gauss, Drude, Free,
                           Sphere, LogNormal, eV, herglotz, stable, dissipative, reciprocal, stoichiometric, classify)
from goo.substance.laws import kk_residual, KK_TOL
from goo.units import um, nm, mPa_s, mN
from conftest import liquid, particle, G, NONE


def sub(name="x", eps=None, morph=None, rho=1000.0, thermo=Thermo(), rheo=Rheo(), reactions=(), reservoir=0.0,
        material="", ev="guessed"):
    p = P(ev, "a source", "CC0") if ev in ("computed", "checked") else P(ev)
    keys = ["eps", "rho", "reservoir"] + (["morph"] if morph else []) + (["reactions"] if reactions else [])
    keys += [f"thermo.{k}" for k, v in thermo.__dict__.items() if v is not None]
    keys += [f"rheo.{k}" for k, v in rheo.__dict__.items() if v is not None]
    return Substance(name, eps or Oscillators(2.25), morph, rho, thermo, rheo, reactions, reservoir,
                     {k: p for k in keys}, material=material)


# ------------------------------------------------------------------------------------------------ the description
def test_every_given_field_needs_a_provenance_and_no_other():
    with pytest.raises(ValueError, match="no provenance"):
        Substance("a", Oscillators(2.0), None, 1000.0, Thermo(gamma=0.07), prov=dict(eps=G, rho=G, reservoir=NONE))
    with pytest.raises(ValueError, match="does not have"):
        Substance("a", Oscillators(2.0), None, 1000.0, prov=dict(eps=G, rho=G, reservoir=NONE, morph=G))
    with pytest.raises(ValueError):
        Substance("a", Oscillators(2.0), None, 1000.0, reservoir=-1.0, prov=dict(eps=G, rho=G, reservoir=NONE))


def test_a_derivable_cohesion_is_refused():
    with pytest.raises(ValueError, match="derivable"):
        Thermo(M=0.018, dH_vap=44e3, cohesion=2e9)


def test_substances_are_immutable_and_hashable():
    a = liquid("a"); b = liquid("a")
    assert hash(a) == hash(b) and a == b and len({a, b}) == 1
    with pytest.raises(Exception):
        a.rho = 3.0
    assert a.provenance("eps").evidence == "guessed"


def test_cohesion_is_derived_from_vaporisation():
    from goo.units import R_gas, T_room
    w = liquid("w", rho=998.0, M=0.018, dH_vap=44e3)
    assert np.isclose(w.cohesion, (44e3 - R_gas * T_room) * 998.0 / 0.018)
    assert set(w.cohesion_keys()) == {"thermo.dH_vap", "thermo.M", "rho"}


# ------------------------------------------------------------------------------------------------------ herglotz
def test_herglotz_proves_passive_oscillators():
    v = herglotz(sub(eps=Oscillators(2.0, (Lorentz(1.0, 6 * eV, 0.1 * eV), Gauss(0.5, 2.3 * eV, 0.2 * eV),
                                           Drude(1 * eV, 0.1 * eV)))))
    assert v.ok and v.method == "proved" and v.shortfall == 0.0


def test_herglotz_measures_gain():
    v = herglotz(sub(eps=Oscillators(2.25, (Gauss(-0.05, 2.3 * eV, 0.15 * eV),))))
    assert not v.ok and v.unit == "1"
    # the gain's depth: -min Im eps = A (1 - exp(-(2 E0 / s)^2)) at E0, to the sampling
    assert v.shortfall == pytest.approx(0.05, rel=2e-3)


def test_herglotz_measures_eps_inf_below_one():
    v = herglotz(sub(eps=Oscillators(0.8, (Lorentz(0.5, 8 * eV, 0.0),))))
    assert not v.ok and v.shortfall == pytest.approx(0.2)


def test_herglotz_measures_acausality():
    lam = np.linspace(350e-9, 1700e-9, 136)
    bad = Free.of(lam, 1.5, 0.3 * np.exp(-((lam - 600e-9) / 40e-9) ** 2))
    v = herglotz(sub(eps=bad))
    assert not v.ok
    rem = dict((p[0], p[1]) for p in v.parts)["KK remainder"]
    assert v.shortfall == pytest.approx(rem - KK_TOL) and rem > 5 * KK_TOL


def test_kk_test_passes_a_causal_table_and_tolerates_its_edges():
    lam = np.linspace(350e-9, 1700e-9, 136)
    osc = Oscillators(2.0, (Gauss(0.4, 2.3 * eV, 0.2 * eV), Lorentz(1.0, 7 * eV, 0.2 * eV)))
    m = osc.nk(lam)
    t = Free.of(lam, m.real, m.imag)
    assert kk_residual(t)[0] < 0.2 * KK_TOL
    assert herglotz(sub(eps=t)).ok


# --------------------------------------------------------------------------------------------- stable, dissipative
def test_stable_passes_and_fails_with_units():
    assert stable(liquid(gamma=0.03, M=0.05)).ok
    v = stable(liquid(gamma=-0.02))
    assert not v.ok and v.unit == "J/m^2" and v.shortfall == pytest.approx(0.02)
    v = stable(sub(rho=-5.0))
    assert not v.ok and v.unit == "kg/m^3" and v.shortfall == pytest.approx(5.0)
    v = stable(sub(rho=0.0))
    assert not v.ok and v.shortfall == 0.0                 # a strict inequality fails at equality
    v = stable(sub(thermo=Thermo(M=0.018, dH_vap=1.0)))   # dH_vap < R T: negative cohesive energy
    assert not v.ok and v.unit == "J/m^3"


def test_dissipative_passes_and_fails_with_units():
    assert dissipative(liquid()).ok
    v = dissipative(liquid(eta=-2 * mPa_s))
    assert not v.ok and v.unit == "Pa s" and v.shortfall == pytest.approx(2e-3)
    rx = Reaction((("x", -1.0), ("y", 1.0)), rate=-1e-3)
    v = dissipative(sub(reactions=(rx,)))
    assert not v.ok and v.unit == "1/s" and v.shortfall == pytest.approx(1e-3)
    v = dissipative(sub(rheo=Rheo(coupling=((1.0, 0.0), (0.0, -0.5)))))
    assert not v.ok and v.shortfall == pytest.approx(0.5)
    assert dissipative(sub(rheo=Rheo(coupling=((2.0, 1.0), (1.0, 2.0))))).ok


# ------------------------------------------------------------------------------------- reciprocal, stoichiometric
def test_reciprocal_passes_and_fails():
    assert reciprocal(liquid()).ok
    assert reciprocal(sub(rheo=Rheo(coupling=((2.0, 0.5), (0.5, 1.0))))).ok
    v = reciprocal(sub(rheo=Rheo(coupling=((2.0, 0.9), (0.1, 1.0)))))
    assert not v.ok and v.shortfall == pytest.approx(0.4)


def test_stoichiometric_passes_and_fails():
    ok = Reaction((("x", -1.0), ("oxygen", -0.1), ("y", 1.1)), rate=1e-5)
    assert stoichiometric(sub(reactions=(ok,))).ok
    v = stoichiometric(sub(reactions=(Reaction((("x", -1.0), ("y", 1.25)), 1e-5),)))
    assert not v.ok and v.unit == "kg/kg" and v.shortfall == pytest.approx(0.25)
    v = stoichiometric(sub(reactions=(Reaction((("z", -1.0), ("y", 1.0)), 1e-5),)))
    assert not v.ok                                         # the substance is not a reactant of its own reaction


# ------------------------------------------------------------------------------------------------------ classify
def test_classify_real_possible_impossible():
    real = sub(ev="computed", material="glass")
    c = classify(real)
    assert c.klass == "real" and set(c.passed) == {"herglotz", "stable", "dissipative", "reciprocal", "stoichiometric"}
    assert classify(sub(ev="computed")).klass == "possible"           # no named material
    c = classify(sub(ev="assumed", material="glass"))
    assert c.klass == "possible" and ("eps", "assumed") in c.blocking
    c = classify(liquid(eta=-1e-3))
    assert c.klass == "impossible" and c.failed == ("dissipative",)


def test_a_state_variable_is_declared_data():
    s = Substance("latex", Oscillators(2.2), Sphere(LogNormal(0.15 * um, 1.2)), 1180.0,
                  state=(StateVar("coalescence", "1", 0.0, 1.0),),
                  prov=dict(eps=G, morph=G, rho=G, reservoir=NONE))
    assert s.state[0].name == "coalescence" and hash(s)
