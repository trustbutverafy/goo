"""Shared helpers for the substance tests (owner: substance)."""
import numpy as np
import pytest
from goo.evidence import Provenance as P
from goo.film import Film, Bands
from goo.substance import Substance, Thermo, Rheo, Oscillators, Sphere, LogNormal
from goo.units import um, nm, mPa_s, mN

G = P("guessed")
NONE = P("exact")


def liquid(name="liq", n=1.40, eta=1 * mPa_s, rho=1000.0, **thermo):
    t = Thermo(**thermo)
    prov = dict(eps=G, rho=G, reservoir=NONE, **{"rheo.eta": G}, **{f"thermo.{k}": G for k in thermo})
    return Substance(name, Oscillators(n * n), None, rho, t, Rheo(eta=eta), prov=prov)


def particle(name="par", n=1.6, d=0.3 * um, sg=1.0, rho=2000.0, k=0.0):
    eps = Oscillators(n * n) if k == 0 else None
    return Substance(name, eps, Sphere(LogNormal(d, sg)), rho, prov=dict(eps=G, morph=G, rho=G, reservoir=NONE))


@pytest.fixture
def bands():
    return Bands.uniform().with_band(1550 * nm, 100 * nm)


@pytest.fixture
def small_bands():
    return Bands(np.array([450e-9, 550e-9, 650e-9]), np.array([20e-9, 20e-9, 20e-9]))
