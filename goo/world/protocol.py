"""What the world needs to know about matter (SPEC §4.4): the protocol `Constitutive`, and the environment.

The world reaches substances only through an object satisfying `Constitutive`. Every function takes a
`goo.film.Film` (any shape [S, H, W]; the world calls them on windows and on compacted films of shape
[S, 1, n]) and must be element-wise over cells: a cell's answer depends on that cell alone. Each returns a
`goo.evidence.Derived` (or a bare array, read as `assumed`). Values must be finite on every cell, including
empty ones (the world uses them as coefficients for matter that may arrive and masks them where it does not).

A function that is absent (or returns None) means the term is absent. A term whose parameter is zero
everywhere is treated exactly as absent (SPEC §6, O8). See docs/DYNAMICS.md for where each one enters.

    viscosity(film)          eta [H,W]        Pa s       required wherever there is flow
    yield_stress(film)       tau_y [H,W]      Pa         >= 0; absent: Newtonian
    surface_tension(film)    gamma [H,W]      J/m^2      lagged per step
    mixing(film, T)          Mixing           J/m^2, J/m^3, J/m^4   f homogeneous of degree 1 in q
    mobility(film)           Lambda [S,H,W]   m^2/(Pa s) = D v/(k_B T), the Stokes mobility (Stokes-Einstein)
    exchange(film, env)      k [S,H,W]        m/(Pa s)   rate coefficient against env.potentials; 0: not volatile
    wetting(film, ground)    (S, ell) [H,W]   J/m^2, m   spreading coefficient S <= 0; wetting length ell > 0
    sorption(film, ground)   (P, kappa) [S,H,W]  Pa, m/(Pa s)   suction into the ground's pores; rate
"""
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
import numpy as np
from goo.evidence import Derived


@dataclass(frozen=True)
class Mixing:
    """The mixing free energy of each cell and its derivatives. `f`, `mu`, `hess` are the convex part
    (treated implicitly); `f_x`, `mu_x` an optional concave part (explicit). Total f + f_x, mu + mu_x.
    f [H,W] J/m^2; mu [S,H,W] J/m^3 (= df/dq_i); hess [S,S,H,W] J/m^4. Where a substance is absent its mu may
    be -inf; the world never uses it there."""
    f: np.ndarray
    mu: np.ndarray
    hess: np.ndarray
    f_x: np.ndarray = None
    mu_x: np.ndarray = None


@runtime_checkable
class Constitutive(Protocol):
    def viscosity(self, film) -> Derived: ...


@dataclass(frozen=True)
class Environment:
    """Reservoirs at fixed potential. T in K. gravity (g_x, g_y, g_n) in m/s^2: g_x along columns, g_y down
    rows, g_n >= 0 pressing the film onto the panel. potentials: substance name -> mu (J/m^3 of substance)
    of the atmosphere (k_B T / v ln a for a volatile at activity a)."""
    T: float = 293.15
    gravity: tuple = (0.0, 0.0, 9.80665)
    potentials: dict = field(default_factory=dict)

    def __post_init__(self):
        g = tuple(float(x) for x in self.gravity)
        if len(g) != 3:
            raise ValueError("gravity: (g_x, g_y, g_n)")
        if g[2] < 0:
            raise ValueError("g_n < 0 (a film under a ceiling) is Rayleigh-Taylor unstable and not on this rung")
        object.__setattr__(self, "gravity", g)
        object.__setattr__(self, "potentials", dict(self.potentials))


def unwrap(v, name, flags):
    """(array, evidence) of a map's answer; records out-of-domain use in `flags` (a list of (map, detail))."""
    if v is None:
        return None
    if isinstance(v, Derived):
        if not v.everywhere_in_domain:
            flags.append((name, f"{name} used outside its domain ({v.map or 'no map named'})"))
        return v.value
    return v


def call(constitutive, name, *args, flags=None):
    fn = getattr(constitutive, name, None)
    if fn is None:
        return None
    return unwrap(fn(*args), name, flags if flags is not None else [])
