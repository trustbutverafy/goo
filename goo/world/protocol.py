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
    exchange(film, env)      (k [S,H,W] m/(Pa s), mu_env [S] J/m^3, activity [S,H,W])   k = 0: not volatile;
                             mu_env: the reservoir's potential (the map turns env.activity into it)
    wetting(film, ground)    (S, ell) [H,W]   J/m^2, m   spreading coefficient S <= 0; wetting length ell > 0
    sorption(film, ground)   (P, kappa) [S,H,W]  Pa, m/(Pa s)   suction into the ground's pores; rate
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import numpy as np
from goo.evidence import Derived, Bounded
from goo.film import Environment          # the environment type (SPEC §5); re-exported


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


def unwrap(v, name, flags):
    """The value of a map's answer as arrays. Derived: its value, flagging out-of-domain use. Bounded: collapsed
    at mid with the note Bounded.collapse returns (SPEC §2), flagged bounds_collapsed. A tuple (or named tuple)
    is unwrapped item by item. `flags` collects (kind, subject, detail)."""
    if v is None:
        return None
    if isinstance(v, Bounded):
        d, note = v.collapse("mid")
        flags.append(("bounds_collapsed", name, note))
        return unwrap(d, name, flags)
    if isinstance(v, Derived):
        if not v.everywhere_in_domain:
            flags.append(("out_of_domain", name, f"{name} used outside its domain ({v.map or 'no map named'})"))
        return unwrap(v.value, name, flags)
    if isinstance(v, tuple):                       # a named tuple keeps its type, so its items keep their names
        items = [unwrap(x, name, flags) for x in v]
        return type(v)(*items) if hasattr(v, "_fields") else tuple(items)
    if isinstance(v, Mixing):
        return Mixing(*(unwrap(getattr(v, f), name, flags) for f in ("f", "mu", "hess", "f_x", "mu_x")))
    return v


def call(constitutive, name, *args, flags=None):
    fn = getattr(constitutive, name, None)
    if fn is None:
        return None
    return unwrap(fn(*args), name, flags if flags is not None else [])
