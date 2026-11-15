"""What goo.light needs from matter: the Optical protocol (SPEC §4.4). Units in docs/LIGHT.md §7.

goo.light never reads a substance. It calls these four functions, passed in as an object, and reads nothing
else about a film or a ground but the film's thickness (Film.h) and the ground's optical depth (Ground.depth).

    optical(film, bands)          -> (K, S, n_host)   each [B, H, W]; K, S in 1/m; n_host dimensionless
    surface(film, bands)          -> (n_top, sigma)   n_top [B, H, W]; sigma [H, W] in m (rms roughness)
    ground_optical(ground, bands) -> (K, S, n_host)   of the ground's body, to its depth Ground.depth (m)
    ground_surface(ground, bands) -> (n_top, sigma)   of the ground's top interface

K and S are the two-flux coefficients (S with the asymmetry taken). Each returned item may be a plain array,
a goo.evidence.Derived or a goo.evidence.Bounded; the whole return may also be one Derived whose value is the
tuple. Any shape that broadcasts to the stated one is accepted. Values in cells where the film is empty
(h = 0) are not read.
"""
from typing import Protocol, runtime_checkable
import numpy as np
from goo.evidence import Derived, Bounded
from goo.ledger import Flag


@runtime_checkable
class Optical(Protocol):
    def optical(self, film, bands): ...
    def surface(self, film, bands): ...
    def ground_optical(self, ground, bands): ...
    def ground_surface(self, ground, bands): ...


def unwrap(ret, n, what, shapes, flags):
    """Plain float arrays from what a protocol function returned, broadcast to `shapes`; Derived values out of
    their domain and Bounded values collapsed (at mid) are recorded in `flags`."""
    if isinstance(ret, Derived):
        if not ret.everywhere_in_domain:
            flags.append(Flag("out_of_domain", ret.map or what, f"{what}: outside its map's domain in "
                              f"{int(np.size(ret.in_domain) - np.count_nonzero(ret.in_domain))} cells"))
        ret = ret.value
    ret = tuple(ret)
    if len(ret) != n:
        raise ValueError(f"{what} returned {len(ret)} items, the Optical protocol says {n}")
    out = []
    for i, (v, shp) in enumerate(zip(ret, shapes)):
        if isinstance(v, Bounded):
            v, note = v.collapse("mid")
            flags.append(Flag("bounds_collapsed", v.map or what, f"{what}[{i}]: {note}"))
        if isinstance(v, Derived):
            if not v.everywhere_in_domain:
                flags.append(Flag("out_of_domain", v.map or what, f"{what}[{i}]: outside its map's domain"))
            v = v.value
        a = np.asarray(v, np.float64)
        try:
            a = np.broadcast_to(a, shp)
        except ValueError:
            raise ValueError(f"{what}[{i}] has shape {a.shape}, expected {shp}") from None
        out.append(a)
    return out
