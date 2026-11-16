"""A constitutive wrapper for timings while goo.substance's particle thermodynamics are being replaced: the library's
maps (goo.substance.maps) with the mobility of every particle substance (one with a morphology) set to zero, so
particles do not interdiffuse (with the present regular-solution interaction they do, at an effective diffusivity near
1e-4 m^2/s, and a quiet megapixel step never finishes). Every other map is the library's. Benchmarks only."""
import numpy as np
from goo.evidence import Derived
from goo.substance import maps as _maps


class NoParticleMobility:
    def __getattr__(self, name):
        return getattr(_maps, name)

    def mobility(self, film):
        d = _maps.mobility(film)
        part = np.array([s.morph is not None for s in film.substances])[:, None, None]
        return Derived(np.where(part, 0.0, d.value), d.evidence, d.in_domain, d.map + " (particles held: bench wrapper)")


maps = NoParticleMobility()
