"""goo.classic.matter.substance: a substance is a point in the space of constitutive descriptions.

    Substance(name, dielectric, morphology, density, ...)   particles: a pigment, a filler, a void
    Liquid(name, dielectric, viscosity, density, ...)        a continuous phase: a carrier, a binder

It presents itself to the engine as response functions of its environment:

    sub.ks(n_s)            K, S (R, G, B, N) in goo units at surrounding index n_s   (goo.classic.matter.response)
    sub.mechanics(liquid)  phi_m, [eta], Krieger exponent, Stokes-Einstein D, Stokes settling in `liquid`
    sub.laws()             the admissibility verdicts (goo.classic.matter.laws); sub.klass: real | possible | impossible

Nothing here is typed in by hand that can be derived: K and S follow from the dielectric function and the
morphology; packing, thickening, diffusion and settling from the morphology and densities.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from goo.classic.matter import response, mechanics
from goo.classic.matter.dielectric import Dielectric


@dataclass(frozen=True)
class Provenance:
    """Where a description comes from: 'measured' (optical constants of a real material, with a citation),
    'computed' (from measured constants by a stated method), or 'fit' (fitted to reproduce a table: not
    provenance of a material). Only measured/computed ties a substance to something that exists."""
    kind: str
    citation: str
    material: str = ""

    @property
    def ties_to_material(self): return self.kind in ("measured", "computed") and bool(self.citation)


@dataclass(frozen=True)
class Change:
    """What a substance turns into: `product` (a name), at `rate` (1/s, of the remaining amount), with volume
    yields: `yield_volume` of product per unit volume consumed, plus volumes sent to named sinks (e.g. vapour)
    and drawn from named sources (e.g. oxygen). Irreversible unless `reversible`."""
    product: str
    rate: float
    yield_volume: float = 1.0
    sinks: tuple = ()                  # (("vapour", 1.0),)
    sources: tuple = ()                # (("oxygen", 0.1),)
    reversible: bool = False
    reverse_rate: float = 0.0


@dataclass
class Substance:
    name: str
    dielectric: Dielectric
    morphology: object = None          # None for a continuous phase
    density: float = 2.0               # g/cm^3 of the solid
    provenance: Provenance | None = None
    changes: tuple = ()
    claims: dict = field(default_factory=dict)   # e.g. {"nonreciprocal": True}
    note: str = ""

    # ---- response functions of the environment
    def table(self):
        if getattr(self, "_table", None) is None:
            self._table = response.ResponseTable(self.dielectric, self.morphology)
        return self._table

    def ks(self, n_s=1.48, exact=False):
        """K, S (4,) in goo units (per 0.025 mm of pure pigment), bands R, G, B, N, at surrounding index n_s
        (scalar, or one per band). exact: compute by Mie at n_s rather than interpolate."""
        if exact:
            ns = np.broadcast_to(np.asarray(n_s, float), (4,))
            if np.allclose(ns, ns[0]):
                K, S = response.km_bands(self.dielectric, self.morphology, [ns[0]]); return K[0], S[0]
            out = [response.km_bands(self.dielectric, self.morphology, [v]) for v in ns]
            return np.array([o[0][0][j] for j, o in enumerate(out)]), np.array([o[1][0][j] for j, o in enumerate(out)])
        return self.table()(n_s)

    def mechanics(self, liquid):
        return mechanics.particle_mechanics(self.morphology, self.density, liquid)

    # ---- admissibility
    def laws(self):
        from goo.classic.matter import laws
        return laws.verdicts(self)

    @property
    def klass(self):
        from goo.classic.matter import laws
        return laws.classify(self)

    def describe(self):
        return dict(name=self.name, dielectric=self.dielectric.describe(),
                    morphology=None if self.morphology is None else self.morphology.describe(),
                    density=self.density, provenance=None if self.provenance is None else self.provenance.__dict__,
                    claims=self.claims)


@dataclass
class Liquid(Substance):
    """A continuous phase. viscosity Pa s; p_sat Pa (0: non-volatile); M kg/mol; surface tension N/m."""
    viscosity: float = 1e-3
    p_sat: float = 0.0
    molar_mass: float = 0.018
    surface_tension: float = 0.072

    def index(self, lam_nm=None):
        """Refractive index per goo band (R, G, B, N) at the box centres, or at lam_nm."""
        lam = np.array([650.0, 550.0, 450.0, 1550.0]) if lam_nm is None else np.asarray(lam_nm, float)
        return self.dielectric.nk(lam).real

    def evaporation_m_s(self, **kw):
        return mechanics.evaporation(self.p_sat, self.molar_mass, self.density, **kw) if self.p_sat > 0 else 0.0

    def mechanics(self, liquid=None):
        return dict(viscosity=self.viscosity, density=self.density, evaporation_m_s=self.evaporation_m_s(),
                    evap_units_per_tick=self.evaporation_m_s() * mechanics.TICK_S / mechanics.CELL_M,
                    n_D=self.dielectric.n_at(589.3), surface_tension=self.surface_tension)
