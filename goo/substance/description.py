"""The description of a substance (SPEC §4.1): what the engine may read about it, and nothing else.

    Substance(name, eps, morph, rho, thermo, rheo, reactions, reservoir, prov, state, material)

Every number is SI. Every parameter that is given has a Provenance in `prov`, keyed by its field ("eps", "morph",
"rho", "reservoir", "reactions", "thermo.<name>", "rheo.<name>"); a parameter left None makes no claim, needs no
provenance, and every map that would need it marks the cells where the substance is present out of its domain (or
says in its docstring what it does instead). A substance is immutable and hashable, so a Film can hold it in a
tuple and a map can cache what it precomputes for it.

Thermo: its terms in the free energy F (all at the reference temperature T_ref = goo.units.T_room, 20 C)

  M          kg/mol    molar mass of a molecular substance. Its molecular volume M / (rho N_A) is the mixing unit of
                       the ideal (Flory-Huggins) entropy. None: a particle population (the unit is the mean particle
                       volume from morph) or a macromolecular phase (no mixing entropy of its own: the limit v -> inf).
  p_sat      Pa        saturation vapour pressure at T_ref; None or 0 for a non-volatile substance
  dH_vap     J/mol     enthalpy of vaporisation (Clausius-Clapeyron for p_sat(T); the cohesive energy density
                       (dH_vap - R T) rho / M of a volatile liquid is DERIVED from it)
  cohesion   J/m^3     cohesive energy density (the square of the Hildebrand solubility parameter) of a substance
                       that does not evaporate. Typing it for a substance that has dH_vap and M is refused: it is
                       derivable. Interaction energy density between i and j: (sqrt(c_i) - sqrt(c_j))^2 phi_i phi_j
                       (Hildebrand-Scatchard regular solution). None: no interaction terms.
  cohesion_p J/m^3     the polar part of the cohesive energy density (Hansen's delta_p^2), and
  cohesion_h J/m^3     its hydrogen-bonding part (delta_h^2); the dispersive part is the rest, cohesion - p - h
                       (DERIVED). Given for both of a pair of liquids, their interaction energy density is the
                       Hansen distance (sqrt d_i - sqrt d_j)^2 + (1/4)(sqrt p_i - sqrt p_j)^2 + (1/4)(sqrt h_i -
                       sqrt h_j)^2 instead of Hildebrand's (hydrogen bonds that both make cost nothing to mix).
  gamma      J/m^2     surface tension of a liquid against air; for a solid, its surface free energy against air
  gamma_d    J/m^2     the dispersive (London) part of gamma (Fowkes; Owens-Wendt): the rest, gamma - gamma_d, is
                       the polar part. Interfacial tension between a and b: gamma_a + gamma_b - 2 sqrt(gd_a gd_b)
                       - 2 sqrt(gp_a gp_b) (geometric-mean rule), which gives wetting and capillary suction
  D_vapour   m^2/s     diffusivity of its vapour in air at T_ref (evaporation through a still-air boundary layer)

Rheo: its terms in the dissipation potential Psi*

  eta        Pa s      Newtonian viscosity at T_ref of a continuous phase (Psi* = (1/2) X^2 / eta per unit volume of
                       flow in the force X conjugate to the shear rate: convex iff eta > 0)
  E_eta      J/mol     viscous activation energy: eta(T) = eta exp(E_eta / R (1/T - 1/T_ref)) (Arrhenius-Andrade)
  separation m         closest approach of two particles of this population (an adsorbed or steric layer): with the
                       Hamaker constant derived from eps, it sets the attraction that gives a yield stress
  phi_percolation 1    the particle volume fraction at which an attractive network first spans (YODEL's phi_0)
  glass      1         a binder's glass composition: the volume fraction of the other liquids around it (its
                       solvent) at or below which its solution is a glass at T_ref: the liquids' viscosity diverges
                       there (+inf: it no longer flows; goo.world's settle() retires it)
  coupling   as declared   symmetric Onsager matrix among the rates of the declared state variables (tuple of
                       tuples); its symmetric part must be positive semi-definite (dissipative) and it must equal its
                       transpose (reciprocal)

Reactions: Reaction(stoich=((name, kg per kg of extent), ...), rate (1/s at T_ref), E_a (J/mol), dG (J per kg of
extent), M_reservoir=((reservoir name, kg/mol), ...)). Reactants carry negative coefficients, products positive;
names are substances or environment reservoirs ("oxygen"). The substance itself must be a reactant. Mass is
conserved iff the coefficients sum to zero. dG is the free energy of reaction per kg of extent at T_ref with every
film substance pure and every reservoir at activity 1 (None: no claim; the affinity is then not derived);
M_reservoir gives the molar mass of each reservoir it draws, for the reservoir's potential (R T / M) ln a per kg.
Provenance: "reactions" for the stoichiometry and rate, "reactions.dG" for dG and M_reservoir.

State: StateVar(name, unit, lo, hi): internal state variables the substance declares (the film's `state` holds them).

reservoir: J/m^3 of the substance: the finite energy an impossible substance may draw when a failure yields power.
0 for every admissible substance.
"""
from dataclasses import dataclass, field, fields
import numpy as np
from goo.units import T_room, R_gas, N_A
from goo.evidence import Provenance, weakest


@dataclass(frozen=True)
class Thermo:
    M: float = None
    p_sat: float = None
    dH_vap: float = None
    cohesion: float = None
    cohesion_p: float = None
    cohesion_h: float = None
    gamma: float = None
    gamma_d: float = None
    D_vapour: float = None

    def __post_init__(self):
        if self.cohesion is not None and self.dH_vap is not None and self.M is not None:
            raise ValueError("Thermo: cohesion is derivable from dH_vap, M and rho; do not type it in")


@dataclass(frozen=True)
class Rheo:
    eta: float = None
    E_eta: float = None
    separation: float = None
    phi_percolation: float = None
    glass: float = None
    coupling: tuple = None


@dataclass(frozen=True)
class Reaction:
    stoich: tuple                      # ((name, kg per kg of extent), ...)
    rate: float                        # 1/s at T_ref, first order in the reacting substance
    E_a: float = 0.0                   # J/mol
    dG: float = None                   # J per kg of extent at T_ref, film substances pure, reservoirs at activity 1
    M_reservoir: tuple = ()            # ((reservoir name, kg/mol), ...)

    def coefficient(self, name):
        return sum(v for n, v in self.stoich if n == name)


@dataclass(frozen=True)
class StateVar:
    name: str
    unit: str
    lo: float = -np.inf
    hi: float = np.inf


def _params(obj):
    return {f.name: getattr(obj, f.name) for f in fields(obj) if getattr(obj, f.name) is not None}


@dataclass(frozen=True)
class Substance:
    name: str
    eps: object
    morph: object
    rho: float
    thermo: Thermo = Thermo()
    rheo: Rheo = Rheo()
    reactions: tuple = ()
    reservoir: float = 0.0
    prov: tuple = ()                   # ((key, Provenance), ...); a dict is accepted and frozen
    state: tuple = ()
    material: str = ""                 # the named real material the description is computed or checked against

    def __post_init__(self):
        p = self.prov
        if isinstance(p, dict):
            p = tuple(sorted(p.items()))
            object.__setattr__(self, "prov", p)
        object.__setattr__(self, "reactions", tuple(self.reactions))
        object.__setattr__(self, "state", tuple(self.state))
        keys = dict(p)
        for k, v in keys.items():
            if not isinstance(v, Provenance):
                raise ValueError(f"{self.name}: prov[{k!r}] is not a Provenance")
        need = set(self.fields_present())
        missing, extra = need - set(keys), set(keys) - need
        if missing:
            raise ValueError(f"{self.name}: no provenance for {sorted(missing)}")
        if extra:
            raise ValueError(f"{self.name}: provenance for fields it does not have: {sorted(extra)}")
        if not np.isfinite(self.reservoir) or self.reservoir < 0:
            raise ValueError(f"{self.name}: reservoir must be finite and >= 0 J/m^3")

    def fields_present(self):
        """The keys of `prov` this description must carry: every field and parameter it gives."""
        out = ["eps", "rho", "reservoir"]
        if self.morph is not None:
            out.append("morph")
        if self.reactions:
            out.append("reactions")
        if any(r.dG is not None or r.M_reservoir for r in self.reactions):
            out.append("reactions.dG")
        out += [f"thermo.{k}" for k in _params(self.thermo)]
        out += [f"rheo.{k}" for k in _params(self.rheo)]
        return out

    def provenance(self, key):
        return dict(self.prov)[key]

    def evidence(self, *keys):
        """The weakest evidence of the given keys that this description has (keys it lacks make no claim)."""
        d = dict(self.prov)
        got = [d[k].evidence for k in keys if k in d]
        return weakest(*got) if got else "exact"

    # ---- quantities derived from the description alone (no film): each says what it reads
    @property
    def is_particle(self):
        return self.morph is not None

    @property
    def molecular_volume(self):
        """m^3: the mixing unit. M / (rho N_A) for a molecular substance, the mean particle volume for a particle
        population, inf for a macromolecular phase."""
        if self.morph is not None:
            return float(self.morph.size.particle_volume())
        if self.thermo.M is not None:
            return self.thermo.M / (self.rho * N_A)
        return np.inf

    def molecular_volume_keys(self):
        return ("morph",) if self.morph is not None else ("thermo.M", "rho")

    @property
    def cohesion(self):
        """J/m^3: (dH_vap - R T_ref) rho / M for a volatile liquid, else the stated cohesion, else None."""
        t = self.thermo
        if t.dH_vap is not None and t.M is not None:
            return (t.dH_vap - R_gas * T_room) * self.rho / t.M
        return t.cohesion

    def cohesion_keys(self):
        t = self.thermo
        if t.dH_vap is not None and t.M is not None:
            return ("thermo.dH_vap", "thermo.M", "rho")
        return ("thermo.cohesion",)

    @property
    def cohesion_parts(self):
        """(dispersive, polar, hydrogen-bonding) parts of the cohesive energy density, J/m^3, or None when the
        polar and hydrogen-bonding parts are not both given. The dispersive part is derived: cohesion - p - h."""
        t = self.thermo
        c = self.cohesion
        if c is None or t.cohesion_p is None or t.cohesion_h is None:
            return None
        return (c - t.cohesion_p - t.cohesion_h, t.cohesion_p, t.cohesion_h)

    def cohesion_parts_keys(self):
        return self.cohesion_keys() + ("thermo.cohesion_p", "thermo.cohesion_h")

    @property
    def volatile(self):
        return bool(self.thermo.p_sat) and self.thermo.p_sat > 0

    def __repr__(self):
        return f"Substance({self.name!r})"
