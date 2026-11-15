"""Grounds by name (docs/PORTS.md §6). A ground is a porous solid substance from the library with a porosity, a
pore radius and a depth. The named grounds are rows of data with provenance, not branches: any library substance
can be a ground's solid with explicit numbers."""
from dataclasses import dataclass
import numpy as np
from goo.evidence import Provenance
from goo.film import Ground
from goo.substance import library
from goo.units import um


@dataclass(frozen=True)
class GroundSpec:
    solid: str
    porosity: float
    pore_radius: float
    depth: float
    prov: Provenance


GROUNDS = {
    "gesso": GroundSpec("calcite", 0.30, 1 * um, 300 * um, Provenance(
        "guessed", note="a chalk gesso: porosity 0.3, pore radius 1 um, 300 um deep. Order of magnitude for a chalk "
                        "and glue ground; not measured")),
    "paper": GroundSpec("cellulose", 0.50, 5 * um, 200 * um, Provenance(
        "guessed", note="an unsized cotton paper: porosity 0.5, pore radius 5 um, 200 um thick. Order of magnitude; "
                        "not measured")),
    "canvas": GroundSpec("cellulose", 0.60, 20 * um, 400 * um, Provenance(
        "guessed", note="an unprimed cotton or linen canvas taken as a porous cellulose sheet: porosity 0.6, pore "
                        "radius 20 um, 400 um thick. The weave is not modelled (no relief); not measured")),
}


def make_ground(name, H, W, solid=None, porosity=None, pore_radius=None, depth=None):
    """(Ground, description) from a named ground or a library substance, with any number overridden. The description
    says where each number came from (the named ground's provenance, or 'given')."""
    if name in GROUNDS:
        spec = GROUNDS[name]
    else:
        if name not in library.BY_NAME:
            raise ValueError(f"ground {name!r}: one of {', '.join(GROUNDS)} or a library substance "
                             f"({', '.join(library.BY_NAME)})")
        if None in (porosity, pore_radius, depth):
            raise ValueError(f"ground {name!r} is a bare substance: give porosity, pore radius and depth")
        spec = GroundSpec(name, porosity, pore_radius, depth, Provenance("assumed", note="given by the painter"))
    vals = dict(solid=solid or spec.solid, porosity=spec.porosity if porosity is None else float(porosity),
                pore_radius=spec.pore_radius if pore_radius is None else float(pore_radius),
                depth=spec.depth if depth is None else float(depth))
    if not 0 <= vals["porosity"] < 1 or vals["pore_radius"] <= 0 or vals["depth"] <= 0:
        raise ValueError("ground: porosity in [0, 1), pore radius and depth > 0")
    given = [k for k, v in dict(solid=solid, porosity=porosity, pore_radius=pore_radius, depth=depth).items()
             if v is not None]
    g = Ground(np.zeros((H, W)), library.get(vals["solid"]), vals["porosity"], vals["pore_radius"], vals["depth"])
    desc = dict(name=name, **vals, evidence=spec.prov.evidence, note=spec.prov.note, given=given)
    return g, desc
