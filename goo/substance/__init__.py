"""goo.substance: descriptions, admissibility, derived maps, data (SPEC §4). Imports goo.units and goo.evidence only.

    Substance, Thermo, Rheo, Reaction, StateVar     the description (goo.substance.description)
    Oscillators, Lorentz, Drude, Gauss, Tabulated, Free, eV     dielectric functions (goo.substance.dielectric)
    LogNormal, Sphere, Coated, Hollow, Shape        morphologies (goo.substance.morphology)
    herglotz, stable, dissipative, reciprocal, stoichiometric, classify     admissibility (goo.substance.laws)
    goo.substance.maps                               the derived maps of SPEC §4.3
    goo.substance.library                            the library (imported on demand)

See docs/SUBSTANCES.md.
"""
from goo.substance.description import Substance, Thermo, Rheo, Reaction, StateVar
from goo.substance.dielectric import Oscillators, Lorentz, Drude, Gauss, Tabulated, Free, eV, sellmeier
from goo.substance.morphology import LogNormal, Sphere, Coated, Hollow, Shape, NotDerivable
from goo.substance.laws import (herglotz, stable, dissipative, reciprocal, stoichiometric, classify, verdicts,
                                Verdict, Classification, LAWS)
from goo.substance import maps

__all__ = ["Substance", "Thermo", "Rheo", "Reaction", "StateVar", "Oscillators", "Lorentz", "Drude", "Gauss",
           "Tabulated", "Free", "eV", "sellmeier", "LogNormal", "Sphere", "Coated", "Hollow", "Shape", "NotDerivable",
           "herglotz", "stable", "dissipative", "reciprocal", "stoichiometric", "classify", "verdicts", "Verdict",
           "Classification", "LAWS", "maps"]
