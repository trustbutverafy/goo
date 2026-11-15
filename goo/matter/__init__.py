"""goo.matter: substances as points in a space of constitutive descriptions (docs/MATTER.md).

A substance is a dielectric function (goo.matter.dielectric) and, for particles, a morphology
(goo.matter.morphology); everything goo uses is derived from it by stated maps: K and S as functions of the
surrounding index by Mie theory (goo.matter.mie, goo.matter.response), packing, thickening, diffusion and
settling (goo.matter.mechanics). Every substance is labelled real, possible or impossible by the
admissibility laws (goo.matter.laws). goo.matter.classic holds the classic pigments as points;
goo.matter.phenotype makes paint in the engine and looks; goo.matter.gaps lists what the engine cannot
consume yet. Nothing here changes the engine.
"""
from goo.matter.dielectric import (Oscillators, Lorentz, Drude, Gauss, Tabulated, Free, WATER, LINSEED, AIR,
                                   sellmeier, HC)
from goo.matter.morphology import LogNormal, Sphere, Coated, Hollow, Shape, NotDerivable
from goo.matter.substance import Substance, Liquid, Provenance, Change
from goo.matter.response import ResponseTable, km_bands, km_spectral, cross_sections, NS_GRID, BANDS
from goo.matter import laws, mie, mechanics

__all__ = ["Oscillators", "Lorentz", "Drude", "Gauss", "Tabulated", "Free", "WATER", "LINSEED", "AIR", "sellmeier",
           "LogNormal", "Sphere", "Coated", "Hollow", "Shape", "NotDerivable", "Substance", "Liquid", "Provenance",
           "Change", "ResponseTable", "km_bands", "km_spectral", "cross_sections", "NS_GRID", "BANDS", "laws", "mie",
           "mechanics"]
