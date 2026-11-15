"""goo.classic.matter: substances as points in a space of constitutive descriptions (classic/docs/MATTER.md).

A substance is a dielectric function (goo.classic.matter.dielectric) and, for particles, a morphology
(goo.classic.matter.morphology); everything goo uses is derived from it by stated maps: K and S as functions of the
surrounding index by Mie theory (goo.classic.matter.mie, goo.classic.matter.response), packing, thickening, diffusion and
settling (goo.classic.matter.mechanics). Every substance is labelled real, possible or impossible by the
admissibility laws (goo.classic.matter.laws). goo.classic.matter.classic holds the classic pigments as points;
goo.classic.matter.phenotype makes paint in the engine and looks; goo.classic.matter.gaps lists what the engine cannot
consume yet. Nothing here changes the engine.
"""
from goo.classic.matter.dielectric import (Oscillators, Lorentz, Drude, Gauss, Tabulated, Free, WATER, LINSEED, AIR,
                                   sellmeier, HC)
from goo.classic.matter.morphology import LogNormal, Sphere, Coated, Hollow, Shape, NotDerivable
from goo.classic.matter.substance import Substance, Liquid, Provenance, Change
from goo.classic.matter.response import ResponseTable, km_bands, km_spectral, cross_sections, NS_GRID, BANDS
from goo.classic.matter import laws, mie, mechanics

__all__ = ["Oscillators", "Lorentz", "Drude", "Gauss", "Tabulated", "Free", "WATER", "LINSEED", "AIR", "sellmeier",
           "LogNormal", "Sphere", "Coated", "Hollow", "Shape", "NotDerivable", "Substance", "Liquid", "Provenance",
           "Change", "ResponseTable", "km_bands", "km_spectral", "cross_sections", "NS_GRID", "BANDS", "laws", "mie",
           "mechanics"]
