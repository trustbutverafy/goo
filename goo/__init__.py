"""goo: a physical paint medium, a painter's tools, and the sensors that look at the result.

    World    goo.engine   Canvas: ground, strata, one wet layer of suspension; the laws; the ledger
    Tools    goo.brush    Brush, Stylus: bodies that touch the wet layer (drag, plough, exchange)
    Optics   goo.optics   the canvas's optics model, read through Canvas.reflectance() and finish(): "classic"
                          (default; Kubelka-Munk, layer by layer, R, G, B and 1.5 um) or "v2" (opt-in)
             goo.surface  Surface.from_canvas(): the World presented to sensors (height, reflectance, roughness, gloss)
    Sensors  goo.sensors  lidar (incoherent spot), coherent (FMCW, speckle), relight (visible light): see only the Surface

    Paints   goo.paints   Paint: a recipe in a medium; preview, mix_for, swatch cards
    Studio   goo.studio   acts, receipts, a journal, rehearsal on forks; goo.easel: the same from a shell
    Data     goo.data     optical constants computed from open sources, each with its provenance
    Matter   goo.matter   substances as descriptions (dielectric function, morphology), derived K and S, laws

See SPEC.md for the contract between them.
"""
from goo.engine import Canvas, PIGMENTS, CARRIERS, GROUND_RGBN
from goo.brush import Brush, Stylus
from goo.surface import Surface

__all__ = ["Canvas", "Brush", "Stylus", "Surface", "PIGMENTS", "CARRIERS", "GROUND_RGBN"]
__version__ = "0.1.0"
