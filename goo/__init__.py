"""goo: a physical paint medium, a painter's tools, and the sensors that look at the result.

    World    goo.engine   Canvas: ground, strata, one wet layer of suspension; the laws; the ledger
    Tools    goo.brush    Brush, Stylus: bodies that touch the wet layer (drag, plough, exchange)
    Optics   goo.engine   Canvas.reflectance(): Kubelka-Munk, layer by layer, in R, G, B and 1.5 um
    Sensors  goo.sensors  lidar (incoherent spot), coherent (FMCW, speckle): see only the Surface

See SPEC.md for the contract between them.
"""
from goo.engine import Canvas, PIGMENTS, CARRIERS, GROUND_RGBN
from goo.brush import Brush, Stylus

__all__ = ["Canvas", "Brush", "Stylus", "PIGMENTS", "CARRIERS", "GROUND_RGBN"]
__version__ = "0.1.0"
