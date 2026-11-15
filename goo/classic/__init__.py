"""goo.classic: a physical paint medium, a painter's tools, and the sensors that look at the result.

    World    goo.classic.engine   Canvas: ground, strata, one wet layer of suspension; the laws; the ledger
    Tools    goo.classic.brush    Brush, Stylus: bodies that touch the wet layer (drag, plough, exchange)
    Optics   goo.classic.optics   the canvas's optics model, read through Canvas.reflectance() and finish(): "classic"
                                  (default; Kubelka-Munk, layer by layer, R, G, B and 1.5 um) or "v2" (opt-in)
             goo.classic.surface  Surface.from_canvas(): the World presented to sensors (height, reflectance, roughness, gloss)
    Sensors  goo.classic.sensors  lidar (incoherent spot), coherent (FMCW, speckle), relight (visible light): see only the Surface

    Paints   goo.classic.paints   Paint: a recipe in a medium; preview, mix_for, swatch cards
    Studio   goo.classic.studio   acts, receipts, a journal, rehearsal on forks; goo.classic.easel: the same from a shell
    Data     goo.classic.data     optical constants computed from open sources, each with its provenance
    Matter   goo.classic.matter   substances as descriptions (dielectric function, morphology), derived K and S, laws

See classic/SPEC.md for the contract between them.
"""
import warnings

warnings.warn("goo.classic is goo's first engine, frozen at 6fcca08 and no longer developed; the kernel is goo. "
              "See classic/SPEC.md.", DeprecationWarning, stacklevel=2)

from goo.classic.engine import Canvas, PIGMENTS, CARRIERS, GROUND_RGBN
from goo.classic.brush import Brush, Stylus
from goo.classic.surface import Surface

__all__ = ["Canvas", "Brush", "Stylus", "Surface", "PIGMENTS", "CARRIERS", "GROUND_RGBN"]
__version__ = "0.1.0"
