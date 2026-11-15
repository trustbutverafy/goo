"""Recipes: binders, paints, grounds and the room, every one built from substances (docs/LAB.md §1).

Nothing about a medium is typed in. "Linseed oil", "water with gum arabic" and "acrylic latex in water" are the names
of recipes: volume fractions of library substances. What each does when it dries comes from the substances'
descriptions acted on by the maps (goo.lab.dry). The recipe numbers (fractions, thicknesses, loadings) are choices of
a drawdown, not measurements of any product: each has a Provenance marked `assumed`.
"""
from dataclasses import dataclass, field
import numpy as np
from goo.units import um
from goo.evidence import Provenance
from goo.film import Film, Ground, Stack, Bands, Environment
from goo.substance import library as lib

RECIPE = Provenance("assumed", note="a recipe choice of the drawdown, not a measured product")


@dataclass(frozen=True)
class Binder:
    """A vehicle: ((substance, volume fraction of the vehicle), ...), and the pigment volume fraction of the wet paint
    a masstone is made at."""
    name: str
    parts: tuple
    pvc: float
    note: str = ""
    prov: Provenance = RECIPE

    def __post_init__(self):
        if abs(sum(f for _, f in self.parts) - 1.0) > 1e-12 or any(f <= 0 for _, f in self.parts):
            raise ValueError(f"{self.name}: vehicle fractions must be positive and sum to 1")
        if not 0 < self.pvc < 1:
            raise ValueError(f"{self.name}: pvc in (0, 1)")

    @property
    def substances(self):
        return tuple(s for s, _ in self.parts)


OIL = Binder("linseed oil", ((lib.LINSEED, 1.0),), 0.20, "raw linseed oil; cures by its reaction with oxygen")
WATERCOLOUR = Binder("water with gum arabic", ((lib.WATER, 0.80), (lib.GUM_ARABIC, 0.20)), 0.10,
                     "a gum solution; dries by losing water")
ACRYLIC = Binder("acrylic latex in water", ((lib.WATER, 0.55), (lib.PMMA_LATEX, 0.45)), 0.12,
                 "PMMA latex particles in water; goo has no coalescence term, so they stay particles")
WATER = Binder("water", ((lib.WATER, 1.0),), 0.20, "particles in water alone, no binder")
BINDERS = (OIL, WATERCOLOUR, ACRYLIC)
BY_NAME = {b.name: b for b in BINDERS + (WATER,)}


@dataclass(frozen=True)
class Paint:
    """A paint recipe: pigments ((substance, fraction of the pigment volume), ...), a binder, and the pigment volume
    fraction of the wet paint (None: the binder's)."""
    name: str
    pigments: tuple
    binder: Binder = None
    pvc: float = None

    def __post_init__(self):
        if abs(sum(f for _, f in self.pigments) - 1.0) > 1e-12:
            raise ValueError(f"{self.name}: pigment fractions must sum to 1")

    @classmethod
    def of(cls, substance, binder=None, pvc=None):
        return cls(substance.name, ((substance, 1.0),), binder, pvc)


WHITE = lib.RUTILE
WET_THICKNESS = 50 * um
BLACK_STRIPE = {lib.PMMA: 19 * um, lib.SOOT: 1 * um}     # a soot-in-acrylic-resin film: 5 % soot by volume, 20 um
ROOM = Environment(activity={"water": 0.5, "oxygen": 0.21})
ROOM_PROV = Provenance("assumed", note="20 C, 50 % relative humidity, oxygen at mole fraction 0.21 (recalled)")


def wet_film(paint, binder, white_share=0.0, h=WET_THICKNESS, white=WHITE, cells=None):
    """The wet film of `paint` in `binder`, with `white_share` of its pigment volume replaced by the white. With
    `cells` (a sequence of (white_share, h) pairs), one cell each, in a row."""
    cells = [(white_share, h)] if cells is None else list(cells)
    pvc = paint.pvc if paint.pvc is not None else binder.pvc
    subs, amounts = [], {}

    def add(s, q, j):
        if s not in amounts:
            subs.append(s); amounts[s] = np.zeros(len(cells))
        amounts[s][j] += q

    has_white = any(w > 0 for w, _ in cells)
    for j, (w, hj) in enumerate(cells):
        for s, f in paint.pigments:
            add(s, pvc * hj * (1 - w) * f, j)
        if has_white:
            add(white, pvc * hj * w, j)
        for s, f in binder.parts:
            add(s, (1 - pvc) * hj * f, j)
    q = np.array([amounts[s][None, :] for s in subs])
    return Film(tuple(subs), q)


def white_ground(W=1):
    """The white: a calcite ground (chalk gesso), bare. 30 % porosity, 1 um pores, 300 um deep (assumed)."""
    return Ground(np.zeros((1, W)), lib.CALCITE, 0.3, 1 * um, 300 * um)


def black_film(W=1, present=None):
    """The black stripe: soot in PMMA, 20 um, on the white ground. `present` [W] bool: where it is laid."""
    p = np.ones(W, bool) if present is None else np.asarray(present, bool)
    subs = tuple(BLACK_STRIPE)
    q = np.array([np.where(p, BLACK_STRIPE[s], 0.0)[None, :] for s in subs])
    return Film(subs, q)


GROUND_PROV = Provenance("assumed", note="white: calcite ground, porosity 0.3, pore radius 1 um, 300 um deep; black: "
                                         "20 um of PMMA with 5 % soot by volume over it. Real stacks, not typed "
                                         "reflectances")


def stack(paint_film, ground_kind, extra_below=()):
    """A drawdown stack: the paint film over 'white' (the bare ground) or 'black' (the black stripe on it)."""
    W = paint_film.q.shape[2]
    films = list(extra_below)
    if ground_kind == "black":
        films.append(black_film(W))
    elif ground_kind != "white":
        raise ValueError("ground: 'black' or 'white'")
    return Stack(white_ground(W), films + [paint_film], 1e-3)


def bands(ir=(1000e-9, 1500e-9), ir_width=100e-9, step=20e-9):
    """The lab's bands: 380-740 nm in `step` bands, and infrared bands."""
    b = Bands.uniform(380e-9, 740e-9, step)
    for c in ir:
        b = b.with_band(c, ir_width)
    return b


LAB_BANDS = bands()
