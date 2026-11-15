"""goo.classic.studio: an instrument for the one holding the brush (classic/SPEC.md §1, the Painter domain).

A `Studio` paints directly, without a motif, in the vocabulary of a painting manual: wet the paper, lay a
graded wash, drop colour in, let it go damp, touch in clean water for a backrun, dry it, glaze over it. Each
technique is a physical act: it chooses and sizes its own brush, charges it, lays strokes that run off the
panel where its region meets the panel's edge and keep the brush's rim inside the region's inner edges,
rinses the brush afterwards, and returns a `Receipt` of what it laid. A technique that lays nothing says why
(`NothingLaid`): the region is too small for any brush, the brush was lifted everywhere, or the medium is wrong.

The studio is painter code. Of the canvas it reads only what optics show and tools and `fix` return:
`reflectance()`, `sheen()`, `touch_dry()`, `finish()`, `W`, `H`; it changes the canvas only through tools,
`fix`, `wait`, `mask` / `unmask`, and it may `fork()` it to rehearse (`classic/tests/test_studio.py` enforces this with
the canvas behind glass). Raking light goes through `goo.classic.sensors.relight`, a camera that sees a `Surface`.

    st = Studio.new(400, 300, ["ultramarine", "rose", "ochre"], medium="watercolour")
    sky = band(0, 172)
    st.wet(sky)
    st.graded_wash(sky, "ultramarine", "2 rose + ochre")
    st.dry("touch")
    st.look()                                   # an sRGB photograph, (H, W, 3) uint8

Everything here is in goo's units (SPEC §2): positions and widths in cells (0.25 mm), thicknesses in volume
units (0.25 mm), time in ticks. Defaults are a painter's: a "wet" wash lays about half a unit of water.

Brushes come from `goo.classic.brush` (round and flat), the knife, the blotter and spatter from `goo.classic.tools`, paints
and colour by intent from `goo.classic.paints`; a reserve is the World's frisket (`Canvas.mask`), physical: no law
moves paint under it.
"""
from __future__ import annotations
import functools, hashlib, inspect, json, os, pickle, re, textwrap
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage
from goo.classic.engine import Canvas, CARRIERS, SHEEN_GLISTENING as GLISTEN, SHEEN_DAMP as DAMP
from goo.classic.brush import Brush
from goo.classic import tools as _tools
from goo.classic import paints as _paints
from goo.classic.paints import Paint

# GLISTEN, DAMP: how the studio reads sheen(), named once in goo.classic.engine (SHEEN_GLISTENING, SHEEN_DAMP)
RUNOFF = 0.5         # a stroke that meets the panel's edge runs on until half the brush is off it
MARGIN = 3           # cells: how far beyond its region a technique changes the painting (the soft rim, a little flow);
                     # wet-in-wet blooms and lifting from a wet wash reach a few cells further (classic/tests/test_studio.py)
WIDTHS = (1.5, 2.5, 3.5, 5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0, 46.0)   # the brush rack, cells across


class NothingLaid(RuntimeError):
    """A technique laid (or, for lift / blot / scrape, took up) nothing. The message says why and what to change."""


class StudioError(ValueError):
    """A request the studio cannot carry out as written. The message says what to change."""


# =====================================================================================================
# The World, tools and paints the studio uses
# =====================================================================================================
def _fork(cv):
    """An independent canvas in the same state (Canvas.fork: a rehearsal)."""
    return cv.fork()


def _new_brush(cv, width, kind, **kw):
    """A tool from the rack's recipes: the knife, the blotter, a flat (`flat=True`) or a round brush."""
    if kind == "knife": return _tools.Knife(cv, width)
    if kind == "blotter": return _tools.Blotter(cv, width)
    if kw.pop("flat", False): return Brush(cv, width, shape="flat", aspect=3.0, **kw)
    return Brush(cv, width, **kw)


def _make_paint(medium, mix, **kw):
    """A paint by recipe through goo.classic.paints's constructors."""
    if medium == "water": return _paints.water()
    if medium not in ("watercolour", "gouache", "acrylic", "oil"):
        raise StudioError(f"unknown medium {medium!r}: use watercolour, gouache, acrylic, oil or water")
    try:
        return getattr(_paints, medium)(mix, **kw)
    except (TypeError, ValueError, KeyError) as e:
        raise StudioError(f"cannot make {medium} from {mix!r} with {kw!r}: {e}") from None


class _Thirst:
    """No paint at all: a squeezed-out brush (it gives nothing, and takes)."""
    medium = "water"; permanent = False
    def load(self, volume): return {}
    def __repr__(self): return "squeezed brush"


# =====================================================================================================
# Regions: where a technique works. A Region is a JSON-able description; its cells come from the panel size.
# =====================================================================================================
class Region:
    """Cells of the panel, by description. Build with `disc`, `band`, `rect`, `polygon`, `below`, `above`,
    `everywhere`; combine with `|` (or), `&` (and), `-` (minus), `~` (not); `grow(n)` / `shrink(n)` in cells
    (shrinking never eats in from the panel's edge). Coordinates are cells: x to the right, y down."""

    def __init__(self, spec):
        self.spec = spec

    def mask(self, H, W):
        return _render(self.spec, H, W)

    def __or__(self, o): return Region({"or": [self.spec, coerce_region(o).spec]})
    def __and__(self, o): return Region({"and": [self.spec, coerce_region(o).spec]})
    def __sub__(self, o): return Region({"minus": [self.spec, coerce_region(o).spec]})
    def __invert__(self): return Region({"not": self.spec})
    def grow(self, n): return Region({"grow": [self.spec, n]})
    def shrink(self, n): return Region({"shrink": [self.spec, n]})
    def to_json(self): return self.spec
    def __repr__(self): return f"Region({json.dumps(self.spec)[:80]})"


def everywhere(): return Region("all")
def disc(x, y, r): return Region({"disc": [x, y, r]})
def band(y0, y1): return Region({"band": [y0, y1]})
def rect(x0, y0, x1, y1): return Region({"rect": [x0, y0, x1, y1]})
def polygon(points): return Region({"polygon": [list(map(float, p)) for p in points]})
def below(curve): return Region({"below": _curve_spec(curve)})
def above(curve): return Region({"above": _curve_spec(curve)})
def mask_region(cells): return Region({"cells": np.asarray(cells, bool)})


def _curve_spec(curve):
    """A curve across the panel: points [(x, y), ...] joined by straight lines (flat beyond the ends)."""
    return [list(map(float, p)) for p in curve]


def coerce_region(r):
    if isinstance(r, Region): return r
    if r is None: return everywhere()
    if isinstance(r, np.ndarray): return mask_region(r)
    return Region(r)                                   # a JSON description


def _render(spec, H, W):
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    if isinstance(spec, str):
        if spec in ("all", "everywhere"): return np.ones((H, W), bool)
        raise StudioError(f"unknown region {spec!r}")
    if not isinstance(spec, dict) or len(spec) != 1:
        raise StudioError(f"a region is one of {sorted(REGION_KINDS)}, written like {{\"band\": [0, 172]}}; got {spec!r}")
    (kind, a), = spec.items()
    try:
        if kind == "disc": x, y, r = a; return np.hypot(xx - x, yy - y) < r
        if kind == "band": y0, y1 = a; return (yy >= y0) & (yy < y1)
        if kind == "rect": x0, y0, x1, y1 = a; return (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        if kind in ("below", "above"):
            c = np.asarray(a, float); c = c[np.argsort(c[:, 0])]
            yc = np.interp(xx[0], c[:, 0], c[:, 1])[None, :]
            return yy > yc if kind == "below" else yy < yc
        if kind == "polygon":
            from matplotlib.path import Path
            pts = np.stack([xx.ravel(), yy.ravel()], 1)
            return Path(np.asarray(a, float)).contains_points(pts).reshape(H, W)
        if kind == "cells": m = np.asarray(a, bool); assert m.shape == (H, W); return m.copy()
        if kind == "or": return np.logical_or.reduce([_render(s, H, W) for s in a])
        if kind == "and": return np.logical_and.reduce([_render(s, H, W) for s in a])
        if kind == "minus": return _render(a[0], H, W) & ~np.logical_or.reduce([_render(s, H, W) for s in a[1:]])
        if kind == "not": return ~_render(a, H, W)
        if kind == "grow": m = _render(a[0], H, W); return ndimage.distance_transform_edt(~m) <= float(a[1])
        if kind == "shrink": m = _render(a[0], H, W); return _depth(m) > float(a[1])
    except StudioError: raise
    except Exception as e:
        raise StudioError(f"region {kind}: could not read {a!r} ({e}); see `help` for the form") from None
    raise StudioError(f"unknown region kind {kind!r}: use one of {sorted(REGION_KINDS)}")


REGION_KINDS = {"all": '"all"', "disc": '{"disc": [x, y, r]}', "band": '{"band": [y0, y1]}', "rect": '{"rect": [x0, y0, x1, y1]}',
                "polygon": '{"polygon": [[x, y], ...]}', "below": '{"below": [[x, y], ...]}  (under a curve)',
                "above": '{"above": [[x, y], ...]}', "or": '{"or": [r1, r2, ...]}', "and": '{"and": [r1, r2, ...]}',
                "minus": '{"minus": [r, cut, ...]}', "not": '{"not": r}', "grow": '{"grow": [r, cells]}', "shrink": '{"shrink": [r, cells]}'}


def _depth(m):
    """Distance (cells) from each cell of m to the nearest cell outside it; the panel's edge is not 'outside'."""
    return ndimage.distance_transform_edt(np.pad(m, 1, constant_values=True))[1:-1, 1:-1] * m


# =====================================================================================================
# Receipts and the journal
# =====================================================================================================
@dataclass
class Receipt:
    """What one act did: `given`, `taken`: {species: volume} given to and taken from the wet layer (summed from
    the tools' own stroke Receipts, goo.classic.brush.Receipt); `strokes`: int, tool passes (or presses, flicks);
    `ticks`: int; `length`: float, path length (cells); `cells`: int, cells the tools touched; `bbox`: their
    (y0, y1, x0, x1) or None; `notes`: list of str, the studio's notes (conditions met, mixes made)."""
    act: str
    tick: int = 0
    given: dict = field(default_factory=dict)
    taken: dict = field(default_factory=dict)
    strokes: int = 0
    ticks: int = 0
    length: float = 0.0
    cells: int = 0
    notes: list = field(default_factory=list)
    bbox: tuple = None

    @property
    def laid(self): return float(sum(self.given.values()))
    @property
    def took(self): return float(sum(v for k, v in self.taken.items() if k not in CARRIERS))   # colour taken up, not water

    def to_json(self):
        r = lambda d: {k: round(v, 3) for k, v in d.items() if v > 1e-9}
        out = dict(act=self.act, tick=int(self.tick), ticks=int(self.ticks), strokes=self.strokes, length=round(float(self.length), 1),
                   cells=self.cells, given=r(self.given), taken=r(self.taken))
        if self.bbox is not None: out["bbox"] = [int(v) for v in self.bbox]
        if self.notes: out["notes"] = self.notes
        return out


def _jsonable(v):
    if isinstance(v, Region): return v.to_json()
    if isinstance(v, Paint): return v.to_json()
    if isinstance(v, np.ndarray):
        return "<cells>" if v.dtype == bool else v.tolist()
    if isinstance(v, (list, tuple)): return [_jsonable(x) for x in v]
    if isinstance(v, dict): return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None: return v
    if isinstance(v, np.generic): return v.item()
    return repr(v)


# =====================================================================================================
# Acts: every technique is registered here, with an example, so the easel's `help` is generated from them
# =====================================================================================================
ACTS = {}
REGION_PARAMS = {"region"}
PAINT_PARAMS = {"paint", "from_", "to", "via"}
ANGLES = {"across": 0.0, "horizontal": 0.0, "down": 90.0, "vertical": 90.0, "diagonal": 45.0, "rising": -45.0, "falling": 45.0}
WETNESS = {"damp": 0.3, "wet": 0.5, "flooded": 0.75}


def act(example, kind="lays"):
    """Register a studio act. kind: 'lays' (NothingLaid unless paint was given), 'takes' (unless paint was taken
    up), or 'other' (no check). The example is an act as the easel takes it."""
    def deco(fn):
        @functools.wraps(fn)
        def run(self, *a, **k):
            return self._perform(fn, a, k, kind)
        run._act = dict(example=example, kind=kind, fn=fn)
        ACTS[fn.__name__] = run
        return run
    return deco


class Studio:
    """A canvas, a rack of brushes, named paints, a clock and a journal. See the module docstring."""

    def __init__(self, canvas, pigments=None, medium="watercolour", seed=0, workdir=None):
        self.cv = canvas
        if canvas.reflectance().shape[-1] != 4:
            raise ValueError("the studio reads four classic bands (R, G, B, 1.5 um): give it a canvas with "
                             "optics='classic' (SPEC §11)")
        self.W, self.H = int(canvas.W), int(canvas.H)
        self.pigments = list(pigments) if pigments is not None else list(canvas.species[len(CARRIERS):])
        self.species = CARRIERS + self.pigments
        self.medium = medium
        self.seed = seed
        self.workdir = workdir or os.path.join("out", "studio")
        self.paints = {}                     # name -> paint
        self.journal = []                    # every act, its parameters, receipt and tick
        self.clock = 0                       # ticks this studio has let pass (strokes, waits)
        self.acts = 0                        # acts performed (seeds each act's own randomness)
        self.reserved = np.zeros((self.H, self.W), bool)   # under frisket (Canvas.mask), as the studio put it there
        self._rack = {}                      # (kind, width) -> brush
        self._key = "genesis"
        self._kept = False
        self._rec = None

    @classmethod
    def new(cls, width, height, pigments, medium="watercolour", ground=None, seed=1, laws=None, workdir=None):
        """A fresh panel. Watercolour and gouache go on paper, acrylic and oil on gesso. An oil panel is given a
        slow cure (its climate) so that it can dry at all; acrylic gets its clear binder as a species."""
        pigments = list(pigments)
        if medium == "acrylic" and _paints.BINDER not in pigments:
            pigments.append(_paints.BINDER)
        ground = ground or ("paper" if medium in ("watercolour", "gouache") else "gesso")
        L = dict(lazy=3)
        if medium == "oil": L.update(cure_rate=1e-4)
        L.update(laws or {})
        cv = Canvas(width, height, pigments, seed=seed, ground=ground, laws=L)
        st = cls(cv, pigments, medium=medium, seed=seed, workdir=workdir)
        st._key = hashlib.sha256(json.dumps([width, height, pigments, medium, ground, seed, L, _goo_version()], sort_keys=True).encode()).hexdigest()
        return st

    # ------------------------------------------------------------------------------------------ looking
    def reflectance(self):
        return self.cv.reflectance()

    def look(self, region=None, raking=None):
        """The painting as a photograph: sRGB (H, W, 3) uint8, under even light, or under a low light through
        goo's relight camera (`raking=True`, or dict(azimuth=20, elevation=18, relief=1.0)). `region` crops to
        the region's bounding box."""
        from goo.classic.palette import linear_to_srgb
        if raking:
            from goo.classic.surface import Surface
            from goo.classic.sensors import relight
            o = dict(azimuth=20, elevation=18, relief=1.0); o.update(raking if isinstance(raking, dict) else {})
            L = relight.light_dir(o["azimuth"], o["elevation"])
            lin = relight.render(Surface.from_canvas(self.cv), L, relief_gain=o["relief"]) / (L[2] + 0.1)
        else:
            lin = self.cv.reflectance()[..., :3]
        im = (linear_to_srgb(lin) * 255 + 0.5).astype(np.uint8)
        if region is not None:
            m = coerce_region(region).mask(self.H, self.W); ys, xs = np.nonzero(m)
            if ys.size: im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return im

    def look_wet(self):
        """How wet the surface looks, (H, W) in 0..1: 1 glistening, in between damp (satin), 0 dry (Canvas.sheen)."""
        return np.asarray(self.cv.sheen())

    def wetness_picture(self):
        """The photograph with wetness shown: glistening cells blue, damp cells amber, not touch-dry cells hatched."""
        im = self.look().astype(float)
        s = self.look_wet(); td = self.cv.touch_dry()
        gl = np.clip((s - GLISTEN) / (1 - GLISTEN), 0, 1)[..., None]; dp = ((s > DAMP) & (s <= GLISTEN))[..., None]
        im = im * (1 - 0.55 * gl) + np.array([40, 120, 255]) * 0.55 * gl
        im = np.where(dp, im * 0.6 + np.array([255, 170, 40]) * 0.4, im)
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        hatch = (~td) & ((xx + yy) % 6 == 0)
        im[hatch] = im[hatch] * 0.5
        return np.clip(im, 0, 255).astype(np.uint8)

    def save(self, path, raking=None):
        """Write the photograph to `path` (PNG) and the journal beside it (same name, .journal.json)."""
        from PIL import Image
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        Image.fromarray(self.look(raking=raking)).save(path)
        with open(os.path.splitext(path)[0] + ".journal.json", "w") as f: json.dump(self.journal, f, indent=1)
        return path

    def _region(self, region):
        r = coerce_region(region); return r, r.mask(self.H, self.W)

    def _under(self, m, n=4096):
        """The reflectance (4 bands, linear) of the cells of m, as `goo.classic.paints.mix_for` takes `over`: a patch of at
        most n cells (an even subsample of a larger region)."""
        R = self.cv.reflectance()[m]
        if not len(R): R = self.cv.reflectance().reshape(-1, 4)
        return R[::max(1, len(R) // n)]

    def _seen(self, m):
        """Mean reflectance (4 bands, linear) of the cells of m: what is there, as the eye sees it."""
        R = self.cv.reflectance()
        return R[m].mean(0) if m.any() else R.reshape(-1, 4).mean(0)

    # ------------------------------------------------------------------------------------------ paints
    def paint(self, spec, over=None, thickness=0.5):
        """A paint from a description: a paint object; a name given with `mix`; a recipe in the studio's medium
        ("ultramarine", "3 ultramarine + rose"); {"watercolour": "2 rose + ochre", "strength": 0.04} (likewise
        gouache, acrylic with body, oil with fat); "water"; a paint as the journal writes it ({"medium": ..., "parts": {...}, ...}: Paint.to_json); or colour by intent: {"colour": "#8899cc"} or
        [r, g, b] in 0..255, optionally with "medium", mixed with mix_for over what is there (`over`, a mask)."""
        if hasattr(spec, "load"): return spec
        if isinstance(spec, str):
            if spec in self.paints: return self.paints[spec]
            if spec == "water": return _make_paint("water", {})
            if spec.startswith("#"): return self.paint({"colour": spec}, over, thickness)
            return self._check(_make_paint(self.medium, self._recipe(spec)))
        if isinstance(spec, (list, tuple)) and len(spec) == 3 and all(isinstance(v, (int, float)) for v in spec):
            return self.paint({"colour": list(spec)}, over, thickness)
        if isinstance(spec, dict):
            s = dict(spec)
            if "parts" in s and "medium" in s:                   # a Paint as written by Paint.to_json (the journal)
                try:
                    return self._check(Paint.from_json(s))
                except (KeyError, TypeError, ValueError) as e:
                    raise StudioError(f"cannot read the paint {spec!r}: {e}") from None
            if "colour" in s or "color" in s:
                col = s.pop("colour", None) or s.pop("color")
                if isinstance(col, str): col = [int(col.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
                medium = s.pop("medium", self.medium)
                under = self._under(over if over is not None else np.ones((self.H, self.W), bool))
                pigs = [q for q in self.pigments if q != _paints.BINDER and not (medium == "watercolour" and q == "white")]
                if medium == "gouache" and "white" not in self.pigments:
                    raise StudioError("gouache is made opaque with white: this panel has no white")
                p, pred, dE = _paints.mix_for(col, medium, thickness=s.pop("thickness", thickness), over=under, pigments=pigs, **s)
                pred = [int(round(255 * v)) for v in np.clip(pred, 0, 1)]
                if self._rec is not None: self._rec.notes.append(f"mixed {col} in {medium}: {p!r} predicts {pred} dry, delta E {dE:.1f}")
                return self._check(p)
            for medium in ("watercolour", "gouache", "acrylic", "oil"):
                if medium in s:
                    mix = s.pop(medium); mix = self._recipe(mix) if isinstance(mix, str) else dict(mix)
                    return self._check(_make_paint(medium, mix, **s))
        raise StudioError(f"cannot read the paint {spec!r}. Write a recipe like \"2 ultramarine + rose\", "
                          f"{{\"watercolour\": \"ochre\", \"strength\": 0.04}}, {{\"colour\": \"#c08060\"}}, or a name from `mix`. "
                          f"Pigments here: {', '.join(self.pigments)}")

    def _recipe(self, text):
        mix = {}
        for term in text.split("+"):
            m = re.fullmatch(r"\s*([0-9.]+)?\s*\*?\s*([a-z_]+)\s*", term)
            if not m: raise StudioError(f"cannot read {term!r} in the recipe {text!r}: write e.g. \"3 ultramarine + rose\"")
            mix[m.group(2)] = mix.get(m.group(2), 0.0) + float(m.group(1) or 1)
        return mix

    def _check(self, paint):
        for k in paint.load(1.0):
            if k not in self.species:
                if k == _paints.BINDER:
                    raise StudioError(f"this panel has no {k!r} (acrylic's clear binder): start the panel with medium "
                                      f"acrylic, or add \"acrylic\" to its pigments; its paints are {', '.join(self.pigments)}")
                raise StudioError(f"this panel has no {k!r}: its paints are {', '.join(self.pigments)}")
        return paint

    def _name(self, paint):
        for k, v in self.paints.items():
            if v is paint: return k
        return repr(paint)

    # ------------------------------------------------------------------------------------------ the rack
    def _brush(self, kind, width):
        key = (kind, float(width))
        b = self._rack.get(key)
        if b is None:
            recipe = dict(mop=dict(exchange=0.5, plough=0.15, bristles=max(1, int(width / 3.3))),
                          round=dict(exchange=0.65, plough=0.15, bristles=max(1, int(width / 2.5))),
                          bristle=dict(exchange=0.7, plough=0.6, bristles=max(1, int(width / 2.5))),
                          flat=dict(exchange=0.7, plough=0.6, bristles=max(1, int(width / 2.0)), flat=True)).get(kind, {})
            b = _new_brush(self.cv, width, kind, **recipe)
            self._rack[key] = b
        return b

    @staticmethod
    def capacity(width):
        """What a brush `width` cells across holds when fully charged (volume units): ~60 w^1.4 (a 46-cell mop ~12000)."""
        return 60.0 * width ** 1.4

    def _charge(self, b, paint, frac=1.0):
        if getattr(b, "_studio_paint", None) is not None and b._studio_paint is not paint and b.load.sum() > 0:
            b.rinse()                                         # a clean brush for every colour
        b.charge(paint.load(self.capacity(b.width) * frac))
        b._studio_paint = paint
        self._used.add(id(b)); self._used_brushes.append(b)

    def _stroke(self, b, path, speed, pressure, target, steps_after=0, heading=None):
        if len(path) < 2: return
        pts = [(p[0], p[1], p[2] if len(p) > 2 else pressure) for p in path]
        kw = {} if heading is None else dict(heading=heading)
        self._book(b.stroke(pts, speed=speed, pressure=pressure, target=target, steps_after=steps_after, **kw))

    def _book(self, r, strokes=1):
        """Add a tool's Receipt (goo.classic.brush.Receipt) to the act's."""
        rec = self._rec
        for k, v in r.by_species("given").items(): rec.given[k] = rec.given.get(k, 0.0) + v
        for k, v in r.by_species("taken").items(): rec.taken[k] = rec.taken.get(k, 0.0) + v
        rec.ticks += r.ticks; self.clock += r.ticks
        rec.strokes += strokes; rec.length += r.length
        if r.cells: self._touched |= r.mask(self.H, self.W)

    def _lay(self, b, paint, paths, target, speed=3.0, pressure=1.0, frac=1.0, below=0.35):
        """Lay strokes along `paths`, recharging the brush with `paint` whenever it runs low."""
        cap = self.capacity(b.width)
        for k, path in enumerate(paths):
            p = paint(k) if callable(paint) and not hasattr(paint, "load") else paint
            if k == 0 or b.load.sum() < below * cap * frac or getattr(b, "_studio_paint", None) is not p:
                self._charge(b, p, frac)
            self._stroke(b, path, speed, pressure, target)

    # ------------------------------------------------------------------------------------------ geometry
    def _angle(self, strokes):
        if isinstance(strokes, str):
            if strokes not in ANGLES: raise StudioError(f"strokes={strokes!r}: use one of {sorted(ANGLES)} or an angle in degrees (0 across, 90 down)")
            return np.radians(ANGLES[strokes])
        return np.radians(float(strokes))

    def _allowed(self, m, r):
        """Where a brush of radius r may put its centre so its rim stays inside m (the panel's edge is not an edge).
        A reserve is tape: brushes pass over it and lay nothing there."""
        return _depth(m) > r

    def _widths(self, m, prefer, smallest=1.5):
        """The rack's brushes for region m, widest first: none wider than `prefer`, nor much wider than the shape."""
        if not m.any(): raise NothingLaid("the region is empty: it holds no cells of the panel")
        prefer = min(prefer, max(2.5, 0.6 * _depth(m).max()))           # a brush that fits the shape a few times over
        ws = [w for w in WIDTHS if smallest <= w <= prefer] or [min(WIDTHS, key=lambda w: abs(w - prefer))]
        return sorted(ws, reverse=True)

    def _too_small(self, m, widths):
        d = _depth(m).max()
        why = "it is all reserved" if self.reserved.any() and not (m & ~self.reserved).any() else \
              f"its widest part is {2 * d:.1f} cells across and the smallest brush allowed here is {min(widths):g}"
        return NothingLaid(f"the region is too small for the brush: {why}. Grow the region or name a smaller brush.")

    def _cells_of(self, paths):
        c = np.zeros((self.H, self.W), bool)
        for path in paths:
            for a, z in zip(path, path[1:]):
                n = max(2, int(np.hypot(z[0] - a[0], z[1] - a[1]) * 2) + 2)
                xs = np.linspace(a[0], z[0], n); ys = np.linspace(a[1], z[1], n)
                k = (xs >= 0) & (xs < self.W) & (ys >= 0) & (ys < self.H)
                c[ys[k].astype(int), xs[k].astype(int)] = True
        return c

    def _hatch(self, ok, angle, spacing, r, step=4.0, wobble=0.0, keep=1.0, rng=None, min_run=0.0):
        """Parallel strokes at `angle` across the cells where the brush's centre may go (`ok`), alternate ways,
        each run off the panel where `ok` meets the panel's edge. A run of a single point still makes a mark."""
        ys, xs = np.nonzero(ok)
        if xs.size == 0: return []
        c, s = np.cos(angle), np.sin(angle)
        px, py = xs + 0.5, ys + 0.5
        off = -s * px + c * py; along = c * px + s * py
        lo, hi = off.min(), off.max()
        n = int(np.floor((hi - lo) / spacing)) + 1
        offs = (lo + hi) / 2 - (n - 1) * spacing / 2 + spacing * np.arange(n)
        ext = RUNOFF * r + 1                             # how far past the panel's edge a stroke runs
        ts = np.arange(along.min() - ext - 1, along.max() + ext + 1.5, 0.5)
        paths = []
        for k, o in enumerate(offs):
            if keep < 1 and rng.random() > keep: continue
            x = ts * c - o * s; y = ts * s + o * c
            near = (x > -ext) & (x < self.W + ext) & (y > -ext) & (y < self.H + ext)
            cx = np.clip(x, 0, self.W - 1e-6).astype(int); cy = np.clip(y, 0, self.H - 1e-6).astype(int)
            inside = near & ok[cy, cx]
            edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
            for a, z in zip(edges[::2], edges[1::2]):
                idx = np.arange(a, z)
                if (idx.size - 1) * 0.5 < min_run: continue
                if idx.size == 1:
                    pts = [(x[a] - 0.5 * c, y[a] - 0.5 * s), (x[a] + 0.5 * c, y[a] + 0.5 * s)]
                else:
                    sel = idx[::max(1, int(step / 0.5))]
                    if sel[-1] != idx[-1]: sel = np.r_[sel, idx[-1]]
                    pts = [(x[i], y[i]) for i in sel]
                    if wobble and rng is not None and len(pts) > 2:
                        j = rng.normal(0, wobble, len(pts)); j[0] = j[-1] = 0
                        pts = [(px_ - s * d, py_ + c * d) for (px_, py_), d in zip(pts, j)]
                paths.append(pts)
        return [p if k % 2 == 0 else p[::-1] for k, p in enumerate(paths)]

    def _fill(self, m, paint, target, strokes=0.0, prefer=46.0, kind="mop", brush=None, spacing=0.48, speed=4.0,
              pressure=1.0, wobble=1.5, keep=1.0, frac=1.0, below=0.35, order=None):
        """Cover region m in parallel strokes: the widest brush that fits first, then narrower ones cutting in where
        its strokes did not reach (edges, corners, around a reserve). `paint` may be a function of the stroke's
        position along `order` (0..1), for graded washes."""
        angle = self._angle(strokes)
        widths = [float(brush)] if brush is not None else self._widths(m, prefer)
        covered = np.zeros_like(m); laid_any = False
        for w in widths:
            r = w / 2 * (0.6 + 0.4 * pressure)
            ok = self._allowed(m, r)
            if laid_any: ok &= ndimage.distance_transform_edt(~(m & ~covered & ~self.reserved)) <= r   # cut in only where missed
            if not ok.any(): continue
            paths = self._hatch(ok, angle, max(1.0, spacing * w), r, step=max(2.0, 1.9 * speed), wobble=wobble * min(1.0, w / 30), keep=keep,
                                rng=self._rng, min_run=1.0 if laid_any else 0.0)   # cutting in: no dabs where a run would be a dot
            if not paths: continue
            b = self._brush(kind, w)
            if order is not None:
                pos = [order(p) for p in paths]; idx = np.argsort(pos, kind="stable")
                paths = [paths[i] for i in idx]; pos = [pos[i] for i in idx]
                self._lay(b, lambda k: paint(pos[k]), paths, target, speed, pressure, frac, below)
            else:
                self._lay(b, paint, paths, target, speed, pressure, frac, below)
            laid_any = True
            covered |= ndimage.distance_transform_edt(~self._cells_of(paths)) <= r
            if keep < 1 or (m & ~covered & ~self.reserved).sum() <= max(0.02 * m.sum(), 3): break
        if not laid_any:
            if brush is not None:
                raise NothingLaid(f"the region is too small for a brush {brush} cells across: leave the brush to the studio, or name a smaller one")
            raise self._too_small(m, widths)

    def _points(self, m, spacing, rng, jitter=0.35):
        """Points scattered through region m about `spacing` apart (a jittered grid), in a random order."""
        ys, xs = np.mgrid[spacing / 2:self.H:spacing, spacing / 2:self.W:spacing]
        xs = xs + rng.uniform(-jitter, jitter, xs.shape) * spacing; ys = ys + rng.uniform(-jitter, jitter, ys.shape) * spacing
        xi = np.clip(xs, 0, self.W - 1).astype(int); yi = np.clip(ys, 0, self.H - 1).astype(int)
        keep = m[yi, xi]
        pts = list(zip(xs[keep], ys[keep]))
        if not pts:
            yy, xx = np.nonzero(m)
            if yy.size: pts = [(xx.mean() + 0.5, yy.mean() + 0.5)]
        order = rng.permutation(len(pts))
        return [pts[i] for i in order]

    def _wet_at(self, pts):
        s = self.look_wet()
        return np.array([s[int(np.clip(y, 0, self.H - 1)), int(np.clip(x, 0, self.W - 1))] for x, y in pts])

    def _settle(self):
        """Let the slow laws catch up (a wait of no time), so what the studio looks at is current."""
        self.cv.wait(0)

    # ------------------------------------------------------------------------------------------ performing
    def _perform(self, fn, a, k, kind):
        sig = inspect.signature(fn)
        try:
            bound = sig.bind(self, *a, **k)
        except TypeError as e:
            raise StudioError(f"{fn.__name__}{_sig(fn)}: {e}") from None
        bound.apply_defaults()
        params = {n.rstrip("_"): _jsonable(v) for n, v in list(bound.arguments.items())[1:]}   # as the easel takes them
        rec = Receipt(fn.__name__, tick=self.clock)
        self._rec, self._used, self._used_brushes = rec, set(), []
        self._touched = np.zeros((self.H, self.W), bool)
        self._rng = np.random.default_rng([int(self.seed), self.acts, 7])
        try:
            out = fn(*bound.args, **bound.kwargs)
        finally:
            for b in {id(b): b for b in self._used_brushes}.values():
                b.rinse(); b._studio_paint = None              # every brush back in the jar after the act
            self._rec = None
        if self._touched.any():
            rec.cells = int(self._touched.sum()); ys, xs = np.nonzero(self._touched)
            rec.bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        self.acts += 1
        entry = dict(n=len(self.journal), act=fn.__name__, params=params, receipt=rec.to_json())
        if getattr(self, "_fixed", None) is not None: entry["receipt"]["fixed"] = round(float(self._fixed), 4); self._fixed = None
        if kind == "lays" and rec.laid <= 1e-9:
            entry["failed"] = "nothing laid"; self.journal.append(entry)
            raise NothingLaid(f"{fn.__name__} laid nothing: the brush never reached the canvas (lifted everywhere, "
                              f"or everything under it was reserved). Check the region and the reserve.")
        if kind == "takes" and rec.took <= 1e-9:
            entry["failed"] = "nothing taken"; self.journal.append(entry)
            raise NothingLaid(f"{fn.__name__} took nothing up: the paint there is dry and fixed, or there is none. "
                              f"Lifting works on wet or re-wettable (unfixed watercolour, gouache) paint.")
        self.journal.append(entry)
        return out if out is not None else rec

    def do(self, a):
        """Perform one act written as the easel takes it: {"do": "wash", "region": {...}, "paint": ...}.
        A list performs each in turn. Returns the receipts."""
        if isinstance(a, list): return [self.do(x) for x in a]
        if not isinstance(a, dict) or "do" not in a:
            raise StudioError(f'an act is a JSON object with "do": one of {", ".join(ACTS)}; got {a!r}')
        a = dict(a); name = a.pop("do")
        if name not in ACTS: raise StudioError(f"unknown act {name!r}: the acts are {', '.join(ACTS)} (see `help`)")
        fn = ACTS[name]._act["fn"]; names = list(inspect.signature(fn).parameters)[1:]
        kw = {}
        for k, v in a.items():
            p = "from_" if k == "from" else k
            if p not in names: raise StudioError(f"{name} has no parameter {k!r}; it takes {', '.join(n.rstrip('_') for n in names)}")
            kw[p] = v
        return getattr(self, name)(**kw)

    # ------------------------------------------------------------------------------------------ rehearsal
    def _state(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_rec") and k not in ("_used", "_used_brushes", "_touched", "_rng")}

    def _restore(self, state):
        self.__dict__.update(state)

    @contextmanager
    def rehearse(self):
        """A studio on a fork of the canvas: try things, look, then `keep()` (the fork becomes the painting) or
        leave the block (nothing happened). The rehearsal holds the twins of the painting's brushes (Brush.twin), so
        an act rehearsed and then kept is exactly the act done on the painting."""
        r = Studio.__new__(Studio)
        st = self._state(); r.__dict__.update(st)
        r.cv = _fork(self.cv)
        r._rack = {k: b.twin(r.cv) for k, b in self._rack.items()}   # the same brushes, on the fork
        r.journal = list(self.journal); r.paints = dict(self.paints); r.reserved = self.reserved.copy()
        r._kept = False; r._rec = None
        yield r
        if r._kept:
            r._kept = False
            r.journal.append(dict(n=len(r.journal), act="keep", params={}, receipt=dict(tick=r.clock)))
            self._restore(r._state())

    def keep(self):
        """Inside `rehearse()`: make this rehearsal the painting."""
        self._kept = True

    def choose(self, variants, by, names=None):
        """Rehearse each variant (a function of a studio, or an act / list of acts) and keep the one `by` prefers:
        `by(look)` scores a photograph (sRGB, (H, W, 3) floats in 0..1); the highest score is kept. Returns its index."""
        scores, kept = [], None
        for v in variants:
            with self.rehearse() as r:
                (v(r) if callable(v) else r.do(v))
                scores.append(float(by(r.look() / 255.0)))
                if scores[-1] == max(scores): kept = r._state()
        i = int(np.argmax(scores))
        self._restore(kept)
        self.journal.append(dict(n=len(self.journal), act="choose", params=dict(variants=names or len(variants)),
                                 receipt=dict(tick=self.clock, kept=i, scores=[round(s, 4) for s in scores])))
        return i

    # ------------------------------------------------------------------------------------------ sittings
    def sitting(self, name, fn, cache=None):
        """Run `fn(studio)` and checkpoint the result under a key made of `name`, the source of `fn` and the
        previous sitting's key, so re-running a script repaints only from the first sitting that changed. A cached
        sitting restores the whole studio (canvas, brushes, clock, journal) exactly. The key also holds the values
        of the globals `fn` names directly (regions, numbers, lists; the source of functions it calls) and a hash
        of goo's own source; what those reach indirectly is not seen."""
        try:
            src = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError):
            raise StudioError(f"sitting {name!r}: the source of {fn!r} cannot be read, so it cannot be cached; pass a def")
        key = hashlib.sha256(f"{self._key}\n{name}\n{src}\n{_globals_digest(fn)}".encode()).hexdigest()
        d = cache or os.path.join(self.workdir, "sittings"); os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')}-{key[:16]}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f: self._restore(pickle.load(f))
            self.journal.append(dict(n=len(self.journal), act="sitting", params=dict(name=name), receipt=dict(tick=self.clock, cached=True)))
            return "cached"
        start = len(self.journal)
        fn(self)
        self._key = key
        self.journal.append(dict(n=len(self.journal), act="sitting", params=dict(name=name), receipt=dict(tick=self.clock, acts=len(self.journal) - start)))
        tmp = path + ".tmp"
        with open(tmp, "wb") as f: pickle.dump(self._state(), f, protocol=5)
        os.replace(tmp, path)
        return "painted"

    # =================================================================================================
    # The techniques
    # =================================================================================================
    @act({"do": "mix", "name": "sky blue", "paint": {"watercolour": "4 ultramarine + rose", "strength": 0.05}}, kind="other")
    def mix(self, name, paint):
        """Mix a paint and give it a name, for later acts. `paint` is a recipe ("3 ultramarine + rose", in the
        studio's medium), {"watercolour": recipe, "strength": pigment fraction}, {"oil": recipe, "fat": 0.5},
        {"acrylic": recipe, "body": 0.7}, {"gouache": recipe}, or a colour by intent: {"colour": "#8899cc"}."""
        p = self.paint(paint)
        self.paints[name] = p
        pv = _paints.preview(p, 0.5)
        to8 = lambda c: [int(round(255 * v)) for v in np.clip(c, 0, 1)]
        self._rec.notes.append(f"preview at 0.5 wet: {to8(pv['wet'])} wet, {to8(pv['dry'])} dry")
        self._rec.notes.append(f"{name} = {p!r}")
        return self._rec

    @act({"do": "wet", "region": {"band": [0, 172]}, "wetness": "wet"})
    def wet(self, region, wetness="wet", strokes="across", brush=None):
        """Wet the paper with clean water, in overlapping strokes with a mop: so that colour laid into it spreads
        and softens (wet-in-wet). Works when the region glistens evenly; it dries back through damp in a few
        hundred ticks on paper. wetness: damp, wet, flooded, or a thickness in volume units."""
        _, m = self._region(region)
        self._fill(m, self.paint("water"), self._wetness(wetness), strokes, prefer=46, brush=brush, wobble=1.5, speed=6.0)

    def _wetness(self, w):
        if isinstance(w, str):
            if w not in WETNESS: raise StudioError(f"wetness={w!r}: use damp, wet, flooded, or a thickness (e.g. 0.4)")
            return WETNESS[w]
        return float(w)

    @act({"do": "wash", "region": {"band": [172, 300]}, "paint": "3 ultramarine + rose", "wetness": "wet"})
    def wash(self, region, paint, wetness="wet", strokes="across", brush=None, wobble=1.5):
        """A flat wash: one colour in overlapping strokes, alternate ways, the brush recharged before it runs dry,
        so the region takes an even film. Works when it dries without stripes; streaks mean too little water."""
        _, m = self._region(region)
        p = self.paint(paint, over=m, thickness=self._wetness(wetness))
        self._fill(m, p, self._wetness(wetness), strokes, prefer=46, brush=brush, wobble=wobble, speed=5.0)

    @act({"do": "graded_wash", "region": {"band": [0, 172]}, "from": "ultramarine", "to": "2 ochre + yellow", "via": ["2 rose + ultramarine"]})
    def graded_wash(self, region, from_, to, via=(), toward="down", wetness="wet", strokes="across", brush=None, wobble=2.0):
        """A graded (variegated) wash: strokes laid in order from one side of the region to the other, each from
        the paint mixed on the palette for its place, from `from` through `via` to `to`. Best into wet paper
        (`wet` first), where the bands melt into one another; on dry paper the steps show. toward: down, up,
        right or left (the direction in which the colour changes)."""
        _, m = self._region(region)
        th = self._wetness(wetness)
        stops = [self.paint(p, over=m, thickness=th) for p in [from_, *via, to]]
        ys, xs = np.nonzero(m)
        ax = {"down": (0, 1), "up": (0, -1), "right": (1, 0), "left": (-1, 0)}.get(toward)
        if ax is None: raise StudioError(f"toward={toward!r}: use down, up, right or left")
        proj = (xs + 0.5) * ax[0] + (ys + 0.5) * ax[1]; lo, hi = proj.min(), proj.max()
        order = lambda path: float(np.clip((np.mean([p[0] * ax[0] + p[1] * ax[1] for p in path]) - lo) / max(hi - lo, 1e-9), 0, 1))
        cache = {}
        def at(t):
            k = round(t * 24) / 24                       # 25 mixes on the palette
            if k not in cache:
                u = k * (len(stops) - 1); i = min(int(u), len(stops) - 2)
                cache[k] = stops[i].blend(stops[i + 1], u - i)
            return cache[k]
        self._fill(m, at, th, strokes, prefer=46, brush=brush, wobble=wobble, order=order, below=1.01, speed=5.0)

    @act({"do": "drop_in", "region": {"disc": [100, 80, 20]}, "paint": "2 rose + ochre"})
    def drop_in(self, region, paint, spacing=None, size=None, strength=1.0):
        """Charge colour into a wet wash: the loaded tip touched down here and there and lifted, so the colour
        blooms outward with soft edges (where the paper is wet beyond the region, it blooms beyond it too). Works while the paper is wet (it glistens); on dry paper the touches stay
        as hard-edged marks (the receipt says how many fell on dry paper)."""
        _, m = self._region(region)
        w = float(size or np.clip(_depth(m).max() / 2, 3, 7))
        pts = self._points(m, float(spacing or 1.6 * w), self._rng)
        wet = self._wet_at(pts)
        if (wet <= DAMP).any(): self._rec.notes.append(f"{int((wet <= DAMP).sum())} of {len(pts)} drops fell on dry paper (hard marks, not blooms)")
        p = self.paint(paint, over=m)
        b = self._brush("round", min(WIDTHS, key=lambda v: abs(v - w)))
        self._lay(b, p, [[(x, y), (x + 0.6, y + 0.3)] for x, y in pts], target=0.8 * strength, speed=1.0, pressure=0.7)

    @act({"do": "backrun", "region": {"rect": [60, 50, 120, 70]}})
    def backrun(self, region, size=None, wetness=0.95):
        """Clean water touched into a damp (no longer glistening) wash: it pushes the pigment out ahead of it and
        leaves a pale bloom with a dark, frilled edge (a cauliflower, a cloud). Works only while the wash is damp:
        into a glistening wash the water merges; on dry paper nothing happens (and nothing is laid)."""
        _, m = self._region(region)
        w = float(size or np.clip(_depth(m).max() * 1.6, 3, 15))
        pts = self._points(m, 1.6 * w, self._rng)
        s = self._wet_at(pts)
        if (s <= DAMP).mean() > 0.75: raise NothingLaid("the wash is dry here: a backrun needs a damp wash. Wet it again, or come sooner (look --wet).")
        if (s <= DAMP).any(): self._rec.notes.append(f"{int((s <= DAMP).sum())} of {len(pts)} touches fell where the wash was already dry")
        if (s > GLISTEN).mean() > 0.5: self._rec.notes.append("the wash still glistens: the water will merge rather than bloom (wait: dry until damp)")
        b = self._brush("round", min(WIDTHS, key=lambda v: abs(v - w)))
        paths = [[(x - w * 0.6, y), (x, y + self._rng.normal(0, 1)), (x + w * 0.6, y)] for x, y in pts]
        self._lay(b, self.paint("water"), paths, target=wetness, speed=2.0)

    @act({"do": "glaze", "region": {"band": [172, 300]}, "paint": "ultramarine", "wetness": "damp"})
    def glaze(self, region, paint, wetness="damp", strokes="across", brush=None):
        """A glaze: a thin, transparent film over dry paint, laid in one light pass without scrubbing, so the colour
        beneath shows through and is modified. Refuses wet paint beneath (it would stir in): dry it first. Over
        unfixed watercolour it may lift a little (dry until fixed to settle the layer)."""
        _, m = self._region(region)
        td = self.cv.touch_dry()
        if not td[m].all():
            raise NothingLaid("the paint under the glaze is still wet: a glaze over wet paint stirs into it. dry(until='touch') first.")
        p = self.paint(paint, over=m, thickness=self._wetness(wetness))
        self._fill(m, p, self._wetness(wetness), strokes, prefer=28, kind="round", brush=brush, pressure=0.7, wobble=0.8)

    @act({"do": "dry_brush", "region": {"band": [250, 300]}, "paint": "2 sienna + ultramarine"})
    def dry_brush(self, region, paint, strokes="across", brush=None, load=0.12, coverage=0.7):
        """Dry brush: a brush with little paint dragged lightly and fast, so it catches on the paper's tooth and
        leaves broken, sparkling marks (grass, rock, glitter on water). load: fraction of a full charge;
        coverage: the fraction of strokes laid."""
        _, m = self._region(region)
        p = self.paint(paint, over=m)
        self._fill(m, p, 0.3, strokes, prefer=14, kind="bristle", brush=brush, spacing=0.9, speed=7.0, pressure=0.55,
                   wobble=1.0, keep=coverage, frac=load, below=0.0)

    @act({"do": "scumble", "region": {"disc": [200, 100, 30]}, "paint": {"gouache": "white + ochre"}})
    def scumble(self, region, paint, brush=None, coverage=0.6):
        """A scumble: an opaque, lighter paint scrubbed thin in short strokes every which way, so the colour beneath
        glimmers through a broken veil. Wants an opaque paint (gouache, acrylic, oil): watercolour gives a tint."""
        _, m = self._region(region)
        p = self.paint(paint, over=m)
        if getattr(p, "medium", None) == "watercolour": self._rec.notes.append("watercolour is transparent: this scumble is a tint, not a veil")
        w = float(brush or min(10.0, max(2.5, _depth(m).max())))
        r = w / 2 * 0.8
        ok = self._allowed(m, r)
        if not ok.any(): raise NothingLaid(f"the region is too small for a brush {w:g} cells across")
        pts = self._points(ok, w * 1.2 / max(coverage, 0.1) ** 0.5, self._rng, jitter=0.45)
        paths = []
        for x, y in pts:
            a = self._rng.uniform(0, np.pi); L = w * self._rng.uniform(0.8, 1.6)
            c, s = np.cos(a) * L / 2, np.sin(a) * L / 2
            paths.append([(x - c, y - s), (x + c * 0.2, y + s * 0.2 + self._rng.normal(0, 0.5)), (x + c, y + s)])
        b = self._brush("bristle", min(WIDTHS, key=lambda v: abs(v - w)))
        self._lay(b, p, paths, target=0.35, speed=2.5, pressure=0.6, frac=0.4)

    @act({"do": "line", "path": [[150, 40], [160, 45], [170, 40]], "paint": "2 sienna + ultramarine", "width": 2.5})
    def line(self, path, paint, width=2.5, taper=True, wetness=0.5, speed=1.6):
        """One stroke along a path of points [x, y] (or [x, y, pressure]): a reed, a mast, a bird, a contour. A
        round brush `width` cells across; with taper the pressure falls toward the end (a lifted stroke)."""
        pts = [tuple(map(float, p)) for p in path]
        if len(pts) < 2: raise StudioError("a line needs at least two points")
        if taper and all(len(p) == 2 for p in pts):
            n = len(pts); pts = [(x, y, 1.0 - 0.55 * (i / max(n - 1, 1)) ** 1.5) for i, (x, y) in enumerate(pts)]
        p = self.paint(paint)
        b = self._brush("round", min(WIDTHS, key=lambda v: abs(v - width)))
        self._lay(b, p, [pts], target=self._wetness(wetness), speed=speed)

    @act({"do": "dab", "at": [[120, 200], [128, 204]], "paint": "rose", "size": 4})
    def dab(self, at, paint, size=4.0, wetness=0.7):
        """Dabs: the brush touched down and lifted at each point [x, y], leaving a soft round mark its own size
        (a flower, a leaf, a pebble, a glint)."""
        p = self.paint(paint)
        b = self._brush("round", min(WIDTHS, key=lambda v: abs(v - size)))
        pts = [tuple(map(float, q))[:2] for q in at]
        self._lay(b, p, [[(x, y), (x + 0.6, y + 0.3)] for x, y in pts], target=self._wetness(wetness), speed=1.0, pressure=0.9)

    def _paste(self, p, what):
        if getattr(p, "medium", None) in ("watercolour", "water"):
            raise NothingLaid(f"{what} needs a paste (oil, acrylic or gouache): watercolour runs flat. Mix it in another medium.")

    @act({"do": "impasto", "region": {"rect": [40, 40, 70, 60]}, "paint": {"oil": "white + yellow"}, "thickness": "thick"})
    def impasto(self, region, paint, thickness="thick", strokes="across", brush=None):
        """Impasto: stiff paint laid thick with a loaded bristle brush, so the strokes stand up and keep their
        ridges and grooves (see them in raking light). thickness: thick (2 units, 0.5 mm), heavy (4), or a
        number. Needs a paste: oil, acrylic or gouache."""
        _, m = self._region(region)
        p = self.paint(paint, over=m); self._paste(p, "impasto")
        th = dict(thick=2.0, heavy=4.0).get(thickness, thickness) if isinstance(thickness, str) else float(thickness)
        if isinstance(th, str): raise StudioError("thickness: thick, heavy, or a number of volume units")
        self._fill(m, p, th, strokes, prefer=10, kind="bristle", brush=brush, spacing=0.7, speed=1.5, wobble=0.5, below=0.5)

    @act({"do": "knife", "region": {"rect": [40, 40, 70, 60]}, "paint": {"oil": "white"}, "thickness": 1.2})
    def knife(self, region, paint, thickness=1.2, strokes="across", width=None):
        """Painting knife: paste spread flat with a rigid blade, leaving smooth planes with a ridge at each edge
        (rock faces, light on water, a crisp wall). Needs a paste: oil, acrylic or gouache."""
        _, m = self._region(region)
        p = self.paint(paint, over=m); self._paste(p, "the knife")
        w = float(width or min(14.0, max(3.5, _depth(m).max())))
        self._fill(m, p, float(thickness), strokes, kind="knife", brush=w, spacing=0.8, speed=2.0, wobble=0.3, below=0.5)

    @act({"do": "scrape", "region": {"rect": [40, 40, 70, 60]}}, kind="takes")
    def scrape(self, region, down_to=0.05, strokes="across", width=None):
        """Scrape: an empty knife drawn through wet paint, taking it down to a thin film (or, edge-on, to the
        ground): light lines in a dark field, texture, a correction. Takes only what is wet."""
        _, m = self._region(region)
        w = float(width or min(10.0, max(2.5, _depth(m).max())))
        r = w / 2
        ok = self._allowed(m, r)
        if not ok.any(): raise NothingLaid(f"the region is too small for a knife {w:g} cells across")
        b = self._brush("knife", w)
        for path in self._hatch(ok, self._angle(strokes), max(1.0, 0.8 * w), r, step=2.0, rng=self._rng):
            self._stroke(b, path, 2.0, 1.0, float(down_to))
        self._used_brushes.append(b)

    @act({"do": "lift", "region": {"disc": [100, 60, 12]}}, kind="takes")
    def lift(self, region, strokes="across", brush=None, scrubs=2):
        """Lift: a clean, damp brush worked over the paint and squeezed out, taking colour back up: soft lights,
        clouds, a correction. Works on wet paint and on re-wettable paint (unfixed watercolour, gouache); a fixed or
        permanent film does not lift."""
        _, m = self._region(region)
        for _ in range(int(scrubs)):
            # a damp brush loosens the paint (it gives a little water where the film is thinner than its target) ...
            self._fill(m, self.paint("water"), 0.15, strokes, prefer=14, kind="round", brush=brush, speed=2.5,
                       pressure=0.9, wobble=0.5, frac=0.08, below=1.01)
            # ... then, squeezed out, the same brush drinks it back up with what it dissolved
            for b in {id(b): b for b in self._used_brushes}.values():
                b.rinse(); b._studio_paint = None
            self._fill(m, _Thirst(), 0.0, strokes, prefer=14, kind="round", brush=brush, speed=2.5, pressure=0.9, wobble=0.5)

    @act({"do": "blot", "region": {"disc": [100, 60, 12]}}, kind="takes")
    def blot(self, region, size=None, presses=1):
        """Blot: tissue pressed onto wet paint and lifted, taking up what is wet under it without moving it:
        soft-edged lights, clouds pressed out of a wet sky. Works only while the paint is wet."""
        _, m = self._region(region)
        w = float(size or min(20.0, max(3.0, 2 * _depth(m).max())))
        r = w / 2
        ok = _depth(m) > r * 0.6
        if not ok.any(): ok = m
        b = self._brush("blotter", min(WIDTHS, key=lambda v: abs(v - w)))
        for _ in range(int(presses)):
            for x, y in self._points(ok, 0.8 * w, self._rng, jitter=0.2):
                self._book(b.press(x, y, pressure=1.0, ticks=1, target=0.0))
            b.wring()
        self._used_brushes.append(b)

    @act({"do": "spatter", "region": {"band": [250, 300]}, "paint": "2 sienna + ultramarine", "drops": 40})
    def spatter(self, region, paint, drops=40, size=3.0):
        """Spatter: droplets flicked from a loaded brush, landing as small round spots scattered over the region
        (sand, pebbles, foam, texture). size: a droplet's mean width in cells as it lands (goo.classic.tools.spatter's
        volume is that of a dome of this width DROP_FILM thick)."""
        _, m = self._region(region)
        ys, xs = np.nonzero(m)
        if not ys.size: raise NothingLaid("the region is empty: it holds no cells of the panel")
        k = self._rng.choice(ys.size, int(drops)); pts = [(xs[i] + self._rng.uniform(0, 1), ys[i] + self._rng.uniform(0, 1)) for i in k]
        p = self.paint(paint, over=m)
        vol = np.pi * (float(size) / 2) ** 2 * _tools.DROP_FILM      # a droplet `size` cells across, landed
        b = self._brush("round", 7.0); self._charge(b, p)
        for x, y in pts:                                              # one flick per droplet, aimed at its point
            self._book(_tools.spatter(b, (x, y), 1, 0.5, vol, self._rng))
        if (self.reserved & coerce_region(region).grow(2).mask(self.H, self.W)).any():
            self._rec.notes.append("droplets that fell on the reserve stayed on the brush")

    @act({"do": "reserve", "region": {"disc": [288, 122, 17]}}, kind="other")
    def reserve(self, region):
        """Reserve: masking fluid (frisket) over the paper, so washes pass over the shape and leave it bare: a sun,
        the white of a sail, glitter on water. The paper must be touch-dry there. `release` takes it off."""
        _, m = self._region(region)
        if not m.any(): raise StudioError("the reserve's region is empty: it holds no cells of the panel")
        wet = m & ~self.cv.touch_dry()
        if wet.any():
            ys, xs = np.nonzero(wet)
            raise StudioError(f"{int(wet.sum())} cells under the reserve are wet (e.g. x={xs[0]}, y={ys[0]}): dry(until='touch') first")
        self.cv.mask(m)
        self.reserved |= m
        self._rec.notes.append(f"frisket over {int(m.sum())} cells: no paint enters or leaves them until released")

    @act({"do": "release", "region": None}, kind="other")
    def release(self, region=None):
        """Release: rub the masking fluid off (all of it, or within a region), uncovering the paper it kept."""
        m = self.reserved.copy() if region is None else self._region(region)[1] & self.reserved
        self.cv.unmask(None if region is None else m)
        self.reserved &= ~m

    @act({"do": "wait", "ticks": 60}, kind="other")
    def wait(self, ticks=60):
        """Let time pass at the room's pace (no lamp): a wet wash goes damp, then dry. For wet-in-wet timing."""
        self.cv.wait(int(ticks)); self.clock += int(ticks); self._rec.ticks += int(ticks)

    @act({"do": "dry", "until": "touch"}, kind="other")
    def dry(self, until="touch", region=None, lamp=None, spell=None, max_ticks=30000):
        """Dry, in short spells, looking after each: until "damp" (nothing glistens: the moment for a backrun or a
        soft drop-in), "touch" (dry to the finger; a permanent paint, acrylic or oil, is then fixed: it becomes a
        layer the next one sits on, while watercolour and gouache stay re-wettable and liftable) or "fixed" (dry,
        and fixed whatever the paint: a settled layer to glaze over). lamp: how much faster than the room it
        dries (default 1 for damp, 10 for touch and fixed). Judged by the surface's sheen and the finger test only."""
        if until not in ("damp", "touch", "fixed"): raise StudioError("until: damp, touch or fixed")
        self._settle()
        m = ~self.cv.touch_dry() if region is None else self._region(region)[1]     # judged where it is wet now
        boost = float(lamp if lamp is not None else (1.0 if until == "damp" else 10.0))
        spell = int(spell or (5 if until == "damp" else 40))
        self._settle()
        def done():
            if not m.any(): return True                                      # nothing wet there
            if until == "damp": return (self.look_wet()[m] > GLISTEN).mean() <= 0.1   # all but a few puddles
            return bool(self.cv.touch_dry()[m].all())
        n = 0; was = None
        while not done():
            if n >= max_ticks:
                raise StudioError(f"still not {until}-dry after {n} ticks: oil needs a canvas with a cure rate, and a lamp "
                                  f"(lamp=...) speeds it; or ask for a smaller region")
            self.cv.wait(spell, evap_boost=boost, cure_boost=boost); n += spell
        self.clock += n; self._rec.ticks += n
        if until == "fixed" or (until == "touch" and self._all_permanent()):
            f = self.cv.fix(region=None if region is None else m)
            self._fixed = f
            self._rec.notes.append(f"fixed {f:.0%} of the wet cells")
        elif until == "touch":
            self._rec.notes.append("left re-wettable (watercolour or gouache on the panel): dry until fixed to settle it")
        self._rec.notes.append(f"{until} after {n} ticks (lamp x{boost:g})")

    def _all_permanent(self):
        """Whether every paint laid since the panel was last fixed is permanent (from the journal: the painter's memory)."""
        media = set()
        for e in reversed(self.journal):
            if e["act"] == "dry" and e["receipt"].get("fixed") is not None: break
            if e.get("failed"): continue
            for k in ("paint", "from", "from_", "to"):
                if k in e["params"]: media.add(self._medium_of(e["params"][k]))
            for v in e["params"].get("via", []) or []: media.add(self._medium_of(v))
        media.discard("water")
        if not media: return self.medium in ("oil", "acrylic")
        return media <= {"oil", "acrylic"}

    def _medium_of(self, spec):
        if isinstance(spec, dict):
            for m in ("watercolour", "gouache", "acrylic", "oil"):
                if m in spec: return m
            return spec.get("medium", self.medium)
        if isinstance(spec, str):
            if spec in self.paints: return getattr(self.paints[spec], "medium", self.medium)
            if spec == "water": return "water"
            m = re.match(r"(\w+)\(", spec)
            if m and m.group(1) in ("watercolour", "gouache", "acrylic", "oil", "water"): return m.group(1)
        return self.medium


def _globals_digest(fn):
    """A hash of the globals a sitting's function names: their pickled values, or the source of functions."""
    h = hashlib.sha256()
    for n in sorted(set(fn.__code__.co_names)):
        if n not in fn.__globals__: continue
        v = fn.__globals__[n]
        if inspect.ismodule(v) or inspect.isclass(v) or inspect.isbuiltin(v): continue
        try:
            b = inspect.getsource(v).encode() if callable(v) and hasattr(v, "__code__") else pickle.dumps(v, protocol=5)
        except Exception:
            b = repr(v).encode()
        h.update(n.encode() + b"=" + b)
    return h.hexdigest()


@functools.lru_cache(None)
def _goo_version():
    """A hash of goo's own source: a cached sitting painted by other laws, tools or techniques is not reused."""
    d = os.path.dirname(os.path.abspath(__file__)); h = hashlib.sha256()
    for root, _, files in sorted(os.walk(d)):
        for f in sorted(files):
            if f.endswith(".py"): h.update(open(os.path.join(root, f), "rb").read())
    return h.hexdigest()


def _sig(fn):
    s = inspect.signature(fn)
    ps = [p for p in list(s.parameters.values())[1:]]
    return "(" + ", ".join(p.name.rstrip("_") + ("" if p.default is inspect.Parameter.empty else f"={p.default!r}") for p in ps) + ")"


def vocabulary():
    """The studio's acts as text: each with its signature, what it is for, and one example (for `easel help`)."""
    out = []
    for name, run in ACTS.items():
        fn = run._act["fn"]; doc = inspect.getdoc(fn) or ""
        out.append(f"{name}{_sig(fn)}\n" + textwrap.indent(textwrap.fill(" ".join(doc.split()), 100), "    ")
                   + f"\n    example: {json.dumps(run._act['example'])}")
    return "\n\n".join(out)


__all__ = ["Studio", "Region", "Receipt", "NothingLaid", "StudioError", "everywhere", "disc", "band", "rect", "polygon",
           "below", "above", "mask_region", "coerce_region", "ACTS", "vocabulary", "MARGIN", "GLISTEN", "DAMP"]
