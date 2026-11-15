"""The act vocabulary: each act's JSON parameters, how they are read, and the bench method it calls.

An act is a JSON object {"act": NAME, ...parameters}. Numbers are SI; strings carry units (goo.journal.quantities).
Regions and paths are plain JSON (goo.ports.geometry), in cells unless they say "units": "m". `vocabulary()` gives
the list with an example of each, and every example is tested to be a valid act."""
import numpy as np
from goo.journal.quantities import quantity, QuantityError
from goo.ports import PortError, GeometryError

REQ = object()
DIRECTIONS = {"right": 0.0, "down": 90.0, "left": 180.0, "up": 270.0}


class ActError(ValueError):
    pass


def _read(kind, v, what):
    if kind in ("length", "time", "volume", "flow", "angle", "temperature", "number"):
        return quantity(v, kind, what)
    if kind == "int":
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
            raise ActError(f"{what}: an integer, not {v!r}")
        return int(v)
    if kind == "str":
        if not isinstance(v, str) or not v:
            raise ActError(f"{what}: a name, not {v!r}")
        return v
    if kind == "bool":
        if not isinstance(v, bool):
            raise ActError(f"{what}: true or false")
        return v
    if kind in ("region", "path"):
        return v                                         # read by the bench, which knows the panel
    if kind == "fractions":
        if not isinstance(v, dict) or not v:
            raise ActError(f"{what}: an object {{name: number}}")
        return {str(k): quantity(x, "number", f"{what}.{k}") for k, x in v.items()}
    if kind == "direction":
        if isinstance(v, str) and v in DIRECTIONS:
            return np.radians(DIRECTIONS[v])
        return quantity(v, "angle", what)
    raise AssertionError(kind)


ACTS = {}


def act(_name, _example, **spec):
    def deco(fn):
        ACTS[_name] = dict(fn=fn, spec=spec, example=_example, doc=(fn.__doc__ or "").strip())
        return fn
    return deco


@act("paint", {"act": "paint", "name": "red", "recipe": {"water": 0.8, "gum_arabic": 0.15, "hematite": 0.05}},
     name=("str", REQ), recipe=("fractions", REQ))
def _paint(b, p):
    """Define a paint: a recipe of volume fractions of the world's substances (summing to 1)."""
    return b.paint(p["name"], p["recipe"])


@act("mix", {"act": "mix", "name": "pink", "parts": {"red": 1, "white": 3}}, name=("str", REQ), parts=("fractions", REQ))
def _mix(b, p):
    """A new paint from existing paints by parts of volume."""
    return b.mix(p["name"], p["parts"])


@act("fill", {"act": "fill", "vessel": "cup", "paint": "red", "volume": "5 mL"},
     vessel=("str", REQ), paint=("str", REQ), volume=("volume", REQ))
def _fill(b, p):
    """Fill a vessel (a body with its own matter account) with a volume of one paint."""
    return b.fill(p["vessel"], p["paint"], p["volume"])


@act("pour", {"act": "pour", "vessel": "cup", "path": [[10, 10], [60, 20]], "volume": "2 uL", "duration": "4 s",
              "width": "2 mm"},
     vessel=("str", REQ), path=("path", REQ), volume=("volume", REQ), duration=("time", REQ), width=("length", REQ),
     min_pieces=("int", 4), edge=("length", 0.0))
def _pour(b, p):
    """A spout along a path (or at a point) for a duration, at constant flow, footprint diameter `width`, its edge
    tapered over `edge` (default 0: flat)."""
    return b.pour(p["vessel"], p["path"], p["volume"], p["duration"], p["width"], p["min_pieces"], edge=p["edge"])


@act("flood", {"act": "flood", "vessel": "cup", "region": "all", "thickness": "20 um"},
     vessel=("str", REQ), region=("region", REQ), thickness=("length", REQ), duration=("time", 0.0),
     pieces=("int", 1), edge=("length", 0.0))
def _flood(b, p):
    """An even film of a thickness over a region (a pour whose spout covers it), its edge tapered over `edge`."""
    return b.flood(p["vessel"], p["region"], p["thickness"], p["duration"], p["pieces"], edge=p["edge"])


@act("drop", {"act": "drop", "vessel": "cup", "at": [30, 20], "volume": "0.5 uL"},
     vessel=("str", REQ), at=("path", REQ), volume=("volume", REQ), width=("length", None))
def _drop(b, p):
    """One drop at a point, laid as a thin cap; spreading is the world's."""
    return b.drop(p["vessel"], p["at"], p["volume"], p["width"])


@act("drip", {"act": "drip", "vessel": "cup", "path": [[10, 30], [50, 30]], "drops": 5, "volume": "0.2 uL",
              "interval": "1 s"},
     vessel=("str", REQ), path=("path", REQ), drops=("int", REQ), volume=("volume", REQ), interval=("time", REQ),
     width=("length", None))
def _drip(b, p):
    """Drops evenly along a path, one every interval."""
    return b.drip(p["vessel"], p["path"], p["drops"], p["volume"], p["interval"], p["width"])


@act("spray", {"act": "spray", "vessel": "cup", "path": [[5, 5], [60, 5]], "flow": "0.5 uL/s", "duration": "2 s",
               "cone_radius": "1.5 mm", "droplet_size": "60 um", "seed": 1},
     vessel=("str", REQ), path=("path", REQ), flow=("flow", REQ), duration=("time", REQ), cone_radius=("length", REQ),
     droplet_size=("length", REQ), seed=("int", None), min_pieces=("int", 4), edge=("length", 0.0))
def _spray(b, p):
    """Droplets from a seeded generator through a cone (its rim tapered over `edge`); what misses the panel is
    reservoir[overspray]."""
    return b.spray(p["vessel"], p["path"], p["flow"], p["duration"], p["cone_radius"], p["droplet_size"], p["seed"],
                   p["min_pieces"], edge=p["edge"])


@act("tape", {"act": "tape", "name": "edge", "region": {"rect": [0, 0, 64, 6]}}, name=("str", REQ),
     region=("region", REQ))
def _tape(b, p):
    """Lay a tape (a shield from sources and, by the privileged clock, a wall) on dry cells."""
    return b.tape(p["name"], p["region"])


@act("peel", {"act": "peel", "name": "edge"}, name=("str", REQ))
def _peel(b, p):
    """Peel a tape; what it caught leaves with it."""
    return b.peel(p["name"])


@act("tilt", {"act": "tilt", "angle": "15 deg", "direction": "down"}, angle=("angle", REQ),
     direction=("direction", "down"))
def _tilt(b, p):
    """Tilt the panel: angle from horizontal; downhill direction 'down' (+rows), 'up', 'left', 'right', or an angle
    from +x (columns) toward +y (rows)."""
    return b.tilt(p["angle"], p["direction"])


@act("level", {"act": "level"})
def _level(b, p):
    """Lay the panel flat."""
    return b.level()


@act("air", {"act": "air", "humidity": 0.6, "temperature": "20 C"}, humidity=("number", None),
     activity=("fractions", None), temperature=("temperature", None), boundary_layer=("length", None))
def _air(b, p):
    """Change the atmosphere: humidity (water's activity), any reservoir's activity, temperature, still-air layer."""
    a = dict(p["activity"] or {})
    if p["humidity"] is not None:
        a["water"] = p["humidity"]
    return b.air(a or None, p["temperature"], p["boundary_layer"])


@act("wait", {"act": "wait", "duration": "30 s"}, duration=("time", REQ), rate=("number", 1.0), where=("region", None),
     steps=("int", 10))
def _wait(b, p):
    """Let time pass; rate (>= 0) scales the clock in `where` (a privilege when != 1)."""
    return b.wait(p["duration"], p["rate"], p["where"], p["steps"])


@act("hold", {"act": "hold", "region": {"disc": [20, 20, 6]}}, region=("region", REQ))
def _hold(b, p):
    """Hold a region's clock at 0 until released (a privilege)."""
    return b.hold(p["region"])


@act("release", {"act": "release", "region": "all"}, region=("region", "all"))
def _release(b, p):
    """Release held cells."""
    return b.release(p["region"])


@act("settle", {"act": "settle"}, where=("region", "all"), horizon=("time", 86400.0), length=("length", 1e-3),
     force=("bool", False))
def _settle(b, p):
    """Turn the wet film that can no longer flow into a stratum (force: all of it, a privilege)."""
    return b.settle(p["where"], p["horizon"], p["length"], p["force"])


def parse(desc):
    """(name, params with defaults filled, as the bench takes them). Raises ActError with the reason."""
    if not isinstance(desc, dict) or "act" not in desc:
        raise ActError('an act is a JSON object with "act": one of ' + ", ".join(ACTS))
    name = desc["act"]
    if name not in ACTS:
        raise ActError(f"unknown act {name!r}: one of {', '.join(ACTS)}")
    spec = ACTS[name]["spec"]
    extra = [k for k in desc if k != "act" and k not in spec]
    if extra:
        raise ActError(f"{name}: unknown parameter(s) {', '.join(extra)}; it takes {', '.join(spec) or 'none'}")
    out = {}
    for k, (kind, default) in spec.items():
        if k in desc and desc[k] is not None:
            try:
                out[k] = _read(kind, desc[k], f"{name}.{k}")
            except QuantityError as e:
                raise ActError(str(e)) from None
        elif default is REQ:
            raise ActError(f"{name}: {k} is required")
        else:
            out[k] = default
    return name, out


def run(bench, desc):
    name, p = parse(desc)
    try:
        return ACTS[name]["fn"](bench, p)
    except (GeometryError, QuantityError) as e:
        raise ActError(f"{name}: {e}") from None


def vocabulary():
    return {k: dict(example=v["example"], doc=v["doc"], parameters={p: kind for p, (kind, _) in v["spec"].items()})
            for k, v in ACTS.items()}
