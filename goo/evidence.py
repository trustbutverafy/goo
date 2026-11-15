"""Evidence classes and the values that carry them (SPEC §2). Imports nothing from goo."""
from dataclasses import dataclass, replace
import numpy as np

EVIDENCE = ("proved", "exact", "computed", "checked", "assumed", "guessed")   # strongest first
_RANK = {e: i for i, e in enumerate(EVIDENCE)}


def weakest(*classes):
    """The weakest of some evidence classes: what a value built from all of them may claim."""
    if not classes:
        raise ValueError("weakest() of nothing")
    for c in classes:
        if c not in _RANK:
            raise ValueError(f"not an evidence class: {c!r} (one of {EVIDENCE})")
    return max(classes, key=_RANK.__getitem__)


@dataclass(frozen=True)
class Provenance:
    """Where a constitutive number comes from. `licence` is required for `computed` and `checked`."""
    evidence: str
    source: str = ""
    licence: str = ""
    note: str = ""

    def __post_init__(self):
        weakest(self.evidence)
        if self.evidence in ("computed", "checked") and not (self.source and self.licence):
            raise ValueError(f"{self.evidence} needs a source and a licence")


@dataclass(frozen=True)
class Derived:
    """A value produced by a stated map. `in_domain` is a bool or a bool array shaped like the cells."""
    value: object
    evidence: str
    in_domain: object = True
    map: str = ""

    def __post_init__(self):
        weakest(self.evidence)

    @property
    def everywhere_in_domain(self):
        return bool(np.all(self.in_domain))


def derive(value, map_evidence, *input_evidence, in_domain=True, map=""):
    """Apply SPEC §2's rule: a derived value's evidence is the weakest of its inputs and its map's, and
    outside the map's domain of validity it is at best `assumed`."""
    ev = weakest(map_evidence, *input_evidence)
    if not np.all(in_domain):
        ev = weakest(ev, "assumed")
    return Derived(value, ev, in_domain, map)


@dataclass(frozen=True)
class Bounded:
    """A value known only between bounds. Carry both ends, or `collapse` and keep the record it returns."""
    lo: object
    hi: object
    evidence: str
    map: str = ""

    def __post_init__(self):
        weakest(self.evidence)
        if np.any(np.asarray(self.lo) > np.asarray(self.hi)):
            raise ValueError("Bounded: lo > hi")

    @property
    def width(self):
        return np.asarray(self.hi) - np.asarray(self.lo)

    def collapse(self, how="mid"):
        """(Derived, note). how: 'lo', 'hi', 'mid', or a number in 0..1 from lo to hi. The result is at best
        `assumed`; the note is for the receipt's `bounds_collapsed` flag."""
        w = {"lo": 0.0, "mid": 0.5, "hi": 1.0}.get(how, how)
        if not isinstance(w, (int, float)) or not 0.0 <= w <= 1.0:
            raise ValueError(f"collapse: how={how!r}")
        v = np.asarray(self.lo) * (1.0 - w) + np.asarray(self.hi) * w
        note = f"{self.map or 'bounded value'} collapsed at {w:g} of its bounds (width up to {float(np.max(self.width)):.3g})"
        return Derived(v, weakest(self.evidence, "assumed"), True, self.map), note
