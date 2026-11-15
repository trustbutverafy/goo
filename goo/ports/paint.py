"""Paints (recipes) and vessels (bodies that hold a finite amount of one paint). docs/PORTS.md §1.

A Paint is a name and a recipe {substance name: volume fraction}. It names no medium: what it does comes from its
substances. A Vessel is a body: it holds a mass of each substance of one paint, its matter account is
`body[name]` in the world's matter ledger, and everything a source deposits is drawn from it and booked from it."""
from dataclasses import dataclass, field
import numpy as np


class PortError(ValueError):
    """An act that cannot be done as asked. The act changes nothing."""


FRACTION_TOL = 1e-9          # how far from 1 a recipe's fractions may sum (rounding of typed decimals); renormalised


@dataclass(frozen=True)
class Paint:
    name: str
    recipe: tuple                      # ((substance name, volume fraction), ...) sorted by name

    @classmethod
    def of(cls, name, recipe, substances):
        """Validate a recipe against the world's substances (names). Fractions are volume fractions, each >= 0,
        summing to 1 within FRACTION_TOL (then renormalised exactly)."""
        if not isinstance(name, str) or not name:
            raise PortError("a paint needs a name")
        if not isinstance(recipe, dict) or not recipe:
            raise PortError(f"paint {name!r}: a recipe is {{substance: volume fraction}}")
        known = [getattr(s, "name", s) for s in substances]
        items = {}
        for k, v in recipe.items():
            k = getattr(k, "name", k)
            if k not in known:
                raise PortError(f"paint {name!r}: {k!r} is not a substance of this world (it has {', '.join(known)})")
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise PortError(f"paint {name!r}: fraction of {k} is not a number") from None
            if not (np.isfinite(v) and v >= 0):
                raise PortError(f"paint {name!r}: fraction of {k} must be >= 0")
            if v > 0:
                items[k] = items.get(k, 0.0) + v
        tot = sum(items.values())
        if not items or abs(tot - 1.0) > FRACTION_TOL:
            raise PortError(f"paint {name!r}: volume fractions sum to {tot:.12g}, not 1")
        return cls(name, tuple(sorted((k, v / tot) for k, v in items.items())))

    def phi(self, substances):
        """Volume fractions [S] in the world's order."""
        d = dict(self.recipe)
        return np.array([d.get(getattr(s, "name", s), 0.0) for s in substances])

    def as_dict(self):
        return dict(self.recipe)

    @staticmethod
    def mix(name, parts, substances):
        """A new paint from {Paint: parts by volume}."""
        if not parts:
            raise PortError("mix: nothing to mix")
        tot = 0.0
        acc = {}
        for p, n in parts.items():
            n = float(n)
            if not (np.isfinite(n) and n >= 0):
                raise PortError("mix: parts must be >= 0")
            tot += n
            for k, v in p.recipe:
                acc[k] = acc.get(k, 0.0) + n * v
        if tot <= 0:
            raise PortError("mix: parts sum to 0")
        return Paint.of(name, {k: v / tot for k, v in acc.items()}, substances)


@dataclass
class Vessel:
    """A body holding one paint. mass [S] kg, in the world's substance order. Its account is body[name]."""
    name: str
    paint: Paint
    mass: np.ndarray
    rho: np.ndarray = field(repr=False, default=None)

    @property
    def account(self):
        return f"body[{self.name}]"

    @property
    def volume(self):
        """m^3 it holds (volumes add)."""
        return float(np.sum(self.mass / self.rho))

    @property
    def dust(self):
        """The volume below which what a vessel holds is the rounding of its draws (1e-12 of what it held), m^3."""
        return 1e-12 * self.filled

    filled: float = 0.0                # m^3 ever put in

    def take(self, volume):
        """The volume it can give of what is asked (m^3): all of it, or what it holds."""
        return min(float(volume), self.volume)
