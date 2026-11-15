"""Films, grounds, stacks, bands and lights: plain data with no laws (SPEC §5). A substance here is any object
with a `name` and a density `rho` in kg/m^3; goo.substance defines the real ones."""
from dataclasses import dataclass, field
import numpy as np
from goo.units import nm, T_room, g_0


@dataclass(frozen=True)
class Bands:
    """Spectral bands: centres and widths in m. No code may assume how many there are."""
    centres: np.ndarray
    widths: np.ndarray

    def __post_init__(self):
        c, w = np.asarray(self.centres, np.float64), np.asarray(self.widths, np.float64)
        if c.ndim != 1 or c.shape != w.shape or np.any(w <= 0) or np.any(c <= 0):
            raise ValueError("Bands: centres and widths must be positive vectors of one length")
        object.__setattr__(self, "centres", c); object.__setattr__(self, "widths", w)

    @property
    def B(self):
        return self.centres.size

    @classmethod
    def uniform(cls, lo=380 * nm, hi=740 * nm, step=20 * nm):
        n = int(round((hi - lo) / step))
        return cls(lo + step * (np.arange(n) + 0.5), np.full(n, step))

    def with_band(self, centre, width):
        return Bands(np.append(self.centres, centre), np.append(self.widths, width))


@dataclass(frozen=True)
class Light:
    """An illuminant and how it arrives. power: relative power per band. geometry: 'diffuse', or
    ('collimated', angle from the panel's normal in radians)."""
    bands: Bands
    power: np.ndarray
    geometry: object = "diffuse"
    name: str = ""

    def __post_init__(self):
        p = np.asarray(self.power, np.float64)
        if p.shape != (self.bands.B,) or np.any(p < 0):
            raise ValueError("Light: power must be non-negative, one value per band")
        object.__setattr__(self, "power", p)


@dataclass
class Film:
    """One layer of matter. q[S, H, W]: volume per area of each substance, in m. state: named fields over the
    cells (cure, coalescence, alignment, temperature), declared by the substances that evolve them."""
    substances: tuple
    q: np.ndarray
    state: dict = field(default_factory=dict)

    def __post_init__(self):
        self.substances = tuple(self.substances)
        self.q = np.asarray(self.q, np.float64)
        if self.q.ndim != 3 or self.q.shape[0] != len(self.substances):
            raise ValueError("Film: q must be [S, H, W] with one plane per substance")
        names = [s.name for s in self.substances]
        if len(set(names)) != len(names):
            raise ValueError(f"Film: a substance appears twice: {names}")

    @classmethod
    def empty(cls, substances, H, W):
        return cls(tuple(substances), np.zeros((len(tuple(substances)), H, W)))

    @classmethod
    def uniform(cls, amounts, H=1, W=1):
        """A film of the same composition everywhere. amounts: {substance: volume per area in m}."""
        subs = tuple(amounts)
        q = np.array([np.full((H, W), float(amounts[s])) for s in subs])
        return cls(subs, q)

    @property
    def shape(self):
        return self.q.shape[1:]

    @property
    def h(self):
        """Thickness, m: the volumes add (SPEC §3, evidence `assumed`)."""
        return self.q.sum(0)

    @property
    def phi(self):
        """Volume fraction of each substance, [S, H, W]; zero where the film is empty."""
        h = self.h
        return np.divide(self.q, h, out=np.zeros_like(self.q), where=h > 0)

    def index(self, name):
        for i, s in enumerate(self.substances):
            if s.name == name:
                return i
        raise KeyError(name)

    def mass(self, dx):
        """kg of each substance in the film, for cells of side dx."""
        rho = np.array([s.rho for s in self.substances], np.float64)
        return rho * self.q.sum((1, 2)) * dx * dx

    def copy(self):
        return Film(self.substances, self.q.copy(), {k: np.array(v) for k, v in self.state.items()})


@dataclass
class Ground:
    """What is painted on: its relief, a porous body (its solid, porosity, pore radius, depth) and what its
    pores hold, held[S, H, W] in m of each of `substances`."""
    z: np.ndarray
    solid: object
    porosity: float
    pore_radius: float
    depth: float
    substances: tuple = ()
    held: np.ndarray = None

    def __post_init__(self):
        self.z = np.asarray(self.z, np.float64)
        self.substances = tuple(self.substances)
        if not 0.0 <= self.porosity < 1.0:
            raise ValueError("Ground: porosity in [0, 1)")
        if self.held is None:
            self.held = np.zeros((len(self.substances),) + self.z.shape)
        self.held = np.asarray(self.held, np.float64)
        if self.held.shape != (len(self.substances),) + self.z.shape:
            raise ValueError("Ground: held must be [S, H, W]")

    @property
    def capacity(self):
        """Volume per area its pores can hold, m."""
        return self.porosity * self.depth

    def copy(self):
        return Ground(self.z.copy(), self.solid, self.porosity, self.pore_radius, self.depth, self.substances,
                      self.held.copy())


@dataclass
class Stack:
    """A ground and films, bottom first, on cells of side dx (m)."""
    ground: Ground
    films: list
    dx: float

    def __post_init__(self):
        self.films = list(self.films)
        for f in self.films:
            if f.shape != self.ground.z.shape:
                raise ValueError("Stack: a film's cells differ from the ground's")

    @property
    def top(self):
        """Height of the top surface above the datum, m."""
        return self.ground.z + sum((f.h for f in self.films), np.zeros_like(self.ground.z))


@dataclass(frozen=True)
class Environment:
    """What the world sits in: reservoirs at fixed potential, and fields. T in K. gravity (g_x, g_y, g_n) in
    m/s^2: g_x along columns, g_y down rows, g_n pressing the film onto the panel. activity: substance name ->
    activity of that reservoir in the atmosphere, > 0 (water: relative humidity; oxygen: its mole fraction).
    boundary_layer: m of still air a vapour diffuses through. Turning an activity into a potential needs the
    substance, so the maps do it (SPEC §4.3, exchange)."""
    T: float = T_room
    gravity: tuple = (0.0, 0.0, g_0)
    activity: dict = field(default_factory=dict)
    boundary_layer: float = 5e-3

    def __post_init__(self):
        g = tuple(float(x) for x in self.gravity)
        if len(g) != 3:
            raise ValueError("Environment: gravity is (g_x, g_y, g_n)")
        if self.T <= 0 or self.boundary_layer <= 0 or any(not a > 0 for a in self.activity.values()):
            raise ValueError("Environment: T, boundary_layer and every activity must be positive (a reservoir "
                             "at activity 0 would take matter at unbounded gain of free energy; leave a "
                             "substance out of `activity` if the atmosphere holds none and cannot take any)")
        object.__setattr__(self, "gravity", g)

    def with_(self, **changes):
        """A changed environment (an act: SPEC §8)."""
        d = dict(T=self.T, gravity=self.gravity, activity=dict(self.activity), boundary_layer=self.boundary_layer)
        d.update(changes)
        return Environment(**d)
