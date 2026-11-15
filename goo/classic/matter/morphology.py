"""goo.classic.matter.morphology: what shape the matter is in. A particle population is a size distribution and a
particle kind; the kind says how its optical cross-sections are derived (or that they are not, yet).

Implemented kinds (their optics is Mie theory, goo.classic.matter.mie):
  Sphere    homogeneous sphere of the substance's own dielectric function
  Coated    a core of the substance's dielectric inside a shell of another (titania-coated mica is a flake, not
            this; this is e.g. a silica- or alumina-coated rutile, or a dye core in a clear shell), radius
            ratio core / outer
  Hollow    a shell of the substance around a void (opaque polymer pigments: hollow styrene-acrylic spheres,
            the void scatters in place of titania), radius ratio void / outer
Declared but not derivable (the description is held; asking for optics raises NotDerivable, so goo says it
does not model it rather than guessing):
  Shape     any other particle: a platelet, a needle, a fractal aggregate; named, with its parameters

A size distribution is log-normal in diameter, given by its volume-median diameter d (um) and geometric
standard deviation sigma_g (>= 1). Integrals over it use Gauss-Hermite quadrature in ln d on the *volume*
distribution, because everything goo needs is per unit particle volume.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


class NotDerivable(NotImplementedError):
    """The description is admissible as data, but goo has no law that derives this quantity from it."""


@dataclass(frozen=True)
class LogNormal:
    d_um: float = 0.5                  # volume-median diameter
    sigma_g: float = 1.5               # geometric standard deviation (1 = monodisperse)

    def nodes(self, q=9):
        """(d_um, w): diameters and volume weights (sum 1) of a q-point Gauss-Hermite rule in ln d."""
        s = np.log(max(self.sigma_g, 1.0))
        if s < 1e-6: return np.array([self.d_um]), np.array([1.0])
        t, w = np.polynomial.hermite.hermgauss(q)
        return self.d_um * np.exp(np.sqrt(2) * s * t), w / np.sqrt(np.pi)

    # number-distribution statistics (the distribution of particle diameters, as a sizer counts them)
    def number_median(self): return self.d_um * np.exp(-3 * np.log(self.sigma_g) ** 2)
    def polydispersity(self):
        """delta = std / mean of the number distribution of diameters: sqrt(exp(s^2) - 1)."""
        return float(np.sqrt(np.expm1(np.log(self.sigma_g) ** 2)))
    def skewness(self):
        e = np.exp(np.log(self.sigma_g) ** 2); return float((e + 2) * np.sqrt(e - 1))
    def sauter(self):
        """Sauter mean diameter d32 (volume / surface), um: the diameter whose volume-to-area ratio the powder
        has. For log-normal: volume median * exp(-s^2 / 2)."""
        return float(self.d_um * np.exp(-0.5 * np.log(self.sigma_g) ** 2))


@dataclass(frozen=True)
class Sphere:
    size: LogNormal = LogNormal()
    kind = "sphere"
    def describe(self): return dict(kind=self.kind, d_um=self.size.d_um, sigma_g=self.size.sigma_g)
    def solid_fraction(self): return 1.0           # of the envelope volume


@dataclass(frozen=True)
class Coated:
    size: LogNormal
    shell: object                      # a goo.classic.matter.dielectric.Dielectric
    ratio: float = 0.8                 # core radius / outer radius
    shell_density: float = 4.0         # g/cm^3
    kind = "coated"
    def describe(self):
        return dict(kind=self.kind, d_um=self.size.d_um, sigma_g=self.size.sigma_g, ratio=self.ratio,
                    shell=self.shell.describe(), shell_density=self.shell_density)
    def solid_fraction(self): return 1.0


@dataclass(frozen=True)
class Hollow:
    size: LogNormal
    ratio: float = 0.55                # void radius / outer radius (opaque polymers: ~0.5-0.6)
    kind = "hollow"
    def describe(self): return dict(kind=self.kind, d_um=self.size.d_um, sigma_g=self.size.sigma_g, ratio=self.ratio)
    def solid_fraction(self): return 1.0 - self.ratio ** 3


@dataclass(frozen=True)
class Shape:
    """A particle goo cannot derive optics or packing for yet: held as data, labelled, never guessed."""
    name: str
    size: LogNormal = LogNormal()
    params: tuple = ()                 # (key, value) pairs, e.g. (("aspect", 20.0),)
    kind = "shape"
    def describe(self): return dict(kind=self.kind, name=self.name, d_um=self.size.d_um,
                                    sigma_g=self.size.sigma_g, params=dict(self.params))
    def solid_fraction(self): return 1.0
