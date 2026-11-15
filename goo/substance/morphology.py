"""What shape a substance's matter is in (SPEC §4.1 `morph`). None means a continuous phase.

A particle population is a size distribution and a particle kind. Sizes are log-normal in diameter, given by the
volume-median diameter d (m) and the geometric standard deviation sigma_g >= 1. Optical integrals use Gauss-Hermite
quadrature in ln d on the volume distribution (everything goo needs is per unit particle volume); the number-
distribution moments used by packing, diffusion and mixing are closed forms of the log-normal.

  Sphere   homogeneous sphere of the substance's own dielectric function
  Coated   a core of the substance inside a shell of another dielectric (radius ratio core / outer)
  Hollow   a shell of the substance around a void (radius ratio void / outer): opaque-polymer pigments
  Shape    any other particle (platelet, needle, aggregate), held as data. Every map that needs a law for it
           raises NotDerivable, or (in goo.substance.maps) marks the cells it occupies out of domain.

Promoted from goo/classic/matter/morphology.py, sizes in m instead of um.
"""
from dataclasses import dataclass
import numpy as np


class NotDerivable(NotImplementedError):
    """The description is admissible as data, but goo has no law that derives this quantity from it."""


@dataclass(frozen=True)
class LogNormal:
    d: float                          # volume-median diameter, m
    sigma_g: float = 1.0              # geometric standard deviation (1: monodisperse)

    @property
    def s(self):
        return float(np.log(max(self.sigma_g, 1.0)))

    def nodes(self, q=16):
        """(d, w): diameters (m) and volume weights (sum 1) of a q-point Gauss-Hermite rule in ln d."""
        if self.s < 1e-9:
            return np.array([self.d]), np.array([1.0])
        t, w = np.polynomial.hermite_e.hermegauss(q)
        return self.d * np.exp(self.s * t), w / w.sum()

    # closed-form moments; number distribution median d_n = d exp(-3 s^2)
    def number_moment(self, k):
        """E_number[d^k], m^k."""
        s2 = self.s ** 2
        return (self.d * np.exp(-3 * s2)) ** k * np.exp(0.5 * k * k * s2)

    def volume_moment(self, k):
        """E_volume[d^k], m^k (the volume-weighted distribution is log-normal with median d)."""
        return self.d ** k * np.exp(0.5 * k * k * self.s ** 2)

    def particle_volume(self):
        """Number-mean particle volume, m^3."""
        return np.pi / 6 * self.number_moment(3)

    def sauter(self):
        """d32 = E_n[d^3] / E_n[d^2], m."""
        return self.number_moment(3) / self.number_moment(2)


@dataclass(frozen=True)
class Sphere:
    size: LogNormal
    kind = "sphere"

    def solid_fraction(self): return 1.0


@dataclass(frozen=True)
class Coated:
    size: LogNormal
    shell: object                     # a goo.substance.dielectric description
    ratio: float = 0.8                # core radius / outer radius
    shell_rho: float = 4000.0         # kg/m^3
    kind = "coated"

    def solid_fraction(self): return 1.0


@dataclass(frozen=True)
class Hollow:
    size: LogNormal
    ratio: float = 0.55               # void radius / outer radius
    kind = "hollow"

    def solid_fraction(self): return 1.0 - self.ratio ** 3


@dataclass(frozen=True)
class Shape:
    """A particle goo cannot derive optics, packing or thickening for yet: held, labelled, never guessed."""
    name: str
    size: LogNormal
    params: tuple = ()
    kind = "shape"

    def solid_fraction(self): return 1.0


SPHERICAL = ("sphere", "coated", "hollow")


def envelope_rho(morph, rho):
    """Density of a particle's envelope (kg/m^3): the substance's own, a void's nothing, a shell's its own."""
    if morph.kind == "hollow":
        return rho * morph.solid_fraction()
    if morph.kind == "coated":
        c = morph.ratio ** 3
        return rho * c + morph.shell_rho * (1 - c)
    return rho
