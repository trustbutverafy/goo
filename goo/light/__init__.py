"""goo.light: layer operators, the star product, looks and the light ledger (SPEC §7). docs/LIGHT.md.

Light reaches matter only through an object satisfying `Optical`, passed to every entry point."""
from goo.light.optical import Optical
from goo.light.operator import (Layer, Solution, identity, symmetric, scalar, channels, star, product, solve,
                                passivity_excess)
from goo.light.bulk import homogeneous
from goo.light.interface import (fresnel, external_diffuse, diffuse_reflectance, boundary, specular_diffuse, entry,
                               tis_exponent)
from goo.light.look import Operators, Look, operators, look

__all__ = ["Optical", "Layer", "Solution", "identity", "symmetric", "scalar", "channels", "star", "product",
           "solve", "passivity_excess", "homogeneous", "fresnel", "external_diffuse", "diffuse_reflectance",
           "boundary", "specular_diffuse", "entry", "tis_exponent", "Operators", "Look", "operators", "look"]
