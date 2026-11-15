"""goo.observe: sensors, measurement models over looks (SPEC §1, §9). Imports goo.light and goo.film (and units,
evidence, ledger) only. docs/OBSERVE.md.

    colour      CIE 1931 observer (analytic fit), illuminants A and a Planckian daylight, XYZ, CIELAB, sRGB, CIEDE2000
    readings    colour under a light, gloss, infrared, contrast ratio (hiding); each a Reading with evidence and flags
    image       PNG of a look
"""
from goo.observe import colour
from goo.observe.colour import (Illuminant, Observer, CIE1931, A, DAYLIGHT, ILLUMINANTS, xyz, lab, ciede2000,
                                srgb_linear, srgb8, encode, weights, white_xyz, lab_to_xyz)
from goo.observe.readings import (Reading, Colour, colour as colour_of, gloss, gloss_light, diffuse_light, infrared,
                                  contrast_ratio, luminance, look_evidence, GLOSS_ANGLE)
from goo.observe.image import write_png, write_ir_png, rgb8

__all__ = ["colour", "Illuminant", "Observer", "CIE1931", "A", "DAYLIGHT", "ILLUMINANTS", "xyz", "lab", "ciede2000",
           "srgb_linear", "srgb8", "encode", "weights", "white_xyz", "lab_to_xyz", "Reading", "Colour", "colour_of",
           "gloss", "gloss_light", "diffuse_light", "infrared", "contrast_ratio", "luminance", "look_evidence",
           "GLOSS_ANGLE", "write_png", "write_ir_png", "rgb8"]
