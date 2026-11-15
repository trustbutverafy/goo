"""Readings a painter or conservator would take of a Look (docs/OBSERVE.md §2).

A sensor is a measurement model over a `goo.light.Look`: it takes a Look (never a world, a stack or a film), reads
its arrays, and returns new ones. It never writes to what it reads (tested with read-only arrays). Every reading
carries an evidence class and the flags of the look it read:

  evidence   the weakest of the evidence the caller states for the look's matter (`matter`, default `assumed`:
             a sensor cannot know where the matter came from), the measurement model's own provenance, and
             `assumed` if the look's receipt says a map was used out of its domain or a bound was collapsed.
  flags      the look's receipt flags (out of domain, bounds collapsed, impossible), unchanged.

Geometry. Colour is read with the specular reflection excluded ("sce", as a 45/0 or d/8 instrument with a trap
does), from `Look.diffuse`, by default; "sci" reads `Look.reflectance`. Gloss is the specular reflectance of a look
under collimated light at 60 degrees, against a polished black glass of index 1.567 (the ISO 2813 reference:
recalled, assumed), in gloss units. Hiding is the contrast ratio Y(over black) / Y(over white) of two looks.
"""
from dataclasses import dataclass, field
import numpy as np
from goo.evidence import Provenance, weakest
from goo.light import Look, fresnel
from goo.observe import colour as C

GLOSS_ANGLE = np.radians(60.0)
GLOSS_REFERENCE_INDEX = 1.567
PROV = {
    "gloss": Provenance("assumed", "ISO 2813 specular gloss at 60 degrees against black glass of index 1.567",
                        note="geometry and reference index recalled, not read; the reference's reflectance is "
                             "computed here by Fresnel's equations, not typed"),
    "infrared": Provenance("exact", note="the look's own band reflectance, read without a model (a reflectogram "
                                         "with a flat detector response across the band)"),
    "contrast": Provenance("exact", note="Y over black divided by Y over white: an identity of two readings"),
}


@dataclass(frozen=True)
class Reading:
    """A measured number (or array) with its unit, evidence class and the flags of the look it was read from."""
    name: str
    value: object
    unit: str
    evidence: str
    flags: tuple = ()
    note: str = ""

    def __post_init__(self):
        weakest(self.evidence)


def _check(look):
    if not isinstance(look, Look):
        raise TypeError(f"a sensor reads a goo.light.Look, not {type(look).__name__}")
    return look


def look_evidence(look, matter="assumed", *model):
    """Evidence of what a look shows: the weakest of `matter`, the model's provenance classes, and `assumed` if the
    look's receipt has an out-of-domain or bounds-collapsed flag."""
    ev = weakest(matter, *model) if model else weakest(matter)
    if any(f.kind in ("out_of_domain", "bounds_collapsed") for f in look.receipt.flags):
        ev = weakest(ev, "assumed")
    return ev


@dataclass(frozen=True)
class Colour:
    """XYZ, CIELAB (relative to the light's white) and sRGB (linear, adapted to the light) [H, W, 3]."""
    illuminant: str
    XYZ: np.ndarray
    lab: np.ndarray
    rgb: np.ndarray
    out_of_gamut: np.ndarray
    above_one: np.ndarray            # [H, W]: some band reflects more than it receives (gain)
    unresolved: np.ndarray           # [H, W]: some band past a lasing threshold (no steady state)
    evidence: str
    flags: tuple
    geometry: str

    @property
    def srgb8(self):
        return C.srgb8(self.rgb)


def colour(look, illuminant, observer=C.CIE1931, geometry="sce", matter="assumed"):
    """The colour of a look under a named illuminant (goo.observe.colour.Illuminant)."""
    _check(look)
    if geometry not in ("sce", "sci"):
        raise ValueError("geometry: 'sce' (specular excluded) or 'sci' (included)")
    R = np.array(look.diffuse if geometry == "sce" else look.reflectance, dtype=float)
    X = C.xyz(R, look.bands, illuminant, observer)
    w = C.white_xyz(look.bands, illuminant, observer)
    L = C.lab(X, w)
    rgb = C.srgb_linear(X, w)
    ev = look_evidence(look, matter, observer.prov.evidence, illuminant.prov.evidence, C.PROV["srgb"].evidence,
                       C.PROV["cielab"].evidence, C.PROV["bands"].evidence)
    return Colour(illuminant.name, X, L, rgb, C.out_of_gamut(rgb), np.any(look.reflectance > 1.0, -1),
                  np.any(look.unresolved > 0, -1), ev, tuple(look.receipt.flags), geometry)


def gloss(look, illuminant, observer=C.CIE1931, matter="assumed"):
    """Specular gloss in gloss units [H, W]: 100 Y(specular) / Y(Fresnel of index 1.567 at 60 degrees), from a look
    the caller made under collimated light at GLOSS_ANGLE (goo.observe.gloss_light). Y-weighted under the light."""
    _check(look)
    W = C.weights(look.bands, illuminant, observer)[1]
    ref = float(fresnel(np.float64(GLOSS_REFERENCE_INDEX), np.cos(GLOSS_ANGLE)))
    val = 100.0 * (np.asarray(look.specular) @ W) / ref
    ev = look_evidence(look, matter, PROV["gloss"].evidence, observer.prov.evidence)
    return Reading("gloss 60", val, "GU", ev, tuple(look.receipt.flags),
                   f"specular at 60 deg over black glass n={GLOSS_REFERENCE_INDEX} (Fresnel {ref:.5f})")


def gloss_light(bands, illuminant, angle=GLOSS_ANGLE):
    """The goo.film.Light a gloss reading needs: the illuminant, collimated at `angle` from the normal."""
    from goo.film import Light
    return Light(bands, illuminant.band_power(bands), ("collimated", float(angle)), f"{illuminant.name}, {np.degrees(angle):g} deg")


def diffuse_light(bands, illuminant):
    from goo.film import Light
    return Light(bands, illuminant.band_power(bands), "diffuse", illuminant.name)


def infrared(look, lo=780e-9, geometry="sce", matter="assumed"):
    """An infrared image: band reflectance [H, W, b] of the bands centred above `lo` (m), with their centres."""
    _check(look)
    sel = np.asarray(look.bands.centres) > lo
    R = np.array(look.diffuse if geometry == "sce" else look.reflectance, dtype=float)[..., sel]
    ev = look_evidence(look, matter, PROV["infrared"].evidence)
    return Reading("infrared", R, "1", ev, tuple(look.receipt.flags),
                   "bands " + ", ".join(f"{c * 1e9:.0f} nm" for c in np.asarray(look.bands.centres)[sel]))


def luminance(look, illuminant, observer=C.CIE1931, geometry="sce"):
    _check(look)
    R = np.asarray(look.diffuse if geometry == "sce" else look.reflectance, float)
    return R @ C.weights(look.bands, illuminant, observer)[1]


def contrast_ratio(over_black, over_white, illuminant, observer=C.CIE1931, geometry="sce", matter="assumed"):
    """Hiding: Y over black / Y over white [H, W] (1 hides completely)."""
    _check(over_black); _check(over_white)
    yb = luminance(over_black, illuminant, observer, geometry)
    yw = luminance(over_white, illuminant, observer, geometry)
    val = np.divide(yb, yw, out=np.zeros_like(yb), where=yw > 0)
    ev = weakest(look_evidence(over_black, matter, PROV["contrast"].evidence, observer.prov.evidence),
                 look_evidence(over_white, matter))
    return Reading("contrast ratio", val, "1", ev, tuple(dict.fromkeys(over_black.receipt.flags + over_white.receipt.flags)),
                   f"Y over black / Y over white, {geometry}")
