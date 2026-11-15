"""Images of looks: a PNG of what a look shows under a named light, through the colorimetric observer."""
import numpy as np
from goo.observe import colour as C
from goo.observe.readings import colour as _colour


def rgb8(look, illuminant, observer=C.CIE1931, geometry="sce"):
    """[H, W, 3] uint8 sRGB of a look (adapted to the light; clipped into the gamut)."""
    return _colour(look, illuminant, observer, geometry).srgb8


def write_png(look, illuminant, path, scale=1, observer=C.CIE1931, geometry="sce"):
    """Write the look's colour as a PNG (each cell `scale` x `scale` pixels). Returns the path."""
    from PIL import Image
    a = rgb8(look, illuminant, observer, geometry)
    if scale > 1:
        a = np.repeat(np.repeat(a, scale, 0), scale, 1)
    Image.fromarray(a, "RGB").save(path)
    return path


def write_ir_png(look, path, band=-1, scale=1):
    """Write one infrared band of a look as a greyscale PNG (reflectance 0..1 -> 0..255, linear)."""
    from PIL import Image
    from goo.observe.readings import infrared
    r = infrared(look).value
    if r.shape[-1] == 0:
        raise ValueError("the look has no band above 780 nm")
    a = np.round(np.clip(r[..., band], 0, 1) * 255).astype(np.uint8)
    if scale > 1:
        a = np.repeat(np.repeat(a, scale, 0), scale, 1)
    Image.fromarray(a, "L").save(path)
    return path
