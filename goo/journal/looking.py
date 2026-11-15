"""Looking at the painting (SPEC §7, §9): goo.light on the world's stack, turned into pictures by goo.observe.

A look reads the world through World.stack() (read-only views) and never writes to it. Exposure is a parameter of
the camera: linear sRGB is multiplied by it before the display transform (the sRGB encoding and its clip), so a
film with gain, which may return more light than white, can be seen without clipping, and a dark passage can be
lifted. Gloss is read as goo.observe.gloss (specular at 60 degrees, in gloss units) and shown as grey with 100 GU
white; that is how a painter sees wetness. The infrared image is the band reflectance above 780 nm, grey, linear."""
from dataclasses import dataclass
import numpy as np
from goo.film import Bands, Light, Stack, Film, Ground
from goo.light import look as solve_look
from goo.substance import maps
from goo.observe import colour as C
from goo.observe import readings as R
from goo.ports import geometry as G


def bands_of(spec):
    """Bands from a JSON spec {"lo": m, "hi": m, "step": m, "ir": [m, ...], "ir_width": m}."""
    s = dict(lo=380e-9, hi=740e-9, step=20e-9, ir=[1000e-9], ir_width=100e-9)
    s.update(spec or {})
    b = Bands.uniform(s["lo"], s["hi"], s["step"])
    for c in s["ir"]:
        b = b.with_band(c, s["ir_width"])
    return b


@dataclass(frozen=True)
class Seen:
    """What a look returns to a painter: the goo.light Look (new arrays), the picture, and the readings asked for.
    box: (row0, row1, col0, col1) of the panel the look covers."""
    look: object
    image: np.ndarray                 # [h, w, 3] uint8 sRGB
    box: tuple
    exposure: float
    light: str
    gloss: object = None              # goo.observe Reading [h, w] GU
    gloss_image: np.ndarray = None    # [h, w] uint8
    infrared: object = None           # goo.observe Reading [h, w, b]
    infrared_image: np.ndarray = None

    @property
    def receipt(self):
        return self.look.receipt


def _box(mask):
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if rows.size == 0:
        raise ValueError("look: the region holds no cell")
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _crop(stack, box):
    y0, y1, x0, x1 = box
    g = stack.ground
    ground = Ground(g.z[y0:y1, x0:x1], g.solid, g.porosity, g.pore_radius, g.depth, g.substances,
                    g.held[:, y0:y1, x0:x1])
    return Stack(ground, [Film(f.substances, f.q[:, y0:y1, x0:x1]) for f in stack.films], stack.dx)


def look(world, bands, light="daylight", exposure=1.0, region=None, gloss=False, infrared=False, n_ambient=1.0):
    """Look at a world (read only). region: a region description; the look covers its bounding box."""
    if light not in C.ILLUMINANTS:
        raise ValueError(f"light: one of {', '.join(C.ILLUMINANTS)}")
    exposure = float(exposure)
    if not (np.isfinite(exposure) and exposure > 0):
        raise ValueError("exposure must be > 0")
    ill = C.ILLUMINANTS[light]
    box = (0, world.H, 0, world.W) if region in (None, "all") else _box(G.region(region, world.H, world.W, world.dx))
    stack = _crop(world.stack(), box)
    L = solve_look(stack, bands, R.diffuse_light(bands, ill), maps, n_ambient=n_ambient, t=world.t)
    col = R.colour(L, ill)
    img = C.srgb8(col.rgb * exposure)
    out = dict(look=L, image=img, box=box, exposure=exposure, light=ill.name)
    if gloss:
        Lg = solve_look(stack, bands, R.gloss_light(bands, ill), maps, n_ambient=n_ambient, t=world.t)
        gr = R.gloss(Lg, ill)
        out["gloss"] = gr
        out["gloss_image"] = np.round(np.clip(gr.value / 100.0, 0, 1) * 255).astype(np.uint8)
    if infrared:
        ir = R.infrared(L)
        if ir.value.shape[-1] == 0:
            raise ValueError("look: the session's bands have none above 780 nm")
        out["infrared"] = ir
        out["infrared_image"] = np.round(np.clip(ir.value[..., -1], 0, 1) * 255).astype(np.uint8)
    return Seen(**out)


def write_png(a, path, scale=1):
    from PIL import Image
    a = np.asarray(a)
    if scale > 1:
        a = np.repeat(np.repeat(a, scale, 0), scale, 1)
    Image.fromarray(a, "RGB" if a.ndim == 3 else "L").save(path)
    return str(path)
