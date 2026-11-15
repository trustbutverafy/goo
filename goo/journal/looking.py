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
    L = solve_look(stack, bands, R.diffuse_light(bands, ill), OPTICS, n_ambient=n_ambient, t=world.t)
    col = R.colour(L, ill)
    img = C.srgb8(col.rgb * exposure)
    out = dict(look=L, image=img, box=box, exposure=exposure, light=ill.name)
    if gloss:
        Lg = solve_look(stack, bands, R.gloss_light(bands, ill), OPTICS, n_ambient=n_ambient, t=world.t)
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


# ---------------------------------------------------------------------------------------------------- pictures
DEDUP_MIN = 4096


def _groups(cols):
    """(representatives, inverse) of bit-identical columns of cols [k, n], verified exactly; None if not worth it."""
    from goo.journal.session import _distinct
    return _distinct(cols) if cols.shape[1] >= DEDUP_MIN else None


def _expand(v, inv, shape):
    """Expand a map's answer computed on [.., 1, n] distinct cells back to [.., H, W] (exact gather)."""
    from dataclasses import replace, is_dataclass
    from goo.evidence import Derived, Bounded
    n = inv.max() + 1 if inv.size else 0
    if isinstance(v, Derived):
        return replace(v, value=_expand(v.value, inv, shape), in_domain=_expand(v.in_domain, inv, shape))
    if isinstance(v, Bounded):
        return replace(v, lo=_expand(v.lo, inv, shape), hi=_expand(v.hi, inv, shape))
    if isinstance(v, tuple):
        items = [_expand(x, inv, shape) for x in v]
        return type(v)(*items) if hasattr(v, "_fields") else tuple(items)
    a = np.asarray(v)
    if a.ndim >= 2 and a.shape[-2] == 1 and a.shape[-1] == n:
        return a[..., 0, :][..., inv].reshape(a.shape[:-2] + shape)
    return v


class Optics:
    """goo.substance.maps as goo.light's Optical protocol, each function evaluated once per distinct cell (cells are
    grouped by the bits of every per-cell input, verified exactly) and gathered back: maps are element-wise (SPEC
    §4.3), so every value is the one maps gives on the whole stack, bit for bit (tested). One difference: the cell
    counts in an out-of-domain reason's text ("[k of n points]") count distinct compositions, not cells."""

    def optical(self, film, bands):
        return self._film(maps.optical, film, bands)

    def surface(self, film, bands):
        return self._film(maps.surface, film, bands)

    def ground_optical(self, ground, bands):
        return self._ground(maps.ground_optical, ground, bands)

    def ground_surface(self, ground, bands):
        return self._ground(maps.ground_surface, ground, bands)

    def _film(self, fn, film, bands):
        q = np.asarray(film.q)
        S, H, W = q.shape
        g = _groups(q.reshape(S, H * W)) if not film.state else None
        if g is None:
            return fn(film, bands)
        rep, inv = g
        return _expand(fn(Film(film.substances, q.reshape(S, -1)[:, rep][:, None, :]), bands), inv, (H, W))

    def _ground(self, fn, ground, bands):
        held, z = np.asarray(ground.held), np.asarray(ground.z)
        H, W = z.shape
        cols = np.concatenate([held.reshape(-1, H * W), z.reshape(1, H * W)], 0)
        g = _groups(cols)
        if g is None:
            return fn(ground, bands)
        rep, inv = g
        cg = Ground(z.reshape(-1)[rep][None, :], ground.solid, ground.porosity, ground.pore_radius, ground.depth,
                    ground.substances, held.reshape(held.shape[0], -1)[:, rep][:, None, :])
        return _expand(fn(cg, bands), inv, (H, W))


OPTICS = Optics()


def _cell_keys(stack):
    """Two 64-bit keys per cell [2, H, W] of everything a cell's look depends on: the ground's z and held, and the
    amounts of each film present in the cell (h > 0), in order. A film absent from a cell is exactly the identity
    in goo.light (tested), so it does not enter the key. Equal keys are taken as equal inputs: a 128-bit hash, not
    verified (a collision would show a stale cell)."""
    H, W = stack.ground.z.shape
    k1 = np.full((H, W), np.uint64(1469598103934665603))
    k2 = np.full((H, W), np.uint64(0x9E3779B97F4A7C15))
    p1, p2 = np.uint64(1099511628211), np.uint64(6364136223846793005)

    def mix(rows, where=None):
        nonlocal k1, k2
        for r in rows:
            r = np.ascontiguousarray(r, np.float64).view(np.uint64)
            n1, n2 = (k1 ^ r) * p1, (k2 ^ r) * p2
            k1 = n1 if where is None else np.where(where, n1, k1)
            k2 = n2 if where is None else np.where(where, n2, k2)
    with np.errstate(over="ignore"):
        mix([stack.ground.z] + list(np.asarray(stack.ground.held)))
        for f in stack.films:
            q = np.asarray(f.q)
            present = q.sum(0) > 0
            if present.any():
                mix([np.full((H, W), 1.0)] + list(q), present)
    return np.stack([k1, k2])


@dataclass(frozen=True)
class Picture:
    """A picture of the painting: sRGB image (and gloss and infrared images if asked) at the asked scale, the box of
    the panel it shows, how many cells' optics were solved for it (the rest came from the cache), and the light
    receipt of that solve."""
    image: np.ndarray
    box: tuple
    scale: float
    exposure: float
    light: str
    solved: int
    cells: int
    receipt: object
    gloss_image: np.ndarray = None
    infrared_image: np.ndarray = None


def _solve_cells(stack, cells, bands, ill, gloss, infrared, n_ambient, t):
    """Linear sRGB [n, 3], gloss [n] and infrared [n, b] of the given flat cell indices, on a compacted stack."""
    H, W = stack.ground.z.shape
    g = stack.ground
    cz = np.asarray(g.z).reshape(-1)[cells][None, :]
    ch = np.asarray(g.held).reshape(len(g.substances), -1)[:, cells][:, None, :]
    ground = Ground(cz, g.solid, g.porosity, g.pore_radius, g.depth, g.substances, ch)
    films = [Film(f.substances, np.asarray(f.q).reshape(len(f.substances), -1)[:, cells][:, None, :])
             for f in stack.films]
    cs = Stack(ground, films, stack.dx)
    L = solve_look(cs, bands, R.diffuse_light(bands, ill), OPTICS, n_ambient=n_ambient, t=t)
    out = dict(rgb=R.colour(L, ill).rgb[0], receipt=L.receipt)
    if gloss:
        Lg = solve_look(cs, bands, R.gloss_light(bands, ill), OPTICS, n_ambient=n_ambient, t=t)
        out["gloss"] = R.gloss(Lg, ill).value[0]
    if infrared:
        ir = R.infrared(L).value[0]
        if ir.shape[-1] == 0:
            raise ValueError("look: the session's bands have none above 780 nm")
        out["ir"] = ir[:, -1]
    return out


def _down(a, f):
    """Area average over f x f blocks (edge blocks average what they hold), in the units given (linear light)."""
    if f == 1:
        return a
    H, W = a.shape[:2]
    Hb, Wb = -(-H // f), -(-W // f)
    pad = [(0, Hb * f - H), (0, Wb * f - W)] + [(0, 0)] * (a.ndim - 2)
    s = np.pad(a, pad).reshape((Hb, f, Wb, f) + a.shape[2:]).sum((1, 3))
    n = np.pad(np.ones((H, W)), pad[:2]).reshape(Hb, f, Wb, f).sum((1, 3))
    return s / n.reshape(n.shape + (1,) * (a.ndim - 2))


def scale_factor(scale):
    """(up, down): an integer scale >= 1 repeats cells; a fraction 1/k averages k x k blocks."""
    scale = float(scale)
    if scale >= 1:
        if scale != int(scale):
            raise ValueError("scale: an integer >= 1 or a fraction 1/k (0.5, 0.25, ...)")
        return int(scale), 1
    k = round(1 / scale)
    if k < 1 or abs(1 / scale - k) > 1e-9 * k:
        raise ValueError("scale: an integer >= 1 or a fraction 1/k (0.5, 0.25, ...)")
    return 1, k


def picture(world, bands, cache, light="daylight", exposure=1.0, region=None, scale=1, gloss=False, infrared=False,
            n_ambient=1.0):
    """A picture of the world (read only), reusing from `cache` (a dict, updated in place) the optics of every cell
    whose inputs have not changed since the cache was filled, under the same light and bands. Exact: a reused cell
    has the value a fresh solve gives (tested), up to the hash in _cell_keys."""
    if light not in C.ILLUMINANTS:
        raise ValueError(f"light: one of {', '.join(C.ILLUMINANTS)}")
    exposure = float(exposure)
    if not (np.isfinite(exposure) and exposure > 0):
        raise ValueError("exposure must be > 0")
    up, down = scale_factor(scale)
    ill = C.ILLUMINANTS[light]
    H, W = world.H, world.W
    box = (0, H, 0, W) if region in (None, "all") else _box(G.region(region, H, W, world.dx))
    stack = world.stack()
    keys = _cell_keys(stack)
    tag = (light, tuple(float(c) for c in bands.centres), tuple(float(w) for w in bands.widths), float(n_ambient))
    if cache.get("tag") != tag or cache.get("keys") is None or cache["keys"].shape != keys.shape:
        cache.clear()
        cache.update(tag=tag, keys=np.zeros_like(keys), known=np.zeros((H, W), bool), rgb=np.zeros((H, W, 3)),
                     gloss=None, ir=None)
    for name, want in (("gloss", gloss), ("ir", infrared)):
        if want and cache[name] is None:
            cache[name] = np.zeros((H, W))
            cache[f"{name}_known"] = np.zeros((H, W), bool)
    y0, y1, x0, x1 = box
    inbox = np.zeros((H, W), bool); inbox[y0:y1, x0:x1] = True
    same = np.all(cache["keys"] == keys, 0) & cache["known"]
    if gloss:
        same &= cache["gloss_known"]
    if infrared:
        same &= cache["ir_known"]
    todo = np.flatnonzero((inbox & ~same).ravel())
    receipt = None
    if todo.size:
        out = _solve_cells(stack, todo, bands, ill, gloss, infrared, n_ambient, world.t)
        receipt = out["receipt"]
        cache["rgb"].reshape(-1, 3)[todo] = out["rgb"]
        cache["known"].ravel()[todo] = True
        cache["keys"].reshape(2, -1)[:, todo] = keys.reshape(2, -1)[:, todo]
        for name in ("gloss", "ir"):
            if name in out:
                cache[name].ravel()[todo] = out[name]
                cache[f"{name}_known"].ravel()[todo] = True
        # cells solved without gloss or infrared lose those cached readings (their inputs may have changed)
        for name, want in (("gloss", gloss), ("ir", infrared)):
            if not want and cache[name] is not None:
                cache[f"{name}_known"].ravel()[todo] = False

    def img(a):
        a = _down(a[y0:y1, x0:x1], down)
        return np.repeat(np.repeat(a, up, 0), up, 1) if up > 1 else a
    rgb = img(cache["rgb"] * exposure)
    out = dict(image=C.srgb8(rgb), box=box, scale=float(scale), exposure=exposure, light=ill.name,
               solved=int(todo.size), cells=int(inbox.sum()), receipt=receipt)
    if gloss:
        out["gloss_image"] = np.round(np.clip(img(cache["gloss"]) / 100.0, 0, 1) * 255).astype(np.uint8)
    if infrared:
        out["infrared_image"] = np.round(np.clip(img(cache["ir"]), 0, 1) * 255).astype(np.uint8)
    return Picture(**out)


def save_cache(cache, path):
    arrays = {k: v for k, v in cache.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(str(path), tag=np.array(repr(cache.get("tag"))), **arrays)


def load_cache(path):
    import ast
    try:
        d = np.load(str(path))
    except (OSError, ValueError):
        return {}
    c = {k: d[k] for k in d.files if k != "tag"}
    c.setdefault("gloss", None); c.setdefault("ir", None)
    try:
        c["tag"] = ast.literal_eval(str(d["tag"]))
    except (ValueError, SyntaxError):
        return {}
    return c
