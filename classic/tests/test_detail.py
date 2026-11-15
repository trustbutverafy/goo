"""The detail painter (goo.classic.detail): the motif split on hand-made cases, a tiny painting end to end, and the
Painter's domain (classic/SPEC.md §1) enforced mechanically: painter code reads the canvas only through
`reflectance` and its sizes, and changes it only through tools, `fix`, `wait`, `step`."""
import sys
import numpy as np
import pytest
from scipy import ndimage
import goo.classic.brush, goo.classic.detail, goo.classic.painter, goo.classic.palette, goo.classic.tools
from goo.classic import Canvas
from goo.classic.detail import split, DetailPainter
from goo.classic.palette import Palette, lab, srgb_to_linear

BG, RED = np.array([0.2, 0.3, 0.6]), np.array([0.8, 0.1, 0.1])


def plain(n=20): return np.tile(BG, (n, n, 1))


def test_diagonal_line_is_one_chain():
    m = plain()
    for i in range(3, 16): m[i, i] = RED
    s = split(m, None, 1)
    assert len(s.chains) == 1 and len(s.chains[0].points) == 13 and s.chains[0].width == 1
    up = np.repeat(np.repeat(m, 4, 0), 4, 1)                      # corner-touching texels, 4 cells each
    s4 = split(up, None, 8)
    assert len(s4.chains) == 1 and s4.chains[0].width == 4


def test_width_decides_feature_or_mass():
    m = plain(); m[8:10, 2:18] = RED                             # a bar two cells wide
    assert split(m, None, 2).fine.sum() == 32 and split(m, None, 2).chains[0].width == 2
    assert split(m, None, 1).fine.sum() == 0 and split(m, None, 1).chains == []


def test_features_do_not_leak_into_the_broad_motif():
    m = plain(); m[5, 2:18] = RED; m[2:18, 12] = RED; m[14, 4] = RED
    ids = np.zeros((20, 20), int); ids[:, 10:] = 1; m[:, 10:] = 0.5 * BG; m[5, 2:18] = RED; m[2:18, 12] = RED
    s = split(m, ids, 2)
    assert not np.all(s.broad == RED, -1).any()
    assert np.array_equal(s.fine, np.all(m == RED, -1))
    assert np.all(s.broad[:, :10] == BG) and np.all(s.broad[:, 10:] == 0.5 * BG)   # each feature takes its own object's colour


def test_a_texture_keeps_its_commonest_colour_as_body():
    m = plain(8); yy, xx = np.mgrid[:8, :8]; m[(yy + xx) % 3 == 0] = RED      # red one in three
    s = split(m, None, 2)
    assert np.all(s.broad == BG) and np.array_equal(s.fine, (yy + xx) % 3 == 0)


# ---------------------------------------------------------------- the domain boundary, mechanically
PAINTER_FILES = {goo.classic.detail.__file__, goo.classic.painter.__file__, goo.classic.palette.__file__}
TOOL_FILES = {goo.classic.brush.__file__, goo.classic.tools.__file__}
PAINTER_MAY = {"reflectance", "fix", "wait", "step", "W", "H"}
# what Brush and Stylus use: the wet layer they exchange with (q, by species name: species, idx), the tool velocity
# fields they write (U, V), the books they keep when charged (led) and their own registration (brushes), the
# clock they advance (step), the panel's size, and for charcoal the surface it catches on (surface). The engine's
# own tools use the tool channel (_q, _U, _V, and _mark / _mark_tool to say where they wrote, _surface_window for
# charcoal: SPEC §9), which keeps the windowed books; the public names still work for a tool but cost a rescan.
TOOL_MAY = {"q", "U", "V", "surface", "_q", "_U", "_V", "_mark", "_mark_tool", "_surface_window",
            "species", "idx", "led", "brushes", "step", "W", "H", "_open"}


class DomainError(AssertionError): pass


class Watched:
    """A canvas behind glass: painter code may read only PAINTER_MAY, tool code only TOOL_MAY; nobody may set
    an attribute. Reads from anywhere else (this test, the engine itself) pass. Records who read what."""
    def __init__(self, cv):
        object.__setattr__(self, "_cv", cv); object.__setattr__(self, "_read", {"painter": set(), "tool": set()})

    def __getattribute__(self, name):
        f = sys._getframe(1).f_code.co_filename
        who = "painter" if f in PAINTER_FILES else "tool" if f in TOOL_FILES else None
        if who:
            object.__getattribute__(self, "_read")[who].add(name)
            if name not in (PAINTER_MAY if who == "painter" else TOOL_MAY):
                raise DomainError(f"{who} code ({f.rsplit('/', 1)[-1]}) read canvas.{name}")
        return getattr(object.__getattribute__(self, "_cv"), name)

    def __setattr__(self, name, value):
        raise DomainError(f"canvas.{name} set from outside the World")


TEX, F = 20, 4
COL = {k: np.array(v) for k, v in dict(light=(230, 222, 200), dark=(44, 44, 50), red=(196, 48, 40), pale=(150, 190, 220), navy=(36, 52, 96)).items()}


def tiny_motif():
    img = np.zeros((TEX, TEX, 3), np.uint8); ids = np.zeros((TEX, TEX), int)
    img[:, :10] = COL["light"]; img[:, 10:] = COL["dark"]; ids[:, 10:] = 1
    img[5, 2:18] = COL["red"]                                     # horizontal, across light and dark
    img[8:18, 14] = COL["pale"]                                   # vertical, on dark
    for i in range(7): img[9 + i, 2 + i] = COL["red"]             # diagonal, on light
    img[2, 4] = COL["navy"]                                       # a dot
    return img, ids


@pytest.fixture(scope="module")
def painted():
    img, ids = tiny_motif()
    up = lambda a: np.repeat(np.repeat(a, F, 0), F, 1)
    pal = ["white", "carbon", "ultramarine", "yellow", "red", "ochre"]
    cv = Canvas(TEX * F, TEX * F, pal, seed=2, laws=dict(lazy=90, cure_rate=1e-5, flow_sub=0.1, diff_sub=0.1))
    glass = Watched(cv)
    p = DetailPainter(glass, up(srgb_to_linear(img / 255.0)), up(ids), Palette(pal, steps=8), width=2 * F, plan={0: np.pi / 2, 1: np.pi / 2})
    p.paint()
    return cv, glass, p, img


def test_painter_stays_in_its_domain(painted):
    cv, glass, p, img = painted
    read = object.__getattribute__(glass, "_read")
    assert read["painter"] <= PAINTER_MAY and "reflectance" in read["painter"] and {"fix", "wait"} <= read["painter"]
    assert read["tool"] <= TOOL_MAY and {"_q", "_U", "_V", "led"} <= read["tool"]
    for attr in ("q", "kappa", "h", "store", "strata", "L"):     # the glass works: painter code is refused
        with pytest.raises(DomainError):
            eval(compile(f"cv.{attr}", goo.classic.detail.__file__, "eval"), {"cv": glass})
    with pytest.raises(DomainError):
        glass.kappa = None


def test_tiny_painting_looks_like_its_motif(painted):
    cv, glass, p, img = painted
    seen = cv.reflectance()[..., :3]
    texel = seen.reshape(TEX, F, TEX, F, 3).mean(axis=(1, 3))
    dE = np.linalg.norm(lab(texel) - lab(srgb_to_linear(img / 255.0)), axis=-1)
    assert dE.mean() < 6.0


def test_thin_lines_are_there_and_leave_no_halo(painted):
    cv, glass, p, img = painted
    L = lab(cv.reflectance()[..., :3]); want, around = lab(p.motif), lab(p.broad)
    for ch in p.chains:                                           # on its chain, a line looks like itself, not its ground
        iy, ix = ch.points[:, 1].astype(int), ch.points[:, 0].astype(int)
        to_line = np.linalg.norm(L[iy, ix] - want[iy, ix], axis=-1); to_ground = np.linalg.norm(L[iy, ix] - around[iy, ix], axis=-1)
        assert (to_line < to_ground).mean() > 0.85
    d = ndimage.distance_transform_edt(~p.fine)
    edge = ndimage.binary_dilation(ndimage.morphological_gradient(p.ids, size=3) > 0, iterations=4)   # where masses meet
    err = np.linalg.norm(L - around, axis=-1)
    ring, far = (d > 2) & (d <= 8) & ~edge, (d > 12) & ~edge
    assert err[ring].mean() < err[far].mean() + 1.5               # beyond two cells of a line, no line colour


def test_books_close(painted):
    cv = painted[0]
    led = cv.ledger()
    assert max(abs(v["residual"]) / max(1.0, v["loaded"]) for v in led["species"].values()) < 1e-9
    assert cv.capped == 0 and cv.q.min() >= 0
