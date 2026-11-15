"""goo.observe readings: sensors read a Look, never a world, and never modify it. Owner: lab and observers."""
import warnings
import numpy as np, pytest
from PIL import Image
from goo.units import um
from goo.film import Film, Ground, Stack, Bands
from goo.substance import maps, library as lib
from goo.light import look, fresnel
from goo import observe as O

BANDS = Bands.uniform(380e-9, 740e-9, 40e-9).with_band(1500e-9, 100e-9)
Z = np.zeros((2, 3))


def gesso():
    return Ground(Z, lib.CALCITE, 0.3, 1 * um, 300 * um)


def paint(pig=lib.HEMATITE, f=0.05, h=30 * um):
    return Film.uniform({lib.LINSEED: h * (1 - f), pig: h * f}, *Z.shape)


def see(films, light=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return look(Stack(gesso(), films, 1e-3), BANDS, light or O.diffuse_light(BANDS, O.DAYLIGHT), maps)


def frozen(L):
    for k, v in vars(L).items():
        if isinstance(v, np.ndarray):
            v.setflags(write=False)
    return L


def test_sensors_take_a_look_and_refuse_a_world_a_stack_or_a_film():
    st = Stack(gesso(), [paint()], 1e-3)
    for thing in (st, st.films[0], st.ground, np.zeros((2, 3, BANDS.B)), {"reflectance": 1}):
        for sensor in (lambda x: O.colour_of(x, O.DAYLIGHT), lambda x: O.gloss(x, O.DAYLIGHT), O.infrared,
                       lambda x: O.contrast_ratio(x, x, O.DAYLIGHT)):
            with pytest.raises(TypeError):
                sensor(thing)


def test_sensors_never_modify_what_they_read():
    L = frozen(see([paint()]))
    before = {k: np.array(v, copy=True) for k, v in vars(L).items() if isinstance(v, np.ndarray)}
    digest = L.receipt.digest()
    c = O.colour_of(L, O.A); O.gloss(L, O.DAYLIGHT); O.infrared(L); O.contrast_ratio(L, L, O.DAYLIGHT)
    O.rgb8(L, O.DAYLIGHT)
    for k, v in before.items():
        assert np.array_equal(getattr(L, k), v), k
    assert L.receipt.digest() == digest
    c.XYZ[...] = 0                                      # a reading's arrays are its own
    assert not np.all(O.colour_of(L, O.A).XYZ == 0)


def test_a_reading_carries_the_look_flags_and_is_at_best_assumed_out_of_domain():
    L = see([paint(f=0.3)])                             # above the independent-scattering bound
    c = O.colour_of(L, O.DAYLIGHT, matter="computed")
    assert any(f.kind == "out_of_domain" for f in c.flags)
    assert c.evidence == "assumed"
    assert O.look_evidence(L, "guessed") == "guessed"


def test_specular_excluded_colour_reads_the_diffuse_part():
    L = see([paint()])
    sce, sci = O.colour_of(L, O.DAYLIGHT, geometry="sce"), O.colour_of(L, O.DAYLIGHT, geometry="sci")
    assert np.all(sci.XYZ[..., 1] > sce.XYZ[..., 1])
    W = O.weights(BANDS, O.DAYLIGHT)
    assert np.allclose(sce.XYZ, L.diffuse @ W.T, rtol=1e-14)


def test_gloss_of_a_smooth_film_is_its_fresnel_reflectance_against_the_black_glass():
    clear = Film.uniform({lib.PMMA: 20 * um}, *Z.shape)
    L = see([clear], O.gloss_light(BANDS, O.DAYLIGHT))
    g = O.gloss(L, O.DAYLIGHT).value
    W = O.weights(BANDS, O.DAYLIGHT)[1]
    n_top = maps.surface(clear, BANDS).index.value[:, 0, 0]
    expect = 100 * (fresnel(n_top, np.cos(O.GLOSS_ANGLE)) @ W) / fresnel(1.567, np.cos(O.GLOSS_ANGLE))
    assert np.allclose(g, expect, rtol=1e-12)
    assert 80 < g.mean() < 100
    bare = see([], O.gloss_light(BANDS, O.DAYLIGHT))
    assert O.gloss(bare, O.DAYLIGHT).value.max() < 1.0           # a chalk ground is matte


def test_contrast_ratio_falls_with_less_paint_and_is_one_for_a_thick_film():
    black = Film.uniform({lib.PMMA: 19 * um, lib.SOOT: 1 * um}, *Z.shape)
    crs = []
    for h in (2 * um, 20 * um, 400 * um):
        p = Film.uniform({lib.LINSEED: h * 0.9, lib.RUTILE: h * 0.1}, *Z.shape)
        crs.append(float(O.contrast_ratio(see([black, p]), see([p]), O.DAYLIGHT).value.mean()))
    assert crs[0] < crs[1] < crs[2] and crs[2] > 0.99


def test_infrared_reads_only_bands_beyond_the_visible():
    r = O.infrared(see([paint()]))
    assert r.value.shape == (2, 3, 1) and "1500 nm" in r.note


def test_png_has_one_pixel_block_per_cell(tmp_path):
    L = see([paint()])
    p = O.write_png(L, O.DAYLIGHT, tmp_path / "x.png", scale=4)
    a = np.asarray(Image.open(p))
    assert a.shape == (8, 12, 3)
    assert np.array_equal(a[::4, ::4], O.rgb8(L, O.DAYLIGHT))
    q = O.write_ir_png(L, tmp_path / "ir.png", scale=2)
    assert np.asarray(Image.open(q)).shape == (4, 6)
