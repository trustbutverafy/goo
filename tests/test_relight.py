"""The visible-light sensor's receipts: exact zero, the white furnace, linearity, energy, shadows, the texel aperture."""
import numpy as np
import pytest
from goo.surface import Surface
from goo.sensors import relight as R
from goo.sensors.lidar import CELL

DIRS = [R.light_dir(a, e) for a, e in [(0, 12), (90, 30), (200, 60), (315, 8), (45, 85), (130, 20)]] + [np.array([0, 0, 1.0])]


def bumpy(seed=0, H=48, W=64):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W]
    Z = 0.4 * np.exp(-((x - 20) ** 2 + (y - 24) ** 2) / 40.0) + 0.02 * rng.standard_normal((H, W))
    refl = rng.uniform(0.05, 0.95, (H, W, 4))
    return Surface(Z, reflectance=refl, gloss=rng.uniform(0, 1, (H, W)))


def flat(rgb=1.0, gloss=0.0, n=6):
    return Surface(np.zeros((n, n)), reflectance=np.full((n, n, 4), rgb), gloss=np.full((n, n), gloss))


def test_zero_light_zero_ambient_is_exactly_zero():
    for gain in (1.0, 3.0):
        p = R.prepare(bumpy(), relief_gain=gain)
        for L in DIRS:
            assert np.all(p.render(L, intensity=0.0, ambient=0.0) == 0)


def test_white_furnace():
    p = R.prepare(flat())
    for L in DIRS:
        Ln = L / np.linalg.norm(L)
        for I, a in ((1.0, 0.1), (2.5, 0.0), (0.3, 0.7)):
            assert np.abs(p.render(L, intensity=I, ambient=a) - (I * Ln[2] + a)).max() < 1e-14


def test_light_term_is_linear_in_intensity():
    p = R.prepare(bumpy(1), relief_gain=2.0)
    for L in DIRS:
        assert np.array_equal(p.render(L, 2.0, ambient=0.0), 2 * p.render(L, 1.0, ambient=0.0))
        (_, a), (_, b) = p.render(L, 2.0, 0.1, return_parts=True), p.render(L, 1.0, 0.1, return_parts=True)
        assert np.array_equal(a["direct"], b["direct"]) and np.array_equal(a["sky"], b["sky"])   # the parts do not depend on it


@pytest.mark.parametrize("gloss", [0.0, 0.3, 1.0])
def test_energy_through_the_sensor(gloss):
    # directional albedo of a flat white cell seen from above = (1/pi) * integral of its radiance over light directions
    p = R.prepare(flat(gloss=gloss, n=2))
    th = (np.arange(3000) + 0.5) * (np.pi / 2) / 3000
    out = np.array([p.render(R.light_dir(0, 90 - np.degrees(t)), 1.0, 0.0)[0, 0, 0] for t in th])
    A = 2 * np.sum(out * np.sin(th)) * (np.pi / 2) / 3000
    assert abs(A - 1) < R.ENERGY_TOL if gloss == 0 else 1 - 0.01 < A <= 1 + R.ENERGY_TOL
    peak = 1 - gloss * R.F0 + gloss * R.F0 * (R.SHININESS + 2) / 2          # at the mirror direction (overhead light)
    assert out.max() <= peak + 1e-9 and p.render([0, 0, 1], 1.0, 0.0)[0, 0, 0] == pytest.approx(peak)
    # and the sky term is the same albedo: a white cell under a white sky of 1 returns A
    assert p.render([0, 0, 1], 0.0, 1.0)[0, 0, 0] == pytest.approx(A, abs=2 * R.ENERGY_TOL)


@pytest.mark.parametrize("tilt", [0, 10, 25, 45, 60])
def test_energy_on_tilted_facets(tilt):
    # the lobe itself on a facet tilted by `tilt`, all light directions above both facet and panel
    b = np.radians(tilt); n = np.array([np.sin(b), 0, np.cos(b)]); t1 = np.array([np.cos(b), 0, -np.sin(b)]); t2 = np.array([0, 1.0, 0])
    nth, nph = 1200, 480
    th = (np.arange(nth) + 0.5) * (np.pi / 2) / nth; ph = (np.arange(nph) + 0.5) * 2 * np.pi / nph
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    Ls = (np.cos(TH)[..., None] * n + np.sin(TH)[..., None] * (np.cos(PH)[..., None] * t1 + np.sin(PH)[..., None] * t2)).reshape(-1, 3)
    dw = (np.sin(TH) * (np.pi / 2 / nth) * (2 * np.pi / nph)).ravel() * (Ls[:, 2] > 0)
    for g in (0.0, 1.0):
        A = (R.reflected(n[None], Ls, np.ones((len(Ls), 1)), np.full(len(Ls), g))[:, 0] * dw).sum() / np.pi
        assert A <= 1 + R.ENERGY_TOL


@pytest.mark.parametrize("az,el", [(0, 20), (0, 40), (90, 30)])
def test_a_block_casts_its_shadow_away_from_the_light(az, el):
    H, W, hb = 60, 60, 1.0                                                   # a 1 mm mesa, 10 x 10 cells
    Z = np.zeros((H, W)); Z[25:35, 25:35] = hb
    p = R.prepare(Surface(Z, reflectance=np.full((H, W, 4), 0.8)))
    V = p.visibility(R.light_dir(az, el))
    expect = hb / np.tan(np.radians(el)) / CELL                             # cells
    if az == 0:   # light on the left: shadow to the right of the block, none to its left
        lane, before = V[30, 35:], V[30, :25]
    else:         # light at the top: shadow below
        lane, before = V[35:, 30], V[:25, 30]
    length = int(np.argmax(lane > 0.5))
    assert abs(length - expect) <= 2, (length, expect)
    assert np.all(before == 1) and np.all(lane[int(expect) + 2:] == 1)
    assert np.all(V[25:35, 25:35] == 1)                                     # the top of the block is lit


def test_texels_conserve_the_mean():
    img = np.random.default_rng(2).uniform(0, 3, (12, 18, 3))
    t = R.texels(img, 3)
    assert t.shape == (4, 6, 3) and abs(t.mean() - img.mean()) < 1e-14
    assert np.allclose(t[1, 2], img[3:6, 6:9].mean((0, 1)))
    with pytest.raises(ValueError): R.texels(img, 5)


def test_refuses_what_it_cannot_see():
    with pytest.raises(ValueError): R.prepare(Surface(np.zeros((4, 4)), np.full((4, 4), 0.9), np.ones((4, 4))))   # nir only
    with pytest.raises(ValueError): R.prepare(flat()).render([0.3, 0, -0.1])                                       # light below
