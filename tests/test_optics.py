"""Optics receipts (SPEC §6): the contract, `classic` bit for bit, and v2's invariants, one test per law.

A failing test's name says which law broke. Tolerances, where a law cannot hold exactly, are stated beside it."""
import copy, inspect, pickle
import numpy as np
import pytest
from goo.engine import Canvas, GROUND_RGBN, PIGMENTS
from goo.brush import Brush
from goo.palette import Palette, lab
from goo.surface import Surface
from goo import optics as O

PIGS = ["white", "ultramarine", "yellow", "carbon", "sienna"]
V2 = O.V2Optics()


def vec(species, **kw):
    """A (species, 1) film from volumes by name."""
    v = np.zeros((len(species), 1))
    for k, x in kw.items(): v[species.index(k), 0] = x
    return v


def painted(optics, ground="gesso"):
    """A canvas with two strata and a wet oil layer, all through Canvas.add / fix (no time passes)."""
    cv = Canvas(36, 24, PIGS, seed=4, ground=ground, optics=optics, laws=dict(evaporation=0, absorption=0))
    yy, xx = np.mgrid[0:24, 0:36]
    cv.add("ultramarine", 0.03 * (xx < 20)); cv.add("water", 0.0 * (xx < 20)); cv.fix()          # a dry lean layer
    band = (np.abs(yy - 12) < 6).astype(float)
    cv.add("sienna", 0.02 * band); cv.add("oil", 0.001 * band); cv.fix()                     # lean, 5% oil
    for k, v in Brush.paint("oil", 0.5, 0.3, {"white": 1, "yellow": 1}).items(): cv.add(k, v * (xx > 12))
    return cv


def dE(a, b): return np.linalg.norm(lab(np.clip(a, 0, 1)) - lab(np.clip(b, 0, 1)), axis=-1)


# ------------------------------------------------------------------------------------------------ the contract

def test_contract_classic_is_the_default_and_bit_identical():
    cv = painted("classic")
    assert Canvas(4, 4, ["white"]).optics.name == "classic" and cv.optics.bands.n == 4
    Rb = np.broadcast_to(GROUND_RGBN, (cv.H, cv.W, 4)).copy()                 # the legacy composition, by hand
    for vol, wet in [(d, False) for d in cv.strata] + [(cv._q, True)]:
        R, T = cv.km_layer(*cv._KS(vol * 10.0, wet)); Rb = R + T * T * Rb / (1 - R * Rb)
    assert np.array_equal(np.clip(Rb, 0, 1), cv.reflectance())
    assert np.array_equal(cv.reflectance(light="D65"), cv.reflectance()[..., :3])   # classic bands are sRGB
    assert np.array_equal(Surface.from_canvas(cv).rgb, cv.reflectance()[..., :3])


def test_contract_models_are_pure_and_picklable():
    for opt in ("classic", "v2"):
        cv = painted(opt); cv.step()                                       # a pending slow-law tick
        before = pickle.dumps(cv.__dict__)
        R, g = cv.reflectance(), cv.finish(); Surface.from_canvas(cv); cv.reflectance(light="A")
        assert pickle.dumps(cv.__dict__) == before                         # nothing read changed anything
        cv2 = pickle.loads(pickle.dumps(cv))
        assert cv2.optics.name == opt and np.array_equal(cv2.reflectance(), R) and np.array_equal(cv2.finish(), g)
    with pytest.raises(ValueError): Canvas(4, 4, ["white"], optics="v3")


def test_contract_layers_peel_back_in_v2():
    cv = painted("v2")
    R0, R1, R2, R3 = (cv.reflectance(layers=k) for k in (0, 1, 2, 3))
    ground = V2.ground_body("gesso", np.zeros((cv.H, cv.W)))[0]
    assert np.allclose(R0, np.broadcast_to(_ground_seen("gesso"), R0.shape), atol=1e-12)
    assert np.array_equal(R3, cv.reflectance()) and not np.allclose(R1, R2)


def _ground_seen(g):
    b = O.SPECTRAL15
    return np.r_[O._band_means(O.fit_reflectance(np.asarray(O.GROUNDS[g]["R"][:3])), b), O.GROUNDS[g]["R"][3]]


# ------------------------------------------------------------------------------------------------ energy

def test_energy_every_layer_and_band_R_plus_T_plus_absorbed_is_one():
    rng = np.random.default_rng(0); sp = ["water", "oil"] + PIGS + ["acrylic", "gum"]
    vol = rng.uniform(0, 1, (len(sp), 4000)) ** 3 * rng.uniform(0, 0.6, (1, 4000))
    for stratum in (False, True):
        K, S, n, sm, cov = V2.film(sp, vol, rng.uniform(0, 1, 4000), stratum)
        R, T = O.km_layer(K, S); A = 1 - R - T
        assert (R >= 0).all() and (T >= 0).all() and (A >= -1e-12).all() and (K >= 0).all() and (S >= 0).all()
    for n in np.linspace(1, 2.8, 50):                                      # the surface as a layer
        for s in (0, 0.5, 1):
            Rf, Rb, Tf, Tb = O.interface(n, s)
            assert 0 <= Rf and Rf + Tf <= 1 + 1e-15 and Rb + Tb == pytest.approx(1)
    R = V2.preview(sp, vol, ground="paper", store=0.05)
    assert R.min() >= 0 and R.max() <= 1


# ------------------------------------------------------------------------------------------------ exact limits

@pytest.mark.parametrize("ground", ["gesso", "paper"])
def test_limit_no_paint_shows_the_ground(ground):
    cv = Canvas(8, 6, PIGS, ground=ground, optics="v2")
    assert np.abs(cv.reflectance() - _ground_seen(ground)).max() < 1e-12     # the table's ground, as seen
    assert np.abs(cv.reflectance()[..., :15].reshape(6, 8, 3, 5).mean(-1)[..., ::-1] - GROUND_RGBN[:3]).max() < 1e-12
    # classic composes its (empty) wet layer with an S floor of 1e-7: the ground to 1e-8, not exactly
    assert np.abs(Canvas(8, 6, PIGS, ground=ground).reflectance() - GROUND_RGBN).max() < 1e-8
    assert cv.finish().max() == 0


def test_limit_a_thick_film_shows_its_own_R_infinity_and_hides_the_ground():
    sp = ["water", "oil", "white", "ultramarine"]
    film = vec(sp, oil=60.0, white=20.0, ultramarine=20.0)                     # optically thick in every band
    on_black = V2.preview(sp, film, under=np.zeros(16)); on_white = V2.preview(sp, film, under=np.ones(16))
    assert np.abs(on_black - on_white).max() < 1e-12
    K, S, n, sm, _ = V2.film(sp, film, np.zeros(1), False)
    Rf, Ri, Tf, Ti = O.interface(n, sm)
    Rinf = O.km_rinf(K / S)
    assert np.abs(on_black - (Rf + Tf * Ti * Rinf / (1 - Ri * Rinf))).max() < 1e-12


def test_limit_a_surrounding_of_the_pigments_index_makes_it_purely_absorbing():
    t = copy.deepcopy(O.SPECIES); t["oil"]["n"] = t["ultramarine"]["n"]; t["oil"]["reference"] = False
    t["ref"] = dict(role="binder", n=1.48, K=[0, 0, 0, 0], S=[0, 0, 0, 0], reference=True)
    m = O.V2Optics(table=t); sp = ["water", "oil", "ultramarine"]
    K, S, *_ = m.film(sp, vec(sp, oil=0.3, ultramarine=0.1), np.zeros(1), False)
    assert np.abs(S).max() < 1e-12 and K.min() > 0
    R, T = O.km_layer(K, S)
    assert np.abs(R).max() < 1e-6 and np.allclose(T, np.exp(-K), rtol=1e-6)   # S floor 1e-7 of km_layer


def test_limit_v2_with_equal_indices_and_no_surface_is_classic():
    """Tolerance 1e-6: classic composes every layer over the whole panel, and its S floor (1e-7) makes an empty
    layer reflect ~1e-7 (1 - Rb)^2; v2 skips empty cells exactly. Water's 1.5 um absorption is set to 0 and the
    wet layer is uncured (v2's yellowing saturates at KAPPA_FULL; classic's does not)."""
    t = copy.deepcopy(O.SPECIES); n0 = t["oil"]["n"]
    for k, e in t.items():
        if e["role"] != "pigment": e["n"] = n0
    t["water"]["K"] = [0, 0, 0, 0]
    gr = copy.deepcopy(O.GROUNDS)
    for g in gr.values(): g["n_bound"] = n0
    m = O.V2Optics(bands="classic", table=t, grounds=gr, surface=False)
    for ground in ("gesso", "paper"):
        cv = painted("classic", ground)
        cv.add("water", 0.2); cv.L["absorption"] = 1.0; cv.step(); cv.flush()   # water in the wet layer and the ground
        assert cv.store.max() > 0
        assert np.abs(m.reflectance(cv) - cv.reflectance()).max() < 1e-6


# ------------------------------------------------------------------------------------------------ composition

def test_composition_is_associative():
    rng = np.random.default_rng(3)
    lay = lambda: tuple(rng.uniform(0, 0.5, 200) for _ in range(4))
    a, b, c, d = lay(), lay(), lay(), lay()
    left = O.over(O.over(O.over(a, b), c), d); right = O.over(a, O.over(b, O.over(c, d))); mid = O.over(O.over(a, b), O.over(c, d))
    for x, y, z in zip(left, right, mid): assert np.abs(x - y).max() < 1e-14 and np.abs(x - z).max() < 1e-14
    # and the canvas's bottom-up rule is the same algebra with symmetric KM layers
    K, S = rng.uniform(0, 3, (2, 200)); R, T = O.km_layer(K, S); Rg = rng.uniform(0, 1, 200)
    assert np.allclose(O.over((R, R, T, T), (Rg, Rg, 0 * Rg, 0 * Rg))[0], R + T * T * Rg / (1 - R * Rg), atol=1e-15)


# ------------------------------------------------------------------------------------------------ monotone

def test_monotone_adding_an_absorber_never_brightens_a_band():
    sp = ["water", "oil"] + PIGS
    rng = np.random.default_rng(5); vol = rng.uniform(0, 0.3, (len(sp), 500)) * (rng.uniform(0, 1, (len(sp), 500)) > 0.4)
    R0 = V2.preview(sp, vol, ground="paper")
    for pig in PIGS:                                          # more K, the same structure (indices, volumes)
        t = copy.deepcopy(O.SPECIES); t[pig] = dict(t[pig], K=[2 * k + 0.1 for k in PIGMENTS[pig]["K"]], S=PIGMENTS[pig]["S"])
        R1 = O.V2Optics(table=t).preview(sp, vol, ground="paper")
        assert (R1 <= R0 + 1e-15).all(), pig


def test_monotone_adding_a_pure_scatterer_over_black_never_darkens():
    t = copy.deepcopy(O.SPECIES); t["white"] = dict(t["white"], K=[0, 0, 0, 0], S=PIGMENTS["white"]["S"])
    m = O.V2Optics(table=t); sp = ["water", "oil", "white"]
    for oil in (0.0, 0.01, 0.05, 0.3):                         # lean to fat: crossing the CPVC on the way
        w = np.linspace(0, 0.6, 400)
        R = m.preview(sp, np.stack([0 * w, oil + 0 * w, w]), under=np.zeros(16))
        assert (np.diff(R, axis=0) >= -1e-12).all(), oil


# ------------------------------------------------------------------------------------------------ continuity

def test_continuity_fix_does_not_change_what_a_cell_looks_like():
    """Cured oil (kappa >= KAPPA_FULL) and a dried wash: identical to 1e-12 across fix(). The wash leaves at
    most 1e-6 of water, which fix() books as evaporated: tolerance 1e-5 for it."""
    cv = Canvas(30, 20, PIGS, optics="v2", laws=dict(evaporation=0, absorption=0))
    for k, v in Brush.paint("oil", 0.5, 0.4, {"white": 1, "ultramarine": 1}).items(): cv.add(k, v)
    cv.kappa[:] = 0.9
    before, g0 = cv.reflectance(), cv.finish(); assert cv.fix() == 1.0 and len(cv.strata) == 1
    assert np.abs(cv.reflectance() - before).max() < 1e-12 and np.abs(cv.finish() - g0).max() < 1e-12
    cv = Canvas(30, 20, PIGS, optics="v2", ground="paper")
    cv.add("water", 0.3); cv.add("ultramarine", 0.03)
    while cv.q[0].max() > 1e-6: cv.wait(50, evap_boost=8.0)
    before = cv.reflectance(); assert cv.fix() == 1.0
    assert np.abs(cv.reflectance() - before).max() < 1e-5


@pytest.mark.parametrize("medium", ["wash", "gum wash", "acrylic", "oil"])
def test_continuity_a_film_drying_changes_colour_continuously(medium):
    sp = ["water", "oil", "ultramarine", "gum", "acrylic"]
    base = dict(wash=dict(ultramarine=0.04), **{"gum wash": dict(ultramarine=0.03, gum=0.03)},
                acrylic=dict(acrylic=0.06, ultramarine=0.02), oil=dict(oil=0.1, ultramarine=0.05))[medium]
    def max_jump(n, store):
        w = np.linspace(0.4, 0, n)
        film = np.zeros((len(sp), n)); film[0] = w
        for k, v in base.items(): film[sp.index(k)] = v
        c = V2.colour(V2.preview(sp, film, ground="paper", store=store))
        return dE(c[1:], c[:-1]).max()
    for store in (0.08, 0.0):
        coarse, fine = max_jump(4001, store), max_jump(40001, store)
        # continuous: ten times finer steps give (about) ten times smaller jumps; a step in colour would not
        assert fine < 0.2 * coarse + 1e-3 and coarse < 1.0, (medium, store, coarse, fine)


# ------------------------------------------------------------------------------------------------ one code path

MEDIA = {
    "oil":        (dict(oil=0.15, white=0.05, ultramarine=0.08), True),
    "watercolour": (dict(water=0.3, ultramarine=0.03), False),
    "gum wash":   (dict(water=0.3, ultramarine=0.03, gum=0.02), False),
    "acrylic":    (dict(water=0.08, acrylic=0.06, yellow=0.04), False),
}


@pytest.mark.parametrize("medium", list(MEDIA))
@pytest.mark.parametrize("ground", ["gesso", "paper"])
def test_one_code_path_palette_preview_equals_the_canvas(medium, ground):
    film, oily = MEDIA[medium]
    sp_extra = [k for k in film if k not in ("water", "oil")]
    pigs = [p for p in sp_extra if p in PIGMENTS] + [p for p in sp_extra if p not in PIGMENTS]
    for opt in (["classic", "v2"] if all(p in PIGMENTS for p in sp_extra) else ["v2"]):
        cv = Canvas(10, 8, pigs, ground=ground, optics=opt, laws=dict(evaporation=0, absorption=0))
        for k, v in film.items(): cv.add(k, v)
        pal = Palette(pigs, steps=2, optics=cv.optics)
        wet = cv.reflectance()[4, 5]
        assert np.abs(pal.preview(film, ground=ground) - wet).max() < 1e-12, (opt, "wet")
        dry_film = {k: v for k, v in film.items() if k != "water"}
        cv = Canvas(10, 8, pigs, ground=ground, optics=opt, laws=dict(evaporation=0, absorption=0))
        for k, v in dry_film.items(): cv.add(k, v)
        if oily: cv.kappa[:] = 0.9
        assert cv.fix() == 1.0
        # classic composes an empty wet layer over the stratum (its S floor, 1e-7): 1e-7 there; v2 is exact
        tol = 1e-7 if opt == "classic" else 1e-12
        assert np.abs(pal.preview(dry_film, ground=ground, cured=True) - cv.reflectance()[4, 5]).max() < tol, (opt, "dry")


def test_one_code_path_the_painter_refuses_a_palette_of_another_model():
    from goo.painter import Painter
    cv = Canvas(20, 20, ["white", "carbon"], optics="v2", laws=dict(cure_rate=1e-5))
    p = Painter(cv, np.full((20, 20, 3), 0.5), np.zeros((20, 20), int), Palette(["white", "carbon"], steps=2))
    with pytest.raises(ValueError, match="optics"): p.see()
    with pytest.raises(ValueError, match="bands"):
        Palette(["white", "carbon"], steps=2, optics=cv.optics).match([0.5] * 3, [0.5] * 3, 0.2)


# ------------------------------------------------------------------------------------------------ sensors

def test_sensors_see_a_surface_that_carries_the_model_bands():
    from goo.sensors import relight
    from goo.sensors.lidar import Scanner
    cv = painted("v2"); s = Surface.from_canvas(cv)
    assert s.reflectance.shape == (cv.H, cv.W, 16) and s.bands is O.SPECTRAL15
    assert np.array_equal(s.nir, cv.reflectance()[..., 15])
    assert np.allclose(s.rgb, cv.reflectance(light="D65")) and np.allclose(s.colour("A"), cv.reflectance(light="A"))
    assert np.allclose(s.under("A").rgb, s.colour("A"))
    img = relight.render(s, relight.light_dir(0, 40)); assert img.shape == (cv.H, cv.W, 3) and np.isfinite(img).all()
    assert np.array_equal(s.gloss, cv.finish())


# ------------------------------------------------------------------------------------------------ bands, light

def test_bands_spectra_reduce_to_the_classic_table():
    """Each species' smooth K, S spectrum has, over the classic boxes, exactly the table's R, G, B (1e-12)."""
    for name, e in list(PIGMENTS.items()) + [(k, v) for k, v in O.SPECIES.items() if "K" in v]:
        for key in ("K", "S"):
            v = O._band_values(dict(e, role="pigment"), key, O.SPECTRAL15)
            boxes = v[:15].reshape(3, 5).mean(1)[::-1]
            assert np.allclose(boxes, np.maximum(e[key][:3], 0) if max(e[key][:3]) > 0 else 0, rtol=1e-12, atol=1e-12), (name, key)
            assert v[15] == e[key][3]


def test_light_white_is_white_in_daylight_and_warm_in_tungsten():
    assert np.allclose(O.white("D65"), 1, atol=1e-12)
    w = O.white("A"); assert w[0] > 1.2 and w[2] < 0.5
    assert np.allclose(O.white("A", adapt=True), 1, atol=1e-9)
    assert np.allclose(O.colour(np.full(4, 0.5), O.CLASSIC_BANDS, "A"), 0.5 * w)          # classic: the camera-RGB approximation


def test_bands_blue_and_yellow_make_green():
    """50/50 ultramarine and yellow in oil: green (CIELAB hue 120-200 degrees) in v2's 15 bands; classic's three
    bands make an olive (hue ~ 90) or worse."""
    sp = ["water", "oil", "ultramarine", "yellow"]
    film = vec(sp, oil=0.3, ultramarine=0.15, yellow=0.15)
    hue = lambda rgb: np.degrees(np.arctan2(lab(np.clip(rgb, 0, 1))[..., 2], lab(np.clip(rgb, 0, 1))[..., 1])) % 360
    h2 = hue(V2.colour(V2.preview(sp, film))[0])
    hc = hue(O._CLASSIC.preview(sp, film, np.broadcast_to(GROUND_RGBN, (1, 4)))[0, :3])
    assert 120 < h2 < 200 and h2 > hc + 20, (h2, hc)


# ------------------------------------------------------------------------------------------------ the surface

def test_surface_finish_is_the_smooth_fraction_that_reflectance_leaves_out():
    sp = ["water", "oil", "carbon"]
    cases = dict(fat=vec(sp, oil=0.3, carbon=0.1), satin=vec(sp, oil=0.0565, carbon=0.1), lean=vec(sp, oil=0.005, carbon=0.1))
    Rs = {}
    for k, f in cases.items():
        K, S, n, sm, cov = V2.film(sp, f, np.zeros(1), True)
        Rs[k] = V2.colour(V2.preview(sp, f, wet=False))[0].mean(); F0 = O.fresnel_normal(n)
        body = V2._seen(V2.preview(sp, f, wet=False) * 0 + O.km_rinf(K / S), n, sm)  # same body, as seen
        rough = V2._seen(O.km_rinf(K / S), n, 0 * sm)
        assert np.allclose(rough - body, sm[..., None] * F0[..., None], atol=1e-15)    # the excluded specular
    assert Rs["fat"] < 0.012 and Rs["satin"] > 2.5 * Rs["fat"]                          # matte darks have a floor
    cv = Canvas(12, 8, ["ultramarine"], optics="v2"); cv.add("water", 0.3); cv.add("ultramarine", 0.03)
    assert cv.finish().min() == 1.0                                                     # wet paint is glossy
    cv.q[0] = 0; assert cv.finish().max() == 0.0                                        # a dried wash is matte


# ------------------------------------------------------------------------------------------------ emergence

def test_emergence_by_composition_alone():
    """No medium in the code: a made-up table (a solvent, a resin, a latex in the solvent, a dust, a felt ground)
    shows all four effects from composition and indices alone; and the v2 code names no species or ground."""
    src = inspect.getsource(O.V2Optics)
    for word in ("water", "oil", "acrylic", "gum", "paper", "gesso", "watercolour", "ultramarine"):
        assert f'"{word}"' not in src and f"'{word}'" not in src, word
    t = {"air": dict(role="void", n=1.0), "solvent": dict(role="volatile", n=1.36, K=[0, 0, 0, 0], S=[0, 0, 0, 0]),
         "resin": dict(role="binder", n=1.52, K=[0, 0, 0, 0], S=[0, 0, 0, 0], reference=True),
         "latex": dict(role="binder", dispersed_in="solvent", n=1.50, K=[0, 0, 0, 0], S=[0.004, 0.004, 0.004, 0.001]),
         "dust": dict(role="pigment", n=1.62, K=[1.5, 1.0, 0.2, 0.05], S=[0.08, 0.08, 0.2, 0.05])}
    gr = {"felt": dict(R=[0.9, 0.9, 0.9, 0.9], n_solid=1.55, bound=0.0, n_bound=1.5)}
    m = O.V2Optics(table=t, grounds=gr); sp = ["solvent", "resin", "dust", "latex"]
    L = lambda f, store=0.0: lab(np.clip(m.colour(m.preview(sp, f, store=store))[0], 0, 1))[0]
    f = lambda **kw: vec(sp, **kw)
    assert L(f(solvent=0.3, dust=0.04), 0.08) < L(f(dust=0.04)) - 5                     # deeper wet, paler dry
    assert L(f(), 0.08) < L(f()) - 5                                                    # wet ground darker
    assert L(f(solvent=0.05, latex=0.06, dust=0.02)) > L(f(latex=0.06, dust=0.02)) + 3  # an emulsion is milky wet
    lean, bound = f(resin=0.02, dust=0.1), f(resin=0.06, dust=0.1)                     # PVC 0.83 vs 0.63 (CPVC 0.64)
    assert L(lean) > L(bound) + 5                                                       # chalky above the CPVC
    assert m.film(sp, lean, np.zeros(1), True)[3][0] == 0                               # and matte


def test_one_code_path_the_looking_painter_works_under_v2():
    """The looking painter inverts v2 through the canvas's own model (a palette built with cv.optics) and gets
    its small motif about as close as under classic (mean CIELAB error, 48 x 48 cells, two passes)."""
    from goo.painter import Painter
    from goo.palette import srgb_to_linear
    H = W = 48; yy, xx = np.mgrid[0:H, 0:W]
    motif = np.zeros((H, W, 3)); ids = np.zeros((H, W), int); disc = (xx - 24) ** 2 + (yy - 24) ** 2 < 144
    motif[:] = srgb_to_linear(np.array([200, 190, 160]) / 255); motif[disc] = srgb_to_linear(np.array([60, 80, 150]) / 255); ids[disc] = 1
    pal = ["white", "ultramarine", "ochre", "carbon"]; err = {}
    for opt in ("classic", "v2"):
        cv = Canvas(W, H, pal, laws=dict(lazy=30, cure_rate=1e-5, flow_sub=0.1, diff_sub=0.1), optics=opt)
        p = Painter(cv, motif, ids, Palette(pal, steps=6, optics=cv.optics))
        for R in (6, 3): p.pass_(R, lambda mix, vol: Brush.paint("oil", 0.5, vol, mix), 0.6, threshold=0 if R == 6 else 4)
        err[opt] = p.error()
    assert err["v2"] < 13 and err["v2"] < err["classic"] + 3, err
