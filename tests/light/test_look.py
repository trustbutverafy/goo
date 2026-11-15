"""Looks: the light ledger (G3), passivity (G1), thick limit (G5), matched index (G6), order (G7), impossible
substances, and that a look reads the stack without touching it."""
import numpy as np, pytest
from goo.evidence import Derived, Bounded
from goo.film import Film, Light, Stack
from goo.units import um
from goo.light import look, operators, product, symmetric, Optical, external_diffuse
from analytic import Sub, Mixture, Particles, bands, diffuse, ground, random_stack
from test_bulk import km_hyperbolic


def lights(b):
    return [diffuse(b), Light(b, np.linspace(1, 3, b.B), ("collimated", 0.0)),
            Light(b, np.ones(b.B), ("collimated", 0.7))]


def test_mixture_satisfies_the_protocol():
    assert isinstance(Mixture(), Optical) and isinstance(Particles(), Optical)


@pytest.mark.parametrize("seed", range(12))
def test_G3_ledger_closes_for_random_admissible_stacks(seed):
    rng = np.random.default_rng(seed)
    st = random_stack(rng, H=4, W=5, nfilms=int(rng.integers(0, 5)), B=5)
    b = bands(5)
    for light in lights(b):
        L = look(st, b, light, Mixture())
        r = L.receipt
        assert r.op == "look" and r.residual["light"] <= 1e-12 and not r.flags
        # the identity per cell and band, and the ledger's accounts are the panel means of the arrays
        tot = L.reflectance + L.absorbed.sum(0) + L.ground + L.transmitted
        assert np.max(np.abs(tot - 1)) <= 1e-12
        assert np.allclose(L.reflectance, L.specular + L.diffuse, rtol=0, atol=4e-16)
        mean = lambda a: a.mean((0, 1))
        assert np.allclose(r.moved("light", "incident", "reflected.specular"), mean(L.specular), atol=1e-16)
        assert np.allclose(r.moved("light", "incident", "reflected.diffuse"), mean(L.diffuse), atol=1e-16)
        for k in range(len(st.films)):
            assert np.allclose(r.moved("light", "incident", f"absorbed[{k}]"), mean(L.absorbed[k]), atol=1e-16)
        assert np.allclose(r.moved("light", "incident", "absorbed.ground"), mean(L.ground), atol=1e-16)
        assert np.allclose(r.moved("light", "incident", "transmitted"), mean(L.transmitted), atol=1e-16)
        if np.isinf(st.ground.depth):
            assert np.all(L.transmitted == 0)
        assert np.max(np.abs(r.moved("light", "incident") - 1)) <= 1e-12
        assert r.moved("light", "debt") == 0.0


@pytest.mark.parametrize("seed", range(6))
def test_G1_passivity_in_a_look(seed):
    rng = np.random.default_rng(100 + seed)
    st = random_stack(rng, H=5, W=5, nfilms=4, B=6)
    b = bands(6)
    for light in lights(b):
        L = look(st, b, light, Mixture())
        assert np.all(L.reflectance <= 1 + 1e-15) and np.all(L.specular >= 0) and np.all(L.diffuse >= -1e-15)
        assert np.all(L.absorbed >= -1e-15) and np.all(L.ground >= -1e-15) and np.all(L.transmitted >= 0)
        assert not L.diverged.any() and np.all(L.debt == 0)


def test_G5_thick_limit_in_a_look():
    b = bands(4)
    paint = Sub("paint", k=np.array([2e3, 5e3, 1e4, 3e4]), s=1e5, n=1.5, sigma=0.05 * um)
    film = Film.uniform({paint: 0.05}, 2, 2)                     # 5 cm of paint
    looks = [look(Stack(ground(2, 2, Sub(nm_, k=k, s=s, n=n_)), [film], 1e-3), b, diffuse(b), Mixture())
             for nm_, k, s, n_ in [("black", 1e7, 0.0, 1.5), ("white", 0.0, 1e6, 1.2), ("glass", 0.0, 0.0, 2.0)]]
    for L in looks[1:]:
        assert np.max(np.abs(L.reflectance - looks[0].reflectance)) < 1e-14
        assert np.max(np.abs(L.ground)) < 1e-14


def test_G6_matched_index_a_particle_in_a_host_of_its_own_index_scatters_nothing():
    b = bands(4)
    host = Sub("host", k=2e3, n=1.5)
    black = ground(3, 3, Sub("black", k=1e6, s=0.0, n=1.5))
    for n_p, scatters in [(1.5, False), (2.0, True), (1.2, True)]:
        part = Sub("particle", k=1e3, n=n_p)
        film = Film.uniform({host: 30 * um, part: 10 * um}, 3, 3)
        optical = Particles()
        K, S, nh = optical.optical(film, b)
        L = look(Stack(black, [film], 1e-3), b, diffuse(b), optical)
        assert L.receipt.residual["light"] < 1e-12
        if scatters:
            assert np.all(S > 0) and np.all(L.diffuse > 1e-6)
        else:
            assert np.all(S == 0)
            assert np.all(L.diffuse == 0)                                  # nothing comes back from inside
            assert np.all(L.specular == external_diffuse(1.5))            # only the smooth surface reflects
            # what enters is absorbed: Beer-Lambert in the film, the rest in the black ground
            tau = np.exp(-(2e3 * 0.75 + 1e3 * 0.25) * 40 * um)
            assert np.allclose(L.absorbed[0], (1 - external_diffuse(1.5)) * (1 - tau), rtol=1e-13)


def commutator(Ra, Ta, Rb, Tb, Rg=None):
    """docs/LIGHT.md §G7: R(A over B) - R(B over A) for symmetric scalar layers, over black or over a ground of
    reflectance Rg."""
    c = (Ra * (1 + Rb ** 2 - Tb ** 2) - Rb * (1 + Ra ** 2 - Ta ** 2)) / (1 - Ra * Rb)
    if Rg is None:
        return c
    Rab = Ra + Ta ** 2 * Rb / (1 - Ra * Rb)
    Rba = Rb + Tb ** 2 * Ra / (1 - Ra * Rb)
    Tab = Ta * Tb / (1 - Ra * Rb)
    return c * (1 - Tab ** 2 * Rg ** 2 / ((1 - Rab * Rg) * (1 - Rba * Rg)))


def test_G7_order_a_glaze_over_body_colour_differs_from_the_reverse_by_the_commutator():
    b = bands(4)
    kg = np.array([4e4, 1e4, 3e3, 1e3])                       # a glaze: absorbs, no scattering
    kb, sb = np.array([1e3, 5e3, 2e4, 3e4]), np.array([2e5, 1.5e5, 1e5, 8e4])
    dg, db = 15 * um, 12 * um
    glaze, body = Sub("glaze", k=kg, n=1.0), Sub("body", k=kb, s=sb, n=1.0)
    A, Bf = Film.uniform({glaze: dg}, 2, 3), Film.uniform({body: db}, 2, 3)
    # independent scalars: Beer-Lambert for the glaze, Kubelka's hyperbolic form for the body and the ground
    Ra, Ta = np.zeros(4), np.exp(-kg * dg)
    Rb, Tb = km_hyperbolic(kb, sb, db)
    # the bare commutator of the operators
    opA = symmetric(Ra[:, None, None], Ta[:, None, None]); opB = symmetric(Rb[:, None, None], Tb[:, None, None])
    c_code = (product([opA, opB]).R - product([opB, opA]).R)[:, 0, 0]
    assert np.max(np.abs(c_code - commutator(Ra, Ta, Rb, Tb))) < 1e-15
    assert np.allclose(commutator(Ra, Ta, Rb, Tb), -Rb * (1 - Ta ** 2), rtol=1e-14)   # a clear glaze: R_a = 0
    # in a look, index-matched everywhere (no interfaces), over a white semi-infinite ground
    kw, sw = 1e2, 1e6
    Rg = 1 + kw / sw - np.sqrt((kw / sw) ** 2 + 2 * kw / sw)
    G = ground(2, 3, Sub("white", k=kw, s=sw, n=1.0))
    over = look(Stack(G, [Bf, A], 1e-3), b, diffuse(b), Mixture())       # glaze over body (films bottom first)
    under = look(Stack(G, [A, Bf], 1e-3), b, diffuse(b), Mixture())
    diff = over.reflectance - under.reflectance
    expect = commutator(Ra, Ta, Rb, Tb, Rg)
    assert np.max(np.abs(diff - expect)) < 1e-14 and np.min(np.abs(expect)) > 1e-3
    # lossless layers commute (the numerator vanishes when R + T = 1)
    R1, T1 = km_hyperbolic(1e-9, 1e5, 10 * um); R2, T2 = km_hyperbolic(1e-9, 3e4, 20 * um)
    assert abs(commutator(R1, T1, R2, T2)) < 1e-9


def test_a_look_never_modifies_the_stack_and_takes_read_only_arrays():
    rng = np.random.default_rng(42)
    st = random_stack(rng, H=3, W=3, nfilms=3, B=4)
    arrays = [f.q for f in st.films] + [st.ground.z, st.ground.held]
    for a in arrays:
        a.flags.writeable = False
    before = [a.copy() for a in arrays]
    ids = [id(f) for f in st.films]
    b = bands(4)
    L1 = look(st, b, diffuse(b), Mixture())
    L2 = look(st, b, diffuse(b), Mixture())
    assert all(np.array_equal(a, c) for a, c in zip(arrays, before))
    assert [id(f) for f in st.films] == ids and all(not a.flags.writeable for a in arrays)
    assert np.array_equal(L1.reflectance, L2.reflectance) and L1.receipt.digest() == L2.receipt.digest()


def test_a_different_light_changes_nothing_in_the_operators():
    rng = np.random.default_rng(8)
    st = random_stack(rng, H=3, W=4, nfilms=3, B=4)
    b = bands(4)
    ops0 = operators(st, b, Mixture())
    looks = [look(st, b, light, Mixture()) for light in lights(b)]
    ops1 = operators(st, b, Mixture())
    for A, B_ in zip(ops0.layers, ops1.layers):
        assert all(np.array_equal(x, y) for x, y in zip(A.arrays(), B_.arrays()))
    # the power spectrum is not in the fractions: same geometry, different power, the same look
    other = Light(b, np.array([5.0, 0.1, 2.0, 9.0]), "diffuse")
    assert np.array_equal(look(st, b, other, Mixture()).reflectance, looks[0].reflectance)
    assert not np.allclose(looks[0].reflectance, looks[2].reflectance)


def test_the_n_channel_path_in_a_look_equals_one_channel():
    rng = np.random.default_rng(21)
    st = random_stack(rng, H=3, W=4, nfilms=3, B=4)
    b = bands(4)
    for light in lights(b):
        L1, L2 = look(st, b, light, Mixture()), look(st, b, light, Mixture(), channels=2)
        for x, y in [(L1.reflectance, L2.reflectance), (L1.specular, L2.specular), (L1.absorbed, L2.absorbed),
                     (L1.ground, L2.ground)]:
            assert np.max(np.abs(x - y)) < 1e-14
        assert L2.receipt.residual["light"] < 1e-12


def test_an_empty_film_is_the_identity_and_its_values_are_not_read():
    rng = np.random.default_rng(13)
    st = random_stack(rng, H=3, W=3, nfilms=2, B=4)
    ghost = Film.empty((Sub("ghost", k=1e9, s=1e9, n=3.0, sigma=1 * um),), 3, 3)   # Mixture gives n = 0 here
    b = bands(4)
    L0 = look(st, b, diffuse(b), Mixture())
    for pos in (0, 1, 2):
        films = list(st.films); films.insert(pos, ghost)
        L = look(Stack(st.ground, films, 1e-3), b, diffuse(b), Mixture())
        assert np.max(np.abs(L.reflectance - L0.reflectance)) < 1e-15 and np.all(L.absorbed[pos] == 0)


def test_index_steps_reflect_and_matched_steps_do_not():
    b = bands(2)
    clear = lambda n: Sub(f"clear{n}", n=n)
    G = ground(1, 1, Sub("black", k=1e7, n=1.5))
    f1 = Film.uniform({clear(1.5): 20 * um}, 1, 1)
    L_one = look(Stack(G, [f1], 1e-3), b, diffuse(b), Mixture())
    L_two_same = look(Stack(G, [f1, Film.uniform({clear(1.5): 5 * um}, 1, 1)], 1e-3), b, diffuse(b), Mixture())
    assert np.max(np.abs(L_one.reflectance - L_two_same.reflectance)) < 1e-15
    L_step = look(Stack(G, [f1, Film.uniform({clear(2.0): 5 * um}, 1, 1)], 1e-3), b, diffuse(b), Mixture())
    assert np.all(L_step.diffuse > 1e-3)                    # the 2.0 | 1.5 interface sends diffuse light back


def test_roughness_moves_surface_reflection_from_specular_to_diffuse():
    b = bands(3)
    G = ground(1, 1, Sub("black", k=1e7, n=1.5))
    out = []
    for sig in [0.0, 0.02 * um, 0.1 * um, 1 * um]:
        f = Film.uniform({Sub("varnish", n=1.5, sigma=sig): 20 * um}, 1, 1)
        out.append(look(Stack(G, [f], 1e-3), b, Light(b, np.ones(3), ("collimated", 0.0)), Mixture()))
    R = [L.reflectance for L in out]
    assert all(np.allclose(r, R[0], rtol=0, atol=1e-15) for r in R)      # roughness divides, it does not add
    spec = [L.specular[0, 0] for L in out]
    assert np.allclose(spec[0], 0.04) and spec[0][0] > spec[1][0] > spec[2][0] > spec[3][0] >= 0
    assert spec[3].max() < 1e-12
    lam = b.centres
    assert np.allclose(spec[1], 0.04 * np.exp(-(4 * np.pi * 0.02e-6 / lam) ** 2), rtol=1e-13)


def test_gain_below_threshold_books_its_excess_to_debt():
    b = bands(3)
    amp = Sub("amp", k=-3e4, s=0.0, n=1.0)
    G = ground(2, 2, Sub("grey", k=1e4, s=1e4, n=1.0))       # R_inf = 0.268: round trip gain < 1
    st = Stack(G, [Film.uniform({amp: 10 * um}, 2, 2)], 1e-3)
    L = look(st, b, diffuse(b), Mixture())
    r = L.receipt
    assert not L.diverged.any() and r.residual["light"] < 1e-12
    assert r.flagged("impossible") and "debt[0]" in r.flagged("impossible")[0].detail
    assert np.all(L.debt[0] > 0) and np.allclose(L.debt[0], -L.absorbed[0])
    assert np.all(L.reflectance > 0.268)                     # amplified
    debt = r.moved("light", "debt")
    assert np.allclose(1 + debt, r.moved("light", "incident", "reflected") + r.moved("light", "incident", "absorbed"),
                       atol=1e-12)
    # independent: amplified twice through the film, Rg = R_inf of the ground
    Rg = 1 + 1 - np.sqrt(1 + 2)
    tau2 = np.exp(2 * 3e4 * 10e-6)
    assert np.allclose(L.reflectance, tau2 * Rg, rtol=1e-12)


def test_past_the_lasing_threshold_a_look_is_defined_and_flagged():
    b = bands(3)
    # a cavity: the top surface's internal reflectance r_i(1.5) = 0.6 above, a lossless white ground (R = 1)
    # below; the round trip gain exp(2 |K| d) r_i exceeds 1 for every band here
    amp = Sub("amp", k=np.array([-1.5e4, -3e4, -1e5]), s=0.0, n=1.5)
    G = ground(2, 2, Sub("white", k=0.0, s=1e6, n=1.5))
    st = Stack(G, [Film.uniform({amp: 20 * um}, 2, 2)], 1e-3)
    L = look(st, b, diffuse(b), Mixture())
    assert L.diverged.all()
    for a in (L.reflectance, L.absorbed, L.ground, L.transmitted, L.debt):
        assert np.all(np.isfinite(a)) and np.all(a == 0)
    assert np.all(L.unresolved == 1) and L.receipt.residual["light"] < 1e-12
    assert any("lasing" in f.detail for f in L.receipt.flagged("impossible"))
    assert np.allclose(L.receipt.moved("light", "incident", "unresolved"), 1)
    # without the cavity (no index step at the top: r_i = 0) the same gain only amplifies: R = exp(2 |K| d)
    amp1, G1 = Sub("amp", k=amp.k, n=1.0), ground(2, 2, Sub("white", k=0.0, s=1e6, n=1.0))
    L = look(Stack(G1, [Film.uniform({amp1: 20 * um}, 2, 2)], 1e-3), b, diffuse(b), Mixture())
    assert not L.diverged.any() and np.allclose(L.reflectance, np.exp(-2 * amp.k * 20e-6), rtol=1e-12)
    assert L.receipt.residual["light"] < 1e-12            # relative to the power entering: incident + debt
    # mixed: a film past its own threshold in one cell only (scattering gain medium, thick in one cell)
    las = Sub("laser", k=-2e4, s=1e4, n=1.0)
    q = np.full((1, 1, 2), 10 * um); q[0, 0, 1] = 500 * um
    st = Stack(ground(1, 2, Sub("black", k=1e7, n=1.0)), [Film((las,), q)], 1e-3)
    L = look(st, b, diffuse(b), Mixture())
    assert not L.diverged[0, 0].any() and L.diverged[0, 1].all() and L.receipt.residual["light"] < 1e-12
    assert np.all(np.isfinite(L.reflectance))


def test_derived_and_bounded_returns_are_unwrapped_and_flagged():
    class Wrapped(Mixture):
        def optical(self, film, bands):
            K, S, n = super().optical(film, bands)
            return Derived((K, S, n), "assumed", in_domain=np.zeros(film.shape, bool), map="test.optical")

        def surface(self, film, bands):
            n, sig = super().surface(film, bands)
            return Bounded(n * 0.99, n * 1.01, "computed", "test.index"), sig
    rng = np.random.default_rng(1)
    st = random_stack(rng, H=2, W=2, nfilms=1, B=3)
    b = bands(3)
    L = look(st, b, diffuse(b), Wrapped())
    assert L.receipt.flagged("out_of_domain") and L.receipt.flagged("bounds_collapsed")
    assert L.receipt.residual["light"] < 1e-12


def test_bands_must_match():
    rng = np.random.default_rng(2)
    st = random_stack(rng, nfilms=1, B=3)
    with pytest.raises(ValueError): look(st, bands(3), diffuse(bands(4)), Mixture())


def test_the_ground_splits_what_it_takes_into_absorbed_and_transmitted():
    b = bands(3)
    paint = Film.uniform({Sub("paint", k=1e3, s=1e4, n=1.5): 20 * um}, 2, 2)
    body = Sub("paper", k=np.array([10.0, 1e3, 1e5]), s=2e4, n=1.5)
    for depth in [100 * um, 1e-3]:
        L = look(Stack(ground(2, 2, body, depth), [paint], 1e-3), b, diffuse(b), Mixture())
        # independently: Kubelka's ground over a non-reflecting back, entered through the film above
        Rg, Tg = km_hyperbolic(body.k, body.s, depth)
        assert L.receipt.residual["light"] < 1e-12
        assert np.all(L.transmitted > 0) and np.all(L.ground > 0)
        r = L.receipt
        assert np.allclose(r.moved("light", "incident", "transmitted"), L.transmitted.mean((0, 1)), atol=1e-16)
    # the split is the ground body's own: flux d entering the body, up flux u leaving it; transmitted = Tg d,
    # absorbed = (1 - Rg - Tg) d for the body alone, so their ratio is Tg / (1 - Rg - Tg), whatever is above
    for above in ([], [paint]):
        L = look(Stack(ground(2, 2, body, 100 * um), above, 1e-3), b, diffuse(b), Mixture())
        Rg, Tg = km_hyperbolic(body.k, body.s, 100 * um)
        assert np.allclose(L.transmitted / L.ground, Tg / (1 - Rg - Tg), rtol=1e-12)
    # an opaque ground transmits nothing, to rounding
    L = look(Stack(ground(2, 2, body, 1.0), [paint], 1e-3), b, diffuse(b), Mixture())
    assert np.all(L.transmitted < 1e-200) and L.receipt.residual["light"] < 1e-12


@pytest.mark.parametrize("seed", range(4))
def test_dedup_blocks_and_threads_change_nothing_bit_for_bit(seed, monkeypatch):
    import sys; lk = sys.modules["goo.light.look"]
    rng = np.random.default_rng(300 + seed)
    st = random_stack(rng, H=12, W=10, nfilms=3, B=4, gain=bool(seed % 2))
    # repeat cells: copy one row of every input over half the panel, so dedup has work to do
    for f in st.films:
        f.q[:, 6:, :] = f.q[:, :1, :]
    b = bands(4)
    for light in lights(b):
        ref = look(st, b, light, Mixture(), dedup=False, threads=1)
        monkeypatch.setattr(lk, "BLOCK", 7)                 # many tiny blocks
        for kw in [dict(), dict(threads=3), dict(dedup=False, threads=2)]:
            L = look(st, b, light, Mixture(), **kw)
            for name in ("reflectance", "specular", "diffuse", "absorbed", "ground", "transmitted", "debt",
                         "diverged", "unresolved"):
                assert np.array_equal(getattr(L, name), getattr(ref, name)), name
            assert L.receipt.digest() == ref.receipt.digest()
        monkeypatch.setattr(lk, "BLOCK", 32768)
    films, ground_, _ = lk._inputs(st, b, Mixture())
    assert lk._distinct_cells(films, ground_, 120) is not None


def test_dedup_never_merges_different_cells():
    import sys; lk = sys.modules["goo.light.look"]
    rng = np.random.default_rng(5)
    st = random_stack(rng, H=10, W=10, nfilms=2, B=3)
    for f in st.films:
        f.q[:] = f.q[:, :1, :1]                              # every cell the same ...
    st.films[0].q[0, 3, 4] += 1e-12                         # ... but one
    films, ground_, _ = lk._inputs(st, bands(3), Mixture())
    rep, inv = lk._distinct_cells(films, ground_, 100)
    assert rep.size == 2 and inv[34] != inv[0] and np.sum(inv == inv[34]) == 1
