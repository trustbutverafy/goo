"""The compiled mixing kernel (goo/substance/_mixkernel.py) against the numpy reference (_freeenergy.evaluate_reference,
hvp_reference): the same formulas, cell by cell. f and mu agree to rounding (measured: bit for bit in all but a few
cells per 1e4, which differ in the last bit: numpy's vectorised log against libm's); the Hessian, f_x and mu_x to a
few ulps of their largest entry (the liquids' interaction product is a BLAS matmul in the reference)."""
import numpy as np
from goo.film import Film
from goo.substance import maps, library as L, _freeenergy as fe

NAMES = ["water", "gum_arabic", "rutile", "soot", "ultramarine", "hematite", "cadmium_sulfide", "ethanol"]


def _cells(S, n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.random((S, n)) * 1e-5
    q = np.where(rng.random((S, n)) < 0.5, q, 0.0)
    q[:, :5] = 0.0                                           # empty
    q[0, 5:10] = 1e-5; q[1:, 5:10] = 0.0                     # pure water
    q[:, 10:20] = 0.0; q[2, 10:20] = 1e-5; q[0, 10:20] = 1e-9  # rutile with a trace of water (past the entry point)
    return q


def test_compiled_mixing_equals_the_reference_to_rounding():
    subs = tuple(L.BY_NAME[x] for x in NAMES if x in L.BY_NAME)
    ms = fe.MixSet.of(subs)
    S, n = len(subs), 20000
    q = _cells(S, n)
    T = np.full(n, 293.15)
    R = fe.evaluate_reference(ms, q, T, True)
    C = fe.evaluate(ms, q, T, True)
    for name, a, b, tol in zip(("f", "mu", "hess", "f_x", "mu_x"), R, C, (1e-15, 1e-15, 1e-14, 1e-14, 1e-14)):
        fin = np.isfinite(a)
        assert np.array_equal(fin, np.isfinite(b)) and np.array_equal(a[~fin], b[~fin]), name
        assert np.max(np.abs(a[fin] - b[fin])) <= tol * np.max(np.abs(a[fin])), name
    dq = np.random.default_rng(1).standard_normal((S, n))
    hr, hc = fe.hvp_reference(ms, q, T, dq), fe.hvp(ms, q, T, dq)
    assert np.max(np.abs(hr - hc)) <= 1e-14 * np.max(np.abs(hr))


def test_entry_points_agree_with_each_other_bit_for_bit():
    """mixing_potentials is mixing without the Hessian; mixing_block is mixing's Hessian cut to rows; mixing_hvp is
    element-wise (blocks of cells change nothing)."""
    subs = (L.WATER, L.GUM_ARABIC, L.RUTILE)
    q = _cells(3, 3000, 2)[:, None, :]
    film = Film(subs, q)
    full = maps.mixing(film).value
    pot = maps.mixing_potentials(film).value
    for a, b in ((full.f, pot.f), (full.mu, pot.mu), (full.f_x, pot.f_x), (full.mu_x, pot.mu_x)):
        assert np.array_equal(a, b)
    rows = [0, 2]
    blk = maps.mixing_block(film, rows).value.hess
    assert np.array_equal(blk, full.hess[np.ix_(rows, rows)])
    dq = np.random.default_rng(3).standard_normal(q.shape)
    whole = maps.mixing_hvp(film, dq).value
    parts = np.concatenate([maps.mixing_hvp(Film(subs, q[:, :, i:i + 1000]), dq[:, :, i:i + 1000]).value
                            for i in range(0, 3000, 1000)], -1)
    assert np.array_equal(whole, parts)
