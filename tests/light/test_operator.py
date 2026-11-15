"""The star product: the monoid (G2), passivity under composition (G1), channels and bracketing."""
import numpy as np, pytest
from goo.light import (Layer, identity, scalar, symmetric, channels, star, product, solve, passivity_excess,
                       homogeneous, boundary)


def random_passive(rng, shape, n=1):
    """A random passive layer on n channels: non-negative R, T, R', T' whose columns of [R; T] and [T'; R']
    each sum to at most 1 (sub-stochastic)."""
    def block():
        M = rng.random(shape + (2 * n, n)) * (rng.random(shape + (2 * n, n)) > 0.2)
        M = M / (M.sum(-2, keepdims=True) + 1e-300) * rng.uniform(0.3, 1.0, shape + (1, n))
        return M[..., :n, :], M[..., n:, :]
    R, T = block()
    Tp, Rp = block()
    return Layer(R, T, Rp, Tp)


def random_film_like(rng, shape):
    K = 10 ** rng.uniform(1, 6, shape); S = 10 ** rng.uniform(1, 6, shape) * (rng.random(shape) > 0.1)
    R, T, _ = homogeneous(K, S, 10 ** rng.uniform(-7, -4, shape))
    b = boundary(rng.uniform(1, 2.5, shape), rng.uniform(1, 2.5, shape))
    return star(b, symmetric(R[..., None, None], T[..., None, None]))


def close(A, B, tol):
    return all(np.max(np.abs(x - y)) <= tol for x, y in zip(A.arrays(), B.arrays()))


@pytest.mark.parametrize("n", [1, 2, 3])
def test_G2_monoid_associative_with_the_empty_film_as_identity(n):
    rng = np.random.default_rng(n)
    shape = (5, 7)
    A, B, C = (random_passive(rng, shape, n) for _ in range(3))
    assert close(star(star(A, B), C), star(A, star(B, C)), 1e-12)
    e = identity(shape, n)
    for X in (A, B, C):
        for Y in (star(e, X), star(X, e)):
            assert all(np.array_equal(x, y) for x, y in zip(X.arrays(), Y.arrays()))   # exactly
    assert all(np.array_equal(x, y) for x, y in zip(star(e, e).arrays(), e.arrays()))
    if n == 1:
        F = [random_film_like(rng, shape) for _ in range(3)]
        assert close(star(star(F[0], F[1]), F[2]), star(F[0], star(F[1], F[2])), 1e-12)


def test_G2_monoid_is_not_commutative():
    rng = np.random.default_rng(7)
    A, B = random_passive(rng, (50,)), random_passive(rng, (50,))
    assert np.max(np.abs(star(A, B).R - star(B, A).R)) > 1e-3


@pytest.mark.parametrize("n", [1, 2])
def test_G1_passivity_power_out_never_exceeds_power_in(n):
    rng = np.random.default_rng(11 + n)
    shape = (200,)
    layers = [random_passive(rng, shape, n) for _ in range(6)]
    for L in layers:
        assert np.max(passivity_excess(L)) <= 1e-15
    P = product(layers)
    assert np.max(passivity_excess(P)) <= 1e-14
    films = [random_film_like(rng, shape) for _ in range(6)]
    assert np.max(passivity_excess(product(films))) <= 1e-14
    # every input, not only unit inputs: random non-negative fluxes from both sides
    a, b = rng.random(shape + (n,)), rng.random(shape + (n,))
    out = ((P.R + P.T) @ a[..., None] + (P.Rp + P.Tp) @ b[..., None])[..., 0].sum(-1)
    assert np.all(out <= a.sum(-1) + b.sum(-1) + 1e-14)


def test_two_uncoupled_identical_channels_equal_one_channel():
    rng = np.random.default_rng(3)
    shape = (4, 6)
    films = [random_film_like(rng, shape) for _ in range(5)]
    one = product(films)
    two = product([channels(L, 2) for L in films])          # the n-channel (matrix) path
    for X1, X2 in zip(one.arrays(), two.arrays()):
        assert np.max(np.abs(X2[..., 0, 0] - X1[..., 0, 0])) < 1e-14
        assert np.max(np.abs(X2[..., 1, 1] - X1[..., 0, 0])) < 1e-14
        assert np.all(X2[..., 0, 1] == 0) and np.all(X2[..., 1, 0] == 0)
    s1, s2 = solve(films), solve([channels(L, 2) for L in films])
    assert np.max(np.abs(s1.reflected - s2.reflected)) < 1e-14
    assert np.max(np.abs(s1.loss - s2.loss)) < 1e-14


def test_result_is_independent_of_bracketing():
    rng = np.random.default_rng(5)
    shape = (3, 8)
    L = [random_film_like(rng, shape) for _ in range(7)]
    left = product(L)
    right = L[-1]
    for X in reversed(L[:-1]):
        right = star(X, right)
    tree = star(star(star(L[0], L[1]), star(L[2], L[3])), star(star(L[4], L[5]), L[6]))
    assert close(left, right, 1e-13) and close(left, tree, 1e-13)


def test_solve_keeps_internal_fluxes_and_attributes_every_loss():
    rng = np.random.default_rng(9)
    shape = (10, 10)
    L = [random_film_like(rng, shape) for _ in range(5)]
    s = solve(L)
    P = product(L)
    assert np.max(np.abs(s.reflected - P.R[..., 0, 0])) < 1e-14
    assert np.max(np.abs(s.transmitted - P.T[..., 0, 0])) < 1e-14
    assert np.max(np.abs(s.absorbed - s.loss)) < 1e-14          # flux differences = each layer's own loss
    assert np.max(np.abs(1 - s.reflected - s.loss.sum(0) - s.transmitted)) < 1e-14
    assert np.all(s.loss >= -1e-16) and np.all(s.margin > 0)
    # each internal flux is what the layers on either side say it is
    for j in range(1, len(L)):
        U, D = product(L[:j]), product(L[j:])
        d = U.T[..., 0, 0] / (1 - U.Rp[..., 0, 0] * D.R[..., 0, 0])
        assert np.max(np.abs(s.down[j][..., 0] - d)) < 1e-14
        assert np.max(np.abs(s.up[j][..., 0] - D.R[..., 0, 0] * d)) < 1e-14
    # a lossless boundary absorbs nothing
    b = boundary(np.full(shape, 1.0), np.full(shape, 1.5))
    sb = solve([b] + L)
    assert np.max(np.abs(sb.loss[0])) < 1e-16


def test_scalar_helper_and_identity_shape():
    L = scalar(np.array([0.1, 0.2]), np.array([0.8, 0.7]))
    assert L.n == 1 and L.shape == (2,) and np.array_equal(L.Rp, L.R)
    assert product([]).R.shape == (1, 1)
