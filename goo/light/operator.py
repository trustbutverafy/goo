"""Layer operators and the star product (SPEC §7). See docs/LIGHT.md §1-§2.

A layer acts on the fluxes entering it from above (a, going down) and from below (b, going up):

    up out of its top        =  R a  +  T' b
    down out of its bottom   =  T a  +  R' b

R, T, R', T' are arrays of shape (..., n, n): the leading axes are bands and cells, the last two are flux
channels. On the first rung n = 1 (two fluxes, one each way, bands independent). Nothing here assumes n = 1;
n = 1 only takes an elementwise fast path that computes the same expressions.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Layer:
    """A passive (or, for an impossible substance, active) operator on fluxes. Arrays (..., n, n)."""
    R: np.ndarray     # reflection of flux arriving from above
    T: np.ndarray     # transmission downward
    Rp: np.ndarray    # reflection of flux arriving from below (R')
    Tp: np.ndarray    # transmission upward (T')

    @property
    def n(self):
        return self.R.shape[-1]

    @property
    def shape(self):
        """The leading (band, cell) shape."""
        return self.R.shape[:-2]

    def arrays(self):
        return (self.R, self.T, self.Rp, self.Tp)


def symmetric(R, T):
    """A layer that looks the same from both sides (a homogeneous film): R' = R, T' = T."""
    return Layer(R, T, R, T)


def identity(shape=(), n=1):
    """The empty film: R = R' = 0, T = T' = 1. The identity of the star product, exactly."""
    z = np.zeros(tuple(shape) + (n, n))
    one = np.broadcast_to(np.eye(n), tuple(shape) + (n, n)).copy()
    return Layer(z, one, z.copy(), one.copy())


def scalar(R, T, Rp=None, Tp=None):
    """A one-channel layer from arrays of the leading shape (adds the two channel axes of size 1)."""
    f = lambda x: np.asarray(x, np.float64)[..., None, None]
    return Layer(f(R), f(T), f(R if Rp is None else Rp), f(T if Tp is None else Tp))


def channels(layer, n):
    """The same layer on n uncoupled identical channels: each operator becomes block-diagonal (I_n kron X).
    For n-channel layers built this way the n-channel path must reproduce the one-channel result."""
    k = layer.n
    def blow(X):
        out = np.zeros(X.shape[:-2] + (n * k, n * k))
        for i in range(n):                                    # a loop over channels, not over cells or bands
            out[..., i * k:(i + 1) * k, i * k:(i + 1) * k] = X
        return out
    return Layer(*(blow(X) for X in layer.arrays()))


def _mm(A, B):
    return A @ B


def star(top, below, report=None):
    """The star product: `top` over `below` (SPEC §7, docs/LIGHT.md §2).

        X   = (1 - R1' R2)^-1             Y = (1 - R2 R1')^-1
        R12 = R1 + T1' R2 X T1            T12 = T2 X T1
        R12'= R2' + T2 R1' Y T2'          T12'= T1' Y T2'

    For scalars this is SPEC's R12 = R1 + T1 R2 T1' (1 - R1' R2)^-1, T12 = T2 (1 - R1' R2)^-1 T1.
    `report`, if a list, receives the round-trip denominator (1 - R1' R2) on the one-channel path, or the
    spectral radius margin 1 - rho(R1' R2) on the n-channel path: where it is <= 0 the geometric series of
    round trips diverges (a cavity past its threshold; only possible with gain)."""
    R1, T1, R1p, T1p = top.arrays()
    R2, T2, R2p, T2p = below.arrays()
    if top.n == 1 and below.n == 1:
        den = 1.0 - R1p * R2
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            X = 1.0 / den
            out = Layer(R1 + T1p * R2 * X * T1, T2 * X * T1, R2p + T2 * R1p * X * T2p, T1p * X * T2p)
        if report is not None:
            report.append(den[..., 0, 0])
        return out
    n = top.n
    I = np.eye(n)
    A = I - _mm(R1p, R2)
    Bm = I - _mm(R2, R1p)
    X = np.linalg.inv(A)
    Y = np.linalg.inv(Bm)
    out = Layer(R1 + _mm(_mm(_mm(T1p, R2), X), T1), _mm(_mm(T2, X), T1),
                R2p + _mm(_mm(_mm(T2, R1p), Y), T2p), _mm(_mm(T1p, Y), T2p))
    if report is not None:
        rho = np.max(np.abs(np.linalg.eigvals(_mm(R1p, R2))), axis=-1)
        report.append(1.0 - rho)
    return out


def product(layers, report=None):
    """The ordered star product of a list of layers, top first. The empty list is the identity (of shape ())."""
    layers = list(layers)
    if not layers:
        return identity()
    acc = layers[0]
    for L in layers[1:]:
        acc = star(acc, L, report)
    return acc


def passivity_excess(layer):
    """How far a layer is from passive: the largest of (power out - power in) over unit inputs from above and
    from below, and of -(any entry). Zero or less (to rounding) for a passive layer. Shape: the leading shape.

    For intensity operators with non-negative entries the worst input is a single channel, so the column sums
    of [R; T] and of [T'; R'] bound the power out of every input."""
    R, T, Rp, Tp = layer.arrays()
    down = (R.sum(-2) + T.sum(-2)).max(-1) - 1.0         # input from above, worst channel
    up = (Rp.sum(-2) + Tp.sum(-2)).max(-1) - 1.0         # input from below
    neg = -np.minimum.reduce([X.min((-2, -1)) for X in (R, T, Rp, Tp)])
    return np.maximum(np.maximum(down, up), neg)


@dataclass(frozen=True)
class Solution:
    """The composite of a list of layers (top first) with its internal fluxes, for incidence `a0` from above.

    down[j], up[j]   the fluxes (..., n) at interface j: j = 0 above the first layer, j = N below the last
    absorbed[j]      net power absorbed by layer j: (down - up) summed over channels at j, minus the same at j+1
    loss[j]          the same power from layer j's own operator applied to the fluxes entering it
    margin           the smallest round-trip margin met in any product (<= 0: diverged), leading shape
    """
    total: Layer
    down: tuple
    up: tuple
    absorbed: np.ndarray        # [N, ...leading]
    loss: np.ndarray            # [N, ...leading]
    margin: np.ndarray

    @property
    def reflected(self):
        return self.up[0].sum(-1)

    @property
    def transmitted(self):
        return self.down[-1].sum(-1)


def solve(layers, a0=None):
    """Compose layers (top first) keeping the internal fluxes, so absorption is attributed to each layer.

    Bottom-up, D_j = L_j * ... * L_{N-1}; top-down, U_j = L_0 * ... * L_{j-1}. At interface j the down flux is
    d_j = (1 - U_j' R_{D_j})^-1 T_{U_j} a0 and the up flux u_j = R_{D_j} d_j. Layer j absorbs
    (d_j - u_j) - (d_{j+1} - u_{j+1}), summed over channels. Nothing is a remainder: the reflectance is the
    composite's, each absorption is a difference of fluxes, and the ledger checks that they add up."""
    layers = list(layers)
    N = len(layers)
    if N == 0:
        raise ValueError("solve: no layers")
    n = layers[0].n
    lead = np.broadcast_shapes(*(L.shape for L in layers))
    if a0 is None:
        a0 = np.full(lead + (n,), 1.0 / n)
    a0 = np.broadcast_to(np.asarray(a0, np.float64), lead + (n,))
    margins = []

    # bottom-up: reflection from above of everything from layer j down
    RD = [None] * (N + 1)
    RD[N] = np.zeros(lead + (n, n))
    D = layers[N - 1]
    RD[N - 1] = np.broadcast_to(D.R, lead + (n, n))
    for j in range(N - 2, -1, -1):
        D = star(layers[j], D, margins)
        RD[j] = D.R
    total_bu = D

    def apply(M, v):
        return (M @ v[..., None])[..., 0]

    downs, ups = [a0], [apply(RD[0], a0)]
    U = None
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for j in range(1, N + 1):
            U = layers[0] if U is None else star(U, layers[j - 1], margins)
            if j < N:
                if n == 1:
                    den = 1.0 - U.Rp[..., 0, 0] * RD[j][..., 0, 0]
                    margins.append(den)
                    d = (U.T[..., 0, 0] * a0[..., 0] / den)[..., None]
                else:
                    A = np.eye(n) - U.Rp @ RD[j]
                    d = np.linalg.solve(A, apply(U.T, a0)[..., None])[..., 0]
                u = apply(RD[j], d)
            else:
                d, u = apply(U.T, a0), np.zeros(lead + (n,))
            downs.append(d)
            ups.append(u)
    net = np.array([(d - u).sum(-1) for d, u in zip(downs, ups)])
    absorbed = net[:-1] - net[1:]
    # each layer's loss from its own operator and the fluxes entering it: (1 - colsum R - colsum T) . d_j
    # + (1 - colsum R' - colsum T') . u_{j+1}. Independent of the telescoping above; the ledger sums these.
    loss = np.array([((1.0 - L.R.sum(-2) - L.T.sum(-2)) * downs[j]).sum(-1)
                     + ((1.0 - L.Rp.sum(-2) - L.Tp.sum(-2)) * ups[j + 1]).sum(-1) for j, L in enumerate(layers)])
    margin = np.min(np.array([np.broadcast_to(m, lead) for m in margins]), axis=0) if margins else np.ones(lead)
    return Solution(U if U is not None else total_bu, tuple(downs), tuple(ups), absorbed, loss, margin)
