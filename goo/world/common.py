"""What the three substeps share: the context of a step on a window, mixing evaluated on compacted cells, and
the increments of F's terms (computed from increments, not differences of totals: DYNAMICS.md §5)."""
from dataclasses import dataclass
import numpy as np
from goo.film import Film
from goo.world.protocol import call, Mixing


class Reject(Exception):
    """A substep whose acceptance condition failed: the step is redone as two half steps."""


@dataclass
class Ctx:
    grid: object
    substances: tuple
    rho: np.ndarray          # [S]
    con: object              # the Constitutive
    env: object
    ground: object
    zb: np.ndarray           # [N]
    r: np.ndarray            # [N] clock rate
    gam_f: np.ndarray        # [nF] lagged
    Swet: np.ndarray         # [N] lagged, or None
    ell: np.ndarray          # [N]
    P: np.ndarray            # [S, N] lagged, or None
    Pb: np.ndarray           # [N]
    cap: float               # pore capacity, m
    a: float                 # gravity split density
    mu_env: np.ndarray       # [S], 0 where no reservoir
    has_env: np.ndarray      # [S] bool
    flags: list
    solver: str = "exact"
    stats: dict = None
    memo: list = None

    @property
    def g(self):
        return self.env.gravity


def film_of(ctx, q, cells=None):
    """A Film of the window (cells None) or of compacted cells, shape [S, 1, n]."""
    if cells is None:
        return Film(ctx.substances, q.reshape(len(ctx.substances), ctx.grid.H, ctx.grid.W))
    return Film(ctx.substances, q[:, cells][:, None, :])


def coeff(ctx, name, q, cells, *extra):
    """A constitutive function evaluated on compacted cells, flattened; None if absent."""
    v = call(ctx.con, name, film_of(ctx, q, cells), *extra, flags=ctx.flags)
    if v is None:
        return None
    if isinstance(v, tuple):
        out = []
        for x in v:
            x = np.asarray(x, np.float64)
            out.append(x.reshape(x.shape[:-2] + (-1,)) if x.ndim >= 2 else x)
        return tuple(out)
    v = np.asarray(v, np.float64)
    return v.reshape(v.shape[:-2] + (-1,))


def mixing(ctx, qc):
    """Mixing on compacted cells qc [S, n]: (f, mu, hess, f_x, mu_x) flattened, with empty cells and absent
    substances masked to zero. None if the term is absent."""
    S, n = qc.shape
    fn = getattr(ctx.con, "mixing", None)
    if fn is None:
        return None
    # the maps are functions of the amounts, so the same amounts give the same numbers: the last few evaluations of
    # a piece are kept and returned for equal input (bit for bit), which the substeps meet at their boundaries
    return _memo(ctx, qc, True)


def mixing_potentials(ctx, qc):
    """As mixing() but the Hessian may be None: through the protocol's optional `mixing_potentials` (the same values
    without the [S, S, n] Hessian), else mixing() itself."""
    return _memo(ctx, qc, False)


def _memo(ctx, qc, hess):
    if getattr(ctx.con, "mixing", None) is None:
        return None
    # the maps are functions of the amounts, so the same amounts give the same numbers: the last few evaluations of
    # a piece are kept and returned for equal input (bit for bit), which the substeps meet at their boundaries
    memo = ctx.memo
    if memo is not None:
        for k, (qm, out) in enumerate(memo):
            if qm.shape == qc.shape and np.array_equal(qm, qc) and (out[2] is not None or not hess):
                return out
    if hess or getattr(ctx.con, "mixing_potentials", None) is None:
        out = _mixing(ctx, qc)
    else:
        out = _mixing(ctx, qc, ctx.con.mixing_potentials)
    if memo is not None and out is not None:
        memo.insert(0, (np.array(qc, copy=True), out))
        del memo[MEMO:]
    return out


def mixing_hvp(ctx, qc, dq):
    """The convex mixing Hessian applied to dq [S, n] on compacted cells qc, 0 for absent substances: through the
    protocol's optional `mixing_hvp` (no [S, S, n] array), else from mixing()'s Hessian."""
    fn = getattr(ctx.con, "mixing_hvp", None)
    present = qc > 0
    if fn is None:
        H = mixing(ctx, qc)[2]
        return np.where(present, np.einsum("ijn,jn->in", H, np.where(present, dq, 0.0)), 0.0)
    from goo.world.protocol import unwrap
    S, n = qc.shape
    dqm = np.where(present, dq, 0.0)
    out = np.empty((S, n))
    # in blocks of cells: the map is element-wise, so blocks change nothing; and goo.substance's hvp of 2026-11-15
    # reuses a name inside its loop over chunks of 65536 distinct compositions (wrong index, or a 42 GB allocation, on a
    # larger call), which blocks of HVP_BLOCK cells never reach
    for b0 in range(0, n, HVP_BLOCK):
        sl = slice(b0, min(n, b0 + HVP_BLOCK))
        with np.errstate(divide="ignore", invalid="ignore"):
            v = unwrap(fn(Film(ctx.substances, qc[:, None, sl]), dqm[:, None, sl], ctx.env.T), "mixing_hvp", ctx.flags)
        out[:, sl] = np.asarray(v, np.float64).reshape(S, -1)
    return np.where(present, out, 0.0)


HVP_BLOCK = 32768


MEMO = 3


def _mixing(ctx, qc, fn=None):
    S, n = qc.shape
    fn = ctx.con.mixing if fn is None else fn
    with np.errstate(divide="ignore", invalid="ignore"):
        from goo.world.protocol import unwrap
        mx = unwrap(fn(Film(ctx.substances, qc[:, None, :]), ctx.env.T), "mixing", ctx.flags)
    if mx is None:
        return None
    if not hasattr(mx, "f"):
        mx = Mixing(*mx)
    present = qc > 0
    wet = qc.sum(0) > 0
    f = np.where(wet, np.asarray(mx.f, np.float64).reshape(n), 0.0)
    mu = np.where(present, np.asarray(mx.mu, np.float64).reshape(S, n), 0.0)
    pp = present[:, None, :] & present[None, :, :]
    H = None if mx.hess is None else np.where(pp, np.asarray(mx.hess, np.float64).reshape(S, S, n), 0.0)
    fx = np.zeros(n) if mx.f_x is None else np.where(wet, np.asarray(mx.f_x, np.float64).reshape(n), 0.0)
    mux = np.zeros((S, n)) if mx.mu_x is None else np.where(present, np.asarray(mx.mu_x, np.float64).reshape(S, n), 0.0)
    return f, mu, H, fx, mux


def dF_grav(ctx, cells, q0, dq):
    """Increment of F_grav on compacted cells for a change dq [S, n]: exact for this quadratic."""
    gx, gy, gn = ctx.g
    h, m = q0.sum(0), ctx.rho @ q0
    dh, dm = dq.sum(0), ctx.rho @ dq
    psi = -(gx * ctx.grid.X[cells] + gy * ctx.grid.Y[cells])
    per = gn * (dm * ctx.zb[cells] + 0.5 * (dm * h + m * dh + dm * dh)) + dm * psi
    return ctx.grid.dx ** 2 * np.sum(per)


def dF_cap(ctx, faces, h0, dh):
    """Increment of F_cap over the given faces for a change dh [N] of the window's thickness."""
    G = ctx.grid
    s = ctx.zb + h0
    d = s[G.fR[faces]] - s[G.fL[faces]]
    dd = dh[G.fR[faces]] - dh[G.fL[faces]]
    return np.sum(ctx.gam_f[faces] * (d * dd + 0.5 * dd * dd))


def dF_wet(ctx, cells, h0, dh):
    """Increment of F_wet = sum (-S)(1 - exp(-h/ell)) for a change dh, finite for any ell > 0: written as
    sign(dh) (-S) exp(-min(h, h')/ell) (1 - exp(-|dh|/ell)), so a molecular ell and a large dh give 0, not 0 * inf."""
    if ctx.Swet is None:
        return 0.0
    S, l = ctx.Swet[cells], ctx.ell[cells]
    lo = np.minimum(h0, h0 + dh)
    per = np.sign(dh) * (-S) * np.exp(-lo / l) * (-np.expm1(-np.abs(dh) / l))
    return ctx.grid.dx ** 2 * np.sum(per)


def dF_exch(ctx, dq):
    """Increment of F_exch for a change dq [S, n]."""
    return -ctx.grid.dx ** 2 * np.sum(ctx.mu_env[:, None] * dq)


def mu_wet(ctx, h, cells=None):
    """d f_wet / dh per volume, J/m^3 (positive: thin films are costly)."""
    if ctx.Swet is None:
        return 0.0 if cells is None else np.zeros(len(cells))
    S = ctx.Swet if cells is None else ctx.Swet[cells]
    l = ctx.ell if cells is None else ctx.ell[cells]
    hh = h if cells is None else h
    return (-S / l) * np.exp(-hh / l)


def pore_B(theta):
    return -theta - np.log1p(-theta)
