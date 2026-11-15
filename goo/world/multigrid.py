"""Aggregation multigrid on the active set, as a preconditioner for a Krylov solve (DYNAMICS.md §9a).

Unknowns carry their cell's absolute panel position (row, column) and a component label. Aggregates are 2 x 2
blocks of cells in absolute coordinates, one per component; coarse unknowns are numbered in raster order of
their absolute positions. So the hierarchy of a system depends on the system and the cells' absolute positions
only: a window and the panel build the same hierarchy for the same matrix, and the iterates are the same bit for
bit (O5). The smoother is damped block Jacobi over the unknowns of a cell; the coarsest level is factorised.
Every count (sweeps, levels, iterations) is fixed or set by the matrix's size, never by the window."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu as _splu0


def splu(A):
    from goo.world.linsolve import _splu
    return _splu(A)

COARSE = 1500        # unknowns at which a level is factorised
OMEGA = 0.7          # block-Jacobi damping
SWEEPS = 2           # pre- and post-smoothing sweeps


def block_inverse(A, cell):
    """The block-diagonal inverse of A over the unknowns sharing a cell, as a sparse matrix."""
    n = A.shape[0]
    if n == 0:
        return sp.csr_matrix((0, 0))
    order = np.argsort(cell, kind="stable")
    cs = cell[order]
    nc = int(cs[-1]) + 1
    cnt = np.bincount(cs, minlength=nc)
    k = int(cnt.max())
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    loc = np.empty(n, np.int64)
    loc[order] = np.arange(n) - start[cs]
    B = np.zeros((nc, k, k))
    if k > 1:
        B[:, np.arange(k), np.arange(k)] = 1.0           # padding for cells with fewer unknowns
        B[cell, loc, loc] = 0.0
    Ac = A.tocoo()
    same = cell[Ac.row] == cell[Ac.col]
    B[cell[Ac.row[same]], loc[Ac.row[same]], loc[Ac.col[same]]] = Ac.data[same]
    if k == 1:
        with np.errstate(divide="ignore"):
            Bi = 1.0 / B[:, 0, 0]
        return sp.csr_matrix((Bi[cell], (np.arange(n), np.arange(n))), shape=(n, n))
    try:
        Bi = np.linalg.inv(B)
    except np.linalg.LinAlgError as e:
        from goo.world.linsolve import SolveFailed
        raise SolveFailed(str(e)) from None
    idx = np.full((nc, k), -1, np.int64)
    idx[cell, loc] = np.arange(n)
    rows, cols, vals = [], [], []
    for a in range(k):
        for b in range(k):
            r, c = idx[:, a], idx[:, b]
            ok = (r >= 0) & (c >= 0)
            rows.append(r[ok]); cols.append(c[ok]); vals.append(Bi[ok, a, b])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def cells_of(iy, ix):
    key = iy.astype(np.int64) * (1 << 32) + ix.astype(np.int64)
    return np.unique(key, return_inverse=True)[1]


class Multigrid:
    """A V-cycle for A (n x n, any sparse format) whose unknowns sit at absolute cells (iy, ix) with labels comp."""

    def __init__(self, A, iy, ix, comp):
        self.levels = []
        A = sp.csr_matrix(A)
        iy, ix, comp = (np.asarray(v, np.int64) for v in (iy, ix, comp))
        while True:
            n = A.shape[0]
            if n <= COARSE:
                self.levels.append((A, splu(sp.csc_matrix(A)) if n else None, None, None))
                break
            Dinv = block_inverse(A, cells_of(iy, ix))
            key = ((iy // 2) * (1 << 32) + (ix // 2)) * 64 + comp
            ukey, agg = np.unique(key, return_inverse=True)
            if ukey.size >= n:                         # nothing aggregates (isolated cells): factorise here
                self.levels.append((A, splu(sp.csc_matrix(A)), None, None))
                break
            P = sp.csr_matrix((np.ones(n), (np.arange(n), agg)), shape=(n, ukey.size))
            self.levels.append((A, Dinv, P, P.T.tocsr()))
            A = (P.T @ A @ P).tocsr()
            comp, cc = ukey % 64, ukey // 64
            iy, ix = cc // (1 << 32), cc % (1 << 32)

    def __call__(self, b):
        return self._cycle(np.asarray(b, np.float64), 0)

    def _cycle(self, b, lev):
        A, D, P, R = self.levels[lev]
        if P is None:
            return D.solve(b) if D is not None else b.copy()
        x = OMEGA * (D @ b)
        for _ in range(SWEEPS - 1):
            x += OMEGA * (D @ (b - A @ x))
        x += P @ self._cycle(R @ (b - A @ x), lev + 1)
        for _ in range(SWEEPS):
            x += OMEGA * (D @ (b - A @ x))
        return x
