"""Linear solves of the step (SPEC §3): `exact` factorises (SuperLU) and is the reference; `iterative` solves the
same system with GMRES preconditioned by an incomplete LU, to a stated relative residual, by a fixed rule (the
same matrix and right-hand side give the same iterates: the systems are assembled on the active set in canonical
order, so a window and the panel give the same numbers, O5)."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spilu, gmres, LinearOperator

RTOL = 1e-10            # relative residual of an iterative solve
RESTART = 60
MAXITER = 20            # restarts
ILU_DROP, ILU_FILL = 1e-5, 20


class SolveFailed(Exception):
    """A linear solve that could not be done (a singular or non-finite system): the substep is rejected and the
    step halves, as for any other failed acceptance condition."""


def _splu(A):
    if not np.all(np.isfinite(A.data)):
        raise SolveFailed("non-finite matrix")
    try:
        return splu(A)
    except (RuntimeError, MemoryError) as e:          # "Factor is exactly singular", or SuperLU out of memory
        raise SolveFailed(str(e)) from None


class _Direct:
    """A factorisation for the step's systems, which are structurally symmetric (grid stencils): first with a
    minimum-degree ordering of A + A^T and diagonal pivots (three times less fill than the default column ordering
    on these matrices), checked on every solve after one step of refinement; a solve whose residual is not at the
    rounding of a pivoted factorisation (1e-9 of |b|) refactorises with SuperLU's default (partial pivoting) and
    keeps it. Deterministic: the same matrix gives the same factors and the same answers."""

    CHECK = 1e-9

    def __init__(self, A):
        self.A = sp.csc_matrix(A)
        if not np.all(np.isfinite(self.A.data)):
            raise SolveFailed("non-finite matrix")
        self.lu, self.safe = None, False
        try:
            self.lu = splu(self.A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                           options=dict(SymmetricMode=True))
        except (RuntimeError, MemoryError):
            self._fallback()

    def _fallback(self):
        self.lu, self.safe = _splu(self.A), True

    def solve(self, b):
        x = self.lu.solve(b)
        if self.safe:
            return x
        r = b - self.A @ x
        x = x + self.lu.solve(r)
        nb = float(np.linalg.norm(b))
        if not (np.all(np.isfinite(x)) and float(np.linalg.norm(b - self.A @ x)) <= self.CHECK * nb):
            self._fallback()
            x = self.lu.solve(b)
        return x


def _finite(x):
    if not np.all(np.isfinite(x)):
        raise SolveFailed("non-finite solution")
    return x


class Factor:
    """A solver for one matrix. solve(b) -> x; records the relative residual and iterations in `stats`."""

    def __init__(self, A, mode="exact", stats=None):
        self.A = A.tocsc()
        self.mode, self.stats = mode, stats if stats is not None else {}
        if mode == "exact":
            self.lu = _Direct(self.A)
        elif mode == "iterative":
            if not np.all(np.isfinite(self.A.data)):
                raise SolveFailed("non-finite matrix")
            try:
                self.ilu = spilu(self.A, drop_tol=ILU_DROP, fill_factor=ILU_FILL)
            except RuntimeError:                       # an incomplete factor hit a zero pivot: factorise fully
                self.mode = "exact"
                self.lu = _splu(self.A)
                self.stats["solver fallbacks"] = self.stats.get("solver fallbacks", 0) + 1
                return
            self.M = LinearOperator(self.A.shape, self.ilu.solve)
        else:
            raise ValueError("solver: 'exact' or 'iterative'")

    def solve(self, b):
        return _finite(self._solve(b))

    def _solve(self, b):
        if self.mode == "exact":
            return self.lu.solve(b)
        nb = float(np.linalg.norm(b))
        if nb == 0.0:
            return np.zeros_like(b)
        count = [0]

        def cb(_):
            count[0] += 1
        x, info = gmres(self.A, b, x0=self.ilu.solve(b), rtol=RTOL, atol=0.0, restart=RESTART, maxiter=MAXITER,
                        M=self.M, callback=cb, callback_type="pr_norm")
        res = float(np.linalg.norm(self.A @ x - b)) / nb
        self.stats["solver iterations"] = self.stats.get("solver iterations", 0) + count[0]
        if not res <= 10 * RTOL:
            # GMRES did not reach the tolerance within its fixed budget: factorise (deterministic, counted)
            self.mode = "exact"
            self.lu = _splu(self.A)
            self.stats["solver fallbacks"] = self.stats.get("solver fallbacks", 0) + 1
            x = self.lu.solve(b)
            res = float(np.linalg.norm(self.A @ x - b)) / nb
        self.stats["solver residual"] = max(self.stats.get("solver residual", 0.0), res)
        return x


NEWTON_RTOL = 1e-10     # relative residual of a linear solve inside a Newton iteration (the iterate is judged on its
                        # own nonlinear residual, so this sets the path, not the answer: DYNAMICS.md §9a)
NEWTON_RESTART, NEWTON_MAXITER = 40, 5


class Inner:
    """The linear solver of a Newton iteration, by a fixed rule: GMRES preconditioned by block Jacobi over each cell's
    unknowns (enough for a system near the identity), then, if that misses NEWTON_RTOL within JACOBI_ITERS, GMRES
    preconditioned by multigrid on the active set (multigrid.py) from where it stopped, then a factorisation (counted
    as a fallback). Unknowns sit at absolute cells (iy, ix) with component labels, so a window and the panel give the
    same iterates (O5)."""

    JACOBI_ITERS = 12

    def __init__(self, A, iy, ix, comp, stats=None):
        from goo.world.multigrid import block_inverse, cells_of
        self.A = sp.csr_matrix(A)
        if not np.all(np.isfinite(self.A.data)):
            raise SolveFailed("non-finite matrix")
        self.iy, self.ix, self.comp = iy, ix, comp
        self.stats = stats if stats is not None else {}
        self.lu = None
        self.mg = None
        D = block_inverse(self.A, cells_of(iy, ix))
        self.D = LinearOperator(self.A.shape, lambda v: D @ v)

    def _gmres(self, b, x0, M, restart, maxiter):
        count = [0]

        def cb(_):
            count[0] += 1
        x, info = gmres(self.A, b, x0=x0, rtol=NEWTON_RTOL, atol=0.0, restart=restart, maxiter=maxiter, M=M,
                        callback=cb, callback_type="pr_norm")
        self.stats["inner iterations"] = self.stats.get("inner iterations", 0) + count[0]
        return x

    def solve(self, b):
        return _finite(self._solve(b))

    def _solve(self, b):
        if self.lu is not None:
            return self.lu.solve(b)
        nb = float(np.linalg.norm(b))
        if nb == 0.0:
            return np.zeros_like(b)
        x = None
        if self.mg is None:
            x = self._gmres(b, None, self.D, self.JACOBI_ITERS, 1)
            if float(np.linalg.norm(self.A @ x - b)) <= NEWTON_RTOL * nb:
                return x
            from goo.world.multigrid import Multigrid
            self.mg = Multigrid(self.A, self.iy, self.ix, self.comp)
            self.M = LinearOperator(self.A.shape, self.mg)
        x = self._gmres(b, x, self.M, NEWTON_RESTART, NEWTON_MAXITER)
        if not float(np.linalg.norm(self.A @ x - b)) <= 10 * NEWTON_RTOL * nb:
            self.lu = _splu(self.A.tocsc())
            self.stats["inner fallbacks"] = self.stats.get("inner fallbacks", 0) + 1
            x = self.lu.solve(b)
        return x

