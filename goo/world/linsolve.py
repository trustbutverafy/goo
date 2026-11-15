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
    pass


class Factor:
    """A solver for one matrix. solve(b) -> x; records the relative residual and iterations in `stats`."""

    def __init__(self, A, mode="exact", stats=None):
        self.A = A.tocsc()
        self.mode, self.stats = mode, stats if stats is not None else {}
        if mode == "exact":
            self.lu = splu(self.A)
        elif mode == "iterative":
            try:
                self.ilu = spilu(self.A, drop_tol=ILU_DROP, fill_factor=ILU_FILL)
            except RuntimeError:                       # an incomplete factor hit a zero pivot: factorise fully
                self.mode = "exact"
                self.lu = splu(self.A)
                self.stats["solver fallbacks"] = self.stats.get("solver fallbacks", 0) + 1
                return
            self.M = LinearOperator(self.A.shape, self.ilu.solve)
        else:
            raise ValueError("solver: 'exact' or 'iterative'")

    def solve(self, b):
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
            self.lu = splu(self.A)
            self.stats["solver fallbacks"] = self.stats.get("solver fallbacks", 0) + 1
            x = self.lu.solve(b)
            res = float(np.linalg.norm(self.A @ x - b)) / nb
        self.stats["solver residual"] = max(self.stats.get("solver residual", 0.0), res)
        return x
