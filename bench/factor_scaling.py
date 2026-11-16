"""How the present flow solve's sparse factorisation scales: the flow system A = I + (dt/dx^2) K H of a water disc
(one density, every face free), assembled as goo/world/flow.py assembles it, factorised as goo/world/linsolve.py's
_Direct does (SuperLU, MMD on A + A^T, diagonal pivots). Prints unknowns, fill, time, memory of the factors.

    PYTHONPATH=$PWD python bench/factor_scaling.py [max_cells]
"""
import sys, os, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
sys.path.insert(0, os.path.dirname(__file__))
import flow_mg as F
from scenes import water
import goo.world.world as WW


def system(H, W, dt=0.05):
    w = water(H, W, 0.5)
    rec = {}
    orig = WW.flow

    def grab(ctx, q, dt_):
        rec.update(ctx=ctx, q=q.copy(), dt=dt_)
        raise KeyboardInterrupt
    WW.flow = grab
    try:
        w.step(dt)
    except KeyboardInterrupt:
        pass
    finally:
        WW.flow = orig
    P = F.problem_from_ctx(rec["ctx"], rec["q"], rec["dt"])
    Hh, Ww, dx = P["H"], P["W"], P["dx"]
    M, fL, fR = P["M"], P["fL"], P["fR"]
    n = Hh * Ww
    act = np.zeros(n, bool); mf = M > 0; act[fL[mf]] = True; act[fR[mf]] = True
    cells = np.flatnonzero(act); pos = np.full(n, -1); pos[cells] = np.arange(cells.size)
    fa = np.flatnonzero(mf); iL, iR = pos[fL[fa]], pos[fR[fa]]; na = fa.size; m = cells.size
    C = sp.csr_matrix((np.r_[np.ones(na), -np.ones(na)], (np.r_[iL, iR], np.r_[np.arange(na), np.arange(na)])), shape=(m, na))
    capf = np.flatnonzero(((pos[fL] >= 0) | (pos[fR] >= 0)) & (P["gam"] > 0))
    pL, pR = pos[fL[capf]], pos[fR[capf]]; gv = P["gam"][capf] / dx ** 2
    kL, kR, kB = pL >= 0, pR >= 0, (pL >= 0) & (pR >= 0)
    Hm = sp.csr_matrix((np.r_[np.full(m, P["g1"]), gv[kL], gv[kR], -gv[kB], -gv[kB]],
                        (np.r_[np.arange(m), pL[kL], pR[kR], pL[kB], pR[kB]], np.r_[np.arange(m), pL[kL], pR[kR], pR[kB], pL[kB]])),
                       shape=(m, m))
    A = sp.identity(m, format="csr") + (P["dt"] / dx ** 2) * (C @ sp.diags(M[fa]) @ C.T @ Hm)
    return A.tocsc()


if __name__ == "__main__":
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 110000
    for H, W in ((48, 64), (96, 128), (136, 180), (192, 256), (272, 362), (384, 512)):
        if H * W * 0.5 > cap:
            break
        t = time.perf_counter(); A = system(H, W); ta = time.perf_counter() - t
        t = time.perf_counter()
        lu = splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
        tf = time.perf_counter() - t
        t = time.perf_counter(); lu.solve(np.ones(A.shape[0])); ts = time.perf_counter() - t
        nnz = lu.L.nnz + lu.U.nnz
        print(f"{H}x{W}: unknowns {A.shape[0]:7d} nnz(A) {A.nnz:8d} fill nnz(L+U) {nnz:10d} ({nnz / A.nnz:5.1f}x) "
              f"factor {tf:7.2f} s solve {ts:5.2f} s factors {nnz * 12 / 1e6:7.0f} MB (assembly {ta:.2f} s)", flush=True)
