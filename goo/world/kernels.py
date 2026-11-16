"""Compiled per-cell kernels of the step (numba). Element-wise over cells: a cell's arithmetic depends on that cell
alone, so a window and the panel give the same bits (O5). No threads, no reductions across cells, fastmath off (no
reassociation, no contraction into fused multiply-adds), compilation cached on disk (DYNAMICS.md §9a).

The maps (goo.substance.maps) are not compiled here: the world reaches substances through them only (SPEC §4.4), so a
kernel receives what the maps returned for its cells."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, nogil=True)
def local_newton_update(idx, mu, Hm, e, av, Eon, Son, ke, ks, menv, P, Pb, w0, C, sig, gJ, ie, ia, single, q0, h0,
                        elo, ehi, hpos):
    """One Newton iteration of the local step (goo/world/local.py) on the live cells idx, in place on e, av [S, n].
    mu [S, m] the film's potentials and Hm [R, R, m] the mixing Hessian's block on the substances that exchange or sorb at
    the iterate (m = idx.size; hpos [S] maps a substance to its row of the block). The unknowns are
    e_i (i in ie) and a_i (i in ia); the residual, the Jacobian, the step limited to 0.9 of what is held and to 0.9 of
    the pores' free space, and the stopping rule are local.py's. The K x K system is solved by Gaussian elimination with
    partial pivoting in a fixed order (K is 1 for one volatile and no pores: the scalar formula of local.py). Returns
    done [m].

    Safeguard (K = 1): the residual e - ke (mu(q0 - e) - mu_env) is increasing in e (mu increases with the amount), so
    each evaluation brackets the root: elo, ehi [n] (start -inf, +inf) keep the bracket, and a Newton step that leaves it
    is replaced by its midpoint. Where Newton stays inside (every cell of the painter's scenes before the gum's glass
    transition) the iterates are unchanged; where it does not (observed: a two-cycle at a vitrifying gum film, whose
    water potential bends sharply) it converges."""
    S = mu.shape[0]
    m = idx.size
    ne = ie.size
    K = ne + ia.size
    done = np.zeros(m, np.bool_)
    J = np.zeros((K, K))
    R = np.zeros(K)
    d = np.zeros(K)
    kk = np.zeros(K)
    piv = np.zeros(K)
    de = np.zeros(S)
    da = np.zeros(S)
    for c in range(m):
        j = idx[c]
        # pores
        th = 0.0
        Jp = 0.0
        if C > 0:
            sa = 0.0
            for i in range(S):
                sa += av[i, j]
            th = (w0[j] + sa) / C
            Jp = Pb[j] / (C * (1 - th) ** 2)
        for r in range(K):
            i = ie[r] if r < ne else ia[r - ne]
            if r < ne:
                Xe = mu[i, c] - menv[i]
                if Eon[i, j]:
                    R[r] = e[i, j] - ke[i, j] * Xe
                    kk[r] = ke[i, j]
                else:
                    R[r] = e[i, j]
                    kk[r] = 0.0
            else:
                mup = -P[i, j] + Pb[j] * th / (1 - th)
                Xa = mu[i, c] - mup
                pa = Son[i, j] and Xa > 0
                if Son[i, j]:
                    R[r] = av[i, j] - ks[i, j] * (Xa if pa else 0.0)
                else:
                    R[r] = av[i, j]
                kk[r] = ks[i, j] if pa else 0.0
        if K == 1 and ne == 1:
            i0 = ie[0]
            if R[0] < 0.0 and e[i0, j] > elo[j]:
                elo[j] = e[i0, j]
            elif R[0] > 0.0 and e[i0, j] < ehi[j]:
                ehi[j] = e[i0, j]
            Jd = kk[0] * (Hm[hpos[i0], hpos[i0], c] + gJ[i0, i0] + sig[j]) + 1.0
            d[0] = -R[0] / Jd
        else:
            for r in range(K):
                ir = ie[r] if r < ne else ia[r - ne]
                for s in range(K):
                    is_ = ie[s] if s < ne else ia[s - ne]
                    J[r, s] = kk[r] * (Hm[hpos[ir], hpos[is_], c] + gJ[ir, is_] + sig[j])
                    if r >= ne and s >= ne:
                        J[r, s] += kk[r] * Jp
                    if r == s:
                        J[r, s] += 1.0
                d[r] = -R[r]
            # Gaussian elimination with partial pivoting, fixed order
            for col in range(K):
                p = col
                best = abs(J[col, col])
                for r in range(col + 1, K):
                    if abs(J[r, col]) > best:
                        best = abs(J[r, col]); p = r
                if p != col:
                    for s in range(K):
                        tmp = J[col, s]; J[col, s] = J[p, s]; J[p, s] = tmp
                    tmp = d[col]; d[col] = d[p]; d[p] = tmp
                for r in range(col + 1, K):
                    f = J[r, col] / J[col, col]
                    for s in range(col, K):
                        J[r, s] -= f * J[col, s]
                    d[r] -= f * d[col]
            for r in range(K - 1, -1, -1):
                acc = d[r]
                for s in range(r + 1, K):
                    acc -= J[r, s] * d[s]
                d[r] = acc / J[r, r]
        for i in range(S):
            de[i] = 0.0
            da[i] = 0.0
        for r in range(K):
            if r < ne:
                de[ie[r]] = d[r]
            else:
                da[ia[r - ne]] = d[r]
        t = 1.0
        lim = np.inf
        if not single[j]:
            for i in range(S):
                dq = -(de[i] + da[i])
                qp = q0[i, j] - e[i, j] - av[i, j]
                if dq < 0 and qp > 0:
                    v = 0.9 * qp / -dq
                    if v < lim:
                        lim = v
        lim2 = np.inf
        if C > 0:
            sda = 0.0
            for i in range(S):
                sda += da[i]
            dth = sda / C
            if dth > 0:
                lim2 = 0.9 * (1 - th) / dth
        if lim < t:
            t = lim
        if lim2 < t:
            t = lim2
        if K == 1 and ne == 1 and np.isfinite(elo[j]) and np.isfinite(ehi[j]):
            i0 = ie[0]
            en = e[i0, j] + t * d[0]
            if not (elo[j] < en < ehi[j]):                 # the step left the bracket: bisect it
                d[0] = 0.5 * (elo[j] + ehi[j]) - e[i0, j]
                t = 1.0
                de[i0] = d[0]
                if not (ehi[j] - elo[j] > 1e-13 * h0[j]):
                    d[0] = 0.0; de[i0] = 0.5 * (elo[j] + ehi[j]) - e[i0, j]
        for i in range(S):
            e[i, j] += t * de[i]
            av[i, j] += t * da[i]
        ok = t == 1.0
        for r in range(K):
            if not (abs(d[r]) <= 1e-13 * h0[j] + 1e-300):
                ok = False
        done[c] = ok
    return done
