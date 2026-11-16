"""The mixing free energy of _freeenergy.py as a compiled per-cell kernel (numba): the same formulas, cell by cell,
with no [S, S, N] or whole-panel temporaries unless the caller asks for the Hessian.

Each cell is evaluated on its volume fractions phi = q / h over the substances present in it (as _freeenergy.evaluate
does on its distinct compositions) and scaled by h. Two cells with the same phi and T give the same bits, whatever
their place in the array (element-wise; no threads, no reductions across cells, fastmath off). The values agree with
_freeenergy.evaluate to rounding, not bit for bit: numpy's vectorised log and its BLAS products round differently
from the compiled ones (measured and tested in tests/substance/test_mixkernel.py)."""
import math
import numpy as np
from numba import njit

_KB = 1.380649e-23        # goo.units.k_B (checked against it at import in _freeenergy's wrapper)


@njit(cache=True, fastmath=False, nogil=True, inline="always")
def _psi(x, phim):
    a = 2.5 - 1.0 / phim
    return -4 * a * phim * x - 4 * phim * (1 + a * phim) * math.log1p(-x / phim)


@njit(cache=True, fastmath=False, nogil=True, inline="always")
def _dpsi(x, phim):
    a = 2.5 - 1.0 / phim
    return 4 * (1 + a * x) / (1 - x / phim)


@njit(cache=True, fastmath=False, nogil=True, inline="always")
def _d2psi(x, phim):
    a = 2.5 - 1.0 / phim
    r = 1 - x / phim
    return 4 * a / r + 4 * (1 + a * x) / (phim * r * r)


@njit(cache=True, fastmath=False, nogil=True)
def evaluate_cells(q, kT, part, c, B, beta, phim, Pi_e, xe, A, need_h, dq, mode, rows):
    """mode 0: (f, mu, None-shaped H, fx, mux); mode 1: with H [S, S, N]; mode 2: the Hessian applied to dq [S, N];
    mode 3: the Hessian's block on the substances `rows` [R], [R, R, N] (in place of H).
    q [S, N]; kT [N]; xe, A [S, N] (each particle population's entry point and linear term at the cell's T)."""
    S, N = q.shape
    f = np.zeros(N); fx = np.zeros(N)
    mu = np.zeros((S, N)); mux = np.zeros((S, N))
    R = rows.size
    Hout = np.zeros((S, S, N)) if mode == 1 else (np.zeros((R, R, N)) if mode == 3 else np.zeros((1, 1, 1)))
    rpos = np.full(S, -1, np.int64)
    for r in range(R):
        rpos[rows[r]] = r
    hv = np.zeros((S, N)) if mode == 2 else np.zeros((1, 1))
    x = np.zeros(S); k = np.zeros(S, np.int64)
    m1 = np.zeros(S); m2 = np.zeros(S); Hc = np.zeros((S, S))
    ln = np.zeros(S); Mx = np.zeros(S); v = np.zeros(S); isliq = np.zeros(S, np.int64)
    xe1 = np.zeros(S); A1 = np.zeros(S)
    for n in range(N):
        h = 0.0
        for i in range(S):
            h += q[i, n]
        if not h > 0:
            for i in range(S):
                mu[i, n] = -np.inf if c[i] > 0 else 0.0
            continue
        K = 0
        for i in range(S):
            if q[i, n] > 0:
                p = q[i, n] / h
                if p > 0:
                    x[K] = p; k[K] = i; K += 1
        for i in range(S):
            xe1[i] = xe[i, n]; A1[i] = A[i, n]
        kTn = kT[n]
        needh = mode != 0

        fc = 0.0
        fxc = 0.0
        for i in range(S):
            m1[i] = 0.0
            m2[i] = 0.0
            isliq[i] = 0                    # here: 1 where substance i is present in the cell
        for a in range(K):
            isliq[k[a]] = 1
        if needh:
            for a in range(K):
                for b in range(K):
                    Hc[a, b] = 0.0
        hx = 0.0
        for a in range(K):
            hx += x[a]
        Lq = 0.0
        nl = 0
        for a in range(K):
            if not part[k[a]]:
                Lq += x[a]
                nl += 1
        hasL = Lq > 0
        Ls = Lq if hasL else 1.0
        # (1) the liquids' solution
        C = 0.0
        q2 = 0.0
        qMq = 0.0
        if nl > 0:
            s1 = 0.0
            first = True
            for a in range(K):
                i = k[a]
                if part[i]:
                    continue
                ln[a] = math.log(x[a] / Ls)
                C += c[i] * x[a]
                t = c[i] * x[a] * ln[a] if c[i] > 0 else 0.0
                s1 = t if first else s1 + t
                first = False
            fc += kTn * s1
            for a in range(K):
                if not part[k[a]]:
                    q2 += x[a] * x[a]
            fc += 0.5 * beta * (q2 / Ls - Lq)
            for a in range(K):
                i = k[a]
                if part[i]:
                    continue
                m = 0.0
                for b in range(K):
                    j = k[b]
                    if part[j]:
                        continue
                    m += (B[i, j] - (beta if i == j else 0.0)) * x[b]
                Mx[a] = m
            for a in range(K):
                if not part[k[a]]:
                    qMq += x[a] * Mx[a]
            fxc += 0.5 * qMq / Ls + 0.5 * beta * Lq
            for a in range(K):
                i = k[a]
                if part[i]:
                    continue
                mul = kTn * ((c[i] * (ln[a] + 1.0) if c[i] > 0 else 0.0) - C / Ls)
                mul += beta * (x[a] / Ls - 0.5 * q2 / (Ls * Ls) - 0.5)
                mxl = Mx[a] / Ls - 0.5 * qMq / (Ls * Ls) + 0.5 * beta
                m1[i] = mul if hasL else 0.0
                m2[i] = mxl if hasL else 0.0
            # absent liquids without a mixing unit (macromolecular): their finite derivatives
            for i in range(S):
                if part[i] or c[i] != 0.0 or isliq[i] == 1:
                    continue
                ba = 0.0
                for b in range(K):
                    j = k[b]
                    if not part[j]:
                        ba += B[i, j] * x[b]
                m1[i] = (kTn * (-C / Ls) + beta * (-0.5 * q2 / (Ls * Ls) - 0.5)) if hasL else 0.0
                m2[i] = (ba / Ls - 0.5 * qMq / (Ls * Ls) + 0.5 * beta) if hasL else 0.0
            if needh and hasL:
                # kTn * fh_hess(c_l, x_l, Ls) + beta * quad_hess(x_l, Ls), on the liquids' block
                for a in range(K):
                    i = k[a]
                    if part[i]:
                        continue
                    ya = x[a] / Ls
                    oa = 0.0
                    cyo = 0.0
                    y2o = 0.0
                    for b in range(K):
                        j = k[b]
                        if part[j] or b == a:
                            continue
                        yb = x[b] / Ls
                        oa += yb
                        cyo += c[j] * yb
                        y2o += yb * yb
                    fhd = ((c[i] * oa * oa / ya) if c[i] > 0 else 0.0) + cyo
                    Hc[a, a] = kTn * (fhd / Ls) + beta * ((oa * oa + y2o) / Ls)
                    for b in range(a + 1, K):
                        j = k[b]
                        if part[j]:
                            continue
                        yb = x[b] / Ls
                        ob = 0.0
                        for e in range(K):
                            if part[k[e]] or e == b:
                                continue
                            ob += x[e] / Ls
                        cyr = 0.0
                        y2r = 0.0
                        for e in range(K):
                            if part[k[e]] or e == a or e == b:
                                continue
                            ye = x[e] / Ls
                            cyr += c[k[e]] * ye
                            y2r += ye * ye
                        fho = (-c[i] * oa - c[j] * ob + cyr) / Ls
                        qo = (-ya * oa - yb * ob + y2r) / Ls
                        Hc[a, b] = kTn * fho + beta * qo
                        Hc[b, a] = Hc[a, b]
        # (2) the particles' ideal entropy, and (3) their excluded volume
        npart = K - nl
        if npart > 0:
            Np = 0.0
            sp = 0.0
            first = True
            for a in range(K):
                i = k[a]
                cp = c[i] if part[i] else 0.0
                Np += cp * x[a]
                if part[i]:
                    ln[a] = math.log(x[a] / hx)
                    t = cp * x[a] * ln[a]
                else:
                    t = 0.0
                sp = t if first else sp + t
                first = False
            fc += kTn * sp
            for a in range(K):
                i = k[a]
                m1[i] += kTn * ((c[i] * (ln[a] + 1.0) if part[i] else 0.0) - Np / hx)
            for i in range(S):
                if not part[i] and c[i] == 0.0 and isliq[i] == 0:
                    m1[i] += kTn * (-Np / hx)
            if needh:
                # kTn * fh_hess(cp, x, hx) over all K present (cp = 0 for liquids)
                for a in range(K):
                    i = k[a]
                    ya = x[a] / hx
                    oa = 0.0
                    cyo = 0.0
                    for b in range(K):
                        if b == a:
                            continue
                        yb = x[b] / hx
                        oa += yb
                        cyo += (c[k[b]] if part[k[b]] else 0.0) * yb
                    ci = c[i] if part[i] else 0.0
                    Hc[a, a] += kTn * ((((ci * oa * oa / ya) if ci > 0 else 0.0) + cyo) / hx)
                    for b in range(a + 1, K):
                        j = k[b]
                        cj = c[j] if part[j] else 0.0
                        yb = x[b] / hx
                        ob = 0.0
                        for e in range(K):
                            if e != b:
                                ob += x[e] / hx
                        cyr = 0.0
                        for e in range(K):
                            if e != a and e != b:
                                cyr += (c[k[e]] if part[k[e]] else 0.0) * (x[e] / hx)
                        w = kTn * ((-ci * oa - cj * ob + cyr) / hx)
                        Hc[a, b] += w
                        Hc[b, a] += w
            for a in range(K):
                i = k[a]
                if not part[i]:
                    continue
                pm = phim[i]
                qa = x[a]
                u = qa + Lq
                xr = qa / u
                hs = (xr < xe1[i]) and hasL
                xs = xr if hs else 0.0
                kTc = kTn * c[i]
                ps = _psi(xs, pm)
                d1 = _dpsi(xs, pm)
                Pi = kTc * xs * xs * d1 if hs else Pi_e[i]
                fc += (kTc * qa * ps - A1[i] * qa) if hs else -Pi_e[i] * Lq
                m1[i] += (kTc * (ps + xs * (1 - xs) * d1) - A1[i]) if hs else 0.0
                for b in range(K):
                    if not part[k[b]]:
                        m1[k[b]] += -Pi
                for j in range(S):
                    if not part[j] and c[j] == 0.0 and isliq[j] == 0:
                        m1[j] += -Pi
                if needh and hs:
                    aa = kTc * (2 * d1 + xs * _d2psi(xs, pm)) / u
                    for b in range(K):
                        v[b] = (Lq / u) if b == a else (-xs if not part[k[b]] else 0.0)
                    for b in range(K):
                        for e in range(K):
                            Hc[b, e] += aa * v[b] * v[e]
        # absent substances with a mixing unit: -inf
        for i in range(S):
            if c[i] > 0 and isliq[i] == 0:
                m1[i] = -np.inf
        f[n] = fc * h
        fx[n] = fxc * h
        for i in range(S):
            mu[i, n] = m1[i]
            mux[i, n] = m2[i]
        if mode == 1:
            ih = 1.0 / h
            for a in range(K):
                for b in range(K):
                    Hout[k[a], k[b], n] = Hc[a, b] * ih
        elif mode == 3:
            ih = 1.0 / h
            for a in range(K):
                ra = rpos[k[a]]
                if ra < 0:
                    continue
                for b in range(K):
                    rb = rpos[k[b]]
                    if rb >= 0:
                        Hout[ra, rb, n] = Hc[a, b] * ih
        elif mode == 2:
            for a in range(K):
                acc = 0.0
                for b in range(K):
                    acc += Hc[a, b] * dq[k[b], n]
                hv[k[a], n] = acc / h
    return f, mu, Hout, fx, mux, hv
