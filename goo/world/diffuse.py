"""Substep 2: relative fluxes of the substances at fixed thickness, Onsager form (DYNAMICS.md §3, §4.2)."""
import numpy as np
import scipy.sparse as sp
from goo.world.linsolve import Factor
from goo.world.common import Reject, coeff, mixing, dF_grav, dF_cap, dF_wet, dF_exch

MAX_NEWTON = 60
LOG_STEP = 8.0       # largest fall of ln q in one iteration (a damping; e^-30 is reached in four)
LOG_RISE = 8.0       # largest rise of ln q in one iteration (a damping)
ENTRY_PECLET = 30.0  # a substance does not enter a cell where its equilibrium share would be below exp(-30) of
                     # its donor's: such amounts are not representable against the fluxes that balance them
START = 1e-3          # first Newton iterate, as a fraction of the neighbour's, for a substance entering a cell (not physics:
                      # the iteration converges to the same solution from any positive start)


def diffuse(ctx, q, dt):
    G = ctx.grid
    dx, N = G.dx, G.N
    S = q.shape[0]
    rep = dict(D_diffusion=0.0, dF=0.0, cells=np.zeros(0, np.int64))
    h = q.sum(0)
    wet = h > 0
    fL, fR = G.fL, G.fR
    rf = np.minimum(ctx.r[fL], ctx.r[fR])
    fa = np.flatnonzero(wet[fL] & wet[fR] & (rf > 0))
    if fa.size == 0:
        return q, rep
    Lam = coeff(ctx, "mobility", q, None)
    if Lam is None or not np.any(Lam[:, wet] > 0):
        return q, rep
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(wet, q / h, 0.0)
    a, b = fL[fa], fR[fa]
    om_kappa = rf[fa] * 0.5 * (h[a] + h[b]) * 0.5 * (Lam[:, a] + Lam[:, b])
    om = om_kappa * 0.5 * (phi[:, a] + phi[:, b])
    W = om.sum(0)
    k = W > 0
    if not k.any():
        return q, rep
    fa, om, W, a, b, om_kappa = fa[k], om[:, k], W[k], a[k], b[k], om_kappa[:, k]
    cells = np.unique(np.concatenate([a, b]))
    n = cells.size
    pos = np.full(N, -1); pos[cells] = np.arange(n)
    iL, iR = pos[a], pos[b]
    q0 = q[:, cells]
    hc = h[cells]
    gx, gy, gn = ctx.g
    psi = gn * (ctx.zb[cells] + 0.5 * hc) - (gx * G.X[cells] + gy * G.Y[cells])
    mx0 = mixing(ctx, q0)
    if mx0 is None:
        raise ValueError("Constitutive: mobility without mixing has no force")
    mux0 = mx0[4]
    grav = ctx.rho[:, None] * psi[None, :]
    # entry into a cell that lacks a substance (DYNAMICS.md §4.2): allowed on a face only if, at exp(-ENTRY_PECLET)
    # of the donor's fraction, the substance would still be driven in (relative to the mixture) at the old state.
    pL, pR = q0[:, iL] > 0, q0[:, iR] > 0
    one = (om > 0) & (pL != pR)
    if one.any():
        mu0 = mx0[1] + mux0 + grav
        two = (om > 0) & pL & pR
        Xn = np.where(two, -(mu0[:, iR] - mu0[:, iL]) / dx, 0.0)            # L -> R
        w2 = np.where(two, om, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            Xbar = np.where(w2.sum(0) > 0, (w2 * Xn).sum(0) / w2.sum(0), 0.0)
        fi, ff = np.nonzero(one)
        dL = pL[fi, ff]
        dcell, ecell = np.where(dL, iL[ff], iR[ff]), np.where(dL, iR[ff], iL[ff])
        probe = q0[:, ecell].copy()
        k = np.arange(fi.size)
        probe[fi, k] = np.exp(-ENTRY_PECLET) * q0[fi, dcell] / hc[dcell] * hc[ecell]
        mp = mixing(ctx, probe)[1][fi, k] + mux0[fi, ecell] + grav[fi, ecell]
        into = -(mp - mu0[fi, dcell]) / dx - np.where(dL, Xbar[ff], -Xbar[ff])
        bad = np.zeros_like(one); bad[fi, ff] = ~(into >= 0)
        om = np.where(bad, 0.0, om)
        om_kappa = np.where(bad | (om <= 0), 0.0, om_kappa)
    W = om.sum(0)
    keep = W > 0
    om, W, iL, iR = om[:, keep], W[keep], iL[keep], iR[keep]
    om_kappa = om_kappa[:, keep]
    if not keep.any():
        return q, rep
    # kappa_i,f: the face's mobility per unit volume fraction of the donor; omega_i = kappa_i phi'_i(donor)
    kap = om_kappa
    on = kap > 0
    act = np.zeros((S, n), bool)
    for i in range(S):
        act[i, iL[on[i]]] = True; act[i, iR[on[i]]] = True
    unk = np.full((S, n), -1)
    flat = np.flatnonzero(act.T.ravel())          # canonical: cell-major, then substance
    pj, pi = flat // S, flat % S
    unk[pi, pj] = np.arange(flat.size)
    nu = flat.size
    qp = q0.copy()
    new = act & (q0 <= 0)
    # first iterate for an entering substance: START of the largest fraction across its faces
    nb = np.zeros((S, n))
    for i in range(S):
        np.maximum.at(nb[i], iL[on[i]], (q0[i, iR] / hc[iR])[on[i]])
        np.maximum.at(nb[i], iR[on[i]], (q0[i, iL] / hc[iL])[on[i]])
    qp[new] = (START * nb * hc[None, :])[new]
    nf = iL.size
    ar = np.arange(nf)

    def state(qp, up=None):
        mx = mixing(ctx, qp)
        mu = mx[1] + mux0 + grav
        X = np.where(on, -(mu[:, iR] - mu[:, iL]) / dx, 0.0)
        phiL, phiR = qp[:, iL] / hc[iL], qp[:, iR] / hc[iR]
        if up is None:                              # donors from the forces with a centred guess
            w = kap * 0.5 * (phiL + phiR)
            Wg = w.sum(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                up = (X - np.where(Wg > 0, (w * X).sum(0) / Wg, 0.0)) >= 0     # True: i flows L -> R, donor L
        omg = kap * np.where(up, phiL, phiR)
        W = omg.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            Xb = np.where(W > 0, (omg * X).sum(0) / W, 0.0)
        j = omg * (X - Xb)
        # a direction changes only where the relative force is clear of rounding (where it is not, the flux is
        # rounding too, and either donor is an admissible mobility)
        fscale = np.max(np.where(on, np.abs(mu[:, iL]) + np.abs(mu[:, iR]), 0.0), axis=0, keepdims=True) / dx
        clear = np.abs(X - Xb) > 1e-12 * fscale        # the rounding of a difference of potentials of this size
        upn = np.where((omg > 0) & clear, (X - Xb) >= 0, up)
        dv = np.empty((S, n))
        for i in range(S):
            dv[i] = np.bincount(iL, j[i], n) - np.bincount(iR, j[i], n)
        return mx, X, j, dv, omg, W, Xb, up, upn

    c = dt / dx ** 2
    up = None
    lu, lu_up, prev = None, None, np.inf
    best = np.inf
    hmax = float(hc.max())
    for it in range(MAX_NEWTON):
        mx, X, j, dv, omg, W, Xb, up, upn = state(qp, up)
        R = qp - q0 + (dt / dx) * dv
        r = R[pi, pj]
        if it == 0 and not np.any(r) and np.array_equal(upn, up):
            break                                      # exactly at rest: nothing to solve
        rn = float(np.max(np.abs(r)))
        # chord Newton: keep the factors of the last Jacobian while the residual falls at least tenfold per
        # iteration and the upwind directions are those it was built with; otherwise refactor
        if lu is None or not np.array_equal(up, lu_up) or rn > 0.1 * prev:
            lu, lu_up = Factor(_jacobian(S, n, nu, unk, iL, iR, up, kap, hc, mx[2], omg, W, X, Xb, c, dt, dx), ctx.solver, ctx.stats), up.copy()
        prev = rn
        d = lu.solve(-r)
        dq = np.zeros((S, n)); dq[pi, pj] = d
        # Newton in u = ln q: the Jacobian in u is J diag(q), so du = dq / q with the same factors; amounts stay
        # positive and may fall by many decades in one iteration. A rise is capped at e^LOG_RISE per iteration.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            du = np.where(act & (qp > 0), np.clip(dq / qp, -LOG_STEP, LOG_RISE), 0.0)
        # backtracking on the residual (deterministic: halve the step up to 6 times, keep the best)
        alpha, qn_ = 1.0, None
        for _ls in range(7):
            trial = qp * np.exp(alpha * du)
            st = state(trial, up)
            rt = float(np.max(np.abs((trial - q0 + (dt / dx) * st[3])[pi, pj])))
            if qn_ is None or rt < r_best:
                qn_, r_best = trial, rt
            if rt <= (1 - 1e-4 * alpha) * rn or rn <= 1e-15 * hmax:
                break                                  # sufficient decrease, or already at the rounding of h
            alpha *= 0.5
        change = qn_ - qp
        qp = qn_
        # an entering amount whose iterate falls below exp(-ENTRY_PECLET) of the cell gets no inflow: it is not
        # represented (as the entry rule says), its faces carry none of it this step, and the system is rebuilt
        dead = act & (q0 <= 0) & (qp < np.exp(-ENTRY_PECLET) * hc[None, :])
        if dead.any():
            qp[dead] = 0.0
            for i in range(S):
                dc = np.flatnonzero(dead[i])
                if dc.size:
                    hit = np.isin(iL, dc) | np.isin(iR, dc)
                    kap[i, hit] = 0.0
            on = kap > 0
            act = np.zeros((S, n), bool)
            for i in range(S):
                act[i, iL[on[i]]] = True; act[i, iR[on[i]]] = True
            act &= (qp > 0) | (q0 > 0)
            unk = np.full((S, n), -1)
            flat = np.flatnonzero(act.T.ravel())
            pj, pi = flat // S, flat % S
            unk[pi, pj] = np.arange(flat.size)
            nu = flat.size
            lu, best = None, np.inf
            up = None
            continue
        same = np.array_equal(upn, up)
        up = upn
        if same and np.all(np.abs(change) <= 1e-12 * qp + 1e-14 * hc):
            fin = state(qp, up)
            if np.all(q0 - (dt / dx) * fin[3] >= 0):
                break                                  # converged, and the flux-form update is non-negative
        if it >= 20 and rn > 0.5 * best and rn > 1e-14 * hmax:
            raise Reject("diffusion: Newton stagnates")
        best = min(best, rn)
    else:
        raise Reject("diffusion: Newton did not converge")
    mx, X, j, dv, omg, W, Xb, up, upn = state(qp, up)
    dqf = -(dt / dx) * dv
    q1 = q0 + dqf
    if np.any(q1 < 0):
        raise Reject("diffusion: negative amount")
    qn = q.copy(); qn[:, cells] = q1
    dh = np.zeros(N); dh[cells] = dqf.sum(0)
    capf = np.flatnonzero((pos[fL] >= 0) | (pos[fR] >= 0))
    mx1 = mixing(ctx, q1)
    dF = (dx ** 2 * np.sum((mx1[0] + mx1[3]) - (mx0[0] + mx0[3])) + dF_grav(ctx, cells, q0, dqf)
          + dF_cap(ctx, capf, h, dh) + dF_wet(ctx, cells, hc, dh[cells]) + dF_exch(ctx, dqf))
    dX = X[:, None, :] - X[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        Dm = np.where(W > 0, 0.5 * np.sum(omg[:, None, :] * omg[None, :, :] * dX * dX, axis=(0, 1)) / W, 0.0)
    rep.update(D_diffusion=dt * dx ** 2 * np.sum(Dm), dF=dF, cells=cells)
    return qn, rep


def _jacobian(S, n, nu, unk, iL, iR, up, kap, hc, H, omg, W, X, Xb, c, dt, dx):
    with np.errstate(divide="ignore", invalid="ignore"):
        Lf = np.where(W > 0, omg[:, None, :] * (np.eye(S)[:, :, None] - omg[None, :, :] / W), 0.0)
        # d j_i / d omega_m, times d omega_m / d q'_m(donor) = kappa_m / h(donor)
        dJw = np.where(W > 0, np.eye(S)[:, :, None] * (X - Xb)[:, None, :] - omg[:, None, :] * (X - Xb)[None, :, :] / W, 0.0)
    dlam = kap / np.where(up, hc[iL], hc[iR])                     # [S, nf]
    don = np.where(up, iL, iR)                                     # donor cell of m on f, [S, nf]
    LHL = np.einsum("ikf,klf->ilf", Lf, H[:, :, iL])
    LHR = np.einsum("ikf,klf->ilf", Lf, H[:, :, iR])
    blocks = [(iL, iL, c * LHL), (iL, iR, -c * LHR), (iR, iL, -c * LHL), (iR, iR, c * LHR)]
    rows, cols, vals = [], [], []
    for rc, cc, blk in blocks:
        for i in range(S):
            for l in range(S):
                ru, cu = unk[i, rc], unk[l, cc]
                ok = (ru >= 0) & (cu >= 0)
                rows.append(ru[ok]); cols.append(cu[ok]); vals.append(blk[i, l][ok])
    for sgn, rc in ((1.0, iL), (-1.0, iR)):
        for i in range(S):
            for mm in range(S):
                ru, cu = unk[i, rc], unk[mm, don[mm]]
                v = sgn * (dt / dx) * dJw[i, mm] * dlam[mm]
                ok = (ru >= 0) & (cu >= 0) & (v != 0)
                rows.append(ru[ok]); cols.append(cu[ok]); vals.append(v[ok])
    J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nu, nu))
    return (J + sp.identity(nu, format="csr")).tocsc()
