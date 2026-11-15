"""Substep 2: relative fluxes of the substances at fixed thickness, Onsager form (DYNAMICS.md §3, §4.2)."""
import numpy as np
import scipy.sparse as sp
from goo.world.linsolve import Inner
from goo.world.common import Reject, coeff, mixing, dF_grav, dF_cap, dF_wet, dF_exch

MAX_NEWTON = 60
LOG_STEP = 8.0       # largest fall of ln q in one iteration (a damping; e^-30 is reached in four)
LOG_RISE = 8.0       # largest rise of ln q in one iteration (a damping)
ENTRY_PECLET = 30.0  # a substance does not enter a cell where its equilibrium share would be below exp(-30) of
                     # its donor's: such amounts are not representable against the fluxes that balance them
START = 1e-3          # first Newton iterate, as a fraction of the neighbour's, for a substance entering a cell (not physics:
                      # the iteration converges to the same solution from any positive start)


def floors(ctx):
    """[S] the amount of one molecule (or particle) of each substance per cell, v_i / dx^2 (0 for a phase with no
    molecular volume, which never diffuses)."""
    v = np.array([float(getattr(s, "molecular_volume", np.inf)) for s in ctx.substances])
    return np.where(np.isfinite(v) & (v > 0), v, 0.0) / ctx.grid.dx ** 2


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
    # R7 (sub-molecular amounts): a substance holding less than one of its molecules (or particles) in a cell,
    # 0 < q_i < v_i / dx^2, is carried exactly but does not diffuse: every face touching that cell carries none of it
    # this step (a zero mobility, which the energy argument allows, DYNAMICS.md §4.2). Above the floor nothing changes.
    qfl = floors(ctx)
    sub = (q > 0) & (q < qfl[:, None])
    if sub.any():
        om_kappa = np.where(sub[:, a] | sub[:, b], 0.0, om_kappa)
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
    # kappa_i,f: the face's mobility per unit volume fraction of the donor; omega_i = kappa_i phi'_i(donor).
    # Only substances that move on some face are unknowns: the others carry no flux (omega = 0 removes their rows and
    # columns of L), so the system is assembled on the moving substances sa alone and the rest pass through.
    sa = np.flatnonzero((om_kappa > 0).any(1))
    Sa = sa.size
    kap = om_kappa[sa]
    on = kap > 0
    q0a, mux0a, grava = q0[sa], mux0[sa], grav[sa]
    qfa = qfl[sa]

    def unknowns(on, extra=None):
        act = np.zeros((Sa, n), bool)
        for i in range(Sa):
            act[i, iL[on[i]]] = True; act[i, iR[on[i]]] = True
        if extra is not None:
            act &= extra
        unk = np.full((Sa, n), -1, np.int64)
        flat = np.flatnonzero(act.T.ravel())          # canonical: cell-major, then substance
        pj, pi = flat // Sa, flat % Sa
        unk[pi, pj] = np.arange(flat.size)
        return act, unk, pj, pi
    act, unk, pj, pi = unknowns(on)
    qp = q0a.copy()
    new = act & (q0a <= 0)
    # first iterate for an entering substance: START of the largest fraction across its faces
    nb = np.zeros((Sa, n))
    for i in range(Sa):
        np.maximum.at(nb[i], iL[on[i]], (q0a[i, iR] / hc[iR])[on[i]])
        np.maximum.at(nb[i], iR[on[i]], (q0a[i, iL] / hc[iL])[on[i]])
    qp[new] = (START * nb * hc[None, :])[new]
    qfull = q0.copy()

    def state(qp, up=None):
        qfull[sa] = qp
        mx = mixing(ctx, qfull)
        mu = mx[1][sa] + mux0a + grava
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
        dv = np.empty((Sa, n)); gr = np.empty((Sa, n))
        for i in range(Sa):
            dv[i] = np.bincount(iL, j[i], n) - np.bincount(iR, j[i], n)
            # the gross amount the fluxes move through a cell, and the flux a force of the potentials' own size would
            # carry (1e-12 of which is the rounding of a difference of potentials, the scale the upwind rule uses)
            aj = np.abs(j[i]) + omg[i] * (np.abs(mu[i, iL]) + np.abs(mu[i, iR])) / dx
            gr[i] = (dt / dx) * (np.bincount(iL, aj, n) + np.bincount(iR, aj, n))
        H = mx[2][sa][:, sa]
        return H, X, j, dv, omg, W, Xb, up, upn, gr

    c = dt / dx ** 2
    up = None
    lu, lu_up, prev = None, None, np.inf
    best = np.inf
    hmax = float(hc.max())
    cur = None                                        # state(qp, up), when already known
    for it in range(MAX_NEWTON):
        if cur is None:
            cur = state(qp, up)
        Hs, X, j, dv, omg, W, Xb, up, upn, gr = cur
        R = qp - q0a + (dt / dx) * dv
        r = R[pi, pj]
        if it == 0 and not np.any(r) and np.array_equal(upn, up):
            break                                      # exactly at rest: nothing to solve
        rn = float(np.max(np.abs(r)))
        # chord Newton: keep the last Jacobian (and its preconditioner) while the residual falls at least tenfold per
        # iteration and the upwind directions are those it was built with; otherwise rebuild
        if lu is None or not np.array_equal(up, lu_up) or rn > 0.1 * prev:
            ucell = cells[pj]
            Hj = Hs
            low = act & (qp < qfa[:, None])
            if low.any():
                # an iterate below one molecule enters the Jacobian (only the Newton path, not the answer, which is
                # judged on the residual) at the floor, where the mixing Hessian (~ 1/q) is finite
                qfull[sa] = np.where(low, qfa[:, None], qp)
                Hj = mixing(ctx, qfull)[2][sa][:, sa]
            Jm = _jacobian(Sa, n, nu_of(pj), unk, iL, iR, up, kap, hc, Hj, omg, W, X, Xb, c, dt, dx)
            if not np.all(np.isfinite(Jm.data)):
                raise Reject("diffusion: non-finite Jacobian")
            lu, lu_up = Inner(Jm, G.IY[ucell], G.IX[ucell], pi, ctx.stats), up.copy()
        prev = rn
        d = lu.solve(-r)
        dq = np.zeros((Sa, n)); dq[pi, pj] = d
        # Newton in u = ln q: the Jacobian in u is J diag(q), so du = dq / q with the same factors; amounts stay
        # positive and may fall by many decades in one iteration. A rise is capped at e^LOG_RISE per iteration.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            du = np.where(act & (qp > 0), np.clip(dq / qp, -LOG_STEP, LOG_RISE), 0.0)
        # backtracking on the residual (deterministic: halve the step up to 6 times, keep the best)
        alpha, qn_ = 1.0, None
        for _ls in range(7):
            trial = qp * np.exp(alpha * du)
            st = state(trial, up)
            rt = float(np.max(np.abs((trial - q0a + (dt / dx) * st[3])[pi, pj])))
            if qn_ is None or rt < r_best:
                qn_, r_best, st_best = trial, rt, st
            if rt <= (1 - 1e-4 * alpha) * rn or rn <= 1e-15 * hmax:
                break                                  # sufficient decrease, or already at the rounding of h
            alpha *= 0.5
        change = qn_ - qp
        qp = qn_
        cur = None
        # an entering amount whose iterate falls below exp(-ENTRY_PECLET) of the cell gets no inflow: it is not
        # represented (as the entry rule says), its faces carry none of it this step, and the system is rebuilt
        dead = act & (q0a <= 0) & (qp < np.exp(-ENTRY_PECLET) * hc[None, :])
        if dead.any():
            qp[dead] = 0.0
            for i in range(Sa):
                dc = np.flatnonzero(dead[i])
                if dc.size:
                    hit = np.isin(iL, dc) | np.isin(iR, dc)
                    kap[i, hit] = 0.0
            on = kap > 0
            act, unk, pj, pi = unknowns(on, (qp > 0) | (q0a > 0))
            lu, best = None, np.inf
            up = None
            continue
        same = np.array_equal(upn, up)
        up = upn
        if same:
            cur = st_best                              # the accepted trial's state, with the same directions
        # rounding of a pair's update: of its amount, of its cell's thickness, and of the gross amount the fluxes
        # carry through it (the flux-form update is a sum of those terms and is not known better)
        rtol_ = 1e-12 * qp + 1e-14 * hc + 1e-12 * (gr if cur is None else cur[9])
        if same and np.all(np.abs(change) <= rtol_):
            Rf = (qp - q0a + (dt / dx) * cur[3])[pi, pj]
            # converged: the update is below rounding, the iterate's own residual is (a guard against an inner solve
            # that stalled, which a small update alone would not reveal), and the flux-form update is non-negative
            if (np.all(np.abs(Rf) <= 100 * rtol_[pi, pj]) and np.all(q0a - (dt / dx) * cur[3] >= 0)):
                break
        if it >= 20 and rn > 0.5 * best and rn > 1e-14 * hmax:
            raise Reject("diffusion: Newton stagnates")
        best = min(best, rn)
    else:
        raise Reject("diffusion: Newton did not converge")
    if cur is None:
        cur = state(qp, up)
    Hs, X, j, dv, omg, W, Xb, up, upn, gr = cur
    dqf = np.zeros((S, n))
    dqf[sa] = -(dt / dx) * dv
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


def nu_of(pj):
    return pj.size


def _jacobian(S, n, nu, unk, iL, iR, up, kap, hc, H, omg, W, X, Xb, c, dt, dx):
    """J = I + d(dt/dx div j)/dq' on the unknowns. Each face f couples its cells L and R through two S x S blocks,
    V_L (columns at L) and V_R (columns at R): rows at L get +V, rows at R get -V. Cell-diagonal blocks are summed per
    cell before assembly, so every entry is written once."""
    with np.errstate(divide="ignore", invalid="ignore"):
        Wp = W > 0
        Lf = np.where(Wp, omg[:, None, :] * (np.eye(S)[:, :, None] - omg[None, :, :] / W), 0.0)
        # d j_i / d omega_m, times d omega_m / d q'_m(donor) = kappa_m / h(donor)
        Xr = X - Xb
        dJw = np.where(Wp, np.eye(S)[:, :, None] * Xr[:, None, :] - omg[:, None, :] * Xr[None, :, :] / W, 0.0)
    mob = (dt / dx) * dJw * (kap / np.where(up, hc[iL], hc[iR]))[None, :, :]
    VL = np.where(up[None, :, :], mob, 0.0)
    VR = np.where(up[None, :, :], 0.0, mob)
    for k in range(S):                                 # (L_f H)_il = sum_k L_ik H_kl, at the face's two cells
        for l in range(S):
            HL, HR = c * H[k, l, iL], c * H[k, l, iR]
            for i in range(S):
                VL[i, l] += Lf[i, k] * HL
                VR[i, l] -= Lf[i, k] * HR
    # cell-diagonal blocks: +V_L at L, -V_R at R
    Dg = np.zeros((S, S, n))
    for i in range(S):
        for l in range(S):
            Dg[i, l] = np.bincount(iL, VL[i, l], n) - np.bincount(iR, VR[i, l], n)
    rows, cols, vals = [], [], []
    cellar = np.arange(n)
    for i in range(S):
        for l in range(S):
            ru, cu = unk[i], unk[l]
            ok = (ru >= 0) & (cu >= 0)
            rows.append(ru[ok]); cols.append(cu[ok]); vals.append(Dg[i, l][ok] + (1.0 if i == l else 0.0))
            for rc, cc, v in ((iL, iR, VR[i, l]), (iR, iL, -VL[i, l])):
                ru, cu = unk[i, rc], unk[l, cc]
                ok = (ru >= 0) & (cu >= 0)
                rows.append(ru[ok]); cols.append(cu[ok]); vals.append(v[ok])
    del cellar
    J = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows).astype(np.int32), np.concatenate(cols).astype(np.int32))),
                      shape=(nu, nu))
    return J.tocsr()
