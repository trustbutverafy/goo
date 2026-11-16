"""Substep 3: exchange with the atmosphere and sorption into the ground, cell by cell (DYNAMICS.md §4.3)."""
import numpy as np
from goo.world.common import Reject, coeff, mixing, mixing_potentials, mixing_block, dF_grav, dF_cap, dF_wet, dF_exch, mu_wet, pore_B
from goo.world.grid import capillary_pressure
from goo.world.kernels import local_newton_update

MAX_NEWTON = 80
BISECT = 80         # halvings of a bracket: to the last bits of a double
DRAIN_EPS = 1e-12   # a cell's last amount of a substance is drained when its residual is >= 0 at this share of it


def local(ctx, q, held, dt):
    """(new q, new held, report). held [S, N] is what the pores hold."""
    G = ctx.grid
    dx, N = G.dx, G.N
    S = q.shape[0]
    rep = dict(D_exchange=0.0, D_sorption=0.0, dF=0.0, e=np.zeros(S), a=np.zeros(S), cells=np.zeros(0, np.int64), drained=0, rest=False)
    h = q.sum(0)
    wet = (h > 0) & (ctx.r > 0)
    if not wet.any():
        rep["rest"] = True
        return q, held, rep
    ex = coeff(ctx, "exchange", q, None, ctx.env)
    k = None if ex is None else ex[0]
    son = None
    if ctx.P is not None and ctx.cap > 0:
        sv = coeff(ctx, "sorption", q, None, ctx.ground)
        kap = None if sv is None else sv[1]
        if kap is not None:
            son = (kap > 0) & (q > 0) & wet
    eon = None
    if k is not None:
        eon = (k > 0) & (q > 0) & wet & ctx.has_env[:, None]
    if eon is None and son is None:
        rep["rest"] = True
        return q, held, rep
    eon = np.zeros((S, N), bool) if eon is None else eon
    son = np.zeros((S, N), bool) if son is None else son
    cells = np.flatnonzero((eon | son).any(0))
    if cells.size == 0:
        rep["rest"] = True
        return q, held, rep
    n = cells.size
    gx, gy, gn = ctx.g
    a_ = ctx.a
    rho = ctx.rho
    q0 = q[:, cells]
    h0, m0 = q0.sum(0), rho @ q0
    rc = ctx.r[cells]
    ke = np.where(eon[:, cells], dt * rc * (k[:, cells] if k is not None else 0.0), 0.0)        # dt r k  [S, n]
    ks = np.where(son[:, cells], dt * rc * (kap[:, cells] if son.any() else 0.0), 0.0)
    Eon, Son = eon[:, cells], son[:, cells]
    s = ctx.zb + h
    Pc = capillary_pressure(G, ctx.gam_f, s)[cells]
    sig = G.cellsum(ctx.gam_f)[cells] / dx ** 2
    mw = mu_wet(ctx, h0, cells)
    psi = -(gx * G.X[cells] + gy * G.Y[cells])
    zb = ctx.zb[cells]
    gexp = -gn * (m0 - a_ * h0)[None, :] * (rho[:, None] - a_) / (4 * a_)
    base = (gn * rho[:, None] * zb[None, :] + gexp + (Pc - sig * h0 + mw)[None, :] + rho[:, None] * psi[None, :])
    mx0 = mixing_potentials(ctx, q0)          # its Hessian is never read: the Newton's cells get theirs below
    nonmix = mx0 is None
    if nonmix:
        mx0 = (np.zeros(n), np.zeros((S, n)), np.zeros((S, S, n)), np.zeros(n), np.zeros((S, n)))
    mux0 = mx0[4]
    single = (q0 > 0).sum(0) == 1
    gJ = gn * np.outer(rho + a_, rho + a_) / (4 * a_)                                  # [S, S]
    menv = ctx.mu_env
    w0 = held[:, cells].sum(0)
    C = ctx.cap
    P = ctx.P[:, cells] if ctx.P is not None else np.zeros((S, n))
    Pb = ctx.Pb[cells] if ctx.P is not None else np.zeros(n)

    def potentials(qp, idx, need_hess=True):
        """mu [S, m] at qp [S, m] for cells idx (positions in 0..n-1), and the mixing Hessian [S, S, m]: the Jacobian
        of mu is that plus gJ + sig (jac())."""
        sg = single[idx]
        Hm = None
        if not nonmix:
            mm = ~sg
            if mm.all():
                # the Newton needs the Hessian only on the substances that exchange or sorb (hrows)
                mxx = mixing_block(ctx, qp, hrows) if need_hess else mixing_potentials(ctx, qp)
                mu_c, Hm = mxx[1], (mxx[2] if need_hess else None)
            else:
                mu_c = mx0[1][:, idx].copy()
                if need_hess:
                    raise AssertionError("local: the Newton's cells are never single-substance cells")
                if mm.any():
                    mxx = mixing_potentials(ctx, qp[:, mm])
                    mu_c[:, mm] = mxx[1]
        else:
            mu_c = np.zeros_like(qp)
        if Hm is None and need_hess:
            Hm = np.zeros((hrows.size, hrows.size, idx.size))
        hp, mp = qp.sum(0), rho @ qp
        mul = (mu_c + mux0[:, idx] + base[:, idx] + gn * (mp + a_ * hp)[None, :] * (rho[:, None] + a_) / (4 * a_)
               + sig[idx][None, :] * hp[None, :])
        return mul, Hm

    def jac(Hm, idx, rows, cols):
        """The Jacobian of mu, d mu_rows / d q_cols [m, len(rows), len(cols)]."""
        return np.moveaxis(Hm[rows][:, cols], 2, 0) + gJ[np.ix_(rows, cols)][None] + sig[idx][:, None, None]

    def pore(av, idx):
        th = (w0[idx] + av.sum(0)) / C if C > 0 else np.zeros(idx.size)
        mup = -P[:, idx] + (Pb[idx] * th / (1 - th))[None, :]
        Jp = (Pb[idx] / (C * (1 - th) ** 2)) if C > 0 else np.zeros(idx.size)
        return th, mup, Jp

    e = np.zeros((S, n)); av = np.zeros((S, n))
    # A cell of several substances in which one substance alone exchanges or is sorbed, and the step would take all of
    # it (a particle film's last water: its potential is bounded below by the menisci's -Pi_e, far above the air's):
    # with the other amounts fixed the residual q_i' - q0_i + e_i(q_i') + a_i(q_i') is increasing in q_i' (as for a
    # single-substance cell), so if it is >= 0 at q_i' = DRAIN_EPS q0_i the root lies within that of zero and the
    # constrained solution is the drain q_i' = 0 (the KKT point of q_i >= 0), split between e and a below like a
    # single-substance cell's (DYNAMICS.md §4.3). The Newton would only creep toward 0 by its 0.9 limiter.
    movers = Eon | Son
    dsub = np.argmax(movers, axis=0)
    mdrain = np.zeros(n, bool)
    cand = np.flatnonzero(~single & (movers.sum(0) == 1))
    if cand.size:
        ic = dsub[cand]; kc = np.arange(cand.size)
        T = q0[ic, cand]
        # mu_i is increasing in q_i: at q0 it bounds what the step can take from above (the potentials at q0, whose
        # mixing part is mx0's: no map call)
        mu_hi = potentials(q0, np.arange(n), need_hess=False)[0][ic, cand]
        th0 = w0[cand] / C if C > 0 else np.zeros(cand.size)
        mup0 = -P[ic, cand] + Pb[cand] * th0 / (1 - th0)
        upper = (np.where(Eon[ic, cand], np.maximum(ke[ic, cand] * (mu_hi - menv[ic]), 0.0), 0.0)
                 + np.where(Son[ic, cand], np.maximum(ks[ic, cand] * (mu_hi - mup0), 0.0), 0.0))
        pre = upper >= T
        if pre.any():
            c2, i2, k2 = cand[pre], ic[pre], kc[pre]
            T2 = T[pre]
            qe = q0[:, c2].copy()
            qe[i2, np.arange(c2.size)] = DRAIN_EPS * T2
            mu_e = potentials(qe, c2, need_hess=False)[0][i2, np.arange(c2.size)]
            ee0 = np.where(Eon[i2, c2], ke[i2, c2] * (mu_e - menv[i2]), 0.0)
            aa0 = np.zeros(c2.size)
            if C > 0 and np.any(Son[i2, c2]):
                lo_, hi_ = np.zeros(c2.size), (C - w0[c2]) * (1 - 1e-12)
                for _ in range(BISECT):
                    mid = 0.5 * (lo_ + hi_)
                    thm = (w0[c2] + mid) / C
                    gm = mid - ks[i2, c2] * Son[i2, c2] * np.maximum(mu_e - (-P[i2, c2] + Pb[c2] * thm / (1 - thm)), 0.0)
                    lo_, hi_ = np.where(gm < 0, mid, lo_), np.where(gm < 0, hi_, mid)
                aa0 = lo_
            mdrain[c2] = DRAIN_EPS * T2 - T2 + ee0 + aa0 >= 0
    live = np.flatnonzero(~single & ~mdrain)        # single-substance cells are bracketed below; drained ones too
    # the unknowns: e_i for the substances that exchange somewhere in the solve, a_i for those that are sorbed
    # somewhere (every other e_i, a_i is identically zero, a row of the identity, and is left out of the system)
    ie = np.flatnonzero(Eon[:, live].any(1)) if live.size else np.zeros(0, np.int64)
    ia = np.flatnonzero(Son[:, live].any(1)) if live.size else np.zeros(0, np.int64)
    ne, K = ie.size, ie.size + ia.size
    if K == 0:
        live = live[:0]
    IK = np.eye(K)
    hrows = np.unique(np.concatenate([ie, ia])).astype(np.int64)    # the Hessian's rows the Newton reads
    hpos = np.full(S, -1, np.int64); hpos[hrows] = np.arange(hrows.size)
    elo, ehi = np.full(n, -np.inf), np.full(n, np.inf)      # the scalar Newton's brackets (kernels.py)
    gJ = np.asarray(gJ)
    for it in range(MAX_NEWTON):
        if live.size == 0:
            break
        idx = live
        qp = q0[:, idx] - e[:, idx] - av[:, idx]
        mu, Hm = potentials(qp, idx)
        # the Newton update, per cell, compiled (goo/world/kernels.py): the residual, the Jacobian, the limited step
        done = local_newton_update(idx, np.ascontiguousarray(mu), np.ascontiguousarray(Hm), e, av, Eon, Son, ke, ks,
                                   menv, P, Pb, w0, float(C), sig, gJ, ie, ia, single, q0, h0, elo, ehi, hpos)
        live = idx[~done]
    else:
        raise Reject("local: Newton did not converge")
    # single-substance cells: the residual q' - q0 + e(q') + a(q') is increasing in q' (the film's potential rises
    # with q' under the convex split) and a(q') solves an increasing equation at fixed q', so both are bracketed and
    # bisected (a fixed number of halvings, element-wise): the root, or the drain h' = 0 when the residual at 0 is
    # still >= 0 (then the KKT split below applies)
    sidx = np.flatnonzero(single)
    drain = np.zeros(n, bool)
    if sidx.size:
        isub = np.argmax(q0[:, sidx] > 0, axis=0)
        cols = np.arange(sidx.size)
        T0 = q0[isub, sidx]
        kE, kA = ke[isub, sidx] * Eon[isub, sidx], ks[isub, sidx] * Son[isub, sidx]

        def mu_at(qv):
            qq = np.zeros((S, sidx.size)); qq[isub, cols] = qv
            return potentials(qq, sidx, need_hess=False)[0][isub, cols]

        def a_at(muv):
            if not np.any(kA > 0) or C <= 0:
                return np.zeros(sidx.size)
            lo, hi = np.zeros(sidx.size), np.full(sidx.size, (C - w0[sidx]) * (1 - 1e-12))
            for _ in range(BISECT):
                mid = 0.5 * (lo + hi)
                th = (w0[sidx] + mid) / C
                g = mid - kA * np.maximum(muv - (-P[isub, sidx] + Pb[sidx] * th / (1 - th)), 0.0)
                lo, hi = np.where(g < 0, mid, lo), np.where(g < 0, hi, mid)
            return lo

        def resid(qv):
            muv = mu_at(qv)
            ev = kE * (muv - menv[isub])
            av_ = a_at(muv)
            return qv - T0 + ev + av_, ev, av_
        r0, e0, a0 = resid(np.zeros(sidx.size))
        drain[sidx] = r0 >= 0
        lo = np.zeros(sidx.size)
        hi = T0.copy()
        for _ in range(60):                                   # condensation can raise q' above q0: widen the bracket
            rh = resid(hi)[0]
            grow = (rh < 0) & ~drain[sidx]
            if not grow.any():
                break
            hi = np.where(grow, 2 * hi + T0, hi)
        for _ in range(BISECT):
            mid = 0.5 * (lo + hi)
            rm = resid(mid)[0]
            lo, hi = np.where(rm < 0, mid, lo), np.where(rm < 0, hi, mid)
        qs = np.where(drain[sidx], 0.0, hi)
        _, es, as_ = resid(qs)
        e[isub, sidx] = np.where(drain[sidx], 0.0, es)
        av[isub, sidx] = np.where(drain[sidx], 0.0, as_)
        # the update is q' = q0 - e - a, exactly in flux form
    q1 = q0 - e - av
    drain = (drain & single) | mdrain
    if drain.any():
        for j in np.flatnonzero(drain):
            i = int(np.flatnonzero(q0[:, j] > 0)[0]) if single[j] else int(dsub[j])
            T = q0[i, j]
            ek, sk = Eon[i, j], Son[i, j]
            if ek and not sk:
                ee, aa = T, 0.0
            elif sk and not ek:
                ee, aa = 0.0, T
            else:
                def g(x):
                    th = (w0[j] + x) / C
                    return (T - x) / ke[i, j] + menv[i] - x / ks[i, j] - (-P[i, j] + Pb[j] * th / (1 - th))
                if g(0.0) <= 0:
                    aa = 0.0
                elif g(T) >= 0:
                    aa = T
                else:
                    lo, hi = 0.0, T
                    for _ in range(200):
                        mid = 0.5 * (lo + hi)
                        if mid == lo or mid == hi:
                            break
                        lo, hi = (mid, hi) if g(mid) > 0 else (lo, mid)
                    aa = lo
                ee = T - aa
            xi = ee / ke[i, j] + menv[i] if ee > 0 else aa / ks[i, j] + (-P[i, j] + Pb[j] * ((w0[j] + aa) / C) / (1 - (w0[j] + aa) / C))
            qz = np.zeros((S, 1))
            if not single[j]:
                qz[:, 0] = q0[:, j]; qz[i, 0] = DRAIN_EPS * T
            mu00, _ = potentials(qz, np.array([j]), need_hess=False)
            if xi > mu00[i, 0] + 1e-9 * abs(mu00[i, 0]) + 1e-300:
                raise Reject("local: drained cell violates its KKT condition")
            e[:, j] = 0.0; av[:, j] = 0.0
            e[i, j], av[i, j] = ee, aa
            if single[j]:
                q1[:, j] = 0.0
            else:
                q1[:, j] = q0[:, j]; q1[i, j] = 0.0          # the others do not move: exactly what they held
        q1[:, ~drain] = (q0 - e - av)[:, ~drain]
    # e < 0 is condensation (the film's potential is below the reservoir's: a gum-rich film takes water from the
    # air); exchange is two-way (Psi* = r k (mu - mu°)^2 / 2), so only amounts and sorption (one-way) must be >= 0
    if np.any(q1 < 0) or np.any(av < 0):
        raise Reject("local: negative amount")
    qn = q.copy(); qn[:, cells] = q1
    heldn = held.copy(); heldn[:, cells] = held[:, cells] + av
    dq = q1 - q0
    dh = np.zeros(N); dh[cells] = dq.sum(0)
    capf = np.flatnonzero(np.isin(G.fL, cells) | np.isin(G.fR, cells))
    dF = dF_grav(ctx, cells, q0, dq) + dF_cap(ctx, capf, h, dh) + dF_wet(ctx, cells, h0, dh[cells]) + dF_exch(ctx, dq)
    if not nonmix:
        mx1 = mixing_potentials(ctx, q1)
        dF += dx ** 2 * np.sum((mx1[0] + mx1[3]) - (mx0[0] + mx0[3]))
    dF += -dx ** 2 * np.sum(ctx.mu_env[:, None] * av)          # the pores' matter is counted against its reservoir
    if ctx.P is not None and C > 0 and av.any():
        th0, th1 = w0 / C, (w0 + av.sum(0)) / C
        dB = -(th1 - th0) - np.log1p(-(th1 - th0) / (1 - th0))
        dF += dx ** 2 * np.sum(-(P * av).sum(0) + Pb * C * dB)
    with np.errstate(divide="ignore", invalid="ignore"):
        De = np.where(Eon, e * e / ke, 0.0)
        Ds = np.where(Son & (av > 0), av * av / ks, 0.0)
    rep.update(D_exchange=dx ** 2 * np.sum(De), D_sorption=dx ** 2 * np.sum(Ds), dF=dF, drained=int(drain.sum()),
               e=e.sum(1) * dx ** 2, a=av.sum(1) * dx ** 2, cells=cells)
    return qn, heldn, rep
