"""Substep 3: exchange with the atmosphere and sorption into the ground, cell by cell (DYNAMICS.md §4.3)."""
import numpy as np
from goo.world.common import Reject, coeff, mixing, dF_grav, dF_cap, dF_wet, dF_exch, mu_wet, pore_B
from goo.world.grid import capillary_pressure

MAX_NEWTON = 80
BISECT = 80         # halvings of a bracket: to the last bits of a double


def local(ctx, q, held, dt):
    """(new q, new held, report). held [S, N] is what the pores hold."""
    G = ctx.grid
    dx, N = G.dx, G.N
    S = q.shape[0]
    rep = dict(D_exchange=0.0, D_sorption=0.0, dF=0.0, e=np.zeros(S), a=np.zeros(S), cells=np.zeros(0, np.int64))
    h = q.sum(0)
    wet = (h > 0) & (ctx.r > 0)
    if not wet.any():
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
        return q, held, rep
    eon = np.zeros((S, N), bool) if eon is None else eon
    son = np.zeros((S, N), bool) if son is None else son
    cells = np.flatnonzero((eon | son).any(0))
    if cells.size == 0:
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
    mx0 = mixing(ctx, q0)
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

    def potentials(qp, idx):
        """mu [S, m] and its Jacobian [m, S, S] at qp [S, m] for cells idx (positions in 0..n-1)."""
        sg, mul = single[idx], np.empty_like(qp)
        Hm = np.zeros((S, S, idx.size))
        if not nonmix:
            mu_c = mx0[1][:, idx].copy(); Hm = mx0[2][:, :, idx].copy()
            mm = ~sg
            if mm.any():
                mxx = mixing(ctx, qp[:, mm])
                mu_c[:, mm] = mxx[1]; Hm[:, :, mm] = mxx[2]
        else:
            mu_c = np.zeros_like(qp)
        hp, mp = qp.sum(0), rho @ qp
        mul = (mu_c + mux0[:, idx] + base[:, idx] + gn * (mp + a_ * hp)[None, :] * (rho[:, None] + a_) / (4 * a_)
               + sig[idx][None, :] * hp[None, :])
        Jm = np.moveaxis(Hm, 2, 0) + gJ[None] + sig[idx][:, None, None]
        return mul, Jm

    def pore(av, idx):
        th = (w0[idx] + av.sum(0)) / C if C > 0 else np.zeros(idx.size)
        mup = -P[:, idx] + (Pb[idx] * th / (1 - th))[None, :]
        Jp = (Pb[idx] / (C * (1 - th) ** 2)) if C > 0 else np.zeros(idx.size)
        return th, mup, Jp

    e = np.zeros((S, n)); av = np.zeros((S, n))
    live = np.flatnonzero(~single)                 # single-substance cells are bracketed below
    I2 = np.eye(2 * S)
    for it in range(MAX_NEWTON):
        if live.size == 0:
            break
        idx = live
        qp = q0[:, idx] - e[:, idx] - av[:, idx]
        mu, Jm = potentials(qp, idx)
        th, mup, Jp = pore(av[:, idx], idx)
        Xe = mu - menv[:, None]
        Xa = mu - mup
        pa = Son[:, idx] & (Xa > 0)
        Re = np.where(Eon[:, idx], e[:, idx] - ke[:, idx] * Xe, e[:, idx])
        Ra = np.where(Son[:, idx], av[:, idx] - ks[:, idx] * np.where(pa, Xa, 0.0), av[:, idx])
        m_ = idx.size
        J = np.zeros((m_, 2 * S, 2 * S))
        kE = np.where(Eon[:, idx], ke[:, idx], 0.0).T[:, :, None]         # [m, S, 1]
        kA = np.where(pa, ks[:, idx], 0.0).T[:, :, None]
        J[:, :S, :S] = kE * Jm
        J[:, :S, S:] = kE * Jm
        J[:, S:, :S] = kA * Jm
        J[:, S:, S:] = kA * (Jm + Jp[:, None, None])
        J += I2[None]
        Rv = np.concatenate([Re, Ra], 0).T                                # [m, 2S]
        d = np.linalg.solve(J, -Rv[:, :, None])[:, :, 0]
        de, da = d[:, :S].T, d[:, S:].T
        dqp = -(de + da)
        t = np.ones(m_)
        with np.errstate(divide="ignore", invalid="ignore"):
            lim = np.where((dqp < 0) & (qp > 0) & ~single[idx][None, :], 0.9 * qp / -dqp, np.inf).min(0)
            dth = da.sum(0) / C if C > 0 else np.zeros(m_)
            lim2 = np.where(dth > 0, 0.9 * (1 - th) / dth, np.inf)
        t = np.minimum(t, np.minimum(lim, lim2))
        e[:, idx] += t * de
        av[:, idx] += t * da
        done = (t == 1.0) & np.all(np.abs(d) <= 1e-13 * h0[idx][:, None] + 1e-300, axis=1)
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
            return potentials(qq, sidx)[0][isub, cols]

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
    drain = drain & single
    if drain.any():
        for j in np.flatnonzero(drain):
            i = int(np.flatnonzero(q0[:, j] > 0)[0])
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
            mu00, _ = potentials(np.zeros((S, 1)), np.array([j]))
            if xi > mu00[i, 0] + 1e-9 * abs(mu00[i, 0]) + 1e-300:
                raise Reject("local: drained cell violates its KKT condition")
            e[:, j] = 0.0; av[:, j] = 0.0
            e[i, j], av[i, j] = ee, aa
            q1[:, j] = 0.0
        q1[:, ~drain] = (q0 - e - av)[:, ~drain]
    if np.any(q1 < 0) or np.any(e < 0) or np.any(av < 0):
        raise Reject("local: negative amount")
    qn = q.copy(); qn[:, cells] = q1
    heldn = held.copy(); heldn[:, cells] = held[:, cells] + av
    dq = q1 - q0
    dh = np.zeros(N); dh[cells] = dq.sum(0)
    capf = np.flatnonzero(np.isin(G.fL, cells) | np.isin(G.fR, cells))
    dF = dF_grav(ctx, cells, q0, dq) + dF_cap(ctx, capf, h, dh) + dF_wet(ctx, cells, h0, dh[cells]) + dF_exch(ctx, dq)
    if not nonmix:
        mx1 = mixing(ctx, q1)
        dF += dx ** 2 * np.sum((mx1[0] + mx1[3]) - (mx0[0] + mx0[3]))
    dF += -dx ** 2 * np.sum(ctx.mu_env[:, None] * av)          # the pores' matter is counted against its reservoir
    if ctx.P is not None and C > 0 and av.any():
        th0, th1 = w0 / C, (w0 + av.sum(0)) / C
        dB = -(th1 - th0) - np.log1p(-(th1 - th0) / (1 - th0))
        dF += dx ** 2 * np.sum(-(P * av).sum(0) + Pb * C * dB)
    with np.errstate(divide="ignore", invalid="ignore"):
        De = np.where(Eon, e * e / ke, 0.0)
        Ds = np.where(Son & (av > 0), av * av / ks, 0.0)
    rep.update(D_exchange=dx ** 2 * np.sum(De), D_sorption=dx ** 2 * np.sum(Ds), dF=dF,
               e=e.sum(1) * dx ** 2, a=av.sum(1) * dx ** 2, cells=cells)
    return qn, heldn, rep
