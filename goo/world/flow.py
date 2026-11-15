"""Substep 1: flow of the film's total volume, substances in donor proportions (DYNAMICS.md §4.1)."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from goo.world.common import Reject, coeff, mixing, dF_grav, dF_cap, dF_wet, mu_wet
from goo.world.grid import capillary_pressure

MAX_CLASS_ITER = 400
RELEASE_TOL = 1e-12  # a stuck face whose force exceeds its threshold by less than this share of the largest
                     # force is at its threshold to rounding and is not released


def _orient(Gp, Gm, allowp, allowm):
    """Each face's direction for this step, from the old state's force: +1 (L to R, donor L), -1, or 0."""
    fwd = allowp & (Gp > 0)
    bwd = allowm & (Gm < 0)
    o = np.zeros(Gp.shape, np.int8)
    o[fwd] = 1
    o[bwd] = -1
    two = fwd & bwd
    if two.any():
        o[two] = np.where((Gp + Gm)[two] >= 0, 1, -1)
    return o


def flow(ctx, q, dt):
    """(new q, report). Raises Reject when the outflow bound or positivity fails."""
    G = ctx.grid
    dx, N = G.dx, G.N
    fL, fR = G.fL, G.fR
    h = q.sum(0)
    m = ctx.rho @ q
    wet = h > 0
    rep = dict(D_viscous=0.0, D_plastic=0.0, dE=0.0, dF_mix=0.0, cells=np.zeros(0, np.int64),
               faces=np.zeros(0, np.int64))
    wL, wR = wet[fL], wet[fR]
    cand = wL | wR
    rf = np.minimum(ctx.r[fL], ctx.r[fR])
    if not (cand & (rf > 0)).any():
        return q, rep
    eta = coeff(ctx, "viscosity", q, None)
    if eta is None:
        raise ValueError("Constitutive: viscosity is required for a wet film")
    eta = np.asarray(eta, np.float64)
    if np.any(np.isnan(eta)) or np.any(eta[wet] <= 0):
        raise ValueError("viscosity must be positive (+inf allowed: nothing flows)")
    with np.errstate(divide="ignore", invalid="ignore"):
        # +inf viscosity (no liquid, or jammed) is a resistance with mobility exactly zero (SPEC §4.3)
        Mc = np.where(wet & np.isfinite(eta), h * h * h / (3.0 * np.where(np.isfinite(eta), eta, 1.0)), 0.0)
    both = wL & wR
    M = np.where(both, 0.5 * (Mc[fL] + Mc[fR]), np.where(wL, Mc[fL], Mc[fR])) * rf
    M = np.where(cand, M, 0.0)
    allowp, allowm = wL & (M > 0), wR & (M > 0)
    if not (allowp | allowm).any():
        return q, rep
    hf = np.where(both, 0.5 * (h[fL] + h[fR]), np.where(wL, h[fL], h[fR]))
    Y = np.zeros(G.nF)
    tau = coeff(ctx, "yield_stress", q, None)
    if tau is not None and np.any(tau[wet] > 0):
        tf = np.where(both, 0.5 * (tau[fL] + tau[fR]), np.where(wL, tau[fL], tau[fR]))
        with np.errstate(divide="ignore", invalid="ignore"):
            Y = np.where(cand, tf / hf, 0.0)
        stuck = np.isposinf(Y)                    # an infinite yield stress: the face never moves
        if stuck.any():
            M = np.where(stuck, 0.0, M)
            Y = np.where(stuck, 0.0, Y)
            allowp, allowm = wL & (M > 0), wR & (M > 0)
    gx, gy, gn = ctx.g
    a = ctx.a
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = np.where(wet, m / h, 0.0)
        phi = np.where(wet, q / h, 0.0)
    A = gn * m * 0.5 + capillary_pressure(G, ctx.gam_f, ctx.zb + h) + mu_wet(ctx, h)
    B = gn * (ctx.zb + 0.5 * h)
    DA = G.D(A)
    DB = G.D(B) + np.where(G.isx, -gx * dx, -gy * dx)
    G0p = -(DA + rb[fL] * DB) / dx
    G0m = -(DA + rb[fR] * DB) / dx
    gamon = ctx.gam_f > 0

    o = _orient(G0p, G0m, allowp, allowm)
    capc = (dx / dt) * h                              # what a cell can give in this step, as a flux
    cache = {}

    def solve(free, sat, o, Gd0, rbd, Y):
        """The step with the faces `free` flowing in their direction, every other face stuck, and the cells
        `sat` giving exactly all they hold (multiplier lam, a reaction force against their outflow). Returns
        the fluxes U, forces Gd, multipliers lam [N], the active cells and the capillary faces touching them."""
        act = np.flatnonzero(free)
        cells = np.unique(np.concatenate([fL[act], fR[act]]))
        n, na = cells.size, act.size
        pos = np.full(N, -1)
        pos[cells] = np.arange(n)
        iL, iR = pos[fL[act]], pos[fR[act]]
        rd = rbd[act]
        Ma = M[act]
        oa = o[act].astype(np.float64)
        G0 = Gd0[act] - oa * Y[act]
        one = np.ones(na)
        ar = np.arange(n)
        capf = np.flatnonzero(((pos[fL] >= 0) | (pos[fR] >= 0)) & gamon)
        cap_rows, cap_cols, cap_vals = [], [], []
        if capf.size:
            pL, pR = pos[fL[capf]], pos[fR[capf]]
            gv = ctx.gam_f[capf] / dx ** 2
            kL, kR, kB = pL >= 0, pR >= 0, (pL >= 0) & (pR >= 0)
            cap_rows = [pL[kL], pR[kR], pL[kB], pR[kB]]
            cap_cols = [pL[kL], pR[kR], pR[kB], pL[kB]]
            cap_vals = [gv[kL], gv[kR], -gv[kB], -gv[kB]]
        one_rho = rd.size > 0 and bool(np.all(rd == rd[0]))
        if one_rho:
            # every donor has one density rho: dm = rho dh exactly, and the system is in dh alone, with
            # H = g_n (rho + a)^2 / (4 a) + the capillary Laplacian (the same equations, half the unknowns)
            rho1 = float(rd[0])
            C = sp.csr_matrix((np.concatenate([one, -one]), (np.concatenate([iL, iR]), np.tile(np.arange(na), 2))),
                              shape=(n, na))
            Hm = sp.csr_matrix((np.concatenate([np.full(n, gn * (rho1 + a) ** 2 / (4 * a))] + cap_vals),
                                (np.concatenate([ar] + cap_rows), np.concatenate([ar] + cap_cols))), shape=(n, n))
            nn = n
        else:
            C = sp.csr_matrix((np.concatenate([one, -one, rd, -rd]),
                               (np.concatenate([iL, iR, n + iL, n + iR]), np.tile(np.arange(na), 4))), shape=(2 * n, na))
            rows = [ar, ar, n + ar, n + ar] + cap_rows
            cols = [ar, n + ar, ar, n + ar] + cap_cols
            vals = [np.full(n, gn * a / 4), np.full(n, gn / 4), np.full(n, gn / 4), np.full(n, gn / (4 * a))] + cap_vals
            Hm = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, 2 * n))
            nn = 2 * n
        K = C @ sp.diags(Ma) @ C.T
        Asys = sp.identity(nn, format="csr") + (dt / dx ** 2) * (K @ Hm)
        rhs = -(dt / dx) * (C @ (Ma * G0))
        dcell = np.where(oa > 0, fL[act], fR[act])
        scells = np.flatnonzero(sat)
        ns = scells.size
        if ns:
            spos = np.full(N, -1); spos[scells] = np.arange(ns)
            fs = spos[dcell]
            k = fs >= 0
            P = sp.csr_matrix((np.ones(k.sum()), (np.flatnonzero(k), fs[k])), shape=(na, ns))
            Bm = -(dt / dx) * (C @ sp.diags(Ma * oa) @ P)
            Cr = (P.T @ sp.diags(Ma * oa)) @ (C.T @ Hm) / dx
            Dr = -sp.diags(np.asarray(P.T @ Ma).ravel())
            r2 = capc[scells] - P.T @ (Ma * oa * G0)
            Afull = sp.bmat([[Asys, Bm], [Cr, Dr]], format="csc")
            bb = np.concatenate([rhs, r2])
            lu = splu(Afull)
            sol = lu.solve(bb)
            sol = sol + lu.solve(bb - Afull @ sol)               # one step of refinement
            dy, lam_s = sol[:nn], sol[nn:]
        else:
            Ac = Asys.tocsc()
            Ac.sort_indices()
            key = (Ac.shape, Ac.indptr.tobytes(), Ac.indices.tobytes(), Ac.data.tobytes())
            if cache.get("key") != key:                 # the same matrix bit for bit: the same factors
                cache["key"], cache["lu"] = key, splu(Ac)
            dy, lam_s = cache["lu"].solve(rhs), np.zeros(0)
        lam = np.zeros(N)
        lam[scells] = lam_s
        dh_ = np.zeros(N); dm_ = np.zeros(N)
        dh_[cells] = dy[:n]
        dm_[cells] = rho1 * dy[:n] if one_rho else dy[n:]
        w_ = gn * (dm_ + a * dh_) / 4                      # potentials' increments on every cell, the
        dA = w_ + capillary_pressure(G, ctx.gam_f, dh_)     # inactive neighbours of active cells included
        dB = w_ / a
        Gd = Gd0 - (G.D(dA) + rbd * G.D(dB)) / dx
        U = np.zeros(G.nF)
        U[act] = Ma * (Gd[act] - oa * (Y[act] + lam[dcell]))
        return U, Gd, lam, cells, capf

    # orientation: the sign of a two-way Newtonian predictor with the old force's donors, where allowed
    nocell = np.zeros(N, bool)
    Gd0 = np.where(o > 0, G0p, G0m)
    rbd = np.where(o > 0, rb[fL], rb[fR])
    if np.any(o != 0):
        Up = solve(o != 0, nocell, o, Gd0, rbd, np.zeros(G.nF))[0]
        o = np.where((Up > 0) & allowp, 1, np.where((Up < 0) & allowm, -1, 0)).astype(np.int8)
    Gd0 = np.where(o > 0, G0p, G0m)                 # the force with the oriented donor's density
    rbd = np.where(o > 0, rb[fL], rb[fR])
    donor = np.where(o > 0, fL, fR)
    scale = float(np.max(np.abs(Gd0[o != 0]))) if np.any(o != 0) else 0.0
    # primal active-set method for the convex problem in u = o U >= 0 with each cell's outflow <= its content
    # (DYNAMICS.md §4.1): iterates stay feasible and each step lowers the objective, so it cannot cycle.
    oriented = o != 0
    free = oriented & ((Y == 0) | (o * Gd0 - Y > 0))
    sat = nocell.copy()
    seen, single = set(), False
    seen_pd, primal = set(), False
    u = np.zeros(G.nF)
    U = np.zeros(G.nF)
    lam = np.zeros(N)
    cells = np.zeros(0, np.int64); capf = np.zeros(0, np.int64)
    for _it in range(MAX_CLASS_ITER):
        sat &= np.bincount(donor[free], minlength=N) > 0
        if not free.any():
            U = np.zeros(G.nF); cells = np.zeros(0, np.int64); capf = np.zeros(0, np.int64); lam = np.zeros(N)
            break
        Us, Gd, lam_s, cells, capf = solve(free, sat, o, Gd0, rbd, Y)
        us = o * Us
        neg = free & (us < 0)
        outs = np.bincount(donor[free], us[free], N)
        over = ~sat & (outs > capc)
        if not neg.any() and not over.any():
            u, U, lam = us, Us, lam_s
            rel_f = oriented & ~free & (o * Gd - Y - lam[donor] > RELEASE_TOL * scale)
            rel_c = sat & (lam < -RELEASE_TOL * scale)
            if not rel_f.any() and not rel_c.any():
                break
            key = (free.tobytes(), sat.tobytes())
            if key in seen:
                single = True                    # a degenerate cycle: from now on release one at a time
            seen.add(key)
            if single:                           # the most violated multiplier only (the textbook rule)
                vf = np.where(rel_f, o * Gd - Y - lam[donor], -np.inf)
                vc = np.where(rel_c, -lam, -np.inf)
                if vf.max() >= vc.max():
                    rel_f = np.zeros_like(rel_f); rel_f[int(np.argmax(vf))] = True; rel_c = np.zeros_like(rel_c)
                else:
                    rel_c = np.zeros_like(rel_c); rel_c[int(np.argmax(vc))] = True; rel_f = np.zeros_like(rel_f)
            free |= rel_f
            sat &= ~rel_c
            continue
        if not primal:
            # primal-dual move: every violator joins the working set at once (fast, usually final)
            key = (free.tobytes(), sat.tobytes())
            if key in seen_pd:
                primal = True                    # it cycled: switch to the primal method, which cannot
                u = np.zeros(G.nF)               # u = 0 is feasible for every working set
            else:
                seen_pd.add(key)
                free &= ~neg
                sat |= over
                continue
        outu = np.bincount(donor[free], u[free], N)
        with np.errstate(divide="ignore", invalid="ignore"):
            rf_ = np.where(neg, u / (u - us), np.inf)
            rc_ = np.where(over, (capc - outu) / (outs - outu), np.inf)
        alpha = float(min(rf_.min(), rc_.min()))
        u = u + alpha * (us - u)
        bf = neg & (rf_ <= alpha)
        u[bf] = 0.0
        free &= ~bf
        sat |= over & (rc_ <= alpha)
    else:
        raise Reject("flow: active set did not settle")
    act = np.flatnonzero(free)
    if not np.any(U):
        return q, rep
    absU = np.abs(U)
    if sat.any():
        # a saturated cell's outflow equals its content to the solver's accuracy; make it exact
        o_s = np.bincount(donor, absU, N)
        fix = sat[donor] & (absU > 0)
        absU[fix] *= (capc[donor[fix]] / o_s[donor[fix]])
        U = o * absU
    out = (dt / dx) * np.bincount(donor, absU, N)
    if np.any(out > h * (1 + 1e-12)):
        raise Reject("flow: a cell's outflow exceeds its content")
    with np.errstate(divide="ignore", invalid="ignore"):
        alph = np.where(h > 0, np.minimum(out / h, 1.0), 0.0)     # out <= h holds exactly; min() absorbs its rounding
    alph[sat] = 1.0                                             # a saturated cell gives all it holds
    recv = np.where(o > 0, fR, fL)
    inflow = np.empty_like(q)
    for i in range(q.shape[0]):
        inflow[i] = (dt / dx) * np.bincount(recv, phi[i, donor] * absU, N)
    qn = q * (1.0 - alph) + inflow
    dq = qn - q
    if np.any(qn < 0):
        raise Reject("flow: negative amount")
    dqc = dq[:, cells]
    dh = dq.sum(0)
    dE = dF_grav(ctx, cells, q[:, cells], dqc) + dF_cap(ctx, capf, h, dh) + dF_wet(ctx, cells, h[cells], dh[cells])
    mx0, mx1 = mixing(ctx, q[:, cells]), mixing(ctx, qn[:, cells])
    dFmix = 0.0 if mx0 is None else dx ** 2 * np.sum((mx1[0] + mx1[3]) - (mx0[0] + mx0[3]))
    au = absU[act]
    rep.update(D_viscous=dt * dx ** 2 * np.sum(au * au / M[act]), D_plastic=dt * dx ** 2 * np.sum(Y[act] * au),
               dE=dE, dF_mix=dFmix, cells=cells, faces=act)
    return qn, rep
