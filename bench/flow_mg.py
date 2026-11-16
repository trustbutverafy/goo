"""Prototype (i): the flow sub-problem of one substance solved matrix-free by geometric multigrid on the pressure
(mixed) form, with an admissible step by construction. No factorisation.

    PYTHONPATH=$PWD python bench/flow_mg.py compare H W     the present flow substep against this one, same inputs
    PYTHONPATH=$PWD python bench/flow_mg.py scale H W [frac] this one alone (a tilted water disc covering frac)

The problem (docs/DYNAMICS.md §4.1, one density): find face fluxes U minimising
    Phi(U) = E(h^n - (dt/dx) C U) + dt dx^2 sum_f U_f^2 / (2 M_f)
with E quadratic (gravity, capillarity; wetting linearised: explicit concave part), flow out of wet cells only, each cell's
outflow <= its content. Algorithm:
 1. Unconstrained Newton step = one linear system A dh = b, A = I + (dt/dx^2) K H (K = C M C^T, H = g' I + capillary
    Laplacian): written as two second-order equations in (dh, p = H dh) and solved by a fixed number of V-cycles of
    geometric multigrid, red-black 2x2 block Gauss-Seidel per cell, rediscretised coarse levels aligned in absolute
    panel coordinates. Matrix-free: every operator is a 5-point stencil on [H, W] arrays.
 2. Fluxes from the force at the new state: U = M (G0 - grad(H dh)); the state is updated in flux form from U only, so
    matter is conserved exactly whatever the solver's error (O1).
 3. Feasibility: a face into a bare cell carries nothing in; a cell whose outflow exceeds its content has its outflowing
    faces scaled down to give exactly its content (a feasible flux; any such projection is admissible).
 4. Ray search: Phi(t U) is a quadratic in t; t = min(t*, t_feasible). At t*, d/dt Phi = 0 on the ray, so (DYNAMICS.md
    §5, "Constraints") D_num = Bregman of E >= 0 by construction for ANY U: the solver's error changes the accuracy of
    the step, never its admissibility (O2, O3). The accuracy is measured against the factorised solve.
"""
import sys, os, time, resource
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------- the multigrid on the mixed form ----------------
class Level:
    def __init__(self, Mx, My, Gx, Gy, act, c, g1, y0, x0):
        # Mx [H, W-1] face mobilities (x faces), My [H-1, W]; Gx, Gy capillary gamma/dx^2 per face; act [H, W] bool;
        # c = dt/dx^2; g1 = g' (gravity's diagonal); (y0, x0) absolute origin of this level
        self.Mx, self.My, self.Gx, self.Gy, self.act, self.c, self.g1 = Mx, My, Gx, Gy, act, c, g1
        self.y0, self.x0 = y0, x0
        H, W = act.shape
        self.SM = _nsum(Mx, My, H, W)
        self.SG = _nsum(Gx, Gy, H, W)
        # red-black colouring in absolute coordinates (O5: a window colours its cells as the panel does)
        iy, ix = np.mgrid[y0:y0 + H, x0:x0 + W]
        self.red = ((iy + ix) % 2 == 0)
        self.det = 1.0 + c * self.SM * (g1 + self.SG)


def _nsum(Fx, Fy, H, W):
    s = np.zeros((H, W))
    s[:, :-1] += Fx; s[:, 1:] += Fx
    s[:-1, :] += Fy; s[1:, :] += Fy
    return s


def _nsum_v(Fx, Fy, v):
    """sum_j F_ij v_j over the 4 neighbours."""
    s = np.zeros_like(v)
    s[:, :-1] += Fx * v[:, 1:]; s[:, 1:] += Fx * v[:, :-1]
    s[:-1, :] += Fy * v[1:, :]; s[1:, :] += Fy * v[:-1, :]
    return s


def apply(L, h, p):
    """(h + c K p, p - H h) with K p = sum_j M_ij (p_i - p_j), H h = g' h + sum_j G_ij (h_i - h_j); inactive: identity."""
    Kp = L.SM * p - _nsum_v(L.Mx, L.My, p)
    Hh = (L.g1 + L.SG) * h - _nsum_v(L.Gx, L.Gy, h)
    r1 = np.where(L.act, h + L.c * Kp, h)
    r2 = np.where(L.act, p - Hh, p)
    return r1, r2


def smooth(L, h, p, b1, b2, sweeps):
    for _ in range(sweeps):
        for colour in (L.red, ~L.red):
            SMP = _nsum_v(L.Mx, L.My, p)
            SGH = _nsum_v(L.Gx, L.Gy, h)
            e1 = b1 + L.c * SMP
            e2 = b2 - SGH
            hn = (e1 - L.c * L.SM * e2) / L.det
            pn = e2 + (L.g1 + L.SG) * hn
            m = colour & L.act
            h = np.where(m, hn, np.where(colour, b1, h))
            p = np.where(m, pn, np.where(colour, b2, p))
    return h, p


def restrict(a):
    H, W = a.shape
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def prolong(a):
    """Bilinear, cell-centred (9/16, 3/16, 3/16, 1/16), with zero outside (the window's margin is inactive)."""
    H, W = a.shape
    z = np.zeros((H + 2, W + 2)); z[1:-1, 1:-1] = a
    out = np.empty((2 * H, 2 * W))
    C = z[1:-1, 1:-1]
    for dy, sy in ((0, -1), (1, 1)):
        for dx_, sx in ((0, -1), (1, 1)):
            Y = z[1 + sy:H + 1 + sy, 1:-1]
            X = z[1:-1, 1 + sx:W + 1 + sx]
            D = z[1 + sy:H + 1 + sy, 1 + sx:W + 1 + sx]
            out[dy::2, dx_::2] = (9 * C + 3 * Y + 3 * X + D) / 16
    return out


def coarsen(L):
    H, W = L.act.shape
    act = restrict(L.act.astype(float)) > 0
    # a coarse face is the two fine faces on its line: their mean (rediscretisation at spacing 2 dx)
    Mx = 0.5 * (L.Mx[0::2, 1::2] + L.Mx[1::2, 1::2])
    My = 0.5 * (L.My[1::2, 0::2] + L.My[1::2, 1::2])
    Gx = 0.25 * 0.5 * (L.Gx[0::2, 1::2] + L.Gx[1::2, 1::2])       # gamma/(2dx)^2
    Gy = 0.25 * 0.5 * (L.Gy[1::2, 0::2] + L.Gy[1::2, 1::2])
    return Level(Mx, My, Gx, Gy, act, L.c / 4, L.g1, L.y0 // 2, L.x0 // 2)


def hierarchy(L0, coarsest=4):
    levels = [L0]
    while min(levels[-1].act.shape) >= 2 * coarsest and all(s % 2 == 0 for s in levels[-1].act.shape):
        levels.append(coarsen(levels[-1]))
    return levels


def vcycle(levels, k, h, p, b1, b2, pre=2, post=2, coarse=60):
    L = levels[k]
    if k == len(levels) - 1:
        return smooth(L, h, p, b1, b2, coarse)
    h, p = smooth(L, h, p, b1, b2, pre)
    r1, r2 = apply(L, h, p)
    r1, r2 = b1 - r1, b2 - r2
    c1, c2 = restrict(r1), restrict(r2)
    e1, e2 = vcycle(levels, k + 1, np.zeros_like(c1), np.zeros_like(c2), c1, c2, pre, post, coarse)
    h, p = h + prolong(e1), p + prolong(e2)
    return smooth(L, h, p, b1, b2, post)


# ---------------------------------------------------------------- the flow sub-problem ---------------------------
def problem_from_ctx(ctx, q, dt):
    """The coefficients of the present flow substep (goo/world/flow.py), one substance, on the window [H, W]."""
    from goo.world.common import coeff, mu_wet
    from goo.world.grid import capillary_pressure
    G = ctx.grid
    H, W, dx = G.H, G.W, G.dx
    h = q.sum(0)
    wet = h > 0
    eta = np.asarray(coeff(ctx, "viscosity", q, None), np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        Mc = np.where(wet & np.isfinite(eta), h ** 3 / (3.0 * np.where(np.isfinite(eta), eta, 1.0)), 0.0)
    fL, fR = G.fL, G.fR
    wL, wR = wet[fL], wet[fR]
    rf = np.minimum(ctx.r[fL], ctx.r[fR])
    M = np.where(wL & wR, 0.5 * (Mc[fL] + Mc[fR]), np.where(wL, Mc[fL], Mc[fR])) * rf
    M = np.where(wL | wR, M, 0.0)
    rho = float(ctx.rho[0])
    gx, gy, gn = ctx.g
    # the potential (pressure) at the old state, J/m^3, and the force on faces
    p0 = gn * rho * (ctx.zb + h) + capillary_pressure(G, ctx.gam_f, ctx.zb + h) + mu_wet(ctx, h) \
        - rho * (gx * G.X + gy * G.Y)
    G0 = -G.D(p0) / dx
    return dict(H=H, W=W, dx=dx, dt=dt, h=h, wet=wet, M=M, G0=G0, gam=np.asarray(ctx.gam_f, np.float64),
                g1=gn * rho, rho=rho, nx=G.nx, fL=fL, fR=fR, y0=int(G.IY[0]), x0=int(G.IX[0]), p0=p0)


K_LEVELS = 6        # blocks of 64 cells: up to 7 levels


def solve_flow(P, cycles=6, verbose=False, limit_iters=0):
    """limit_iters > 0: before the final projection, re-solve up to that many times with the mobility of every face
    out of an over-drawn cell scaled by content/outflow (a frozen mobility >= 0 is admissible: DYNAMICS.md §5), warm
    started. This is a different constraint model from the present KKT active set (a cell that would empty flows more
    slowly instead of giving exactly all it holds), arguably closer to M ~ h^3."""
    if limit_iters:
        P = dict(P); M = P["M"].copy(); t_acc = 0.0
        for it in range(limit_iters):
            S = solve_flow(dict(P, M=M), cycles, False, 0)
            t_acc += S["t_all"]
            Uu = S["U_raw"]; donor = np.where(Uu > 0, P["fL"], P["fR"])
            out = (P["dt"] / P["dx"]) * np.bincount(donor, np.abs(Uu), P["H"] * P["W"])
            over = out > P["h"]
            if not over.any():
                break
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(over, P["h"] / out, 1.0)
            M = M * f[donor]
        P["M"] = M
        S = solve_flow(P, cycles, verbose, 0)
        S["t_all"] += t_acc; S["limit_iters_used"] = it + 1
        return S
    H, W, dx, dt = P["H"], P["W"], P["dx"], P["dt"]
    nx, fL, fR, M, G0, gam, h, wet = P["nx"], P["fL"], P["fR"], P["M"], P["G0"], P["gam"], P["h"], P["wet"]
    t0 = time.perf_counter()
    # one-way valves: a face between a wet and a bare cell carries flow out of the wet cell only. In the linear solve
    # every face with M > 0 is two-way (one density: no orientation is needed for convexity); infeasible inflow
    # into a wet cell from a bare one is removed in step 3.
    act = np.zeros(H * W, bool); mf = M > 0
    act[fL[mf]] = True; act[fR[mf]] = True
    capon = (gam > 0) & (act[fL] | act[fR])
    Gf = np.where(capon, gam, 0.0) / dx ** 2
    Mx, My = M[:nx].reshape(H, W - 1), M[nx:].reshape(H - 1, W)
    Gx, Gy = Gf[:nx].reshape(H, W - 1), Gf[nx:].reshape(H - 1, W)
    # pad to blocks of 2^K aligned in absolute panel coordinates, with inactive cells (O5: a window and the panel
    # build the same hierarchy on the same cells)
    B = 2 ** K_LEVELS
    y0, x0 = P["y0"], P["x0"]
    py, px = y0 % B, x0 % B
    Hp, Wp = -(-(H + py) // B) * B, -(-(W + px) // B) * B
    def pad(a, dy, dx_):
        z = np.zeros((Hp - dy, Wp - dx_)); z[py:py + a.shape[0], px:px + a.shape[1]] = a; return z
    F0 = M * G0
    b = -(dt / dx) * (np.bincount(fL, F0, H * W) - np.bincount(fR, F0, H * W))
    L0 = Level(pad(Mx, 0, 1), pad(My, 1, 0), pad(Gx, 0, 1), pad(Gy, 1, 0), pad(act.reshape(H, W), 0, 0) > 0,
               dt / dx ** 2, P["g1"], y0 - py, x0 - px)
    levels = hierarchy(L0)
    b1, b2 = pad(b.reshape(H, W), 0, 0), np.zeros((Hp, Wp))
    dh, p = np.zeros((Hp, Wp)), np.zeros((Hp, Wp))
    nb = np.linalg.norm(b1)
    hist = []
    for k in range(cycles):
        dh, p = vcycle(levels, 0, dh, p, b1, b2)
        if verbose:
            r1, r2 = apply(L0, dh, p)
            hist.append(float(np.linalg.norm(b1 - r1) / max(nb, 1e-300)))
    dh, p = dh[py:py + H, px:px + W], p[py:py + H, px:px + W]
    t_solve = time.perf_counter() - t0
    # 2. fluxes from the force at the new state (dh's potential increment p = H dh)
    pf = p.ravel()
    U = M * (G0 - (pf[fR] - pf[fL]) / dx)
    U_raw = U.copy()
    # 3. feasibility: no flow into a wet cell out of a bare one; outflow <= content
    U = np.where((U > 0) & ~wet[fL], 0.0, U)
    U = np.where((U < 0) & ~wet[fR], 0.0, U)
    donor = np.where(U > 0, fL, fR)
    aU = np.abs(U)
    out = (dt / dx) * np.bincount(donor, aU, H * W)
    with np.errstate(divide="ignore", invalid="ignore"):
        lim = np.where(out > h, h / out, 1.0)
    nlim = int((out > h).sum())
    U = U * lim[donor]
    # 4. ray search on Phi(tU) = t Phi1 + t^2 Phi2 (the explicit wetting part is linear in Phi: it is in p0)
    dhU = -(dt / dx) * (np.bincount(fL, U, H * W) - np.bincount(fR, U, H * W))
    ds = dhU[fR] - dhU[fL]
    QE = 0.5 * dx ** 2 * P["g1"] * float(np.sum(dhU * dhU)) + 0.5 * float(np.sum(np.where(capon, gam, 0.0) * ds * ds))
    with np.errstate(divide="ignore", invalid="ignore"):
        QD = 0.5 * dt * dx ** 2 * float(np.sum(np.where(M > 0, U * U / M, 0.0)))
    Phi1 = dx ** 2 * float(np.sum(P["p0"] * dhU))
    tstar = -Phi1 / (2 * (QE + QD)) if QE + QD > 0 else 0.0
    # feasibility along the ray: outflow t * out' <= h, already <= h at t = 1 for every cell that was limited
    out1 = (dt / dx) * np.bincount(np.where(U > 0, fL, fR), np.abs(U), H * W)
    with np.errstate(divide="ignore", invalid="ignore"):
        tfeas = float(np.min(np.where(out1 > 0, h / out1, np.inf)))
    t = min(max(tstar, 0.0), tfeas)
    U = t * U
    dh_final = t * dhU
    hn = h + dh_final
    # ledger of the step: dE exactly as the increments of the quadratic E plus the linear part, D_mech from fluxes
    dE = t * Phi1 + t * t * QE
    D = t * t * 2 * QD
    Dnum = -dE - D
    return dict(h=hn, U=U, U_raw=U_raw, t=t, tstar=tstar, tfeas=tfeas, limited=nlim, dE=dE, D_viscous=D, D_num=Dnum,
                bregman=t * t * QE, t_solve=t_solve, t_all=time.perf_counter() - t0, hist=hist,
                levels=len(levels), active=int(act.sum()))


# ---------------------------------------------------------------- harness -----------------------------------------
def capture(world, dt):
    """Run one step of the present code and capture the flow substep's inputs, output and time."""
    import goo.world.world as WW
    from goo.world import flow as F
    rec = {}
    orig = WW.flow

    def wrapped(ctx, q, dt_):
        t = time.perf_counter()
        out = orig(ctx, q, dt_)
        if "ctx" not in rec:
            rec.update(ctx=ctx, q=q.copy(), dt=dt_, out=out[0].copy(), rep=out[1], t=time.perf_counter() - t)
        return out
    WW.flow = wrapped
    try:
        r = world.step(dt)
    finally:
        WW.flow = orig
    return rec, r


def main():
    from scenes import water
    mode, H, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    dt = float(os.environ.get("DT", 0.5))
    w = water(H, W, frac, noise=float(os.environ.get("NOISE", 0.05)))
    if mode == "compare":
        rec, r = capture(w, dt)
        P = problem_from_ctx(rec["ctx"], rec["q"], rec["dt"])
        h_ref = rec["out"].sum(0)
        dref = h_ref - P["h"]
        print(f"present flow substep: {rec['t']:.2f} s (pieces of the step: {r.report['pieces']:.0f}); "
              f"D_viscous {rec['rep']['D_viscous']:.4e} dE {rec['rep']['dE']:.4e}")
        for cyc in (2, 4, 8, 12):
            S = solve_flow(P, cycles=cyc, verbose=True)
            err = np.linalg.norm(S["h"] - h_ref) / np.linalg.norm(dref)
            print(f"MG {cyc:2d} V-cycles: {S['t_all']:.3f} s (solve {S['t_solve']:.3f}), levels {S['levels']}, "
                  f"rel. residual {S['hist'][-1]:.1e}, |h - h_present| / |dh_present| = {err:.2e}, t* {S['tstar']:.6f} "
                  f"t {S['t']:.6f} limited cells {S['limited']}, D_visc {S['D_viscous']:.4e} dE {S['dE']:.4e} "
                  f"D_num {S['D_num']:.3e} (Bregman {S['bregman']:.3e}); min h {S['h'].min():.2e}; "
                  f"mass change {abs(S['h'].sum() - P['h'].sum()) / P['h'].sum():.1e}")
        for li in (2, 4, 8):
            S2 = solve_flow(P, cycles=8, limit_iters=li)
            err = np.linalg.norm(S2["h"] - h_ref) / np.linalg.norm(dref)
            print(f"MG 8 cycles + mobility limiting (<= {li} re-solves, used {S2['limit_iters_used']}): {S2['t_all']:.3f} s, "
                  f"|h - h_present| / |dh_present| = {err:.2e}, t {S2['t']:.4f}, limited at the end {S2['limited']}, "
                  f"D_visc {S2['D_viscous']:.4e} dE {S2['dE']:.4e} D_num {S2['D_num']:.3e}")
        print("residual history (12 cycles):", " ".join(f"{x:.1e}" for x in S["hist"]))
        print(f"present: cells emptied {int(((h_ref == 0) & (P['h'] > 0)).sum())} of {int((P['h'] > 0).sum())} wet; "
              f"max |dh|/h {np.max(np.abs(dref)[P['wet']] / P['h'][P['wet']]):.3f}")
    else:
        import goo.world.world as WW
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
        t = time.perf_counter()
        P = problem_from_ctx(rec["ctx"], rec["q"], rec["dt"])
        tp = time.perf_counter() - t
        for cyc in (4, 8):
            S = solve_flow(P, cycles=cyc, verbose=(cyc == 8))
            print(f"{H}x{W} wet {int(P['wet'].sum())} active {S['active']}: coefficients {tp:.2f} s, MG {cyc} cycles "
                  f"{S['t_all']:.2f} s (solve {S['t_solve']:.2f}); levels {S['levels']}; t {S['t']:.6f}; D_num {S['D_num']:.3e} "
                  f"Bregman {S['bregman']:.3e}; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
        print("residual history:", " ".join(f"{x:.1e}" for x in S["hist"]))


if __name__ == "__main__":
    main()
