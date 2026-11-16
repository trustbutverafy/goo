"""Prototype (ii): exact rest detection. A state is a rest state of the step (O4) when, at the old state,
  flow       no face's force exceeds its threshold in a direction it may flow (flow.py's "quiet faces" rule, the
             KKT argument of DYNAMICS.md §4.1: u = 0 is then THE minimiser),
  diffusion  on every face with mobility, every moving substance's relative force X_i - Xbar is exactly zero
             (then Newton's first residual is exactly 0 and it does not iterate: DYNAMICS.md §5, O4),
  local      every exchanging substance has mu = mu_env exactly (or k = 0) and no sorption is driven,
so step() would return the state bit for bit. Rest is a property of (state, environment, rate, dt-independent):
once proved it stays proved until one of them changes, so a world can memoise it by its state version and a held
or resting panel then costs O(1) per step.

    PYTHONPATH=$PWD python bench/rest.py H W

Measures, on an exactly resting paste (uniform, no reservoir) and on the quiet scene: the present step, a one-pass
detector (maps evaluated once; mixing by the numba kernel of bench/numba_kernels.py), and the memoised second step.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.simplefilter("ignore")


def ctx_of(w, dt):
    """The window context of World._piece (copied from goo/world/world.py), for the whole panel."""
    from goo.world.common import Ctx
    from goo.world.grid import Grid, panel_faces
    y0, y1, x0, x1 = 0, w.H, 0, w.W
    G = Grid(w.H, w.W, w.dx, 0, 0)
    c = w._lagged
    flat = lambda arr: None if arr is None else np.ascontiguousarray(arr).reshape(arr.shape[:-2] + (-1,))
    menv, has = w._mu_env()
    r = np.ones((w.H, w.W))
    return Ctx(G, w.substances, w.rho, w.con, w.env, w._ground_box(w._full), flat(w._zb), flat(r),
               panel_faces(c["gx"], c["gy"], y0, y1, x0, x1), flat(c["Sw"]) if w._has_wet else None,
               flat(c["ell"]) if w._has_wet else None, flat(c["P"]) if w._has_pore else None,
               flat(c["Pb"]) if w._has_pore else None, w.ground.capacity, w.a, menv, has, [], "exact", {}, [])


def flow_quiet(ctx, q, dt):
    """flow.py up to its quiet-faces return (copied): True when no face can move."""
    from goo.world.common import coeff, mu_wet
    from goo.world.grid import capillary_pressure
    from goo.world.flow import RELEASE_TOL
    G = ctx.grid; dx = G.dx; fL, fR = G.fL, G.fR
    h = q.sum(0); m = ctx.rho @ q; wet = h > 0
    wL, wR = wet[fL], wet[fR]; cand = wL | wR
    rf = np.minimum(ctx.r[fL], ctx.r[fR])
    if not (cand & (rf > 0)).any():
        return True
    eta = np.asarray(coeff(ctx, "viscosity", q, None), np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        Mc = np.where(wet & np.isfinite(eta), h * h * h / (3.0 * np.where(np.isfinite(eta), eta, 1.0)), 0.0)
    both = wL & wR
    M = np.where(both, 0.5 * (Mc[fL] + Mc[fR]), np.where(wL, Mc[fL], Mc[fR])) * rf
    M = np.where(cand, M, 0.0)
    allowp, allowm = wL & (M > 0), wR & (M > 0)
    if not (allowp | allowm).any():
        return True
    hf = np.where(both, 0.5 * (h[fL] + h[fR]), np.where(wL, h[fL], h[fR]))
    Y = np.zeros(G.nF)
    tau = coeff(ctx, "yield_stress", q, None)
    if tau is not None and np.any(tau[wet] > 0):
        tf = np.where(both, 0.5 * (tau[fL] + tau[fR]), np.where(wL, tau[fL], tau[fR]))
        with np.errstate(divide="ignore", invalid="ignore"):
            Y = np.where(cand, tf / hf, 0.0)
        stuck = np.isposinf(Y)
        if stuck.any():
            M = np.where(stuck, 0.0, M); Y = np.where(stuck, 0.0, Y)
            allowp, allowm = wL & (M > 0), wR & (M > 0)
    gx, gy, gn = ctx.g
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = np.where(wet, m / h, 0.0)
    A = gn * m * 0.5 + capillary_pressure(G, ctx.gam_f, ctx.zb + h) + mu_wet(ctx, h)
    B = gn * (ctx.zb + 0.5 * h)
    DA = G.D(A); DB = G.D(B) + np.where(G.isx, -gx * dx, -gy * dx)
    G0p = -(DA + rb[fL] * DB) / dx; G0m = -(DA + rb[fR] * DB) / dx
    sc0 = max(float(np.max(np.abs(G0p[allowp]), initial=0.0)), float(np.max(np.abs(G0m[allowm]), initial=0.0)))
    aA, aB = np.abs(A), np.abs(B)
    pr = (aA[fL] + aA[fR]) / dx
    ex = np.abs(DB - G.D(B))
    pp_, pm_ = pr + rb[fL] * (aB[fL] + aB[fR] + ex) / dx, pr + rb[fR] * (aB[fL] + aB[fR] + ex) / dx
    tolp, tolm = RELEASE_TOL * np.maximum(sc0, pp_), RELEASE_TOL * np.maximum(sc0, pm_)
    movp = allowp & np.where(Y > 0, G0p - Y > tolp, G0p > 0)
    movm = allowm & np.where(Y > 0, -G0m - Y > tolm, -G0m > 0)
    return not (movp | movm).any()


def diffusion_rest(ctx, q, mixfn):
    """True when every relative force on every face with mobility is exactly zero (diffuse.py's first residual is 0)."""
    from goo.world.common import coeff
    G = ctx.grid; dx = G.dx; fL, fR = G.fL, G.fR
    h = q.sum(0); wet = h > 0
    fa = np.flatnonzero(wet[fL] & wet[fR])
    if fa.size == 0:
        return True
    Lam = coeff(ctx, "mobility", q, None)
    if Lam is None or not np.any(Lam[:, wet] > 0):
        return True
    a, b = fL[fa], fR[fa]
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(wet, q / h, 0.0)
    om = (0.5 * (h[a] + h[b]) * 0.5 * (Lam[:, a] + Lam[:, b])) * 0.5 * (phi[:, a] + phi[:, b])
    gx, gy, gn = ctx.g
    psi = gn * (ctx.zb + 0.5 * h) - (gx * G.X + gy * G.Y)
    f, mu, H, fx, mux = mixfn(q)
    mu = mu + mux + ctx.rho[:, None] * psi[None, :]
    on = om > 0
    if np.any(on & ~(np.isfinite(mu[:, a]) & np.isfinite(mu[:, b]))):
        return False                                  # a substance entering a cell that lacks it: not at rest
    X = np.where(on, -(mu[:, b] - mu[:, a]) / dx, 0.0)
    W = om.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Xb = np.where(W > 0, (om * X).sum(0) / W, 0.0)
    return not np.any(on & (X - Xb != 0.0))


def local_rest(ctx, q):
    """True when local.py has nothing to do: no substance present exchanges with a reservoir (k > 0 and a reservoir)
    and nothing is sorbed. (A reservoir exactly at equilibrium with every cell, mu = mu_env to the bit, would also be
    rest; it is not checked here: it does not occur in practice.)"""
    from goo.world.common import coeff
    pores = ctx.P is not None and ctx.cap > 0
    ex = coeff(ctx, "exchange", q, None, ctx.env)
    if ex is not None:
        k = np.asarray(ex[0])
        if np.any((k > 0) & (q > 0) & ctx.has_env[:, None]):
            return False
    return not pores


def main():
    from goo.units import um, mm
    from goo.film import Ground, Environment
    from goo.substance import maps, library as L
    from goo.world import World
    from numba_kernels import mixing_nb, mixing_params
    from scenes import quiet
    H, W = int(sys.argv[1]), int(sys.argv[2])
    subs = (L.WATER, L.GUM_ARABIC, L.RUTILE); phi = 0.35
    g = Ground(np.zeros((H, W)), L.CELLULOSE, 0.0, 5 * um, 400 * um)
    h = np.full((H, W), 30 * um)
    q = np.stack([h * (1 - phi - 0.05), h * 0.05, h * phi])
    scenes = [("exact rest (uniform paste, no reservoir)", World(maps, subs, g, 0.25 * mm, Environment(), q=q)),
              ("quiet scene (paste, evaporating)", quiet(H, W))]
    c, kT, beta, B = mixing_params(subs)
    mixing_nb(q.reshape(3, -1)[:, :8].copy(), c, kT, beta, B)
    for name, w in scenes:
        t = time.perf_counter(); w.step(2.0); tp = time.perf_counter() - t
        ctx = ctx_of(w, 2.0)
        qf = np.ascontiguousarray(w.q.reshape(3, -1))
        t = time.perf_counter()
        a = flow_quiet(ctx, qf, 2.0); t1 = time.perf_counter() - t
        b = diffusion_rest(ctx, qf, lambda qq: mixing_nb(np.ascontiguousarray(qq), c, kT, beta, B)); t2 = time.perf_counter() - t - t1
        cl = local_rest(ctx, qf); t3 = time.perf_counter() - t - t1 - t2
        memo = {}
        key = (w._version, repr(w.env.activity), w.env.gravity)
        t = time.perf_counter()
        if a and b and cl:
            memo[key] = True
        hit = memo.get((w._version, repr(w.env.activity), w.env.gravity), False)
        tm = time.perf_counter() - t
        print(f"{name} {H}x{W}: present step {tp:.2f} s; detector: flow quiet {a} ({t1:.2f} s), diffusion at rest {b} "
              f"({t2:.2f} s), local at rest {cl} ({t3:.2f} s); total {t1 + t2 + t3:.2f} s; memoised rest lookup {tm * 1e6:.0f} us "
              f"(hit {hit})")


if __name__ == "__main__":
    main()
