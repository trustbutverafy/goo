"""Prototype (v): the diffusion substep as ONE explicit Onsager flux with an exact ray search, admissible by
construction. No Newton, no linear solve.

    PYTHONPATH=$PWD python bench/diffuse_ray.py H W      (quiet scene, bench/scenes.py; compared with goo's diffuse)

Step (DYNAMICS.md §3, §4.2 notation): at the old state, X_i = -(mu_i,R - mu_i,L)/dx (full potentials: mixing,
both parts, plus rho_i psi), upwind omega_i = kappa_i phi_i(donor), Xbar = sum omega X / sum omega, j_i = omega_i
(X_i - Xbar) (sum_i j_i = 0 exactly up to rounding of the flux, so h does not change by more than rounding... see
note), delta = -(dt/dx) div j. Then choose t in [0, t_feasible] minimising
    Phi(t) = F(q0 + t delta) + (t^2 / 2) D1,   D1 = dt dx^2 sum_f sum_i omega_i (X_i - Xbar)^2,
with F's convex mixing part evaluated exactly and the concave part (and gravity, linear at fixed h) linearised at q0
(Eyre). Phi is convex in t, so a few Newton steps on Phi'(t) (each one mixing evaluation) give t*. At t*,
t Phi'(t*) = 0, i.e. <grad F_c(q_t) + grad F_x(q0), q_t - q0> = -D_mech with D_mech = t^2 D1 (the flux t j is driven by
the force t (X - Xbar) through the same L), and convexity gives D_num = -dF - D_mech >= 0: O3 by construction, for any
dt. Positivity: t_feasible keeps every amount >= 0 (donor-proportional outflow). When t* < 1 by more than a
tolerance the step is too long for one explicit flux (the problem is stiff at that dt) and the world would halve it,
as it does now on a failed acceptance condition: the report says t*.

Note on volume: sum_i j_i = sum omega (X - Xbar) = 0 in exact arithmetic; in floating point it is rounding, so h
moves by rounding. The present Newton has the same property (its update is in flux form from the same j).
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.simplefilter("ignore")


def capture(world, dt):
    import goo.world.world as WW
    rec = {}
    orig = WW.diffuse

    def wrapped(ctx, q, dt_):
        t = time.perf_counter()
        out = orig(ctx, q, dt_)
        rec.update(ctx=ctx, q=q.copy(), dt=dt_, out=out[0].copy(), rep=out[1], t=time.perf_counter() - t)
        return out
    WW.diffuse = wrapped
    try:
        world.step(dt)
    finally:
        WW.diffuse = orig
    return rec


def diffuse_ray(ctx, q, dt, mix, newton=4):
    """mix(q) -> (f, mu, hess, f_x, mu_x) on [S, n]. Returns (q1, report)."""
    from goo.world.common import coeff
    G = ctx.grid; dx, N = G.dx, G.N; fL, fR = G.fL, G.fR
    S = q.shape[0]
    h = q.sum(0); wet = h > 0
    fa = np.flatnonzero(wet[fL] & wet[fR])
    Lam = coeff(ctx, "mobility", q, None)
    a, b = fL[fa], fR[fa]
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(wet, q / h, 0.0)
    kap = 0.5 * (h[a] + h[b]) * 0.5 * (Lam[:, a] + Lam[:, b])
    gx, gy, gn = ctx.g
    psi = gn * (ctx.zb + 0.5 * h) - (gx * G.X + gy * G.Y)
    f0, mu0, _, fx0, mux0 = mix(q)
    lin = mux0 + ctx.rho[:, None] * psi[None, :]                  # explicit part of the potential (concave + gravity)
    mu = mu0 + lin
    X = -(mu[:, b] - mu[:, a]) / dx
    # donors: the centred guess's relative force, then upwind (as diffuse.py's first state())
    w = kap * 0.5 * (phi[:, a] + phi[:, b])
    Wg = w.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        up = (X - np.where(Wg > 0, (w * X).sum(0) / Wg, 0.0)) >= 0
    om = kap * np.where(up, phi[:, a], phi[:, b])
    W = om.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Xb = np.where(W > 0, (om * X).sum(0) / W, 0.0)
    j = om * (X - Xb)
    dv = np.empty((S, N))
    for i in range(S):
        dv[i] = np.bincount(a, j[i], N) - np.bincount(b, j[i], N)
    delta = -(dt / dx) * dv
    D1 = dt * dx ** 2 * float(np.sum(om * (X - Xb) ** 2))
    # feasibility: what leaves each cell is proportional to what it holds (upwind), so t <= q / outflow
    out = np.zeros((S, N))
    for i in range(S):
        jp, jm = np.maximum(j[i], 0.0), np.maximum(-j[i], 0.0)
        out[i] = (dt / dx) * (np.bincount(a, jp, N) + np.bincount(b, jm, N))
    with np.errstate(divide="ignore", invalid="ignore"):
        tfeas = float(np.min(np.where(out > 0, q / out, np.inf)))
    dx2 = dx ** 2
    # Phi'(t) = dx^2 sum (mu_c(q0 + t delta) + lin) . delta + t D1 ; Phi''(t) = dx^2 delta^T H_c delta + D1
    g1 = dx2 * float(np.sum(lin * delta))
    t = min(1.0, 0.5 * tfeas)
    hist = []
    for k in range(newton):
        qt = q + t * delta
        ft, mut, Ht, _, _ = mix(qt)
        d1 = dx2 * float(np.sum(np.where(qt > 0, mut, 0.0) * delta)) + g1 + t * D1
        d2 = dx2 * float(np.einsum("in,ijn,jn->", delta, Ht, delta)) + D1
        tn = t - d1 / d2
        hist.append((t, d1))
        t = min(max(tn, 0.0), tfeas * (1 - 1e-12))
    q1 = q + t * delta
    # ledger: dF from the mixing values (convex part exact, concave part linear), gravity linear
    f1 = mix(q1)[0]
    dF = dx2 * (float(np.sum(f1 - f0)) + float(np.sum(lin * (q1 - q))))
    Dm = t * t * D1
    return q1, dict(t=t, tfeas=tfeas, dF=dF, D_diffusion=Dm, D_num=-dF - Dm, hist=hist, faces=fa.size)


def main():
    from scenes import quiet
    from numba_kernels import mixing_nb, mixing_params
    H, W = int(sys.argv[1]), int(sys.argv[2])
    w = quiet(H, W)
    w.step(2.0)
    rec = capture(w, 2.0)
    ctx, q0 = rec["ctx"], rec["q"]
    c, kT, beta, B = mixing_params(ctx.substances)
    mixing_nb(np.ascontiguousarray(q0[:, :8]), c, kT, beta, B)
    present = q0 > 0

    def mix(qq):
        f, mu, Hh, fx, mux = mixing_nb(np.ascontiguousarray(qq), c, kT, beta, B)
        return f, mu, Hh, fx, mux
    t = time.perf_counter(); q1, rep = diffuse_ray(ctx, q0, rec["dt"], mix); tr = time.perf_counter() - t
    dref = rec["out"] - q0
    err = np.linalg.norm(q1 - rec["out"]) / np.linalg.norm(dref)
    pd = rec["rep"]
    print(f"diffusion, quiet scene {H}x{W}: present diffuse {rec['t']:.2f} s (dF {pd['dF']:.4e}, D {pd['D_diffusion']:.4e}, "
          f"D_num {-pd['dF'] - pd['D_diffusion']:.3e}); ray step {tr:.2f} s: t* {rep['t']:.6f} (feasible to {rep['tfeas']:.3g}), "
          f"dF {rep['dF']:.4e} D {rep['D_diffusion']:.4e} D_num {rep['D_num']:.3e}; |q_ray - q_present| / |dq_present| "
          f"= {err:.2e}; min q {q1[present].min():.2e}; newton on t: " + ", ".join(f"t={a:.6f} dPhi={b:.1e}" for a, b in rep["hist"]))


if __name__ == "__main__":
    main()
