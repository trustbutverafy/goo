"""Prototypes (iii) and (iv): the per-cell local exchange solve, and the mixing map, as numba kernels over all cells.

    PYTHONPATH=$PWD python bench/numba_kernels.py mixing [n] [S]   maps.mixing against a fused kernel (1 and 2 threads)
    PYTHONPATH=$PWD python bench/numba_kernels.py local H W        the present local substep against a jitted one, on the
                                                                   quiet scene (bench/scenes.py), same inputs

The kernels copy the formulas of goo/substance/maps.py (mixing: Flory-Huggins + Hildebrand-Scatchard, convex split)
and goo/world/local.py (exchange of one volatile, Newton per cell with the same update, limiter and stopping rule).
They are element-wise: one cell's answer depends on that cell alone, so a window and the panel give the same bits
(O5), and the parallel loop (prange) assigns cells to threads without any reduction (O6 holds for any thread count).
"""
import sys, os, time
import numpy as np
import numba as nb
from numba import njit, prange

sys.path.insert(0, os.path.dirname(__file__))


@njit(cache=True, fastmath=False)
def _mix_cell(q, c, kT, beta, B, S, f_out, mu_out, H_out, fx_out, mux_out, j):
    """One cell of maps.mixing: writes f, mu [S], hess [S, S] (convex part) and f_x, mu_x (concave part)."""
    h = 0.0
    for i in range(S):
        h += q[i]
    if h <= 0.0:
        f_out[j] = 0.0; fx_out[j] = 0.0
        for i in range(S):
            mu_out[i, j] = 0.0; mux_out[i, j] = 0.0
            for k in range(S):
                H_out[i, k, j] = 0.0
        return
    cphi = 0.0
    fent = 0.0
    qq = 0.0
    for i in range(S):
        cphi += c[i] * (q[i] / h)
        qq += q[i] * q[i]
        if q[i] > 0.0:
            fent += c[i] * q[i] * np.log(q[i] / h)
    fent *= kT
    # concave part: M = B - beta I
    qBq = 0.0
    for i in range(S):
        s = 0.0
        for k in range(S):
            s += B[i, k] * q[k]
        qBq += q[i] * s
    qMq_x = qBq - beta * qq
    qMq_c = beta * qq
    f_out[j] = fent + 0.5 * qMq_c / h - 0.5 * beta * h
    fx_out[j] = 0.5 * qMq_x / h + 0.5 * beta * h
    for i in range(S):
        Mq_x = -beta * q[i]
        for k in range(S):
            Mq_x += B[i, k] * q[k]
        mux_out[i, j] = Mq_x / h - 0.5 * qMq_x / (h * h) + 0.5 * beta
        muc = beta * q[i] / h - 0.5 * qMq_c / (h * h) - 0.5 * beta
        if q[i] > 0.0:
            ment = kT * ((c[i] * (np.log(q[i] / h) + 1.0) if c[i] > 0.0 else 0.0) - cphi)
            mu_out[i, j] = ment + muc
        else:
            mu_out[i, j] = -np.inf if c[i] > 0.0 else muc
        for k in range(S):
            if q[i] > 0.0 and q[k] > 0.0:
                d = (c[i] / q[i]) if i == k else 0.0
                He = kT * (d - (c[i] + c[k]) / h + cphi / h)
                Hc = (beta / h if i == k else 0.0) - (beta * q[i] + beta * q[k]) / (h * h) + qMq_c / (h * h * h)
                H_out[i, k, j] = He + Hc
            else:
                H_out[i, k, j] = 0.0


@njit(cache=True, inline="always")
def _mix_col(Q, j, c, kT, beta, B, S, f_out, mu_out, H_out, fx_out, mux_out):
    """_mix_cell reading cell j of Q [S, n] in place (no per-cell allocation)."""
    h = 0.0
    for i in range(S):
        h += Q[i, j]
    if h <= 0.0:
        f_out[j] = 0.0; fx_out[j] = 0.0
        for i in range(S):
            mu_out[i, j] = 0.0; mux_out[i, j] = 0.0
            for k in range(S):
                H_out[i, k, j] = 0.0
        return
    cphi = 0.0; fent = 0.0; qq = 0.0; qBq = 0.0
    for i in range(S):
        qi = Q[i, j]
        cphi += c[i] * (qi / h)
        qq += qi * qi
        if qi > 0.0:
            fent += c[i] * qi * np.log(qi / h)
        s = 0.0
        for k in range(S):
            s += B[i, k] * Q[k, j]
        qBq += qi * s
    fent *= kT
    qMq_x = qBq - beta * qq
    qMq_c = beta * qq
    f_out[j] = fent + 0.5 * qMq_c / h - 0.5 * beta * h
    fx_out[j] = 0.5 * qMq_x / h + 0.5 * beta * h
    for i in range(S):
        qi = Q[i, j]
        Mq_x = -beta * qi
        for k in range(S):
            Mq_x += B[i, k] * Q[k, j]
        mux_out[i, j] = Mq_x / h - 0.5 * qMq_x / (h * h) + 0.5 * beta
        muc = beta * qi / h - 0.5 * qMq_c / (h * h) - 0.5 * beta
        if qi > 0.0:
            ment = kT * ((c[i] * (np.log(qi / h) + 1.0) if c[i] > 0.0 else 0.0) - cphi)
            mu_out[i, j] = ment + muc
        else:
            mu_out[i, j] = -np.inf if c[i] > 0.0 else muc
        for k in range(S):
            qk = Q[k, j]
            if qi > 0.0 and qk > 0.0:
                d = (c[i] / qi) if i == k else 0.0
                He = kT * (d - (c[i] + c[k]) / h + cphi / h)
                Hc = (beta / h if i == k else 0.0) - (beta * qi + beta * qk) / (h * h) + qMq_c / (h * h * h)
                H_out[i, k, j] = He + Hc
            else:
                H_out[i, k, j] = 0.0


@njit(parallel=True, cache=True)
def mixing_nb(q, c, kT, beta, B):
    S, n = q.shape
    f = np.empty(n); fx = np.empty(n); mu = np.empty((S, n)); mux = np.empty((S, n)); H = np.empty((S, S, n))
    for j in prange(n):
        _mix_col(q, j, c, kT, beta, B, S, f, mu, H, fx, mux)
    return f, mu, H, fx, mux


def mixing_params(subs, T=None):
    from goo.substance import maps
    from goo.units import k_B
    F = maps._F(subs, np.zeros((len(subs), 1, 1)), np.zeros((1, 1)), maps.T_room if T is None else T)
    v = np.array([s.molecular_volume for s in subs])
    c = np.where(np.isfinite(v), 1.0 / v, 0.0)
    Bm = maps._interaction(F)
    beta = max(0.0, float(np.linalg.eigvalsh(Bm).max()))
    return c, k_B * float(F.T.ravel()[0]), beta, Bm


# ------------------------------------------------------------------------- the local exchange solve (one volatile)
@njit(parallel=True, cache=True)
def local_scalar_nb(q0, i0, ke, base, mux0, sig, rho, a_, gn, menv_i0, c, kT, beta, B, maxit):
    """local.py's Newton for cells with one exchanging substance i0 (the quiet scene): residual
    R = e - ke (mu_i0(q0 - e) - menv), J = 1 + ke (hess_i0i0 + gJ_i0i0 + sig), step limited to 0.9 of what is held,
    stop when the full step is taken and |d| <= 1e-13 h0. Returns e [n] and iterations [n]. Cells are split into
    fixed blocks (scratch allocated once per block); each cell's arithmetic is independent of the block it is in."""
    S, n = q0.shape
    e = np.zeros(n)
    its = np.zeros(n, np.int64)
    gJ = gn * (rho[i0] + a_) * (rho[i0] + a_) / (4 * a_)
    BL = 4096
    nblk = (n + BL - 1) // BL
    for b in prange(nblk):
        Qp = np.empty((S, 1))
        f1 = np.empty(1); fx1 = np.empty(1); mu1 = np.empty((S, 1)); mux1 = np.empty((S, 1)); H1 = np.empty((S, S, 1))
        for j in range(b * BL, min(n, (b + 1) * BL)):
            h0 = 0.0
            for i in range(S):
                h0 += q0[i, j]
            ej = 0.0
            for it in range(maxit):
                for i in range(S):
                    Qp[i, 0] = q0[i, j]
                Qp[i0, 0] = q0[i0, j] - ej
                _mix_col(Qp, 0, c, kT, beta, B, S, f1, mu1, H1, fx1, mux1)
                hp = 0.0; mp = 0.0
                for i in range(S):
                    hp += Qp[i, 0]; mp += rho[i] * Qp[i, 0]
                mu = (mu1[i0, 0] + mux0[i0, j] + base[i0, j] + gn * (mp + a_ * hp) * (rho[i0] + a_) / (4 * a_)
                      + sig[j] * hp)
                R = ej - ke[j] * (mu - menv_i0)
                Jd = ke[j] * (H1[i0, i0, 0] + gJ + sig[j]) + 1.0
                d = -R / Jd
                t = 1.0
                if d > 0.0 and Qp[i0, 0] > 0.0:
                    lim = 0.9 * Qp[i0, 0] / d
                    if lim < t:
                        t = lim
                ej += t * d
                its[j] = it + 1
                if t == 1.0 and abs(d) <= 1e-13 * h0 + 1e-300:
                    break
            e[j] = ej
    return e, its


def capture_local(world, dt):
    import goo.world.world as WW
    rec = {}
    orig = WW.local

    def wrapped(ctx, q, held, dt_):
        t = time.perf_counter()
        out = orig(ctx, q, held, dt_)
        rec.update(ctx=ctx, q=q.copy(), held=held.copy(), dt=dt_, out=out, t=time.perf_counter() - t)
        return out
    WW.local = wrapped
    try:
        world.step(dt)
    finally:
        WW.local = orig
    return rec


def local_inputs(ctx, q, dt):
    """The per-cell inputs of local.py (copied from its first lines): one exchanging substance, no sorption."""
    from goo.world.common import coeff, mu_wet, mixing
    from goo.world.grid import capillary_pressure
    G = ctx.grid
    dx = G.dx
    h = q.sum(0)
    wet = (h > 0) & (ctx.r > 0)
    k = coeff(ctx, "exchange", q, None, ctx.env)[0]
    eon = (k > 0) & (q > 0) & wet & ctx.has_env[:, None]
    cells = np.flatnonzero(eon.any(0))
    ie = np.flatnonzero(eon[:, cells].any(1))
    assert ie.size == 1, "this prototype covers one exchanging substance"
    i0 = int(ie[0])
    gx, gy, gn = ctx.g
    a_, rho = ctx.a, ctx.rho
    q0 = np.ascontiguousarray(q[:, cells])
    h0, m0 = q0.sum(0), rho @ q0
    ke = np.where(eon[i0, cells], dt * ctx.r[cells] * k[i0, cells], 0.0)
    s = ctx.zb + h
    Pc = capillary_pressure(G, ctx.gam_f, s)[cells]
    sig = G.cellsum(ctx.gam_f)[cells] / dx ** 2
    mw = mu_wet(ctx, h0, cells)
    psi = -(gx * G.X[cells] + gy * G.Y[cells])
    zb = ctx.zb[cells]
    gexp = -gn * (m0 - a_ * h0)[None, :] * (rho[:, None] - a_) / (4 * a_)
    base = (gn * rho[:, None] * zb[None, :] + gexp + (Pc - sig * h0 + mw)[None, :] + rho[:, None] * psi[None, :])
    mux0 = mixing(ctx, q0)[4]
    return dict(cells=cells, i0=i0, q0=q0, ke=ke, base=np.ascontiguousarray(base), mux0=np.ascontiguousarray(mux0),
                sig=sig, rho=np.asarray(rho, float), a=float(a_), gn=float(gn), menv=float(ctx.mu_env[i0]),
                single=(q0 > 0).sum(0) == 1)


def main():
    mode = sys.argv[1]
    if mode == "mixing":
        from goo.substance import maps, library as L
        from goo.film import Film
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 786432
        S = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        names = ["water", "gum_arabic", "rutile", "soot", "ultramarine", "rose", "cadmium_sulfide", "hematite",
                 "green", "dioxazine", "calcite"]
        subs = tuple(L.BY_NAME[x] for x in names[:S] if x in L.BY_NAME)
        S = len(subs)
        rng = np.random.default_rng(0)
        q = rng.random((S, n)) * 1e-5
        if S > 4:                                  # a cell holds 4 of S: water, gum and two pigments chosen at random
            keep = np.zeros((S, n), bool); keep[0] = keep[1] = True
            pick = rng.integers(2, S, size=(2, n)); keep[pick[0], np.arange(n)] = True; keep[pick[1], np.arange(n)] = True
            q = np.where(keep, q, 0.0)
        c, kT, beta, B = mixing_params(subs)
        film = Film(subs, q[:, None, :])
        t = time.perf_counter(); ref = maps.mixing(film).value; tm = time.perf_counter() - t
        mixing_nb(q[:, :10].copy(), c, kT, beta, B)          # compile
        out = {}
        for th in (1, 2):
            nb.set_num_threads(th)
            best = 1e9
            for _ in range(3):
                t = time.perf_counter(); res = mixing_nb(q, c, kT, beta, B); best = min(best, time.perf_counter() - t)
            out[th] = best
        f, mu, H, fx, mux = res
        pres = q > 0
        rf = np.max(np.abs(f - ref.f.reshape(n)) / (np.abs(ref.f.reshape(n)) + 1e-30))
        m = pres
        rmu = np.max(np.abs(mu - ref.mu.reshape(S, n))[m] / (np.abs(ref.mu.reshape(S, n))[m] + 1e-30))
        pp = pres[:, None] & pres[None]
        rH = np.max(np.abs(H - ref.hess.reshape(S, S, n))[pp] / (np.abs(ref.hess.reshape(S, S, n))[pp] + 1e-30))
        same = np.mean(H[pp] == ref.hess.reshape(S, S, n)[pp])
        print(f"mixing, n {n} cells, S {S}: maps.mixing {tm:.3f} s; numba 1 thread {out[1]:.3f} s, 2 threads {out[2]:.3f} s "
              f"({tm / out[1]:.0f}x, {tm / out[2]:.0f}x); max rel diff f {rf:.1e} mu {rmu:.1e} hess {rH:.1e}; "
              f"hess entries bit-identical {same:.3f}")
    elif mode == "local":
        from scenes import quiet
        H, W = int(sys.argv[2]), int(sys.argv[3])
        w = quiet(H, W)
        w.step(2.0)
        rec = capture_local(w, 2.0)
        ctx, q = rec["ctx"], rec["q"]
        t = time.perf_counter(); I = local_inputs(ctx, q, rec["dt"]); tin = time.perf_counter() - t
        assert not I["single"].any()
        c, kT, beta, B = mixing_params(ctx.substances)
        local_scalar_nb(I["q0"][:, :4].copy(), I["i0"], I["ke"][:4].copy(), I["base"][:, :4].copy(), I["mux0"][:, :4].copy(),
                        I["sig"][:4].copy(), I["rho"], I["a"], I["gn"], I["menv"], c, kT, beta, B, 80)
        res = {}
        for th in (1, 2):
            nb.set_num_threads(th)
            t = time.perf_counter()
            e, its = local_scalar_nb(I["q0"], I["i0"], I["ke"], I["base"], I["mux0"], I["sig"], I["rho"], I["a"],
                                     I["gn"], I["menv"], c, kT, beta, B, 80)
            res[th] = time.perf_counter() - t
        qn_ref = rec["out"][0][:, I["cells"]]
        e_ref = I["q0"][I["i0"]] - qn_ref[I["i0"]]
        rel = np.max(np.abs(e - e_ref) / np.abs(e_ref))
        bits = np.mean(e == e_ref)
        print(f"local, quiet scene {H}x{W}, {I['cells'].size} cells: present local {rec['t']:.3f} s; inputs (numpy) {tin:.3f} s; "
              f"numba Newton 1 thread {res[1]:.3f} s, 2 threads {res[2]:.3f} s; iterations {its.min()}-{its.max()} "
              f"(mean {its.mean():.2f}); max rel diff of e {rel:.1e}; bit-identical {bits:.3f}")


if __name__ == "__main__":
    main()
