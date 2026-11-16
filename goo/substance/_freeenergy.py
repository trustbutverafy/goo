"""The mixing free energy of a film's cell, and its derivatives (the kernel of goo.substance.maps.mixing).

A cell holds liquids (substances without a morphology, a continuous phase) and particle populations. Its mixing
free energy per area is homogeneous of degree 1 in the amounts q (m), the sum of three parts:

  (1) the liquids' solution, among the liquids alone (particles are not solutes: they hold no solvent):
        F_sol = k T sum_l c_l q_l ln(q_l / L) + (1/2) q_l^T B q_l / L          L = sum_l q_l, c_l = 1 / v_l
      Flory-Huggins with v_l the molecular volume (c = 0 for a macromolecular phase). B_ij is the interaction
      energy density of a pair of liquids: the Hansen distance (sqrt d_i - sqrt d_j)^2 + (1/4)(sqrt p_i - sqrt p_j)^2
      + (1/4)(sqrt h_i - sqrt h_j)^2 when both give their polar and hydrogen-bonding parts, else Hildebrand-Scatchard
      (sqrt c_i - sqrt c_j)^2 when both have a cohesive energy density, else 0.
  (2) the particles' ideal (translational) entropy, k T sum_p c_p q_p ln(q_p / h), c_p = 1 / v_p the number of
      particles per volume of the population (h = L + P). Per volume it is ~1e-8 of a molecular liquid's.
  (3) the particles' excluded volume (hard spheres), each population against the liquid it sits in:
        F_hs,p = k T c_p q_p psi_p(x),   x = q_p / (q_p + L)
      with the excess free energy per particle psi whose compressibility factor is
        Z = 1 + x psi'(x) = 1 + 4 x (1 + a x) / (1 - x / phi_m),   a = 5/2 - 1 / phi_m
      exact to the second and third virial coefficients (Z = 1 + 4 x + 10 x^2 + ...) and, at the population's packing
      fraction phi_m (Desmond & Weeks from its own size distribution), diverging as (phi_m - x)^-1, as free-volume
      theory has it and as Kamien & Liu (2007) fit the metastable branch. Within 4 % of Carnahan-Starling up to
      x = 0.3 and 11 % at x = 0.5. Its osmotic pressure Pi = k T c_p x^2 psi'(x) rises without bound toward phi_m;
      the film does not compress its particles beyond the pressure at which air enters the pores between them,
      the Laplace pressure of the menisci Pi_e = gamma * 6 phi_m / ((1 - phi_m) d32) (hydraulic radius of the packed
      bed, perfect wetting; gamma the largest surface tension among the film's liquid substances, so that the
      function is the same in every cell). Beyond x_e, where Pi(x_e) = Pi_e, F_hs continues linearly (voids open):
      F_hs,p = A_p q_p - Pi_e L, so the liquid's potential there is -Pi_e, and the term is convex and C^1 everywhere.
      A linear term -A_p q_p (A_p = F_hs per particle volume at x_e plus Pi_e (1/x_e - 1)) makes a dry powder's
      free energy 0, as every pure substance's is.

Convex and concave parts (goo.world's convex splitting): (1) less its interaction, (2), (3) and
(beta / 2)(q_l^T q_l / L - L) are convex; the rest of the interaction, (1/2) q_l^T (B - beta I) q_l / L + (beta/2) L,
is concave, with beta = max(0, largest eigenvalue of B over the film's liquids) (one number per set of substances).

Everything is computed on the substances present in each cell, once per distinct composition (exact equality of the
volume fractions and the temperature), and scaled by h: f = h f(phi), mu = mu(phi), hess = hess(phi) / h.
"""
import numpy as np
from goo.units import k_B, T_room

SPHERICAL = ("sphere", "coated", "hollow")
DW_DELTA_MAX = 0.6


def hs_a(phim):
    return 2.5 - 1.0 / phim


def hs_psi(x, phim):
    """psi(x): excess free energy per particle / kT."""
    a = hs_a(phim)
    return -4 * a * phim * x - 4 * phim * (1 + a * phim) * np.log1p(-x / phim)


def hs_dpsi(x, phim):
    return 4 * (1 + hs_a(phim) * x) / (1 - x / phim)


def hs_d2psi(x, phim):
    a = hs_a(phim)
    r = 1 - x / phim
    return 4 * a / r + 4 * (1 + a * x) / (phim * r * r)


def hs_Z(x, phim):
    return 1 + x * hs_dpsi(x, phim)


def population_packing(s):
    """phi_m of one spherical population alone: Desmond & Weeks (2014) on its number distribution of diameters."""
    z = s.morph.size
    m1, m2, m3 = (z.number_moment(k) for k in (1, 2, 3))
    sd = np.sqrt(max(m2 - m1 * m1, 0.0))
    delta = sd / m1 if m1 > 0 else 0.0
    skew = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / sd ** 3 if sd > 0 else 0.0
    dl = min(delta, DW_DELTA_MAX)
    return float(min(0.634 + 0.0658 * dl + 0.0398 * skew * dl * dl, 0.87))


def hs_xe(Phat, phim):
    """x_e solving x^2 psi'(x) = Phat (the reduced entry pressure) on (0, phi_m); vectorised over Phat."""
    Phat = np.asarray(Phat, float)
    lo = np.zeros_like(Phat); hi = np.full_like(Phat, phim)
    for _ in range(200):
        m = 0.5 * (lo + hi)
        g = 4 * m * m * (1 + hs_a(phim) * m) - Phat * (1 - m / phim)
        lo = np.where(g < 0, m, lo); hi = np.where(g < 0, hi, m)
    return np.where(Phat > 0, 0.5 * (lo + hi), 0.0)


def pair_interaction(si, sj):
    """B_ij (J/m^3) of two liquids, and which rule gave it ("hansen", "hildebrand", None)."""
    pi, pj = si.cohesion_parts, sj.cohesion_parts
    if pi is not None and pj is not None:
        d = (np.sqrt(max(pi[0], 0.0)) - np.sqrt(max(pj[0], 0.0))) ** 2
        p = (np.sqrt(max(pi[1], 0.0)) - np.sqrt(max(pj[1], 0.0))) ** 2
        h = (np.sqrt(max(pi[2], 0.0)) - np.sqrt(max(pj[2], 0.0))) ** 2
        return d + 0.25 * p + 0.25 * h, "hansen"
    ci, cj = si.cohesion, sj.cohesion
    if ci is not None and cj is not None:
        return (np.sqrt(max(ci, 0.0)) - np.sqrt(max(cj, 0.0))) ** 2, "hildebrand"
    return 0.0, None


class MixSet:
    """What the free energy needs of a tuple of substances, computed once per tuple."""

    _cache = {}

    @classmethod
    def of(cls, subs):
        subs = tuple(subs)
        if subs not in cls._cache:
            cls._cache[subs] = cls(subs)
        return cls._cache[subs]

    def __init__(self, subs):
        self.subs = subs
        S = len(subs)
        self.S = S
        self.part = np.array([s.morph is not None for s in subs], bool)
        v = np.array([s.molecular_volume for s in subs], float)
        self.c = np.where(np.isfinite(v) & (v > 0), 1.0 / np.where(v > 0, v, 1.0), 0.0)
        B = np.zeros((S, S))
        for i in range(S):
            for j in range(S):
                if i != j and not self.part[i] and not self.part[j]:
                    B[i, j] = pair_interaction(subs[i], subs[j])[0]
        self.B = B
        li = np.flatnonzero(~self.part)
        self.beta = max(0.0, float(np.linalg.eigvalsh(B[np.ix_(li, li)]).max())) if li.size else 0.0
        gs = [s.thermo.gamma for s in subs if s.morph is None and s.thermo.gamma is not None and s.thermo.gamma > 0]
        self.gamma_entry = float(max(gs)) if gs else 0.0
        self.phim = np.full(S, np.nan); self.Pi_e = np.zeros(S)
        for i, s in enumerate(subs):
            if self.part[i]:
                self.phim[i] = population_packing(s) if s.morph.kind in SPHERICAL else 0.634
                d32 = s.morph.size.sauter()
                self.Pi_e[i] = self.gamma_entry * 6 * self.phim[i] / ((1 - self.phim[i]) * d32) if d32 > 0 else 0.0
        self._xe = {}

    def entry(self, i, T):
        """(x_e, A) of population i at temperature(s) T [n]: where its excess osmotic pressure reaches Pi_e, and the
        linear term that makes its dry powder's free energy 0."""
        T = np.asarray(T, float)
        u, inv = np.unique(T, return_inverse=True)
        key = (i, u.tobytes())
        if key not in self._xe:
            kTc = k_B * u * self.c[i]
            phim = self.phim[i]
            Phat = self.Pi_e[i] / kTc
            xe = hs_xe(Phat, phim)
            with np.errstate(divide="ignore", invalid="ignore"):
                A = np.where(xe > 0, kTc * hs_psi(xe, phim) + self.Pi_e[i] * (1 / np.where(xe > 0, xe, 1) - 1), 0.0)
            self._xe[key] = (xe, A)
        xe, A = self._xe[key]
        return xe[inv].reshape(T.shape), A[inv].reshape(T.shape)


def fh_hess(c, x, tot):
    """[K, K, n] Hessian of sum_l c_l x_l ln(x_l / tot) (tot = sum x), written without cancellation: with y = x / tot,
    diagonal (c_l o_l^2 / y_l + sum_{k != l} c_k y_k) / tot, off-diagonal (-c_l o_l - c_m o_m + sum_{k != l, m} c_k
    y_k) / tot, o_l = sum_{k != l} y_k (= 1 - y_l). Exactly zero for one component."""
    K = x.shape[0]
    y = x / tot
    cy = c[:, None] * y
    eye = np.eye(K, dtype=bool)
    o = np.array([y[np.arange(K) != l].sum(0) for l in range(K)]) if K > 1 else np.zeros_like(y)
    H = np.zeros((K, K) + x.shape[1:])
    for l in range(K):
        others = np.arange(K) != l
        H[l, l] = (np.where(c[l] > 0, c[l] * o[l] ** 2 / np.where(y[l] > 0, y[l], 1.0), 0.0) + cy[others].sum(0)) / tot
        for m in range(l + 1, K):
            rest = others & (np.arange(K) != m)
            v = (-c[l] * o[l] - c[m] * o[m] + cy[rest].sum(0)) / tot
            H[l, m] = v; H[m, l] = v
    return H


def quad_hess(x, tot):
    """[K, K, n] Hessian of (1/2)(sum x^2 / tot - tot) (tot = sum x), without cancellation: with y = x / tot,
    diagonal (o_l^2 + sum_{k != l} y_k^2) / tot, off-diagonal (-y_l o_l - y_m o_m + sum_{k != l, m} y_k^2) / tot.
    Exactly zero for one component."""
    K = x.shape[0]
    y = x / tot
    y2 = y * y
    o = np.array([y[np.arange(K) != l].sum(0) for l in range(K)]) if K > 1 else np.zeros_like(y)
    H = np.zeros((K, K) + x.shape[1:])
    for l in range(K):
        others = np.arange(K) != l
        H[l, l] = (o[l] ** 2 + y2[others].sum(0)) / tot
        for m in range(l + 1, K):
            rest = others & (np.arange(K) != m)
            v = (-y[l] * o[l] - y[m] * o[m] + y2[rest].sum(0)) / tot
            H[l, m] = v; H[m, l] = v
    return H


def kernel(ms, k, x, T, need_hess=True):
    """Mixing on rows x [len(k), n]: the amounts (any scale) of the substances k, exactly those present in them.

    Returns f [n], mu [S, n], hess [len(k), len(k), n] (or None), f_x [n], mu_x [S, n] at these amounts (not
    scaled): the absent substances' mu is -inf where they have a mixing unit (c > 0), else their finite derivative;
    their mu_x the concave part's derivative (0 for particles)."""
    S = ms.S
    n = x.shape[1]
    kT = k_B * np.asarray(T, float) * np.ones(n)
    k = np.asarray(k)
    K = k.size
    part, c = ms.part, ms.c
    lq = ~part[k]; pq = part[k]
    h = x.sum(0)
    Lq = x[lq].sum(0) if lq.any() else np.zeros(n)
    hasL = Lq > 0
    Ls = np.where(hasL, Lq, 1.0)
    ck = c[k]
    f = np.zeros(n); fx = np.zeros(n)
    mu = np.zeros((S, n)); mux = np.zeros((S, n))
    H = np.zeros((K, K, n)) if need_hess else None
    # absent liquids that have no mixing unit: their finite derivative; absent with a unit: -inf
    absent = np.setdiff1d(np.arange(S), k)
    abs_mac = absent[(~part[absent]) & (c[absent] == 0)]
    li_k = k[lq]
    # ---------------------------------------------------------------- (1) solution among the liquids, and beta
    if lq.any():
        xl = x[lq]; cl = ck[lq]
        with np.errstate(divide="ignore", invalid="ignore"):
            ln = np.log(xl / Ls)
        C = (cl[:, None] * xl).sum(0)
        f += kT * np.sum(np.where(cl[:, None] > 0, cl[:, None] * xl * ln, 0.0), 0)
        mul = kT * (np.where(cl[:, None] > 0, cl[:, None] * (ln + 1.0), 0.0) - C / Ls)
        beta = ms.beta
        q2 = (xl * xl).sum(0)
        f += 0.5 * beta * (q2 / Ls - Lq)
        mul += beta * (xl / Ls - 0.5 * q2 / Ls ** 2 - 0.5)
        Bl = ms.B[np.ix_(li_k, li_k)] - beta * np.eye(li_k.size)
        Mx = Bl @ xl
        qMq = (xl * Mx).sum(0)
        fx += 0.5 * qMq / Ls + 0.5 * beta * Lq
        muxl = Mx / Ls - 0.5 * qMq / Ls ** 2 + 0.5 * beta
        mu[li_k] = np.where(hasL, mul, 0.0); mux[li_k] = np.where(hasL, muxl, 0.0)
        if abs_mac.size:
            Ba = ms.B[np.ix_(abs_mac, li_k)] @ xl
            mu[abs_mac] = np.where(hasL, kT * (-C / Ls) + beta * (-0.5 * q2 / Ls ** 2 - 0.5), 0.0)
            mux[abs_mac] = np.where(hasL, Ba / Ls - 0.5 * qMq / Ls ** 2 + 0.5 * beta, 0.0)
        if need_hess:
            ia = np.flatnonzero(lq)
            Hl = kT * fh_hess(cl, xl, Ls)
            Hl += beta * quad_hess(xl, Ls)
            H[np.ix_(ia, ia)] = np.where(hasL, Hl, 0.0)
    # ---------------------------------------------------------------- (2) particles' ideal entropy
    if pq.any():
        cp = np.where(pq, ck, 0.0)
        N = (cp[:, None] * x).sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lnp = np.log(np.where(pq[:, None], x, 1.0) / h)
        f += kT * np.sum(np.where(pq[:, None], cp[:, None] * x * lnp, 0.0), 0)
        mu[k] += kT * (np.where(pq[:, None], cp[:, None] * (lnp + 1.0), 0.0) - N / h)
        if abs_mac.size:
            mu[abs_mac] += kT * (-N / h)
        if need_hess:
            H += kT * fh_hess(cp, x, h)
        # ------------------------------------------------------------ (3) excluded volume, per population
        lidx = np.flatnonzero(lq)
        for a in np.flatnonzero(pq):
            i = k[a]
            phim = ms.phim[i]
            xe, A = ms.entry(i, kT / k_B)
            qa = x[a]
            u = qa + Lq
            xr = qa / u
            hs = (xr < xe) & hasL
            xs = np.where(hs, xr, 0.0)
            kTc = kT * c[i]
            psi, d1 = hs_psi(xs, phim), hs_dpsi(xs, phim)
            Pi = np.where(hs, kTc * xs * xs * d1, ms.Pi_e[i])            # the liquid's suction
            f += np.where(hs, kTc * qa * psi - A * qa, -ms.Pi_e[i] * Lq)
            mu[i] += np.where(hs, kTc * (psi + xs * (1 - xs) * d1) - A, 0.0)
            mu[li_k] += -Pi
            if abs_mac.size:
                mu[abs_mac] += -Pi
            if need_hess:
                aa = np.where(hs, kTc * (2 * d1 + xs * hs_d2psi(xs, phim)) / np.where(hs, u, 1.0), 0.0)
                v = np.zeros((K, n)); v[a] = np.where(hs, Lq / u, 0.0); v[lidx] = -xs
                H += aa * v[:, None] * v[None, :]
    # absent substances with a mixing unit: -inf
    abs_unit = absent[c[absent] > 0]
    mu[abs_unit] = -np.inf
    return f, mu, H, fx, mux


def group(cols):
    """(representatives, inverse) of the distinct rows of cols [n, m] (float64), by exact equality of their bits:
    a 64-bit hash sorts them, and every row is then compared with its group's representative; rows whose hash
    collides with another's are grouped by an exact sort of their bytes instead."""
    cols = np.ascontiguousarray(cols)
    n, m = cols.shape
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    b = cols.view(np.uint64)
    mult = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5,
                     0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53, 0x85EBCA77C2B2AE63], np.uint64)
    hsh = np.zeros(n, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(m):
            hsh ^= b[:, j] * mult[j % mult.size] + np.uint64(j + 1)
            hsh = (hsh ^ (hsh >> np.uint64(29))) * np.uint64(0xBF58476D1CE4E5B9)
    _, first, inv = np.unique(hsh, return_index=True, return_inverse=True)
    inv = inv.ravel()
    rep = first[inv]
    if np.array_equal(b, b[rep]):
        return first, inv
    v = cols.view(np.dtype((np.void, 8 * m))).ravel()
    _, first, inv = np.unique(v, return_index=True, return_inverse=True)
    return first, inv.ravel()


def _distinct(q, wi, hw, T, N):
    """The substances present somewhere (act), the distinct (volume fractions, T) of the wet cells wi on them, and
    their presence patterns: (act, U [len(act), nu], TU [nu], first, inv, patterns, pattern of each distinct row)."""
    act = np.flatnonzero((q[:, wi] > 0).any(1))
    phi = q[np.ix_(act, wi)] / hw
    Tw = np.broadcast_to(np.asarray(T, float), (N,))[wi]
    first, inv = group(np.concatenate([phi, Tw[None]], 0).T)
    U = phi[:, first]; TU = Tw[first]
    bits = np.zeros(first.size, np.int64)
    for a in range(act.size):
        bits |= (U[a] > 0).astype(np.int64) << a
    pats, pinv = np.unique(bits, return_inverse=True)
    return act, U, TU, first, inv, pats, pinv.ravel()


def _rows(U, sel, rows):
    return U[np.ix_(sel, rows)]


def evaluate(ms, q, T, need_hess=True):
    """f [N], mu [S, N], hess [S, S, N] (or None), f_x [N], mu_x [S, N] on cells q [S, N] at T [N]."""
    S, N = q.shape
    h = q.sum(0)
    wet = h > 0
    f = np.zeros(N); fx = np.zeros(N)
    mu = np.zeros((S, N)); mux = np.zeros((S, N))
    mu[:, ~wet] = np.where(ms.c > 0, -np.inf, 0.0)[:, None]
    H = np.zeros((S, S, N)) if need_hess else None
    wi = np.flatnonzero(wet)
    if wi.size == 0:
        return f, mu, H, fx, mux
    hw = h[wi]
    act, U, TU, first, inv, pats, pinv = _distinct(q, wi, hw, T, N)
    nu = first.size
    fU = np.zeros(nu); fxU = np.zeros(nu); muU = np.zeros((S, nu)); muxU = np.zeros((S, nu))
    cell_pat = pinv[inv]                                        # pattern of each wet cell
    for pi_, pb in enumerate(pats):
        rows = np.flatnonzero(pinv == pi_)
        sel = [a for a in range(act.size) if (int(pb) >> a) & 1]
        k = act[sel]
        fr, mr, Hr, fxr, mxr = kernel(ms, k, _rows(U, sel, rows), TU[rows], need_hess)
        fU[rows] = fr; fxU[rows] = fxr; muU[:, rows] = mr; muxU[:, rows] = mxr
        if need_hess:
            cells = np.flatnonzero(cell_pat == pi_)
            pos = np.full(nu, -1); pos[rows] = np.arange(rows.size)
            j = pos[inv[cells]]
            ih = 1.0 / hw[cells]
            dst = wi[cells]
            whole = dst.size == N
            for a in range(k.size):
                for b in range(a, k.size):
                    v = Hr[a, b, j] * ih
                    if whole:
                        H[k[a], k[b]] = v
                        if b != a:
                            H[k[b], k[a]] = v
                    else:
                        H[k[a], k[b], dst] = v
                        if b != a:
                            H[k[b], k[a], dst] = v
    f[wi] = fU[inv] * hw; fx[wi] = fxU[inv] * hw
    mu[:, wi] = muU[:, inv]; mux[:, wi] = muxU[:, inv]
    return f, mu, H, fx, mux


def hvp(ms, q, T, dq, chunk=65536):
    """The convex Hessian applied to dq [S, N], per cell, without forming [S, S, N]: blocks of the present substances
    on the distinct compositions, applied chunk by chunk."""
    S, N = q.shape
    h = q.sum(0)
    out = np.zeros((S, N))
    wi = np.flatnonzero(h > 0)
    if wi.size == 0:
        return out
    hw = h[wi]
    act, U, TU, first, inv, pats, pinv = _distinct(q, wi, hw, T, N)
    nu = first.size
    cell_pat = pinv[inv]
    for pi_, pb in enumerate(pats):
        rows = np.flatnonzero(pinv == pi_)
        sel = [a for a in range(act.size) if (int(pb) >> a) & 1]
        k = act[sel]
        cells = np.flatnonzero(cell_pat == pi_)
        pos = np.full(nu, -1); pos[rows] = np.arange(rows.size)
        for c0 in range(0, rows.size, chunk):
            r = rows[c0:c0 + chunk]
            Hr = kernel(ms, k, _rows(U, sel, r), TU[r], True)[2]
            cs = cells[(pos[inv[cells]] >= c0) & (pos[inv[cells]] < c0 + chunk)]
            j = pos[inv[cs]] - c0
            d = dq[np.ix_(k, wi[cs])]
            acc = np.zeros(d.shape)
            for b in range(k.size):                              # a fixed order of summation: element-wise bits
                acc += Hr[:, b, j] * d[b]
            out[np.ix_(k, wi[cs])] = acc / hw[cs]
    return out
