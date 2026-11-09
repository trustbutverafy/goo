"""lidar_coherent.py: an FMCW scanner whose spot is a speckle field, not an average.

Each beam's spot is sampled by N scatterers drawn from its (Gaussian) intensity profile, each ray-marched to
its own first hit on the surface. Every scatterer returns a field  a_i exp(j 4 pi nu(t) R_i / c): amplitude
from the 1550 nm reflectance and Lambert's cosine; range R_i from the geometry plus micro-roughness h_r
(hashed per 4 um patch, so the same patch always gives the same phase). With the optical frequency swept
by B over the chirp, that is a beat tone at R_i / dR cycles per chirp (dR = c / 2B, the range resolution)
with optical phase 4 pi R_i / lambda. The receiver sums the fields, adds thermal noise, and reports the
spectral peak (zoomed DTFT + parabolic interpolation). Nothing else is assumed: speckle, speckle-induced
range outliers (Baumann et al. 2014), mixed 'flying' pixels at edges, and reflectance-dependent precision
all come out of that sum.

The ground truth for each beam (true boresight intersection, the true surface under each reported point)
is kept beside the measurement, never mixed into it.
"""
import numpy as np
from goo.sensors.lidar import bilinear, CELL

C = 299_792_458.0


def hash_normal(x, y, pitch, seed):
    """A standard-normal value fixed to each pitch x pitch patch of the surface (x, y in mm)."""
    ix = np.floor(x / pitch).astype(np.int64); iy = np.floor(y / pitch).astype(np.int64)
    def mix(k):
        z = (k + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))
    with np.errstate(over="ignore"):
        k = (ix.astype(np.uint64) * np.uint64(73856093)) ^ (iy.astype(np.uint64) * np.uint64(19349663)) ^ np.uint64(seed * 83492791)
        u1 = (mix(k) >> np.uint64(11)).astype(np.float64) / 2.0 ** 53
        u2 = (mix(k ^ np.uint64(0xABCDEF)) >> np.uint64(11)).astype(np.float64) / 2.0 ** 53
    return np.sqrt(-2 * np.log(np.maximum(u1, 1e-300))) * np.cos(2 * np.pi * u2)


def march(o, d, Z, zmax, zmin, step=0.03, bisect=12):
    t0 = (zmax - o[:, 2]) / d[:, 2]; t1 = (zmin - o[:, 2]) / d[:, 2]
    horiz = np.hypot(d[:, 0], d[:, 1]) / np.maximum(np.abs(d[:, 2]), 1e-9)
    n = int(np.ceil(((zmax - zmin) * horiz.max()) / step)) + 2
    lo = t0.copy(); hi = t1.copy(); found = np.zeros(len(o), bool); prev = t0.copy()
    for s in range(1, n + 1):
        t = t0 + (t1 - t0) * s / n; p = o + t[:, None] * d
        below = (p[:, 2] <= bilinear(Z, p[:, 0], p[:, 1])) & ~found
        lo = np.where(below, prev, lo); hi = np.where(below, t, hi); found |= below; prev = t
    for _ in range(bisect):
        m = 0.5 * (lo + hi); p = o + m[:, None] * d
        b = p[:, 2] <= bilinear(Z, p[:, 0], p[:, 1]); hi = np.where(b, m, hi); lo = np.where(b, lo, m)
    p = o + hi[:, None] * d
    H, W = Z.shape
    inside = (p[:, 0] > 0) & (p[:, 0] < W * CELL) & (p[:, 1] > 0) & (p[:, 1] < H * CELL)
    return hi, found & inside


class Surface:
    """What the World presents to a sensor: height (mm), 1550 nm reflectance, roughness (um)."""
    def __init__(self, Z, nir, rough_um):
        self.Z, self.nir, self.rough = Z, nir, rough_um
        gy, gx = np.gradient(Z, CELL); self.gx, self.gy = gx, gy


class Geometry:
    """Everything about the beams that doesn't depend on the waveform: computed once, measured many ways."""
    def __init__(self, surf, pos, spacing=0.14, spot=0.12, n_sc=32, aim_box=None, seed=0, chunk=600_000):
        rng = np.random.default_rng(seed)
        Z = surf.Z; H, W = Z.shape; pos = np.asarray(pos, float)
        x0, x1, y0, y1 = aim_box or (0.5, W * CELL - 0.5, 0.5, H * CELL - 0.5)
        gx_, gy_ = np.meshgrid(np.arange(x0, x1, spacing), np.arange(y0, y1, spacing))
        zref = float(Z.mean())
        aim = np.stack([gx_.ravel(), gy_.ravel(), np.full(gx_.size, zref)], 1)
        d = aim - pos; d /= np.linalg.norm(d, axis=1, keepdims=True)
        e1 = np.cross(d, [0, 0, 1.0]); e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(d, e1)
        B = len(d); zmax, zmin = Z.max() + 0.02, Z.min() - 0.02
        self.pos, self.d, self.B, self.N = pos, d, B, n_sc
        self.t_plane = (zref - pos[2]) / d[:, 2]                     # what the scanner knows a priori: the panel plane
        tc, okc = march(np.broadcast_to(pos, (B, 3)), d, Z, zmax, zmin)
        self.t_true = np.where(okc, tc, np.nan)                      # ground truth: the boresight's true hit
        sig = spot / 2.355
        R = np.full((B, n_sc), np.nan, np.float64); A = np.zeros((B, n_sc)); X = np.zeros((B, n_sc)); Y = np.zeros((B, n_sc))
        rows = max(1, chunk // n_sc)
        for s in range(0, B, rows):
            sl = slice(s, min(B, s + rows)); b = sl.stop - sl.start
            a_, b_ = rng.normal(0, sig, (2, b, n_sc))
            o = pos + a_[..., None] * e1[sl, None, :] + b_[..., None] * e2[sl, None, :]
            dd = np.broadcast_to(d[sl, None, :], (b, n_sc, 3)).reshape(-1, 3)
            t, ok = march(o.reshape(-1, 3), dd, Z, zmax, zmin)
            p = o.reshape(-1, 3) + t[:, None] * dd
            nx, ny = -bilinear(surf.gx, p[:, 0], p[:, 1]), -bilinear(surf.gy, p[:, 0], p[:, 1])
            cosi = np.clip(-(dd[:, 0] * nx + dd[:, 1] * ny + dd[:, 2]) / np.sqrt(nx * nx + ny * ny + 1), 0, 1)
            amp = np.sqrt(np.clip(bilinear(surf.nir, p[:, 0], p[:, 1]), 0, 1) * cosi)
            # micro-roughness: a height per 4 um patch, scaled by the local roughness; it moves the range too
            hr = hash_normal(p[:, 0], p[:, 1], 0.004, 11) * bilinear(surf.rough, p[:, 0], p[:, 1]) * 1e-3
            # a collimated beam: each sub-ray's path is its own march length; a patch raised by h_r is met
            # h_r / |d_z| earlier along the ray
            Rr = t - hr / np.abs(dd[:, 2])
            R[sl] = np.where(ok, Rr, np.nan).reshape(b, n_sc); A[sl] = np.where(ok, amp, 0).reshape(b, n_sc)
            X[sl] = p[:, 0].reshape(b, n_sc); Y[sl] = p[:, 1].reshape(b, n_sc)
        self.R, self.A, self.X, self.Y = R, A, X, Y
        self.valid = np.isfinite(self.t_true) & (np.isfinite(R).sum(1) >= n_sc // 2)
        Rg = np.where(np.isfinite(R), R, np.nan)
        self.spread = np.nanstd(Rg, axis=1)                          # truth-side: depth spread inside the spot
        self.surf = surf

    def measure(self, bandwidth=100e9, lams=(1550e-9,), M=64, rho_ref=84.0, mode="coherent", combine="median",
                window_mm=1.4, seed=1, return_looks=False):
        """Range (mm) and intensity per beam. rho_ref: per-sample SNR of a speckle-averaged white (0.9),
        normal-incidence return; it sets the thermal precision to 8 um at 100 GHz. mode='incoherent' is the
        old model (amplitude^2-weighted mean range + the same thermal precision), for comparison."""
        rng = np.random.default_rng(seed)
        dR = C / (2 * bandwidth) * 1e3                               # range resolution, mm
        Pref = self.N * 0.9
        sig_n = np.sqrt(Pref / rho_ref)
        tm = (np.arange(M) + 0.5) / M - 0.5
        df = 1 / 32; fk = np.arange(-window_mm / dR, window_mm / dR + df, df)
        E = np.exp(-2j * np.pi * np.outer(tm, fk)).astype(np.complex64)
        Rs, Is = [], []
        Rm = np.nan_to_num(self.R - self.t_plane[:, None]); A = self.A
        for k, lam in enumerate(lams):
            if mode == "incoherent":
                w = A * A; Rbar = (w * Rm).sum(1) / np.maximum(w.sum(1), 1e-12)
                snr = rho_ref * w.sum(1) / Pref
                sd = np.sqrt(6 / ((2 * np.pi) ** 2 * np.maximum(snr, 1e-9) * M)) * dR
                Rs.append(self.t_plane + Rbar + rng.normal(0, 1, self.B) * sd); Is.append(w.sum(1) / self.N); continue
            ph0 = (4 * np.pi / (lam * 1e3)) * np.nan_to_num(self.R)  # optical phase (R in mm, lambda in mm)
            ph0 = np.mod(ph0, 2 * np.pi)
            Rh = np.empty(self.B); Ih = np.empty(self.B)
            for s in range(0, self.B, 6000):
                sl = slice(s, min(self.B, s + 6000)); b = sl.stop - sl.start
                u = (Rm[sl] / dR)                                    # tone frequency, cycles per chirp
                z = (A[sl, :, None] * np.exp(1j * (ph0[sl, :, None] + 2 * np.pi * u[:, :, None] * tm[None, None, :]))).sum(1)
                z = z + sig_n / np.sqrt(2) * (rng.standard_normal((b, M)) + 1j * rng.standard_normal((b, M)))
                S = np.abs(z.astype(np.complex64) @ E)
                k0 = np.clip(S.argmax(1), 1, len(fk) - 2); r = np.arange(b)
                a_, b_, c_ = S[r, k0 - 1], S[r, k0], S[r, k0 + 1]
                den = a_ - 2 * b_ + c_; dl = np.where(np.abs(den) > 1e-12, 0.5 * (a_ - c_) / den, 0.0)
                Rh[sl] = self.t_plane[sl] + (fk[k0] + np.clip(dl, -0.5, 0.5) * df) * dR
                Ih[sl] = (b_ / M) ** 2 / self.N
            Rs.append(Rh); Is.append(Ih)
        Rs, Is = np.array(Rs), np.array(Is)
        if combine == "median": R = np.median(Rs, 0)
        elif combine == "strongest": R = Rs[Is.argmax(0), np.arange(self.B)]
        else: R = (Rs * Is).sum(0) / np.maximum(Is.sum(0), 1e-12)
        out = dict(R=R, I=Is.mean(0), dR=dR)
        if return_looks: out.update(R_looks=Rs, I_looks=Is)
        return out

    def points(self, R):
        P = self.pos + R[:, None] * self.d
        return P

    def errors(self, R):
        """Truth-side scoring: along-beam error vs the boresight's true hit, and the reported point's height
        error vs the true surface under it (what a point cloud's user actually suffers)."""
        P = self.points(R)
        ez = P[:, 2] - bilinear(self.surf.Z, P[:, 0], P[:, 1])
        return dict(e_beam=R - self.t_true, e_z=ez, P=P)
