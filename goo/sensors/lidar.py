"""lidar.py: scan the painting.

The Sensor domain, kept apart from the World: the scanner sees only the surface (ground + strata + wet layer,
as a height field in mm) and the reflectance the paint actually has. It never reads the paint's species.

Model: a monostatic 1550 nm FMCW scanner (the band where reflectography sees through paint, and where
frequency-modulated ranging reaches micron precision at short range), raster-scanning from two stations
either side of the panel. Every beam is ray-marched against the height field (so impasto casts occlusion
shadows); its spot has a finite width, sampled by sub-rays whose ranges average (so edges give the mixed
'flying' pixels real scanners give); the return intensity is the near-infrared reflectance times
Lambert's cosine, plus a specular glint from glossy oil when the surface faces the beam, with shot noise;
the range has Gaussian noise. Output: a merged point cloud (x, y, z, intensity, rgb from a co-registered
camera) and a PLY file.
"""
import numpy as np
from scipy import ndimage

CELL = 0.25   # mm per cell, and per unit of paint thickness


def bilinear(F, x, y):
    """F sampled at cell centres (i+0.5)*CELL; x, y in mm."""
    H, W = F.shape
    u = np.clip(x / CELL - 0.5, 0, W - 1.001); v = np.clip(y / CELL - 0.5, 0, H - 1.001)
    i, j = v.astype(int), u.astype(int); fu, fv = u - j, v - i
    return (F[i, j] * (1 - fu) * (1 - fv) + F[i, j + 1] * fu * (1 - fv) + F[i + 1, j] * (1 - fu) * fv + F[i + 1, j + 1] * fu * fv)


class Scanner:
    def __init__(self, pos, spacing=0.15, spot=0.12, sub=4, sigma_range=0.008, glint=0.6, seed=0):
        self.pos = np.asarray(pos, float); self.spacing, self.spot, self.sub = spacing, spot, sub
        self.sigma, self.glint = sigma_range, glint
        self.rng = np.random.default_rng(seed)

    def scan(self, Z, nir, gloss, rgb):
        """Z: (H, W) surface height in mm. nir: (H, W) reflectance at 1550 nm. gloss: (H, W) 0..1. rgb: (H, W, 3)."""
        H, W = Z.shape; Wm, Hm = W * CELL, H * CELL
        zmax, zmin = Z.max() + 0.05, Z.min() - 0.05
        # aim points on a grid over the panel at z = mean; directions from the station through them
        gx, gy = np.meshgrid(np.arange(0.5, Wm - 0.5, self.spacing), np.arange(0.5, Hm - 0.5, self.spacing))
        aim = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, Z.mean())], 1)
        d0 = aim - self.pos; d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
        # the spot: sub-rays offset perpendicular to the beam
        e1 = np.cross(d0, [0, 0, 1.0]); e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(d0, e1)
        ts, hits = [], []
        for k in range(self.sub):
            a, b = self.rng.normal(0, self.spot / 2.355, (2, len(d0)))
            o = self.pos + a[:, None] * e1 + b[:, None] * e2
            t, ok = self._march(o, d0, Z, zmax, zmin)
            ts.append(np.where(ok, t, np.nan)); hits.append(ok)
        T = np.array(ts); ok = np.array(hits).sum(0) >= max(1, self.sub // 2)
        t = np.nanmean(np.where(np.isfinite(T), T, np.nan), axis=0)       # FMCW reports the spot's mean range
        t = t + self.rng.normal(0, self.sigma, t.shape)
        P = self.pos + t[:, None] * d0
        P, d, t = P[ok], d0[ok], t[ok]
        # intensity: near-IR reflectance x Lambert + glint off glossy oil, with shot noise
        gyz, gxz = np.gradient(Z, CELL)
        nx, ny = -bilinear(gxz, P[:, 0], P[:, 1]), -bilinear(gyz, P[:, 0], P[:, 1])
        n = np.stack([nx, ny, np.ones_like(nx)], 1); n /= np.linalg.norm(n, axis=1, keepdims=True)
        cosi = np.clip(-(n * d).sum(1), 0, 1)
        rho = bilinear(nir, P[:, 0], P[:, 1]); gl = bilinear(gloss, P[:, 0], P[:, 1])
        spec = self.glint * gl * np.exp(-(np.arccos(np.clip(cosi, -1, 1))) ** 2 / (2 * 0.05 ** 2))
        I = rho * cosi + spec
        I = self.rng.poisson(np.clip(I, 0, None) * 4000) / 4000.0
        col = np.stack([bilinear(rgb[..., c], P[:, 0], P[:, 1]) for c in range(3)], 1)
        return dict(xyz=P, intensity=I, rgb=col, cos=cosi, n_rays=len(d0), n_hits=int(ok.sum()))

    def _march(self, o, d, Z, zmax, zmin, step=0.04):
        """First intersection of each ray with the height field: march from where it enters the slab, bisect."""
        t0 = (zmax - o[:, 2]) / d[:, 2]; t1 = (zmin - o[:, 2]) / d[:, 2]
        horiz = np.hypot(d[:, 0], d[:, 1]) / np.maximum(np.abs(d[:, 2]), 1e-9)
        n = int(np.ceil(((zmax - zmin) * horiz.max()) / step)) + 2
        lo = t0.copy(); found = np.zeros(len(o), bool); hi = t1.copy()
        prev = t0.copy()
        for s in range(1, n + 1):
            t = t0 + (t1 - t0) * s / n
            p = o + t[:, None] * d
            below = (p[:, 2] <= bilinear(Z, p[:, 0], p[:, 1])) & ~found
            lo = np.where(below, prev, lo); hi = np.where(below, t, hi); found |= below
            prev = t
        for _ in range(10):
            m = 0.5 * (lo + hi); p = o + m[:, None] * d
            b = p[:, 2] <= bilinear(Z, p[:, 0], p[:, 1])
            hi = np.where(b, m, hi); lo = np.where(b, lo, m)
        p = o + hi[:, None] * d
        inside = (p[:, 0] > 0) & (p[:, 0] < Z.shape[1] * CELL) & (p[:, 1] > 0) & (p[:, 1] < Z.shape[0] * CELL)
        return hi, found & inside


def grid(points, values, shape, agg="mean"):
    """Bin a cloud back onto the canvas grid (mean per cell; NaN where nothing landed)."""
    H, W = shape
    j = np.clip((points[:, 0] / CELL).astype(int), 0, W - 1); i = np.clip((points[:, 1] / CELL).astype(int), 0, H - 1)
    k = i * W + j
    s = np.bincount(k, values, H * W); c = np.bincount(k, None, H * W)
    out = np.where(c > 0, s / np.maximum(c, 1), np.nan).reshape(H, W)
    return out, c.reshape(H, W)


def write_ply(path, xyz, intensity, rgb):
    n = len(xyz)
    hdr = ("ply\nformat binary_little_endian 1.0\ncomment simulated 1550 nm FMCW scan, units mm\n"
           f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\nproperty float intensity\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"], rec["i"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], intensity
    c = (np.clip(rgb, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8); rec["r"], rec["g"], rec["b"] = c[:, 0], c[:, 1], c[:, 2]
    with open(path, "wb") as f: f.write(hdr.encode()); f.write(rec.tobytes())
