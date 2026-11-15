"""relight.py: a visible-light camera. Photographs a Surface under one distant light and a uniform sky.

It is constructed from a `goo.surface.Surface` and reads nothing else from the World.

Conventions
-----------
* Image axes: x to the right (column index), y DOWN the image (row index), z out of the panel toward the viewer.
  A height field z = Z(x, y) has the normal n ~ (-dZ/dx, -dZ/dy, 1).
* The camera looks straight down, orthographic: view direction v = (0, 0, 1); one output pixel per cell.
* L is the unit vector FROM the surface TOWARD a distant light, lz > 0 (it is normalised on entry). A light on
  the left has lx < 0, a light at the top of the image ly < 0; bumps cast shadows away from the light.
  `light_dir(azimuth, elevation)`: azimuth 0 = light at the left, 90 = at the top.
* Output: linear radiance (H, W, 3), RGB, float64, in units where a white Lambertian facing a light of
  intensity I returns I. No exposure compensation with angle: a raking light gives a darker picture.

Shading per cell (rho = surface.rgb, g = surface.gloss)
-------------------------------------------------------
    radiance = intensity * V * (n.L)+ * [ rho (1 - g F) + g F Lam ]  +  ambient * AO * [ rho (1 - g Fd) + g Fs ]

* n: normal of relief_gain * Z_mm, central differences (np.gradient, one-sided at the border).
* F = Schlick Fresnel at h.v, h = (L + v)/|L + v|, F0 = 0.04 (an oil film, n ~ 1.5). The skin reflects F of the
  light specularly; the body (diffuse) gets the rest, (1 - g F). It is the same F in both terms, so for a white
  body no energy is created by the coupling.
* Lam = (s + 2)/2 * max(0, r.L)^s, r = 2 (n.v) n - v the mirror of the view direction: the normalised (modified)
  Phong lobe, shininess s (default 100). For n = v, (1/pi) * integral of Lam (n.L) over the hemisphere is 1.
* Energy: the directional albedo of a cell seen from v (by reciprocity, the integral of its radiance over light
  directions, / pi) is  A = rho (1 - g Fbar) + g <F Lam>,  Fbar = <F>, <x> = (1/pi) * integral of x (n.L) dw.
  For n = v, F increases and Lam decreases with the angle of L from the normal, so by Chebyshev's integral
  inequality <F Lam> <= <F><Lam> = Fbar, and A <= rho + g Fbar (1 - rho) <= 1 for rho <= 1: proved, with
  equality only for a white matte body. For tilted facets the lobe is clipped by the facet's horizon; measured
  in tests/test_relight.py, A <= 1 + ENERGY_TOL over 0..60 degrees of tilt (ENERGY_TOL = 1e-3: quadrature).
  The PEAK radiance of a glossy cell can exceed a white Lambertian's (at the mirror direction it is
  rho (1 - g F0) + g F0 (s + 2)/2, 3.0 for a white, fully glossy cell at s = 100): a glint is concentrated, not
  extra, energy.
* V: cast-shadow visibility. Horizon along the light's azimuth by marching 1-cell steps (bilinear, edge-clamped)
  out to the distance beyond which nothing can rise above the light (zrange / tan(elevation - SUN_RADIUS));
  V = smoothstep of (elevation - horizon) over +-SUN_RADIUS (a light disc 3 degrees across). V = 1 for a
  zenith light. V multiplies both direct terms.
* Ambient: a uniform white sky of radiance `ambient`, reflected by the cell's directional albedo at n = v
  (Fd = Fbar, Fs = <F Lam> at s; constants computed once) times AO = mean over AO_DIRS azimuths of
  cos^2(max(horizon, 0)) out to AO_RADIUS cells (the cosine-weighted open sky measured from the panel plane;
  1 exactly on a flat panel). Ambient has no shadowing beyond AO and no glint.
* relief_gain scales the height before normals, shadows and AO; colour and gloss are unchanged by it.

Prepare once, render many: `prepare(surface, relief_gain)` computes normals and AO (the costly part, independent
of the light); `Prepared.render(L, ...)` is one horizon march plus arithmetic. `render(surface, L, ...)` does both.
`texels(image, n)`: the aperture of a texel, the mean of n x n cells in linear radiance.
"""
from __future__ import annotations
import numpy as np

SUN_RADIUS = np.radians(1.5)   # angular radius of the light disc
AO_DIRS = 16                   # azimuths for ambient occlusion
AO_RADIUS = 48                 # AO horizon search radius, cells (12 mm)
SHININESS = 100.0              # Phong exponent s of an oil skin (lobe half-width at half maximum ~ 6.7 deg)
F0 = 0.04                      # specular reflectance at normal incidence (oil, n ~ 1.5)
ENERGY_TOL = 1e-3              # stated tolerance on the directional albedo of a white cell (see docstring)


def schlick(cos_hv, f0=F0):
    return f0 + (1 - f0) * (1 - np.clip(cos_hv, 0, 1)) ** 5


def lobe(rL, s):
    """Normalised Phong lobe (s + 2)/2 * max(0, r.L)^s."""
    return (s + 2) / 2 * np.maximum(rL, 0.0) ** s


def ambient_constants(s=SHININESS, f0=F0, n=20001):
    """(Fbar, <F Lam>) for n = v: cosine-weighted hemisphere means, by quadrature in the polar angle."""
    th = np.linspace(0, np.pi / 2, n); w = 2 * np.cos(th) * np.sin(th)      # (1/pi) dw cos -> 2 cos sin dth
    F = schlick(np.cos(th / 2), f0); Lm = lobe(np.cos(th), s)
    return float(np.trapezoid(F * w, th)), float(np.trapezoid(F * Lm * w, th))


def reflected(n, L, rho, gloss, s=SHININESS, f0=F0):
    """Direct reflected radiance per unit light intensity, unshadowed: (n.L)+ [rho (1 - g F) + g F Lam].
    n (..., 3) unit normals, L (3,) or (..., 3) unit light directions, rho (..., C), gloss (...). Viewer v = (0, 0, 1)."""
    L = np.asarray(L, np.float64); v = np.array([0.0, 0.0, 1.0])
    nl = np.maximum((n * L).sum(-1), 0.0)
    h = L + v; h = h / np.linalg.norm(h, axis=-1, keepdims=True)
    F = schlick(h[..., 2], f0)                                             # h.v
    r = 2 * n[..., 2:3] * n - v                                            # mirror of v about n
    Lm = lobe((r * L).sum(-1), s)
    gF = gloss * F
    return nl[..., None] * (rho * (1 - gF)[..., None] + (gF * Lm)[..., None])


def light_dir(azimuth_deg, elevation_deg):
    """azimuth 0 = light at the LEFT (lx < 0), 90 = at the TOP of the image (ly < 0); elevation above the panel."""
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array([-np.cos(a) * np.cos(e), -np.sin(a) * np.cos(e), np.sin(e)])


def _shift_bilinear(Zp, pad, H, W, ox, oy):
    fx, fy = np.floor(ox), np.floor(oy)
    ax, ay = ox - fx, oy - fy
    x0, y0 = pad + int(fx), pad + int(fy)
    a = Zp[y0:y0 + H, x0:x0 + W]; b = Zp[y0:y0 + H, x0 + 1:x0 + 1 + W]
    c = Zp[y0 + 1:y0 + 1 + H, x0:x0 + W]; d = Zp[y0 + 1:y0 + 1 + H, x0 + 1:x0 + 1 + W]
    return (1 - ay) * ((1 - ax) * a + ax * b) + ay * ((1 - ax) * c + ax * d)


def horizon_tan(Zg, cell, dx, dy, t_max):
    """max over t = 1, 2, .. <= ceil(t_max) cells of (Zg(p + t (dx, dy)) - Zg(p)) / (t cell); -inf if no step."""
    H, W = Zg.shape
    n = int(np.ceil(t_max))
    best = np.full((H, W), -np.inf)
    if n <= 0: return best
    pad = n + 2
    Zp = np.pad(Zg, pad, mode="edge")
    for t in range(1, n + 1):
        z = _shift_bilinear(Zp, pad, H, W, t * dx, t * dy)
        np.maximum(best, (z - Zg) / (t * cell), out=best)
    return best


def _smoothstep(e0, e1, x):
    u = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return u * u * (3 - 2 * u)


class Prepared:
    """A Surface with its light-independent geometry done: scaled height, normals, AO."""

    def __init__(self, surface, relief_gain=1.0, shininess=SHININESS, f0=F0):
        rgb = np.asarray(surface.rgb, np.float64)
        if not np.isfinite(rgb).all():
            raise ValueError("this Surface has no visible reflectance (built from nir only?)")
        self.rgb, self.gloss = rgb, np.asarray(surface.gloss, np.float64)
        self.cell = float(surface.cell_mm); self.relief_gain = float(relief_gain)
        self.s, self.f0 = float(shininess), float(f0)
        self.Zg = np.asarray(surface.Z_mm, np.float64) * self.relief_gain
        gy, gx = np.gradient(self.Zg, self.cell)                          # axis 0 = y (down), axis 1 = x (right)
        n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        self.n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        self.zrange = float(self.Zg.max() - self.Zg.min())
        ao = np.zeros(self.Zg.shape)
        if self.zrange > 0:
            for k in range(AO_DIRS):
                ph = 2 * np.pi * k / AO_DIRS
                hp = np.maximum(horizon_tan(self.Zg, self.cell, np.cos(ph), np.sin(ph), AO_RADIUS), 0.0)
                ao += 1.0 / (1.0 + hp * hp)                              # cos^2(atan t)
            ao /= AO_DIRS
        else:
            ao[:] = 1.0
        self.ao = ao
        Fd, Fs = ambient_constants(self.s, self.f0)
        self.sky = rgb * (1 - self.gloss * Fd)[..., None] + (self.gloss * Fs)[..., None]   # albedo at n = v

    def visibility(self, L):
        """Cast-shadow visibility toward L for a light disc of angular radius SUN_RADIUS."""
        L = _unit(L)
        rxy = np.hypot(L[0], L[1])
        if rxy < 1e-12 or self.zrange <= 0: return np.ones(self.Zg.shape)
        e = np.arctan2(L[2], rxy); lo = e - SUN_RADIUS
        diag = float(np.hypot(*self.Zg.shape))
        t_max = min(self.zrange / (self.cell * np.tan(lo)), diag) if lo > 1e-6 else diag
        hz = np.arctan(horizon_tan(self.Zg, self.cell, L[0] / rxy, L[1] / rxy, t_max))
        return _smoothstep(-SUN_RADIUS, SUN_RADIUS, e - hz)

    def render(self, L, intensity=1.0, ambient=0.10, return_parts=False):
        """Linear radiance (H, W, 3) under a distant light toward L of the given intensity, plus the sky."""
        L = _unit(L)
        direct = reflected(self.n, L, self.rgb, self.gloss, self.s, self.f0)
        V = self.visibility(L)
        direct = direct * V[..., None]
        sky = self.sky * self.ao[..., None]
        out = intensity * direct + ambient * sky
        if return_parts: return out, dict(direct=direct, sky=sky, V=V)
        return out


def _unit(L):
    L = np.asarray(L, np.float64)
    L = L / np.linalg.norm(L)
    if not L[2] > 0: raise ValueError("the light must be above the panel (lz > 0)")
    return L


def prepare(surface, relief_gain=1.0, shininess=SHININESS, f0=F0):
    """Do the light-independent work once; then call .render(L, ...) for each direction."""
    return Prepared(surface, relief_gain, shininess, f0)


def render(surface, L, intensity=1.0, ambient=0.10, relief_gain=1.0, shininess=SHININESS, f0=F0, return_parts=False):
    """Linear radiance (H, W, 3) of `surface` under a distant light toward L (from the surface), viewer above."""
    return prepare(surface, relief_gain, shininess, f0).render(L, intensity, ambient, return_parts)


def texels(image, n):
    """The aperture of a texel: the mean of each n x n block of cells, in linear radiance. H and W must be
    multiples of n (crop first)."""
    image = np.asarray(image, np.float64); H, W = image.shape[:2]
    if H % n or W % n: raise ValueError(f"image {H} x {W} is not a whole number of {n} x {n} texels; crop it first")
    return image.reshape(H // n, n, W // n, n, *image.shape[2:]).mean(axis=(1, 3))
