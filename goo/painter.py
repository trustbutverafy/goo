"""goo.painter: a painter that looks.

The painter has a motif (a reference: colour, which object is where, where the highlights are), a palette,
brushes, and eyes: it sees the canvas only as colour (Canvas.reflectance), never as volumes. Each pass is
coarse-to-fine curved strokes after Hertzmann (1998), 'Painterly rendering with curved brush strokes of
multiple sizes': where the canvas is furthest from the motif (seen at the brush's own scale), start a stroke;
follow the forms (across the local colour gradient, from the motif's structure tensor), stop at an object's
edge or where going on would make things worse; mix the stroke's colour by inverting the optics over what
the canvas shows there; then let a real brush carry it.
"""
import numpy as np
from scipy import ndimage
from goo.brush import Brush
from goo.palette import Palette, lab


class Painter:
    def __init__(self, cv, motif, ids, palette, seed=0, plan=None):
        """plan: {object id: angle (radians) | 'around'}: how the painter means to lay strokes where the motif
        itself has no structure to follow (a flat wall, a table top, the turning of a round form)."""
        self.cv, self.motif, self.ids = cv, np.asarray(motif, float), ids
        H, W = ids.shape; yy, xx = np.mgrid[0:H, 0:W]
        pdx, pdy = np.ones((H, W)), np.zeros((H, W))
        for oid, how in (plan or {}).items():
            m = ids == oid
            if not m.any(): continue
            if how == "around":
                cy, cx = np.argwhere(m).mean(0); ry, rx = yy - cy, xx - cx; r = np.hypot(rx, ry) + 1e-9
                pdx[m], pdy[m] = (-ry / r)[m], (rx / r)[m]
            else:
                pdx[m], pdy[m] = np.cos(how), np.sin(how)
        self.plan_dx, self.plan_dy = pdx, pdy
        self._swatches = {}
        self.palette = palette if isinstance(palette, Palette) else Palette(palette, steps=12)
        self.rng = np.random.default_rng(seed)
        self.log = []

    def _simplify(self, sigma):
        """The motif at the brush's scale, the way a painter simplifies it: each object blurred within itself
        (normalised convolution over its own mask), so masses simplify without bleeding into their neighbours."""
        if sigma <= 0: return self.motif.copy()
        out = np.zeros_like(self.motif)
        for oid in np.unique(self.ids):
            m = (self.ids == oid).astype(float)
            den = ndimage.gaussian_filter(m, sigma) + 1e-9
            for c in range(3):
                out[..., c] += m * ndimage.gaussian_filter(self.motif[..., c] * m, sigma) / den
        return out

    def swatch(self, width, target, drag, plough, phi, exchange, speed):
        """Try the brush on a scrap of panel first: what thickness does a stroke actually lay?"""
        key = (round(width, 2), target, drag, plough, phi, exchange, speed)
        if key in self._swatches: return self._swatches[key]
        from goo.engine import Canvas
        L = max(60, int(10 * width)); scrap = Canvas(L + 20, int(4 * width) + 20, ["white"], laws=dict(lazy=90, flow_sub=0.1, diff_sub=0.1))
        b = Brush(scrap, width, bristles=max(1, int(width / 2.5)), exchange=exchange, drag=drag, plough=plough)
        b.charge(Brush.paint("oil", phi, 2 * L * width * target, {"white": 1}))
        y = scrap.H / 2; pts = [(10, y), (10 + L / 3, y), (10 + 2 * L / 3, y), (10 + L, y)]
        b.stroke([(x, yv, 0.65 + 0.35 * np.sin(np.pi * (j + .5) / 4)) for j, (x, yv) in enumerate(pts)], speed=max(1.5, speed * width), target=target)
        scrap.flush()
        band = scrap.h[int(y - width / 2):int(y + width / 2), 10 + int(L * 0.1):10 + int(L * 0.9)]
        band = band[band > 0.02 * band.max()]
        self._swatches[key] = np.quantile(band, np.linspace(0.05, 0.95, 7)); return self._swatches[key]

    def see(self):
        return self.cv.reflectance()[..., :3]

    def _orientation(self, img, R):
        """Stroke directions: the minor eigenvector of the colour structure tensor, smoothed at scale R."""
        L = lab(img)
        J = np.zeros((3,) + img.shape[:2])
        for c in range(3):
            gy, gx = np.gradient(ndimage.gaussian_filter(L[..., c], 0.5 * R))
            J[0] += gx * gx; J[1] += gx * gy; J[2] += gy * gy
        J = np.stack([ndimage.gaussian_filter(j, R) for j in J])
        ang = 0.5 * np.arctan2(2 * J[1], J[0] - J[2])               # direction of greatest change
        lam1 = 0.5 * (J[0] + J[2] + np.sqrt((J[0] - J[2]) ** 2 + 4 * J[1] ** 2))
        coh = np.sqrt((J[0] - J[2]) ** 2 + 4 * J[1] ** 2) / (J[0] + J[2] + 1e-9)
        strength = np.sqrt(lam1)                                     # dE per cell across the structure
        return np.cos(ang + np.pi / 2), np.sin(ang + np.pi / 2), coh * np.clip(strength / 1.0, 0, 1)

    def pass_(self, R, paint, target, threshold=6.0, curvature=0.6, min_len=2, max_len=14, blur=1.0,
              region=None, colour_tol=14.0, label="", width_scale=1.8, pressure=1.0, load_factor=1.6,
              max_strokes=None, bristles=None, transparent=False, phi=0.5, speed=0.5, drag=1.0, plough=0.5, exchange=0.85, opacity_bias=6.0, cell_stat="mean"):
        """One layer of strokes with brush radius R (cells). `paint(mix) -> load` builds the brush load from a
        mix; `target` is the thickness the brush lays. Returns the number of strokes."""
        cv, H, W = self.cv, self.cv.H, self.cv.W
        ref = self._simplify(blur * R)
        seen = self.see()
        Lref, Lseen = lab(ref), lab(seen)
        diff = np.linalg.norm(Lref - Lseen, axis=-1)
        if region is not None: diff = diff * region
        dx_, dy_, coh = self._orientation(ref, R)
        # follow the motif where it has structure, the plan where it hasn't
        wgt = np.clip(coh / 0.35, 0, 1)
        sgn = np.sign(dx_ * self.plan_dx + dy_ * self.plan_dy + 1e-12)
        dx_ = wgt * dx_ * sgn + (1 - wgt) * self.plan_dx; dy_ = wgt * dy_ * sgn + (1 - wgt) * self.plan_dy
        g = max(1, int(R))
        starts = []
        for y0 in range(0, H, g):
            for x0 in range(0, W, g):
                cell = diff[y0:y0 + g, x0:x0 + g]
                v = cell.max() if cell_stat == "max" else cell.mean()           # 'max': a painter's eye for a stray mark
                if v > threshold:
                    k = np.argmax(cell); starts.append((v, y0 + k // cell.shape[1], x0 + k % cell.shape[1]))
        if threshold <= 0:                                                  # a first pass covers the panel to its edges
            for v in range(g // 2, H, g): starts += [(99.0, v, 1), (99.0, v, W - 2)]
            for u in range(g // 2, W, g): starts += [(99.0, 1, u), (99.0, H - 2, u)]
        if max_strokes and len(starts) > max_strokes:                     # attend to the worst first
            starts.sort(key=lambda s_: -s_[0]); starts = starts[:max_strokes]
        starts = [(y, x) for _, y, x in starts]; self.rng.shuffle(starts)
        width = max(2.0, width_scale * R)
        brush = Brush(cv, width, bristles=bristles or max(1, int(width / 2.5)), exchange=exchange, drag=drag, plough=plough)
        laid = self.swatch(width, target, drag, plough, phi, exchange, speed)    # what this brush really lays
        solids = laid * phi                                                  # seven thicknesses, not one
        painted = np.zeros((H, W), bool)                                     # where this pass has already been
        n = 0
        for (y, x) in starts:
            if painted[y, x]: continue                                       # this pass has painted here already
            col = ref[y, x]; oid = self.ids[y, x]
            pts = [(x + 0.5, y + 0.5)]; px, py = x + 0.5, y + 0.5; ldx = ldy = 0.0
            for i in range(max_len):
                iy, ix = int(py), int(px)
                if i > min_len and np.linalg.norm(Lref[iy, ix] - Lseen[iy, ix]) < np.linalg.norm(Lref[iy, ix] - lab(col)): break
                dx, dy = dx_[iy, ix], dy_[iy, ix]
                if dx * ldx + dy * ldy < 0: dx, dy = -dx, -dy
                if i > 0: dx, dy = curvature * dx + (1 - curvature) * ldx, curvature * dy + (1 - curvature) * ldy
                nrm = np.hypot(dx, dy) + 1e-12; dx, dy = dx / nrm, dy / nrm
                nx, ny = px + R * dx, py + R * dy
                if not (1 <= nx < W - 1 and 1 <= ny < H - 1): break
                if self.ids[int(ny), int(nx)] != oid: break                      # the painter respects the edge
                if np.linalg.norm(Lref[int(ny), int(nx)] - lab(col)) > colour_tol: break
                px, py, ldx, ldy = nx, ny, dx, dy; pts.append((px, py))
            if len(pts) < 2:                                                     # a dab
                a = np.arctan2(dy_[y, x], dx_[y, x]) + self.rng.normal(0, 0.2); pts = [(x + 0.5 - 0.5 * R * np.cos(a), y + 0.5 - 0.5 * R * np.sin(a)), (x + 0.5 + 0.5 * R * np.cos(a), y + 0.5 + 0.5 * R * np.sin(a))]
            ix_ = np.clip([int(q[1]) for q in pts], 0, H - 1); jx_ = np.clip([int(q[0]) for q in pts], 0, W - 1)
            under = seen[ix_, jx_].mean(0)                                   # what the whole stroke will cover
            mix, pred, dE = self.palette.match(col, under, solids, oil=laid * (1 - phi), opacity_bias=opacity_bias)
            length = sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])) + R
            brush.charge(paint(mix, load_factor * length * width * target))
            m = len(pts)
            prof = [(p[0], p[1], pressure * (0.65 + 0.35 * np.sin(np.pi * (j + 0.5) / m))) for j, p in enumerate(pts)]
            brush.stroke(prof, speed=max(1.5, speed * width), target=target)
            for (qx, qy, _) in prof:
                r0 = int(width * 0.35); yy0, xx0 = int(qy), int(qx)
                painted[max(0, yy0 - r0):yy0 + r0 + 1, max(0, xx0 - r0):xx0 + r0 + 1] = True
            n += 1
        self.log.append(dict(label=label, R=R, strokes=n, t=cv.t, swatch=float(np.median(laid)), mean_dE_before=float(diff[diff > 0].mean()) if (diff > 0).any() else 0.0))
        return n

    def error(self, blur=0.0):
        """Mean CIELAB distance between what the painter sees and the motif (optionally both blurred)."""
        ref = self.motif if blur == 0 else np.stack([ndimage.gaussian_filter(self.motif[..., c], blur) for c in range(3)], -1)
        seen = self.see()
        if blur: seen = np.stack([ndimage.gaussian_filter(seen[..., c], blur) for c in range(3)], -1)
        return float(np.linalg.norm(lab(ref) - lab(seen), axis=-1).mean())
