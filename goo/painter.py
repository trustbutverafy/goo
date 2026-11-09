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
    def __init__(self, cv, motif, ids, palette, seed=0):
        self.cv, self.motif, self.ids = cv, np.asarray(motif, float), ids
        self.palette = palette if isinstance(palette, Palette) else Palette(palette, steps=12)
        self.rng = np.random.default_rng(seed)
        self.log = []

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
        coh = np.sqrt((J[0] - J[2]) ** 2 + 4 * J[1] ** 2) / (J[0] + J[2] + 1e-9)
        return np.cos(ang + np.pi / 2), np.sin(ang + np.pi / 2), coh  # strokes run along the isophotes

    def pass_(self, R, paint, target, threshold=6.0, curvature=0.6, min_len=2, max_len=14, blur=1.0,
              region=None, colour_tol=14.0, label="", width_scale=1.8, pressure=1.0, load_factor=1.6,
              max_strokes=None, bristles=None, transparent=False, phi=0.5):
        """One layer of strokes with brush radius R (cells). `paint(mix) -> load` builds the brush load from a
        mix; `target` is the thickness the brush lays. Returns the number of strokes."""
        cv, H, W = self.cv, self.cv.H, self.cv.W
        ref = np.stack([ndimage.gaussian_filter(self.motif[..., c], blur * R) for c in range(3)], -1)
        seen = self.see()
        Lref, Lseen = lab(ref), lab(seen)
        diff = np.linalg.norm(Lref - Lseen, axis=-1)
        if region is not None: diff = diff * region
        dx_, dy_, coh = self._orientation(ref, R)
        g = max(1, int(R))
        starts = []
        for y0 in range(0, H, g):
            for x0 in range(0, W, g):
                cell = diff[y0:y0 + g, x0:x0 + g]
                if cell.mean() > threshold:
                    k = np.argmax(cell); starts.append((y0 + k // cell.shape[1], x0 + k % cell.shape[1]))
        self.rng.shuffle(starts)
        if max_strokes: starts = starts[:max_strokes]
        width = max(2.0, width_scale * R)
        brush = Brush(cv, width, bristles=bristles or max(1, int(width / 2.5)), exchange=0.35)
        solids = target * phi * 0.8
        n = 0
        for (y, x) in starts:
            col = ref[y, x]; oid = self.ids[y, x]
            pts = [(x + 0.5, y + 0.5)]; px, py = x + 0.5, y + 0.5; ldx = ldy = 0.0
            for i in range(max_len):
                iy, ix = int(py), int(px)
                if i > min_len and np.linalg.norm(Lref[iy, ix] - Lseen[iy, ix]) < np.linalg.norm(Lref[iy, ix] - lab(col)): break
                dx, dy = dx_[iy, ix], dy_[iy, ix]
                if coh[iy, ix] < 0.02 and i > 0: dx, dy = ldx, ldy           # no structure here: keep going straight
                if dx * ldx + dy * ldy < 0: dx, dy = -dx, -dy
                if i > 0: dx, dy = curvature * dx + (1 - curvature) * ldx, curvature * dy + (1 - curvature) * ldy
                nrm = np.hypot(dx, dy) + 1e-12; dx, dy = dx / nrm, dy / nrm
                nx, ny = px + R * dx, py + R * dy
                if not (1 <= nx < W - 1 and 1 <= ny < H - 1): break
                if self.ids[int(ny), int(nx)] != oid: break                      # the painter respects the edge
                if np.linalg.norm(Lref[int(ny), int(nx)] - lab(col)) > colour_tol: break
                px, py, ldx, ldy = nx, ny, dx, dy; pts.append((px, py))
            if len(pts) < 2:                                                     # a dab
                a = self.rng.uniform(0, np.pi); pts = [(x + 0.5 - 0.5 * R * np.cos(a), y + 0.5 - 0.5 * R * np.sin(a)), (x + 0.5 + 0.5 * R * np.cos(a), y + 0.5 + 0.5 * R * np.sin(a))]
            under = seen[y, x]
            mix, pred, dE = self.palette.match(col, under, solids, oil=target * (1 - phi) * 0.8)
            length = sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])) + R
            brush.charge(paint(mix, load_factor * length * width * target))
            m = len(pts)
            prof = [(p[0], p[1], pressure * (0.65 + 0.35 * np.sin(np.pi * (j + 0.5) / m))) for j, p in enumerate(pts)]
            brush.stroke(prof, speed=max(1.5, 0.3 * width), target=target)
            n += 1
        self.log.append(dict(label=label, R=R, strokes=n, t=cv.t, mean_dE_before=float(diff[diff > 0].mean()) if (diff > 0).any() else 0.0))
        return n

    def error(self, blur=0.0):
        """Mean CIELAB distance between what the painter sees and the motif (optionally both blurred)."""
        ref = self.motif if blur == 0 else np.stack([ndimage.gaussian_filter(self.motif[..., c], blur) for c in range(3)], -1)
        seen = self.see()
        if blur: seen = np.stack([ndimage.gaussian_filter(seen[..., c], blur) for c in range(3)], -1)
        return float(np.linalg.norm(lab(ref) - lab(seen), axis=-1).mean())
