"""brush.py: tools that touch the paint.

A Brush is a body: a row of bristles and a load (volumes of each species). Where it touches, it does two
things. It imposes its velocity on the paint (bristles drag harder than the gaps), and it exchanges paint
with the canvas toward a target thickness under each bristle: where the canvas is thinner it gives from its
load (in the load's proportions), where thicker it picks up (in the canvas's proportions). So a brush gets
dirty, mixes on the canvas, runs dry and stops giving, all by the same rule.

A Stylus is a brush with no load and no exchange: it only drags (sgraffito, combing, marbling rods).
Charcoal is a brush whose load is a dry pigment with no carrier: it deposits only on the ground's tooth,
and the deposit (phi = 1) is jammed solid where it lands.
"""
import numpy as np


class Brush:
    def __init__(self, cv, width, bristles=None, load=None, exchange=0.6, tooth_only=False, drag=1.0, plough=0.5):
        self.cv, self.width, self.exchange, self.tooth_only, self.drag, self.plough = cv, width, exchange, tooth_only, drag, plough
        self.bristles = bristles if bristles is not None else max(1, int(width / 2.5))
        self.load = np.zeros(len(cv.species))
        # a real brush: bristles clump unevenly; its profile across the stroke is its own, fixed for its life
        rng = np.random.default_rng(1000 + len(cv.brushes))
        s = np.linspace(-1, 1, 129); nb = self.bristles
        c = np.sort(rng.uniform(-1, 1, nb)) if nb > 1 else np.zeros(1)
        prof = 0.45 + sum(a * np.exp(-((s - ci) / wi) ** 2) for ci, a, wi in zip(c, rng.uniform(0.4, 1.0, nb), rng.uniform(0.4, 1.1, nb) / max(nb, 1)))
        self.profile = prof / prof.max() if nb > 1 else np.ones_like(s)
        cv.brushes.append(self)
        if load: self.charge(load)

    def charge(self, load):
        """Load paint: {species: volume}. What was left goes in the rinse jar (counted as rinsed, not lost)."""
        cv = self.cv
        self.cv.led.setdefault("rinsed", np.zeros(len(cv.species)))
        cv.led["rinsed"] += self.load
        cv.led["loaded"] -= self.load            # the jar returns it to the books as not-on-canvas
        self.load = np.zeros(len(cv.species))
        for k, v in load.items(): self.load[cv.idx(k)] += v
        cv.led["loaded"] += self.load
        return self

    @staticmethod
    def paint(kind, phi, volume, mix):
        """A paint of a given carrier ('water' or 'oil'), solid fraction and pigment mix {pigment: share}."""
        tot = sum(mix.values())
        d = {kind: volume * (1 - phi)}
        for k, v in mix.items(): d[k] = volume * phi * v / tot
        return d

    def touch(self, x, y, dx, dy, pressure, target):
        cv = self.cv
        sp = np.hypot(dx, dy); tx, ty = (dx / sp, dy / sp) if sp > 1e-9 else (1.0, 0.0)
        r = self.width / 2 * (0.6 + 0.4 * pressure)
        # the footprint is swept: the capsule from where the brush was (x - dx, y - dy) to where it is
        xs_, ys_ = (x - dx, x), (y - dy, y)
        x0, x1 = int(max(0, min(xs_) - r - 2)), int(min(cv.W, max(xs_) + r + 3)); y0, y1 = int(max(0, min(ys_) - r - 2)), int(min(cv.H, max(ys_) + r + 3))
        if x0 >= x1 or y0 >= y1: return
        yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
        along = (xx - x) * tx + (yy - y) * ty; across = -(xx - x) * ty + (yy - y) * tx
        along = along - np.clip(along, -sp, 0.0)                    # inside the swept segment: no along-distance
        foot = np.clip(1.6 - np.hypot(along / (0.55 * r + 0.8), across / r) * 1.6, 0, 1)   # soft-edged oval
        if not foot.any(): return
        nb = self.bristles
        bris = np.interp(np.clip(across / max(r, 1e-6), -1, 1), np.linspace(-1, 1, 129), self.profile)
        w = foot * bris
        if self.tooth_only:
            gl = cv.surface()[y0:y1, x0:x1]; w = w * np.clip((gl - np.percentile(gl, 45)) / 0.03, 0, 1)
        # the tool's velocity, imposed where it touches
        if self.drag > 0:
            cv.U[0, y0:y1, x0:x1] += w * dx * self.drag; cv.U[1, y0:y1, x0:x1] += w * dy * self.drag
        # the plough: paint is squeezed sideways, down the gradient of contact pressure (bristles cut grooves,
        # a stylus throws up a burr on each side); not spread, it acts only under the tool
        if self.plough > 0 and not self.tooth_only:
            pa = np.gradient(w * pressure, axis=1) * -ty + np.gradient(w * pressure, axis=0) * tx   # d/d(across)
            k = -self.plough * pa * (r / nb)                  # displaces by a fraction of one bristle's spacing
            cv.V[0, y0:y1, x0:x1] += k * -ty; cv.V[1, y0:y1, x0:x1] += k * tx
        if self.exchange <= 0: return
        q = cv.q[:, y0:y1, x0:x1]; h = q.sum(0)
        # contact pressure sets how fast paint is exchanged, not how thick it ends up: the rim of the footprint
        # moves toward the same target as the middle, only more slowly
        ex = getattr(self, "_ex_touch", self.exchange)       # per-touch rate (stroke() derives it from the per-pass rate)
        contact = np.clip(2.0 * foot, 0, 1) * (w > 0.05)       # pressed paint is in full contact; the soft rim less
        if self.tooth_only: contact = contact * np.clip(w / np.maximum(foot * bris, 1e-9), 0, 1)
        want = ex * contact * (target * pressure * (0.55 + 0.45 * bris) - h)   # each bristle lays its own ribbon
        give = np.clip(want, 0, None); take = np.clip(-want, 0, None) * 0.25
        L = self.load.sum()
        if L > 1e-12 and give.sum() > 0:
            g = min(1.0, L * 0.25 / give.sum())                 # a brush can only give part of what it holds
            add = (self.load / L)[:, None, None] * (give * g)[None]
            q += add; self.load -= add.sum(axis=(1, 2))
        if take.sum() > 0:                                     # it picks up what's under it, in proportion
            fr = q / np.maximum(h, 1e-12)
            rem = np.minimum(fr * take[None], q)
            q -= rem; self.load += rem.sum(axis=(1, 2))
        np.maximum(self.load, 0, out=self.load)

    def stroke(self, path, speed=2.5, pressure=1.0, target=0.6, steps_after=0):
        """path: [(x, y) or (x, y, pressure)]; the brush moves `speed` cells per step; the canvas steps as it goes."""
        pts = [(p[0], p[1], p[2] if len(p) > 2 else pressure) for p in path]
        # exchange is per pass of the brush over a cell, not per tick: a cell under the footprint is touched about
        # (footprint length / speed) times, so each touch takes its share of the pass's exchange
        foot = 2 * (0.55 * self.width / 2 + 0.8)
        k = max(1.0, (foot + speed) / max(speed, 1e-6))            # touches per cell under a swept footprint
        self._ex_touch = 1 - (1 - min(self.exchange, 0.999)) ** (1 / k)
        seg = [(a, b) for a, b in zip(pts, pts[1:])]
        for (xa, ya, pa), (xb, yb, pb) in seg:
            L = np.hypot(xb - xa, yb - ya); n = max(1, int(np.ceil(L / speed)))
            for i in range(n):
                t0, t1 = i / n, (i + 1) / n
                x, y = xa + (xb - xa) * t1, ya + (yb - ya) * t1
                self.touch(x, y, (xb - xa) / n, (yb - ya) / n, pa + (pb - pa) * t1, target)
                self.cv.step()
        for _ in range(steps_after): self.cv.step()
        del self._ex_touch


class Stylus(Brush):
    def __init__(self, cv, width=2.0, drag=1.0, plough=1.5):
        super().__init__(cv, width, bristles=1, exchange=0.0, drag=drag, plough=plough)
