"""brush.py: tools that touch the paint.

A Brush is a body: a row of bristles and a load (volumes of each species). Where it touches, it does two
things. It imposes its velocity on the paint (bristles drag harder than the gaps), and it exchanges paint
with the canvas toward a target thickness under each bristle: where the canvas is thinner it gives from its
load (in the load's proportions), where thicker it picks up (in the canvas's proportions). So a brush gets
dirty, mixes on the canvas, runs dry and stops giving, all by the same rule.

A round brush (the default) is an oval footprint that follows its direction of travel, bristles across it.
A flat brush (shape="flat") holds its long axis at a fixed `heading` whatever the direction of travel, with
the bristles along that axis: on its face (moving across the long axis) it lays a band as wide as the brush,
on its edge (moving along it) a line width / aspect wide, so a turning stroke goes thick and thin.

A Stylus is a brush with no load and no exchange: it only drags (sgraffito, combing, marbling rods).
Charcoal is a brush whose load is a dry pigment with no carrier: it deposits only on the ground's tooth,
and the deposit (phi = 1) is jammed solid where it lands.

Every stroke returns a Receipt: what it gave and took (per species), how many cells it touched, how far it
went and how many ticks it took. The knife, the blotter and spatter are in goo.tools.
"""
import itertools
import numpy as np

_IDS = itertools.count()                                  # every brush made in this process gets its own number


class Volumes(np.ndarray):
    """Per-species volumes (a float array in the canvas's species order) with `.volume`, their total."""
    def __new__(cls, a): return np.array(a, dtype=float).view(cls)

    @property
    def volume(self): return float(np.asarray(self).sum())


class Receipt:
    """What a stroke (or several, summed with +) did. Fields:

    - `species`: list of str, the canvas's species in order (CARRIERS, then its pigments);
    - `given`, `taken`: `Volumes` (float arrays in that order, `.volume` their total; volume units, SPEC §2):
      paint that left the tool for the wet layer, and paint the tool picked up from it. `given.sum() == 0`
      means the stroke laid nothing;
    - `length`: float, path length in cells; `ticks`: int, canvas steps taken;
    - `cells`: int, how many distinct cells its footprint touched (open cells only: tape is not touched);
    - `bbox`: (y0, y1, x0, x1) of those cells, or None; `mask(H, W)`: them as an (H, W) bool array.
    `by_species("given" | "taken")` gives {species: volume} without the zeros."""

    def __init__(self, species, given=None, taken=None, length=0.0, ticks=0, touched=None):
        self.species = list(species)
        n = len(self.species)
        self.given = Volumes(np.zeros(n) if given is None else given)
        self.taken = Volumes(np.zeros(n) if taken is None else taken)
        self.length, self.ticks = float(length), int(ticks)
        self._touched = touched                           # (y0, x0, bool block) or None

    @property
    def cells(self):
        return 0 if self._touched is None else int(self._touched[2].sum())

    @property
    def bbox(self):
        """(y0, y1, x0, x1) of the touched cells (panel coordinates), or None if it touched none."""
        if self._touched is None: return None
        y0, x0, m = self._touched
        ys, xs = np.nonzero(m)
        return int(y0 + ys.min()), int(y0 + ys.max() + 1), int(x0 + xs.min()), int(x0 + xs.max() + 1)

    def mask(self, H, W):
        """The touched cells as an (H, W) bool array (a panel of that size)."""
        out = np.zeros((H, W), bool)
        if self._touched is None: return out
        y0, x0, m = self._touched
        out[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m[:H - y0, :W - x0]
        return out

    def _touch(self, y0, x0, m):
        """Record that the cells where m (a bool block at (y0, x0)) is True were touched."""
        if not m.any(): return
        self._touched = _merge(self._touched, (y0, x0, m))

    def by_species(self, which="given"):
        a = getattr(self, which); return {s: float(v) for s, v in zip(self.species, a) if v != 0}

    def __add__(self, other):
        if other == 0: return self                        # so sum(receipts) works
        if not isinstance(other, Receipt): return NotImplemented
        if other.species != self.species: raise ValueError("receipts from canvases with different species")
        return Receipt(self.species, self.given + other.given, self.taken + other.taken, self.length + other.length,
                       self.ticks + other.ticks, _merge(self._touched, other._touched))
    __radd__ = __add__

    def __repr__(self):
        return (f"Receipt(given={self.given.volume:.4g}, taken={self.taken.volume:.4g}, cells={self.cells}, "
                f"length={self.length:.4g}, ticks={self.ticks})")


def _merge(a, b):
    if a is None: return b
    if b is None: return a
    (ay, ax, am), (by, bx, bm) = a, b
    y0, x0 = min(ay, by), min(ax, bx)
    y1, x1 = max(ay + am.shape[0], by + bm.shape[0]), max(ax + am.shape[1], bx + bm.shape[1])
    m = np.zeros((y1 - y0, x1 - x0), bool)
    m[ay - y0:ay - y0 + am.shape[0], ax - x0:ax - x0 + am.shape[1]] |= am
    m[by - y0:by - y0 + bm.shape[0], bx - x0:bx - x0 + bm.shape[1]] |= bm
    return (y0, x0, m)


class Brush:
    def __init__(self, cv, width, bristles=None, load=None, exchange=0.6, tooth_only=False, drag=1.0, plough=0.5,
                 shape="round", aspect=3.0):
        if shape not in ("round", "flat"): raise ValueError(f"shape must be 'round' or 'flat', not {shape!r}")
        if shape == "flat" and not aspect >= 1: raise ValueError("a flat brush's aspect (long axis / short) is >= 1")
        self.cv, self.width, self.exchange, self.tooth_only, self.drag, self.plough = cv, width, exchange, tooth_only, drag, plough
        self.shape, self.aspect = shape, float(aspect)
        self.bristles = bristles if bristles is not None else max(1, int(width / 2.5))
        self.load = np.zeros(len(cv.species))
        # a real brush: bristles clump unevenly; its profile across the stroke is its own, fixed for its life
        rng = np.random.default_rng(1000 + len(cv.brushes))
        s = np.linspace(-1, 1, 129); nb = self.bristles
        c = np.sort(rng.uniform(-1, 1, nb)) if nb > 1 else np.zeros(1)
        prof = 0.45 + sum(a * np.exp(-((s - ci) / wi) ** 2) for ci, a, wi in zip(c, rng.uniform(0.4, 1.0, nb), rng.uniform(0.4, 1.1, nb) / max(nb, 1)))
        self.profile = prof / prof.max() if nb > 1 else np.ones_like(s)
        self._rec = None                                  # the Receipt of the stroke in progress
        self._id = next(_IDS)                             # kept by a fork's copy: how twin() knows its twin
        cv.brushes.append(self)
        if load: self.charge(load)

    def twin(self, fork):
        """This brush's twin on `fork` (a `Canvas.fork()` of this brush's canvas): same load, same bristle profile,
        bound to the fork. A rehearsal holds the same brush the painting does."""
        i = next((k for k, b in enumerate(self.cv.brushes) if b is self), None)
        if i is None: raise ValueError("this brush is not registered with its canvas")
        tw = fork.brushes[i] if i < len(fork.brushes) else None
        if tw is None or type(tw) is not type(self) or getattr(tw, "_id", None) != getattr(self, "_id", None):
            raise ValueError("that canvas is not a fork of this brush's canvas")
        return tw

    def rinse(self):
        """Empty the load into the rinse jar: booked as rinsed and out of `loaded` (I1). Returns what was rinsed."""
        cv = self.cv
        cv.led.setdefault("rinsed", np.zeros(len(cv.species)))
        out = self.load
        cv.led["rinsed"] += self.load
        cv.led["loaded"] -= self.load            # the jar returns it to the books as not-on-canvas
        self.load = np.zeros(len(cv.species))
        return Volumes(out)

    def charge(self, load):
        """Load paint: {species: volume}. What was left goes in the rinse jar (counted as rinsed, not lost)."""
        cv = self.cv
        self.rinse()
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

    # ---- the receipt of the stroke in progress (touch() called alone books nothing)
    def _gave(self, a):
        if self._rec is not None: self._rec.given += a

    def _took(self, a):
        if self._rec is not None: self._rec.taken += a

    def _seen(self, y0, x0, m):
        if self._rec is not None: self._rec._touch(y0, x0, m)

    def _flat_foot(self, xx, yy, x, y, dx, dy, r, heading):
        """Footprint of a flat brush swept from (x - dx, y - dy) to (x, y): the ellipse of semi-axes r (along
        `heading`) and r / aspect, Minkowski-summed with the segment (an exact distance in the ellipse's own
        metric). Returns (foot, offset along the long axis from the nearest point of the segment)."""
        a = r; b = max(r / self.aspect, 0.8)
        cl, sl = np.cos(heading), np.sin(heading)
        px, py = xx - x, yy - y
        pl = (px * cl + py * sl) / a; ps = (-px * sl + py * cl) / b
        dl = -(dx * cl + dy * sl) / a; ds = -(-dx * sl + dy * cl) / b
        dd = dl * dl + ds * ds
        t = np.clip((pl * dl + ps * ds) / dd, 0, 1) if dd > 1e-18 else 0.0
        el, es = pl - t * dl, ps - t * ds
        foot = np.clip(1.6 - np.hypot(el, es) * 1.6, 0, 1)
        return foot, el * a

    def _semi_along(self, tx, ty, heading):
        """Half-length of the footprint along the direction of travel (r at full pressure)."""
        r = self.width / 2
        if self.shape == "round": return 0.55 * r + 0.8
        a, b = r, max(r / self.aspect, 0.8)
        c = np.cos(heading) * tx + np.sin(heading) * ty        # cos of the angle between travel and the long axis
        return float(np.sqrt((a * c) ** 2 + (b * b) * (1 - c * c)))

    def touch(self, x, y, dx, dy, pressure, target, heading=None):
        cv = self.cv
        flat = self.shape == "flat"
        if heading is not None and not flat: raise ValueError("a round brush has no heading; use shape='flat'")
        sp = np.hypot(dx, dy); tx, ty = (dx / sp, dy / sp) if sp > 1e-9 else (1.0, 0.0)
        r = self.width / 2 * (0.6 + 0.4 * pressure)
        # the footprint is swept: the capsule from where the brush was (x - dx, y - dy) to where it is
        xs_, ys_ = (x - dx, x), (y - dy, y)
        x0, x1 = int(max(0, min(xs_) - r - 2)), int(min(cv.W, max(xs_) + r + 3)); y0, y1 = int(max(0, min(ys_) - r - 2)), int(min(cv.H, max(ys_) + r + 3))
        if x0 >= x1 or y0 >= y1: return
        yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
        along = (xx - x) * tx + (yy - y) * ty; across = -(xx - x) * ty + (yy - y) * tx
        if not flat:
            along = along - np.clip(along, -sp, 0.0)                    # inside the swept segment: no along-distance
            foot = np.clip(1.6 - np.hypot(along / (0.55 * r + 0.8), across / r) * 1.6, 0, 1)   # soft-edged oval
            lng = across                                                # the bristles lie across the travel
        else:
            hd = np.arctan2(ty, tx) + np.pi / 2 if heading is None else heading   # no heading: held square to the travel
            foot, lng = self._flat_foot(xx, yy, x, y, dx, dy, r, hd)    # the bristles lie along the long axis
        if not foot.any(): return
        op = _opening(cv, y0, y1, x0, x1)                               # tape: 1 open, 0 masked; None: no mask
        self._seen(y0, x0, foot > 0 if op is None else (foot > 0) & (op > 0))
        nb = self.bristles
        bris = np.interp(np.clip(lng / max(r, 1e-6), -1, 1), np.linspace(-1, 1, 129), self.profile)
        w = foot * bris
        if self.tooth_only:
            gl = cv._surface_window(y0, y1, x0, x1); w = w * np.clip((gl - np.percentile(gl, 45)) / 0.03, 0, 1)
        wv = w if op is None else w * op
        # the tool's velocity, imposed where it touches (and the window marked, so the canvas looks only there)
        if self.drag > 0:
            cv._U[0, y0:y1, x0:x1] += wv * dx * self.drag; cv._U[1, y0:y1, x0:x1] += wv * dy * self.drag
            cv._mark_tool(y0, y1, x0, x1)
        # the plough: paint is squeezed sideways, down the gradient of contact pressure (bristles cut grooves,
        # a stylus throws up a burr on each side); not spread, it acts only under the tool
        if self.plough > 0 and not self.tooth_only:
            pa = np.gradient(w * pressure, axis=1) * -ty + np.gradient(w * pressure, axis=0) * tx   # d/d(across)
            k = -self.plough * pa * (r / nb)                  # displaces by a fraction of one bristle's spacing
            if op is not None: k = k * op
            cv._V[0, y0:y1, x0:x1] += k * -ty; cv._V[1, y0:y1, x0:x1] += k * tx
            cv._mark_tool(y0, y1, x0, x1)
        if self.exchange <= 0: return
        q = cv._q[:, y0:y1, x0:x1]; h = q.sum(0)
        # contact pressure sets how fast paint is exchanged, not how thick it ends up: the rim of the footprint
        # moves toward the same target as the middle, only more slowly
        ex = getattr(self, "_ex_touch", self.exchange)       # per-touch rate (stroke() derives it from the per-pass rate)
        contact = np.clip(2.0 * foot, 0, 1) * (w > 0.05)       # pressed paint is in full contact; the soft rim less
        if self.tooth_only: contact = contact * np.clip(w / np.maximum(foot * bris, 1e-9), 0, 1)
        if op is not None: contact = contact * op
        want = ex * contact * (target * pressure * (0.55 + 0.45 * bris) - h)   # each bristle lays its own ribbon
        give = np.clip(want, 0, None); take = np.clip(-want, 0, None) * 0.25
        L = self.load.sum()
        if L > 1e-12 and give.sum() > 0:
            g = min(1.0, L * 0.25 / give.sum())                 # a brush can only give part of what it holds
            add = (self.load / L)[:, None, None] * (give * g)[None]
            q += add; a = add.sum(axis=(1, 2)); self.load -= a; self._gave(a); cv._mark(y0, y1, x0, x1)
        if take.sum() > 0:                                     # it picks up what's under it, in proportion
            fr = q / np.maximum(h, 1e-12)
            rem = np.minimum(fr * take[None], q)
            q -= rem; a = rem.sum(axis=(1, 2)); self.load += a; self._took(a)
        np.maximum(self.load, 0, out=self.load)

    def _ex_per_touch(self, speed, tx=1.0, ty=0.0, heading=None):
        """Per-touch exchange from the per-pass rate: a cell under the swept footprint is touched about
        (footprint length along the travel + speed) / speed times."""
        foot = 2 * self._semi_along(tx, ty, heading)
        k = max(1.0, (foot + speed) / max(speed, 1e-6))
        return 1 - (1 - min(self.exchange, 0.999)) ** (1 / k)

    def stroke(self, path, speed=2.5, pressure=1.0, target=0.6, steps_after=0, heading=None):
        """path: [(x, y) or (x, y, pressure)]; the brush moves `speed` cells per step; the canvas steps as it goes.
        heading (flat brushes only): the angle of the footprint's long axis, radians in the image frame (x right,
        y down); None holds it square to the travel. Returns a Receipt."""
        if heading is not None and self.shape != "flat":
            raise ValueError("a round brush has no heading; use shape='flat'")
        rec = Receipt(self.cv.species); self._rec = rec
        pts = [(p[0], p[1], p[2] if len(p) > 2 else pressure) for p in path]
        # exchange is per pass of the brush over a cell, not per tick: a cell under the footprint is touched about
        # (footprint length / speed) times, so each touch takes its share of the pass's exchange
        follows = self.shape != "round"                          # its length along the travel depends on the heading
        if not follows: self._ex_touch = self._ex_per_touch(speed)
        seg = [(a, b) for a, b in zip(pts, pts[1:])]
        try:
            for (xa, ya, pa), (xb, yb, pb) in seg:
                L = np.hypot(xb - xa, yb - ya); n = max(1, int(np.ceil(L / speed)))
                if follows:
                    tx, ty = ((xb - xa) / L, (yb - ya) / L) if L > 1e-9 else (1.0, 0.0)
                    hd = np.arctan2(ty, tx) + np.pi / 2 if heading is None else heading
                    self._ex_touch = self._ex_per_touch(speed, tx, ty, hd)
                for i in range(n):
                    t0, t1 = i / n, (i + 1) / n
                    x, y = xa + (xb - xa) * t1, ya + (yb - ya) * t1
                    if heading is None: self.touch(x, y, (xb - xa) / n, (yb - ya) / n, pa + (pb - pa) * t1, target)
                    else: self.touch(x, y, (xb - xa) / n, (yb - ya) / n, pa + (pb - pa) * t1, target, heading=heading)
                    self.cv.step(); rec.ticks += 1
                rec.length += float(L)
            for _ in range(steps_after): self.cv.step(); rec.ticks += 1
        finally:
            if hasattr(self, "_ex_touch"): del self._ex_touch
            self._rec = None
        return rec


    def press(self, x, y, pressure=1.0, ticks=1, target=0.0):
        """Hold the tool down at (x, y) without moving it for `ticks` ticks (the canvas steps each one): a dab,
        a touch of a blotter. Returns a Receipt."""
        rec = Receipt(self.cv.species); self._rec = rec
        try:
            for _ in range(int(ticks)):                       # each tick a touch at the tool's own `exchange`
                self.touch(x, y, 0.0, 0.0, pressure, target); self.cv.step(); rec.ticks += 1
        finally:
            self._rec = None
        return rec


class Stylus(Brush):
    def __init__(self, cv, width=2.0, drag=1.0, plough=1.5):
        super().__init__(cv, width, bristles=1, exchange=0.0, drag=drag, plough=plough)


from goo.tools import opening as _opening   # noqa: E402  (tools.py defines it before it imports this module)
