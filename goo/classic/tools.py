"""tools.py: the tools that are not brushes. A knife, a blotter, and spatter flicked from a brush.

All of them are Tools in the sense of classic/SPEC.md §1 and §5: they change the World only by writing the tool
velocities and by exchanging paint between the wet layer and a load (booked in the ledger, I1), through the
tool channel (cv._q, cv._U, cv._V, marked with cv._mark / cv._mark_tool, I8). None touches a stratum (I5).
Where the canvas has tape (cv._open, a float window: 1 open, 0 masked), every contact is multiplied by it.

  Knife(cv, width)     a rigid blade: lays to a flat target, takes all the excess under it, ploughs to its ends
  Blotter(cv, width)   tissue, sponge, a thirsty brush: takes what is wet, gives nothing, writes no velocity
  spatter(brush, ...)  droplets from a brush's own load, placed by the rng it is given

One interface for every tool (Brush, Knife, Blotter; spatter acts through a Brush):

  tool.charge({species: volume})                    load it (the old load goes to the rinse jar first); booked
  tool.stroke(path, speed=2.5, pressure=1.0, target=..., steps_after=0, heading=None) -> Receipt
                                                    draw it along path [(x, y) or (x, y, pressure)], cells per
                                                    tick; `target` the thickness it works toward (Brush 0.6,
                                                    Knife 0.6, Blotter its own `target`, default 0); `heading`
                                                    the blade's or flat's long axis (radians), None square to
                                                    the travel (a Blotter is a disc and has none)
  tool.press(x, y, pressure=1.0, ticks=1, target=...) -> Receipt
                                                    hold it down, unmoving, for `ticks` ticks
  tool.rinse() -> Volumes                           empty its load into the rinse jar (booked); Blotter.wring()
                                                    and Knife.wipe() are the same act by their own names
  spatter(brush, at, n, spread, size, rng) -> Receipt
                                                    n droplets from brush's load around at = (x, y)
  tool.twin(fork)                                   the same tool on a Canvas.fork() (a rehearsal)

Every Receipt has given / taken (per species), cells, bbox, mask(H, W), length, ticks (goo.classic.brush.Receipt).
"""
import numpy as np


def opening(cv, y0, y1, x0, x1):
    """The canvas's tape over [y0:y1, x0:x1] as a float window (1 open, 0 masked), or None when it has no mask
    (or no masking at all). Every tool multiplies its contact (giving, taking, velocity) by it."""
    f = getattr(cv, "_open", None)
    if f is None: return None
    o = f(y0, y1, x0, x1)
    return None if o is None else np.asarray(o, dtype=float)


from goo.classic.brush import Brush, Receipt, Volumes     # noqa: E402  (brush.py imports `opening` from here)
from goo.classic.engine import CARRIERS                   # noqa: E402

NC = len(CARRIERS)          # a canvas's species are CARRIERS + its pigments (SPEC §4)
PHI_M = 0.64                # the engine's default jamming fraction (Canvas.L["phi_m"])


def _seg_dist(px, py, ax, ay, bx, by):
    """Distance from points (px, py) to the segment a-b."""
    vx, vy = bx - ax, by - ay; vv = vx * vx + vy * vy
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / vv, 0, 1) if vv > 1e-18 else 0.0
    return np.hypot(px - ax - t * vx, py - ay - t * vy)


class Knife(Brush):
    """A painting knife: a rigid, straight edge `width` cells long, held at `heading` (radians, image frame; by
    default square to the travel). It has no bristles and does not give under pressure.

    - Lays paint from its load toward a flat `target` (no bristle ribbons), at `exchange` per pass.
    - Takes ALL the wet paint above the target under the blade (a brush takes a quarter of the excess per touch),
      in the canvas's proportions, onto the blade (its load). Unloaded with a low target it scrapes the wet layer
      back toward what is beneath; strata are never touched (I5).
    - Ploughs: in the last `edge` cells at each end of the blade it takes nothing and pushes the paint outward
      along the blade (plough cells per tick), so a pass leaves a flat floor with a ridge at each side.
    - Drags the paint it meets (a bead rides in front of the blade) like a brush, at `drag`.
    """
    HALF = 0.6              # half the thickness of the edge's contact (cells); the rim is anti-aliased over 1 cell

    def __init__(self, cv, width, load=None, exchange=1.0, drag=0.0, plough=2.0, edge=1.5):
        super().__init__(cv, width, bristles=1, load=load, exchange=exchange, drag=drag, plough=plough,
                         shape="flat", aspect=max(1.0, width / (2 * self.HALF)))
        self.edge = edge

    def wipe(self):
        """Wipe the blade clean into the rinse jar (booked as rinsed). Returns what was on it."""
        return self.rinse()

    def _semi_along(self, tx, ty, heading):
        c = np.cos(heading) * tx + np.sin(heading) * ty
        return float(self.width / 2 * abs(c) + self.HALF + 0.5)

    def touch(self, x, y, dx, dy, pressure, target, heading=None):
        cv = self.cv
        sp = np.hypot(dx, dy); tx, ty = (dx / sp, dy / sp) if sp > 1e-9 else (1.0, 0.0)
        hd = np.arctan2(ty, tx) + np.pi / 2 if heading is None else heading
        ex_, ey_ = np.cos(hd), np.sin(hd); a = self.width / 2
        m = a + self.HALF + 2
        xs_, ys_ = (x - dx, x), (y - dy, y)
        x0, x1 = int(max(0, min(xs_) - m)), int(min(cv.W, max(xs_) + m + 1)); y0, y1 = int(max(0, min(ys_) - m)), int(min(cv.H, max(ys_) + m + 1))
        if x0 >= x1 or y0 >= y1: return
        yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
        px, py = xx - x, yy - y
        # the swept blade: p = u e - t d, |u| <= a, 0 <= t <= 1 (a parallelogram; a segment when edge-on)
        A, B = (-a * ex_, -a * ey_), (a * ex_, a * ey_)
        d0 = np.minimum(_seg_dist(px, py, *A, *B), _seg_dist(px, py, A[0] - dx, A[1] - dy, B[0] - dx, B[1] - dy))
        cr = ex_ * dy - ey_ * dx
        if abs(cr) > 1e-9:
            u = (dx * py - dy * px) / -cr; t = (ex_ * py - ey_ * px) / -cr
            inside = (np.abs(u) <= a) & (t >= 0) & (t <= 1)
            d0 = np.minimum(d0, np.minimum(_seg_dist(px, py, *B, B[0] - dx, B[1] - dy), _seg_dist(px, py, *A, A[0] - dx, A[1] - dy)))
            dist = np.where(inside, 0.0, d0)
            endw = np.clip((np.abs(u) - (a - self.edge)) / max(self.edge, 1e-9), 0, 1)   # at the blade's ends
        else:                                           # edge-on: the blade cuts a line along its own length
            u = px * ex_ + py * ey_; dist = d0; endw = np.zeros_like(dist)
        foot = np.clip(self.HALF + 0.5 - dist, 0, 1)
        if not foot.any(): return
        op = opening(cv, y0, y1, x0, x1)
        if op is not None: foot = foot * op
        self._seen(y0, x0, foot > 0)
        if self.drag > 0:
            cv._U[0, y0:y1, x0:x1] += foot * dx * self.drag; cv._U[1, y0:y1, x0:x1] += foot * dy * self.drag
            cv._mark_tool(y0, y1, x0, x1)
        if self.plough > 0 and endw.any():              # out along the blade, away from its middle
            k = self.plough * foot * endw * np.sign(u)
            cv._V[0, y0:y1, x0:x1] += k * ex_; cv._V[1, y0:y1, x0:x1] += k * ey_
            cv._mark_tool(y0, y1, x0, x1)
        q = cv._q[:, y0:y1, x0:x1]; h = q.sum(0)
        ex = getattr(self, "_ex_touch", self.exchange)
        give = ex * foot * min(pressure, 1.0) * np.clip(target - h, 0, None)        # a flat target: no ribbons
        take = foot * (1 - endw) * np.clip(h - target, 0, None)                     # all the excess under the blade
        L = self.load.sum()
        if L > 1e-12 and give.sum() > 0:
            g = min(1.0, L * 0.25 / give.sum())
            add = (self.load / L)[:, None, None] * (give * g)[None]
            q += add; s = add.sum(axis=(1, 2)); self.load -= s; self._gave(s); cv._mark(y0, y1, x0, x1)
        if take.sum() > 0:
            fr = q / np.maximum(h, 1e-12)
            rem = np.minimum(fr * take[None], q)
            q -= rem; s = rem.sum(axis=(1, 2)); self.load += s; self._took(s)
        np.maximum(self.load, 0, out=self.load)


class Blotter(Brush):
    """Contact without motion: tissue, a sponge, a thirsty brush. A disc `width` cells across (swept, if it is
    drawn along a path with stroke()). Where it touches it takes wet paint toward a low `target`, at `exchange`
    per touch (pass, for a stroke), and gives nothing; it writes no velocity.

    What it takes: liquid wicks into the sheet; pigment comes with it only as far as it is suspended. All the
    carrier (water, oil) is mobile; of the pigment, the fraction (1 - phi / phi_m)+, phi the cell's solid
    fraction. A dilute wash is taken nearly in the canvas's proportions (so blotting lightens it); a stiff paste
    gives up its carrier and keeps its pigment; paint with no carrier left (dry, charcoal) gives up nothing.

    It holds at most `capacity` volume units (default 0.25 per cell of its area): it stops taking when full.
    `wring()` empties it into the rinse jar, booked."""

    def __init__(self, cv, width, capacity=None, exchange=0.5, target=0.0):
        super().__init__(cv, width, bristles=1, exchange=exchange, drag=0.0, plough=0.0)
        self.capacity = float(capacity) if capacity is not None else 0.25 * np.pi * (width / 2) ** 2
        self.target = target

    @property
    def room(self): return max(0.0, self.capacity - float(self.load.sum()))

    def wring(self):
        """Empty what it soaked up into the rinse jar (booked as rinsed). Returns it."""
        return self.rinse()

    def _semi_along(self, tx, ty, heading): return self.width / 2

    def touch(self, x, y, dx, dy, pressure, target=None, heading=None):
        cv = self.cv
        target = self.target if target is None else target
        r = self.width / 2
        x0, x1 = int(max(0, min(x - dx, x) - r - 2)), int(min(cv.W, max(x - dx, x) + r + 3))
        y0, y1 = int(max(0, min(y - dy, y) - r - 2)), int(min(cv.H, max(y - dy, y) + r + 3))
        if x0 >= x1 or y0 >= y1: return
        yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
        foot = np.clip(r + 0.5 - _seg_dist(xx - x, yy - y, 0.0, 0.0, -dx, -dy), 0, 1)
        if not foot.any(): return
        op = opening(cv, y0, y1, x0, x1)
        if op is not None: foot = foot * op
        self._seen(y0, x0, foot > 0)
        room = self.room
        if room <= 0: return
        q = cv._q[:, y0:y1, x0:x1]; h = q.sum(0)
        ex = getattr(self, "_ex_touch", self.exchange)
        pig = q[NC:].sum(0); phi = pig / np.maximum(h, 1e-12)
        s = np.clip(1 - phi / PHI_M, 0, 1)                         # the suspended share of the pigment
        mob = np.concatenate([np.ones((NC,) + h.shape), np.broadcast_to(s, (len(q) - NC,) + h.shape)]) * q
        mobile = mob.sum(0)
        want = np.minimum(ex * foot * min(pressure, 1.0) * np.clip(h - target, 0, None), mobile)
        if not want.sum() > 0: return
        want = want * min(1.0, room / want.sum())                   # it stops taking when full
        rem = np.minimum(mob / np.maximum(mobile, 1e-300) * want, q)
        q -= rem; a = rem.sum(axis=(1, 2)); self.load += a; self._took(a)

    def press(self, x, y, pressure=1.0, ticks=1, target=None):
        """Hold it down at (x, y) for `ticks` ticks (the canvas steps each one). Returns a Receipt."""
        return super().press(x, y, pressure=pressure, ticks=ticks, target=self.target if target is None else target)

    def stroke(self, path, speed=2.5, pressure=1.0, target=None, steps_after=0, heading=None):
        """A thirsty brush drawn along a path: lifts as it goes, drags nothing. Returns a Receipt."""
        return super().stroke(path, speed=speed, pressure=pressure, target=self.target if target is None else target,
                              steps_after=steps_after)


DROP_FILM = 0.3   # thickness (volume units, 75 um) a landed droplet spreads to before the laws take over


def spatter(brush, at, n, spread, size, rng):
    """Flick `n` droplets from `brush`'s load around `at` = (x, y). Each droplet's volume is `size` times a
    Gamma(2, 1/2) draw (mean `size`, volume units) and its landing point is `at` plus `spread` times a standard
    normal draw in x and y, all from `rng` (so a given rng gives the same spatter). A droplet lands as a small
    dome (radius from its volume at DROP_FILM thick, at least 0.75 cells) with no drag and no velocity, in the
    load's proportions; it is clipped to the load, so the total given never exceeds it. The part of a droplet
    that would land off the panel or on tape is not given (it stays on the brush). The canvas steps once.
    Returns a Receipt."""
    cv = brush.cv
    rec = Receipt(cv.species)
    vols = size * rng.gamma(2.0, 0.5, n)
    pos = np.asarray(at, float)[None, :] + spread * rng.standard_normal((n, 2))
    for v, (cx, cy) in zip(vols, pos):
        L = brush.load.sum()
        if L <= 1e-12: break
        v = min(v, L)
        rho = max(0.75, np.sqrt(v / (np.pi * DROP_FILM)))
        X0, X1 = int(np.floor(cx - rho)), int(np.ceil(cx + rho)) + 1; Y0, Y1 = int(np.floor(cy - rho)), int(np.ceil(cy + rho)) + 1
        yy, xx = np.mgrid[Y0:Y1, X0:X1] + 0.5
        k = np.clip(1 - ((xx - cx) ** 2 + (yy - cy) ** 2) / rho ** 2, 0, None)
        if not k.sum() > 0: continue
        k = k / k.sum()                                  # the whole droplet, on and off the panel
        x0, x1, y0, y1 = max(0, X0), min(cv.W, X1), max(0, Y0), min(cv.H, Y1)
        if x0 >= x1 or y0 >= y1: continue
        k = k[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
        op = opening(cv, y0, y1, x0, x1)
        if op is not None: k = k * op
        if not k.sum() > 0: continue
        add = (brush.load / L)[:, None, None] * (v * k)[None]
        cv._q[:, y0:y1, x0:x1] += add; s = add.sum(axis=(1, 2))
        brush.load -= s; np.maximum(brush.load, 0, out=brush.load); rec.given += s
        cv._mark(y0, y1, x0, x1); rec._touch(y0, x0, k > 0)
    cv.step(); rec.ticks += 1
    return rec


__all__ = ["Knife", "Blotter", "spatter", "opening", "Receipt", "Volumes"]
