"""goo.classic.detail: painting a motif with fine features (pixel art, line work, lettering, small glyphs).

A one-cell line cannot be painted by the brush that lays the masses: a broad brush that carries a line's
colour leaves a haze of it around the line, and a line laid wet into wet body colour dissolves into it. So a
painter copying such a motif takes it apart first and paints it in the order the paint needs:

  split()        the motif's broad masses (every feature narrower than `width` cells replaced by the colour
                 around it) and its fine features, as 8-connected chains a small brush can follow
  DetailPainter  a Painter that paints the broad motif region by region (ground, then body colour, then a
                 second coat wherever a region still does not look like the motif), dries it, and then traces
                 the chains with small brushes, one colour at a time, drying between colours; optional raised
                 work last, in short loaded paint

Analysis is a pure function of the motif (its colours and object ids). The painter sees the canvas only as
colour (`reflectance`) and changes it only through brushes, `fix`, `wait` and `step` (classic/SPEC.md §1); it tells
paint is dry the way a painter does, by trying: it waits, fixes, and waits longer while `fix` reports cells it
could not fix.
"""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from goo.classic.brush import Brush
from goo.classic.painter import Painter
from goo.classic.palette import lab

N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class Chain:
    """A fine feature as a brush follows it: cell centres (x, y) in order along it, its colour (as the motif
    gives it) and its width in cells (area over length of the feature it was traced from, rounded)."""
    points: np.ndarray
    colour: np.ndarray
    width: int


@dataclass
class Split:
    broad: np.ndarray        # the motif with every fine feature replaced by the colour around it
    fine: np.ndarray         # (H, W) bool: the cells of fine features
    chains: list             # [Chain], light colours first


def _labels(a):
    """Integer label per cell for the distinct values along the last axis of `a`."""
    flat = a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.reshape(-1, 1)
    vals, inv = np.unique(flat, axis=0, return_inverse=True)
    return vals, inv.reshape(a.shape[:2])


def split(motif, ids=None, width=2):
    """Split a motif (H, W, 3) into broad masses and fine features.

    A cell is fine if, among the cells of its own colour and object (4- or 8-connected alike), it lies in no
    (width + 1)-square: everything `width` cells wide or less is a feature. Fine cells take the colour of the
    nearest broad cell of their own object (an object with no broad cells, a pure texture, takes its commonest
    colour); a feature in the colour around it is not a feature. Features are then traced, by colour, as
    8-connected chains (a diagonal line of corner-touching cells is one chain, not a row of dots)."""
    motif = np.asarray(motif); H, W = motif.shape[:2]
    ids = np.zeros((H, W), int) if ids is None else np.asarray(ids)
    cols, cid = _labels(motif)
    _, key = _labels(np.stack([ids, cid], -1))
    box = np.ones((width + 1, width + 1), bool); pad = width + 1
    fine = np.zeros((H, W), bool)
    for k, sl in enumerate(ndimage.find_objects(key + 1)):
        if sl is None: continue
        ys = slice(max(0, sl[0].start - pad), min(H, sl[0].stop + pad)); xs = slice(max(0, sl[1].start - pad), min(W, sl[1].stop + pad))
        m = key[ys, xs] == k
        fine[ys, xs] |= m & ~ndimage.binary_opening(m, box)
    broad = motif.copy()
    for oid in np.unique(ids):
        obj = ids == oid; todo = obj & fine
        if not todo.any(): continue
        src = obj & ~fine
        if src.any():
            sl = ndimage.find_objects(obj.astype(int))[0]
            _, (iy, ix) = ndimage.distance_transform_edt(~src[sl], return_indices=True)
            t = todo[sl]; b = broad[sl]; b[t] = motif[sl][iy[t], ix[t]]
        else:
            c = np.bincount(cid[obj]).argmax(); broad[todo] = cols[c]
    fine &= np.any(motif != broad, axis=-1)
    chains = []
    for c in np.unique(cid[fine]):
        lbl, n = ndimage.label(fine & (cid == c), structure=np.ones((3, 3)))
        for j, sl in enumerate(ndimage.find_objects(lbl)):
            m = np.pad(lbl[sl] == j + 1, 1)
            sk = skeletonize(m)
            if not sk.any(): sk = m
            w = int(max(1, round(min(m.sum() / sk.sum(), 2 * ndimage.distance_transform_edt(m).max()))))
            cy, cx = np.nonzero(m); r = w / 2 + 0.5
            for path in _trace(sk):                            # each point to the middle of the cells around it
                near = [(np.abs(cy - y) <= r) & (np.abs(cx - x) <= r) for y, x in path]   # (an even width's skeleton is off
                pts = np.array([(cx[k].mean() + sl[1].start - 0.5, cy[k].mean() + sl[0].start - 0.5) for k in near])   # by half a cell)
                chains.append(Chain(pts, cols[c].astype(float), w))
    chains.sort(key=lambda ch: (-float(lab(np.asarray(ch.colour, float) / (255.0 if motif.dtype == np.uint8 else 1.0))[0]),
                                tuple(ch.colour), ch.points[0][1], ch.points[0][0]))
    return Split(broad, fine, chains)


def _trace(sk):
    """Ordered 8-connected paths through a one-cell-thin skeleton: start at an end, keep going as straight as
    the skeleton allows; a branch is a path of its own, from its end to the cell before the junction (the brush's
    footprint closes that cell; running into the junction would lay it twice)."""
    pts = set(zip(*np.nonzero(sk)))
    nb = {p: [(p[0] + dy, p[1] + dx) for dy, dx in N8 if (p[0] + dy, p[1] + dx) in pts] for p in pts}
    free = set(pts); paths = []
    while free:
        start = min(free, key=lambda p: (sum(q in free for q in nb[p]), p))
        path = [start]; free.discard(start)
        while True:
            p = path[-1]; cand = [q for q in nb[p] if q in free]
            if not cand: break
            if len(path) > 1:
                d = np.subtract(p, path[max(0, len(path) - 4)]).astype(float); d /= np.hypot(*d) + 1e-12
            else:
                d = np.zeros(2)
            q = min(cand, key=lambda q: (-np.dot(np.subtract(q, p), d) / np.hypot(*np.subtract(q, p)), abs(q[0] - p[0]) + abs(q[1] - p[1]), q))
            path.append(q); free.discard(q)
        paths.append(path)
    return paths


# the paints and brushes of each kind of work: (target thickness, solid fraction), (exchange, drag, plough)
PAINT = dict(ground=(0.4, 0.45), body=(0.45, 0.5), line=(0.4, 0.5), raised=(2.0, 0.58))
BRUSH = dict(body=(0.85, 0.6, 0.3), line=(0.85, 0.2, 0.08), raised=(0.95, 0.15, 0.05))


class DetailPainter(Painter):
    """A Painter for motifs of flat colours with fine features.

    width:  features this many cells wide or less are traced, not filled (2 texels at 4 cells per texel: 8)
    raised: optional (H, W) bool, the cells whose features are laid last in short, loaded paint, standing proud
    lamp:   the drying lamp: `wait(..., cure_boost=lamp)` between sittings (the World sets how fast oil cures;
            the lamp is how much the painter speeds it up)
    plan:   as for Painter: the stroke direction of each object's regions (an angle); default across the page
    largest: the largest brush it owns (cells)

    It sees only colour; everything it knows about where things are comes from the motif."""

    def __init__(self, cv, motif, ids, palette, width=8, raised=None, lamp=1000.0, plan=None, seed=0, largest=32.0):
        motif = np.asarray(motif, float); ids = np.zeros(motif.shape[:2], int) if ids is None else np.asarray(ids)
        super().__init__(cv, motif, ids, palette, seed=seed, plan=plan)
        self.width, self.lamp, self.largest = width, lamp, largest
        s = split(motif, ids, width)
        self.broad, self.fine = s.broad, s.fine
        rmask = np.zeros(ids.shape, bool) if raised is None else np.asarray(raised, bool)
        at = lambda ch: rmask[ch.points[:, 1].astype(int), ch.points[:, 0].astype(int)].mean() > 0.5
        self.chains = [c for c in s.chains if not at(c)]
        self.raised = [c for c in s.chains if at(c)]
        # the regions of the broad motif: each flat colour of each object, 4-connected
        _, key = _labels(np.concatenate([ids[..., None].astype(float), self.broad], -1))
        self.regions = []
        for k, sl in enumerate(ndimage.find_objects(key + 1)):
            if sl is None: continue
            lbl, n = ndimage.label(key[sl] == k)
            for j, sl2 in enumerate(ndimage.find_objects(lbl)):
                ys = slice(sl[0].start + sl2[0].start, sl[0].start + sl2[0].stop); xs = slice(sl[1].start + sl2[1].start, sl[1].start + sl2[1].stop)
                m = lbl[sl2] == j + 1
                a = np.arctan2(self.plan_dy[ys, xs][m].mean(), self.plan_dx[ys, xs][m].mean())
                self.regions.append(((ys, xs), m, self.broad[ys, xs][m][0], a))
        self.regions.sort(key=lambda r: -r[1].sum())
        self._brushes, self._mixes, self._wet = {}, {}, False
        self.small = max(3.0, (width + 1) / 2)                 # the smallest brush for masses (they are wider than `width`)

    # ---- tools
    def brush(self, width, kind):
        k = (round(float(width), 2), kind)
        if k not in self._brushes:
            ex, dr, pl = BRUSH[kind]
            self._brushes[k] = Brush(self.cv, k[0], bristles=1 if kind == "raised" else max(1, int(width / 2.5)), exchange=ex, drag=dr, plough=pl)
        return self._brushes[k]

    def stroke(self, brush, pts, col, under, kind, speed, taper=0.35):
        """One stroke through pts in the paint of `kind`, mixed so that over `under` it looks like `col`."""
        target, phi = PAINT[kind]; ex, dr, pl = BRUSH.get(kind, BRUSH["body"])
        key = (tuple(np.round(col, 4)), tuple(np.round(under, 3)), brush.width, kind)
        if key not in self._mixes:
            laid = self.swatch(brush.width, target, dr, pl, phi, ex, speed / brush.width)
            self._mixes[key] = self.mix(col, under, laid, phi, opacity_bias=2.0, cured=True)   # it will dry before it is seen
        load = Brush.paint("oil", phi, 1.6 * (self.length(pts) + brush.width) * brush.width * target, self._mixes[key])
        self.lay(brush, pts, load, target, speed, taper=taper); self._wet = True

    # ---- drying, by trying
    def dry(self, ticks=200, tries=7):
        """Wait under the lamp, then fix; while `fix` reports wet cells it could not fix, wait twice as long and
        try again. Waits in short spells (the World integrates its slow laws over each spell). Returns ticks waited."""
        if not self._wet: return 0
        waited = 0
        for _ in range(tries):
            for _ in range(ticks // 10): self.cv.wait(10, cure_boost=self.lamp)
            waited += ticks
            if self.cv.fix() >= 1.0: self._wet = False; return waited
            ticks *= 2
        raise RuntimeError(f"paint still wet after {waited} ticks under the lamp: does this World's oil cure (cure_rate > 0)?")

    # ---- masses
    def fill(self, where, mask, col, under, angle, kind):
        """Cover `mask` (cells, in the window `where`) with parallel strokes along `angle`, the biggest brush that
        fits first; each smaller brush only where the bigger ones could not reach. A brush's rim stays inside the
        mask (the smallest's within a cell of it), except at the panel's edge, which strokes run off."""
        (ys, xs), H, W = where, self.cv.H, self.cv.W
        d, nrm = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])
        inscribed = 2 * ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1].max()
        widths = [self.small]
        while widths[-1] * 2 <= min(self.largest, 0.8 * inscribed): widths.append(widths[-1] * 2)
        P = int(widths[-1]) + 2; pm = np.pad(mask, P, mode="edge")          # past the panel's edge a brush
        if ys.start > 0: pm[:P] = False                                     # may run off; past another
        if ys.stop < H: pm[-P:] = False                                     # region's edge it may not
        if xs.start > 0: pm[:, :P] = False
        if xs.stop < W: pm[:, -P:] = False
        covered = np.zeros_like(pm); n = 0
        yy, xx = np.nonzero(pm); v_all = (xx + 0.5) * d[0] + (yy + 0.5) * d[1]
        vs = np.arange(v_all.min() - 1, v_all.max() + 2, 1.0)
        for w in widths[::-1]:
            e = max(1, int(np.ceil(0.3 * w)))                               # the band a stroke covers fully
            if w > self.small: ok = ndimage.binary_erosion(pm, iterations=max(1, int(np.ceil(w / 2)) - 1))   # its rim stays in
            else: ok = ndimage.binary_erosion(pm); ok |= pm & ~ndimage.binary_dilation(ok)               # the smallest goes everywhere
            need = pm & ~covered
            if need[P:-P, P:-P].sum() < 2: break
            cand = ok & ndimage.binary_dilation(need, iterations=e)
            if not cand.any(): continue
            b = self.brush(w, "body"); sp = max(1.5, 0.5 * w); gap = max(2.0, 0.45 * w)
            cy_, cx_ = np.nonzero(cand); u = (cx_ + 0.5) * nrm[0] + (cy_ + 0.5) * nrm[1]
            us = np.linspace(u.min(), u.max(), int(np.ceil((u.max() - u.min()) / gap)) + 1)   # outermost lines on the edge
            line = np.zeros_like(pm)
            for u0 in us:
                px, py = u0 * nrm[0] + vs * d[0], u0 * nrm[1] + vs * d[1]
                iy, ix = np.floor(py).astype(int), np.floor(px).astype(int)
                on = (iy >= 0) & (iy < pm.shape[0]) & (ix >= 0) & (ix < pm.shape[1])
                hit = np.zeros(len(vs), bool); hit[on] = cand[iy[on], ix[on]]
                edges = np.diff(np.r_[0, hit.astype(np.int8), 0]); starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
                for s0, e0 in zip(starts, ends):
                    a, z = vs[s0], vs[e0 - 1] + 1.0
                    if z - a < 2: a, z = 0.5 * (a + z) - 1, 0.5 * (a + z) + 1
                    for q0 in np.arange(a, z, 64.0):
                        q1 = min(z, q0 + 65.0); k = max(2, int((q1 - q0) / max(w, 4)) + 2)
                        pts = [(u0 * nrm[0] + v * d[0] + xs.start - P, u0 * nrm[1] + v * d[1] + ys.start - P) for v in np.linspace(q0, q1, k)]
                        self.stroke(b, pts, col, under, kind, sp, taper=0.35 if w > self.small else 0.1); n += 1   # the edge brush presses to its ends
                    line[iy[s0:e0], ix[s0:e0]] = True
            covered |= ndimage.binary_dilation(line, iterations=e) & pm
        return n

    def turns(self):
        """The regions in turns such that no two regions of one turn touch (greedy colouring of who touches
        whom): a region is painted only next to dry paint, never wet into a wet neighbour."""
        H, W = self.ids.shape; lbl = np.zeros((H, W), int)
        for i, ((ys, xs), m, col, a) in enumerate(self.regions): lbl[ys, xs][m] = i + 1
        turn = {}
        for i, ((ys, xs), m, col, a) in enumerate(self.regions):
            box = (slice(max(0, ys.start - 2), ys.stop + 2), slice(max(0, xs.start - 2), xs.stop + 2))
            near = set(np.unique(lbl[box][ndimage.binary_dilation(lbl[box] == i + 1, iterations=2)]).tolist()) - {0, i + 1}
            taken = {turn[j - 1] for j in near if j - 1 in turn}
            turn[i] = min(set(range(len(taken) + 1)) - taken)
        return [[i for i in turn if turn[i] == t] for t in range(max(turn.values()) + 1)] if turn else []

    def masses(self, kind="body", only=None):
        """One coat of the broad motif, region by region, each in its own colour over what is seen there, in
        turns of regions that do not touch, each turn dried before the next.
        only: {region index: cell mask} to paint just those cells (a second coat). Returns strokes laid."""
        n = 0
        for group in self.turns():
            seen = self.see()
            for i in group:
                where, m, col, angle = self.regions[i]
                mm = m if only is None else only.get(i)
                if mm is None or not mm.any(): continue
                n += self.fill(where, mm, col, seen[where][mm].mean(0), angle, kind)
            self.dry()
        self.log.append(dict(label=kind, strokes=n, error=self.error())); return n

    def off(self, tol=4.0):
        """Where the painter sees a region still unlike the motif (CIELAB distance > tol, not counting the cells
        along its border with another region, where neighbours meet): {region index: cells to cover again}."""
        seen = lab(self.see()); want = lab(self.broad); bad = {}
        min_cells = int(np.ceil(self.small))
        for i, (where, m, col, angle) in enumerate(self.regions):
            (ys, xs), H, W = where, self.cv.H, self.cv.W
            core = ndimage.binary_erosion(np.pad(m, 1, mode="edge"), border_value=1)[1:-1, 1:-1]
            if ys.start > 0: core[0] = False
            if ys.stop < H: core[-1] = False
            if xs.start > 0: core[:, 0] = False
            if xs.stop < W: core[:, -1] = False
            far = core & (np.linalg.norm(seen[where] - want[where], axis=-1) > tol)
            if far.sum() >= min_cells: bad[i] = m & ndimage.binary_dilation(far, iterations=2)
        return bad

    # ---- features
    def trace(self, chains, kind="line", seen=None):
        """Lay each chain as a stroke along it, with a brush as wide as the feature (a dab for a dot)."""
        seen = self.see() if seen is None else seen
        for ch in chains:
            w = max(2.0, float(ch.width))
            b = self.brush(w, "raised" if kind == "raised" else "line")
            p = ch.points; step = max(2.0, w)
            keep = [0]
            for i in range(1, len(p)):
                if np.hypot(*(p[i] - p[keep[-1]])) >= step: keep.append(i)
            if keep[-1] != len(p) - 1: keep.append(len(p) - 1)
            pts = [tuple(p[i]) for i in keep]
            if self.length(pts) < 0.6 * w:                         # a dot: a short dab along the plan
                cx, cy = p.mean(0); a = 0.3 * w; ux, uy = self.plan_dx[int(cy), int(cx)], self.plan_dy[int(cy), int(cx)]
                pts = [(cx - a * ux, cy - a * uy), (cx + a * ux, cy + a * uy)]
            self.stroke(b, pts, ch.colour, self.under(seen, pts), kind, 0.6 if kind == "raised" else max(1.5, 0.5 * w), taper=0.15)

    def features(self, chains=None, kind="line", tol=8.0):
        """The chains, one colour at a time (light first), each colour dried and fixed before the next; a
        second coat on any chain the painter still sees off by more than `tol` once dry."""
        chains = self.chains if chains is None else chains
        groups = {}
        for ch in chains: groups.setdefault(tuple(np.round(ch.colour, 6)), []).append(ch)
        for col, g in groups.items():
            self.trace(g, kind); self.dry()
            seen = self.see(); L = lab(seen); want = lab(np.array(col))
            again = [ch for ch in g if np.linalg.norm(L[ch.points[:, 1].astype(int), ch.points[:, 0].astype(int)] - want, axis=-1).mean() > tol]
            if again: self.trace(again, kind, seen); self.dry()
            self.log.append(dict(label=f"{kind} {np.round(col, 3)}", chains=len(g), again=len(again), error=self.error()))

    def paint(self, coats=2, tol=4.0):
        """All the sittings: ground, body colour, second coats where needed, the features, the raised work."""
        self.masses("ground")
        self.masses("body")
        for _ in range(coats):
            bad = self.off(tol)
            if not bad: break
            self.masses("body", only=bad)
        self.features()
        if self.raised: self.features(self.raised, kind="raised")
        return self.log

