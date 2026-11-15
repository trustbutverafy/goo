"""Regions, paths and footprints as plain data (docs/PORTS.md §2).

Coordinates are CELL coordinates unless a description says `"units": "m"`: x runs along columns, y down rows, and
cell (row i, column j) has its centre at (x, y) = (j + 1/2, i + 1/2). A cell is inside a shape when its centre is.
In metres, x = column coordinate times dx, from the panel's top-left corner. A `units` key applies to the
description it is in and to every description nested in it (a child may say otherwise).

Regions (JSON):
    "all"
    {"rect": [x0, y0, x1, y1]}                 x0 <= x < x1, y0 <= y < y1 (cell centres)
    {"disc": [cx, cy, r]}
    {"polygon": [[x, y], ...]}                 even-odd rule
    {"band": {"rows": [y0, y1]}}  or  {"band": {"cols": [x0, x1]}}
    {"union": [R, ...]}, {"intersection": [R, ...]}, {"difference": [R, S, ...]}  (R minus the others)
Paths (JSON):
    [x, y]                                     a point
    [[x, y], ...]                              a polyline
    {"points": [[x, y], ...], "smooth": n, "units": "m"}   n rounds of Chaikin corner cutting (ends kept)
"""
import numpy as np


class GeometryError(ValueError):
    pass


def _scale(desc, units):
    u = desc.get("units", units) if isinstance(desc, dict) else units
    if u not in ("cells", "m"):
        raise GeometryError(f"units: 'cells' or 'm', not {u!r}")
    return u


def _num(x, what):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise GeometryError(f"{what}: a number, not {x!r}") from None
    if not np.isfinite(v):
        raise GeometryError(f"{what}: not finite")
    return v


def region(desc, H, W, dx, units="cells"):
    """A bool mask [H, W] from a region description."""
    if isinstance(desc, str):
        if desc == "all":
            return np.ones((H, W), bool)
        raise GeometryError(f"a region is 'all' or a JSON object, not {desc!r}")
    if not isinstance(desc, dict):
        raise GeometryError(f"a region is 'all' or a JSON object, not {type(desc).__name__}")
    u = _scale(desc, units)
    k = 1.0 / dx if u == "m" else 1.0
    keys = [key for key in desc if key != "units"]
    if len(keys) != 1:
        raise GeometryError(f"a region has one shape key (rect, disc, polygon, band, union, intersection, "
                            f"difference), got {keys}")
    kind, v = keys[0], desc[keys[0]]
    X = np.arange(W)[None, :] + 0.5
    Y = np.arange(H)[:, None] + 0.5
    if kind == "rect":
        if len(v) != 4:
            raise GeometryError("rect: [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (_num(a, "rect") * k for a in v)
        return (X >= min(x0, x1)) & (X < max(x0, x1)) & (Y >= min(y0, y1)) & (Y < max(y0, y1))
    if kind == "disc":
        if len(v) != 3:
            raise GeometryError("disc: [cx, cy, r]")
        cx, cy, r = (_num(a, "disc") * k for a in v)
        return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    if kind == "polygon":
        pts = np.array([[_num(a, "polygon") * k for a in p] for p in v])
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise GeometryError("polygon: at least three [x, y] points")
        Xb, Yb = np.broadcast_to(X, (H, W)), np.broadcast_to(Y, (H, W))
        inside = np.zeros((H, W), bool)
        n = len(pts)
        for i in range(n):
            (xa, ya), (xb, yb) = pts[i], pts[(i + 1) % n]
            if ya == yb:
                continue
            cross = (Yb >= min(ya, yb)) & (Yb < max(ya, yb))
            xi = xa + (Yb - ya) * (xb - xa) / (yb - ya)
            inside ^= cross & (Xb < xi)
        return inside
    if kind == "band":
        if not isinstance(v, dict) or len(v) != 1 or next(iter(v)) not in ("rows", "cols"):
            raise GeometryError('band: {"rows": [y0, y1]} or {"cols": [x0, x1]}')
        axis, (a, b) = next(iter(v.items()))
        a, b = sorted((_num(a, "band") * k, _num(b, "band") * k))
        C = Y if axis == "rows" else X
        return np.broadcast_to((C >= a) & (C < b), (H, W)).copy()
    if kind in ("union", "intersection", "difference"):
        if not isinstance(v, list) or not v:
            raise GeometryError(f"{kind}: a list of regions")
        parts = [region(p, H, W, dx, u) for p in v]
        if kind == "union":
            return np.logical_or.reduce(parts)
        if kind == "intersection":
            return np.logical_and.reduce(parts)
        out = parts[0].copy()
        for p in parts[1:]:
            out &= ~p
        return out
    raise GeometryError(f"unknown region shape {kind!r}")


def path(desc, dx, units="cells"):
    """Points [n, 2] (x, y) in cell coordinates from a path description (a point is a path of one point)."""
    smooth = 0
    if isinstance(desc, dict):
        u = _scale(desc, units)
        smooth = int(desc.get("smooth", 0))
        if smooth < 0 or smooth > 8:
            raise GeometryError("smooth: 0 to 8 rounds")
        pts = desc.get("points")
        if pts is None:
            raise GeometryError('a path object needs "points"')
    else:
        u, pts = units, desc
    k = 1.0 / dx if u == "m" else 1.0
    try:
        a = np.array(pts, dtype=float)
    except (TypeError, ValueError):
        raise GeometryError(f"a path is [x, y] or [[x, y], ...], not {pts!r}") from None
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != 2 or len(a) == 0 or not np.all(np.isfinite(a)):
        raise GeometryError(f"a path is [x, y] or [[x, y], ...], not {pts!r}")
    a = a * k
    for _ in range(smooth):
        if len(a) < 3:
            break
        p, q = a[:-1], a[1:]
        mid = np.empty((2 * len(p), 2))
        mid[0::2] = 0.75 * p + 0.25 * q
        mid[1::2] = 0.25 * p + 0.75 * q
        a = np.vstack([a[:1], mid[1:-1], a[-1:]])
    return a


def length(points):
    return float(np.sum(np.hypot(*np.diff(points, axis=0).T))) if len(points) > 1 else 0.0


def along(points, s):
    """Points at arc lengths s (array, in cells) along a polyline."""
    s = np.asarray(s, float)
    if len(points) == 1:
        return np.repeat(points, s.size, 0).reshape(s.shape + (2,))
    seg = np.hypot(*np.diff(points, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] == 0:
        return np.repeat(points[:1], s.size, 0).reshape(s.shape + (2,))
    x = np.interp(s, cum, points[:, 0])
    y = np.interp(s, cum, points[:, 1])
    return np.stack([x, y], -1)


def footprint(p, q, radius, H, W):
    """Where a spout of `radius` (cells) moving from p to q (cell coordinates) lays its matter: weights [H, W] on the
    panel and the fraction that falls off it, together summing to 1. Every cell whose centre is within the radius
    of the segment gets the same weight (a flat-topped footprint: a profile tapering to zero, such as a swept disc's
    chord, puts vanishingly thin films at its edge, which today's world cannot step: docs/PORTS.md §8). If no cell
    centre is within the radius, the nearest cell to the segment's midpoint takes it all."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    r = float(radius)
    lo = np.floor(np.minimum(p, q) - r - 1).astype(int)
    hi = np.ceil(np.maximum(p, q) + r + 1).astype(int)
    cols = np.arange(lo[0], hi[0] + 1)
    rows = np.arange(lo[1], hi[1] + 1)
    Xc, Yc = np.meshgrid(cols + 0.5, rows + 0.5)
    d2 = _seg_dist2(Xc, Yc, p, q)
    w = (d2 <= r * r).astype(float)
    if not np.any(w > 0):
        m = 0.5 * (p + q)
        j, i = int(np.floor(m[0])), int(np.floor(m[1]))
        out = np.zeros((H, W))
        if 0 <= i < H and 0 <= j < W:
            out[i, j] = 1.0
            return out, 0.0
        return out, 1.0
    w = w / w.sum()
    out = np.zeros((H, W))
    ri = (rows >= 0) & (rows < H)
    ci = (cols >= 0) & (cols < W)
    if not (ri.any() and ci.any()):
        return out, 1.0
    sub = w[np.ix_(ri, ci)]
    out[rows[ri][0]:rows[ri][-1] + 1, cols[ci][0]:cols[ci][-1] + 1] = sub
    return out, max(0.0, 1.0 - float(sub.sum()))


def _seg_dist2(X, Y, p, q):
    d = q - p
    L2 = float(d @ d)
    if L2 == 0.0:
        return (X - p[0]) ** 2 + (Y - p[1]) ** 2
    t = np.clip(((X - p[0]) * d[0] + (Y - p[1]) * d[1]) / L2, 0.0, 1.0)
    return (X - p[0] - t * d[0]) ** 2 + (Y - p[1] - t * d[1]) ** 2
