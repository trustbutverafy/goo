"""The motif: a still life set up on a table and lit from a window, rendered by a small sphere tracer.

This is the painter's model, not the painting. It gives the painter what a painter has in front of them:
colour (as reflectance-like linear values), which object is where, and where the highlights are.
Units are centimetres; the camera looks at a walnut table against a dark wall.
"""
import numpy as np
from scipy import ndimage

IDS = dict(wall=0, table=1, front=2, jug=3, lemon=4, apple=5, pear=6, knife_blade=7, knife_handle=8)


def _len(*a): return np.sqrt(sum(x * x for x in a))
def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1); return b * (1 - h) + a * h - k * h * (1 - h)
def ellipsoid(x, y, z, rx, ry, rz):
    k0 = _len(x / rx, y / ry, z / rz); k1 = _len(x / rx ** 2, y / ry ** 2, z / rz ** 2)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-9)
def box(x, y, z, bx, by, bz):
    qx, qy, qz = np.abs(x) - bx, np.abs(y) - by, np.abs(z) - bz
    return _len(np.maximum(qx, 0), np.maximum(qy, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0)
def rot_y(x, z, a): c, s = np.cos(a), np.sin(a); return c * x - s * z, s * x + c * z


def noise3(x, y, z, seed=0, scale=1.0):
    """Cheap smooth value noise from a few sines (deterministic, no tables)."""
    rng = np.random.default_rng(seed); v = 0
    for o in range(4):
        k = rng.normal(0, 1, 3) * scale * 2 ** o; ph = rng.uniform(0, 6.28)
        v = v + np.sin(k[0] * x + k[1] * y + k[2] * z + ph) / 2 ** o
    return v / 1.9


def scene(x, y, z):
    """Signed distances of every object; returns the stack (n_obj, ...) in IDS order (wall..knife_handle)."""
    wall = 25.0 - z
    table = box(x, y + 1.5, z - 5, 60, 1.5, 20)              # top at y = 0, front edge at z = -15
    front = table + 1e9                                       # front face is the same solid; told apart by normal
    # the jug: an ellipsoid body, a neck, a lip, a handle
    jx, jz = x + 8.5, z - 7.0
    body = ellipsoid(jx, y - 6.6, jz, 6.0, 6.8, 6.0)
    neck = np.maximum(_len(jx, jz) - 3.1, np.abs(y - 14.5) - 2.6)
    lip = _len(_len(jx, jz) - 3.35, y - 17.0) - 0.45
    hx, hz = rot_y(jx, jz, 0.0)
    handle = _len(_len(hx - 5.6, y - 10.0) - 3.0, hz) - 0.62
    handle = np.maximum(handle, -(hx - 5.0))
    jug = smin(smin(smin(body, neck, 1.2), lip, 0.4), handle, 0.5)
    jug = np.maximum(jug, -(y - 0.02))
    # two lemons
    def lemon(cx, cz, a, s):
        lx, lz = rot_y(x - cx, z - cz, a); ly = y - 2.55 * s
        b = ellipsoid(lx, ly, lz, 3.6 * s, 2.6 * s, 2.6 * s)
        nub = smin(_len(lx - 3.7 * s, ly, lz) - 0.55 * s, _len(lx + 3.7 * s, ly, lz) - 0.5 * s, 0.1)
        return smin(b, nub, 0.7 * s)
    lemons = np.minimum(lemon(7.0, -3.5, 0.35, 1.0), lemon(12.8, 2.0, -0.9, 0.95))
    # the apple: a sphere with a dimple and a stem
    ax_, ay, az = x - 2.5, y - 4.0, z - 6.0
    apple = _len(ax_, ay * 1.06, az) - 4.2
    apple = -smin(-apple, _len(ax_, ay - 4.6, az) - 1.4, 0.9)
    stem = np.maximum(_len(ax_ - 0.15 * (ay - 3.0), az), 0) - 0.18
    stem = np.maximum(stem, np.abs(ay - 4.3) - 1.1)
    apple = np.minimum(apple, stem)
    # the pear, front left, leaning
    px, pz = x + 18.5, z + 4.5; py = y
    c, s = np.cos(0.18), np.sin(0.18); qx, qy = c * px + s * (py - 4.5), -s * px + c * (py - 4.5) + 4.5
    pb = _len(qx, qy - 3.7, pz) - 3.9; ps = _len(qx + 0.4, qy - 8.3, pz) - 2.2
    pear = smin(pb, ps, 2.2)
    pstem = np.maximum(_len(qx + 0.3 - 0.25 * (qy - 10.2), pz) - 0.16, np.abs(qy - 11.3) - 1.0)
    pear = np.minimum(pear, pstem)
    # the knife, over the table's front edge
    kx, kz = rot_y(x - 9.5, z + 11.5, -1.22)
    blade = box(kx - 4.0, y - 0.18, kz, 5.0, 0.07, 0.75 - 0.25 * np.clip((kx - 4.0) / 5.0, 0, 1))
    khandle = box(kx + 3.5, y - 0.5, kz, 3.0, 0.45, 0.6) - 0.15
    return np.stack([wall, table, front, jug, lemons, apple, pear, blade, khandle])


def sdf(x, y, z): return scene(x, y, z).min(0)


def render(W=640, H=480, ss=2, seed=0):
    import os; ss = int(os.environ.get("SS", ss))
    Wr, Hr = W * ss, H * ss
    cam = np.array([0.0, 15.0, -74.0]); look = np.array([-1.5, 6.0, 0.0]); up = np.array([0, 1.0, 0])
    f = look - cam; f /= np.linalg.norm(f); r = np.cross(up, f); r /= np.linalg.norm(r); u = np.cross(f, r)
    fov = np.radians(33); tw = np.tan(fov / 2)
    sx = (np.arange(Wr) + 0.5) / Wr * 2 - 1; sy = 1 - (np.arange(Hr) + 0.5) / Hr * 2
    SX, SY = np.meshgrid(sx * tw, sy * tw * Hr / Wr)
    D = f[None, None] + SX[..., None] * r + SY[..., None] * u; D /= np.linalg.norm(D, axis=-1, keepdims=True)
    D = D.reshape(-1, 3); O = np.broadcast_to(cam, D.shape)
    t = np.zeros(len(D)); hit = np.zeros(len(D), bool)
    for _ in range(260):
        P = O + t[:, None] * D; d = sdf(P[:, 0], P[:, 1], P[:, 2])
        hit |= d < 2e-3; t = np.where(hit, t, t + 0.8 * d)
        if hit.all(): break
    P = O + t[:, None] * D
    S = scene(P[:, 0], P[:, 1], P[:, 2]); oid = S.argmin(0)
    e = 2e-3
    N = np.stack([sdf(P[:, 0] + e, P[:, 1], P[:, 2]) - sdf(P[:, 0] - e, P[:, 1], P[:, 2]),
                  sdf(P[:, 0], P[:, 1] + e, P[:, 2]) - sdf(P[:, 0], P[:, 1] - e, P[:, 2]),
                  sdf(P[:, 0], P[:, 1], P[:, 2] + e) - sdf(P[:, 0], P[:, 1], P[:, 2] - e)], 1)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    oid = np.where((oid == IDS["table"]) & (N[:, 2] < -0.7), IDS["front"], oid)
    # light: a window up and to the left, a little in front; soft shadows by penumbra marching
    Ld = np.array([-0.62, 0.66, -0.42]); Ld /= np.linalg.norm(Ld)
    sh = np.ones(len(P)); ts = np.full(len(P), 0.05)
    Ps = P + N * 4e-3
    for _ in range(200):
        q = Ps + ts[:, None] * Ld; d = sdf(q[:, 0], q[:, 1], q[:, 2])
        sh = np.minimum(sh, np.clip(9.0 * d / ts, 0, 1)); ts = ts + np.clip(d, 0.015, 1.0)
        if (ts > 60).all(): break
    sh = sh * sh * (3 - 2 * sh)
    # ambient occlusion
    ao = np.ones(len(P))
    for k in range(1, 6):
        hh = 0.35 * k; q = P + N * hh; ao -= (hh - sdf(q[:, 0], q[:, 1], q[:, 2])) * (0.55 / 2 ** k)
    ao = np.clip(ao, 0, 1)
    # materials
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    alb = np.zeros((len(P), 3)); spec = np.zeros(len(P)); shin = np.full(len(P), 30.0)
    def put(i, a, s_=0.0, n_=30.0):
        m = oid == i; alb[m] = a[m] if np.ndim(a) == 2 else a; spec[m] = s_; shin[m] = n_
    nw = noise3(x * 0.25, y * 0.25, z * 0.25, 1)
    put(IDS["wall"], np.stack([0.24 + 0.03 * nw, 0.215 + 0.025 * nw, 0.165 + 0.02 * nw], 1))
    grain = np.sin(x * 1.3 + 2.5 * noise3(x * 0.2, 0, z * 0.6, 2)) * 0.5 + 0.5
    wood = np.stack([0.30 + 0.07 * grain, 0.17 + 0.04 * grain, 0.085 + 0.02 * grain], 1)
    put(IDS["table"], wood, 0.06, 18); put(IDS["front"], wood * 0.5, 0.04, 18)
    jx, jz = x + 8.5, z - 7.0
    band = np.exp(-((y - 8.6) / 0.75) ** 2) + 0.6 * np.exp(-((y - 6.3) / 0.35) ** 2)
    glaze = np.array([0.80, 0.80, 0.76]) * (1 - np.clip(band, 0, 1))[:, None] + np.array([0.10, 0.17, 0.48]) * np.clip(band, 0, 1)[:, None]
    put(IDS["jug"], glaze, 0.55, 90)
    nl = noise3(x * 3, y * 3, z * 3, 3) * 0.04
    put(IDS["lemon"], np.stack([0.80 + nl, 0.62 + nl, 0.06 + 0 * nl], 1), 0.18, 25)
    na = noise3(x * 1.4, y * 2.5, z * 1.4, 4)
    streak = np.clip(0.5 + 0.8 * na, 0, 1)
    blush = np.clip((y - 3.5) / 3.0, 0, 1) * (1 - streak)
    put(IDS["apple"], np.stack([0.55 + 0.1 * streak, 0.07 + 0.25 * blush, 0.04 + 0.03 * blush], 1), 0.35, 60)
    npr = noise3(x * 0.9, y * 0.9, z * 0.9, 5); blush_p = np.clip(0.5 + npr, 0, 1)
    spk = (noise3(x * 9, y * 9, z * 9, 6) > 0.62).astype(float) * 0.5
    put(IDS["pear"], np.stack([0.50 + 0.10 * blush_p - 0.15 * spk, 0.56 - 0.10 * blush_p - 0.2 * spk, 0.17 - 0.05 * blush_p - 0.05 * spk], 1), 0.12, 20)
    put(IDS["knife_blade"], np.array([0.62, 0.62, 0.64]), 0.9, 140)
    put(IDS["knife_handle"], np.array([0.12, 0.07, 0.04]), 0.15, 30)
    # shading: key + sky fill + warm bounce from the table, then a specular glint
    ndl = np.clip(N @ Ld, 0, 1)
    window = np.clip(1.55 - 0.016 * _len(x + 38, y - 34, z + 8), 0.35, 1.25)            # falloff away from the window
    key = ndl * sh * window * 1.0
    sky = (0.5 + 0.5 * N[:, 1]) * 0.22 * ao
    bounce = (np.clip(-N[:, 1], 0, 1) * 0.18 * ao)[:, None] * np.array([[1.0, 0.65, 0.4]])
    col = alb * (key[:, None] + sky[:, None] + bounce)
    Hv = Ld - D; Hv /= np.linalg.norm(Hv, axis=1, keepdims=True)
    glint = spec * np.clip((N * Hv).sum(1), 0, 1) ** shin * sh * 1.1
    col += glint[:, None]
    hl = glint.reshape(Hr, Wr)
    col = col.reshape(Hr, Wr, 3); oid = oid.reshape(Hr, Wr)
    # down to the canvas grid
    col = col.reshape(H, ss, W, ss, 3).mean((1, 3)); hl = hl.reshape(H, ss, W, ss).mean((1, 3))
    oid = oid[::ss, ::ss]
    col = 0.9 * col / (1 + 0.15 * col)                                                   # a gentle shoulder
    col = np.clip(col, 0.015, 0.9)
    return dict(rgb=col, ids=oid, highlight=hl, normal=N.reshape(Hr, Wr, 3)[::ss, ::ss])


if __name__ == "__main__":
    import time, matplotlib.pyplot as plt, os
    t = time.time(); R = render()
    os.makedirs("out", exist_ok=True)
    np.savez_compressed("out/still-life-reference.npz", **R)
    plt.imsave("out/still-life-reference.png", np.clip(R["rgb"], 0, 1) ** (1 / 2.2))
    print(f"{time.time() - t:.0f}s", R["rgb"].min(), R["rgb"].max())
