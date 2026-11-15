"""goo.classic.engine: the World. One paint layer for every medium, on a ground, over the strata it has dried into.

A depth-averaged layer of suspension on a ground. Every cell holds volumes of each species: two carriers
(water, which evaporates and soaks into the ground; oil, which stays and cures) and any number of pigments
(solids). Thickness h = sum of volumes; solid fraction phi = solids / h. Everything else is a consequence:

  viscosity     eta = eta0 (1 - phi/phi_m)^-2 (1 + cure_stiff * kappa)        Krieger-Dougherty, curing
  yield stress  tau_y = tau0 ((phi - phi_y)+ / (phi_m - phi))^1.5 (1 + cure_yield * kappa^2)
  leveling      Bingham thin-film flux down the gradient of P = G (h + ground): plugs where stress < yield
  tools         brushes and styluses impose a velocity where they touch; it spreads sideways over a range
                set by what the paint sits on (short on a ground); paint is carried by it with a 2nd-order,
                limited (MUSCL / van Leer) scheme, so streaks survive
  colour        pigments diffuse among the solids at D = D0 eta0 / eta (Stokes-Einstein): clouds when thin,
                streaks when thick
  drying        water evaporates (fastest at a wet edge) and soaks into the ground; oil cures (kappa -> 1)

Robustness: every flux is in conservative form; each species' outflow from a cell is limited to what the
cell holds (no clamping, no creation); substeps are chosen from the stability limits of every process each
step; the ledger accounts for every unit loaded onto a brush, and exact-zero switches turn each law off.
"""
from __future__ import annotations
from collections.abc import Sequence
import copy
import numpy as np
from scipy import ndimage

CARRIERS = ["water", "oil"]
# Kubelka-Munk per unit volume: K and S for (R, G, B, N), N = near-infrared at ~1.5 um (1550 nm, where the
# simulated lidar and reflectography look)
PIGMENTS = {
    "carbon":      dict(K=[3.2, 3.2, 3.2, 3.0], S=[0.10, 0.10, 0.10, 0.10]),
    "white":       dict(K=[0.02, 0.02, 0.03, 0.01], S=[3.2, 3.2, 3.0, 0.45]),   # scattering falls with wavelength
    "ultramarine": dict(K=[2.3, 1.7, 0.20, 0.04], S=[0.05, 0.06, 0.30, 0.05]),
    "ochre":       dict(K=[0.25, 0.55, 1.70, 0.05], S=[0.45, 0.40, 0.25, 0.10]),
    "sienna":      dict(K=[0.30, 1.25, 2.40, 0.08], S=[0.30, 0.20, 0.10, 0.12]),
    "rose":        dict(K=[0.10, 2.30, 0.85, 0.02], S=[0.03, 0.02, 0.04, 0.02]),
    "green":       dict(K=[1.60, 0.30, 1.90, 0.10], S=[0.06, 0.08, 0.04, 0.05]),
    "yellow":      dict(K=[0.04, 0.12, 2.80, 0.02], S=[0.90, 0.85, 0.35, 0.30]),    # cadmium-like, opaque
    "red":         dict(K=[0.06, 2.60, 2.20, 0.03], S=[0.90, 0.30, 0.25, 0.30]),    # vermilion-like, opaque
    "ivory_black": dict(K=[4.6, 4.8, 5.0, 4.5], S=[0.016, 0.016, 0.016, 0.02]),    # bone/ivory black, a deeper black than carbon (K/S ~ 300); plausible, not measured
    "dioxazine":   dict(K=[1.1, 3.4, 0.55, 0.03], S=[0.04, 0.03, 0.06, 0.03]),     # dioxazine violet (PV23), a transparent violet for glazes; plausible, not measured
    "acrylic":     dict(K=[0.004, 0.005, 0.007, 0.006], S=[0.02, 0.02, 0.02, 0.02]),  # clear polymer binder of acrylic paint (a solid once dry); plausible, not measured
}
GROUND_RGBN = np.array([0.93, 0.92, 0.88, 0.90])   # gesso, visible and near-infrared

# The finish rule (Canvas.finish): which films are glossy
SKIN = 0.3        # thickness (volume units, 0.25 mm each) at which a film is thick enough to be fully glossy
FAT = 0.3         # oil fraction (oil / film volume) at which a film is oil-rich enough to be fully glossy
HAS_PAINT = 1e-9  # a layer "has paint" at a cell when its volume there exceeds this (the wet threshold of the laws)

# The sheen rule (Canvas.sheen): how wet a surface looks
DRY_WATER = 1e-6  # water at or below this is dry (fix() books it as evaporated)
LEAN = 0.05       # oil at or below this fraction of the film is lean: it is held in the paint, not free (fix())
CURE_TOUCH = 0.8  # cure at which oil is dry to the touch (fix()'s default cure_min)
GLISTEN = 0.02    # standing liquid (volume units, 5 um) above the pigment bed at which a surface fully glistens
# How a painter reads sheen() (named once, here; goo.classic.studio and goo.classic.easel use these):
SHEEN_GLISTENING = 0.5   # sheen above this: liquid stands above the pigment bed (it glistens; a drop-in spreads freely)
SHEEN_DAMP = 0.02        # sheen above this, up to SHEEN_GLISTENING: damp, satin (the moment for a backrun); at or below: dry-looking


def _open_faces(m):
    """Open faces of a masked (H, W) region: (x-faces (H, W-1), y-faces (H-1, W)), False where a face touches a
    masked cell."""
    return ~(m[:, :-1] | m[:, 1:]), ~(m[:-1, :] | m[1:, :])


def film_gloss(film, oil_index):
    """Gloss 0..1 of one film (species, H, W): clip(h / SKIN, 0, 1) * clip(oil / h / FAT, 0, 1); 0 where h = 0."""
    h = film.sum(0)
    oilfrac = film[oil_index] / np.maximum(h, 1e-9)
    return np.clip(h / SKIN, 0, 1) * np.clip(oilfrac / FAT, 0, 1)


def limit(Fx, Fy, q):
    """Scale each species' face fluxes so no cell sends out more than it holds. Conservative, positive."""
    out = np.zeros_like(q)
    out[..., :, :-1] += np.maximum(Fx, 0); out[..., :, 1:] += np.maximum(-Fx, 0)
    out[..., :-1, :] += np.maximum(Fy, 0); out[..., 1:, :] += np.maximum(-Fy, 0)
    r = np.where(out > q, q / np.maximum(out, 1e-300), 1.0)
    return np.where(Fx > 0, Fx * r[..., :, :-1], Fx * r[..., :, 1:]), np.where(Fy > 0, Fy * r[..., :-1, :], Fy * r[..., 1:, :])


def vanleer(a, b):
    """van Leer limited slope from the two one-sided differences."""
    ab = a * b
    return np.where(ab > 0, 2 * ab / (a + b + np.where(a + b == 0, 1e-300, 0)), 0.0)


def muscl_flux(q, u, axis, dt):
    """Face fluxes of q (species-leading array) carried by face velocity u along axis, MUSCL + van Leer."""
    d = np.diff(q, axis=axis)
    # slopes at interior cells
    sl = np.zeros_like(q)
    if axis == -1:
        sl[..., :, 1:-1] = vanleer(d[..., :, :-1], d[..., :, 1:])
        qL = q[..., :, :-1] + 0.5 * (1 - np.clip(u, 0, None) * dt) * sl[..., :, :-1]
        qR = q[..., :, 1:] - 0.5 * (1 - np.clip(-u, 0, None) * dt) * sl[..., :, 1:]
    else:
        sl[..., 1:-1, :] = vanleer(d[..., :-1, :], d[..., 1:, :])
        qL = q[..., :-1, :] + 0.5 * (1 - np.clip(u, 0, None) * dt) * sl[..., :-1, :]
        qR = q[..., 1:, :] - 0.5 * (1 - np.clip(-u, 0, None) * dt) * sl[..., 1:, :]
    return np.where(u > 0, qL, qR) * u * dt


class Strata(Sequence):
    """The dried layers, bottom first, stored compact: each stratum is the (species, h, w) block of its bounding
    box, its offset, and its per-species totals (summed as the dense layer would be, so the ledger is unchanged).
    Read as a sequence of dense (species, H, W) arrays: `strata[i]` materialises one, read-only (I5: a stratum
    is immutable); `block(i)` gives the compact form without materialising anything. Only `Canvas.fix` appends."""

    def __init__(self, shape):
        self.shape = tuple(shape)                       # (species, H, W) of a dense stratum
        self._items = []                                # (y0, y1, x0, x1, block, totals)

    def __len__(self): return len(self._items)

    def __getitem__(self, i):
        if isinstance(i, slice): return [self[j] for j in range(*i.indices(len(self)))]
        y0, y1, x0, x1, blk, _ = self._items[i]
        d = np.zeros(self.shape); d[:, y0:y1, x0:x1] = blk
        d.flags.writeable = False
        return d

    def block(self, i):
        """(y0, y1, x0, x1, block): stratum i is `block` at [:, y0:y1, x0:x1] and zero elsewhere (read-only)."""
        return self._items[i][:5]

    def totals(self, i):
        """Volume of each species in stratum i (bit-identical to the dense stratum's .sum(axis=(1, 2)))."""
        return self._items[i][5]

    @property
    def nbytes(self): return sum(it[4].nbytes for it in self._items)

    def _append(self, y0, y1, x0, x1, blk, totals):
        blk = np.ascontiguousarray(blk); blk.flags.writeable = False
        totals = np.array(totals, dtype=float); totals.flags.writeable = False
        self._items.append((y0, y1, x0, x1, blk, totals))

    def __repr__(self):
        return f"Strata({len(self)} layers of {self.shape}, compact: {self.nbytes / 1e6:.1f} MB)"


def _dense_sum(shape, y0, y1, x0, x1, a):
    """Sum of an (H, W) array that is `a` in [y0:y1, x0:x1] and zero elsewhere, bit-identical to summing the
    dense array: the zeros are kept, so numpy's pairwise summation groups the terms exactly as it would."""
    buf = np.zeros(shape); buf[y0:y1, x0:x1] = a
    return buf.sum()


def _dense_species_sums(shape, y0, y1, x0, x1, blk):
    """Per-species totals of a (species, H, W) layer that is `blk` in the window and zero elsewhere,
    bit-identical to the dense layer's .sum(axis=(1, 2)), using one (1, H, W) buffer at a time."""
    buf = np.zeros((1,) + tuple(shape)); out = np.zeros(len(blk))
    for s in range(len(blk)):
        if not blk[s].any(): continue                    # all zeros: the dense sum is +0.0 as well
        buf[0, y0:y1, x0:x1] = blk[s]; out[s] = buf.sum(axis=(1, 2))[0]
    return out


def _bbox(mask):
    """(y0, y1, x0, x1) of the True cells of a 2-D mask, or None."""
    rows = np.flatnonzero(mask.any(1))
    if rows.size == 0: return None
    cols = np.flatnonzero(mask.any(0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _union(a, b):
    if a is None: return b
    if b is None: return a
    return min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])


class Canvas:
    # windowed slow laws: margin (cells) around the live paint; live cells must stay at least _NEAR cells from
    # any window side that is not the panel's own (the leveling faces, the 3-cell wet-edge filter, np.gradient)
    _MARGIN = 16
    _NEAR = 2
    _mask = None          # frisket: (H, W) bool where the panel is under tape, or None (a class default, so canvases
                          # pickled before the frisket existed load unmasked)

    def __init__(self, W, H, pigments, seed=1, ground="gesso", laws=None, optics="classic"):
        self.W, self.H = W, H
        self.species = CARRIERS + list(pigments)
        self.nc = len(CARRIERS)
        self._q = np.zeros((len(self.species), H, W))
        self._kappa = np.zeros((H, W))                      # oil cure, 0..1
        self._store = np.zeros((H, W))                      # water soaked into the ground
        rng = np.random.default_rng(seed)
        n = lambda s: ndimage.gaussian_filter(rng.standard_normal((H, W)), s)
        tex = 0.5 * n(1.0) / n(1.0).std() + 0.6 * n(4.0) / n(4.0).std() + 0.25 * n((0.8, 12)) / n((0.8, 12)).std()
        self.g = 0.03 * tex                                 # ground relief (gesso brush marks and tooth)
        self.strata = Strata(self._q.shape)                 # dried / fixed layers, bottom first (compact)
        self.zb = self.g.copy()                             # what the wet layer sits on: ground + strata
        self.L = dict(G=1.0, eta0=0.12, phi_m=0.64, phi_y=0.2, tau0=0.05, tau_cure=1.0, D0=0.15, evap=0.0006, edge_evap=0.006,
                      absorb=0.004 if ground == "gesso" else 0.04, store_cap=0.08, store_evap=0.002,
                      cure_rate=0.0, cure_stiff=400.0, cure_yield=300.0, drag_range=1.4,
                      flow=1.0, drag=1.0, diffusion=1.0, evaporation=1.0, absorption=1.0, curing=1.0, max_sub=600, flow_sub=48, h_pin=0.08, lazy=1, diff_sub=1e9,
                      windowed=1)
        if laws: self.L.update(laws)
        self._U = np.zeros((2, H, W))                       # tool velocity imposed this step (cells / step)
        self._V = np.zeros((2, H, W))                       # plough: paint squeezed out from under a tool
        self.led = dict(loaded=np.zeros(len(self.species)), evaporated=0.0, store_evaporated=0.0)
        self.brushes = []
        self.t = 0; self.substeps = 0; self.capped = 0; self.pending = 0
        # windowed bookkeeping: where anything can be live (paint, water in the ground, cure), and where tools
        # wrote U, V since they were last cleared; both supersets, kept by add, the tools, step, flush
        self._box = None; self._tw = None
        self._ext_live = False; self._ext_tool = False      # raw state handed out: rescan the panel once
        # >>> optics hook (goo/classic/optics.py): the canvas holds an optics model; reflectance() and finish() ask it
        from goo.classic.optics import make_optics
        self.ground = ground
        self.optics = make_optics(optics)
        # <<< optics hook

    # ---- the raw state. Reading one of these from outside the engine hands out a writable array, so the next
    #      windowed step/flush rescans the whole panel once (anything may have been written into it).
    @property
    def q(self): self._ext_live = True; return self._q
    @q.setter
    def q(self, v): self._ext_live = True; self._q = v
    @property
    def kappa(self): self._ext_live = True; return self._kappa
    @kappa.setter
    def kappa(self, v): self._ext_live = True; self._kappa = v
    @property
    def store(self): self._ext_live = True; return self._store
    @store.setter
    def store(self, v): self._ext_live = True; self._store = v
    @property
    def U(self): self._ext_tool = True; return self._U
    @U.setter
    def U(self, v): self._ext_tool = True; self._U = v
    @property
    def V(self): self._ext_tool = True; return self._V
    @V.setter
    def V(self, v): self._ext_tool = True; self._V = v

    def _mark(self, y0, y1, x0, x1):
        """Something may have become live (paint, ground water, cure) in [y0:y1, x0:x1]."""
        self._box = _union(self._box, (y0, y1, x0, x1))

    def _mark_tool(self, y0, y1, x0, x1):
        """A tool wrote U or V in [y0:y1, x0:x1]."""
        self._tw = _union(self._tw, (y0, y1, x0, x1))

    # ---- putting paint down by hand (setups, tests): booked as loaded, like anything a brush brings
    def add(self, name, field):
        """Add volume `field` (H, W, or scalar) of species `name` to the wet layer, and book it."""
        f = np.broadcast_to(np.asarray(field, float), (self.H, self.W))
        if (f < 0).any(): raise ValueError("add() takes volume, not a debit")
        i = self.idx(name); self._q[i] += f; self.led["loaded"][i] += f.sum()
        b = _bbox(f != 0)
        if b is not None: self._mark(*b)

    # ---- derived fields
    def idx(self, name): return self.species.index(name)
    @property
    def h(self): return self._q.sum(0)
    @staticmethod
    def _phi(q, h, nc):
        return np.where(h > 1e-12, q[nc:].sum(0) / np.maximum(h, 1e-12), 1.0)
    def phi(self): return self._phi(self._q, self.h, self.nc)
    def _eta(self, phi, kappa):
        L = self.L; return L["eta0"] * (1 - np.minimum(phi, 0.99 * L["phi_m"]) / L["phi_m"]) ** -2 * (1 + L["cure_stiff"] * kappa)
    def _tau(self, phi, kappa):
        L = self.L; pm = L["phi_m"]
        return (L["tau0"] * (np.maximum(phi - L["phi_y"], 0) / (pm - np.minimum(phi, 0.99 * pm))) ** 1.5 * (1 + L["cure_yield"] * kappa ** 2)
                + L["tau_cure"] * kappa ** 2)                # a skin forms on any cured oil, lean or fat
    def eta(self, phi): return self._eta(phi, self._kappa)
    def tau(self, phi): return self._tau(phi, self._kappa)
    def surface(self): return self.zb + self.h
    def _surface_window(self, y0, y1, x0, x1):
        """surface()[y0:y1, x0:x1], computed in the window only (same arithmetic, same summation order)."""
        if not self.L["windowed"]: return self.surface()[y0:y1, x0:x1]
        return self.zb[y0:y1, x0:x1] + np.ascontiguousarray(self._q[:, y0:y1, x0:x1]).sum(0)
    def porous(self): return np.exp(-(self.zb - self.g) / 0.02)   # a dried film seals the ground

    # ---- the tool velocity: imposed where tools touch, spread sideways by a screened Poisson solve
    def _drag_velocity(self):
        """The tool velocity where tools touched this tick: U spread sideways by a screened Poisson solve
        (range drag_range), plus the unspread plough V. Solved in a window around the touch, not the panel.
        Returns (u[2, h, w], (y0, y1, x0, x1)) or None."""
        if self.L["drag"] == 0: return None
        if self.L["windowed"] and not self._ext_tool:
            if self._tw is None: return None
            wy0, wy1, wx0, wx1 = self._tw                  # look only where tools wrote (Brush.touch marks it)
            act = (self._U[:, wy0:wy1, wx0:wx1] != 0).any(0) | (self._V[:, wy0:wy1, wx0:wx1] != 0).any(0)
        else:
            wy0 = wx0 = 0
            act = (self._U != 0).any(0) | (self._V != 0).any(0)
        self._ext_tool = False
        if not act.any(): self._tw = None; return None
        ys, xs = np.nonzero(act); ys = ys + wy0; xs = xs + wx0; m = 12   # the kernel has decayed by e^-8 at 12 cells
        y0, y1 = max(0, ys.min() - m), min(self.H, ys.max() + 1 + m); x0, x1 = max(0, xs.min() - m), min(self.W, xs.max() + 1 + m)
        pad = 12; Hh, Ww = (y1 - y0) + 2 * pad, (x1 - x0) + 2 * pad
        ky = np.fft.fftfreq(Hh)[:, None] * 2 * np.pi; kx = np.fft.rfftfreq(Ww)[None, :] * 2 * np.pi
        filt = 1 / (1 + self.L["drag_range"] ** 2 * (kx ** 2 + ky ** 2))
        out = []
        for c in range(2):
            a = np.pad(self._U[c, y0:y1, x0:x1], pad)
            out.append(np.fft.irfft2(np.fft.rfft2(a) * filt, s=(Hh, Ww))[pad:-pad, pad:-pad])
        u = (np.stack(out) + self._V[:, y0:y1, x0:x1]) * self.L["drag"]
        return u * (self._q[:, y0:y1, x0:x1].sum(0) > 1e-9), (y0, y1, x0, x1)   # only paint is carried

    # ---- time. Two clocks: tools act every tick, locally; the slow laws (leveling, colour, drying, curing)
    #      catch up every L["lazy"] ticks with the whole elapsed time, at their own stability limits.
    #      Conservative either way; with lazy = 1 every law runs every tick.
    def step(self):
        L = self.L
        d = self._drag_velocity()
        if d is not None:
            u, (Y0, Y1, X0, X1) = d
            umax = np.abs(u).max()
            if umax > 0:
                live = (np.abs(u) > 1e-3 * umax).any(0)
                ys, xs = np.nonzero(live)
                y0, y1 = max(0, ys.min() - 3), min(Y1 - Y0, ys.max() + 4); x0, x1 = max(0, xs.min() - 3), min(X1 - X0, xs.max() + 4)
                nt = int(np.ceil(max(1.0, umax / 0.5)))
                if nt > L["max_sub"]: self.capped += 1; nt = L["max_sub"]
                ub = u[:, y0:y1, x0:x1]
                ux = 0.5 * (ub[0][:, :-1] + ub[0][:, 1:]); uy = 0.5 * (ub[1][:-1, :] + ub[1][1:, :])
                for _ in range(nt):
                    qb = self._q[:, Y0 + y0:Y0 + y1, X0 + x0:X0 + x1]
                    Fx, Fy = muscl_flux(qb, ux, -1, 1.0 / nt), muscl_flux(qb, uy, -2, 1.0 / nt)
                    if self._mask is not None:                # frisket: no face touching tape carries paint
                        ox, oy = _open_faces(self._mask[Y0 + y0:Y0 + y1, X0 + x0:X0 + x1])
                        Fx = np.where(ox, Fx, 0.0); Fy = np.where(oy, Fy, 0.0)
                    Fx, Fy = limit(Fx, Fy, qb)
                    qb[..., :, :-1] -= Fx; qb[..., :, 1:] += Fx; qb[..., :-1, :] -= Fy; qb[..., 1:, :] += Fy
                    np.maximum(qb, 0, out=qb)
                self.substeps += nt
                self._mark(Y0 + y0, Y0 + y1, X0 + x0, X0 + x1)   # paint moved only inside this box
            self._U[:, Y0:Y1, X0:X1] = 0; self._V[:, Y0:Y1, X0:X1] = 0
            self._tw = None                                   # the cleared window held every nonzero U, V
        self.t += 1; self.pending += 1
        if self.pending >= L["lazy"]: self.flush()

    def _nsub(self, q, kappa, zb, dt):
        """Substeps for dt ticks from every process's stability limit (the maxima are over the panel; in a window,
        over the window, which holds every live cell and at least one cell like all those outside it)."""
        L = self.L
        h = q.sum(0); phi = self._phi(q, h, self.nc); eta = self._eta(phi, kappa)
        # Bingham mobility from the yielded depth Y only: a paste below its yield stress doesn't set the step
        gy, gx = np.gradient(L["G"] * (h + zb)); ag = np.hypot(gx, gy) + 1e-300
        Y = np.clip(h - self._tau(phi, kappa) / ag, 0, None)
        mob = (Y * Y * (3 * h - Y) / (6 * eta)).max() * L["flow"]
        D = (L["D0"] * L["eta0"] / eta).max() * L["diffusion"]
        n_flow = min(4.4 * L["G"] * mob / 0.8, L["flow_sub"]) * dt  # beyond flow_sub the face cap keeps it stable
        n_diff = min(4 * D / 0.8, L["diff_sub"]) * dt              # likewise for colour diffusion (face cap)
        n = int(np.ceil(max(1.0, n_flow, n_diff)))
        if n > L["max_sub"] * dt: self.capped += 1; n = int(L["max_sub"] * dt)
        return n

    def flush(self):
        """Run the slow laws over the elapsed ticks not yet accounted for."""
        dt = self.pending
        if dt <= 0: return
        if not self.L["windowed"]:
            n = self._nsub(self._q, self._kappa, self.zb, dt)
            for _ in range(n): self._sub(self._q, self._kappa, self._store, self.zb, self.g, dt / n)
            self.substeps += n; self.pending = 0
            self._ext_live = True                            # the windowed books were not kept: rescan if switched
            return
        self._flush_windowed(dt)

    # ---- the slow laws in a window. Outside the cells that hold paint, water in the ground or cure ("live"),
    #      every law is exactly zero: no flux leaves or enters a dry cell from a dry neighbour, nothing evaporates,
    #      soaks in or cures. So the laws run in the live cells' bounding box plus a margin, on contiguous copies
    #      of the state (the same arithmetic, element for element, as on the panel). Paint spreads at most one
    #      cell per substep; before live cells come within _NEAR cells of a window side that is not the panel's,
    #      the window is moved. The only panel-wide reductions, the ledger's sums of what evaporated, are taken
    #      over a zero-filled panel-sized buffer, so they round exactly as on the panel.
    def _live_bbox(self):
        """Exact bounding box of the live cells, or None."""
        if self._ext_live: region = (0, self.H, 0, self.W); self._ext_live = False
        elif self._box is None: return None
        else: region = self._box
        y0, y1, x0, x1 = region
        live = (self._q[:, y0:y1, x0:x1].any(0) | (self._store[y0:y1, x0:x1] != 0) | (self._kappa[y0:y1, x0:x1] != 0))
        b = _bbox(live)
        self._box = None if b is None else (b[0] + y0, b[1] + y0, b[2] + x0, b[3] + x0)
        return self._box

    def _window_around(self, b):
        M = self._MARGIN
        if b is None: return 0, min(2, self.H), 0, min(2, self.W)   # nothing live: any two dry cells set the limits
        return max(0, b[0] - M), min(self.H, b[1] + M), max(0, b[2] - M), min(self.W, b[3] + M)

    def _room(self, win, b):
        """Substeps that can run in window `win` before live cells (bounding box b, panel coordinates) could come
        within _NEAR cells of a window side that is not the panel's own."""
        if b is None: return np.inf
        y0, y1, x0, x1 = win; d = []
        if y0 > 0: d.append(b[0] - y0)
        if y1 < self.H: d.append(y1 - b[1])
        if x0 > 0: d.append(b[2] - x0)
        if x1 < self.W: d.append(x1 - b[3])
        return np.inf if not d else min(d) - self._NEAR + 1

    def _take(self, win):
        y0, y1, x0, x1 = win
        c = lambda a: a[..., y0:y1, x0:x1].copy()           # C-contiguous copies, like the panel's own arrays
        return [c(self._q), c(self._kappa), c(self._store), c(self.zb), c(self.g)]

    def _put(self, win, st):
        y0, y1, x0, x1 = win
        self._q[:, y0:y1, x0:x1] = st[0]; self._kappa[y0:y1, x0:x1] = st[1]; self._store[y0:y1, x0:x1] = st[2]

    def _flush_windowed(self, dt):
        b = self._live_bbox()
        win = self._window_around(b)
        st = self._take(win)
        n = self._nsub(st[0], st[1], st[3], dt)
        k = 0
        while k < n and b is not None:
            room = self._room(win, b)
            if room < 1:                                     # paint is near the window's side: move the window
                self._put(win, st); win = self._window_around(b); st = self._take(win); continue
            run = int(min(room, n - k))
            for _ in range(run): self._sub(*st, dt / n, win=win)
            k += run
            y0, y1, x0, x1 = win
            bl = _bbox(st[0].any(0) | (st[2] != 0) | (st[1] != 0))
            b = None if bl is None else (bl[0] + y0, bl[1] + y0, bl[2] + x0, bl[3] + x0)
        # k < n with nothing live left: the remaining substeps are exactly no-ops (no paint, no water, no cure)
        self._put(win, st)
        self._box = b
        self.substeps += n; self.pending = 0

    def _sub(self, q, kappa, store, zb, g, dt, win=None):
        """One substep of the slow laws on state arrays (the panel's own, or contiguous copies of a window)."""
        L = self.L
        h = q.sum(0); phi = self._phi(q, h, self.nc); eta = self._eta(phi, kappa); tau = self._tau(phi, kappa)
        wet = h > 1e-9
        frac = q / np.maximum(h, 1e-300)
        Hn, Wn = h.shape
        Fx = np.zeros((len(q), Hn, Wn - 1)); Fy = np.zeros((len(q), Hn - 1, Wn))
        # 1. leveling: Bingham thin film down the pressure gradient
        if L["flow"] > 0:
            P = L["G"] * (h + zb)
            for axis, F in ((1, Fx), (0, Fy)):
                a = (slice(None), slice(None, -1)) if axis == 1 else (slice(None, -1), slice(None))
                b = (slice(None), slice(1, None)) if axis == 1 else (slice(1, None), slice(None))
                gr = P[b] - P[a]; ag = np.abs(gr) + 1e-300
                hd = np.where(gr < 0, h[a], h[b]); rw = np.where(gr < 0, wet[b], wet[a])   # donor, receiver wet?
                # a contact line is pinned: onto dry ground a film advances only by its thickness above h_pin
                hf = np.where(rw, 0.5 * (h[a] + h[b]), np.maximum(hd - L["h_pin"], 0))
                ef = 2 / (1 / eta[a] + 1 / eta[b]); tf = np.maximum(tau[a], tau[b])
                Y = np.clip(hf - tf / ag, 0, None)
                Q = ag * Y * Y * (3 * hf - Y) / (6 * ef) * dt * L["flow"]
                Q = -np.sign(gr) * np.minimum(Q, 0.2 * ag / L["G"])   # no overshoot: a face never moves more than
                                                                     # a fifth of the surface step (max principle)
                F += np.where(Q > 0, frac[(slice(None),) + a], frac[(slice(None),) + b]) * Q
        # 3. colour: pigments exchange among the solids, Stokes-Einstein
        if L["diffusion"] > 0 and len(q) > self.nc + 1:
            s = q[self.nc:]; st = s.sum(0); x = s / np.maximum(st, 1e-300)
            Dc = L["D0"] * L["eta0"] / eta * L["diffusion"] * dt
            for axis, F in ((1, Fx), (0, Fy)):
                a = (slice(None), slice(None, -1)) if axis == 1 else (slice(None, -1), slice(None))
                b = (slice(None), slice(1, None)) if axis == 1 else (slice(1, None), slice(None))
                # each face exchanges at most a fifth of the composition difference (max principle): stable at any dt
                Df = np.minimum(np.minimum(Dc[a], Dc[b]), 0.2) * (wet[a] & wet[b]) * np.minimum(st[a], st[b])
                F[self.nc:] += Df * (x[(slice(None),) + a] - x[(slice(None),) + b])
        mk = None
        if self._mask is not None:                     # frisket: a boundary condition on every law
            mk = self._mask if win is None else self._mask[win[0]:win[1], win[2]:win[3]]
            ox, oy = _open_faces(mk)
            Fx = np.where(ox, Fx, 0.0); Fy = np.where(oy, Fy, 0.0)
        Fx, Fy = limit(Fx, Fy, q)
        q[..., :, :-1] -= Fx; q[..., :, 1:] += Fx; q[..., :-1, :] -= Fy; q[..., 1:, :] += Fy
        np.maximum(q, 0, out=q)                        # only round-off can go negative here
        # the ledger's sums: over the panel, or over a zero-filled panel-sized buffer (the same rounding)
        total = (lambda a: a.sum()) if win is None else (lambda a: _dense_sum((self.H, self.W), *win, a) if a.any() else None)
        # 4. water: evaporates (faster at a wet edge) and soaks into the ground
        w = q[0]
        if L["evaporation"] > 0:
            m = (w > 1e-3).astype(float)
            edge = m * (1 - ndimage.uniform_filter(m, 3))
            e = np.minimum(w, (L["evap"] + L["edge_evap"] * edge) * dt * L["evaporation"] * (w > 0))
            if mk is not None: e = np.where(mk, 0.0, e)
            w -= e; s_ = total(e)
            if s_ is not None: self.led["evaporated"] += s_
        if L["absorption"] > 0:
            a = np.minimum(w, L["absorb"] * np.exp(-(zb - g) / 0.02) * dt * np.maximum(L["store_cap"] - store, 0) / L["store_cap"] * L["absorption"])
            if mk is not None: a = np.where(mk, 0.0, a)
            w -= a; store += a
            se = np.minimum(store, L["store_evap"] * dt * L["evaporation"])
            if mk is not None: se = np.where(mk, 0.0, se)
            store -= se; s_ = total(se)
            if s_ is not None: self.led["store_evaporated"] += s_
        # 5. oil cures where there is oil, faster at the surface of thin films (oxygen)
        if L["curing"] > 0 and L["cure_rate"] > 0:
            oil = q[1]; hh = np.maximum(q.sum(0), 1e-12)
            dk = L["cure_rate"] * dt * L["curing"] * (oil / hh) * (1 - kappa) / (0.3 + hh)
            if mk is not None: dk = np.where(mk, 0.0, dk)
            kappa += dk

    # ---- stratigraphy: what is dry becomes a layer the wet paint sits on and can no longer move
    def fix(self, cure_min=0.8, region=None):
        """Fix every cell whose paint is dry: no water left, and either lean (little oil) or cured. With `region`
        ((H, W) bool), only the dry cells inside it; the fraction returned is then of the region's wet cells."""
        self.flush()
        if region is not None:
            region = np.asarray(region, dtype=bool)
            if region.shape != (self.H, self.W): raise ValueError(f"fix region must be ({self.H}, {self.W}), not {region.shape}")
        if not self.L["windowed"]:
            win = (0, self.H, 0, self.W); self._ext_live = True
        else:
            win = self._live_bbox()                          # dry paint is live paint: nothing outside can fix
            if win is None: return 0.0
        y0, y1, x0, x1 = win
        full = win == (0, self.H, 0, self.W)
        q = self._q if full else self._q[:, y0:y1, x0:x1]
        h = q.sum(0) if full else np.ascontiguousarray(q).sum(0)
        kap = self._kappa[y0:y1, x0:x1]
        ok = (h > 0) & (q[0] <= 1e-6) & ((q[1] <= 0.05 * np.maximum(h, 1e-12)) | (kap >= cure_min))
        held = h > 0
        if region is not None:
            ok &= region[y0:y1, x0:x1]; held &= region[y0:y1, x0:x1]
        self.led["evaporated"] += (q[0] * ok).sum() if full else _dense_sum((self.H, self.W), *win, q[0] * ok)
        layer = q * ok; layer[0] = 0
        q *= ~ok
        if layer.any():
            b = _bbox(layer.any(0))                          # the stratum, cropped to what it covers
            tot = layer.sum(axis=(1, 2)) if full else _dense_species_sums((self.H, self.W), *win, layer)
            self.strata._append(b[0] + y0, b[1] + y0, b[2] + x0, b[3] + x0, layer[:, b[0]:b[1], b[2]:b[3]], tot)
            if full: self.zb = self.zb + layer.sum(0)
            else: zb = self.zb.copy(); zb[y0:y1, x0:x1] = zb[y0:y1, x0:x1] + np.ascontiguousarray(layer).sum(0); self.zb = zb
        kap[ok] = 0.0
        return float(ok.sum()) / max(1, held.sum())

    def wait(self, steps, evap_boost=1.0, cure_boost=1.0, flow=None):
        """Let time pass (a drying lamp, a day, a month): the same laws, with drying and curing sped up."""
        self.flush(); keep = dict(self.L)
        self.L["evap"] *= evap_boost; self.L["edge_evap"] *= evap_boost; self.L["store_evap"] *= evap_boost
        self.L["cure_rate"] *= cure_boost
        for _ in range(steps): self.step()
        self.flush()
        self.L.update(keep)

    # ---- the touch test: what fix() would fix, without fixing it
    def _dry_cells(self, cure_min):
        """fix()'s test of dryness on the state as it stands (no flush): (H, W) bool, and the volume h."""
        q = self._q; h = q.sum(0)
        return (h > 0) & (q[0] <= DRY_WATER) & ((q[1] <= LEAN * np.maximum(h, 1e-12)) | (self._kappa >= cure_min)), h

    def touch_dry(self, cure_min=0.8):
        """The finger test, (H, W) bool: True where the wet layer holds nothing, or holds only what fix(cure_min)
        would fix now; False where it holds paint fix() would leave wet. Bare cells are touch-dry whatever water
        the ground beneath holds (sheen() shows that). Pure: it never flushes, steps or writes. With slow-law
        ticks pending (lazy > 1) it answers for the state fix() would see, by flushing a scratch fork."""
        if self.pending > 0:
            f = self.fork(); f.flush(); return f.touch_dry(cure_min)
        ok, h = self._dry_cells(cure_min)
        return ok | (h <= 0)

    # ---- the frisket: tape on the panel, a boundary condition on every law
    def mask(self, region):
        """Put frisket over `region` ((H, W) bool), added to any already there. Every law treats a masked cell as
        a wall: no face flux (leveling, colour, tool advection) crosses a face touching it, and nothing evaporates,
        soaks in or cures under it; tools that ask `_open` exchange nothing there. The pending slow laws run
        first (flush), so the tape goes down at a definite time. Raises ValueError if any cell to be masked is
        not touch_dry()."""
        region = np.asarray(region, dtype=bool)
        if region.shape != (self.H, self.W): raise ValueError(f"mask region must be ({self.H}, {self.W}), not {region.shape}")
        self.flush()
        bad = region & ~self.touch_dry()
        if bad.any():
            y, x = np.argwhere(bad)[0]
            raise ValueError(f"cannot mask {int(bad.sum())} cells that are not touch-dry (e.g. y={y}, x={x}): let them dry first")
        m = region.copy() if self._mask is None else (self._mask | region)
        self._mask = m if m.any() else None

    def unmask(self, region=None):
        """Lift the frisket from `region` ((H, W) bool), or all of it (None). Flushes first, like mask()."""
        self.flush()
        if self._mask is None: return
        if region is None: self._mask = None; return
        m = self._mask & ~np.asarray(region, dtype=bool)
        self._mask = m if m.any() else None

    def _open(self, y0, y1, x0, x1):
        """For tools: None when nothing is masked, else the (y1 - y0, x1 - x0) float window, 1 open, 0 masked."""
        if self._mask is None: return None
        return (~self._mask[y0:y1, x0:x1]).astype(float)

    # ---- rehearsal: an independent canvas in the same state
    def fork(self):
        """An independent Canvas in exactly this state (a World service: a rehearsal). The strata's compact blocks
        and totals are immutable (I5) and shared, not copied; everything mutable is copied: the wet layer, cure,
        ground water, relief, tool fields, the ledger, clocks and counters, the windowed books, the laws, the
        frisket, any random state. Brushes are cloned in order (`fork.brushes[i]` is the twin of `brushes[i]`,
        its `.cv` the fork), whatever their class. Continuing a fork exactly as the original continues gives
        bit-identical arrays, ledger and counters (I4); each ledger closes on its own. Changes nothing here."""
        memo = {}
        for it in self.strata._items:                        # share the immutable blocks and totals
            memo[id(it[4])] = it[4]; memo[id(it[5])] = it[5]
        for b in self.brushes:                               # a brush held through a proxy of this canvas: the
            if b.cv is not self: memo[id(b.cv)] = b.cv       # proxy is not copied (the twin is rebound below)
        f = copy.deepcopy(self, memo)
        for b in f.brushes: b.cv = f
        return f

    # ---- bookkeeping
    def ledger(self):
        inb = sum((b.load for b in self.brushes), np.zeros(len(self.species)))
        on = self._q.sum(axis=(1, 2)) + sum((self.strata.totals(i) for i in range(len(self.strata))), np.zeros(len(self.species)))
        res = self.led["loaded"] - inb - on
        res[0] -= self.led["evaporated"] + self.led["store_evaporated"] + self._store.sum()
        return dict(t=self.t, substeps=self.substeps, capped_steps=self.capped,
                    species={s: dict(loaded=float(self.led["loaded"][i]), on_canvas=float(on[i]), in_brushes=float(inb[i]), residual=float(res[i]))
                             for i, s in enumerate(self.species)},
                    water=dict(evaporated=self.led["evaporated"], in_ground=float(self._store.sum()), ground_evaporated=self.led["store_evaporated"]))

    # ---- the observer: Kubelka-Munk in R, G, B and near-infrared (~1.5 um, where reflectography looks),
    #      one layer at a time, composed bottom-up over the ground with Kubelka's layer formula
    def _KS(self, vol, wet):
        K = np.zeros(vol.shape[1:] + (4,)); S = np.zeros(vol.shape[1:] + (4,))
        for i, s in enumerate(self.species[self.nc:]):
            p = PIGMENTS[s]; v = vol[self.nc + i][..., None]
            K += v * np.array(p["K"]); S += v * np.array(p["S"])
        k = self._kappa[..., None] if wet else 1.0
        K += vol[1][..., None] * np.array([0.02, 0.03, 0.06, 0.01]) * (1 + 2 * k)    # linseed yellows a touch
        return K, S

    @staticmethod
    def km_layer(K, S):
        """Reflectance and transmittance of a layer with total absorption K and scattering S (both x thickness)."""
        S = np.maximum(S, 1e-7); a = 1 + K / S; b = np.sqrt(np.maximum(a * a - 1, 1e-14))
        bS = np.minimum(b * S, 30.0); sh, ch = np.sinh(bS), np.cosh(bS)
        D = a * sh + b * ch
        return sh / D, b / D

    # >>> optics hook: reflectance() and finish() are the canvas's optics model's (goo/classic/optics.py, SPEC §6)
    def reflectance(self, gain=10.0, layers=None, light=None, adapt=False):
        """What is seen, (H, W, nb) in the optics model's bands (classic: R, G, B, 1.5 um). gain converts volume
        units (0.25 mm) to the K, S table's units. layers: how many strata to include (None = all, plus the wet
        layer); lets you peel the painting back like a cross-section. light: a light's name ("D65", "A") or
        spectrum: return the colour under it instead, linear sRGB (H, W, 3); adapt: as an eye adapted to it."""
        from goo.classic.optics import optics_of
        return optics_of(self).reflectance(self, gain=gain, layers=layers, light=light, adapt=adapt)

    def sheen(self):
        """How wet the surface looks, (H, W) in 0..1: 1 glistening, in between satin (damp), 0 dry. Pure, like
        reflectance(): it reads and changes nothing.

        Free liquid is the wet layer's water above DRY_WATER plus its oil where the film without its water is not
        lean (oil > LEAN * (h - water)) and not yet cured to the touch (kappa < CURE_TOUCH). The pigment bed, packed at phi_m,
        holds bed = solids * (1 - phi_m) / phi_m of liquid in its pores. Liquid standing above the bed glistens:
        sheen = 0.5 + 0.5 * clip((free - bed) / GLISTEN, 0, 1). Otherwise the film has sunk into the bed and the
        surface is satin as far as its pores and the ground are filled: sheen = 0.5 * clip((free + store) /
        (bed + store_cap), 0, 1). Bare dry ground is 0. Monotone: sheen never rises when water evaporates, soaks
        into the ground or the ground dries, or when oil cures."""
        q = self._q; L = self.L; pm = L["phi_m"]
        h = q.sum(0); solids = q[self.nc:].sum(0)
        film = h - q[0]                                    # lean is judged on the film without its water, so
        free = np.where(q[0] > DRY_WATER, q[0], 0.0) + np.where((q[1] > LEAN * film) & (self._kappa < CURE_TOUCH), q[1], 0.0)  # drying never frees oil
        bed = solids * (1 - pm) / pm
        stand = free - bed
        satin = 0.5 * np.clip((free + self._store) / (bed + L["store_cap"]), 0, 1)
        return np.where(stand > 0, 0.5 + 0.5 * np.clip(stand / GLISTEN, 0, 1), satin)

    def finish(self):
        """Gloss 0..1 of the top surface at each cell, (H, W), as the optics model reads it (classic: from the
        top film's thickness and oil fraction, SPEC §6). A pure function of the state: it reads and changes nothing."""
        from goo.classic.optics import optics_of
        return optics_of(self).finish(self)
    # <<< optics hook
