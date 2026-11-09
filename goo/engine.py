"""goo.engine: the World. One paint layer for every medium, on a ground, over the strata it has dried into.

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
import numpy as np
from scipy import ndimage

CARRIERS = ["water", "oil"]
# Kubelka-Munk per unit volume: K and S for (R, G, B, NIR ~ 900 nm)
PIGMENTS = {
    "carbon":      dict(K=[3.2, 3.2, 3.2, 3.0], S=[0.10, 0.10, 0.10, 0.10]),
    "white":       dict(K=[0.02, 0.02, 0.03, 0.01], S=[3.2, 3.2, 3.0, 0.45]),   # scattering falls with wavelength
    "ultramarine": dict(K=[2.3, 1.7, 0.20, 0.04], S=[0.05, 0.06, 0.30, 0.05]),
    "ochre":       dict(K=[0.25, 0.55, 1.70, 0.05], S=[0.45, 0.40, 0.25, 0.10]),
    "sienna":      dict(K=[0.30, 1.25, 2.40, 0.08], S=[0.30, 0.20, 0.10, 0.12]),
    "rose":        dict(K=[0.10, 2.30, 0.85, 0.02], S=[0.03, 0.02, 0.04, 0.02]),
    "green":       dict(K=[1.60, 0.30, 1.90, 0.10], S=[0.06, 0.08, 0.04, 0.05]),
}
GROUND_RGBN = np.array([0.93, 0.92, 0.88, 0.90])   # gesso, visible and near-infrared


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


class Canvas:
    def __init__(self, W, H, pigments, seed=1, ground="gesso", laws=None):
        self.W, self.H = W, H
        self.species = CARRIERS + list(pigments)
        self.nc = len(CARRIERS)
        self.q = np.zeros((len(self.species), H, W))
        self.kappa = np.zeros((H, W))                       # oil cure, 0..1
        self.store = np.zeros((H, W))                       # water soaked into the ground
        rng = np.random.default_rng(seed)
        n = lambda s: ndimage.gaussian_filter(rng.standard_normal((H, W)), s)
        tex = 0.5 * n(1.0) / n(1.0).std() + 0.6 * n(4.0) / n(4.0).std() + 0.25 * n((0.8, 12)) / n((0.8, 12)).std()
        self.g = 0.03 * tex                                 # ground relief (gesso brush marks and tooth)
        self.strata = []                                    # dried / fixed layers, bottom first: (species, H, W)
        self.zb = self.g.copy()                             # what the wet layer sits on: ground + strata
        self.L = dict(G=1.0, eta0=0.12, phi_m=0.64, phi_y=0.2, tau0=0.05, tau_cure=1.0, D0=0.15, evap=0.0006, edge_evap=0.006,
                      absorb=0.004 if ground == "gesso" else 0.04, store_cap=0.08, store_evap=0.002,
                      cure_rate=0.0, cure_stiff=400.0, cure_yield=300.0, drag_range=1.4,
                      flow=1.0, drag=1.0, diffusion=1.0, evaporation=1.0, absorption=1.0, curing=1.0, max_sub=600, flow_sub=48, h_pin=0.08)
        if laws: self.L.update(laws)
        self.U = np.zeros((2, H, W))                        # tool velocity imposed this step (cells / step)
        self.V = np.zeros((2, H, W))                        # plough: paint squeezed out from under a tool
        self.led = dict(loaded=np.zeros(len(self.species)), evaporated=0.0, store_evaporated=0.0)
        self.brushes = []
        self.t = 0; self.substeps = 0; self.capped = 0

    # ---- putting paint down by hand (setups, tests): booked as loaded, like anything a brush brings
    def add(self, name, field):
        """Add volume `field` (H, W, or scalar) of species `name` to the wet layer, and book it."""
        f = np.broadcast_to(np.asarray(field, float), (self.H, self.W))
        if (f < 0).any(): raise ValueError("add() takes volume, not a debit")
        i = self.idx(name); self.q[i] += f; self.led["loaded"][i] += f.sum()

    # ---- derived fields
    def idx(self, name): return self.species.index(name)
    @property
    def h(self): return self.q.sum(0)
    def phi(self):
        h = self.h; return np.where(h > 1e-12, self.q[self.nc:].sum(0) / np.maximum(h, 1e-12), 1.0)
    def eta(self, phi):
        L = self.L; return L["eta0"] * (1 - np.minimum(phi, 0.99 * L["phi_m"]) / L["phi_m"]) ** -2 * (1 + L["cure_stiff"] * self.kappa)
    def tau(self, phi):
        L = self.L; pm = L["phi_m"]
        return (L["tau0"] * (np.maximum(phi - L["phi_y"], 0) / (pm - np.minimum(phi, 0.99 * pm))) ** 1.5 * (1 + L["cure_yield"] * self.kappa ** 2)
                + L["tau_cure"] * self.kappa ** 2)                # a skin forms on any cured oil, lean or fat
    def surface(self): return self.zb + self.h
    def porous(self): return np.exp(-(self.zb - self.g) / 0.02)   # a dried film seals the ground

    # ---- the tool velocity: imposed where tools touch, spread sideways by a screened Poisson solve
    def _drag_velocity(self):
        if self.L["drag"] == 0 or not (self.U.any() or self.V.any()): return None
        pad = 24; H, W = self.H + 2 * pad, self.W + 2 * pad
        ky = np.fft.fftfreq(H)[:, None] * 2 * np.pi; kx = np.fft.rfftfreq(W)[None, :] * 2 * np.pi
        filt = 1 / (1 + self.L["drag_range"] ** 2 * (kx ** 2 + ky ** 2))
        out = []
        for c in range(2):
            a = np.pad(self.U[c], pad)
            out.append(np.fft.irfft2(np.fft.rfft2(a) * filt, s=(H, W))[pad:-pad, pad:-pad])
        u = (np.stack(out) + self.V) * self.L["drag"]
        return u * (self.h > 1e-9)                          # only paint is carried

    # ---- one step of unit time, in as many substeps as the stiffest process needs
    def step(self):
        L = self.L
        u = self._drag_velocity()
        # operator split: (a) the tools carry paint, in the box where their velocity matters, at the tool's CFL;
        #                 (b) leveling, colour diffusion, drying over the whole panel, at their own limits
        if u is not None:
            umax = np.abs(u).max()
            live = (np.abs(u) > 1e-3 * umax).any(0)
            ys, xs = np.nonzero(live)
            y0, y1 = max(0, ys.min() - 3), min(self.H, ys.max() + 4); x0, x1 = max(0, xs.min() - 3), min(self.W, xs.max() + 4)
            nt = int(np.ceil(max(1.0, umax / 0.5)))
            if nt > L["max_sub"]: self.capped += 1; nt = L["max_sub"]
            ub = u[:, y0:y1, x0:x1]
            ux = 0.5 * (ub[0][:, :-1] + ub[0][:, 1:]); uy = 0.5 * (ub[1][:-1, :] + ub[1][1:, :])
            for _ in range(nt):
                qb = self.q[:, y0:y1, x0:x1]
                Fx, Fy = limit(muscl_flux(qb, ux, -1, 1.0 / nt), muscl_flux(qb, uy, -2, 1.0 / nt), qb)
                qb[..., :, :-1] -= Fx; qb[..., :, 1:] += Fx; qb[..., :-1, :] -= Fy; qb[..., 1:, :] += Fy
                np.maximum(qb, 0, out=qb)
            self.substeps += nt
        h = self.h; phi = self.phi(); eta = self.eta(phi)
        # Bingham mobility from the yielded depth Y only: a paste below its yield stress doesn't set the step
        gy, gx = np.gradient(L["G"] * (h + self.zb)); ag = np.hypot(gx, gy) + 1e-300
        Y = np.clip(h - self.tau(phi) / ag, 0, None)
        mob = (Y * Y * (3 * h - Y) / (6 * eta)).max() * L["flow"]
        D = (L["D0"] * L["eta0"] / eta).max() * L["diffusion"]
        n_flow = min(4.4 * L["G"] * mob / 0.8, L["flow_sub"])    # beyond this the face cap keeps it stable
        n = int(np.ceil(max(1.0, n_flow, 4 * D / 0.8)))
        if n > L["max_sub"]: self.capped += 1; n = L["max_sub"]
        for _ in range(n): self._sub(1.0 / n)
        self.substeps += n; self.t += 1
        self.U[:] = 0; self.V[:] = 0

    def _sub(self, dt):
        L, q = self.L, self.q
        h = q.sum(0); phi = self.phi(); eta = self.eta(phi); tau = self.tau(phi)
        wet = h > 1e-9
        frac = q / np.maximum(h, 1e-300)
        Fx = np.zeros((len(q), self.H, self.W - 1)); Fy = np.zeros((len(q), self.H - 1, self.W))
        # 1. leveling: Bingham thin film down the pressure gradient
        if L["flow"] > 0:
            P = L["G"] * (h + self.zb)
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
                Df = np.minimum(Dc[a], Dc[b]) * (wet[a] & wet[b]) * np.minimum(st[a], st[b])
                F[self.nc:] += Df * (x[(slice(None),) + a] - x[(slice(None),) + b])
        Fx, Fy = limit(Fx, Fy, q)
        q[..., :, :-1] -= Fx; q[..., :, 1:] += Fx; q[..., :-1, :] -= Fy; q[..., 1:, :] += Fy
        np.maximum(q, 0, out=q)                        # only round-off can go negative here
        # 4. water: evaporates (faster at a wet edge) and soaks into the ground
        w = q[0]
        if L["evaporation"] > 0:
            m = (w > 1e-3).astype(float)
            edge = m * (1 - ndimage.uniform_filter(m, 3))
            e = np.minimum(w, (L["evap"] + L["edge_evap"] * edge) * dt * L["evaporation"] * (w > 0))
            w -= e; self.led["evaporated"] += e.sum()
        if L["absorption"] > 0:
            a = np.minimum(w, L["absorb"] * self.porous() * dt * np.maximum(L["store_cap"] - self.store, 0) / L["store_cap"] * L["absorption"])
            w -= a; self.store += a
            se = np.minimum(self.store, L["store_evap"] * dt * L["evaporation"]); self.store -= se; self.led["store_evaporated"] += se.sum()
        # 5. oil cures where there is oil, faster at the surface of thin films (oxygen)
        if L["curing"] > 0 and L["cure_rate"] > 0:
            oil = q[1]; hh = np.maximum(q.sum(0), 1e-12)
            self.kappa += L["cure_rate"] * dt * L["curing"] * (oil / hh) * (1 - self.kappa) / (0.3 + hh)

    # ---- stratigraphy: what is dry becomes a layer the wet paint sits on and can no longer move
    def fix(self, cure_min=0.8):
        """Fix every cell whose paint is dry: no water left, and either lean (little oil) or cured."""
        q = self.q; h = q.sum(0)
        ok = (h > 0) & (q[0] <= 1e-6) & ((q[1] <= 0.05 * np.maximum(h, 1e-12)) | (self.kappa >= cure_min))
        self.led["evaporated"] += (q[0] * ok).sum()
        layer = q * ok; layer[0] = 0
        q *= ~ok
        if layer.any(): self.strata.append(layer); self.zb = self.zb + layer.sum(0)
        self.kappa[ok] = 0.0
        return float(ok.sum()) / max(1, (h > 0).sum())

    def wait(self, steps, evap_boost=1.0, cure_boost=1.0, flow=None):
        """Let time pass (a drying lamp, a day, a month): the same laws, with drying and curing sped up."""
        keep = dict(self.L)
        self.L["evap"] *= evap_boost; self.L["edge_evap"] *= evap_boost; self.L["store_evap"] *= evap_boost
        self.L["cure_rate"] *= cure_boost
        for _ in range(steps): self.step()
        self.L.update(keep)

    # ---- bookkeeping
    def ledger(self):
        inb = sum((b.load for b in self.brushes), np.zeros(len(self.species)))
        on = self.q.sum(axis=(1, 2)) + sum((s.sum(axis=(1, 2)) for s in self.strata), np.zeros(len(self.species)))
        res = self.led["loaded"] - inb - on
        res[0] -= self.led["evaporated"] + self.led["store_evaporated"] + self.store.sum()
        return dict(t=self.t, substeps=self.substeps, capped_steps=self.capped,
                    species={s: dict(loaded=float(self.led["loaded"][i]), on_canvas=float(on[i]), in_brushes=float(inb[i]), residual=float(res[i]))
                             for i, s in enumerate(self.species)},
                    water=dict(evaporated=self.led["evaporated"], in_ground=float(self.store.sum()), ground_evaporated=self.led["store_evaporated"]))

    # ---- the observer: Kubelka-Munk in R, G, B and near-infrared (~1.5 um, where reflectography looks),
    #      one layer at a time, composed bottom-up over the ground with Kubelka's layer formula
    def _KS(self, vol, wet):
        K = np.zeros((self.H, self.W, 4)); S = np.zeros((self.H, self.W, 4))
        for i, s in enumerate(self.species[self.nc:]):
            p = PIGMENTS[s]; v = vol[self.nc + i][..., None]
            K += v * np.array(p["K"]); S += v * np.array(p["S"])
        k = self.kappa[..., None] if wet else 1.0
        K += vol[1][..., None] * np.array([0.02, 0.03, 0.06, 0.01]) * (1 + 2 * k)    # linseed yellows a touch
        return K, S

    @staticmethod
    def km_layer(K, S):
        """Reflectance and transmittance of a layer with total absorption K and scattering S (both x thickness)."""
        S = np.maximum(S, 1e-7); a = 1 + K / S; b = np.sqrt(np.maximum(a * a - 1, 1e-14))
        bS = np.minimum(b * S, 30.0); sh, ch = np.sinh(bS), np.cosh(bS)
        D = a * sh + b * ch
        return sh / D, b / D

    def reflectance(self, gain=10.0, layers=None):
        """gain converts volume units (0.25 mm) to the K, S table's units. layers: how many strata to include
        (None = all, plus the wet layer); lets you peel the painting back like a cross-section."""
        Rb = np.broadcast_to(GROUND_RGBN, (self.H, self.W, 4)).copy()
        stack = [(s, False) for s in self.strata] + [(self.q, True)]
        if layers is not None: stack = stack[:layers]
        for vol, wet in stack:
            K, S = self._KS(vol * gain, wet)
            R, T = self.km_layer(K, S)
            Rb = R + T * T * Rb / (1 - R * Rb)
        return np.clip(Rb, 0, 1)
