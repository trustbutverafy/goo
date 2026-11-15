"""The world: state, F, the step, and the matter and energy ledgers (SPEC §3, §6, §8; docs/DYNAMICS.md)."""
import copy
import hashlib
import numpy as np
from goo.film import Film, Ground, Stack
from goo.ledger import Ledger, Receipt, Flag
from goo.world.protocol import Environment, call
from goo.world.grid import Grid, panel_faces
from goo.world.common import Ctx, Reject, mixing, pore_B
from goo.world.flow import flow
from goo.world.diffuse import diffuse
from goo.world.local import local

MAX_HALVINGS = 24
TERMS = ("gravity", "capillary", "wetting", "mixing", "exchange", "sorption")
MECHANISMS = ("viscous", "plastic", "diffusion", "exchange", "sorption")


class World:
    """A depth-averaged wet film of `substances` on `ground`, with strata (bottom first), an environment, a
    clock and two ledgers. `constitutive` satisfies goo.world.Constitutive. q: [S, H, W] volume per area (m)."""

    def __init__(self, constitutive, substances, ground, dx, environment=None, q=None, strata=(),
                 slope_bound=0.1, margin=3):
        self.con = constitutive
        self.substances = tuple(substances)
        S = len(self.substances)
        self.rho = np.array([float(s.rho) for s in self.substances])
        H, W = np.asarray(ground.z).shape
        self.H, self.W, self.dx = H, W, float(dx)
        g = ground.copy()
        if tuple(g.substances) != self.substances:
            if g.substances and np.any(g.held):
                raise ValueError("ground holds substances that are not the world's")
            g = Ground(g.z, g.solid, g.porosity, g.pore_radius, g.depth, self.substances, None)
        self.ground = g
        self.q = np.zeros((S, H, W)) if q is None else np.array(q, np.float64)
        if self.q.shape != (S, H, W) or np.any(self.q < 0):
            raise ValueError("q must be [S, H, W] and non-negative")
        self.strata = [np.array(s, np.float64) for s in strata]
        self.env = environment if environment is not None else Environment()
        self.t = 0.0
        self.slope_bound, self.margin = float(slope_bound), int(margin)
        self.a = float(np.sqrt(self.rho.min() * self.rho.max()))
        self.matter = Ledger("matter", (S,), tuple(s.name for s in self.substances))
        self.energy = Ledger("energy")
        self._grid = Grid(H, W, self.dx)
        self._coef_flags = []
        self._lagged = self._coefficients()
        self.matter.post("origin", "wet", self._mass(self.q), "initial film")
        if np.any(self.ground.held):
            self.matter.post("origin", "ground", self._mass(self.ground.held), "initial pores")
        for k, s in enumerate(self.strata):
            self.matter.post("origin", f"stratum[{k}]", self._mass(s), "initial stratum")
        F0 = self.F()
        self.energy.post("work[origin]", "stored", sum(F0.values()), "initial free energy")
        self.receipt0 = Receipt("create", 0.0, 0.0, self.matter.since(0) + self.energy.since(0),
                                self._residuals(F0, 0.0), {k: (0.0, v) for k, v in F0.items()})

    # ---- state -------------------------------------------------------------------------------------------
    @property
    def film(self):
        return Film(self.substances, self.q)

    @property
    def zb(self):
        return self.ground.z + sum((s.sum(0) for s in self.strata), np.zeros((self.H, self.W)))

    def _mass(self, q):
        return self.rho * np.asarray(q).sum((1, 2)) * self.dx ** 2

    def _mu_env(self):
        pot = self.env.potentials
        has = np.array([s.name in pot for s in self.substances])
        return np.array([float(pot.get(s.name, 0.0)) for s in self.substances]), has

    # ---- lagged coefficients (DYNAMICS.md §2) -------------------------------------------------------------
    def _coefficients(self):
        flags = []
        film = Film(self.substances, self.q)
        wet = self.q.sum(0) > 0
        gam = call(self.con, "surface_tension", film, flags=flags)
        if gam is None or not np.any(np.asarray(gam) > 0):
            gx, gy = np.zeros((self.H, self.W - 1)), np.zeros((self.H - 1, self.W))
        else:
            gam = np.where(np.isfinite(gam), np.asarray(gam, np.float64), 0.0)

            def face(gL, gR, wL, wR):
                return np.where(wL == wR, 0.5 * (gL + gR), np.where(wL, gL, gR))
            gx = face(gam[:, :-1], gam[:, 1:], wet[:, :-1], wet[:, 1:])
            gy = face(gam[:-1, :], gam[1:, :], wet[:-1, :], wet[1:, :])
        wv = call(self.con, "wetting", film, self.ground, flags=flags)
        Sw = ell = None
        if wv is not None:
            Sw, ell = (np.asarray(x, np.float64) for x in wv)
            if np.any(Sw > 0) or np.any((Sw != 0) & ~(ell > 0)):
                raise ValueError("wetting: spreading coefficient must be <= 0 with a positive length")
            if not np.any(Sw != 0):
                Sw = ell = None
        sv = call(self.con, "sorption", film, self.ground, flags=flags)
        P = Pb = None
        if sv is not None:
            P = np.asarray(sv[0], np.float64)
            if not np.any(P != 0) or self.ground.capacity <= 0:
                P = None
            else:
                Pb = P.max(0)
        self._coef_flags = flags
        return dict(gx=gx, gy=gy, Sw=Sw, ell=ell, P=P, Pb=Pb)

    # ---- free energy -------------------------------------------------------------------------------------
    def F(self, lagged=None):
        """F by term, J (DYNAMICS.md §2), with the lagged coefficients in force (or those given)."""
        c = self._lagged if lagged is None else lagged
        dx2 = self.dx ** 2
        q = self.q
        h, m = q.sum(0), np.tensordot(self.rho, q, 1)
        zb = self.zb
        gxv, gyv, gn = self.env.gravity
        G = self._grid
        X, Y = G.X.reshape(self.H, self.W), G.Y.reshape(self.H, self.W)
        out = {}
        out["gravity"] = dx2 * np.sum(gn * m * (zb + 0.5 * h) - m * (gxv * X + gyv * Y))
        s = zb + h
        cap = 0.0
        for gam, sl in ((c["gx"], (np.s_[:, 1:], np.s_[:, :-1])), (c["gy"], (np.s_[1:, :], np.s_[:-1, :]))):
            dh, ds, dz = h[sl[0]] - h[sl[1]], s[sl[0]] - s[sl[1]], zb[sl[0]] - zb[sl[1]]
            cap += np.sum(0.5 * gam * dh * (ds + dz))
        out["capillary"] = cap
        out["wetting"] = 0.0 if c["Sw"] is None else dx2 * np.sum((-c["Sw"]) * (-np.expm1(-h / c["ell"])))
        wet = np.flatnonzero(h.ravel() > 0)
        mixf = 0.0
        if wet.size:
            ctx = self._ctx_stub()
            mx = mixing(ctx, q.reshape(len(self.substances), -1)[:, wet])
            if mx is not None:
                mixf = dx2 * np.sum(mx[0] + mx[3])
        out["mixing"] = mixf
        menv, _ = self._mu_env()
        out["exchange"] = -dx2 * np.sum(menv * q.sum((1, 2)))
        pore = 0.0
        if c["P"] is not None:
            C = self.ground.capacity
            w = self.ground.held
            pore = dx2 * np.sum(-(c["P"] * w).sum(0) + c["Pb"] * C * pore_B(w.sum(0) / C))
        out["sorption"] = pore
        return {k: float(v) for k, v in out.items()}

    def _ctx_stub(self):
        menv, has = self._mu_env()
        return Ctx(self._grid, self.substances, self.rho, self.con, self.env, self.ground, None, None, None, None,
                   None, None, None, 0.0, self.a, menv, has, [])

    # ---- the step ----------------------------------------------------------------------------------------
    def step(self, dt, rate=None, window="auto"):
        """Advance by dt (s). rate: None, a scalar or an [H, W] field >= 0 multiplying every Psi* (SPEC §8).
        window: 'auto' (the wet cells' bounding box plus a margin) or 'panel'. Returns a Receipt."""
        dt = float(dt)
        if not dt > 0:
            raise ValueError("dt must be positive")
        r = np.ones((self.H, self.W)) if rate is None else np.array(np.broadcast_to(np.asarray(rate, np.float64), (self.H, self.W)))
        if not np.all(np.isfinite(r)):
            raise ValueError("rate must be finite")
        if np.any(r < 0):
            raise ValueError("rate < 0 reverses the world (SPEC §8, impossible); reverse steps are not implemented yet")
        mm, me = self.matter.mark(), self.energy.mark()
        t0 = self.t
        flags = []
        Fb = self.F()
        self._relabel(Fb, flags)
        Fb = self.F()
        if rate is not None and not np.all(r == 1.0):
            flags.append(Flag("privilege", "painter", f"clock rate in [{r.min():g}, {r.max():g}]", float(dt * r.mean())))
        acc = {k: 0.0 for k in MECHANISMS}
        acc.update(num_time=0.0, num_transport=0.0, moved=0.0)
        e = np.zeros(len(self.substances)); a = np.zeros(len(self.substances))
        ctxflags = []
        if np.any(r > 0):
            for rep in self._advance(dt, r, window, 0, ctxflags):
                for k in MECHANISMS:
                    acc[k] += rep[k]
                acc["num_time"] += rep["num_time"]; acc["num_transport"] += rep["num_transport"]
                acc["moved"] += rep["moved"]
                e += rep["e"]; a += rep["a"]
        self.t = t0 + dt
        if np.any(e):
            self.matter.post("wet", "reservoir[atmosphere]", self.rho * e, "exchange")
        if np.any(a):
            self.matter.post("wet", "ground", self.rho * a, "sorption")
        for k in MECHANISMS:
            if acc[k] != 0.0:
                self.energy.post("stored", f"dissipated[{k}]", acc[k], k)
        if acc["num_time"] != 0.0:
            self.energy.post("stored", "dissipated[numerical]", acc["num_time"], "time")
        if acc["num_transport"] != 0.0:
            self.energy.post("stored", "dissipated[numerical]", acc["num_transport"], "transport")
        Fa = self.F()
        flags += self._flags(ctxflags)
        sl = self.slope()
        if sl > self.slope_bound:
            flags.append(Flag("slope", "film", f"max surface slope {sl:.3g} above the bound {self.slope_bound:g}", sl))
        res = self._residuals(Fa, acc["moved"])
        rcpt = Receipt("step", t0, self.t, self.matter.since(mm) + self.energy.since(me), res,
                       {k: (Fb[k], Fa[k]) for k in TERMS}, tuple(flags))
        object.__setattr__(rcpt, "report", dict(acc))
        return rcpt

    def _flags(self, pairs):
        seen, out = set(), []
        for name, detail in list(self._coef_flags) + list(pairs):
            if (name, detail) not in seen:
                seen.add((name, detail)); out.append(Flag("out_of_domain", name, detail))
        return out

    def _relabel(self, Fb, flags):
        """Re-evaluate the lagged coefficients; book the change of F at fixed state (DYNAMICS.md §2)."""
        new = self._coefficients()
        Fn = self.F(new)
        d = sum(Fn.values()) - sum(Fb.values())
        self._lagged = new
        if d != 0.0:
            self.energy.post("work[constitutive]", "stored", d, "lagged coefficients re-evaluated")
            flags.append(Flag("out_of_domain", "lagged coefficient",
                              "a coefficient of F changed with the state; Marangoni-type coupling not modelled", d))

    def _advance(self, dt, r, window, depth, ctxflags):
        try:
            return [self._piece(dt, r, window, ctxflags)]
        except Reject:
            if depth >= MAX_HALVINGS:
                raise RuntimeError(f"step could not be accepted after {MAX_HALVINGS} halvings")
            return self._advance(dt / 2, r, window, depth + 1, ctxflags) + self._advance(dt / 2, r, window, depth + 1, ctxflags)

    def _window(self, window):
        wet = self.q.sum(0) > 0
        if window == "panel":
            return 0, self.H, 0, self.W
        if window != "auto":
            raise ValueError("window: 'auto' or 'panel'")
        if not wet.any():
            return None
        rows, cols = np.flatnonzero(wet.any(1)), np.flatnonzero(wet.any(0))
        M = self.margin
        return (max(0, rows[0] - M), min(self.H, rows[-1] + 1 + M), max(0, cols[0] - M), min(self.W, cols[-1] + 1 + M))

    def _piece(self, dt, r, window, ctxflags):
        rep = {k: 0.0 for k in MECHANISMS}
        S = len(self.substances)
        rep.update(num_time=0.0, num_transport=0.0, moved=0.0, e=np.zeros(S), a=np.zeros(S))
        if not np.any(self.q):
            return rep
        win = self._window(window)
        if win is None:
            return rep
        y0, y1, x0, x1 = win
        Hh, Ww = y1 - y0, x1 - x0
        G = Grid(Hh, Ww, self.dx, y0, x0)
        c = self._lagged
        flat = lambda arr: None if arr is None else np.ascontiguousarray(arr[..., y0:y1, x0:x1]).reshape(arr.shape[:-2] + (-1,))
        menv, has = self._mu_env()
        flags = []
        ctx = Ctx(G, self.substances, self.rho, self.con, self.env, self.ground, flat(self.zb), flat(r),
                  panel_faces(c["gx"], c["gy"], y0, y1, x0, x1), flat(c["Sw"]), flat(c["ell"]), flat(c["P"]),
                  flat(c["Pb"]), self.ground.capacity, self.a, menv, has, flags)
        q = flat(self.q).copy()
        held = flat(self.ground.held).copy()
        q1, fr = flow(ctx, q, dt)
        q2, dr = diffuse(ctx, q1, dt)
        q3, held3, lr = local(ctx, q2, held, dt)
        # commit
        self.q[:, y0:y1, x0:x1] = q3.reshape(S, Hh, Ww)
        self.ground.held[:, y0:y1, x0:x1] = held3.reshape(S, Hh, Ww)
        ctxflags.extend(flags)
        rep["viscous"], rep["plastic"] = fr["D_viscous"], fr["D_plastic"]
        rep["diffusion"] = dr["D_diffusion"]
        rep["exchange"], rep["sorption"] = lr["D_exchange"], lr["D_sorption"]
        rep["num_time"] = ((-fr["dE"] - fr["D_viscous"] - fr["D_plastic"]) + (-dr["dF"] - dr["D_diffusion"])
                           + (-lr["dF"] - lr["D_exchange"] - lr["D_sorption"]))
        rep["num_transport"] = -fr["dF_mix"]
        rep["moved"] = (abs(fr["dE"]) + abs(fr["dF_mix"]) + fr["D_viscous"] + fr["D_plastic"] + abs(dr["dF"])
                        + dr["D_diffusion"] + abs(lr["dF"]) + lr["D_exchange"] + lr["D_sorption"])
        rep["e"], rep["a"] = lr["e"], lr["a"]
        return rep

    # ---- residuals, flags --------------------------------------------------------------------------------
    def _residuals(self, F, moved):
        state = {"wet": self._mass(self.q), "ground": self._mass(self.ground.held)}
        for k, s in enumerate(self.strata):
            state[f"stratum[{k}]"] = self._mass(s)
        scale = float(sum(np.sum(v) for v in state.values())) or 1.0
        rm = max(float(np.max(np.abs(self.matter.balance(k) - v))) for k, v in state.items()) / scale
        Ftot = sum(F.values())
        escale = max(sum(abs(v) for v in F.values()), moved, 1e-300)
        re = abs(float(self.energy.balance("stored")) - Ftot) / escale
        return {"matter": rm, "energy": re}

    def slope(self):
        """Largest surface slope |grad s| over faces touching a wet cell."""
        s, wet = self.zb + self.q.sum(0), self.q.sum(0) > 0
        out = 0.0
        for d, w in ((np.diff(s, axis=1), wet[:, 1:] | wet[:, :-1]), (np.diff(s, axis=0), wet[1:, :] | wet[:-1, :])):
            if w.any():
                out = max(out, float(np.max(np.abs(d[w]))) / self.dx)
        return out

    # ---- ports, retirement, environment ------------------------------------------------------------------
    def add(self, amounts, src="origin"):
        """Put matter into the wet film: amounts [S, H, W] (m), or {substance or name: field or scalar}. Books its
        mass from `src` and the free energy it brings from work[src]."""
        S = len(self.substances)
        if isinstance(amounts, dict):
            arr = np.zeros((S, self.H, self.W))
            for key, v in amounts.items():
                name = getattr(key, "name", key)
                i = [s.name for s in self.substances].index(name)
                arr[i] = np.broadcast_to(np.asarray(v, np.float64), (self.H, self.W))
        else:
            arr = np.array(np.broadcast_to(np.asarray(amounts, np.float64), (S, self.H, self.W)))
        if np.any(arr < 0) or not np.all(np.isfinite(arr)):
            raise ValueError("add: amounts must be finite and non-negative (taking matter out is a body's act)")
        mm, me, t = self.matter.mark(), self.energy.mark(), self.t
        Fb = self.F()
        self.q += arr
        Fa = self.F()
        self.matter.post(src, "wet", self._mass(arr), "added")
        self.energy.post(f"work[{src}]", "stored", sum(Fa.values()) - sum(Fb.values()), "free energy brought")
        return Receipt("add", t, t, self.matter.since(mm) + self.energy.since(me), self._residuals(Fa, 0.0),
                       {k: (Fb[k], Fa[k]) for k in TERMS}, tuple(self._flags([])))

    def retire(self, mask=None, substances=None):
        """Turn the given substances (default all) of the wet film in `mask` (default every wet cell) into a new
        stratum on top of the others (SPEC §6). Matter wet -> stratum[k]; the change of the live F stored -> frozen."""
        sel = np.ones(len(self.substances), bool) if substances is None else np.isin(
            [s.name for s in self.substances], [getattr(x, "name", x) for x in substances])
        mk = (self.q.sum(0) > 0) if mask is None else np.asarray(mask, bool)
        moved = np.where(sel[:, None, None] & mk[None], self.q, 0.0)
        mm, me, t = self.matter.mark(), self.energy.mark(), self.t
        Fb = self.F()
        self.q = self.q - moved
        self.q[sel[:, None, None] & mk[None]] = 0.0
        k = len(self.strata)
        self.strata.append(moved)
        Fa = self.F()
        self.matter.post("wet", f"stratum[{k}]", self._mass(moved), "retired")
        self.energy.post("stored", "frozen", sum(Fb.values()) - sum(Fa.values()), f"retired into stratum[{k}]")
        return Receipt("retire", t, t, self.matter.since(mm) + self.energy.since(me), self._residuals(Fa, 0.0),
                       {kk: (Fb[kk], Fa[kk]) for kk in TERMS}, ())

    def set_environment(self, environment):
        """Change the reservoirs: an act whose ΔF is work by the environment's port."""
        mm, me, t = self.matter.mark(), self.energy.mark(), self.t
        Fb = self.F()
        self.env = environment
        Fa = self.F()
        self.energy.post("work[environment]", "stored", sum(Fa.values()) - sum(Fb.values()), "environment changed")
        return Receipt("environment", t, t, self.energy.since(me), self._residuals(Fa, 0.0),
                       {k: (Fb[k], Fa[k]) for k in TERMS}, ())

    # ---- forks, looks, digests ---------------------------------------------------------------------------
    def fork(self):
        new = copy.copy(self)
        new.q = self.q.copy()
        new.ground = self.ground.copy()
        new.strata = [s.copy() for s in self.strata]
        new.matter, new.energy = self.matter.fork(), self.energy.fork()
        new._lagged = {k: (None if v is None else v.copy()) for k, v in self._lagged.items()}
        new._coef_flags = list(self._coef_flags)
        return new

    def stack(self):
        """A goo.film.Stack of the world for a look: read-only views, nothing copied, nothing loaded."""
        def ro(a):
            v = a.view(); v.flags.writeable = False; return v
        g = self.ground
        ground = Ground(ro(g.z), g.solid, g.porosity, g.pore_radius, g.depth, g.substances, ro(g.held))
        films = [Film(self.substances, ro(s)) for s in self.strata] + [Film(self.substances, ro(self.q))]
        return Stack(ground, films, self.dx)

    def digest(self):
        h = hashlib.sha256(repr((self.t, self.env.T, self.env.gravity, sorted(self.env.potentials.items()))).encode())
        for a in [self.q, self.ground.z, self.ground.held] + self.strata:
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(self.matter.digest().encode()); h.update(self.energy.digest().encode())
        return h.hexdigest()

    def state_digest(self):
        """Digest of the state arrays alone (no ledgers)."""
        h = hashlib.sha256(repr(self.t).encode())
        for a in [self.q, self.ground.z, self.ground.held] + self.strata:
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()
