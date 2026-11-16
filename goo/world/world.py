"""The world: state, F, the step, and the matter and energy ledgers (SPEC §3, §6, §8; docs/DYNAMICS.md)."""
import copy
import hashlib
import numpy as np
from goo.film import Film, Ground, Stack
from goo.ledger import Ledger, Receipt, Flag
from goo.world.protocol import Environment, call, unwrap
from goo.world.grid import Grid, panel_faces
from goo.world.common import Ctx, Reject, mixing, mixing_potentials, pore_B
from goo.units import k_B
from goo.world.flow import flow
from goo.world.diffuse import diffuse
from goo.world.local import local
from goo.world.linsolve import SolveFailed, RTOL

MAX_HALVINGS = 24


class StepFailed(RuntimeError):
    """A step that could not be accepted after MAX_HALVINGS halvings. The world is left exactly as it was before the
    step (nothing of it is committed), so the caller may retry with another dt."""


def _union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))


def _inside(a, b):
    """Box a inside box b."""
    return b is not None and a[0] >= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] <= b[3]


def _sl(box):
    y0, y1, x0, x1 = box
    return np.s_[..., y0:y1, x0:x1]
TERMS = ("gravity", "capillary", "wetting", "mixing", "exchange", "sorption")
MECHANISMS = ("viscous", "plastic", "diffusion", "exchange", "sorption")


class World:
    """A depth-averaged wet film of `substances` on `ground`, with strata (bottom first), an environment, a
    clock and two ledgers. `constitutive` satisfies goo.world.Constitutive. q: [S, H, W] volume per area (m)."""

    def __init__(self, constitutive, substances, ground, dx, environment=None, q=None, strata=(),
                 slope_bound=0.1, margin=3, solver="exact"):
        if solver not in ("exact", "iterative"):
            raise ValueError("solver: 'exact' (factorising, the reference) or 'iterative' (SPEC §3)")
        self.solver = solver
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
        self._strata = []                     # (box, q of the box, mass [S]): a stratum is stored on its bounding box
        self._zb = np.array(self.ground.z, np.float64)
        for st in strata:
            self._push_stratum(np.array(st, np.float64))
        self._shared = False
        self._version = 0                     # counts changes of the state (q, held, z_b, environment)
        self.env = environment if environment is not None else Environment()
        self.t = 0.0
        self.slope_bound, self.margin = float(slope_bound), int(margin)
        self.a = float(np.sqrt(self.rho.min() * self.rho.max()))
        self.matter = Ledger("matter", (S,), tuple(s.name for s in self.substances))
        self.energy = Ledger("energy")
        self._grid = Grid(H, W, self.dx)
        self._coef_flags = []
        self._store_coefficients(self._full, self._coefficients(self._full))
        self._dirty = None
        self.matter.post("origin", "wet", self._mass(self.q), "initial film")
        if np.any(self.ground.held):
            self.matter.post("origin", "ground", self._mass(self.ground.held), "initial pores")
        for k, st in enumerate(self._strata):
            self.matter.post("origin", f"stratum[{k}]", st[2], "initial stratum")
        self._Ftot = self._F_box(self._full)
        F0 = self.F()
        self.energy.post("work[origin]", "stored", sum(F0.values()), "initial free energy")
        self.receipt0 = Receipt("create", 0.0, 0.0, self.matter.since(0) + self.energy.since(0),
                                self._residuals(abs(sum(F0.values())), sum(F0.values()), 0.0,
                                                sum(abs(v) for v in F0.values()), H * W),
                                {k: (0.0, v) for k, v in F0.items()})

    # ---- state -------------------------------------------------------------------------------------------
    @property
    def film(self):
        return Film(self.substances, self.q)

    @property
    def _full(self):
        return (0, self.H, 0, self.W)

    @property
    def zb(self):
        """The relief under the wet film: the ground and the strata, m (kept, not recomputed)."""
        return self._zb

    @property
    def strata(self):
        """The strata as full [S, H, W] arrays, bottom first (built on demand from their boxes)."""
        out = []
        for box, qb, _ in self._strata:
            a = np.zeros((len(self.substances), self.H, self.W))
            a[_sl(box)] = qb
            out.append(a)
        return out

    def _push_stratum(self, full):
        nz = full.sum(0) > 0
        if nz.any():
            rows, cols = np.flatnonzero(nz.any(1)), np.flatnonzero(nz.any(0))
            box = (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)
        else:
            box = (0, 0, 0, 0)
        qb = full[_sl(box)].copy()
        self._strata.append((box, qb, self._mass(qb)))
        self._zb[_sl(box)[1:]] += qb.sum(0)

    def _own(self):
        """Copy on write: a fork shares its arrays until one of the two changes them."""
        if self._shared:
            self.q = self.q.copy()
            self.ground = self.ground.copy()
            self._zb = self._zb.copy()
            self._lagged = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self._lagged.items()}
            self._shared = False

    def _mass(self, q):
        return self.rho * np.asarray(q).sum((1, 2)) * self.dx ** 2

    def _mu_env(self):
        """The reservoirs' potentials [S] J/m^3, from the exchange map (SPEC §4.3), and which substances have one.
        They depend on the environment and the substances only, so they are read on a one-cell film."""
        S = len(self.substances)
        fn = getattr(self.con, "exchange", None)
        if fn is None:
            return np.zeros(S), np.zeros(S, bool)
        one = Film(self.substances, np.zeros((S, 1, 1)))
        v = unwrap(fn(one, self.env), "exchange", [])
        mu = np.asarray(v[1], np.float64).reshape(S)
        has = np.isfinite(mu)          # a reservoir of activity 0 has mu = -inf: no finite term, refused below
        return np.where(has, mu, 0.0), has

    # ---- lagged coefficients (DYNAMICS.md §2) -------------------------------------------------------------
    def _coefficients(self, box):
        """The lagged coefficients evaluated on a box of the panel (box-shaped), from the protocol."""
        flags = []
        sl = _sl(box)
        y0, y1, x0, x1 = box
        film = Film(self.substances, self.q[sl])
        hb, wb = y1 - y0, x1 - x0
        wet = film.h > 0
        gam = call(self.con, "surface_tension", film, flags=flags)
        if gam is None:
            gx, gy = np.zeros((hb, max(wb - 1, 0))), np.zeros((max(hb - 1, 0), wb))
        else:
            gam = np.asarray(gam, np.float64)
            gam = np.where(np.isfinite(gam), gam, 0.0)

            def face(gL, gR, wL, wR):
                return np.where(wL == wR, 0.5 * (gL + gR), np.where(wL, gL, gR))
            gx = face(gam[:, :-1], gam[:, 1:], wet[:, :-1], wet[:, 1:])
            gy = face(gam[:-1, :], gam[1:, :], wet[:-1, :], wet[1:, :])
        wv = call(self.con, "wetting", film, self._ground_box(box), flags=flags)
        if wv is None:
            Sw, ell = np.zeros((hb, wb)), np.ones((hb, wb))
        else:
            Sw, ell = (np.array(np.broadcast_to(np.asarray(x, np.float64), (hb, wb))) for x in wv[:2])
            if np.any(Sw > 0) or np.any((Sw != 0) & ~(ell > 0)):
                raise ValueError("wetting: spreading coefficient must be <= 0 with a positive length")
            ell = np.where(ell > 0, ell, 1.0)
        sv = call(self.con, "sorption", film, self._ground_box(box), flags=flags)
        S = len(self.substances)
        if sv is None or self.ground.capacity <= 0:
            P = np.zeros((S, hb, wb))
        else:
            P = np.array(np.broadcast_to(np.asarray(sv[0], np.float64), (S, hb, wb)))
        self._coef_flags = flags
        return dict(gx=gx, gy=gy, Sw=Sw, ell=ell, P=P, Pb=P.max(0) if S else np.zeros((hb, wb)))

    def _ground_box(self, box):
        """The ground on a box of the panel (views), for maps of (film, ground) evaluated on that box."""
        g = self.ground
        sl = _sl(box)
        return Ground(g.z[sl[1:]], g.solid, g.porosity, g.pore_radius, g.depth, g.substances, g.held[sl])

    def _lag_box(self, box, lag=None):
        """The panel's lagged coefficients on a box (views)."""
        c = self._lagged if lag is None else lag
        y0, y1, x0, x1 = box
        return dict(gx=c["gx"][y0:y1, x0:max(x1 - 1, x0)], gy=c["gy"][y0:max(y1 - 1, y0), x0:x1],
                    Sw=c["Sw"][y0:y1, x0:x1], ell=c["ell"][y0:y1, x0:x1], P=c["P"][:, y0:y1, x0:x1],
                    Pb=c["Pb"][y0:y1, x0:x1])

    def _store_coefficients(self, box, cb):
        if box == self._full:
            self._lagged = cb
        else:
            for k, v in self._lag_box(box).items():
                v[...] = cb[k]
        self._has_wet = bool(np.any(self._lagged["Sw"] != 0))
        self._has_pore = bool(np.any(self._lagged["P"] != 0)) and self.ground.capacity > 0

    # ---- free energy -------------------------------------------------------------------------------------
    def F(self):
        """F by term, J (DYNAMICS.md §2): kept, and changed by each operation's increment over the region it
        touched (outside it the state, and so F's density, is unchanged). F_recomputed() sums it afresh."""
        return dict(self._Ftot)

    def F_recomputed(self):
        return self._F_box(self._full)

    LAGGED_TERMS = ("capillary", "wetting", "sorption")       # the terms whose coefficients are lagged (§2)

    def _F_box(self, box, q=None, held=None, zb=None, lag=None, terms=TERMS):
        """F by term over the cells of a box and the faces inside it, J (only the terms named)."""
        sl = _sl(box)
        y0, y1, x0, x1 = box
        q = self.q[sl] if q is None else q
        held = self.ground.held[sl] if held is None else held
        zb = self._zb[sl[1:]] if zb is None else zb
        c = self._lag_box(box, lag)
        dx2 = self.dx ** 2
        h, m = q.sum(0), np.tensordot(self.rho, q, 1)
        gxv, gyv, gn = self.env.gravity
        X = ((x0 + np.arange(x1 - x0)) + 0.5) * self.dx
        Y = (((y0 + np.arange(y1 - y0)) + 0.5) * self.dx)[:, None]
        out = {}
        if "gravity" in terms:
            out["gravity"] = dx2 * np.sum(gn * m * (zb + 0.5 * h) - m * (gxv * X[None, :] + gyv * Y))
        if "capillary" in terms:
            s = zb + h
            cap = 0.0
            for gam, a_, b_ in ((c["gx"], np.s_[:, 1:], np.s_[:, :-1]), (c["gy"], np.s_[1:, :], np.s_[:-1, :])):
                dh, ds, dz = h[a_] - h[b_], s[a_] - s[b_], zb[a_] - zb[b_]
                cap += np.sum(0.5 * gam * dh * (ds + dz))
            out["capillary"] = cap
        if "wetting" in terms:
            out["wetting"] = dx2 * np.sum((-c["Sw"]) * (-np.expm1(-h / c["ell"]))) if self._has_wet else 0.0
        if "mixing" in terms:
            wetc = np.flatnonzero(h.ravel() > 0)
            mixf = 0.0
            if wetc.size:
                mx = mixing_potentials(self._ctx_stub(), q.reshape(len(self.substances), -1)[:, wetc])
                if mx is not None:
                    mixf = dx2 * np.sum(mx[0] + mx[3])
            out["mixing"] = mixf
        if "exchange" in terms:
            menv, _ = self._mu_env()
            out["exchange"] = -dx2 * np.sum(menv * (q.sum((1, 2)) + held.sum((1, 2))))
        if "sorption" in terms:
            pore = 0.0
            if self._has_pore:
                C = self.ground.capacity
                pore = dx2 * np.sum(-(c["P"] * held).sum(0) + c["Pb"] * C * pore_B(held.sum(0) / C))
            out["sorption"] = pore
        return {k: float(v) for k, v in out.items()}

    def _ctx_stub(self):
        menv, has = self._mu_env()
        return Ctx(self._grid, self.substances, self.rho, self.con, self.env, self.ground, None, None, None, None,
                   None, None, None, 0.0, self.a, menv, has, [])

    # ---- the step ----------------------------------------------------------------------------------------
    def step(self, dt, rate=None, window="auto", solver=None):
        """Advance by dt (s). rate: None, a scalar or an [H, W] field >= 0 multiplying every Psi* (SPEC §8).
        window: 'auto' (the wet cells' bounding box plus a margin) or 'panel'. Returns a Receipt. Its work is on
        the region R that can change (the window and what changed in the last step); F, the coefficients and the
        slope are evaluated there, and the rest of the panel is not read."""
        dt = float(dt)
        if not dt > 0:
            raise ValueError("dt must be positive")
        r = np.ones((self.H, self.W)) if rate is None else np.array(np.broadcast_to(np.asarray(rate, np.float64), (self.H, self.W)))
        if not np.all(np.isfinite(r)):
            raise ValueError("rate must be finite")
        if np.any(r < 0):
            raise ValueError("rate < 0 reverses the world (SPEC §8, impossible); reverse steps are not implemented yet")
        self._own()
        self._solver_now = self.solver if solver is None else solver
        if self._solver_now not in ("exact", "iterative"):
            raise ValueError("solver: 'exact' or 'iterative'")
        self._stats = {}
        # rest memo (DYNAMICS.md §9a): a step that every substep left on an exit decided on the old state alone is a
        # step of every size; until the state, the environment, the clock's rate or the solver changes, the next step
        # is the same rest step and is not recomputed. The key's version counts every change of q, held, z_b and the
        # environment (every act goes through a method that counts it); the arrays' identities guard a replacement.
        rkey = None if rate is None else (hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest()
                                          if np.ndim(rate) else float(rate))
        key = (self._version, id(self.q), id(self.ground.held), window, self._solver_now, rkey)
        memo = getattr(self, "_rest", None)
        if memo is not None and memo[0] == key:
            return self._rest_step(dt, r, rate, memo)
        mm, me = self.matter.mark(), self.energy.mark()
        t0 = self.t
        flags = []
        sb = float(self.energy.balance("stored"))
        Fb0 = self.F()
        R = _union(self._window(window), self._dirty)
        rest = True
        acc = {k: 0.0 for k in MECHANISMS}
        acc.update(num_time=0.0, num_transport=0.0, moved=0.0, pieces=0.0, num_local=0.0, drained=0.0, num_constraint=0.0)
        e = np.zeros(len(self.substances)); a = np.zeros(len(self.substances))
        rnd = 0.0
        chg = 0.0
        relabel = 0.0
        dF = 0.0
        sl = 0.0
        self._Fopen = None
        if R is not None:
            relabel, rnd = self._relabel(R, flags)
        Fb = self.F()
        if rate is not None and not np.all(r == 1.0):
            flags.append(Flag("privilege", "painter", f"clock rate in [{r.min():g}, {r.max():g}]; amount: integral of r over cells and time, m^2 s",
                              float(dt * np.sum(r) * self.dx ** 2)))
        ctxflags = []
        if R is not None and np.any(r > 0):
            self._saved = [R, self.q[_sl(R)].copy(), self.ground.held[_sl(R)].copy()]
            try:
                reps = self._advance(dt, r, window, 0, ctxflags)
            except StepFailed:
                R0, q0, h0 = self._saved                # undo every accepted piece: the step is all or nothing
                self.q[_sl(R0)] = q0
                self.ground.held[_sl(R0)] = h0
                del self._saved
                self._version += 1
                raise
            rest = len(reps) == 1 and bool(reps[0]["rest"])
            for rep in reps:
                for k in MECHANISMS:
                    acc[k] += rep[k]
                acc["num_time"] += rep["num_time"]; acc["num_transport"] += rep["num_transport"]
                acc["num_local"] += rep["num_local"]; acc["drained"] += rep["drained"]
                acc["num_constraint"] += rep["num_constraint"]
                acc["moved"] += rep["moved"]
                acc["pieces"] += 1
                e += rep["e"]; a += rep["a"]
            R, q0, h0 = self._saved
            del self._saved
            # the largest change of any amount in a cell, relative to the cell's thickness (before or after, the
            # larger, and at least a tenth of the thickest cell of the region, so that a front entering bare cells
            # is measured against the film and not against the nothing it enters): advance's bound on change per
            # step and the scale of its error estimate (§9a). Cells outside the wet region are 0 either way, so the
            # number is the same in a window and on the panel.
            q1 = self.q[_sl(R)]
            hb, ha = q0.sum(0), q1.sum(0)
            href = np.maximum(np.maximum(hb, ha), 0.1 * max(float(hb.max(initial=0.0)), float(ha.max(initial=0.0))))
            with np.errstate(divide="ignore", invalid="ignore"):
                rc = np.where(href > 0, np.max(np.abs(q1 - q0), axis=0) / href, 0.0)
            chg = float(np.max(rc)) if rc.size else 0.0
            op = getattr(self, "_Fopen", None)
            before = dict(op[1]) if op is not None and op[0] == R else self._F_box(R, q=q0, held=h0)
            after = self._F_box(R)
            self._version += 1
            self._Fcache = (R, self._version, after)
            for k in TERMS:
                self._Ftot[k] += after[k] - before[k]
                dF += after[k] - before[k]
            rnd += sum(abs(v) for v in before.values()) + sum(abs(v) for v in after.values())
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
        if R is not None:
            sl = self.slope(R)
            self._dirty = R
        if sl > self.slope_bound:
            flags.append(Flag("slope", "film", f"max surface slope {sl:.3g} above the bound {self.slope_bound:g}", sl))
        ncell = 0 if R is None else (R[1] - R[0]) * (R[3] - R[2])
        res = self._residuals(acc["moved"] + abs(relabel), dF + relabel, sb, rnd, ncell)
        report = {f"D {k}": float(acc[k]) for k in MECHANISMS}
        if self._solver_now == "iterative":
            report["max solver residual"] = float(self._stats.get("solver residual", 0.0))
            report["solver iterations"] = float(self._stats.get("solver iterations", 0))
            report["solver tolerance"] = RTOL
            report["solver fallbacks"] = float(self._stats.get("solver fallbacks", 0))

        # how diffusion was solved (DYNAMICS.md §4.2a): by the ray, or by the Newton where the ray fell short
        for k in ("diffusion rays", "diffusion fallbacks", "max diffusion ray shortfall"):
            report[k] = float(self._stats.get(k, 0.0))
        nwet = 0 if R is None else int(np.count_nonzero(self.q[_sl(R)].sum(0) > 0))     # the same in a window
        # the time part's sign is the construction's (§5); the transport part's is not established for a mixing f
        # with a concave interaction part (a stated gap, §5 "Flow and mixing"), so only the time part is judged here
        dn = acc["num_time"]
        if dn < -1e-12 * max(acc["moved"], 1e-300) - 16 * np.finfo(float).eps * rnd:
            flags.append(Flag("solver", self._solver_now,
                              "D_num negative beyond rounding" + (": its sign holds only to the solver's tolerance"
                                                                  if self._solver_now == "iterative" else ""), dn))
        report.update({"max energy floor": float(nwet * k_B * self.env.T), "D_num time": float(acc["num_time"]), "D_num transport": float(acc["num_transport"]),
                       "D_num local": float(acc["num_local"]), "D_num constraint": float(acc["num_constraint"]), "drained cells": float(acc["drained"]),
                       "energy moved": float(acc["moved"]), "max slope": sl, "pieces": acc["pieces"],
                       "max relative change": chg,
                       "rate integral": float(dt * np.sum(r) * self.dx ** 2)})
        out = Receipt("step", t0, self.t, self.matter.since(mm) + self.energy.since(me), res,
                      {k: (Fb[k], Fa[k]) for k in TERMS}, tuple(flags), report)
        self._rest = None
        if rest and not out.entries and relabel == 0.0:
            # a rest step: remember what it reported, except what depends on dt (the rate's integral, the privilege)
            key = (self._version, id(self.q), id(self.ground.held), window, self._solver_now, rkey)   # after the step
            self._rest = (key, tuple(f for f in flags if not (f.kind == "privilege" and f.subject == "painter")),
                          dict(report), {k: (Fb[k], Fa[k]) for k in TERMS}, res, R)
        return out

    def _rest_step(self, dt, r, rate, memo):
        """The step at a memoised rest state: what step() would compute, without computing it (O4: the state is
        returned bit for bit, nothing is posted). Tested against the computed step receipt by receipt."""
        _, flags, report, F, res, R = memo
        t0 = self.t
        self.t = t0 + dt
        flags = list(flags)
        if rate is not None and not np.all(r == 1.0):
            flags.insert(0, Flag("privilege", "painter", f"clock rate in [{r.min():g}, {r.max():g}]; amount: integral of r over cells and time, m^2 s",
                                 float(dt * np.sum(r) * self.dx ** 2)))
        report = dict(report)
        report["rate integral"] = float(dt * np.sum(r) * self.dx ** 2)
        self._dirty = R
        return Receipt("step", t0, self.t, (), dict(res), dict(F), tuple(flags), report)

    ADVANCE_TOL = 1e-3          # D_num (time part) per step, as a share of the energy the step moved
    ADVANCE_DT0 = 1e-3          # s: the first step when the world has no step-size history

    ADVANCE_MIN = 1e-9          # a rejected step shrinks fourfold; below this share of the duration advance stops
    ADVANCE_CHANGE = 0.1        # a step that changes no amount by more than this share of its cell's thickness is
                                # held to its error in proportion: e is scaled by (change / ADVANCE_CHANGE)
    ADVANCE_CHANGE_MAX = 0.25   # the bound on change per step: above it the next step halves, and dt only doubles
                                # when the doubled step's change would stay below it

    def advance(self, duration, rate=None, solver=None, dt=None, tol=None):
        """Cover `duration` seconds of world time with as many steps as needed; return the summed Receipt.

        Step-size rule (deterministic: it reads only receipts, which are the same in a window and on the panel):
        the time part of D_num is the Bregman divergence of the step, O(dt^2), so e = D_num time / energy moved
        is its error estimate. After a step with e <= tol/4 and no halving the next dt doubles; with e > tol it
        halves; a step that needed halvings is followed by dt / (its pieces). The last step lands on the end.
        A step the world cannot take (StepFailed: it is left untouched) is retried at a quarter of its size; if that
        falls below ADVANCE_MIN of the duration, advance stops where it is and returns what it did with a `solver`
        flag (t1 of the receipt says how far it got). rate: as in step (a scalar or an [H, W] field). The size
        reached is kept for the next call."""
        T = float(duration)
        if not T > 0:
            raise ValueError("duration must be positive")
        tol = self.ADVANCE_TOL if tol is None else float(tol)
        h = float(dt) if dt is not None else getattr(self, "_dt_next", self.ADVANCE_DT0)
        t_start = self.t
        t_end = self.t + T
        total = None
        nsteps, nfail = 0, 0
        stopped = None
        last, hold = getattr(self, "_adv_state", (None, None))
        power = 0.0                                  # energy moved per second, averaged over the recent steps
        priv = 0.0
        while self.t < t_end:
            left = t_end - self.t
            take = min(h, left)
            if left - take < 1e-6 * take:            # do not leave a sliver
                take = left
            try:
                r = self.step(take, rate=rate, solver=solver)
            except StepFailed as exc:
                nfail += 1
                h = take / 4
                if h < self.ADVANCE_MIN * T:
                    stopped = (take, str(exc))
                    break
                continue
            except RuntimeError as exc:                # not the world's own refusal: nothing says what it left behind
                raise RuntimeError(f"advance: could not take a step of {take:.4g} s at t = {self.t:.6g} s: {exc}") from exc
            nsteps += 1
            priv += sum(f.amount for f in r.flags if f.kind == "privilege" and f.subject == "painter")
            total = r if total is None else total + r
            # the estimate is read against the energy the step moved, floored at k_B T per cell of its region (below
            # that, D_num is rounding and says nothing about dt); only a full step (not the remainder that lands
            # on the end) sets the next size
            # (and against the power of the recent steps times this one's size: a step during which little happens
            # is not held to a relative error on the little it moved)
            mv = r.report.get("energy moved", 0.0)
            moved = max(mv, r.report.get("max energy floor", 0.0), 0.5 * power * take)
            power = mv / take if power == 0.0 else 0.5 * (power + mv / take)
            # a cell drained dry is the local step's constraint: its share of D_num is O(1) in dt (a smaller dt only
            # postpones the drain), so the local part is left out of the estimate when any cell drained
            dn = r.report.get("D_num time", 0.0) - r.report.get("D_num constraint", 0.0)
            if r.report.get("drained cells", 0.0) > 0:
                dn -= r.report.get("D_num local", 0.0)
            e = dn / moved if moved > 0 else 0.0
            # The noise floor (DYNAMICS.md §9a): e is a relative error of the energy the step moved, which does not
            # vanish as a decaying mode dies out (implicit Euler's D_num / energy moved is about lambda dt / 2 at
            # every amplitude), and near equilibrium both are rounding. What the painter sees is the state, so a step
            # that changes no amount by more than ADVANCE_CHANGE of its cell is held to its error only in proportion
            # to how much it changed: an absolute tolerance on the state, ADVANCE_CHANGE * tol.
            chg = r.report.get("max relative change", 0.0)
            e *= min(1.0, chg / self.ADVANCE_CHANGE)
            pieces = r.report.get("pieces", 1.0) or 1.0
            full = take >= h * (1 - 1e-9)
            # A shrink that does not lower e by a quarter shows an error that is not the time step's (a constraint's
            # reaction, a drained or saturated cell, whose share of D_num is O(1) in dt): the size goes back to the
            # one before and is held there until e falls below tol/4.
            if pieces > 1:
                h = min(h, take / pieces)
            elif e > tol and full:
                shrunk = last is not None and abs(last[0] - 2 * take) <= 1e-9 * take
                if shrunk and e > 0.75 * last[1]:
                    hold = 2 * take
                    h = hold
                elif hold is not None and take / 2 < hold * (1 - 1e-9):
                    h = take
                else:
                    h = take / 2
            elif chg > self.ADVANCE_CHANGE_MAX and full:
                h = take / 2                            # the bound on change per step
            elif e <= tol / 4 and full and 2 * chg <= self.ADVANCE_CHANGE_MAX:
                hold = None
                h = 2 * h
            if full:
                last = (take, e)
            if self.t >= t_end or abs(self.t - t_end) <= 1e-12 * max(abs(t_end), 1.0):
                self.t = t_end
                break
        self._dt_next = h
        self._adv_state = (last, hold)
        # equal flags of equal steps would merge when receipts add: the clock's privilege is summed over every step
        flags = () if total is None else tuple(f for f in total.flags if not (f.kind == "privilege" and f.subject == "painter"))
        if priv:
            flags = (Flag("privilege", "painter", f"clock rate != 1 in {nsteps} steps; amount: integral of r over cells and "
                          "time, m^2 s", float(priv)),) + flags
        if stopped is not None:
            flags += (Flag("solver", "advance", f"stopped at t = {self.t:.6g} s of [{t_start:.6g}, {t_end:.6g}] s: a "
                           f"step of {stopped[0]:.3g} s could not be accepted ({stopped[1]})", float(t_end - self.t)),)
        if total is None:
            return Receipt("advance", t_start, self.t, (), {"matter": 0.0, "energy": 0.0}, {}, flags,
                           {"steps": 0.0, "rejected steps": float(nfail)})
        report = dict(total.report)
        report["steps"] = float(nsteps)
        report["rejected steps"] = float(nfail)
        return Receipt("advance", total.t0, self.t, total.entries, total.residual, total.F, flags, report)

    def _flags(self, pairs):
        seen, out = set(), []
        for kind, name, detail in list(self._coef_flags) + list(pairs):
            if (kind, name, detail) not in seen:
                seen.add((kind, name, detail)); out.append(Flag(kind, name, detail))
        return out

    def _relabel(self, R, flags):
        """Re-evaluate the lagged coefficients on R; book the change of F at fixed state (DYNAMICS.md §2).
        Returns (the change, the magnitude of the F values it was computed from)."""
        # only the lagged terms change at fixed state; the others are the same numbers before and after (so their
        # difference is exactly 0) and are read once, from the last step's closing evaluation if the state and the
        # box are those it was computed on
        Fo = self._F_box(R, terms=self.LAGGED_TERMS)
        self._store_coefficients(R, self._coefficients(R))
        cache = getattr(self, "_Fcache", None)
        rest = tuple(k for k in TERMS if k not in self.LAGGED_TERMS)
        if cache is not None and cache[0] == R and cache[1] == self._version:
            Fn = {k: cache[2][k] for k in rest}
        else:
            Fn = self._F_box(R, terms=rest)
        Fn.update(self._F_box(R, terms=self.LAGGED_TERMS))
        Fo.update({k: Fn[k] for k in rest})
        self._Fopen = (R, Fn)                     # F on R at the step's start, with the new coefficients
        d = 0.0
        for k in TERMS:
            self._Ftot[k] += Fn[k] - Fo[k]
            d += Fn[k] - Fo[k]
        if d != 0.0:
            self.energy.post("work[constitutive]", "stored", d, "lagged coefficients re-evaluated")
            flags.append(Flag("out_of_domain", "lagged coefficient",
                              "a coefficient of F changed with the state; Marangoni-type coupling not modelled", d))
        return d, sum(abs(v) for v in Fo.values()) + sum(abs(v) for v in Fn.values())

    def _advance(self, dt, r, window, depth, ctxflags):
        try:
            return [self._piece(dt, r, window, ctxflags)]
        except (Reject, SolveFailed, np.linalg.LinAlgError, FloatingPointError):
            if depth >= MAX_HALVINGS:
                raise StepFailed(f"step could not be accepted after {MAX_HALVINGS} halvings")
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
        rep.update(num_time=0.0, num_transport=0.0, moved=0.0, e=np.zeros(S), a=np.zeros(S), num_local=0.0, drained=0.0, num_constraint=0.0)
        rep["rest"] = True
        if not np.any(self.q):
            return rep
        win = self._window(window)
        if win is None:
            return rep
        sv = getattr(self, "_saved", None)
        if sv is not None and not _inside(win, sv[0]):
            # the window reaches beyond the region saved at the step's start: outside that region nothing has
            # changed yet, so the state there is still the start's and is saved now
            U = _union(win, sv[0])
            nq, nh = self.q[_sl(U)].copy(), self.ground.held[_sl(U)].copy()
            o = (sv[0][0] - U[0], sv[0][1] - U[0], sv[0][2] - U[2], sv[0][3] - U[2])
            nq[_sl(o)] = sv[1]; nh[_sl(o)] = sv[2]
            self._saved = [U, nq, nh]
        y0, y1, x0, x1 = win
        Hh, Ww = y1 - y0, x1 - x0
        G = Grid(Hh, Ww, self.dx, y0, x0)
        c = self._lagged
        flat = lambda arr: None if arr is None else np.ascontiguousarray(arr[..., y0:y1, x0:x1]).reshape(arr.shape[:-2] + (-1,))
        menv, has = self._mu_env()
        flags = []
        wetting, pores = self._has_wet, self._has_pore          # decided on the panel, so a window decides alike
        ctx = Ctx(G, self.substances, self.rho, self.con, self.env, self._ground_box(win), flat(self._zb), flat(r),
                  panel_faces(c["gx"], c["gy"], y0, y1, x0, x1), flat(c["Sw"]) if wetting else None,
                  flat(c["ell"]) if wetting else None, flat(c["P"]) if pores else None,
                  flat(c["Pb"]) if pores else None, self.ground.capacity, self.a, menv, has, flags,
                  self._solver_now, {}, [])
        q = flat(self.q).copy()
        held = flat(self.ground.held).copy()
        pst = ctx.stats
        q1, fr = flow(ctx, q, dt)
        q2, dr = diffuse(ctx, q1, dt)
        q3, held3, lr = local(ctx, q2, held, dt)
        if not (np.all(np.isfinite(q3)) and np.all(np.isfinite(held3))):
            raise Reject("non-finite state")
        # commit
        self.q[:, y0:y1, x0:x1] = q3.reshape(S, Hh, Ww)
        self.ground.held[:, y0:y1, x0:x1] = held3.reshape(S, Hh, Ww)
        ctxflags.extend(flags)
        for k, v in pst.items():                    # only an accepted piece's solves count
            self._stats[k] = (max(self._stats.get(k, 0.0), v) if (k == "solver residual" or k.startswith("max "))
                              else self._stats.get(k, 0) + v)
        rep["viscous"], rep["plastic"] = fr["D_viscous"], fr["D_plastic"]
        rep["diffusion"] = dr["D_diffusion"]
        rep["exchange"], rep["sorption"] = lr["D_exchange"], lr["D_sorption"]
        rep["num_time"] = ((-fr["dE"] - fr["D_viscous"] - fr["D_plastic"]) + (-dr["dF"] - dr["D_diffusion"])
                           + (-lr["dF"] - lr["D_exchange"] - lr["D_sorption"]))
        rep["num_local"] = -lr["dF"] - lr["D_exchange"] - lr["D_sorption"]
        rep["num_constraint"] = fr["D_constraint"] + fr["D_wetting"]
        rep["drained"] = float(lr["drained"])
        rep["num_transport"] = -fr["dF_mix"]
        rep["moved"] = (abs(fr["dE"]) + abs(fr["dF_mix"]) + fr["D_viscous"] + fr["D_plastic"] + abs(dr["dF"])
                        + dr["D_diffusion"] + abs(lr["dF"]) + lr["D_exchange"] + lr["D_sorption"])
        rep["e"], rep["a"] = lr["e"], lr["a"]
        rep["rest"] = bool(fr.get("rest") and dr.get("rest") and lr.get("rest"))
        return rep

    # ---- residuals, flags --------------------------------------------------------------------------------
    def _residuals(self, moved, dF, sb, mag, ncells):
        """Closure residuals of an operation. Matter: every account against the state, over the mass present.
        Energy: the change the ledger booked to `stored` against the change of F recomputed on the region it
        touched (dF), beyond the rounding of those F values (a few ulps of their magnitude `mag`), over the energy
        the operation moved, floored at k_B T per cell of the region (below which no number in F means anything)."""
        state = {"wet": self._mass(self.q), "ground": self._mass(self.ground.held)}
        for k, st in enumerate(self._strata):
            state[f"stratum[{k}]"] = st[2]
        scale = float(sum(np.sum(v) for v in state.values())) or 1.0
        rm = max(float(np.max(np.abs(self.matter.balance(k) - v))) for k, v in state.items()) / scale
        dS = float(self.energy.balance("stored")) - sb
        floor = max(ncells, 1) * k_B * self.env.T
        rnd = 16 * np.finfo(float).eps * mag
        re = max(0.0, abs(dS - dF) - rnd) / max(moved, abs(dF), floor)
        return {"matter": rm, "energy": re}

    def slope(self, box=None):
        """Largest surface slope |grad s| over faces touching a wet cell (in a box of the panel, or all of it)."""
        box = self._full if box is None else box
        sl = _sl(box)
        h = self.q[sl].sum(0)
        s, wet = self._zb[sl[1:]] + h, h > 0
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
        self._own()
        mm, me, t = self.matter.mark(), self.energy.mark(), self.t
        box = self._box_of(arr.sum(0) > 0)
        Fb = self.F(); sb = float(self.energy.balance("stored"))
        d, mag = self._change(box, lambda: self.q.__iadd__(arr))
        Fa = self.F()
        self.matter.post(src, "wet", self._mass(arr), "added")
        self.energy.post(f"work[{src}]", "stored", d, "free energy brought")
        return Receipt("add", t, t, self.matter.since(mm) + self.energy.since(me),
                       self._residuals(abs(d), d, sb, mag, self._ncells(box)),
                       {k: (Fb[k], Fa[k]) for k in TERMS}, tuple(self._flags([])))

    def _box_of(self, mask):
        """The bounding box of a mask, grown by one cell (the faces around it), or None."""
        if not mask.any():
            return None
        rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        return (max(0, int(rows[0]) - 1), min(self.H, int(rows[-1]) + 2), max(0, int(cols[0]) - 1),
                min(self.W, int(cols[-1]) + 2))

    @staticmethod
    def _ncells(box):
        return 0 if box is None else (box[1] - box[0]) * (box[3] - box[2])

    def _change(self, box, act):
        """Run `act` (which changes the state only inside `box`), update F's totals by its increment on the box,
        and mark the box for the next coefficient refresh. Returns (dF, the magnitude of the F values used)."""
        if box is None:
            act()
            return 0.0, 0.0
        before = self._F_box(box)
        act()
        self._version += 1
        after = self._F_box(box)
        d = 0.0
        for k in TERMS:
            self._Ftot[k] += after[k] - before[k]
            d += after[k] - before[k]
        self._dirty = _union(self._dirty, box)
        return d, sum(abs(v) for v in before.values()) + sum(abs(v) for v in after.values())

    def retire(self, mask=None, substances=None):
        """Turn the given substances (default all) of the wet film in `mask` (default every wet cell) into a new
        stratum on top of the others (SPEC §6). Matter wet -> stratum[k]; the change of the live F stored -> frozen."""
        self._own()
        sel = np.ones(len(self.substances), bool) if substances is None else np.isin(
            [s.name for s in self.substances], [getattr(x, "name", x) for x in substances])
        mk = (self.q.sum(0) > 0) if mask is None else np.asarray(mask, bool)
        moved = np.where(sel[:, None, None] & mk[None], self.q, 0.0)
        mm, me, t = self.matter.mark(), self.energy.mark(), self.t
        Fb = self.F(); sb = float(self.energy.balance("stored"))
        k = len(self._strata)

        def act():
            self.q -= moved
            self.q[sel[:, None, None] & mk[None]] = 0.0
            self._push_stratum(moved)
        d, mag = self._change(self._box_of(moved.sum(0) > 0), act)
        Fa = self.F()
        self.matter.post("wet", f"stratum[{k}]", self._strata[k][2], "retired")
        self.energy.post("stored", "frozen", -d, f"retired into stratum[{k}]")
        return Receipt("retire", t, t, self.matter.since(mm) + self.energy.since(me),
                       self._residuals(abs(d), d, sb, mag, self._ncells(self._box_of(moved.sum(0) > 0))),
                       {kk: (Fb[kk], Fa[kk]) for kk in TERMS}, ())

    def settle(self):
        """Retire, as a law, every wet cell whose film can no longer flow: its viscosity is +inf (no liquid left,
        or particles jammed), so its mobility is exactly zero and it has no lateral freedom left to lose. All of
        such a cell's matter becomes one new stratum (SPEC §6). A film that still holds a liquid is left as it is,
        however slow. Returns the retirement's Receipt (an empty one if nothing settles)."""
        wet = self.q.sum(0) > 0
        box = self._box_of(wet)
        t = self.t
        nwet = float(wet.sum())
        if box is None:
            return Receipt("settle", t, t, report={"cells retired": 0.0, "cells still wet": 0.0})
        flags = []
        eta = call(self.con, "viscosity", Film(self.substances, self.q[_sl(box)]), flags=flags)
        stuck = np.zeros((self.H, self.W), bool)
        stuck[_sl(box)[1:]] = np.isposinf(np.asarray(eta, np.float64)) & wet[_sl(box)[1:]]
        if not stuck.any():
            return Receipt("settle", t, t, report={"cells retired": 0.0, "cells still wet": nwet})
        r = self.retire(mask=stuck)
        rep = dict(r.report)
        rep.update({"cells retired": float(stuck.sum()), "cells still wet": nwet - float(stuck.sum())})
        return Receipt("settle", r.t0, r.t1, r.entries, r.residual, r.F, r.flags + tuple(self._flags([])), rep)

    def set_environment(self, environment):
        """Change the reservoirs: an act whose ΔF is work by the environment's port. It changes F everywhere, so
        F is evaluated on the whole panel."""
        mm, me, t = self.matter.mark(), self.energy.mark(), self.t
        Fb = self.F(); sb = float(self.energy.balance("stored"))
        before = self._F_box(self._full)
        self.env = environment
        self._version += 1
        after = self._F_box(self._full)
        d = 0.0
        for k in TERMS:
            self._Ftot[k] += after[k] - before[k]
            d += after[k] - before[k]
        Fa = self.F()
        self.energy.post("work[environment]", "stored", d, "environment changed")
        mag = sum(abs(v) for v in before.values()) + sum(abs(v) for v in after.values())
        return Receipt("environment", t, t, self.energy.since(me), self._residuals(abs(d), d, sb, mag, self.H * self.W),
                       {k: (Fb[k], Fa[k]) for k in TERMS}, ())

    # ---- forks, looks, digests ---------------------------------------------------------------------------
    def fork(self):
        """An independent world that continues identically (O6). Arrays are shared, copy on write: each of the two
        copies them the first time it changes them, so a fork costs the ledgers' bookkeeping, not the panel."""
        new = copy.copy(self)
        self._shared = new._shared = True
        new._strata = list(self._strata)                 # strata are immutable once retired
        new.matter, new.energy = self.matter.fork(), self.energy.fork()
        new._Ftot = dict(self._Ftot)
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

    def _hash_state(self, h):
        for a in (self.q, self.ground.z, self.ground.held):
            h.update(np.ascontiguousarray(a).tobytes())
        for box, qb, _ in self._strata:
            h.update(repr(box).encode()); h.update(np.ascontiguousarray(qb).tobytes())

    def digest(self):
        h = hashlib.sha256(repr((self.t, self.env.T, self.env.gravity, sorted(self.env.activity.items()),
                                   self.env.boundary_layer)).encode())
        self._hash_state(h)
        h.update(self.matter.digest().encode()); h.update(self.energy.digest().encode())
        return h.hexdigest()

    def state_digest(self):
        """Digest of the state arrays alone (no ledgers)."""
        h = hashlib.sha256(repr(self.t).encode())
        self._hash_state(h)
        return h.hexdigest()
