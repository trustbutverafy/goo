"""The bench: a World and the bodies coupled to it (SPEC §8; docs/PORTS.md).

Every act is a method that returns one goo.ledger.Receipt: the world's receipts of the act (adds, steps, retirements,
environment changes) summed, with the bench's own entries (vessels filled, overspray, spill, what tapes caught), a
matter closure over every account the act could touch (the world's and the bodies'), and privileges consolidated
into one flag with the integral of the clock rate.

What the world lacks and how it is worked around is in docs/PORTS.md §8. In short:
  * World.advance / World.settle are used when the world has them; otherwise a fixed rule here (documented).
  * World.step is called with window="panel": with a porous ground and a window smaller than the panel,
    World._coefficients passes the whole ground with a windowed film to `sorption`, which fails to broadcast.
  * Tapes are walls by the privileged clock (rate 0 in their cells); the world has no walls yet.
"""
import copy
import inspect
import time
import numpy as np
from goo.ledger import Receipt, Flag
from goo.film import Film
from goo.units import g_0
from goo.world.protocol import call
from goo.ports.paint import Paint, Vessel, PortError
from goo.ports import geometry as G

RESERVOIRS = ("overspray", "spill")
WALL_NOTE = ("tape held as cells of clock rate 0: the privileged clock doing a wall's job until the world has "
             "real walls (nothing crosses into or out of a held cell)")


class Tape:
    """A mask laid on the panel: it shields its cells from every source (what would land there is booked to its own
    account, body[name], and leaves with it when peeled) and, by the clock, nothing flows into its cells."""

    def __init__(self, name, mask, S):
        self.name, self.mask, self.mass, self.on = name, mask, np.zeros(S), True

    @property
    def account(self):
        return f"body[{self.name}]"


class Bench:
    def __init__(self, world):
        self.world = world
        self.subs = world.substances
        self.names = tuple(s.name for s in self.subs)
        self.rho = np.array([s.rho for s in self.subs], float)
        self.paints = {}
        self.vessels = {}
        self.tapes = {}                       # name -> Tape (peeled ones stay, on=False: their account remains)
        self.held = np.zeros((world.H, world.W), bool)
        self.reservoirs = {k: np.zeros(len(self.subs)) for k in RESERVOIRS}
        self.wall_seconds = 0.0

    # ---- shape --------------------------------------------------------------------------------------------
    @property
    def H(self):
        return self.world.H

    @property
    def W(self):
        return self.world.W

    @property
    def dx(self):
        return self.world.dx

    def region(self, desc, units="cells"):
        return G.region(desc, self.H, self.W, self.dx, units)

    def path(self, desc, units="cells"):
        return G.path(desc, self.dx, units)

    def fork(self):
        new = copy.copy(self)
        new.world = self.world.fork()
        new.paints = dict(self.paints)
        new.vessels = {k: Vessel(v.name, v.paint, v.mass.copy(), v.rho, v.filled) for k, v in self.vessels.items()}
        new.tapes = {}
        for k, t in self.tapes.items():
            n = Tape(t.name, t.mask, len(self.subs)); n.mass = t.mass.copy(); n.on = t.on
            new.tapes[k] = n
        new.held = self.held.copy()
        new.reservoirs = {k: v.copy() for k, v in self.reservoirs.items()}
        return new

    # ---- receipts -----------------------------------------------------------------------------------------
    def _open(self):
        w = self.world
        return dict(mm=w.matter.mark(), me=w.energy.mark(), t0=w.t, F=w.F(), parts=[], report={}, flags=[],
                    wall=time.perf_counter())

    def _close(self, op, ctx):
        w = self.world
        parts = ctx["parts"]
        residual = {"matter": 0.0, "energy": 0.0}
        flags, report = [], {}
        priv = 0.0
        npriv = 0
        seen = set()
        for r in parts:
            for k, v in r.residual.items():
                residual[k] = max(residual.get(k, 0.0), float(v))
            for f in r.flags:
                if f.kind == "privilege" and f.subject == "painter":
                    priv += f.amount; npriv += 1
                elif (f.kind, f.subject, f.detail) not in seen:
                    seen.add((f.kind, f.subject, f.detail))
                    flags.append(f)
            for k, v in r.report.items():
                report[k] = v if k not in report else (max(report[k], v) if k.startswith("max ") else report[k] + v)
        if npriv:
            ctx["flags"].insert(0, Flag("privilege", "painter", "clock rate != 1 in some cells during this act "
                                        f"({npriv} world steps); amount: integral of r over cells and time, m^2 s",
                                        priv))
        flags = ctx["flags"] + flags
        report.update(ctx["report"])
        residual["matter"] = max(residual["matter"], self.closure())
        Fa = w.F()
        entries = w.matter.since(ctx["mm"]) + w.energy.since(ctx["me"])
        self.wall_seconds = time.perf_counter() - ctx["wall"]
        return Receipt(op, ctx["t0"], w.t, entries, residual, {k: (ctx["F"][k], Fa[k]) for k in Fa}, tuple(flags),
                       report)

    def state_mass(self):
        """Every account the bench can check against state: kg per substance."""
        w = self.world
        dx2 = w.dx ** 2
        out = {"wet": self.rho * w.q.sum((1, 2)) * dx2, "ground": self.rho * w.ground.held.sum((1, 2)) * dx2}
        for k, st in enumerate(w.strata):
            out[f"stratum[{k}]"] = self.rho * st.sum((1, 2)) * dx2
        for v in self.vessels.values():
            out[v.account] = v.mass
        for t in self.tapes.values():
            out[t.account] = t.mass
        for k, m in self.reservoirs.items():
            if np.any(m) or f"reservoir[{k}]" in w.matter.accounts("reservoir["):
                out[f"reservoir[{k}]"] = m
        return out

    def closure(self):
        """Matter closure of the whole bench: max over accounts of |ledger - state| / total mass held."""
        st = self.state_mass()
        scale = float(sum(np.sum(v) for v in st.values())) or 1.0
        return max(float(np.max(np.abs(self.world.matter.balance(k) - v))) for k, v in st.items()) / scale

    # ---- clocks -------------------------------------------------------------------------------------------
    def rate_field(self, rate=1.0, where=None):
        """The clock rate field for a stretch of time: `rate` in `where` (default everywhere), 0 in held and taped
        cells, 1 elsewhere. None when it is 1 everywhere (so the world runs unprivileged, bit for bit)."""
        rate = float(rate)
        if not np.isfinite(rate):
            raise PortError("rate must be finite")
        if rate < 0:
            raise PortError(f"rate {rate:g} < 0 would run the world backwards (SPEC §8: impossible, breaks "
                            "`dissipative`); reverse is not implemented and is refused")
        r = np.ones((self.H, self.W))
        if where is None:
            r[:] = rate
        else:
            r[where] = rate
        r[self.held] = 0.0
        for t in self.tapes.values():
            if t.on:
                r[t.mask] = 0.0
        return None if np.all(r == 1.0) else r

    def _clock_flags(self, ctx, r):
        if r is None:
            return
        notes = []
        if self.held.any():
            notes.append(f"{int(self.held.sum())} cells held (hold)")
        taped = [t for t in self.tapes.values() if t.on]
        if taped:
            notes.append(f"{int(np.logical_or.reduce([t.mask for t in taped]).sum())} cells under tape: " + WALL_NOTE)
        if notes and not any(f.kind == "privilege" and f.subject == "clock" for f in ctx["flags"]):
            ctx["flags"].append(Flag("privilege", "clock", "; ".join(notes), 0.0))

    def _run(self, ctx, duration, r, steps=1):
        """Let the world run for `duration` s under the rate field r. Uses World.advance when the world has it; else
        `steps` implicit steps whose sizes double (the last one takes the remainder so the clock lands exactly)."""
        if duration <= 0:
            return
        self._clock_flags(ctx, r)
        w = self.world
        adv = getattr(w, "advance", None)
        if adv is not None:
            kw = {}
            sig = inspect.signature(adv).parameters
            if "rate" in sig:
                kw["rate"] = r
            elif r is not None:
                raise PortError("World.advance takes no rate: holds and tapes cannot be honoured")
            f = w.fork()
            try:
                out = f.advance(duration, **kw)
            except RuntimeError as e:
                raise PortError(f"the world could not advance {duration:.4g} s: {e}") from None
            self.world = f
            ctx["parts"].extend(out if isinstance(out, (list, tuple)) else [out])
            ctx["report"]["stepper"] = 1.0
            return
        n = max(1, int(steps))
        sizes = duration * (2.0 ** np.arange(n)) / (2.0 ** n - 1.0)
        for k in range(n):
            self._step(ctx, float(sizes[k]), r, 0)
        ctx["report"]["stepper"] = 0.0

    STEP_RETRY_DEPTH = 2

    def _step(self, ctx, dt, r, depth):
        """One World.step on a fork, adopted if it is accepted. A step the world gives up on (RuntimeError after its
        own halvings) has already committed part of itself, so it is never taken on the bench's world; it is
        retried as three steps of dt/3, to a depth of STEP_RETRY_DEPTH, each retry counted in the receipt. (The
        world's acceptance can flip with the last bits of dt: out/ports/repro/step_fails_one_ulp.py.)"""
        f = self.world.fork()
        try:
            rec = f.step(dt, rate=r, window="panel")
        except RuntimeError as e:
            if depth >= self.STEP_RETRY_DEPTH:
                raise PortError(f"the world could not take a step of {dt:.4g} s even split {3 ** depth} ways: {e}")
            ctx["report"]["world step retries"] = ctx["report"].get("world step retries", 0.0) + 1.0
            for _ in range(3):
                self._step(ctx, dt / 3.0, r, depth + 1)
            return
        self.world = f
        ctx["parts"].append(rec)

    # ---- paints and vessels -------------------------------------------------------------------------------
    def paint(self, name, recipe):
        if name in self.paints:
            raise PortError(f"there is already a paint called {name!r}")
        ctx = self._open()
        self.paints[name] = Paint.of(name, recipe, self.subs)
        return self._close("paint", ctx)

    def mix(self, name, parts):
        if name in self.paints:
            raise PortError(f"there is already a paint called {name!r}")
        ctx = self._open()
        missing = [p for p in parts if p not in self.paints]
        if missing:
            raise PortError(f"mix: no paint called {', '.join(missing)}")
        self.paints[name] = Paint.mix(name, {self.paints[k]: v for k, v in parts.items()}, self.subs)
        return self._close("mix", ctx)

    def _body_name_free(self, name):
        if name in self.tapes or name in RESERVOIRS:
            raise PortError(f"{name!r} already names a body")

    def fill(self, vessel, paint, volume):
        """Put `volume` m^3 of a paint into a vessel (a new one, or one already holding that paint)."""
        if paint not in self.paints:
            raise PortError(f"fill: no paint called {paint!r} (define it with a 'paint' act)")
        if not (np.isfinite(volume) and volume > 0):
            raise PortError("fill: volume must be > 0")
        p = self.paints[paint]
        v = self.vessels.get(vessel)
        if v is None:
            self._body_name_free(vessel)
            v = Vessel(vessel, p, np.zeros(len(self.subs)), self.rho)
        elif v.paint != p:
            raise PortError(f"vessel {vessel!r} holds {v.paint.name!r}; a vessel holds one paint")
        ctx = self._open()
        m = self.rho * p.phi(self.subs) * volume
        e = self.world.matter.post("origin", v.account, m, f"filled with {p.name}")
        v.mass = v.mass + e.amount
        v.filled += volume
        self.vessels[vessel] = v
        return self._close("fill", ctx)

    def _vessel(self, name):
        if name not in self.vessels:
            raise PortError(f"no vessel called {name!r} (fill one first)")
        return self.vessels[name]

    # ---- deposition ---------------------------------------------------------------------------------------
    def _deposit(self, ctx, v, V, weights, off, where_off):
        """Lay volume V (m^3) of vessel v's paint: `weights` [H, W] the fraction of V on each cell, `off` the
        fraction off the panel (to reservoir[where_off]). Taped cells' share goes to the tape. Returns the volume
        actually laid out of the vessel (it may run dry)."""
        take = v.take(V)
        if take <= v.dust:
            take = 0.0                                     # what is left is the rounding of earlier draws
        if take < V:
            ctx["report"]["shortfall m^3"] = ctx["report"].get("shortfall m^3", 0.0) + (V - take)
        if take <= 0:
            return 0.0
        w = self.world
        phi = v.paint.phi(self.subs)
        thick = take * weights / w.dx ** 2
        taken = np.zeros(len(self.subs))
        for t in reversed(list(self.tapes.values())):        # the latest tape lies on top
            if not t.on:
                continue
            caught = thick * t.mask
            if np.any(caught):
                m = self.rho * phi * caught.sum() * w.dx ** 2
                e = w.matter.post(v.account, t.account, m, f"caught by tape {t.name}")
                t.mass = t.mass + e.amount
                taken = taken + e.amount
                thick = np.where(t.mask, 0.0, thick)
        if np.any(thick > 0):
            r = w.add(phi[:, None, None] * thick[None], src=v.account)
            ctx["parts"].append(r)
            for e in r.entries:
                if e.ledger == "matter" and e.src == v.account:
                    taken = taken + e.amount
        if off > 0:
            m = self.rho * phi * take * off
            e = w.matter.post(v.account, f"reservoir[{where_off}]", m, f"missed the panel ({where_off})")
            self.reservoirs[where_off] = self.reservoirs[where_off] + e.amount
            taken = taken + e.amount
        v.mass = v.mass - taken
        if np.any(v.mass < 0):                                # rounding of the last drop: never below zero
            v.mass = np.maximum(v.mass, 0.0)
        ctx["report"]["laid m^3"] = ctx["report"].get("laid m^3", 0.0) + take
        return take

    def _pieces(self, pts, radius_cells, min_pieces):
        L = G.length(pts)
        return max(int(min_pieces), int(np.ceil(L / max(radius_cells, 0.5))), 1), L

    def pour(self, vessel, path, volume, duration, width, min_pieces=4, units="cells"):
        """A spout moving along `path` at constant speed for `duration` s, laying `volume` m^3 at a constant flow
        rate through a round footprint of diameter `width` m. Resolved in pieces: the spout moves at most half its
        footprint per piece (at least `min_pieces`); each piece lays its share, then the world runs for its time."""
        v = self._vessel(vessel)
        pts = self.path(path, units)
        if not (volume > 0 and width > 0 and duration >= 0):
            raise PortError("pour: volume and width > 0, duration >= 0")
        r = 0.5 * width / self.dx
        n, L = self._pieces(pts, r, min_pieces)
        ctx = self._open()
        rate = self.rate_field()
        s = np.linspace(0.0, L, n + 1)
        ends = G.along(pts, s)
        for k in range(n):
            wts, off = G.footprint(ends[k], ends[k + 1], r, self.H, self.W)
            self._deposit(ctx, v, volume / n, wts, off, "spill")
            self._run(ctx, duration / n, rate)
        ctx["report"]["pieces laid"] = float(n)
        return self._close("pour", ctx)

    def flood(self, vessel, region, thickness, duration=0.0, pieces=1, units="cells"):
        """An even film of `thickness` m over a region: a pour whose spout covers the region, in `pieces` equal
        parts over `duration` s."""
        v = self._vessel(vessel)
        mask = self.region(region, units)
        n = int(mask.sum())
        if n == 0:
            raise PortError("flood: the region holds no cell")
        if not (thickness > 0 and duration >= 0 and int(pieces) >= 1):
            raise PortError("flood: thickness > 0, duration >= 0, pieces >= 1")
        ctx = self._open()
        rate = self.rate_field()
        V = thickness * n * self.dx ** 2
        wts = mask / n
        k = int(pieces)
        for _ in range(k):
            self._deposit(ctx, v, V / k, wts, 0.0, "spill")
            self._run(ctx, duration / k, rate)
        return self._close("flood", ctx)

    def drop(self, vessel, at, volume, width=None, units="cells"):
        """One drop of `volume` m^3 laid at a point as a thin cap (paraboloid) of diameter `width` m, by default that of
        a hemisphere of the drop's volume. No time passes; the world spreads it when it next runs."""
        v = self._vessel(vessel)
        p = self.path(at, units)[0]
        if not volume > 0:
            raise PortError("drop: volume > 0")
        R = (0.5 * width if width else (3 * volume / (2 * np.pi)) ** (1 / 3)) / self.dx
        ctx = self._open()
        wts, off = _cap(p, R, self.H, self.W)
        self._deposit(ctx, v, volume, wts, off, "spill")
        return self._close("drop", ctx)

    def drip(self, vessel, path, drops, volume, interval, width=None, units="cells"):
        """`drops` drops of `volume` m^3 each, evenly spaced along `path`, one every `interval` s (the world runs
        between drops)."""
        v = self._vessel(vessel)
        pts = self.path(path, units)
        drops = int(drops)
        if not (drops >= 1 and volume > 0 and interval >= 0):
            raise PortError("drip: drops >= 1, volume > 0, interval >= 0")
        L = G.length(pts)
        at = G.along(pts, np.linspace(0.0, L, drops) if drops > 1 else np.zeros(1))
        R = (0.5 * width if width else (3 * volume / (2 * np.pi)) ** (1 / 3)) / self.dx
        ctx = self._open()
        rate = self.rate_field()
        for k in range(drops):
            wts, off = _cap(at[k], R, self.H, self.W)
            self._deposit(ctx, v, volume, wts, off, "spill")
            if k < drops - 1:
                self._run(ctx, interval, rate)
        return self._close("drip", ctx)

    MAX_DROPLETS = 20_000_000

    def spray(self, vessel, path, flow, duration, cone_radius, droplet_size, seed, min_pieces=4, units="cells"):
        """A spray gun moving along `path` for `duration` s at `flow` m^3/s. Droplets of diameter `droplet_size` m
        (monodisperse; their number is the volume over one droplet's, rounded, each carrying an equal share) leave the
        nozzle at times uniform over the act and land uniformly in a disc of `cone_radius` m around the nozzle's
        point. Positions come from numpy's default_rng(seed). A droplet lands in the cell holding its centre;
        one whose centre is off the panel is overspray, booked to reservoir[overspray]. Not modelled: evaporation in
        flight (flash-off), impact, splash and bounce, drift, a droplet's spread over more than one cell."""
        v = self._vessel(vessel)
        pts = self.path(path, units)
        if not (flow > 0 and duration > 0 and cone_radius > 0 and droplet_size > 0):
            raise PortError("spray: flow, duration, cone radius and droplet size > 0")
        if seed is None:
            raise PortError("spray: a seed is required (it goes in the journal)")
        V = flow * duration
        vd = np.pi * droplet_size ** 3 / 6
        N = max(1, int(round(V / vd)))
        if N > self.MAX_DROPLETS:
            raise PortError(f"spray: {N} droplets (more than {self.MAX_DROPLETS}); use larger droplets or less paint")
        R = cone_radius / self.dx
        n, L = self._pieces(pts, R, min_pieces)
        rng = np.random.default_rng(int(seed))
        counts = np.diff(np.round(np.linspace(0, N, n + 1)).astype(np.int64))
        s = np.linspace(0.0, L, n + 1)
        ctx = self._open()
        rate = self.rate_field()
        offs = 0
        for k in range(n):
            c = int(counts[k])
            if c:
                u = rng.random((c, 3))
                centre = G.along(pts, s[k] + (s[k + 1] - s[k]) * u[:, 0])
                rad = R * np.sqrt(u[:, 1])
                ang = 2 * np.pi * u[:, 2]
                x = centre[:, 0] + rad * np.cos(ang)
                y = centre[:, 1] + rad * np.sin(ang)
                j, i = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
                on = (i >= 0) & (i < self.H) & (j >= 0) & (j < self.W)
                cnt = np.bincount(i[on] * self.W + j[on], minlength=self.H * self.W).reshape(self.H, self.W)
                offs += int(c - on.sum())
                self._deposit(ctx, v, V * c / N, cnt / c, float(c - on.sum()) / c, "overspray")
            self._run(ctx, duration / n, rate)
        ctx["report"]["droplets"] = float(N)
        ctx["report"]["droplets off the panel"] = float(offs)
        ctx["report"]["seed"] = float(seed)
        return self._close("spray", ctx)

    # ---- masks --------------------------------------------------------------------------------------------
    def tape(self, name, region, units="cells"):
        """Lay a tape over a region: a shield from every source and, by the clock, a wall to flow. Refused over wet
        film (a tape is laid on dry ground or dry strata)."""
        if name in self.vessels or (name in self.tapes and self.tapes[name].on) or name in RESERVOIRS:
            raise PortError(f"{name!r} already names a body")
        if name in self.tapes:
            raise PortError(f"tape {name!r} was peeled; a tape is laid once (use a new name)")
        mask = self.region(region, units)
        if not mask.any():
            raise PortError("tape: the region holds no cell")
        wet = int(np.count_nonzero(mask & (self.world.q.sum(0) > 0)))
        if wet:
            raise PortError(f"tape: {wet} cells of the region are wet; a tape is laid on dry ground or dry strata")
        ctx = self._open()
        self.tapes[name] = Tape(name, mask, len(self.subs))
        ctx["report"]["cells taped"] = float(mask.sum())
        ctx["flags"].append(Flag("privilege", "clock", f"tape {name}: " + WALL_NOTE, 0.0))
        return self._close("tape", ctx)

    def peel(self, name):
        """Peel a tape off: its cells are free again; what it caught leaves with it (its account keeps it)."""
        t = self.tapes.get(name)
        if t is None or not t.on:
            raise PortError(f"no tape called {name!r} is on the panel")
        ctx = self._open()
        t.on = False
        ctx["report"]["peeled kg"] = float(t.mass.sum())
        return self._close("peel", ctx)

    # ---- environment --------------------------------------------------------------------------------------
    def tilt(self, angle, direction):
        """Tilt the panel by `angle` (rad) from horizontal, its downhill direction `direction` (rad from +x, along
        columns, toward +y, down rows). Gravity's in-plane part is g0 sin(angle), its normal part g0 cos(angle)."""
        if not (0 <= angle <= np.pi / 2):
            raise PortError("tilt: angle between 0 and 90 degrees")
        gt = g_0 * np.sin(angle)
        g = (gt * np.cos(direction), gt * np.sin(direction), g_0 * np.cos(angle))
        g = tuple(0.0 if abs(c) < 1e-15 * g_0 else float(c) for c in g)
        return self._env("tilt", gravity=g)

    def level(self):
        return self._env("level", gravity=(0.0, 0.0, g_0))

    def air(self, activity=None, T=None, boundary_layer=None):
        """Change the atmosphere: activities of its reservoirs (water's is the relative humidity), temperature (K),
        the still-air layer a vapour crosses (m)."""
        ch = {}
        if activity is not None:
            act = dict(self.world.env.activity)
            for k, a in activity.items():
                if not (np.isfinite(a) and a > 0):
                    raise PortError(f"air: activity of {k} must be > 0 (at 0 the air would take matter at unbounded "
                                    "gain of free energy; remove a reservoir by leaving it out)")
                act[k] = float(a)
            ch["activity"] = act
        if T is not None:
            ch["T"] = float(T)
        if boundary_layer is not None:
            ch["boundary_layer"] = float(boundary_layer)
        return self._env("air", **ch)

    def _env(self, op, **changes):
        ctx = self._open()
        try:
            env = self.world.env.with_(**changes)
        except ValueError as e:
            raise PortError(str(e)) from None
        ctx["parts"].append(self.world.set_environment(env))
        return self._close(op, ctx)

    # ---- time ---------------------------------------------------------------------------------------------
    def wait(self, duration, rate=1.0, where=None, steps=10, units="cells"):
        """Let `duration` s pass. rate (>= 0) multiplies every dissipation potential in `where` (default everywhere):
        0 holds, < 1 slows, > 1 hastens; held and taped cells stay at 0. Any rate != 1 is a privilege."""
        if not duration > 0:
            raise PortError("wait: duration > 0")
        mask = None if where is None else self.region(where, units)
        r = self.rate_field(rate, mask)
        ctx = self._open()
        if float(rate) != 1.0:
            ctx["flags"].append(Flag("privilege", "clock", f"wait at rate {float(rate):g} in "
                                     f"{'all cells' if mask is None else str(int(mask.sum())) + ' cells'}", 0.0))
        self._run(ctx, duration, r, steps)
        return self._close("wait", ctx)

    def hold(self, region, units="cells"):
        """Hold a region's clock at 0 until released: nothing in it changes and nothing crosses its edge."""
        mask = self.region(region, units)
        ctx = self._open()
        self.held |= mask
        ctx["flags"].append(Flag("privilege", "clock", f"hold: {int(self.held.sum())} cells held at rate 0", 0.0))
        ctx["report"]["cells held"] = float(self.held.sum())
        return self._close("hold", ctx)

    def release(self, region="all", units="cells"):
        mask = self.region(region, units)
        ctx = self._open()
        self.held &= ~mask
        ctx["report"]["cells held"] = float(self.held.sum())
        return self._close("release", ctx)

    # ---- settle -------------------------------------------------------------------------------------------
    def can_flow(self, horizon, length):
        """Where the wet film could still flow appreciably within `horizon` s over a lateral scale `length` m:
        a Look-free rule from the state and the constitutive maps (used only when the world has no settle).

            rate = h^3 / (3 eta) [ (rho g_n + gamma k^2) k^2 + rho |g_t| / (h length) ],   k = 2 pi / length
            can flow  iff  eta < inf,  rate * horizon >= 1,  and the yield stress (if any) is below
                           h (rho |g_t| + (rho g_n + gamma k^2) h k)

        The first bracket is the levelling of a ripple of wavelength `length` under gravity and capillarity, the
        second the drainage of a film down a tilt over `length`."""
        w = self.world
        film = Film(w.substances, w.q)
        h = film.h
        wet = h > 0
        out = np.zeros_like(wet)
        if not wet.any():
            return out
        flags = []
        eta = np.asarray(call(w.con, "viscosity", film, flags=flags), float)
        gam = call(w.con, "surface_tension", film, flags=flags)
        gam = np.zeros_like(h) if gam is None else np.where(np.isfinite(gam), np.asarray(gam, float), 0.0)
        ty = call(w.con, "yield_stress", film, flags=flags)
        ty = np.zeros_like(h) if ty is None else np.asarray(ty, float)
        rho = np.tensordot(self.rho, film.phi, 1)
        gx, gy, gn = w.env.gravity
        gt = np.hypot(gx, gy)
        k = 2 * np.pi / length
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            mob = np.where(np.isfinite(eta) & (eta > 0), h ** 3 / (3 * eta), 0.0)
            rate = mob * ((rho * gn + gam * k * k) * k * k + np.where(h > 0, rho * gt / (h * length), 0.0))
            drive = h * (rho * gt + (rho * gn + gam * k * k) * h * k)
        out = wet & (rate * horizon >= 1.0) & ~(ty >= drive)
        return out

    def settle(self, where="all", horizon=86400.0, length=1e-3, force=False, units="cells"):
        """Turn the wet film that can no longer flow into a stratum. With World.settle, the world decides; otherwise
        `can_flow` decides (a horizon of `horizon` s over `length` m). force=True retires every wet cell in `where`,
        still liquid or not: a privilege, flagged with the number of liquid cells retired."""
        w = self.world
        mask = self.region(where, units)
        ctx = self._open()
        wet = w.q.sum(0) > 0
        ws = getattr(w, "settle", None)
        if ws is not None and not force and where == "all":
            out = ws()
            ctx["parts"].extend(out if isinstance(out, (list, tuple)) else [out])
            ctx["report"]["rule"] = 1.0
            return self._close("settle", ctx)
        fl = self.can_flow(horizon, length)
        sel = mask & wet & (np.ones_like(wet) if force else ~fl)
        liquid = int(np.count_nonzero(sel & fl))
        if liquid:
            ctx["flags"].append(Flag("privilege", "painter", f"settle(force): retired {liquid} cells of film that "
                                     f"could still flow within {horizon:g} s", float(liquid)))
        ctx["report"]["cells retired"] = float(sel.sum())
        ctx["report"]["cells still wet"] = float((wet & ~sel).sum())
        ctx["report"]["rule"] = 0.0
        if sel.any():
            ctx["parts"].append(w.retire(mask=sel))
        return self._close("settle", ctx)


def _cap(p, R, H, W):
    """Weights of a drop's footprint: a flat disc of radius R cells centred at p (see geometry.footprint), and the
    fraction off the panel."""
    return G.footprint(p, p, max(float(R), 0.5), H, W)
