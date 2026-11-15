"""A painting as a function of its journal (SPEC §9; docs/JOURNAL.md).

A Session owns a World and its bodies (a goo.ports.Bench) and a list of entries. Every act goes through `do` and
appends one entry: the act's name, its parameters as JSON (regions and paths by description), its seed, its clock,
the digest of its receipt, the digest of the world after it, a one-line summary, and the wall time (the only field
replay does not reproduce). An act that fails changes nothing: it is done on a fork and kept only if it succeeds.

The painter sees the world only through `look` (pictures and readings of a goo.light Look) and through receipts.
No method returns a World, a Film, a Stack, a Ground or an array of the world's state."""
import copy
import json
import time
import numpy as np
from goo.film import Environment
from goo.world import World
from goo.ports import Bench, make_ground, PortError
from goo.substance import maps, library
from goo.journal import acts as A
from goo.journal.quantities import quantity
from goo.journal.looking import look as _look, bands_of

KEEP = 32            # forks kept in memory for back()


class ReplayError(AssertionError):
    pass


class Refused(ValueError):
    """An act the session did not do, and why. Nothing changed."""


HESSIAN_FLOOR = 1e-250        # m: amounts below this are evaluated here for the mixing Hessian only (see Maps)


class Maps:
    """goo.substance.maps as the world's Constitutive, by delegation (a module cannot be pickled; this can), with one
    numerical guard. The world's diffusion and exchange Newton iterations drive a vanishing amount down by many
    decades (iterates of 1e-155 m and below are seen), and maps.mixing's Hessian, c / q on the diagonal, overflows
    to inf (and to NaN off the diagonal) below about 1e-280 m; the Jacobian then cannot be factored and the step
    fails for any dt (docs/PORTS.md §8). The Hessian enters only the Newton Jacobians, never a residual, a force or
    F, so here it is evaluated with every positive amount raised to at least HESSIAN_FLOOR, while f and mu are
    those of the true amounts. The converged state is the solution of the same equations."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(maps, name)

    def mixing(self, film, T=None):
        out = maps.mixing(film, T)
        q = np.asarray(film.q)
        tiny = (q > 0) & (q < HESSIAN_FLOOR)
        if not tiny.any():
            return out
        from goo.film import Film
        from dataclasses import replace
        guard = maps.mixing(Film(film.substances, np.where(tiny, HESSIAN_FLOOR, q), film.state), T)
        v, g = out.value, guard.value
        return replace(out, value=v._replace(hess=g.hess))


def normalise_initial(initial):
    """The initial state's description as JSON with defaults filled: size [W, H] in cells, dx (m), ground (name and
    any overrides), substances (library names, in order), air, bands, seed."""
    d = dict(initial)
    unknown = set(d) - {"size", "dx", "ground", "substances", "air", "bands", "seed"}
    if unknown:
        raise ValueError(f"initial: unknown keys {sorted(unknown)}")
    if "size" not in d or "substances" not in d:
        raise ValueError("initial: size [W, H] and substances are required")
    W, H = (int(v) for v in d["size"])
    if W < 2 or H < 2:
        raise ValueError("initial: size at least 2 x 2")
    g = d.get("ground", {"name": "paper"})
    g = {"name": g} if isinstance(g, str) else dict(g)
    subs = list(d["substances"])
    for s in subs:
        if s not in library.BY_NAME:
            raise ValueError(f"initial: no substance {s!r} in the library ({', '.join(library.BY_NAME)})")
    if len(set(subs)) != len(subs):
        raise ValueError("initial: a substance is listed twice")
    air = dict({"humidity": 0.5, "temperature": "20 C"}, **d.get("air", {}))
    return dict(size=[W, H], dx=d.get("dx", "0.25 mm"), ground=g, substances=subs, air=air, bands=d.get("bands", {}),
                seed=int(d.get("seed", 0)))


def build(initial):
    """A Bench from a normalised initial description."""
    W, H = initial["size"]
    dx = quantity(initial["dx"], "length", "dx")
    g = initial["ground"]
    kw = {k: quantity(g[k], "length" if k in ("pore_radius", "depth") else "number", f"ground.{k}")
          for k in ("porosity", "pore_radius", "depth") if k in g}
    ground, gdesc = make_ground(g["name"], H, W, solid=g.get("solid"), **kw)
    air = initial["air"]
    act = {k: quantity(v, "number", f"air.{k}") for k, v in air.get("activity", {}).items()}
    if air.get("humidity") is not None:
        act["water"] = quantity(air["humidity"], "number", "air.humidity")
    env = Environment(T=quantity(air.get("temperature", "20 C"), "temperature", "air.temperature"), activity=act)
    subs = tuple(library.get(s) for s in initial["substances"])
    w = World(Maps(), subs, ground, dx, env)
    b = Bench(w)
    b.ground_description = gdesc
    return b


def _bands(initial):
    spec = {k: (v if k == "ir" else quantity(v, "length", f"bands.{k}")) for k, v in initial["bands"].items()}
    if "ir" in spec:
        spec["ir"] = [quantity(c, "length", "bands.ir") for c in spec["ir"]]
    return bands_of(spec)


def summary(rec, wall=None):
    """One line: matter moved by account, energy by mechanism and work, residuals, flags, world time, wall time."""
    mv = {}
    for e in rec.entries:
        if e.ledger == "matter":
            key = f"{e.src}->{e.dst}"
            mv[key] = mv.get(key, 0.0) + float(np.sum(e.amount))
    parts = [f"{rec.op} t {rec.t0:.4g}->{rec.t1:.4g} s"]
    if mv:
        parts.append("matter " + ", ".join(f"{k} {v:.3g} kg" for k, v in mv.items() if v != 0))
    D = {k[2:]: v for k, v in rec.report.items() if k.startswith("D ") and v}
    num = rec.report.get("D_num time", 0.0) + rec.report.get("D_num transport", 0.0)
    if D or num:
        parts.append("dissipated " + ", ".join(f"{k} {v:.3g} J" for k, v in D.items()) + f", numerical {num:.3g} J")
    work = {}
    for e in rec.entries:
        if e.ledger == "energy" and e.src.startswith("work["):
            work[e.src] = work.get(e.src, 0.0) + float(e.amount)
    if work:
        parts.append("work " + ", ".join(f"{k} {v:.3g} J" for k, v in work.items()))
    parts.append("residual " + ", ".join(f"{k} {v:.1e}" for k, v in sorted(rec.residual.items())))
    fl = {}
    for f in rec.flags:
        fl[f.kind] = fl.get(f.kind, 0) + 1
    priv = [f for f in rec.flags if f.kind == "privilege"]
    if fl:
        parts.append("flags " + ", ".join(f"{k} x{n}" for k, n in fl.items()))
    for f in priv:
        parts.append(f"PRIVILEGE {f.subject}: {f.detail}" + (f" (amount {f.amount:.3g})" if f.amount else ""))
    for k in ("shortfall m^3", "world step retries", "droplets off the panel", "peeled kg", "cells retired",
              "cells still wet"):
        if k in rec.report:
            parts.append(f"{k} {rec.report[k]:.4g}")
    if rec.report.get("shortfall m^3"):
        parts.append("VESSEL RAN DRY")
    if wall is not None:
        parts.append(f"wall {wall:.2f} s")
    return "; ".join(parts)


class Session:
    """A painting in progress. `do(act)` with an act's JSON (or a list, done all or nothing); `look(...)`;
    `back(n)`; `fork()` for rehearsal; `entries` the journal; `replay(entries, initial)` reproduces it."""

    def __init__(self, initial):
        self._initial = normalise_initial(initial)
        self._bench = build(self._initial)
        self._bandset = _bands(self._initial)
        self._entries = []
        self._kept = {0: self._bench.fork()}

    # ---- what a painter may know -------------------------------------------------------------------------
    @property
    def initial(self):
        return copy.deepcopy(self._initial)

    @property
    def entries(self):
        return copy.deepcopy(self._entries)

    @property
    def time(self):
        return float(self._bench.world.t)

    @property
    def size(self):
        """(W, H) cells and dx (m)."""
        return self._initial["size"][0], self._initial["size"][1], float(self._bench.dx)

    def digest(self):
        return self._bench.world.digest()

    def vessels(self):
        """What the painter's own bodies hold: {vessel: (paint, volume m^3)}, {tape: (on the panel, kg caught)}."""
        b = self._bench
        return ({k: (v.paint.name, v.volume) for k, v in b.vessels.items()},
                {k: (t.on, float(t.mass.sum())) for k, t in b.tapes.items()})

    def paints(self):
        return {k: p.as_dict() for k, p in self._bench.paints.items()}

    def log(self):
        return [f"{e['n']:4d}  {e['summary']}" for e in self._entries]

    # ---- acting -------------------------------------------------------------------------------------------
    def do(self, act):
        """Do an act (a JSON object) or a list of acts (all or nothing). Returns the receipt (or a list of them).
        Raises Refused, with the reason, if any act cannot be done; then nothing has changed."""
        acts = act if isinstance(act, list) else [act]
        bench = self._bench.fork()
        new, kept, recs = [], [], []
        for a in acts:
            a = json.loads(json.dumps(a))                 # JSON only: what the journal can hold
            if isinstance(a, dict) and a.get("act") == "spray" and a.get("seed") is None:
                a["seed"] = self._initial["seed"] + len(self._entries) + len(new) + 1
            t = time.perf_counter()
            try:
                rec = A.run(bench, a)
            except (ValueError, RuntimeError) as e:
                raise Refused(f"{a.get('act', '?') if isinstance(a, dict) else '?'}: {e}") from None
            wall = time.perf_counter() - t
            n = len(self._entries) + len(new) + 1
            entry = dict(n=n, act=a["act"], params={k: v for k, v in a.items() if k != "act"}, seed=a.get("seed"),
                         clock=[rec.t0, rec.t1], receipt=rec.digest(), world=bench.world.digest(),
                         summary=summary(rec, wall), wall=wall)
            new.append(entry); recs.append(rec); kept.append(bench.fork())
        self._bench = bench
        for e, k in zip(new, kept):
            self._entries.append(e)
            self._kept[e["n"]] = k
        for n in sorted(self._kept)[:-KEEP]:
            if n != 0:
                del self._kept[n]
        return recs if isinstance(act, list) else recs[0]

    def __getattr__(self, name):
        """session.pour(vessel=..., ...) is session.do({"act": "pour", ...})."""
        if name in A.ACTS:
            return lambda **kw: self.do(dict(act=name, **kw))
        raise AttributeError(name)

    # ---- looking ------------------------------------------------------------------------------------------
    def look(self, light="daylight", exposure=1.0, region=None, gloss=False, infrared=False):
        """What the painting shows: a goo.observe picture of a goo.light Look, and gloss and infrared readings if
        asked. Changes nothing."""
        return _look(self._bench.world, self._bandset, light, exposure, region, gloss, infrared)

    # ---- time travel and rehearsal -----------------------------------------------------------------------
    def back(self, n=1):
        """Return to the state after entry len(entries) - n (by a kept fork if there is one, else by replay)."""
        n = int(n)
        if not 0 <= n <= len(self._entries):
            raise Refused(f"back: 0 to {len(self._entries)} entries")
        if n == 0:
            return
        k = len(self._entries) - n
        if k in self._kept:
            self._bench = self._kept[k].fork()
        else:
            self._bench = replay(self._entries[:k], self._initial)._bench
        self._entries = self._entries[:k]
        for j in [j for j in self._kept if j > k]:
            del self._kept[j]

    def fork(self):
        """An independent session that continues identically: rehearse on it, look, and drop it."""
        new = object.__new__(Session)
        new._initial = copy.deepcopy(self._initial)
        new._bench = self._bench.fork()
        new._bandset = self._bandset
        new._entries = copy.deepcopy(self._entries)
        new._kept = dict(self._kept)
        return new

    # ---- persistence --------------------------------------------------------------------------------------
    def journal(self):
        return dict(initial=self.initial, entries=self.entries)

    def __getstate__(self):
        return dict(_initial=self._initial, _bench=self._bench, _entries=self._entries, _kept={})

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._bandset = _bands(self._initial)
        self._kept = {}


def replay(entries, initial, check=True):
    """A session rebuilt from the initial description by doing each entry's act again. With check, every receipt's
    digest and the world's digest after every entry must equal the journal's, bit for bit."""
    s = Session(initial)
    for e in entries:
        s.do(dict(act=e["act"], **e["params"]))
        got = s._entries[-1]
        if check and (got["receipt"] != e["receipt"] or got["world"] != e["world"]):
            raise ReplayError(f"entry {e['n']} ({e['act']}): replay differs from the journal")
    return s
