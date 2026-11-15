"""The dry film: the composition a wet film reaches in equilibrium with the atmosphere (docs/LAB.md §2).

    The rule. A film is dry when (1) every reaction whose rate in the environment is positive has run to completion,
    i.e. until one of its reactants in the film is exhausted (reactants the film lacks are drawn from the
    environment's reservoirs, without limit), and (2) then every volatile substance (one whose description has
    p_sat > 0) and that the environment names has left or entered until its activity in the film, as
    `goo.substance.maps.evaporation` gives it, equals the environment's activity of that substance. A volatile the
    environment does not name has no reservoir (as in `maps.exchange`: k = 0) and stays. Nothing else
    changes: non-volatile, unreacting matter stays. Whatever the remaining liquids do not fill between the packed
    particles is void, and the maps already treat it as such.

Which substances are volatile, which react and what they become are read from their descriptions, never from their
names. Products are looked up by name in a catalogue (the film's own substances, then the library): a reaction's
stoichiometry names its products. This is the t -> infinity limit of the world's exchange and reaction terms at
fixed environment; the referee tests that the world's drying converges to it (tests/limits/).
"""
from dataclasses import dataclass
import numpy as np
from goo.evidence import weakest
from goo.film import Film
from goo.ledger import Ledger, Receipt, Flag
from goo.substance import maps, library as lib

ACT_TOL = 1e-10          # |activity - activity_env| at the solution, absolute
MAX_SWEEPS = 40          # Gauss-Seidel sweeps over the volatiles
BISECT = 90


@dataclass(frozen=True)
class Dried:
    film: Film                   # the dry film (wet substances, then products)
    wet: Film
    volatiles: dict              # name -> dict(q_wet, q_dry, activity, activity_env) arrays [H, W]
    reactions: tuple             # (substance, products, extent kg/m^2 [H, W]) per reaction run
    drawn: dict                  # environment reservoir -> kg/m^2 [H, W] taken into the film
    receipt: Receipt
    evidence: str                # of the activities (the mixing model) and the reactions used


def _activity(subs, q, i, env):
    ex = maps.evaporation(Film(subs, q), env)
    return ex[subs[i].name].activity


def _equilibrate(subs, q, i, a_env, env, flags):
    """Set q[i] per cell so that the activity of subs[i] equals a_env, by bisection (activity increases with q)."""
    if a_env <= 0:
        q[i] = 0.0
        return
    rest = q.sum(0) - q[i]
    lo = np.zeros(q.shape[1:])
    hi = np.maximum(q[i], np.maximum(rest, 1e-12))
    for _ in range(80):                                   # bracket: grow hi until activity(hi) >= a_env
        qq = q.copy(); qq[i] = hi
        a = _activity(subs, qq, i, env).value
        short = a < a_env
        if not short.any():
            break
        hi = np.where(short, hi * 2, hi)
    else:
        flags.append(Flag("out_of_domain", f"dry[{subs[i].name}]", f"its activity never reaches the environment's "
                          f"{a_env:g}: the film takes it up without limit"))
    for _ in range(BISECT):
        mid = 0.5 * (lo + hi)
        qq = q.copy(); qq[i] = mid
        a = _activity(subs, qq, i, env).value
        up = a < a_env
        lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
    q[i] = 0.5 * (lo + hi)


def _catalogue(film, extra):
    cat = dict(lib.BY_NAME)
    cat.update({s.name: s for s in (extra or ())})
    cat.update({s.name: s for s in film.substances})
    return cat


def dry(film, env, catalogue=()):
    """The dry film of `film` in environment `env` (goo.film.Environment or goo.substance.maps.Atmosphere), by the
    rule in this module's docstring. `catalogue`: extra substances products may be found among."""
    cat = _catalogue(film, catalogue)
    subs = list(film.substances)
    q = film.q.copy()
    shape = q.shape[1:]
    flags, runs, drawn = [], [], {}
    ev = ["assumed"]

    # (1) reactions to completion, repeated while any product reacts in turn
    for _round in range(10):
        moved = False
        F = Film(tuple(subs), q)
        for rr in maps.reactions(F, env):
            i = subs.index(next(s for s in subs if s.name == rr.substance))
            go = (np.asarray(rr.rate.value) > 0) & (q[i] > 0)
            if not go.any():
                continue
            ev.append(rr.rate.evidence)
            rx = rr.reaction
            ci = rx.coefficient(subs[i].name)
            ext = np.where(go, subs[i].rho * q[i] / -ci, 0.0)                # kg/m^2 of extent
            for n, c in rx.stoich:
                if c < 0 and n != subs[i].name and n in [s.name for s in subs]:
                    j = [s.name for s in subs].index(n)
                    ext = np.minimum(ext, np.where(go, subs[j].rho * q[j] / -c, 0.0))
            prods = []
            for n, c in rx.stoich:
                names = [s.name for s in subs]
                if n in names:
                    j = names.index(n)
                    q[j] = np.maximum(q[j] + c * ext / subs[j].rho, 0.0)
                    if c > 0:
                        prods.append(n)
                elif c < 0:
                    drawn[n] = drawn.get(n, np.zeros(shape)) - c * ext
                elif n in cat:
                    subs.append(cat[n]); q = np.concatenate([q, (c * ext / cat[n].rho)[None]], 0); prods.append(n)
                else:
                    flags.append(Flag("out_of_domain", f"dry[{subs[i].name}]", f"its product {n} is not a known "
                                      "substance: the reaction cannot be run to completion"))
            runs.append((subs[i].name, tuple(prods), ext))
            moved = True
        if not moved:
            break

    # (2) volatiles to the environment's activity (Gauss-Seidel over several volatiles)
    act = dict(env.activity)
    vol = [i for i, s in enumerate(subs) if s.volatile and s.name in act]      # no reservoir: no exchange
    q_after_rx = q.copy()
    for _sweep in range(MAX_SWEEPS):
        prev = q.copy()
        for i in vol:
            _equilibrate(tuple(subs), q, i, float(act[subs[i].name]), env, flags)
        if np.max(np.abs(q - prev)) <= 1e-15 + 1e-12 * np.max(np.abs(q)):
            break
    out = Film(tuple(subs), q)
    volatiles = {}
    if vol:
        ex = maps.evaporation(out, env)
        for i in vol:
            s = subs[i]
            a = np.asarray(ex[s.name].activity.value)
            ev.append(ex[s.name].activity.evidence)
            a_env = float(act.get(s.name, 0.0))
            have = q[i] > 0
            resid = np.where(have, np.abs(a - a_env), 0.0)
            if np.any(resid > 1e-6):
                flags.append(Flag("out_of_domain", f"dry[{s.name}]", f"activity not equalised: residual up to "
                                  f"{float(resid.max()):.3g} (phase separation or a non-monotone activity)"))
            q_wet = film.q[film.substances.index(s)] if s in film.substances else np.zeros(shape)
            volatiles[s.name] = dict(q_wet=q_wet, q_after_reactions=q_after_rx[i], q_dry=q[i].copy(), activity=a,
                                     activity_env=a_env)

    # the matter ledger of drying, kg/m^2 (panel mean), per substance and environment reservoir
    labels = [s.name for s in subs] + sorted(drawn)
    L = Ledger("matter", (len(labels),), labels)
    nS = len(subs)

    def vec(pairs):
        v = np.zeros(len(labels))
        for k, x in pairs:
            v[k] += x
        return v

    m0 = np.array([s.rho * (film.q[film.substances.index(s)].mean() if s in film.substances else 0.0) for s in subs])
    L.post("origin", "wet", np.r_[m0, np.zeros(len(drawn))], "the wet film")
    for k, name in enumerate(sorted(drawn)):
        L.post(f"reservoir[{name}]", "wet", vec([(nS + k, float(drawn[name].mean()))]), "drawn by reactions")
    m_rx = np.array([s.rho * q_after_rx[i].mean() for i, s in enumerate(subs)])
    delta = np.r_[m_rx - m0, -np.array([drawn[n].mean() for n in sorted(drawn)])]
    L.post("wet", "reaction", -np.minimum(delta, 0), "reactants consumed")
    L.post("reaction", "wet", np.maximum(delta, 0), "products made")
    m1 = np.array([s.rho * q[i].mean() for i, s in enumerate(subs)])
    L.post("wet", "reservoir[atmosphere]", np.r_[np.maximum(m_rx - m1, 0), np.zeros(len(drawn))], "volatiles left")
    L.post("reservoir[atmosphere]", "wet", np.r_[np.maximum(m1 - m_rx, 0), np.zeros(len(drawn))], "volatiles taken up")
    actual = np.r_[m1, np.zeros(len(drawn))]
    scale = max(float(np.abs(m0).sum() + sum(float(d.mean()) for d in drawn.values())), 1e-300)
    resid = float(np.max(np.abs(L.balance("wet") - actual))) / scale
    rx_net = abs(float(L.balance("reaction").sum())) / scale      # a reaction conserves total mass, not each substance's
    receipt = Receipt("dry", 0.0, float("inf"), L.since(0), {"matter": max(resid, rx_net)}, {}, tuple(flags),
                      {"max dry thickness / wet": float(np.max(np.divide(out.h, film.h, out=np.zeros(shape),
                                                                           where=film.h > 0)))})
    return Dried(out, film, volatiles, tuple(runs), drawn, receipt, weakest(*ev))
