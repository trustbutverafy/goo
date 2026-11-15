"""drawdown: what a substance looks like as paint, made by composition and looked at (SPEC §10, docs/LAB.md §3).

A Card: masstone and a tint ladder with the white, each over the black stripe and over the white ground, wet and dry,
in each binder, under each illuminant, with gloss, hiding (contrast ratio) and an infrared reading. Every number is
stored as {"value", "unit", "evidence", "flags"} and the flags as a table of every flag any look or dry raised.
"""
from dataclasses import dataclass, field
import json, warnings
import numpy as np
from goo.evidence import Derived, Bounded, weakest
from goo.film import Film
from goo.light import look
from goo.substance import maps, classify
from goo import observe as O
from goo.lab import recipes as R
from goo.lab.dry import dry

TINTS = (0.0, 0.5, 0.75, 0.9, 0.97)
ILLUMINANTS = (O.DAYLIGHT, O.A)
CLAIM = ("Mechanistic claim only (SPEC §11): this is what goo's declared world produces from these descriptions, "
         "recipes and maps. It is not a claim that any real product looks like this.")


def _ev_of(ret):
    """Weakest evidence of what a map returned (Derived, Bounded, a tuple of them)."""
    if isinstance(ret, (Derived, Bounded)):
        return ret.evidence
    out = [_ev_of(x) for x in ret if isinstance(x, (Derived, Bounded, tuple))]
    return weakest(*out) if out else "exact"


def matter_evidence(stack, bands):
    """The weakest evidence of every map value a look at this stack reads (films present and the ground)."""
    ev = [_ev_of(maps.ground_optical(stack.ground, bands)), _ev_of(maps.ground_surface(stack.ground, bands))]
    for f in stack.films:
        if np.any(f.h > 0):
            ev += [_ev_of(maps.optical(f, bands)), _ev_of(maps.surface(f, bands))]
    return weakest(*ev)


class FlagTable:
    def __init__(self):
        self.rows, self._ids = [], {}

    def add(self, flags):
        ids = []
        for f in flags:
            key = (f.kind, f.subject, f.detail)
            if key not in self._ids:
                self._ids[key] = len(self.rows)
                self.rows.append(dict(id=len(self.rows), kind=f.kind, subject=f.subject, detail=f.detail,
                                      amount=float(f.amount)))
            ids.append(self._ids[key])
        return sorted(set(ids))


def weak_sources(film, ev, dried=None):
    """What makes a film's numbers as weak as `ev`: each substance present with a field of that class, and the
    drying (its reactions and mixing model) when its evidence is that class."""
    out = []
    for s, q in zip(film.substances, film.q):
        if np.any(q > 0):
            keys = [k for k, p in s.prov if p.evidence == ev]
            if keys:
                out.append(f"{s.name} ({', '.join(keys)})")
    if dried is not None and dried.evidence == ev:
        rx = [r[0] for r in dried.reactions]
        out.append("drying (" + (f"reactions of {', '.join(dict.fromkeys(rx))}" if rx else "mixing model") + ")")
    return out


NO_STEADY = "no steady state: the look is past a lasing threshold in some band (goo has no gain saturation)"


def num(value, unit, evidence, flags=(), unresolved=False):
    """A number with its unit, evidence and flags. unresolved: the look it comes from has no steady state in a band it
    reads, so there is no number (value None, with the reason)."""
    if unresolved or value is None:
        return dict(value=None, unit=unit, evidence=evidence, flags=list(flags), note=NO_STEADY)
    v = np.asarray(value, float)
    return dict(value=(float(v) if v.ndim == 0 else v.tolist()), unit=unit, evidence=evidence, flags=list(flags))


def _see(stack, bands, light):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return look(stack, bands, light, maps)


@dataclass
class Card:
    subject: str
    data: dict
    flags: list = field(default_factory=list)

    def to_json(self):
        return json.dumps(dict(subject=self.subject, **self.data, flags=self.flags), indent=1, default=_jsonable)

    def numbers(self):
        """{key: value} of every scalar number on the card (for sensitivity ranges)."""
        out = {}
        for s in self.data["swatches"]:
            k = f"{s['binder']}|{s['state']}|{s['white_share']:g}|{s['ground']}"
            for il, c in s["colour"].items():
                if c["lab"]["value"] is not None:
                    for j, ax in enumerate(("L*", "a*", "b*")):
                        out[f"{k}|{il}|{ax}"] = c["lab"]["value"][j]
            if s["infrared"]["R"]["value"] is not None:
                for j, b in enumerate(s["infrared"]["bands_nm"]):
                    out[f"{k}|IR {b:.0f} nm"] = s["infrared"]["R"]["value"][j]
        for kind in ("gloss", "hiding"):
            for g in self.data[kind]:
                out[f"{g['binder']}|{g['state']}|{g['white_share']:g}|{kind}"] = g["number"]["value"]
        for g in self.data["drying_shift"]:
            out[f"{g['binder']}|{g['white_share']:g}|{g['ground']}|drying dE00"] = g["number"]["value"]
        return out

    def apply_ranges(self, rng):
        """Write {key: [lo, hi]} (keys of `numbers`) beside each number as 'range'."""
        for s in self.data["swatches"]:
            k = f"{s['binder']}|{s['state']}|{s['white_share']:g}|{s['ground']}"
            for il, c in s["colour"].items():
                r = [rng.get(f"{k}|{il}|{ax}") for ax in ("L*", "a*", "b*")]
                if all(x is not None for x in r):
                    c["lab"]["range"] = [[x[0] for x in r], [x[1] for x in r]]
            r = [rng.get(f"{k}|IR {b:.0f} nm") for b in s["infrared"]["bands_nm"]]
            if r and all(x is not None for x in r):
                s["infrared"]["R"]["range"] = [[x[0] for x in r], [x[1] for x in r]]
        for kind in ("gloss", "hiding"):
            for g in self.data[kind]:
                r = rng.get(f"{g['binder']}|{g['state']}|{g['white_share']:g}|{kind}")
                if r is not None:
                    g["number"]["range"] = r
        for g in self.data["drying_shift"]:
            r = rng.get(f"{g['binder']}|{g['white_share']:g}|{g['ground']}|drying dE00")
            if r is not None:
                g["number"]["range"] = r


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def _paint_of(subject):
    if isinstance(subject, R.Paint):
        return subject
    return R.Paint.of(subject)


def classification_flags(paint):
    """An `impossible` flag per impossible substance of the paint, naming each failed law with its shortfall (the
    lab adds it: a look flags gain through K < 0, but not, for example, an acausal eps)."""
    from goo.ledger import Flag
    out = []
    for s, _ in paint.pigments:
        c = classify(s)
        if c.klass == "impossible":
            for v in c.verdicts:
                if not v.ok:
                    out.append(Flag("impossible", s.name, f"fails {v.law} by {v.shortfall:.3g} {v.unit}: {v.detail}",
                                    float(v.shortfall)))
    return out


def drawdown(subject, binders=R.BINDERS, tints=TINTS, illuminants=ILLUMINANTS, h_wet=R.WET_THICKNESS, env=R.ROOM,
             bands=R.LAB_BANDS, white=R.WHITE, catalogue=()):
    """A Card for a substance or a goo.lab.recipes.Paint. Films are made by composition (recipes.wet_film), dried by
    goo.lab.dry.dry, laid over real black and white stacks, and looked at with goo.light.look through goo.maps."""
    paint = _paint_of(subject)
    if paint.binder is not None:
        binders = (paint.binder,)
    table = FlagTable()
    cls_ids = table.add(classification_flags(paint))
    swatches, gloss, hiding, shift = [], [], [], []
    gl_light = O.gloss_light(bands, illuminants[0])
    df_light = O.diffuse_light(bands, illuminants[0])
    recipe = []
    for b in binders:
        wet = R.wet_film(paint, b, cells=[(w, h_wet) for w in tints], white=white)
        d = dry(wet, env, catalogue)
        dry_ids = table.add(d.receipt.flags)
        recipe.append(dict(binder=b.name, vehicle={s.name: f for s, f in b.parts}, pvc=b.pvc if paint.pvc is None
                           else paint.pvc, note=b.note, evidence=b.prov.evidence,
                           dry=dict(receipt_residual=d.receipt.residual, evidence=d.evidence,
                                    volatiles={k: dict(activity_env=v["activity_env"]) for k, v in d.volatiles.items()},
                                    reactions=[dict(substance=r[0], products=list(r[1])) for r in d.reactions],
                                    drawn_kg_m2={k: float(v.mean()) for k, v in d.drawn.items()})))
        labs = {}
        weak = {}
        for state, film in (("wet", wet), ("dry", d.film)):
            for j, w in enumerate(tints):
                cell = Film(film.substances, film.q[:, :, j:j + 1])
                ys = {}
                for gk in ("black", "white"):
                    st = R.stack(cell, gk)
                    mev = matter_evidence(st, bands)
                    if state == "dry":
                        mev = weakest(mev, d.evidence)
                    L = _see(st, bands, df_light)
                    ids = sorted(set(table.add(L.receipt.flags) + cls_ids + (dry_ids if state == "dry" else [])))
                    col = {}
                    for il in illuminants:
                        c = O.colour_of(L, il, matter=mev)
                        key = il.name.split(" ")[0]
                        un = bool(c.unresolved[0, 0])
                        col[key] = dict(illuminant=il.name, XYZ=num(c.XYZ[0, 0], "1", c.evidence, ids, un),
                                        lab=num(c.lab[0, 0], "CIELAB", c.evidence, ids, un),
                                        srgb8=c.srgb8[0, 0].tolist(), out_of_gamut=bool(c.out_of_gamut[0, 0]),
                                        above_one=bool(c.above_one[0, 0]), unresolved=bool(c.unresolved[0, 0]))
                        labs[(state, w, gk, key)] = None if un else c.lab[0, 0]
                    ir = O.infrared(L, matter=mev)
                    ir_un = bool(np.any(L.diverged[0, 0][np.asarray(bands.centres) > 780e-9]))
                    ys[gk] = L
                    swatches.append(dict(
                        binder=b.name, state=state, white_share=w, ground=gk,
                        thickness=num(cell.h[0, 0], "m", d.evidence if state == "dry" else "exact"),
                        composition={s.name: float(q) for s, q in zip(cell.substances, cell.q[:, 0, 0]) if q > 0},
                        colour=col, infrared=dict(bands_nm=[float(x) * 1e9 for x in np.asarray(bands.centres)
                                                            [np.asarray(bands.centres) > 780e-9]],
                                                  R=num(ir.value[0, 0], "1", ir.evidence, ids, ir_un)),
                        evidence=col[next(iter(col))]["lab"]["evidence"], matter_evidence=mev, flags=ids))
                cr = O.contrast_ratio(ys["black"], ys["white"], illuminants[0], matter=mev)
                hiding.append(dict(binder=b.name, state=state, white_share=w,
                                   number=num(cr.value[0, 0], "1", cr.evidence, table.add(cr.flags) + cls_ids,
                                              bool(ys["black"].diverged.any() or ys["white"].diverged.any()))))
                Lg = _see(R.stack(cell, "white"), bands, gl_light)
                g = O.gloss(Lg, illuminants[0], matter=matter_evidence(R.stack(cell, "white"), bands))
                gloss.append(dict(binder=b.name, state=state, white_share=w,
                                  number=num(g.value[0, 0], "GU", g.evidence, table.add(g.flags) + cls_ids,
                                             bool(Lg.diverged.any()))))
            ev_state = weakest(*[s["evidence"] for s in swatches if s["binder"] == b.name and s["state"] == state])
            weak[state] = dict(evidence=ev_state, because=weak_sources(film, ev_state, d if state == "dry" else None))
        recipe[-1]["weakest"] = weak
        for w in tints:
            for gk in ("white", "black"):
                key = illuminants[0].name.split(" ")[0]
                a_, b_ = labs[("wet", w, gk, key)], labs[("dry", w, gk, key)]
                dE = None if a_ is None or b_ is None else O.ciede2000(a_, b_)
                ev = weakest(*[s["evidence"] for s in swatches if s["binder"] == b.name and s["white_share"] == w
                               and s["ground"] == gk])
                shift.append(dict(binder=b.name, white_share=w, ground=gk, number=num(dE, "dE00", ev)))
    cls = classify(paint.pigments[0][0]) if len(paint.pigments) == 1 else None
    data = dict(
        kind="drawdown card", claim=CLAIM,
        substance=None if cls is None else dict(
            name=paint.pigments[0][0].name, klass=cls.klass, failed=list(cls.failed), material=paint.pigments[0][0].material,
            blocking=[list(x) for x in cls.blocking],
            prov={k: dict(evidence=p.evidence, source=p.source, licence=p.licence, note=p.note)
                  for k, p in paint.pigments[0][0].prov}),
        pigments=[dict(name=s.name, fraction=f, klass=classify(s).klass) for s, f in paint.pigments],
        recipe=dict(binders=recipe, tints=list(tints), white=white.name, wet_thickness_m=h_wet,
                    grounds=R.GROUND_PROV.note, environment=dict(activity=dict(env.activity), note=R.ROOM_PROV.note),
                    evidence="assumed"),
        observer=dict(cmf=O.CIE1931.name, illuminants=[il.name for il in illuminants], geometry="specular excluded "
                      "(colour from Look.diffuse)", prov={k: dict(evidence=p.evidence, source=p.source, note=p.note)
                                                          for k, p in O.colour.PROV.items()}),
        bands_nm=[float(c) * 1e9 for c in bands.centres],
        swatches=swatches, gloss=gloss, hiding=hiding, drying_shift=shift)
    return Card(paint.name, data, table.rows)
