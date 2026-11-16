"""contribute: one JSON file per substance, with sources and licence for every field, and `validate` (docs/LAB.md §6).

Keys carry their SI unit as a suffix (`rho_kg_m3`, `d_m`, `E0_J`); a value outside the plausible range of its unit is
refused as a probable unit error. Every field the description gives needs a provenance entry, and every `computed`
or `checked` entry needs a source and a licence. `validate` builds the Substance, classifies it with the five laws,
and checks a declared class against the computed one. Runtime dependencies: the standard library and numpy.
"""
from dataclasses import dataclass, field, fields
import json, math, pathlib
import numpy as np
from goo.evidence import Provenance, EVIDENCE
from goo.substance import (Substance, Thermo, Rheo, Reaction, StateVar, Oscillators, Lorentz, Drude, Gauss, Tabulated, Free,
                           LogNormal, Sphere, Coated, Hollow, Shape, classify)

FORMAT = "goo.substance/1"
# key -> (attribute, plausible SI range); a value outside the range is refused as a probable unit error
# The fields are discovered from the dataclasses (Substance, Thermo, Rheo, Reaction, StateVar), never listed by hand:
# a field added to the description is carried by the format. UNIT gives the SI suffix of a key (a field not in it is
# carried as <name>_SI and validated without a range check, with a warning); RANGE the plausible SI range used to
# refuse probable unit errors.
UNIT = {"M": "kg_mol", "p_sat": "Pa", "dH_vap": "J_mol", "cohesion": "J_m3", "gamma": "J_m2", "gamma_d": "J_m2",
        "D_vapour": "m2_s", "eta": "Pa_s", "E_eta": "J_mol", "separation": "m", "phi_percolation": "", "coupling": "",
        "rate": "1_s", "E_a": "J_mol", "rho": "kg_m3", "reservoir": "J_m3", "lo": "", "hi": ""}
RANGE = {"M": (1e-3, 1e3), "p_sat": (0.0, 1e7), "dH_vap": (1e3, 1e6), "cohesion": (-1e11, 1e11), "gamma": (-1.0, 1.0),
         "gamma_d": (-1.0, 1.0), "D_vapour": (-1e-2, 1e-2), "eta": (-1e15, 1e15), "E_eta": (-1e6, 1e6),
         "separation": (-1e-6, 1e-6), "phi_percolation": (-1.0, 2.0), "rate": (-1e6, 1e6), "E_a": (-1e7, 1e7)}
TOP_FIELDS = {"name", "eps", "morph", "rho", "thermo", "rheo", "reactions", "reservoir", "prov", "state", "material"}


def key_of(name):
    u = UNIT.get(name, "SI")
    return name if u == "" else f"{name}_{u}"


def _params_json(obj):
    """{key with unit: value} of every field of a Thermo or Rheo that is given (not None)."""
    out = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        if v is None:
            continue
        out[key_of(f.name)] = [list(r) for r in v] if isinstance(v, tuple) else v
    return out


def _params_read(cls, d, where, errors, warns):
    names = {key_of(f.name): f.name for f in fields(cls)}
    out = {}
    for k, x in (d or {}).items():
        if k not in names:
            errors.append(f"{where}.{k}: unknown key (one of {sorted(names)})")
            continue
        n = names[k]
        if isinstance(x, list):
            out[n] = tuple(tuple(y if isinstance(y, str) else float(y) for y in row) for row in x)   # (name, number) pairs keep their names
            continue
        if n not in RANGE:
            warns.append(f"{where}.{k}: no plausible range known for this field; unit not checked")
        out[n] = _number(x, f"{where}.{k}", errors, RANGE.get(n, (-np.inf, np.inf)))
    return out


RANGES = {"rho_kg_m3": (1e-3, 3e4), "d_m": (1e-10, 1e-2), "sigma_g": (1.0, 10.0), "lam_m": (1e-8, 1e-3),
          "E_J": (1e-23, 1e-15), "reservoir_J_m3": (0.0, 1e12), "rate_1_s": (-1e6, 1e6), "E_a_J_mol": (-1e7, 1e7)}
TOP = {"format", "name", "material", "eps", "morph", "rho_kg_m3", "thermo", "rheo", "reactions", "reservoir_J_m3",
       "prov", "state", "class", "note"}


# ------------------------------------------------------------------------------------------------ writing
def _eps_json(e):
    if isinstance(e, Oscillators):
        terms = []
        for t in e.terms:
            if isinstance(t, Lorentz):
                terms.append(dict(kind="lorentz", f=t.f, E0_J=t.E0, gamma_J=t.gamma))
            elif isinstance(t, Drude):
                terms.append(dict(kind="drude", Ep_J=t.Ep, gamma_J=t.gamma))
            elif isinstance(t, Gauss):
                terms.append(dict(kind="gauss", A=t.A, E0_J=t.E0, s_J=t.s))
            else:
                raise TypeError(f"no format for oscillator term {type(t).__name__}")
        return dict(kind="oscillators", eps_inf=e.eps_inf, terms=terms)
    if isinstance(e, Tabulated):
        return dict(kind="free" if isinstance(e, Free) else "tabulated", lam_m=list(e.lam),
                    parts=[dict(weight=w, n=list(n), k=list(k)) for w, n, k in e.parts_nk], lo_m=e.lo, hi_m=e.hi,
                    citation=e.citation)
    raise TypeError(f"no format for a dielectric of kind {type(e).__name__}")


def _size_json(z):
    return dict(d_m=z.d, sigma_g=z.sigma_g)


def _morph_json(m):
    if m is None:
        return None
    if isinstance(m, Sphere):
        return dict(kind="sphere", **_size_json(m.size))
    if isinstance(m, Coated):
        return dict(kind="coated", **_size_json(m.size), shell=_eps_json(m.shell), ratio=m.ratio,
                    shell_rho_kg_m3=m.shell_rho)
    if isinstance(m, Hollow):
        return dict(kind="hollow", **_size_json(m.size), ratio=m.ratio)
    if isinstance(m, Shape):
        return dict(kind="shape", name=m.name, **_size_json(m.size), params=[list(p) if isinstance(p, tuple) else p
                                                                            for p in m.params])
    raise TypeError(f"no format for morphology {type(m).__name__}")


def to_json(s, claim_class=True):
    """The contribution format of a Substance, as a dict (json.dumps it to write the file)."""
    unknown = {f.name for f in fields(s)} - TOP_FIELDS
    if unknown:
        raise TypeError(f"Substance has fields the contribution format does not know: {sorted(unknown)}")
    t, r = _params_json(s.thermo), _params_json(s.rheo)
    out = dict(format=FORMAT, name=s.name, material=s.material, eps=_eps_json(s.eps), morph=_morph_json(s.morph),
               rho_kg_m3=s.rho, thermo=t, rheo=r,
               reactions=[dict(stoich=[[n, c] for n, c in rx.stoich],
                               **{key_of(f.name): getattr(rx, f.name) for f in fields(rx) if f.name != "stoich"})
                          for rx in s.reactions],
               state=[{f.name: getattr(v, f.name) for f in fields(v)} for v in s.state],
               reservoir_J_m3=s.reservoir,
               prov={k: dict(evidence=p.evidence, source=p.source, licence=p.licence, note=p.note) for k, p in s.prov})
    if claim_class:
        out["class"] = classify(s).klass
    return out


def write(s, path):
    path = pathlib.Path(path)
    path.write_text(json.dumps(to_json(s), indent=1))
    return path


# ------------------------------------------------------------------------------------------------ reading
@dataclass
class Validation:
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    substance: object = None
    klass: str = None
    failed: tuple = ()

    def text(self):
        head = f"{'VALID' if self.ok else 'INVALID'}" + (f": {self.substance.name} is {self.klass}" if self.substance
                                                         else "")
        if self.failed:
            head += f" (fails {', '.join(self.failed)})"
        return "\n".join([head] + [f"  error: {e}" for e in self.errors] + [f"  warning: {w}" for w in self.warnings])


class _Err(Exception):
    pass


def _number(x, key, errors, rng=None, unit_key=None):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        errors.append(f"{key}: not a number ({x!r})")
        return None
    x = float(x)
    if not math.isfinite(x):
        errors.append(f"{key}: not finite")
        return None
    r = rng if rng is not None else RANGES.get(unit_key or key.split(".")[-1])
    if r is not None and not (r[0] <= x <= r[1]):
        errors.append(f"{key} = {x:g} is outside the plausible SI range {r[0]:g}..{r[1]:g}: wrong unit?")
    return x


def _eps(e, key, errors):
    if not isinstance(e, dict) or "kind" not in e:
        errors.append(f"{key}: a dict with 'kind'")
        raise _Err()
    k = e["kind"]
    n0 = len(errors)
    if k == "oscillators":
        extra = set(e) - {"kind", "eps_inf", "terms"}
        if extra:
            errors.append(f"{key}: unknown keys {sorted(extra)}")
        terms = []
        for i, t in enumerate(e.get("terms", [])):
            tk = t.get("kind")
            if tk == "lorentz":
                terms.append(Lorentz(_number(t.get("f"), f"{key}.terms[{i}].f", errors, (-1e6, 1e6)),
                                     _number(t.get("E0_J"), f"{key}.terms[{i}].E0_J", errors, RANGES["E_J"]),
                                     _number(t.get("gamma_J", 0.0), f"{key}.terms[{i}].gamma_J", errors,
                                             (-1e-15, 1e-15))))
            elif tk == "drude":
                terms.append(Drude(_number(t.get("Ep_J"), f"{key}.terms[{i}].Ep_J", errors, RANGES["E_J"]),
                                   _number(t.get("gamma_J"), f"{key}.terms[{i}].gamma_J", errors, (-1e-15, 1e-15))))
            elif tk == "gauss":
                terms.append(Gauss(_number(t.get("A"), f"{key}.terms[{i}].A", errors, (-1e6, 1e6)),
                                   _number(t.get("E0_J"), f"{key}.terms[{i}].E0_J", errors, RANGES["E_J"]),
                                   _number(t.get("s_J"), f"{key}.terms[{i}].s_J", errors, (-1e-15, 1e-15))))
            else:
                errors.append(f"{key}.terms[{i}]: unknown oscillator kind {tk!r}")
        if len(errors) > n0:
            raise _Err()
        return Oscillators(_number(e.get("eps_inf", 1.0), f"{key}.eps_inf", errors, (-1e6, 1e6)), tuple(terms))
    if k in ("tabulated", "free"):
        lam = e.get("lam_m")
        if not isinstance(lam, list) or len(lam) < 2:
            errors.append(f"{key}.lam_m: a list of wavelengths in m")
            raise _Err()
        la = np.asarray(lam, float)
        if la.min() < RANGES["lam_m"][0] or la.max() > RANGES["lam_m"][1]:
            errors.append(f"{key}.lam_m spans {la.min():g}..{la.max():g}: wavelengths must be in m (nm or um given?)")
        parts = []
        for i, p in enumerate(e.get("parts", [])):
            n, kk = p.get("n"), p.get("k")
            if not (isinstance(n, list) and isinstance(kk, list) and len(n) == len(lam) == len(kk)):
                errors.append(f"{key}.parts[{i}]: n and k must be lists on lam_m")
                continue
            parts.append((float(p.get("weight", 1.0)), tuple(float(x) for x in n), tuple(float(x) for x in kk)))
        if len(errors) > n0:
            raise _Err()
        cls = Free if k == "free" else Tabulated
        try:
            return cls(tuple(float(x) for x in lam), tuple(parts), _number(e.get("lo_m"), f"{key}.lo_m", errors,
                                                                           RANGES["lam_m"]),
                       _number(e.get("hi_m"), f"{key}.hi_m", errors, RANGES["lam_m"]), e.get("citation", ""))
        except (ValueError, TypeError) as x:
            errors.append(f"{key}: {x}")
            raise _Err()
    errors.append(f"{key}: unknown dielectric kind {k!r}")
    raise _Err()


def _morph(m, errors):
    if m is None:
        return None
    k = m.get("kind")
    size = LogNormal(_number(m.get("d_m"), "morph.d_m", errors, RANGES["d_m"]),
                     _number(m.get("sigma_g", 1.0), "morph.sigma_g", errors, RANGES["sigma_g"]))
    if k == "sphere":
        return Sphere(size)
    if k == "coated":
        return Coated(size, _eps(m.get("shell"), "morph.shell", errors), _number(m.get("ratio"), "morph.ratio", errors,
                                                                                  (0.0, 1.0)),
                      _number(m.get("shell_rho_kg_m3"), "morph.shell_rho_kg_m3", errors, RANGES["rho_kg_m3"]))
    if k == "hollow":
        return Hollow(size, _number(m.get("ratio"), "morph.ratio", errors, (0.0, 1.0)))
    if k == "shape":
        return Shape(str(m.get("name", "")), size, tuple(tuple(p) if isinstance(p, list) else p
                                                          for p in m.get("params", [])))
    errors.append(f"morph: unknown kind {k!r}")
    raise _Err()


def from_json(d):
    """A Substance from the contribution format (raises ValueError with every error found)."""
    v = validate(d)
    if v.substance is None:
        raise ValueError("; ".join(v.errors))
    return v.substance


def validate(src):
    """Check a contribution (a path, a JSON string or a dict): format and units, provenance completeness, that nothing
    is `computed` or `checked` without a source and a licence, the description itself, and admissibility (the class
    is computed by the five laws; a declared 'class' must agree)."""
    errors, warns = [], []
    if isinstance(src, (str, pathlib.Path)) and not str(src).lstrip().startswith("{"):
        d = json.loads(pathlib.Path(src).read_text())
    elif isinstance(src, str):
        d = json.loads(src)
    else:
        d = src
    if d.get("format") != FORMAT:
        errors.append(f"format must be {FORMAT!r}")
    extra = set(d) - TOP
    if extra:
        errors.append(f"unknown keys {sorted(extra)} (units belong in the key: rho_kg_m3, d_m, E0_J ...)")
    for k in ("name", "eps", "rho_kg_m3", "prov"):
        if k not in d:
            errors.append(f"missing {k!r}")
    if errors:
        return Validation(False, errors, warns)
    # provenance: every entry well formed; computed and checked need a source and a licence
    prov = {}
    for key, p in d["prov"].items():
        ev = p.get("evidence")
        if ev not in EVIDENCE:
            errors.append(f"prov[{key}]: evidence {ev!r} is not one of {EVIDENCE}")
            continue
        if ev in ("computed", "checked") and not (p.get("source") and p.get("licence")):
            errors.append(f"prov[{key}]: marked {ev} without a source and a licence")
            continue
        prov[key] = Provenance(ev, p.get("source", ""), p.get("licence", ""), p.get("note", ""))
    eps = morph = None
    try:
        eps = _eps(d["eps"], "eps", errors)
    except _Err:
        pass
    try:
        morph = _morph(d.get("morph"), errors)
    except _Err:
        pass
    rho = _number(d["rho_kg_m3"], "rho_kg_m3", errors)
    th = _params_read(Thermo, d.get("thermo"), "thermo", errors, warns)
    rh = _params_read(Rheo, d.get("rheo"), "rheo", errors, warns)
    rx = []
    for i, r in enumerate(d.get("reactions") or []):
        st = r.get("stoich")
        if not isinstance(st, list) or not all(isinstance(p, list) and len(p) == 2 for p in st):
            errors.append(f"reactions[{i}].stoich: a list of [name, kg per kg]")
            continue
        kw = _params_read(Reaction, {k: v for k, v in r.items() if k != "stoich"}, f"reactions[{i}]", errors, warns)
        try:
            rx.append(Reaction(tuple((str(n), float(c)) for n, c in st), **kw))
        except TypeError as x:
            errors.append(f"reactions[{i}]: {x}")
    sv = []
    for i, v in enumerate(d.get("state") or []):
        try:
            sv.append(StateVar(**{k: (float(x) if k in ("lo", "hi") else x) for k, x in v.items()}))
        except TypeError as x:
            errors.append(f"state[{i}]: {x}")
    res = _number(d.get("reservoir_J_m3", 0.0), "reservoir_J_m3", errors)
    if errors:
        return Validation(False, errors, warns)
    try:
        s = Substance(str(d["name"]), eps, morph, rho, Thermo(**th), Rheo(**rh), tuple(rx), res, prov,
                      state=tuple(sv), material=str(d.get("material", "")))
    except (ValueError, TypeError) as x:
        errors.append(f"description refused: {x}")
        return Validation(False, errors, warns)
    try:
        c = classify(s)
    except Exception as x:                                   # a law that cannot be evaluated is an error, not a pass
        errors.append(f"admissibility could not be evaluated: {type(x).__name__}: {x}")
        return Validation(False, errors, warns, s)
    if "class" in d and d["class"] != c.klass:
        errors.append(f"declares class {d['class']!r}, the laws give {c.klass!r}"
                      + (f" (fails {', '.join(c.failed)})" if c.failed else
                         f" (blocking: {', '.join(k for k, _ in c.blocking)})" if c.blocking else ""))
    if c.klass == "impossible":
        warns.append(f"impossible: fails {', '.join(c.failed)}; allowed, and flagged wherever it appears")
    return Validation(not errors, errors, warns, s, c.klass, tuple(c.failed))
