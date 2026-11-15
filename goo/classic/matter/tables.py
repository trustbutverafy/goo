"""goo.classic.matter.tables: holding measured tables as descriptions.

goo's optical data table (`goo/classic/data/optics.json`, schema "goo.optics-data/1", computed elsewhere from CC0
optical constants by Mie theory) gives, per species, n and k on a wavelength grid with the sources of every
value, and an assumed size distribution. Each becomes a Substance with a Tabulated dielectric (the citation
of its sources) and a Sphere morphology: `real` when its constants have a cited source and it passes the laws,
otherwise `possible` (or `impossible`, if the table's n and k violate a law). A species whose n or k is
'guessed' at any wavelength (per the table's own per-wavelength provenance) is not tied to a material. The table's own K and S are kept
beside the description (`.table_KS`) so the two Mie syntheses can be compared. Nothing here requires the file:
`load()` returns {} when it is absent.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from goo.classic.matter.dielectric import Tabulated
from goo.classic.matter.morphology import LogNormal, Sphere
from goo.classic.matter.substance import Substance, Provenance

DEFAULT = Path(__file__).resolve().parents[1] / "data" / "optics.json"


def _citation(sources):
    out = []
    for s in sources or []:
        if isinstance(s, str): out.append(s)
        elif isinstance(s, dict): out.append(s.get("reference") or s.get("file") or json.dumps(s)[:200])
    return "; ".join(out)


def load(path=None, roles=("pigment",)):
    """{name: Substance} from an optics.json (default: goo/classic/data/optics.json, or $GOO_OPTICS_JSON)."""
    path = Path(path or os.environ.get("GOO_OPTICS_JSON") or DEFAULT)
    if not path.exists(): return {}
    d = json.load(open(path))
    lam = np.asarray(d["wavelength_nm"], float)
    out = {}
    for name, e in d.get("species", {}).items():
        if roles and e.get("role") not in roles: continue
        if "n" not in e or "k" not in e: continue
        cit = _citation(e.get("sources"))
        extra = [s for s in (e.get("sources") or []) if isinstance(s, dict) and s.get("used_for")]
        bw = (e.get("provenance") or {}).get("by_wavelength", {}) if isinstance(e.get("provenance"), dict) else {}
        guessed = [lam[i] for key in ("n", "k") for i, v in enumerate(bw.get(key, [])) if v == "guessed"]
        measured = bool(cit) and not extra and not guessed
        diel = Tabulated(lam, np.asarray(e["n"], float), np.asarray(e["k"], float), citation=cit or None)
        size = e.get("size") or {}
        morph = Sphere(LogNormal(float(size.get("d50_um", 0.5)), float(size.get("sigma_g", 1.5)))) if size else None
        prov = Provenance("computed" if measured else ("partly guessed" if guessed else "assumed"),
                          f"{cit} (via {path.name}; extrapolated at {e.get('extrapolated_nm') or 'none'} nm; "
                          f"size {size.get('provenance', 'unstated')})" if cit else "", e.get("material", ""))
        sub = Substance(name, diel, morph, provenance=prov, note=e.get("note", ""))
        sub.table_KS = {k: (np.asarray(e["K"][k], float), np.asarray(e["S"][k], float)) for k in e.get("K", {}) if k in e.get("S", {})}
        sub.table_lam = lam
        sub.guessed_nm = sorted(set(guessed))
        sub.table_units_mm = True                            # mm^-1 of pure pigment; goo units = x 0.025
        out[name] = sub
    return out
