"""sensitivity: every card and sheet number again with the particle size halved and doubled and the index perturbed.

The variants of a substance (`variants`): its morphology with the size distribution's median diameter times 0.5 and 2
(sigma_g kept), and its dielectric function with the real index shifted by -DN and +DN at every wavelength (k kept;
written as a Tabulated on the library's 350-1700 nm grid, or the source's own grid). The variants are not classified:
an index shifted without its absorption is not Kramers-Kronig consistent; they probe the numbers, not the matter.
Each number's range is [min, max] over the nominal and the four variants; it is stored beside the number as "range".
"""
from dataclasses import replace
import numpy as np
from goo.evidence import Provenance
from goo.substance import Tabulated
from goo.substance import _nk_data as D
from goo.lab import recipes as R

SIZES = (0.5, 2.0)
DN = 0.05
PROV = Provenance("assumed", note=f"perturbation sizes: diameter x{SIZES[0]:g} and x{SIZES[1]:g}, index +-{DN:g} "
                                  "(of the order of the spread between sources of n for one mineral; chosen, not measured)")


def shifted(eps, dn):
    """eps with n + dn at every wavelength (k unchanged), as a Tabulated; a uniaxial eps has each part shifted."""
    if isinstance(eps, Tabulated):
        lam = np.asarray(eps.lam, float)
        lo, hi = eps.lo, eps.hi
        parts = tuple((w, tuple((np.real(m) + dn).tolist()), tuple(np.imag(m).tolist())) for w, m in eps.parts(lam))
    else:
        lam = np.asarray(D.GRID_M, float)
        lo, hi = lam[0], lam[-1]
        m = eps.nk(lam)
        parts = ((1.0, tuple((np.real(m) + dn).tolist()), tuple(np.imag(m).tolist())),)
    return Tabulated(tuple(lam.tolist()), parts, lo, hi, f"index shifted by {dn:+g}")


def variants(s):
    """{label: substance}: size halved and doubled (particles only), index -DN and +DN. Each variant has its own name,
    so that a variant of the white is a pigment beside the (unchanged) reference white."""
    out = {}
    if s.morph is not None:
        for f in SIZES:
            m = replace(s.morph, size=replace(s.morph.size, d=s.morph.size.d * f))
            out[f"d x{f:g}"] = replace(s, name=f"{s.name} (d x{f:g})", morph=m)
    for dn in (-DN, DN):
        out[f"n {dn:+g}"] = replace(s, name=f"{s.name} (n {dn:+g})", eps=shifted(s.eps, dn))
    return out


def _paint_variants(paint):
    """Variants of every pigment of a paint, one pigment at a time."""
    out = {}
    for i, (s, f) in enumerate(paint.pigments):
        for label, v in variants(s).items():
            pig = list(paint.pigments); pig[i] = (v, f)
            out[f"{s.name} {label}"] = replace(paint, pigments=tuple(pig))
    return out


def ranges(values):
    """{key: [lo, hi]} over a list of {key: value} (scalars or arrays; None skipped)."""
    out = {}
    keys = set().union(*values)
    for k in keys:
        vs = [np.asarray(v[k], float) for v in values if k in v and v[k] is not None]
        if vs:
            a = np.stack(vs)
            out[k] = [np.min(a, 0).tolist(), np.max(a, 0).tolist()]
    return out


def _sheet_values(sh):
    return {k: v["value"] for k, v in sh.items() if isinstance(v, dict) and "value" in v}


def sensitivity(subject, card=None, sheet=None, **kw):
    """Recompute the card's and the sheet's numbers for every variant; write each number's [min, max] beside it as
    'range'. Returns {"card": ranges, "sheet": ranges, "variants": labels, "prov": ...}."""
    from goo.lab.drawdown import drawdown, _paint_of
    from goo.lab.sheet import sheet as make_sheet
    from goo import observe as O
    paint = _paint_of(subject)
    vs = _paint_variants(paint)
    card_vals, sheet_vals, labs = [], [], []
    if card is not None:
        card_vals.append(card.numbers())
    if sheet is not None:
        sheet_vals.append(_sheet_values(sheet))
    for label, p in vs.items():
        if card is not None:
            c = drawdown(p, **kw)
            card_vals.append(c.numbers())
            labs.append(c)
        if sheet is not None:
            sheet_vals.append(_sheet_values(make_sheet(p, **kw)))
    out = dict(variants=list(vs), prov=dict(evidence=PROV.evidence, note=PROV.note))
    if card is not None:
        rc = ranges(card_vals)
        card.apply_ranges(rc)
        # the largest CIEDE2000 any variant moves each swatch's colour (first illuminant)
        for k, s in enumerate(card.data["swatches"]):
            key = next(iter(s["colour"]))
            nom = s["colour"][key]["lab"]["value"]
            dE = [float(O.ciede2000(np.asarray(nom), np.asarray(v))) for c in labs
                  for v in [c.data["swatches"][k]["colour"][key]["lab"]["value"]] if nom is not None and v is not None]
            s["sensitivity_dE00"] = max(dE) if dE else None
        out["card"] = rc
    if sheet is not None:
        rs = ranges(sheet_vals)
        for k, r in rs.items():
            if isinstance(sheet.get(k), dict):
                sheet[k]["range"] = r
        out["sheet"] = rs
    return out
