"""sheet: the phenotype of SPEC §10, each number measured by making films and looking, never read off the description.

    hiding        the dry film thickness at which the masstone's contrast ratio Y(black) / Y(white) reaches CR_HIDE,
                  from a ladder of wet thicknesses (log-spaced), each dried and looked at over both grounds; linear
                  interpolation in log thickness between the two rungs that bracket it
    tinting       the K/S the pigment adds to the white at TINT_RATIO (pigment : white by volume), at the band where
                  the tint's K/S is largest: (K/S)_tint - (K/S)_white, with K/S = (1 - R)^2 / (2 R) of the specular-
                  excluded reflectance of an opaque dry film over the white ground (Kubelka-Munk's remission function:
                  assumed, it ignores the surface)
    drying shift  CIEDE2000 between the wet and the dry masstone over the white ground at the drawdown thickness
    gloss         60-degree gloss of the dry masstone over the white ground
    infrared      the dry masstone's reflectance in each infrared band over white and over black, and their
                  difference (how much of the black stripe an infrared reflectogram would see through it)
"""
import numpy as np
from goo.evidence import weakest
from goo.film import Film
from goo import observe as O
from goo.lab import recipes as R
from goo.lab.dry import dry
from goo.lab.drawdown import (_see, matter_evidence, num, FlagTable, classification_flags, _paint_of, ILLUMINANTS)

CR_HIDE = 0.98
TINT_RATIO = 1.0 / 10.0
LADDER = np.geomspace(1e-6, 3e-3, 36)            # wet thicknesses, m
OPAQUE = 2e-3                                     # wet thickness of the "opaque" tint film, m


def _row(paint, binder, cells, white=R.WHITE):
    return R.wet_film(paint, binder, cells=cells, white=white)


def _looks(film, bands, light):
    out = {}
    for gk in ("black", "white"):
        st = R.stack(film, gk)
        out[gk] = (_see(st, bands, light), matter_evidence(st, bands))
    return out


def sheet(subject, binders=R.BINDERS, env=R.ROOM, bands=R.LAB_BANDS, illuminant=ILLUMINANTS[0], h_wet=R.WET_THICKNESS,
          white=R.WHITE, catalogue=()):
    """The phenotype of a substance (or Paint) as {name: number}, each number {value, unit, evidence, flags}, and the
    flag table under the key 'flags'."""
    paint = _paint_of(subject)
    if paint.binder is not None:
        binders = (paint.binder,)
    table = FlagTable()
    cls_ids = table.add(classification_flags(paint))
    light = O.diffuse_light(bands, illuminant)
    out = {}
    for b in binders:
        tag = b.name
        # hiding: a ladder of wet thicknesses, dried, over black and white
        wet = _row(paint, b, [(0.0, h) for h in LADDER], white)
        d = dry(wet, env, catalogue)
        lk = _looks(d.film, bands, light)
        cr = O.contrast_ratio(lk["black"][0], lk["white"][0], illuminant)
        c = cr.value[0]
        ev = weakest(lk["black"][1], lk["white"][1], d.evidence, cr.evidence)
        ids = table.add(cr.flags) + table.add(d.receipt.flags) + cls_ids
        hd = d.film.h[0]
        above = np.nonzero(c >= CR_HIDE)[0]
        if lk["black"][0].diverged.any() or lk["white"][0].diverged.any():
            out[f"{tag}: hiding thickness (dry, CR {CR_HIDE})"] = num(None, "m", ev, ids, True)
        elif above.size == 0:
            out[f"{tag}: hiding thickness (dry, CR {CR_HIDE})"] = dict(
                value=None, unit="m", evidence=ev, flags=ids,
                note=f"not reached by {hd[-1] * 1e6:.0f} um dry (CR {c[-1]:.3f} there)")
        else:
            k = above[0]
            if k == 0:
                h = hd[0]
            else:
                t = (CR_HIDE - c[k - 1]) / (c[k] - c[k - 1])
                h = float(np.exp(np.log(hd[k - 1]) + t * (np.log(hd[k]) - np.log(hd[k - 1]))))
            out[f"{tag}: hiding thickness (dry, CR {CR_HIDE})"] = num(h, "m", ev, ids)
        # drying shift and gloss at the drawdown thickness; tinting strength on an opaque film
        w_tint = 1.0 / (1.0 + TINT_RATIO)
        wet = _row(paint, b, [(0.0, h_wet), (w_tint, OPAQUE), (1.0, OPAQUE)], white)
        d = dry(wet, env, catalogue)
        lw = _looks(wet, bands, light)["white"]
        ld = _looks(d.film, bands, light)
        cw = O.colour_of(lw[0], illuminant, matter=lw[1])
        cd = O.colour_of(ld["white"][0], illuminant, matter=weakest(ld["white"][1], d.evidence))
        dE = float(O.ciede2000(cw.lab[0, 0], cd.lab[0, 0]))
        ids = table.add(cw.flags + cd.flags) + table.add(d.receipt.flags) + cls_ids
        un_w, un_d = bool(cw.unresolved[0, 0]), bool(cd.unresolved[0, 0])
        out[f"{tag}: drying shift (masstone over white, wet to dry)"] = num(dE, "dE00", weakest(cw.evidence, cd.evidence),
                                                                            ids, un_w or un_d)
        out[f"{tag}: masstone L* a* b* (dry, over white)"] = num(cd.lab[0, 0], "CIELAB", cd.evidence, ids, un_d)
        cell = Film(d.film.substances, d.film.q[:, :, 0:1])
        stw = R.stack(cell, "white")
        Lg = _see(stw, bands, O.gloss_light(bands, illuminant))
        g = O.gloss(Lg, illuminant, matter=matter_evidence(stw, bands))
        out[f"{tag}: gloss 60 (dry masstone)"] = num(g.value[0, 0], "GU", weakest(g.evidence, d.evidence),
                                                     table.add(g.flags) + cls_ids, bool(Lg.diverged.any()))
        vis = np.asarray(bands.centres) < 780e-9
        Rt = np.clip(np.asarray(ld["white"][0].diffuse[0, 1]), 1e-9, None)
        Rw = np.clip(np.asarray(ld["white"][0].diffuse[0, 2]), 1e-9, None)
        ks_t, ks_w = (1 - Rt) ** 2 / (2 * Rt), (1 - Rw) ** 2 / (2 * Rw)
        j = int(np.argmax(np.where(vis, ks_t, -np.inf)))
        out[f"{tag}: tinting strength (K/S added to the white at 1:{1 / TINT_RATIO:.0f}, at "
            f"{bands.centres[j] * 1e9:.0f} nm)"] = num(ks_t[j] - ks_w[j], "1",
                                                      weakest(ld["white"][1], d.evidence, "assumed"),
                                                      table.add(ld["white"][0].receipt.flags) + cls_ids,
                                                      bool(ld["white"][0].diverged[0, 1:].any()))
        ir_w = O.infrared(ld["white"][0]).value[0, 0]
        ir_b = O.infrared(ld["black"][0]).value[0, 0]
        ev_ir = weakest(ld["white"][1], ld["black"][1], d.evidence)
        ids_ir = table.add(ld["white"][0].receipt.flags + ld["black"][0].receipt.flags) + cls_ids
        dv = ld["white"][0].diverged[0, 0][~vis] | ld["black"][0].diverged[0, 0][~vis]
        for bnm, a, c_, u in zip(np.asarray(bands.centres)[~vis], ir_w, ir_b, dv):
            out[f"{tag}: IR {bnm * 1e9:.0f} nm over white"] = num(a, "1", ev_ir, ids_ir, bool(u))
            out[f"{tag}: IR {bnm * 1e9:.0f} nm see-through (white - black)"] = num(a - c_, "1", ev_ir, ids_ir, bool(u))
    out["flags"] = table.rows
    out["method"] = __doc__
    return out
