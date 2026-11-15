"""Card images: the evidence drawn on the picture, not only stored in the JSON (docs/LAB.md §4).

Every swatch carries a tag: its evidence class (C computed, K checked, A assumed, G guessed) and its flags (D a map
used out of its domain, B bounds collapsed, X impossible). A pigment with a guessed field is overlaid with hatching
and the word GUESS; an impossible one is framed in red with the laws it fails; a swatch with no steady state (past a
lasing threshold) is drawn as a crossed black box, never as a colour.
"""
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EV_TAG = dict(proved="P", exact="E", computed="C", checked="K", assumed="A", guessed="G")
CLASS_COLOUR = dict(real=(40, 120, 60), possible=(40, 80, 150), impossible=(190, 30, 30))
INK, PAPER, GREY = (20, 20, 20), (250, 249, 246), (110, 110, 110)


def font(size, bold=False):
    try:
        from matplotlib import font_manager
        path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans",
                                                                 weight="bold" if bold else "normal"))
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def ratio(w):
    if w <= 0:
        return "mass"
    if w >= 1:
        return "white"
    return f"1:{w / (1 - w):.0f}"


def flag_reasons(flags, ids=None):
    """Short unique reasons from a card's flag table (optionally only the ids given): 'D', 'B' or 'D/B' and the
    reason (a map's named domain condition), or 'impossible: subject: what'."""
    kinds, imp = {}, []
    for f in flags:
        if ids is not None and f["id"] not in ids:
            continue
        if f["kind"] == "impossible":
            r = f"impossible: {f['subject']}: {f['detail'].split(':')[0]}"
            if r not in imp:
                imp.append(r)
            continue
        k = "B" if f["kind"] == "bounds_collapsed" else "D"
        text = f["subject"]
        if "out of domain: " in text:
            body = text.split("out of domain: ", 1)[1].split("; bounded")[0]
            parts = [p.split(" [")[0] for p in re.split(r"(?<=\])\s*;\s*", body)]
        elif "bounded" in text:
            parts = [text.split(";")[0].replace("_", " ").replace(".", " ") + ": Wiener bounds collapsed at mid"]
        else:
            parts = [text.split(";")[0]]
        for p in parts:
            kinds.setdefault(p, set()).add(k)
    return imp + [f"{'/'.join(sorted(v, reverse=True))} {r}" for r, v in kinds.items()]


def tag_of(swatch, flags_by_id):
    t = EV_TAG.get(swatch["evidence"], "?")
    kinds = {flags_by_id[i]["kind"] for i in swatch["flags"]}
    if "out_of_domain" in kinds:
        t += "D"
    if "bounds_collapsed" in kinds:
        t += "B"
    if "impossible" in kinds:
        t += "X"
    return t


def _hatch(img, box, colour=(0, 0, 0, 70), step=14, width=3):
    x0, y0, x1, y1 = box
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for k in range(-(y1 - y0), x1 - x0, step):
        d.line([(x0 + k, y1), (x0 + k + (y1 - y0), y0)], fill=colour, width=width)
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    over.putalpha(Image.composite(over.getchannel("A"), Image.new("L", img.size, 0), mask))
    img.alpha_composite(over)


def _crossed(d, box):
    d.rectangle(box, fill=(0, 0, 0))
    d.line([box[:2], box[2:]], fill=(220, 40, 40), width=3)
    d.line([(box[0], box[3]), (box[2], box[1])], fill=(220, 40, 40), width=3)


def _swatch(d, box, sw, key):
    c = sw["colour"][key]
    if c["unresolved"]:
        _crossed(d, box)
        return
    d.rectangle(box, fill=tuple(c["srgb8"]))


def _fmt(n, digits=2):
    v = n["value"]
    if v is None:
        return "none: lases" if "steady" in n.get("note", "") else "none"
    s = f"{v:.{digits}f}"
    if "range" in n:
        lo, hi = n["range"]
        s += f" [{lo:.{digits}f}-{hi:.{digits}f}]"
    return s


def _pick(rows, **kw):
    for r in rows:
        if all(r[k] == v for k, v in kw.items()):
            return r
    return None


def render_card(card, path, scale=1.0):
    """Draw a drawdown Card (goo.lab.drawdown) to a PNG. Returns the path."""
    D = card.data
    flags = card.flags
    fid = {f["id"]: f for f in flags}
    sw = D["swatches"]
    binders = list(dict.fromkeys(s["binder"] for s in sw))
    tints = D["recipe"]["tints"]
    keys = list(sw[0]["colour"])
    W = 1200
    SW, SH, AW = 126, 104, 24               # swatch width, height, the second illuminant's strip
    LEFT = 96
    block = 40 + 2 * (SH + 8) + 16
    sub = D.get("substance") or {}
    klass = sub.get("klass", "mixture")
    guessed = any(p["evidence"] == "guessed" for p in (sub.get("prov") or {}).values())
    reasons = flag_reasons(flags)
    foot_lines = 5 + min(len(reasons), 8)
    sheet = D.get("sheet")
    H = 150 + len(binders) * block + (130 if sheet else 0) + 30 * foot_lines + 30
    img = Image.new("RGBA", (W, H), PAPER + (255,))
    d = ImageDraw.Draw(img)
    f_title, f_big, f_med, f_small, f_tag = font(46, True), font(28, True), font(22), font(18), font(16, True)

    # header
    d.text((24, 16), card.subject, font=f_title, fill=INK)
    badge = klass.upper()
    bw = d.textlength(badge, font=f_big) + 28
    d.rectangle([W - bw - 24, 20, W - 24, 64], fill=CLASS_COLOUR.get(klass, GREY))
    d.text((W - bw - 10, 24), badge, font=f_big, fill=(255, 255, 255))
    prov = sub.get("prov") or {}
    ev_line = "  ".join(f"{k} {p['evidence']}" for k, p in prov.items() if k in ("eps", "morph", "rho"))
    d.text((24, 74), (sub.get("material") or "no material named") + "   |   " + ev_line, font=f_small, fill=GREY)
    if klass == "impossible":
        fails = [r for r in reasons if r.startswith(f"impossible: {sub.get('name')}")]
        d.text((24, 100), ("; ".join(f.split(": ", 2)[-1] for f in fails) or "fails " + ", ".join(sub.get("failed", [])))[:90],
               font=f_med, fill=CLASS_COLOUR["impossible"])
    elif sub.get("blocking"):
        t = "not real because: " + ", ".join(f"{k.replace('rheo.', '')} {e}" for k, e in sub["blocking"])
        while d.textlength(t, font=f_small) > W - 48:
            t = t[:t.rfind(",")] + " ..."
        d.text((24, 100), t, font=f_small, fill=GREY)

    y = 150
    swatch_area = [LEFT, y, LEFT + len(tints) * (SW + AW + 8), y + len(binders) * block]
    for b in binders:
        rows = [s for s in sw if s["binder"] == b]
        evs = {st: min((s["evidence"] for s in rows if s["state"] == st), key=lambda e: -list(EV_TAG).index(e))
               for st in ("wet", "dry")}
        d.text((24, y + 4), b, font=f_big, fill=INK)
        rec = next((r for r in D["recipe"]["binders"] if r["binder"] == b), {})
        why = rec.get("weakest", {}).get("dry", {}).get("because", [])
        txt = f"wet {evs['wet']}, dry {evs['dry']}" + (f" ({'; '.join(w.split(' (')[0] for w in why[:4])})" if why else "")
        d.text((24 + d.textlength(b, font=f_big) + 16, y + 12), txt[:95], font=f_small, fill=GREY)
        y += 40
        for st in ("wet", "dry"):
            r0 = _pick(rows, state=st, white_share=tints[0], ground="white")
            d.text((10, y + 8), st, font=f_med, fill=INK)
            d.text((10, y + 38), f"{r0['thickness']['value'] * 1e6:.1f} um", font=f_small, fill=GREY)
            for j, w in enumerate(tints):
                x = LEFT + j * (SW + AW + 8)
                bl, wh = _pick(rows, state=st, white_share=w, ground="black"), _pick(rows, state=st, white_share=w,
                                                                                     ground="white")
                _swatch(d, [x, y, x + SW, y + SH // 2], bl, keys[0])
                _swatch(d, [x, y + SH // 2, x + SW, y + SH], wh, keys[0])
                if len(keys) > 1:
                    _swatch(d, [x + SW, y, x + SW + AW, y + SH // 2], bl, keys[1])
                    _swatch(d, [x + SW, y + SH // 2, x + SW + AW, y + SH], wh, keys[1])
                d.rectangle([x - 1, y - 1, x + SW + (AW if len(keys) > 1 else 0), y + SH], outline=(120, 120, 120), width=1)
                d.line([(x + SW, y), (x + SW, y + SH - 1)], fill=(160, 160, 160), width=1)
                tg = tag_of(wh, fid)
                tw = d.textlength(tg, font=f_tag) + 8
                d.rectangle([x + 2, y + SH - 22, x + 2 + tw, y + SH - 2], fill=(255, 255, 255))
                d.text((x + 6, y + SH - 21), tg, font=f_tag, fill=INK)
                if any(sw_["colour"][keys[0]]["out_of_gamut"] for sw_ in (bl, wh)):
                    d.text((x + SW - 18, y + 2), "!", font=f_tag, fill=(255, 255, 255))
            # numbers for this row: masstone gloss, hiding, drying shift, IR
            x = LEFT + len(tints) * (SW + AW + 8) + 4
            g = _pick(D["gloss"], binder=b, state=st, white_share=tints[0])["number"]
            h = _pick(D["hiding"], binder=b, state=st, white_share=tints[0])["number"]
            ir_w, ir_b = r0["infrared"]["R"], _pick(rows, state=st, white_share=tints[0], ground="black")["infrared"]["R"]
            gu = " GU" if g["value"] is not None else ""
            lines = [f"gloss {_fmt(g, 0)}{gu} {EV_TAG[g['evidence']]}", f"CR {_fmt(h)} {EV_TAG[h['evidence']]}"]
            lines += [f"IR {bnm / 1000:.1f}um {a:.2f} w / {c:.2f} k" for bnm, a, c in
                      zip(r0["infrared"]["bands_nm"], ir_w["value"], ir_b["value"])]
            if st == "dry":
                sh = _pick(D["drying_shift"], binder=b, white_share=tints[0], ground="white")["number"]
                lines.append(f"wet->dry dE00 {_fmt(sh, 1)}")
            sens = r0.get("sensitivity_dE00")
            if sens is not None and r0["colour"][keys[0]]["lab"]["value"] is not None:
                lines.append(f"size/n: dE00 <= {sens:.1f}")
            fs = font(16)
            for k, t in enumerate(lines):
                d.text((x, y + 20 * k - 2), t, font=fs, fill=INK)
            y += SH + 8
        y += 16
    # column heads under the area
    for j, w in enumerate(tints):
        d.text((LEFT + j * (SW + AW + 8) + 4, 150 - 26), ratio(w), font=f_small, fill=GREY)
    d.text((LEFT + len(tints) * (SW + AW + 8) + 4, 150 - 26), "masstone, dry/wet", font=f_small, fill=GREY)
    if guessed:
        _hatch(img, tuple(int(v) for v in swatch_area), (0, 0, 0, 38), step=18, width=2)
        d = ImageDraw.Draw(img)
        fg = font(120, True)
        cx, cy = (swatch_area[0] + swatch_area[2]) // 2, (swatch_area[1] + swatch_area[3]) // 2
        txt = Image.new("RGBA", (700, 170), (0, 0, 0, 0))
        ImageDraw.Draw(txt).text((10, 10), "GUESS", font=fg, fill=(255, 255, 255, 90), stroke_width=3,
                                 stroke_fill=(40, 40, 40, 200))
        txt = txt.rotate(18, expand=True)
        img.alpha_composite(txt, (cx - txt.width // 2, cy - txt.height // 2))
        d = ImageDraw.Draw(img)
    if klass == "impossible":
        d.rectangle([2, 2, W - 3, H - 3], outline=CLASS_COLOUR["impossible"], width=10)

    if sheet:
        first = binders[0]
        d.text((24, y), f"sheet, {first} (films made and looked at; drying shift per binder)", font=f_med, fill=INK)
        items = sheet_lines(sheet, first)
        for k, t in enumerate(items[:8]):
            d.text((24 + (k % 2) * 585, y + 32 + 22 * (k // 2)), t, font=font(16), fill=INK)
        y += 130
    # footer
    d.line([(24, y), (W - 24, y)], fill=GREY, width=1)
    y += 8
    foot = [f"top half over the black stripe, bottom over the white ground; left {keys[0]}, strip {keys[1] if len(keys) > 1 else ''} "
            "(adapted); specular excluded",
            "tags: C computed  K checked  A assumed  G guessed | D out of domain  B bounds collapsed  X impossible  ! out of gamut",
            "numbers: masstone; IR = reflectance over white (w) / over black (k); [lo-hi] = size x0.5..x2, n -0.05..+0.05",
            "daylight = Planck 6504 K, not CIE D65; observer: CIE 1931 analytic fit (assumed); recipes assumed",
            "mechanistic only: what goo's declared world makes from these descriptions, not a real product"]
    foot += reasons[:8]
    for t in foot:
        d.text((24, y), t[:150], font=font(17), fill=GREY if not t.startswith("impossible") else CLASS_COLOUR["impossible"])
        y += 30
    out = img.convert("RGB")
    if scale != 1.0:
        out = out.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    out.save(path)
    return path


SHORT = [("hiding thickness (dry, CR 0.98)", "hiding (CR 0.98)"), ("drying shift (masstone over white, wet to dry)",
         "drying shift"), ("gloss 60 (dry masstone)", "gloss 60"), ("tinting strength (K/S added to the white at 1:10, at",
         "tint strength 1:10 at"), ("see-through (white - black)", "see-through"), (" nm over white", " nm, over white")]


def sheet_lines(sheet, first=None):
    """Short lines of a sheet: every scalar number of the first binder, then the other binders' drying shifts."""
    out, rest = [], []
    for k, v in sheet.items():
        if not isinstance(v, dict) or "value" not in v or isinstance(v["value"], list):
            continue
        binder, _, name = k.partition(": ")
        for a, b in SHORT:
            name = name.replace(a, b)
        val, unit = v["value"], v.get("unit", "")
        if val is None:
            note = v.get("note", "none")
            s = "none (no steady state)" if "steady" in note else note
        elif unit == "m":
            s = f"{val * 1e6:.1f} um" + (f" [{v['range'][0] * 1e6:.1f}-{v['range'][1] * 1e6:.1f}]" if "range" in v else "")
        else:
            s = f"{val:.3g}" + (f" [{v['range'][0]:.3g}-{v['range'][1]:.3g}]" if "range" in v else "")
            s += f" {unit}" if unit not in ("1", "") else ""
        line = f"{name}: {s} {EV_TAG.get(v['evidence'], '?')}"
        if first is None or binder == first:
            out.append(line)
        elif name == "drying shift":
            rest.append(f"{binder.split(' ')[0]} {line}")
    return out + rest


def thumbnail(card, width=1080):
    """A compact row for a contact sheet: name, class, evidence, and the masstone-to-tint strip per binder, wet over
    dry, each swatch half over black and half over white (daylight)."""
    D = card.data
    sub = D.get("substance") or {}
    klass = sub.get("klass", "mixture")
    sw = D["swatches"]
    binders = list(dict.fromkeys(s["binder"] for s in sw))
    tints = D["recipe"]["tints"]
    key = list(sw[0]["colour"])[0]
    fid = {f["id"]: f for f in card.flags}
    TW = 330
    cw, ch = (width - TW - 20) // len(binders), 56
    sw_w = (cw - 14) // len(tints)
    H = 42 + 2 * (ch + 6) + 44 if klass != "impossible" else 42 + 2 * (ch + 6) + 60
    img = Image.new("RGBA", (width, H), PAPER + (255,))
    d = ImageDraw.Draw(img)
    size = 34
    while size > 18 and d.textlength(card.subject, font=font(size, True)) > TW - 24:
        size -= 2
    d.text((14, 8 + (34 - size) // 2), card.subject, font=font(size, True), fill=INK)
    d.rectangle([14, 56, 14 + d.textlength(klass.upper(), font=font(24, True)) + 16, 88], fill=CLASS_COLOUR.get(klass, GREY))
    d.text((22, 58), klass.upper(), font=font(24, True), fill=(255, 255, 255))
    prov = sub.get("prov") or {}
    ev = prov.get("eps", {}).get("evidence", "?")
    d.text((14, 96), f"eps {ev}", font=font(26, ev == "guessed"), fill=INK if ev != "guessed" else (160, 60, 0))
    mo = prov.get("morph", {}).get("evidence", "-")
    d.text((14, 128), f"size {mo}, rho {prov.get('rho', {}).get('evidence', '?')}", font=font(22), fill=GREY)
    for i, b in enumerate(binders):
        x0 = TW + i * cw
        d.text((x0, 8), b.replace("acrylic latex in water", "acrylic latex").replace("water with gum arabic", "gum water"),
               font=font(22, True), fill=INK)
        for r, st in enumerate(("wet", "dry")):
            y0 = 42 + r * (ch + 6)
            for j, w in enumerate(tints):
                x = x0 + j * sw_w
                bl = _pick(sw, binder=b, state=st, white_share=w, ground="black")
                wh = _pick(sw, binder=b, state=st, white_share=w, ground="white")
                _swatch(d, [x, y0, x + sw_w - 3, y0 + ch // 2], bl, key)
                _swatch(d, [x, y0 + ch // 2, x + sw_w - 3, y0 + ch], wh, key)
                d.rectangle([x - 1, y0 - 1, x + sw_w - 2, y0 + ch], outline=(120, 120, 120), width=1)
            d.text((x0 + len(tints) * sw_w - 2, y0 + 14), st[0], font=font(20, True), fill=GREY)
        tg = tag_of(_pick(sw, binder=b, state="dry", white_share=0.0, ground="white"), fid)
        d.text((x0, 42 + 2 * (ch + 6) + 4), f"dry tag {tg}", font=font(20), fill=INK)
    if any(p["evidence"] == "guessed" for p in prov.values()):
        _hatch(img, (TW, 42, width - 10, 42 + 2 * (ch + 6)), (0, 0, 0, 45), step=16, width=2)
        d = ImageDraw.Draw(img)
        d.text((width - 190, 42 + ch - 18), "GUESS", font=font(40, True), fill=(255, 255, 255), stroke_width=3,
               stroke_fill=(0, 0, 0))
    if klass == "impossible":
        d.rectangle([1, 1, width - 2, H - 2], outline=CLASS_COLOUR["impossible"], width=6)
        fails = sub.get("failed", [])
        d.text((14, 160), "fails " + ", ".join(fails), font=font(22, True), fill=CLASS_COLOUR["impossible"])
    return img.convert("RGB")


def contact_sheet(cards, path, width=1080):
    rows = [thumbnail(c, width) for c in cards]
    H = sum(r.height for r in rows) + 8 * len(rows) + 64
    img = Image.new("RGB", (width, H), (200, 200, 200))
    d = ImageDraw.Draw(img)
    d.text((14, 10), "goo lab drawdowns: top half over black, bottom over white; wet row, dry row", font=font(24, True),
           fill=INK)
    d.text((14, 38), "tags: C computed A assumed G guessed | D out of domain B bounds X impossible", font=font(20), fill=INK)
    y = 64
    for r in rows:
        img.paste(r, (0, y))
        y += r.height + 8
    img.save(path)
    return path
