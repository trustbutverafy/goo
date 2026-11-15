"""goo.classic.paints: what is on the brush, and what colour it will be.

A `Paint` is a recipe in goo's own species: the two carriers (water, oil), pigments, and for acrylic the
clear polymer solid `"acrylic"`. It knows its medium, whether it becomes permanent when dry (oil, acrylic)
or stays re-wettable (watercolour, gouache), and how to fill a brush (`load`, for `Brush.charge`).

Units are SPEC §2: volumes in cell units (one unit of thickness is 0.25 mm; a volume of 1 in one cell is
0.0156 µL). A brushful: a loaded filbert holds ~6000 units (0.1 mL), a wash mop ~12000 (0.19 mL).

`preview(paint, thickness, over)` is pure optics: the colour a film of that wet thickness shows as laid and
once dry, computed with the engine's own Kubelka-Munk (`Canvas._KS`, `Canvas.km_layer`, the composition of
`Canvas.reflectance`), so it cannot drift from what a canvas shows. `mix_for` inverts it over the stock
pigments (reusing `goo.classic.palette.Palette`'s mixes) in CIEDE2000; `swatch_card` draws a manufacturer's chart.

What the optics do and do not show (they are the World's, not this module's): water carries no K or S, so
a water-borne film looks the same wet and dry (no wet darkening, no drying lighter); only oil changes, its
linseed yellowing as it cures (a stratum is shown fully cured).
"""
from __future__ import annotations
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
from goo.classic.engine import PIGMENTS, CARRIERS, GROUND_RGBN, Canvas
from goo.classic.palette import Palette, lab, srgb_to_linear, linear_to_srgb, delta_e2000

BINDER = "acrylic"            # the clear polymer of acrylic paint (PIGMENTS["acrylic"], in the engine): a solid species
                              # that is not a colour, never offered as a pigment by mix_for
GAIN = 10.0                   # Canvas.reflectance's default gain (SPEC §2: K, S per 0.025 mm of pigment)
MEDIA = ("watercolour", "gouache", "acrylic", "oil", "water")

# Per medium: the wet thickness a brush lays by default (volume units; 1 = 0.25 mm) and a brushful (volume).
THICKNESS = dict(watercolour=0.5, gouache=0.3, acrylic=0.3, oil=0.25, water=0.5)
BRUSHFUL = dict(watercolour=12000.0, gouache=6000.0, acrylic=6000.0, oil=6000.0, water=12000.0)


def _norm(mix):
    mix = {k: float(v) for k, v in dict(mix).items() if v > 0}
    for k in mix:
        if k not in PIGMENTS: raise KeyError(f"unknown pigment {k!r}; known: {sorted(PIGMENTS)}")
        if k in CARRIERS: raise ValueError(f"{k!r} is a carrier, not a pigment")
    tot = sum(mix.values())
    if tot <= 0: raise ValueError("a mix needs at least one pigment with a positive share")
    return {k: v / tot for k, v in mix.items()}


class Paint:
    """What is on the brush: `parts` = {species: fraction of the wet paint's volume} (sums to 1).

    `medium` one of MEDIA; `mix` the colour pigments' shares (sum 1, binder and opacifier not included);
    `permanent` True if it becomes a stratum when dry and cannot be lifted (oil once cured, acrylic once
    its water has gone), False if it stays re-wettable (watercolour, gouache, water); `thickness` the wet
    film a brush lays by default, `brushful` a full brush (both volume units, SPEC §2); `params` the
    constructor's arguments."""

    def __init__(self, medium, parts, mix, permanent, params=None):
        if medium not in MEDIA: raise ValueError(f"medium must be one of {MEDIA}")
        tot = sum(parts.values())
        self.medium, self.mix, self.permanent = medium, dict(mix), bool(permanent)
        self.parts = {k: v / tot for k, v in parts.items() if v > 0}
        self.params = dict(params or {})
        self.thickness, self.brushful = THICKNESS[medium], BRUSHFUL[medium]

    # what the canvas must hold
    @property
    def pigments(self):
        """The solid species of this paint (pigments and binder), for `Canvas(W, H, pigments)`."""
        return [s for s in self.parts if s not in CARRIERS]

    @property
    def solids(self):
        """Solid fraction of the wet paint (phi as laid)."""
        return sum(v for s, v in self.parts.items() if s not in CARRIERS)

    def dried(self):
        """{species: volume per unit of wet paint} once dry: water gone (to the air and the ground), the rest
        stays (oil does not evaporate; it cures)."""
        return {s: v for s, v in self.parts.items() if s != "water"}

    def dry_thickness(self, thickness):
        return thickness * sum(self.dried().values())

    def load(self, volume=None):
        """{species: volume} for `Brush.charge`. Default: one brushful (`self.brushful`)."""
        volume = self.brushful if volume is None else float(volume)
        return {s: volume * v for s, v in self.parts.items()}

    def __repr__(self):
        mix = ", ".join(f"{k} {v:.2f}" for k, v in self.mix.items())
        par = ", ".join(f"{k}={v:g}" for k, v in self.params.items() if isinstance(v, (int, float)))
        return f"Paint({self.medium}: {mix or 'clear'}{'; ' + par if par else ''})"

    def blend(self, other, t):
        """This paint and `other` mixed on the palette in proportion t (0: this, 1: other), by volume of wet
        paint: what a graded wash dips into. Both must be the same medium, or one of them clean water (which
        dilutes: a watercolour blended toward water is a paler watercolour). Returns a new Paint."""
        t = float(t)
        if not 0 <= t <= 1: raise ValueError("blend: t is in 0..1")
        media = {self.medium, other.medium} - {"water"}
        if len(media) > 1: raise ValueError(f"blend: cannot mix {self.medium} with {other.medium}")
        medium = media.pop() if media else "water"
        parts = {}
        for p, w in ((self, 1 - t), (other, t)):
            for k, v in p.parts.items(): parts[k] = parts.get(k, 0.0) + w * v
        colour = {}
        for p, w in ((self, 1 - t), (other, t)):
            share = sum(p.parts.get(k, 0.0) for k in p.mix)       # the colour pigments' share of this paint
            for k, v in p.mix.items(): colour[k] = colour.get(k, 0.0) + w * share * v
        tot = sum(colour.values())
        mix = {k: v / tot for k, v in colour.items()} if tot > 0 else {}
        permanent = self.permanent if self.medium == medium else other.permanent
        return Paint(medium, parts, mix, permanent, params=dict(blend=t, between=[self.label, other.label]))

    def to_json(self):
        """A JSON-able dict that `Paint.from_json` turns back into this paint exactly."""
        return dict(medium=self.medium, parts=dict(self.parts), mix=dict(self.mix), permanent=self.permanent,
                    params=dict(self.params))

    @classmethod
    def from_json(cls, d):
        """The Paint `to_json` described."""
        p = cls(d["medium"], dict(d["parts"]), dict(d["mix"]), d["permanent"], d.get("params"))
        p.parts = {k: float(v) for k, v in d["parts"].items()}   # exactly as written (no renormalising round-off)
        return p

    @property
    def label(self):
        """A short name for charts: the pigments, largest share first, with shares when mixed."""
        if not self.mix: return "clean water"
        items = sorted(self.mix.items(), key=lambda kv: -kv[1])
        if len(items) == 1: return items[0][0].replace("_", " ")
        return " + ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in items)


def watercolour(mix, strength=0.04):
    """Watercolour: pigment carried in water, re-wettable when dry. `strength` is the solid fraction of the
    wash on the brush: 0.04 (default) is a medium wash, a few percent of pigment in water; 0.15 is near tube
    consistency; 0.005 a pale tint. Light colours come from dilution, not white. A mop holds ~12000 units
    (0.19 mL); a wash is laid ~0.5 units wet (125 µm) and dries to 0.5 * strength (5 µm at 0.04): the water
    evaporates or soaks into the gesso. (goo has no gum arabic: the dry film is pigment alone.)"""
    mix = _norm(mix)
    if not 0 < strength < 1: raise ValueError("strength is a solid fraction, 0 < strength < 1")
    parts = {"water": 1 - strength, **{k: strength * v for k, v in mix.items()}}
    return Paint("watercolour", parts, mix, permanent=False, params=dict(strength=strength))


def gouache(mix, strength=0.35, opacity=0.05):
    """Gouache: water with a high load of pigment, made opaque with white; re-wettable when dry. `strength`
    0.35: the solid fraction of gouache thinned to a cream on the brush (a tube is ~0.5); `opacity` 0.05: the
    share of the solids that is white (a designer's gouache is opacified by its filler; goo's only scattering
    filler is white). Laid ~0.3 units wet (75 µm), it dries to 0.3 * 0.35 = 0.105 units (26 µm)."""
    mix = _norm(mix)
    if not 0 <= opacity < 1: raise ValueError("opacity is a share of the solids, 0 <= opacity < 1")
    sol = {k: (1 - opacity) * v for k, v in mix.items()}
    sol["white"] = sol.get("white", 0.0) + opacity
    parts = {"water": 1 - strength, **{k: strength * v for k, v in sol.items()}}
    return Paint("gouache", parts, mix, permanent=False, params=dict(strength=strength, opacity=opacity))


def acrylic(mix, body=0.7, strength=0.5):
    """Acrylic: pigment and a clear polymer (`"acrylic"`) carried in water; short, dries by evaporation and
    is permanent. `body` 0.7: the polymer's share of the solids (artist acrylics are binder-rich: a pigment
    volume concentration of ~30%); `strength` 0.5: the solid fraction of the wet paint, so a film loses half
    its thickness as it dries (heavy-body acrylic is ~50% solids by volume). Laid ~0.3 units wet (75 µm),
    it dries to 0.15 units (37 µm) of which 0.045 (11 µm) is pigment."""
    mix = _norm(mix)
    if not 0 <= body < 1: raise ValueError("body is the polymer's share of the solids, 0 <= body < 1")
    parts = {"water": 1 - strength, BINDER: strength * body, **{k: strength * (1 - body) * v for k, v in mix.items()}}
    return Paint("acrylic", parts, mix, permanent=True, params=dict(body=body, strength=strength))


def oil(mix, fat=0.5):
    """Oil paint: pigment ground in linseed; permanent once cured (the canvas needs a cure rate). `fat` 0.5:
    the oil's share of the paint by volume (tube colours run 0.35-0.65; above ~0.3 the film is glossy,
    SPEC §6). Nothing evaporates: laid ~0.25 units (62 µm) it stays 0.25 units, and its oil yellows a little
    as it cures. A filbert holds ~6000 units (0.1 mL)."""
    mix = _norm(mix)
    if not 0 < fat < 1: raise ValueError("fat is the oil's share of the paint, 0 < fat < 1")
    parts = {"oil": fat, **{k: (1 - fat) * v for k, v in mix.items()}}
    return Paint("oil", parts, mix, permanent=True, params=dict(fat=fat))


def water():
    """Clean water: for wetting the paper, lifting, softening an edge. A mop holds ~12000 units."""
    return Paint("water", {"water": 1.0}, {}, permanent=False)


# ---- optics: the engine's own functions, applied to a film that exists nowhere but here

def _optics(species, shape):
    """A stand-in for a canvas, carrying only what Canvas._KS reads (species, nc, kappa = 0: freshly laid)."""
    return SimpleNamespace(species=list(species), nc=len(CARRIERS), _kappa=np.zeros(shape))


def _film(parts, thickness, wet):
    """R, T (4,) of a film of `thickness` holding `parts` {species: volume per unit thickness}: Canvas._KS
    (wet: the laid film, kappa 0; dry: a stratum, its oil fully cured) then Canvas.km_layer."""
    species = CARRIERS + [s for s in parts if s not in CARRIERS]
    vol = np.zeros((len(species), 1, 1))
    for s, v in parts.items(): vol[species.index(s), 0, 0] = v * thickness
    K, S = Canvas._KS(_optics(species, (1, 1)), vol * GAIN, wet)
    R, T = Canvas.km_layer(K, S)
    return R[0, 0], T[0, 0]


def _under(over):
    """The reflectance underneath, as an (n, 4) array of samples: bare ground, one colour, or a patch."""
    if over is None: return GROUND_RGBN[None, :].astype(float)
    a = np.asarray(over, float)
    if a.shape[-1] != 4: raise ValueError("over is a reflectance in R, G, B, N: shape (4,) or (H, W, 4)")
    return a.reshape(-1, 4)


def _compose(R, T, Rb):
    """Canvas.reflectance's composition of a layer over what is beneath it, averaged over the samples."""
    return np.clip(R + T * T * Rb / (1 - R * Rb), 0, 1).mean(0)


def preview(paint, thickness=None, over=None):
    """The colour a film of `paint` of wet thickness `thickness` (volume units; default the medium's usual
    film) shows as laid and once dry, over bare ground (`over=None`), a reflectance (4,) or a patch
    (H, W, 4) (each cell composed, then averaged, as the eye averages). Dry: water-borne paints have lost
    their water (thinner, all solids); oil is the cured stratum (linseed yellowed). Pure: no canvas exists
    or changes. Returns dict(wet, dry: sRGB (3,) in 0..1; wet_R, dry_R: reflectance (4,);
    dry_thickness: volume units)."""
    t = paint.thickness if thickness is None else float(thickness)
    Rb = _under(over)
    Rw, Tw = _film(paint.parts, t, wet=True)
    Rd, Td = _film(paint.dried(), t, wet=False)
    wet_R, dry_R = _compose(Rw, Tw, Rb), _compose(Rd, Td, Rb)
    return dict(wet=linear_to_srgb(wet_R[:3]), dry=linear_to_srgb(dry_R[:3]), wet_R=wet_R, dry_R=dry_R,
                dry_thickness=paint.dry_thickness(t))


# ---- mixing for a colour: Palette's mixes, the medium's recipe, the dry film, CIEDE2000

WASH_STRENGTHS = tuple(np.geomspace(0.002, 0.15, 16))   # watercolour dilutions searched: pale tint .. tube


def _make(medium, mix, strength=None, kw=None):
    kw = dict(kw or {})
    if medium == "watercolour": return watercolour(mix, strength if strength is not None else kw.get("strength", 0.04))
    if medium == "gouache": return gouache(mix, **kw)
    if medium == "acrylic": return acrylic(mix, **kw)
    if medium == "oil": return oil(mix, **kw)
    raise ValueError(f"mix_for mixes a coloured medium: {MEDIA[:4]}")


def default_pigments(medium):
    """Every pigment in PIGMENTS but the binder; for watercolour also without white (dilution makes light)."""
    return tuple(p for p in PIGMENTS if p != BINDER and not (medium == "watercolour" and p == "white"))


@lru_cache(maxsize=16)
def _palette(pigments, steps, max_pigments):
    return Palette(list(pigments), steps=steps, max_pigments=max_pigments, gain=GAIN)


@lru_cache(maxsize=32)
def _table(medium, pigments, thickness, kw, steps, max_pigments):
    """Dry-film R, T for every mix of the palette (and, for watercolour, every dilution): arrays (n, 4),
    with the (mix index, strength) of each row. Built from per-unit K, S that Canvas._KS itself gives."""
    pal = _palette(pigments, steps, max_pigments)
    kw = dict(kw)
    # unit K, S of each species in a dry film, from the engine (so a change there reaches the search)
    extra = {}
    if medium == "gouache": extra["white"] = kw.get("opacity", 0.05)
    if medium == "acrylic": extra[BINDER] = kw.get("body", 0.7)
    species = list(pigments) + [s for s in extra if s not in pigments]
    unit = np.zeros((len(CARRIERS) + len(species), len(CARRIERS) + len(species), 1))
    for i in range(len(unit)): unit[i, i, 0] = GAIN
    Ku, Su = Canvas._KS(_optics(CARRIERS + species, (len(unit), 1)), unit, False)
    Ku, Su = Ku[:, 0], Su[:, 0]                                   # (species, 4) per unit volume, cured
    oilK = Ku[1]
    Kp, Sp = Ku[len(CARRIERS):len(CARRIERS) + len(pigments)], Su[len(CARRIERS):len(CARRIERS) + len(pigments)]
    W = np.zeros((len(pal.mixes), len(pigments)))
    for m, mx in enumerate(pal.mixes):
        for c, w in mx.items(): W[m, c] = w
    Kmix, Smix = W @ Kp, W @ Sp                                   # per unit volume of colour pigment
    Kx = sum((f * Ku[len(CARRIERS) + species.index(s)] for s, f in extra.items()), np.zeros(4))
    Sx = sum((f * Su[len(CARRIERS) + species.index(s)] for s, f in extra.items()), np.zeros(4))
    colour = 1 - sum(extra.values())
    if medium == "watercolour": strengths = WASH_STRENGTHS
    elif medium == "oil": strengths = (1 - kw.get("fat", 0.5),)
    elif medium == "gouache": strengths = (kw.get("strength", 0.35),)
    else: strengths = (kw.get("strength", 0.5),)
    Rs, Ts, idx = [], [], []
    for s in strengths:
        oil_f = kw.get("fat", 0.5) if medium == "oil" else 0.0
        K = thickness * (s * (colour * Kmix + Kx) + oil_f * oilK)
        S = thickness * s * (colour * Smix + Sx)
        R, T = Canvas.km_layer(K, S)
        Rs.append(R); Ts.append(T); idx += [(m, s) for m in range(len(pal.mixes))]
    return pal, np.concatenate(Rs), np.concatenate(Ts), idx


def _samples(over, n=8):
    """At most n representative reflectances of what is underneath (a patch is split by lightness)."""
    Rb = _under(over)
    if len(Rb) <= n: return Rb
    order = np.argsort(Rb[:, :3].mean(1))
    return np.stack([Rb[c].mean(0) for c in np.array_split(order, n)])


def mix_for(colour_srgb, medium, thickness=None, over=None, pigments=None, steps=10, max_pigments=3, refine=True, **kw):
    """The paint of `medium` whose dry film comes closest to `colour_srgb` in CIEDE2000.

    - `colour_srgb`: the colour wanted, sRGB, 0..1 (or 0..255 if any value exceeds 1).
    - `thickness`: the WET film laid (volume units, 0.25 mm each; default the medium's usual film,
      `THICKNESS[medium]`), as one even film. The colour judged is of that film once DRY (water gone, oil
      cured; `preview(...)["dry"]`). A brush lays ribbons, not a slab: what a stroke leaves averages a little
      thinner than its target.
    - `over`: what the film lies on, as linear reflectance in the four bands R, G, B, N (what
      `Canvas.reflectance()` returns): None for bare gesso (`GROUND_RGBN`), one reflectance (4,), or a patch
      (H, W, 4) or (n, 4) of them (each composed, then averaged, as the eye averages).
    - `pigments`: those to mix from (default `default_pigments(medium)`); mixes of up to `max_pigments` on
      Palette's simplex grid, and for watercolour the dilution too; then the best mix's shares (and dilution)
      are refined continuously. `kw` are the medium's fixed recipe (body, fat, opacity, strength).

    Returns (Paint, predicted dry sRGB (3,) in 0..1, delta E 2000): the error is the gamut's honest answer,
    not hidden."""
    c = np.asarray(colour_srgb, float)[:3]
    if (c > 1).any(): c = c / 255.0
    lab_t = lab(srgb_to_linear(c))
    t = THICKNESS[medium] if thickness is None else float(thickness)
    pigments = tuple(pigments) if pigments is not None else default_pigments(medium)
    kwk = tuple(sorted(kw.items()))
    pal, R, T, idx = _table(medium, pigments, round(t, 9), kwk, steps, max_pigments)
    Rb = _samples(over)
    Rt = np.clip(R[None, :, :3] + T[None, :, :3] ** 2 * Rb[:, None, :3] / (1 - R[None, :, :3] * Rb[:, None, :3]), 0, 1).mean(0)
    dE = delta_e2000(lab(Rt), lab_t)
    best = int(np.argmin(dE)); m, s = idx[best]
    names = [pigments[i] for i in pal.mixes[m]]
    w0 = np.array([pal.mixes[m][i] for i in pal.mixes[m]])
    s_best, w_best = s, w0
    if refine:
        from scipy.optimize import minimize

        def unpack(x):
            w = np.exp(np.r_[0.0, x[:len(names) - 1]]); w /= w.sum()
            return w, (float(np.exp(x[-1])) if medium == "watercolour" else s)

        def err(x):
            w, st = unpack(x)
            if medium == "watercolour" and not (0.0005 <= st <= 0.3): return 1e3
            p = _make(medium, dict(zip(names, w)), st, kw)
            Rd, Td = _film(p.dried(), t, wet=False)
            return float(delta_e2000(lab(_compose(Rd, Td, Rb)[:3]), lab_t))

        x0 = list(np.log(w0[1:] / w0[0]))
        if medium == "watercolour": x0.append(np.log(s))
        if x0:
            res = minimize(err, np.array(x0), method="Nelder-Mead", options=dict(maxfev=120, xatol=1e-3, fatol=1e-3))
            if res.fun < dE[best]: w_best, s_best = unpack(res.x)
    paint = _make(medium, dict(zip(names, w_best)), s_best, kw)
    pred = preview(paint, t, _samples(over))["dry"]               # the answer is what preview says, exactly
    return paint, pred, float(delta_e2000(lab(srgb_to_linear(pred)), lab_t))


# ---- the chart

def _font(size):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try: return ImageFont.truetype(name, size)
        except OSError: pass
    return ImageFont.load_default()


def black_stripe():
    """The reflectance of the chart's black stripe: a dry stroke of strong carbon-black ink on gesso."""
    return preview(watercolour({"carbon": 1}, strength=0.15), 1.0)["dry_R"]


def swatch_card(paints, path, thicknesses=None, title=None, cell=84):
    """A manufacturer's chart: one row per paint, three films (thin, medium, thick: by default 0.4x, 1x and
    3x the medium's usual film), each a square over white gesso with a carbon-black stripe across its
    middle (opacity shows there), wet on the left half and dry on the right. Pure optics (`preview`).
    Writes a PNG to `path` and returns the image as an (H, W, 3) uint8 array."""
    from PIL import Image, ImageDraw
    paints = list(paints)
    f_title, f, f_small = _font(18), _font(13), _font(11)
    label_w, gap, head = 250, 14, 58 + (22 if title else 0)
    W = label_w + 3 * (cell + gap) + gap
    Hh = head + len(paints) * (cell + 18) + 16
    img = Image.new("RGB", (W, Hh), (246, 244, 238)); d = ImageDraw.Draw(img)
    y = 10
    if title: d.text((12, y), title, fill=(20, 20, 20), font=f_title); y += 26
    d.text((12, y), "wet | dry, each over gesso with a black stripe; thickness = wet film laid", fill=(80, 80, 80), font=f_small)
    stripe = black_stripe()
    to8 = lambda srgb: tuple(int(round(255 * v)) for v in np.clip(srgb, 0, 1))
    for r, p in enumerate(paints):
        y0 = head + r * (cell + 18)
        ts = thicknesses or (0.4 * p.thickness, p.thickness, 3 * p.thickness)
        d.text((12, y0 + 6), p.label, fill=(20, 20, 20), font=f)
        par = ", ".join(f"{k} {v:g}" for k, v in p.params.items() if isinstance(v, (int, float)))
        d.text((12, y0 + 26), f"{p.medium}{' - ' + par if par else ''}", fill=(90, 90, 90), font=f_small)
        d.text((12, y0 + 42), "permanent when dry" if p.permanent else "re-wettable", fill=(90, 90, 90), font=f_small)
        for j, t in enumerate(ts):
            x0 = label_w + gap + j * (cell + gap)
            if r == 0:
                d.text((x0, head - 18), f"{t * 250:.0f} µm wet", fill=(60, 60, 60), font=f_small)
            g, b = preview(p, t), preview(p, t, stripe)
            hw = cell // 2; s0, s1 = y0 + cell // 3, y0 + 2 * cell // 3
            d.rectangle([x0, y0, x0 + hw - 1, y0 + cell - 1], fill=to8(g["wet"]))
            d.rectangle([x0 + hw, y0, x0 + cell - 1, y0 + cell - 1], fill=to8(g["dry"]))
            d.rectangle([x0, s0, x0 + hw - 1, s1], fill=to8(b["wet"]))
            d.rectangle([x0 + hw, s0, x0 + cell - 1, s1], fill=to8(b["dry"]))
            d.line([x0 + hw, y0, x0 + hw, y0 + cell - 1], fill=(246, 244, 238), width=1)
            d.text((x0 + 2, y0 + cell + 1), "wet", fill=(110, 110, 110), font=f_small)
            d.text((x0 + hw + 2, y0 + cell + 1), "dry", fill=(110, 110, 110), font=f_small)
    img.save(path)
    return np.asarray(img)
