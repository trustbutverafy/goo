"""studies/swatches.py: a manufacturer's chart for each medium, and where each medium's gamut ends.

For each medium a card of every stock pigment (thin, medium, thick films; over gesso and over a black
stripe; wet beside dry) goes to out/swatches_<medium>.png. Then `mix_for` is asked for a grid of hues at
three lightnesses (CIELAB L* 35, 55, 75, chroma as high as sRGB allows up to 60) and the CIEDE2000 it
reaches is printed per medium and hue: the regions the stock pigments cannot mix.

    PYTHONPATH=. python studies/swatches.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from goo import paints as P
from goo.palette import linear_to_srgb

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
MAKERS = dict(watercolour=P.watercolour, gouache=P.gouache, acrylic=P.acrylic, oil=P.oil)


def starter(medium):
    """Every stock pigment alone, and two classic mixes: a green-grey and a violet."""
    pigs = [p for p in P.PIGMENTS if p != P.BINDER]
    rows = [MAKERS[medium]({p: 1}) for p in pigs]
    rows += [MAKERS[medium]({"ultramarine": 0.5, "sienna": 0.5}), MAKERS[medium]({"ultramarine": 0.5, "rose": 0.5})]
    return rows


_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])


def lab_to_linear(L, a, b):
    """Inverse of goo.palette.lab (D65)."""
    fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
    f = np.array([fx, fy, fz])
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) * 27 / 24389) * np.array([0.95047, 1.0, 1.08883])
    return np.linalg.solve(_M, xyz)


def target(L, hue, cmax=60.0):
    """The most chromatic sRGB colour (up to cmax) at lightness L and hue (degrees), in sRGB 0..1."""
    for C in np.arange(cmax, 0, -1.0):
        rgb = lab_to_linear(L, C * np.cos(np.radians(hue)), C * np.sin(np.radians(hue)))
        if (rgb >= 0).all() and (rgb <= 1).all(): return linear_to_srgb(rgb), C
    return linear_to_srgb(lab_to_linear(L, 0, 0)), 0.0


def gamut():
    hues = range(0, 360, 30); Ls = (35, 55, 75)
    print("CIEDE2000 reached by mix_for (rows: hue; columns: L* 35/55/75; chroma as high as sRGB allows, <= 60)")
    for medium in MAKERS:
        t0 = time.time(); errs = {}; n = 0
        for h in hues:
            for L in Ls:
                c, C = target(L, h)
                paint, pred, dE = P.mix_for(c, medium); errs[h, L] = (dE, C, paint); n += 1
        dt = (time.time() - t0) / n
        allE = np.array([v[0] for v in errs.values()])
        print(f"\n{medium}: median dE {np.median(allE):.1f}, mean {allE.mean():.1f}, worst {allE.max():.1f}; {dt * 1e3:.0f} ms per call")
        for h in hues:
            row = "  ".join(f"{errs[h, L][0]:5.1f}(C{errs[h, L][1]:.0f})" for L in Ls)
            print(f"  hue {h:3d}: {row}   best at L55: {errs[h, 55][2].label}")


def main():
    os.makedirs(OUT, exist_ok=True)
    for medium in MAKERS:
        path = os.path.join(OUT, f"swatches_{medium}.png")
        P.swatch_card(starter(medium), path, title=f"goo {medium}: stock pigments")
        print("wrote", path)
    gamut()


if __name__ == "__main__":
    main()
