"""goo.classic.palette: a painter's knowledge of their paints.

Mixing by inverting the optics: for a colour wanted over a known underlayer, find the mix of up to three
pigments (on a simplex grid) whose Kubelka-Munk layer, at the thickness the brush will lay, composited over
that underlayer, comes closest in CIELAB. The painter knows its paints the way the World's optics do, but it
sees only colour on the canvas, never volumes.
"""
from itertools import combinations
import numpy as np
from goo.classic.engine import PIGMENTS, Canvas


def srgb_to_linear(c): c = np.asarray(c, float); return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def linear_to_srgb(c): c = np.clip(c, 0, 1); return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def lab(rgb_linear):
    """Linear sRGB reflectance (..., 3) -> CIELAB (D65)."""
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = np.asarray(rgb_linear) @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def delta_e2000(lab1, lab2):
    """CIEDE2000 between CIELAB arrays (..., 3) (broadcast), via scikit-image."""
    from skimage.color import deltaE_ciede2000
    a, b = np.broadcast_arrays(np.asarray(lab1, float), np.asarray(lab2, float))
    return deltaE_ciede2000(a, b)


class Palette:
    def __init__(self, pigments, steps=8, max_pigments=3, gain=10.0, optics=None):
        """optics: None (classic, the original code path) or the canvas's optics model (`goo.classic.optics`, e.g.
        `Palette(pigments, optics=cv.optics)` for a v2 canvas): mixes are then previewed by that model."""
        self.pigments = list(pigments); self.gain = gain
        # >>> optics: a palette previews through its canvas's optics model (one code path, goo/classic/optics.py)
        if optics is not None and getattr(optics, "name", None) == "classic": optics = None
        self.optics = optics
        # <<< optics
        mixes = []
        for k in range(1, max_pigments + 1):
            for combo in combinations(range(len(pigments)), k):
                if k == 1: mixes.append({combo[0]: 1.0}); continue
                for w in self._simplex(k, steps):
                    if (w > 0).all(): mixes.append({c: float(x) for c, x in zip(combo, w)})
        self.mixes = mixes
        if self.optics is not None: return                     # optics: the model previews; no classic K, S tables
        self.K = np.array([[sum(w * PIGMENTS[self.pigments[c]]["K"][b] for c, w in m.items()) for b in range(4)] for m in mixes])
        self.S = np.array([[sum(w * PIGMENTS[self.pigments[c]]["S"][b] for c, w in m.items()) for b in range(4)] for m in mixes])

    @staticmethod
    def _simplex(k, n):
        if k == 1: return [np.array([1.0])]
        out = []
        for i in range(1, n):
            for rest in Palette._simplex(k - 1, n - i): out.append(np.r_[i / n, rest * (n - i) / n])
        return out

    def layer(self, solids, oil=0.0, cured=False):
        """R, T (mix, band) of each mix as a layer with `solids` pigment volume per cell (and `oil` carrier);
        `cured`: as it will look once dry, its linseed yellowed (the World's optics, at full cure)."""
        K = self.K * solids * self.gain + oil * self.gain * np.array([0.02, 0.03, 0.06, 0.01]) * (3.0 if cured else 1.0)
        return Canvas.km_layer(K, self.S * solids * self.gain)

    # >>> optics: previews through the model (classic or v2), and the v2 match
    def preview(self, film, under=None, ground="gesso", cured=False, kappa=0.0):
        """What a film looks like (..., nb in the model's bands): `film` {species: volume (scalar or array)},
        laid on a bare dry `ground` or over a body reflectance `under` (..., nb); `cured`: as a dried stratum.
        The same functions as Canvas.reflectance() (goo.classic.optics), so the two agree to round-off."""
        from goo.classic import optics as O
        m = self.optics if self.optics is not None else O._CLASSIC
        sp = list(O._engine.CARRIERS) + [k for k in film if k not in O._engine.CARRIERS]
        shp = np.broadcast(*[np.asarray(v, float) for v in film.values()]).shape
        vol = np.zeros((len(sp),) + shp)
        for k, v in film.items(): vol[sp.index(k)] = v
        if m.name == "classic":
            Rb = np.broadcast_to(O._engine.GROUND_RGBN if under is None else np.asarray(under, float), shp + (4,))
            out = m.preview(sp, vol, Rb, gain=self.gain, wet=not cured, kappa=kappa)
            return np.where((vol.sum(0) > 0)[..., None], out, Rb) if under is None else out
        return m.preview(sp, vol, under=under, ground=ground, gain=self.gain, wet=not cured, kappa=kappa)

    def _match_model(self, target_linear, under_bands, solids, oil, opacity_bias, cured):
        from goo.classic import optics as O
        m = self.optics; under = np.asarray(under_bands, float)
        if under.shape[-1] != m.bands.n:
            raise ValueError(f"a {m.name} palette mixes over what is seen in its model's {m.bands.n} bands; it was "
                             f"given {under.shape[-1]} values (a colour?). Pass the band reflectance under the stroke "
                             f"(Painter does; DetailPainter does not yet)")
        sol = np.atleast_1d(np.asarray(solids, float)); oil_ = np.broadcast_to(np.atleast_1d(np.asarray(oil, float)), sol.shape)
        sp = list(O._engine.CARRIERS) + self.pigments
        W = np.zeros((len(self.pigments), len(self.mixes)))
        for j, mx in enumerate(self.mixes):
            for c, w in mx.items(): W[c, j] = w
        Rt = 0; Tm = 0
        for s_, o_ in zip(sol, oil_):
            vol = np.zeros((len(sp), len(self.mixes))); vol[1] = o_; vol[2:] = W * s_
            R = m.preview(sp, vol, under=np.broadcast_to(under, (len(self.mixes), m.bands.n)), gain=self.gain, wet=not cured)
            Rt = Rt + m.colour(R) / len(sol)
            K, S, *_ = m.film(sp, vol, np.zeros(len(self.mixes)), cured, self.gain)
            Tm = Tm + O.km_layer(K, S)[1][:, :m.bands.nir].mean(1) / len(sol)
        dE = np.linalg.norm(lab(np.clip(Rt, 0, None)) - lab(np.asarray(target_linear)[None, :3]), axis=1)
        if opacity_bias: dE = dE + opacity_bias * Tm
        i = int(np.argmin(dE))
        return {self.pigments[c]: w for c, w in self.mixes[i].items()}, Rt[i], float(dE[i])
    # <<< optics

    def match(self, target_linear, under_linear, solids, oil=0.0, opacity_bias=0.0, cured=False):
        """Best mix for `target` (linear RGB) laid at `solids` over `under` (observed linear RGB).
        `solids` (and `oil`) may be arrays of samples: the thicknesses a stroke actually lays (bristle ribbons,
        tapering ends); the predicted colour is the average over them, as the eye averages a striped film.
        `cured`: match the colour the paint will have once dry rather than wet.
        Returns (dict pigment->share, predicted linear RGB, delta E).
        With a v2 palette (`optics=`), `under` is the band reflectance seen under the stroke (the model's bands)."""
        if self.optics is not None: return self._match_model(target_linear, under_linear, solids, oil, opacity_bias, cured)
        sol = np.atleast_1d(np.asarray(solids, float)); oil_ = np.broadcast_to(np.atleast_1d(np.asarray(oil, float)), sol.shape)
        Rb = np.asarray(under_linear)[None, :3]
        Rt = 0; Tm = 0
        for s_, o_ in zip(sol, oil_):
            R, T = self.layer(s_, o_, cured)
            Rt = Rt + (R[:, :3] + T[:, :3] ** 2 * Rb / (1 - R[:, :3] * Rb)) / len(sol); Tm = Tm + T[:, :3].mean(1) / len(sol)
        dE = np.linalg.norm(lab(Rt) - lab(np.asarray(target_linear)[None, :3]), axis=1)
        if opacity_bias: dE = dE + opacity_bias * Tm             # a painter covers with opaque paint rather than tint
        i = int(np.argmin(dE))
        return {self.pigments[c]: w for c, w in self.mixes[i].items()}, Rt[i], float(dE[i])
