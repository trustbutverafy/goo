"""goo.palette: a painter's knowledge of their paints.

Mixing by inverting the optics: for a colour wanted over a known underlayer, find the mix of up to three
pigments (on a simplex grid) whose Kubelka-Munk layer, at the thickness the brush will lay, composited over
that underlayer, comes closest in CIELAB. The painter knows its paints the way the World's optics do, but it
sees only colour on the canvas, never volumes.
"""
from itertools import combinations
import numpy as np
from goo.engine import PIGMENTS, Canvas


def srgb_to_linear(c): c = np.asarray(c, float); return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def linear_to_srgb(c): c = np.clip(c, 0, 1); return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def lab(rgb_linear):
    """Linear sRGB reflectance (..., 3) -> CIELAB (D65)."""
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = np.asarray(rgb_linear) @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


class Palette:
    def __init__(self, pigments, steps=8, max_pigments=3, gain=10.0):
        self.pigments = list(pigments); self.gain = gain
        mixes = []
        for k in range(1, max_pigments + 1):
            for combo in combinations(range(len(pigments)), k):
                if k == 1: mixes.append({combo[0]: 1.0}); continue
                for w in self._simplex(k, steps):
                    if (w > 0).all(): mixes.append({c: float(x) for c, x in zip(combo, w)})
        self.mixes = mixes
        self.K = np.array([[sum(w * PIGMENTS[self.pigments[c]]["K"][b] for c, w in m.items()) for b in range(4)] for m in mixes])
        self.S = np.array([[sum(w * PIGMENTS[self.pigments[c]]["S"][b] for c, w in m.items()) for b in range(4)] for m in mixes])

    @staticmethod
    def _simplex(k, n):
        if k == 1: return [np.array([1.0])]
        out = []
        for i in range(1, n):
            for rest in Palette._simplex(k - 1, n - i): out.append(np.r_[i / n, rest * (n - i) / n])
        return out

    def layer(self, solids, oil=0.0):
        """R, T (mix, band) of each mix as a layer with `solids` pigment volume per cell (and `oil` carrier)."""
        K = self.K * solids * self.gain + oil * self.gain * np.array([0.02, 0.03, 0.06, 0.01])
        return Canvas.km_layer(K, self.S * solids * self.gain)

    def match(self, target_linear, under_linear, solids, oil=0.0, opacity_bias=0.0):
        """Best mix for `target` (linear RGB) laid at `solids` over `under` (observed linear RGB).
        Returns (dict pigment->share, predicted linear RGB, delta E)."""
        R, T = self.layer(solids, oil)
        Rb = np.asarray(under_linear)[None, :3]
        Rt = R[:, :3] + T[:, :3] ** 2 * Rb / (1 - R[:, :3] * Rb)
        dE = np.linalg.norm(lab(Rt) - lab(np.asarray(target_linear)[None, :3]), axis=1)
        if opacity_bias: dE = dE + opacity_bias * T[:, :3].mean(1)
        i = int(np.argmin(dE))
        return {self.pigments[c]: w for c, w in self.mixes[i].items()}, Rt[i], float(dE[i])
