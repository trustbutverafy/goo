"""goo.classic.surface: what the World presents to a sensor (classic/SPEC.md §1, Optics).

A `Surface` is the whole of what a sensor may know about a painting: the height of the paint's top surface,
its reflectance per band, its micro-roughness and its gloss, on the canvas grid. `Surface.from_canvas(cv)` is
the one place where a canvas is turned into a Surface. It is a pure function of the canvas state: it reads, it
never writes, flushes or steps (a canvas with pending slow-law ticks is presented as it stands).

Sensors are constructed from a Surface and read nothing else from the World.
"""
from __future__ import annotations
import numpy as np
from goo.classic.sensors.lidar import CELL
from goo.classic.engine import SKIN, FAT, HAS_PAINT, film_gloss   # the finish rule lives with the World's optics (Canvas.finish)

ROUGH_MATTE_UM = 1.5   # micro-roughness of a matte surface (gesso, dried watercolour, lean paint), um RMS
ROUGH_GLOSS_UM = 0.3   # micro-roughness of a fully glossy oil skin, um RMS


def roughness_from_gloss(gloss):
    """Micro-roughness (um RMS): ROUGH_MATTE_UM where matte, ROUGH_GLOSS_UM where fully glossy, linear between."""
    return ROUGH_MATTE_UM * (1 - gloss) + ROUGH_GLOSS_UM * gloss


def _frozen(a, shape=None):
    a = np.array(a, dtype=np.float64, copy=True)
    if shape is not None and a.shape != shape: raise ValueError(f"expected shape {shape}, got {a.shape}")
    a.setflags(write=False)
    return a


class Surface:
    """What a sensor knows. All arrays are on the canvas grid (row = y down the panel, column = x right), read-only.

        Z_mm         (H, W)     height of the top surface above the panel plane, mm (ground + strata + wet layer)
        reflectance  (H, W, nb) diffuse reflectance in the optics model's bands, 0..1 (SPEC §6); classic: R, G, B, 1.5 um
        rough_um     (H, W)     micro-roughness, um RMS
        gloss        (H, W)     0 (matte) .. 1 (a fully formed oil skin)
        cell_mm      float      cell pitch, mm

    Conveniences: `nir` = reflectance[..., 3], `rgb` = reflectance[..., :3], `Z` = Z_mm, `rough` = rough_um,
    `gx`, `gy` = dZ/dx, dZ/dy (central differences, mm/mm).

    Bands. A Surface carries its optics model's bands (`bands`, a goo.classic.optics.Bands; classic R, G, B, N by
    default). With spectral bands, `reflectance` is (H, W, nb), `rgb` is the colour under daylight (D65,
    linear sRGB, through the model's observer) and `nir` is the 1.5 um band; `colour(light)` gives the colour
    under another light, and `under(light)` a Surface whose `rgb` is that colour (for a visible-light sensor
    lit by a coloured lamp).

    Construction: `Surface.from_canvas(cv)` for a painting. By hand (tests, synthetic surfaces):
    `Surface(Z_mm, reflectance=..., gloss=..., rough_um=...)`. The older positional form
    `Surface(Z, nir, rough_um)` is still accepted: it knows the 1.5 um band only (R, G, B are NaN, and a
    visible-light sensor refuses it) and is matte unless `gloss` is given.
    """

    def __init__(self, Z_mm, nir=None, rough_um=None, *, reflectance=None, gloss=None, cell_mm=CELL, bands=None, light="D65"):
        Z = _frozen(Z_mm)
        if Z.ndim != 2: raise ValueError("Z_mm must be (H, W)")
        H, W = Z.shape
        if reflectance is None:
            if nir is None: raise ValueError("give reflectance (H, W, 4) or, in the older form, nir (H, W)")
            refl = np.full((H, W, 4), np.nan); refl[..., 3] = np.broadcast_to(nir, (H, W))
        else:
            if nir is not None: raise ValueError("give reflectance or nir, not both")
            refl = reflectance
        self.Z_mm = Z
        # >>> optics: the model's bands (classic R, G, B, N unless given)
        from goo.classic import optics as _optics
        self.bands = _optics.CLASSIC_BANDS if bands is None else (_optics._BANDS[bands] if isinstance(bands, str) else bands)
        self.light = light
        self.reflectance = _frozen(refl, (H, W, self.bands.n))
        if self.bands.name == "classic" and light == "D65":
            self._rgb = self.reflectance[..., :3]
        else:
            self._rgb = _frozen(_optics.colour(self.reflectance, self.bands, light))
        # <<< optics
        self.gloss = _frozen(np.zeros((H, W)) if gloss is None else np.broadcast_to(gloss, (H, W)), (H, W))
        if (self.gloss < 0).any() or (self.gloss > 1).any(): raise ValueError("gloss must be in 0..1")
        self.rough_um = _frozen(roughness_from_gloss(self.gloss) if rough_um is None else np.broadcast_to(rough_um, (H, W)), (H, W))
        self.cell_mm = float(cell_mm)
        gy, gx = np.gradient(self.Z_mm, self.cell_mm)
        self.gx, self.gy = _frozen(gx), _frozen(gy)

    shape = property(lambda self: self.Z_mm.shape)
    Z = property(lambda self: self.Z_mm)
    rough = property(lambda self: self.rough_um)
    nir = property(lambda self: self.reflectance[..., self.bands.nir])   # the 1.5 um band, in any band set
    rgb = property(lambda self: self._rgb)                                # linear sRGB under self.light (D65)

    def colour(self, light="D65", adapt=False):
        """Linear sRGB (H, W, 3) of this surface's diffuse reflectance under `light` (goo.classic.optics.colour)."""
        from goo.classic import optics as _optics
        return _optics.colour(self.reflectance, self.bands, light, adapt)

    def under(self, light, adapt=False):
        """The same surface with `rgb` the colour under `light` (a visible-light sensor's coloured lamp).
        Only the diffuse colour changes: a sensor's specular term stays white (SPEC §8)."""
        out = Surface(self.Z_mm, reflectance=self.reflectance, gloss=self.gloss, rough_um=self.rough_um,
                      cell_mm=self.cell_mm, bands=self.bands, light="D65")
        out._rgb = _frozen(self.colour(light, adapt)); out.light = light
        return out

    @classmethod
    def from_canvas(cls, cv, gain=10.0):
        """The World presented to a sensor, through the canvas's three optics functions and nothing else:

            cv.surface()           top-surface height, volume units  -> Z_mm = cv.surface() * CELL
            cv.reflectance(gain)   (H, W, 4)                         -> reflectance
            cv.finish()            gloss of the top film, (H, W)     -> gloss, and from it the roughness

        It reads no raw state and changes nothing (no flush, no step; arrays are copied).

        Finish. The *top film* at a cell is the wet layer if it has paint there (wet-layer volume h > HAS_PAINT),
        otherwise the topmost stratum with paint there (stratum volume > HAS_PAINT), otherwise the bare ground.
        For that film, with h its volume at the cell and f = oil / h its oil fraction (water counts in h for
        the wet layer; strata hold no water):

            gloss = clip(h / SKIN, 0, 1) * clip(f / FAT, 0, 1),   SKIN = 0.3 units (75 um), FAT = 0.3
            gloss = 0 on bare ground (gesso is matte)

        So a film is glossy when it is thick enough to form a skin and oil-rich; lean paint, dried watercolour
        (no oil) and bare gesso are matte. Wet and cured oil films of the same composition and thickness have
        the same gloss: curing and fixing do not change the finish.
        Roughness: rough_um = ROUGH_MATTE_UM (1 - gloss) + ROUGH_GLOSS_UM gloss (1.5 um matte, 0.3 um glossy).

        Limits of the rule: only the top film counts (a film thinner than SKIN is less glossy even over a glossy
        film beneath; a vanishing wet residue above HAS_PAINT masks the film under it); no sinking-in (a fat
        film over an absorbent lean one does not lose gloss); no varnish; no dependence on cure, on the pigment
        or its load beyond the oil fraction, or on the brushwork's micro-relief below one cell.
        """
        gloss = cv.finish()
        from goo.classic.optics import optics_of                                # the model's bands (metadata, not state)
        return cls(cv.surface() * CELL, reflectance=cv.reflectance(gain=gain), gloss=gloss,
                   rough_um=roughness_from_gloss(gloss), cell_mm=CELL, bands=optics_of(cv).bands)
