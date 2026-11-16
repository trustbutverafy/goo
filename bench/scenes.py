"""Scenes for the benchmarks (bench/): built from the library through goo.substance.maps, as the painter's tests are.

quiet(H, W)    a paste with a yield stress everywhere, losing water to the air (quiet.py's scene)
water(H, W, frac)   a disc of pure water, 40 um, on a sealed ground tilted 5 deg: moving paint, one substance; the disc
               covers `frac` of the panel
paint3(H, W, frac)  the same disc of gum water with hematite (three substances)
"""
import warnings; warnings.simplefilter("ignore")
import numpy as np
from goo.units import um, mm
from goo.film import Ground, Environment
from goo.substance import maps, library as L
from goo.world import World

G0 = 9.80665
DX = 0.25 * mm


def disc(H, W, frac):
    y, x = np.mgrid[:H, :W]
    r2 = frac * H * W / np.pi
    return ((y + 0.5 - H / 2) ** 2 + (x + 0.5 - W / 2) ** 2) <= r2


def quiet(H, W, phi=0.35, solver="exact"):
    subs = (L.WATER, L.GUM_ARABIC, L.RUTILE)
    g = Ground(np.zeros((H, W)), L.CELLULOSE, 0.0, 5 * um, 400 * um)
    env = Environment(activity={"water": 0.5})
    h = 30 * um * (1.0 + 0.3 * np.random.default_rng(0).random((H, W)))
    q = np.stack([h * (1 - phi - 0.05), h * 0.05, h * phi])
    return World(maps, subs, g, DX, env, q=q, solver=solver)


def _tilted(H, W, subs, fr, frac, h0=40 * um, solver="exact", noise=0.05):
    m = disc(H, W, frac)
    rng = np.random.default_rng(1)
    h = np.where(m, h0 * (1 + noise * rng.random((H, W))), 0.0)
    q = np.stack([h * f for f in fr])
    a = np.radians(5)
    env = Environment(gravity=(G0 * np.sin(a), 0.0, G0 * np.cos(a)), activity={"water": 0.5})
    g = Ground(np.zeros((H, W)), L.CALCITE, 0.0, 1 * um, 100 * um, subs, None)
    return World(maps, subs, g, DX, env, q=q, solver=solver)


def water(H, W, frac=0.5, solver="exact", noise=0.05):
    return _tilted(H, W, (L.WATER,), (1.0,), frac, solver=solver, noise=noise)


def paint3(H, W, frac=0.5, solver="exact"):
    return _tilted(H, W, (L.WATER, L.GUM_ARABIC, L.HEMATITE), (0.8, 0.15, 0.05), frac, solver=solver)


SCENES = dict(quiet=quiet, water=water, paint3=paint3)
