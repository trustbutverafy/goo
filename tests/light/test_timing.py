"""Timing of a look on 400 x 300 cells, 16 bands, 6 films. Skipped unless GOO_BENCH is set (it is a
measurement, not a contract); run with GOO_BENCH=1 python -m pytest -q -s tests/light/test_timing.py."""
import os, time
import numpy as np, pytest
from goo.film import Film, Bands, Light, Stack
from goo.units import um, nm
from goo.light import look
from analytic import Sub, Mixture, ground


@pytest.mark.skipif(not os.environ.get("GOO_BENCH"), reason="benchmark: set GOO_BENCH=1")
@pytest.mark.parametrize("geometry", ["diffuse", ("collimated", 0.3)])
def test_timing_400x300_16_bands_6_films(geometry):
    rng = np.random.default_rng(0)
    H, W, B = 300, 400, 16
    b = Bands(np.linspace(400, 700, B) * nm, np.full(B, 20 * nm))
    films = []
    for j in range(6):
        subs = (Sub(f"binder{j}", k=rng.uniform(0, 1e3, B), n=1.45 + 0.02 * j, sigma=0.05 * um),
                Sub(f"pigment{j}", k=rng.uniform(1e3, 1e6, B), s=rng.uniform(1e4, 1e6, B), n=1.6 + 0.1 * j,
                    sigma=0.3 * um))
        q = rng.uniform(0, 20, (2, H, W)) * um
        films.append(Film(subs, q))
    st = Stack(ground(H, W, Sub("gesso", k=rng.uniform(10, 1e3, B), s=1e6, n=1.5, sigma=0.5 * um), 300 * um),
               films, 1e-4)
    light = Light(b, np.ones(B), geometry)
    look(st, b, light, Mixture())                      # warm-up
    t = time.perf_counter()
    L = look(st, b, light, Mixture())
    dt = time.perf_counter() - t
    print(f"\nlook {H}x{W} cells, {B} bands, 6 films, {geometry}: {dt:.2f} s, residual {L.receipt.residual['light']:.2e}")
    assert L.receipt.residual["light"] < 1e-12


@pytest.mark.skipif(not os.environ.get("GOO_BENCH"), reason="benchmark: set GOO_BENCH=1")
def test_timing_painting_like_400x300():
    """6 films laid as rectangular strokes of uniform paint over a uniform ground: many cells share inputs."""
    rng = np.random.default_rng(1)
    H, W, B = 300, 400, 16
    b = Bands(np.linspace(400, 700, B) * nm, np.full(B, 20 * nm))
    films = []
    for j in range(6):
        subs = (Sub(f"binder{j}", k=rng.uniform(0, 1e3, B), n=1.48, sigma=0.05 * um),
                Sub(f"pigment{j}", k=rng.uniform(1e3, 1e6, B), s=rng.uniform(1e4, 1e6, B), n=1.6 + 0.1 * j,
                    sigma=0.3 * um))
        q = np.zeros((2, H, W))
        for _ in range(8):
            y, x = rng.integers(0, H - 60), rng.integers(0, W - 80)
            q[:, y:y + 60, x:x + 80] = np.array([15, 5])[:, None, None] * um
        films.append(Film(subs, q))
    st = Stack(ground(H, W, Sub("gesso", k=rng.uniform(10, 1e3, B), s=1e6, n=1.5, sigma=0.5 * um), 300 * um),
               films, 1e-4)
    light = Light(b, np.ones(B), "diffuse")
    look(st, b, light, Mixture())
    for dedup in (True, False):
        t = time.perf_counter()
        L = look(st, b, light, Mixture(), dedup=dedup)
        print(f"\npainting-like look, dedup={dedup}: {time.perf_counter() - t:.2f} s")
    assert L.receipt.residual["light"] < 1e-12
