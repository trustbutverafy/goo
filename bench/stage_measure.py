"""The lead's stage measurements on 1024 x 768, with the library's maps (NOPARTICLE=1: bench/noparticle.py's wrapper):
    quiet     one quiet step (paste with a yield stress, evaporating): time and peak RSS
    dry       advance a full-panel 30 um paste to equilibrium (1 h of world time): steps, wall time
    drop      a drop of ~300 cells running down a 30 degree tilt beside a wet painting (a 900-cell wet patch in the far
              corner): time per step
    PYTHONPATH=$PWD python bench/stage_measure.py quiet|dry|drop [H W]"""
import sys, os, time, resource
import numpy as np
import warnings; warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from goo.units import um, mm
from goo.film import Ground, Environment
from goo.substance import library as L
from goo.world import World
if os.environ.get("NOPARTICLE"):
    from noparticle import maps
else:
    from goo.substance import maps

what = sys.argv[1]
H, W = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (768, 1024)
G0 = 9.80665


def paste():
    subs = (L.WATER, L.GUM_ARABIC, L.RUTILE)
    g = Ground(np.zeros((H, W)), L.CELLULOSE, 0.0, 5 * um, 400 * um)
    h = 30 * um * (1.0 + 0.3 * np.random.default_rng(0).random((H, W)))
    return World(maps, subs, g, 0.25 * mm, Environment(activity={"water": 0.5}), q=np.stack([h * 0.6, h * 0.05, h * 0.35]))


rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
if what == "quiet":
    w = paste()
    for k in range(3):
        t = time.perf_counter(); r = w.step(2.0); el = time.perf_counter() - t
        print(f"quiet {W}x{H} step {k}: {el:.2f} s, peak RSS {rss():.0f} MB, diffusion rays {r.report.get('diffusion rays')}, "
              f"fallbacks {r.report.get('diffusion fallbacks')}", flush=True)
elif what == "dry":
    w = paste(); t0 = time.perf_counter(); steps = 0
    for k in range(24):
        r = w.advance(300.0); steps += int(r.report["steps"])
        print(f"dry {W}x{H}: t {w.t:.0f} s, steps {steps}, wall {time.perf_counter() - t0:.0f} s, water {w.q[0].mean() / um:.4f} um, "
              f"dt next {w._dt_next:.3g}, peak RSS {rss():.0f} MB", flush=True)
        if r.report.get("max relative change", 1.0) < 1e-6:
            s = w.settle()
            print(f"settle: {s.report}, wall {time.perf_counter() - t0:.0f} s", flush=True)
            if s.report.get("cells still wet", 1) == 0:
                break
elif what == "drop":
    subs = (L.WATER, L.GUM_ARABIC, L.HEMATITE)
    y, x = np.mgrid[:H, :W]
    m = ((y - 40) ** 2 + (x - 60) ** 2) <= 10 ** 2
    h = np.where(m, 60 * um, 0.0); q = np.stack([h * .8, h * .15, h * .05])
    far = (y > H - 30) & (x > W - 30)
    q[:, far] += (np.array([0.5, 0.5, 0.0]) * 20 * um)[:, None]
    a = np.radians(30)
    env = Environment(gravity=(G0 * np.sin(a), 0, G0 * np.cos(a)), activity={"water": 0.5})
    w = World(maps, subs, Ground(np.zeros((H, W)), L.CALCITE, 0.0, 1 * um, 100 * um, subs, None), 0.25 * mm, env, q=q)
    for k in range(3):
        t = time.perf_counter(); w.step(0.01); print(f"drop beside a wet patch {W}x{H} step {k}: {time.perf_counter() - t:.2f} s", flush=True)
    t = time.perf_counter(); r = w.advance(0.5)
    print(f"advance 0.5 s: {time.perf_counter() - t:.1f} s wall, {r.report['steps']:.0f} steps")
