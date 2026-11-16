"""Profile one step of the present code: PYTHONPATH=$PWD python bench/profile_step.py SCENE H W [dt] [solver]

Takes one warm-up step, then profiles the next; prints wall time, peak RSS, the time per substep and per module, and
the top functions by own time. Measured on this machine."""
import sys, time, resource, cProfile, pstats, collections
sys.path.insert(0, __import__("os").path.dirname(__file__))
from scenes import SCENES

scene, H, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dt = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
solver = sys.argv[5] if len(sys.argv) > 5 else "exact"
t = time.perf_counter(); w = SCENES[scene](H, W, solver=solver); tb = time.perf_counter() - t
t = time.perf_counter(); w.step(dt); t1 = time.perf_counter() - t
pr = cProfile.Profile()
t = time.perf_counter(); pr.enable(); r = w.step(dt); pr.disable(); t2 = time.perf_counter() - t
wet = int((w.q.sum(0) > 0).sum())
print(f"{scene} {H}x{W} wet {wet} dt {dt} solver {solver}: build {tb:.2f}s step1 {t1:.2f}s step2 {t2:.2f}s "
      f"(profiled) peakRSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB pieces {r.report.get('pieces')}")
st = pstats.Stats(pr)
cum = {k: v for k, v in st.stats.items()}
def cumof(fn):
    return sum(v[3] for k, v in cum.items() if k[2] == fn and ("goo/world" in k[0]))
for name in ("flow", "diffuse", "local", "_relabel", "_F_box", "slope", "_coefficients"):
    print(f"  cum {name:14s} {cumof(name):7.2f}s")
mod = collections.Counter()
for k, v in cum.items():
    f = k[0]
    key = ("goo/" + f.split("/goo/")[-1]) if "/goo/" in f else ("scipy.sparse.linalg" if "linalg" in f and "scipy" in f else
           "scipy.sparse" if "scipy/sparse" in f else "numpy" if "numpy" in f else f if f.startswith("~") else "other")
    mod[key] += v[2]
print("  own time by module:", ", ".join(f"{k} {v:.2f}" for k, v in mod.most_common(9)))
st.sort_stats("tottime").print_stats(14)
