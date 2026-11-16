"""The cost of a look (goo.light.look) against cells, films and bands, and where it goes: maps (optical, surface:
Mie per distinct composition) against the per-cell layer algebra. Every cell differs (random amounts).

    PYTHONPATH=$PWD python bench/look_cost.py H W FILMS BANDS [S]
Measured on this machine; peak RSS from getrusage."""
import sys, time, resource, cProfile, pstats
import numpy as np
import warnings; warnings.simplefilter("ignore")
from goo.units import um, nm
from goo.film import Film, Ground, Stack, Bands, Light
from goo.substance import maps, library as L
from goo.light.look import look

H, W, NF, B = (int(x) for x in sys.argv[1:5])
S = int(sys.argv[5]) if len(sys.argv) > 5 else 6
names = ["water", "gum_arabic", "rutile", "soot", "ultramarine", "hematite", "cadmium_sulfide", "rose", "green", "dioxazine"]
subs = tuple(L.BY_NAME[n] for n in names if n in L.BY_NAME)[:S]
S = len(subs)
bands = Bands(np.linspace(400, 700, B) * nm, np.full(B, 300 / B) * nm)
light = Light(bands, np.ones(B))
rng = np.random.default_rng(0)
films = []
PALETTE = len(sys.argv) > 6 and sys.argv[6] == "palette"
for k in range(NF):
    q = np.zeros((S, H, W))
    if PALETTE:      # a film laid from one recipe: one composition (phi), thickness varying cell to cell
        hh = 8 * um * rng.random((H, W))
        phi = np.zeros(S); phi[1] = 0.6
        for j in rng.choice(np.arange(2, S), size=2, replace=False):
            phi[j] = 0.2
        q = phi[:, None, None] * hh[None]
    else:
        q[1] = 5 * um * rng.random((H, W))                         # gum
        for j in rng.choice(np.arange(2, S), size=2, replace=False):  # two pigments a film
            q[j] = 2 * um * rng.random((H, W))
    films.append(Film(subs, q))
g = Ground(np.zeros((H, W)), L.CELLULOSE, 0.5, 5 * um, 200 * um)
st = Stack(g, films, 2.5e-4)
pr = cProfile.Profile()
t = time.perf_counter(); pr.enable(); lk = look(st, bands, light, maps); pr.disable(); el = time.perf_counter() - t
stt = pstats.Stats(pr)
cum = lambda fn: max([v[3] for k, v in stt.stats.items() if k[2] == fn] or [0])
print(f"look {H}x{W} films {NF} bands {B} S {S}: {el:.2f} s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB; "
      f"_inputs (maps) {cum('_inputs'):.2f} s [optical {cum('optical'):.2f}, surface {cum('surface'):.2f}], solve {cum('solve'):.2f} s, "
      f"_build {cum('_build'):.2f} s, dedup {cum('_distinct_cells'):.2f} s; per cell-band-film {el / (H * W * B * NF) * 1e9:.0f} ns; "
      f"closure residual {lk.receipt.residual['light']:.1e}")
