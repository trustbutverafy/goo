# goo

A physical paint medium, the tools that move it, and the sensors that scan the result.

Watercolour and oil are one material here: a depth-averaged layer of pigment suspended in water or oil, on a
gesso ground, over the strata it has dried into. Leveling is Bingham thin-film flow with a Krieger–Dougherty
viscosity and a yield stress that grows as the paint jams, so a dilute wash levels its brush marks, a short
paste holds them, and curing oil freezes. Brushes drag, plough and exchange paint with a load. Colour is
Kubelka–Munk, layer by layer, in R, G, B and 1.5 µm, so an underdrawing hidden in the visible shows in the
infrared. A simulated FMCW lidar scans the surface, speckle and all.

Nothing is drawn: every edge, groove, burr and bloom comes out of the laws (see `SPEC.md`).

```
pip install -e ".[dev]"
python -m pytest -q
python paintings/pear.py           # ~8 min on 2 cores; writes out/
python studies/scan.py             # scan it
```

| | |
|---|---|
| `goo/engine.py` | the World: `Canvas`, its laws and ledger, its optics |
| `goo/brush.py` | `Brush`, `Stylus` |
| `goo/sensors/` | `lidar` (averaged spot, direct detection), `coherent` (FMCW, speckle) |
| `paintings/` | painters: scripts that move tools |
| `studies/` | measurements: scans, sensor studies, figures |

MIT licensed.
