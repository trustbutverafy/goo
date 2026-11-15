# goo

A paint is a constitutive description. A painting is the history of accounted work done on one
persistent, stratified world. A picture is what a non-loading observer reads from that world.

goo is being rebuilt on that statement. The kernel treats paint as an arbitrary material: a substance
is a dielectric function, a morphology and its terms in a free energy and a dissipation potential,
and everything the engine uses is derived from that description. Known pigments are points in the
space. So are substances nobody has made, and substances that cannot exist, each labelled with the
law it breaks. Three ledgers (matter, energy, light) account for every operation.

- `SPEC.md` is the contract. Its §12 says what exists today, which is not much yet.
- `docs/VISION.md` gives the reasons. `docs/MATTER-SPEC.md` is the decision about substances.
- `AGENTS.md` has the working rules.

```sh
pip install -e ".[dev]"
python -m pytest -q
```

## goo classic

The first engine is in `goo.classic`. It is deprecated and frozen at commit `6fcca08` (branch
`goo-classic`): a thin-film paint simulator with brushes, a studio, an easel and sensors, tuned by
eye. It works, it is tested, and it is not developed further. Importing it raises a
`DeprecationWarning`. Its contract, paintings, studies and tests are under `classic/`:

```sh
python -m pytest classic/tests -q
python classic/paintings/sampler/paint.py
python -m goo.classic.easel out/s1 new --size 400x300 --pigments ultramarine,ochre --medium watercolour
```

The kernel imports nothing from classic and does not owe its pictures.

## Data and licence

MIT. Shipped optical numbers are computed from CC0 optical constants (refractiveindex.info) with
attribution, or are goo's own guesses and marked so. Restricted references are read and cited for
checks; nothing from them is included.
