# Where things stand

A checkpoint written on 2026-11-15 for whoever picks this up next, on whatever machine. `SPEC.md` §12 is the
status of record; this is the working picture around it.

## Run it

```sh
pip install -e ".[dev]"                     # numpy, scipy, numba, pillow, matplotlib, scikit-image, pytest
python -m pytest -q                         # the kernel: 552 tests, about 7 minutes on two slow cores
python -m pytest classic/tests -q           # the deprecated first engine: 215 tests
PYTHONPATH=. python paintings/nocturne/paint.py 256 192 dry out/cartoon.png 0.6     # a picture, 1 minute
python -m goo.journal out/s1 new --size 160x120 --dx 0.5mm --ground paper --porosity 0 \
    --substances water,gum_arabic,rutile,ultramarine          # then: do '<json acts>', look, log, vocab
python -m goo.lab card rutile               # a drawdown card into out/
```

The first call into the world or the maps compiles numba kernels (some tens of seconds, cached afterwards).
Every timing in the docs was measured on two cores of a 2.3 GHz Xeon whose second core gives no speed-up; a
laptop should be several times faster.

## What exists

| Module | State |
|---|---|
| `goo.substance` | Descriptions with provenance, five admissibility laws, derived maps, a library of 22 substances on one physical scale. Particles are a colloid, liquids a solution; gum arabic vitrifies. |
| `goo.light` | Layer operators, star product, Fresnel surfaces, looks with a light ledger; gain booked as debt. G1 to G7 tested. |
| `goo.world` | A depth-averaged thin film as a gradient flow; matter and energy ledgers; O1 to O8 tested against analytic laws; real substances step; `advance`, `settle`, forks. |
| `goo.ports`, `goo.journal` | Paints, vessels, pour, flood, drop, drip, spray, tape, tilt, air, clocks, settle; a journal with bit-identical replay; a command line. No brush. |
| `goo.observe`, `goo.lab` | Colour and readings from looks; drawdown cards, sheets, admissibility reports, search, a contribution format. |
| `goo.classic` | The first engine, frozen and deprecated. |

## Measured, on a 1024 x 768 panel with the real maps

- A quiet step (nothing flows): 5.5 to 6.3 s, 1.3 to 1.7 GB.
- A full-panel 30 um paste of water, gum arabic and rutile dried until `settle()` retired every cell: 151 steps,
  12 minutes.
- A look at a painted panel: about a minute, and only tile by tile (a whole-panel look was killed at 6 GB).
- Laying 328 acts with the clock held: 15 minutes.
- Moving paint: the exact solver takes 4 to 10 s a step at 25,000 active cells and does not reach 300,000.

## Next, in order

1. **Stage 3 of `docs/PERFORMANCE.md`: multigrid flow.** The prototype in `bench/flow_mg.py` matches the factorised
   step to 3e-13 where no constraint binds (0.18 s at 25,000 cells, 2.0 s at 395,000). Move it into `goo/world` as the
   `iterative` solver with the ray condition, a Courant bound in `advance`, several densities and a yield stress.
   Until this lands only paint that holds its shape can be dried at full size, and a painting is mostly glazes.
2. **Stage 2: activity per substep**, so a drop beside a wet painting costs what the drop costs (2 s a step now,
   46 ms on an empty panel).
3. **Fuse the world's per-cell numpy work** (about 2 to 3 s of the quiet step).
4. **Stage 4: looks.** Maps keyed on composition, dried strata cached, tiles by default.
5. **Journal:** a session's description should carry substances that are not in the library (the Nocturne
   registers its luminous paint by hand, so its journal does not replay on its own).
6. **Re-lay the Nocturne wet** and let it dry: the test that the kernel works as a painting engine.
7. **A referee:** tests of SPEC §3, §6, §7 written from the contract alone, plus limits (the lab's dry film is what
   the world's drying converges to) and comparisons with classic. Nothing has been refereed.
8. **A brush and a blade:** the step needs to accept a wall riding on the film (docs/DYNAMICS.md sketches it).

## Known wrong or missing

- A pigment in a solution (gum water) still diffuses at the liquid's molecular mobility; in a single liquid it is
  right. Three attempted fixes are recorded in `docs/DYNAMICS.md` §3.
- Water held in a ground's pores never evaporates; there is no lateral wicking; strata do not impede sorption.
- Cure and dissolution are not terms of the step yet (xfail); the reaction's free energy now exists.
- Surface tension is held fixed within a step (no Marangoni stress); the change is booked to `work[constitutive]`.
- Guesses that matter: the optics of ultramarine, rose, green, dioxazine and ochre; gum arabic's glass fraction
  (0.194); linseed oil's reaction energy; every particle size. `classify` says what keeps each substance from
  being real. Under the contract's rule only water and clear PMMA are.
- The observer's colour-matching fit is unverified and the daylight is a 6504 K blackbody.
- `docs/LAB.md` §8 lists the lab's known wrong results; `docs/PORTS.md` §8 what the tools lack.

## Things that bite

- A pickled session only loads with the code that wrote it. Journals replay; pickles do not travel.
- `settle(force=True)` freezes a wet film, water included. It says so in its receipt.
- The world's results are bit-identical across forks, windows and replays on one machine. Nothing promises
  bit-identity across machines or numba versions.
