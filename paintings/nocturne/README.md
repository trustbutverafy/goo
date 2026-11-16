# Nocturne: the far shore

The first picture made with the kernel's tools. `docs/nocturne.png` is a look at it: 1024 x 768 cells of 0.4 mm
(a panel of 41 x 31 cm), under a 6504 K light, camera exposure 0.6, specular excluded.

```sh
PYTHONPATH=. python paintings/nocturne/paint.py 1024 768 dry out/nocturne.png 0.6    # about 16 minutes, 3 GB
PYTHONPATH=. python paintings/nocturne/paint.py 256 192 dry out/cartoon.png 0.6     # the same acts on a coarse grid, 1 minute
```

The act list is written in metres, so the same painting can be laid on any grid. `journal.json` is the journal
of the full-size session: 366 entries (253 sprays, 43 floods, 16 tapes laid and peeled, 11 settles).

## What it is, exactly

- **Matter.** Thirteen paints mixed from library substances: rutile, soot, hematite and cadmium sulfide (optics
  computed from CC0 constants); ultramarine, rose, green and dioxazine (optics guessed); gum arabic as binder. 5.9 mL
  of paint solids in all, 5.0 mL of it the white priming.
- **Light.** Every colour is the light solver's answer for the stack of films at that cell. Glazes are dark over
  the white priming and vanish over dark; cadmium yellow and thin rutile stay visible over dark. The moon, the
  lamps and the masthead light are reserved white priming (tape, peeled at the end) under a thin film of `lamp`,
  a substance with optical gain. It fails passivity and is flagged `impossible` in every look it appears in. The
  moon returns up to 1.31 times the light that falls on it, the excess booked as debt in the light ledger, which
  still closes (residual 9e-16). The camera is exposed so that this reads as brighter than white paint.
- **Tools.** Spray (seeded droplets through a cone), floods inside regions, tape. No brush: the world cannot yet
  take a wall riding on the film.

## What it is not

- **Nothing flowed and nothing dried.** Every film was laid already dry, with the clock held over the whole
  panel, and force-settled; each of those is flagged as a privilege in its receipt. The world can step real paint
  on small panels, and a drop was run down this panel on a tilt as a test, but a step on 786,000 cells takes
  16 to 30 s today, too slow to dry a dozen layers. So this is a picture built by composition and seen by real
  optics, not yet a painting in the full sense the kernel means.
- **Not replayable from the journal alone.** `lamp` is not in the library, and a session's description can only
  name library substances; `paint.py` registers it by hand before starting.
- **Not a claim about real paint.** Half the palette's optics are guesses and say so.
- A look at the whole panel at once runs out of memory at this size; `paint.py` looks tile by tile.
