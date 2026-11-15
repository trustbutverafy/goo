# goo.observe: sensors over looks

Owner: lab and observers. Code: `goo/observe/`. Tests: `tests/observe/`. `goo.observe` imports `goo.light` and
`goo.film` (and `goo.units`, `goo.evidence`, `goo.ledger`), nothing else of goo, and nothing of `goo.classic`.

A sensor is a measurement model over a `goo.light.Look`. It takes a Look, never a world, a stack or a film (anything
else raises `TypeError`), reads its arrays and returns new ones. It writes nothing it reads: the tests freeze every
array of a Look and compare digests before and after
(`test_sensors_take_a_look_and_refuse_a_world_a_stack_or_a_film`, `test_sensors_never_modify_what_they_read`).

## 1. Colorimetry (`goo.observe.colour`)

```text
XYZ = W R         W[3, B] = sum over 1 nm in each band of cmf(lambda) E(lambda), normalised so a perfect diffuser has Y = 1
CIELAB            relative to the light's own white (the perfect diffuser under that light)
sRGB              Bradford adaptation of the light's white to sRGB's white (an adapted observer), the IEC matrix
                  derived from the primaries' chromaticities, the sRGB transfer function; clipped only for display
CIEDE2000         Sharma, Wu and Dalal (2005), implemented here
```

Provenance of every table and fit (`goo.observe.colour.PROV`):

| item | what | evidence |
|---|---|---|
| observer | CIE 1931 2-degree, the multi-lobe analytic fit of Wyman, Sloan and Shirley (2013), copied from `goo/classic/optics.py` | assumed |
| A | Planck's law at 2856 K, c2 = h c / k_B from the exact SI constants | exact (the identification with CIE A is recalled) |
| daylight | Planck's law at 6504 K | assumed: **it is not CIE D65** |
| sRGB | IEC 61966-2-1 primaries and white, recalled; matrix derived here | assumed |
| Bradford | adaptation matrix, recalled | assumed |
| CIELAB | CIE 15 definition, recalled | assumed |
| CIEDE2000 | Sharma et al. 2005 | assumed; equal to scikit-image's implementation to 1e-10 on 5000 random pairs; Sharma's pair 1 gives 2.0425 |
| bands | reflectance constant within a band, held at the nearest band beyond 360-830 nm | assumed |

**Not verified.** The fit's error against the CIE 1931 table was not measured: the table is CIE's (not CC0) and could
not be read in this session (cvrl.org is unreachable from here; reading a copy shipped in a third-party package for
the check was refused). Classic's docstring says "well under 1 % of peak", recalled and unverified. The daylight is a
Planckian radiator, not D65 (D65 is a table; it was not shipped and not compared). Until both are checked, every
colour goo reports is `assumed` at best.

## 2. Readings (`goo.observe.readings`)

| reading | definition | unit |
|---|---|---|
| `colour_of(look, light)` | XYZ, CIELAB, linear sRGB [H, W, 3]; from `Look.diffuse` (specular excluded, as a 45/0 or d/8 instrument with a trap) or `Look.reflectance` ("sci"); masks: out of gamut, a band above 1 (gain), a band unresolved (past a lasing threshold) | 1 |
| `gloss(look, light)` | 100 Y(specular) / Y(Fresnel at 60 degrees of black glass, n = 1.567) from a look under `gloss_light` (collimated, 60 degrees) | GU |
| `infrared(look)` | the band reflectance above 780 nm, unweighted (a flat detector across each band) | 1 |
| `contrast_ratio(black, white, light)` | Y over black / Y over white | 1 |
| `luminance` | Y | 1 |

The ISO 2813 geometry and reference index are recalled (`assumed`); the reference's reflectance is Fresnel's equations,
not a typed number. Every reading is a `Reading` (or `Colour`) with an evidence class and the look's flags:

```text
evidence = weakest(matter, the measurement model's provenance, "assumed" if the look has out_of_domain or bounds_collapsed)
```

`matter` is supplied by the caller (the lab computes it from the maps' own Derived and Bounded evidence); a sensor
cannot know where matter came from, so it defaults to `assumed`.

## 3. Images

`write_png(look, light, path, scale)` writes the sRGB of each cell as a `scale` x `scale` block; `write_ir_png` writes
one infrared band as grey (reflectance 0..1 linear). Cards are drawn by `goo.lab.render`.
