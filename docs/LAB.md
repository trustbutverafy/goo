# goo.lab: what a substance would look like as paint, and how it is known

Owner: lab and observers. Code: `goo/lab/`. Tests: `tests/lab/`. `goo.lab` imports `goo.substance`, `goo.film`,
`goo.light`, `goo.observe` (and units, evidence, ledger). It never imports `goo.world` or `goo.ports`
(`tests/lab/test_lab_layering.py` checks that importing it loads neither). Films are made by composition, not by flow.

Claims (SPEC §11): every card says what goo's declared world produces from these descriptions, recipes and maps. None
says that a real product looks like this.

## 1. Recipes (`goo.lab.recipes`)

Nothing about a medium is typed in. Binders are recipes, volume fractions of library substances; what each does when
it dries comes from the substances' descriptions through the maps. Every recipe number is a choice of the drawdown,
`assumed`.

| recipe | vehicle (volume fractions) | masstone pigment volume fraction of the wet paint |
|---|---|---|
| linseed oil | linseed_oil 1 | 0.20 |
| water with gum arabic | water 0.80, gum_arabic 0.20 | 0.10 |
| acrylic latex in water | water 0.55, pmma_latex 0.45 | 0.12 |
| water (search (b) only) | water 1 | 0.20 |

A tint replaces a share `w` of the pigment volume with the white (rutile): 0 (masstone), 1:1, 1:3, 1:9, 1:32 by
volume. Wet film: 50 um. Grounds are real stacks: **white** is a bare calcite ground (porosity 0.3, 1 um pores,
300 um deep); **black** is 20 um of PMMA holding 5 % soot by volume, on that ground. The room: 20 C, water activity
0.5, oxygen activity 0.21.

## 2. The dry film (`goo.lab.dry`)

**The rule.** A film is dry when

1. every reaction whose rate in the environment, by `maps.reactions(film, env)`, is positive has run to completion:
   its extent grows until one of its reactants in the film is exhausted; reactants the film lacks are drawn from the
   environment's reservoirs without limit; products are added by the stoichiometry (by mass, volume = mass / rho);
   repeated while a product itself reacts; and then
2. every volatile substance (one whose description has `thermo.p_sat > 0`) that the environment names has left (or
   entered) until its activity in the film, `maps.evaporation(film, env)[name].activity`, equals the environment's
   activity of that substance, `env.activity[name]` (> 0). A volatile the environment does not name has no
   reservoir (`maps.exchange` gives it k = 0) and stays. With several volatiles they are equalised
   together (Gauss-Seidel to a fixed point).

Nothing else changes: non-volatile, unreacting matter stays bit for bit. Whatever the remaining liquids do not fill
between the packed particles is void, and the maps treat it as air (Wiener-bounded host, receding liquid surface).
Volatility, reactivity and products are read from descriptions, never names (tested by renaming water). This is the
t -> infinity limit, at a fixed environment, of the world's exchange and reaction terms; the referee is to test that
the world's drying converges to it (`tests/limits/`). Tolerances: each activity within 1e-10 of the environment's; the
matter ledger of drying (`Dried.receipt`, kg/m^2 per substance and reservoir) closes to 1e-12; the dry state is a fixed
point of the rule.

What the rule gives, and what it shows about the maps:

- **Oil** has no volatile. Its cure reaction (linseed_oil + 0.12 oxygen -> 1.12 linoxyn, `guessed`) runs to completion:
  50 um wet becomes 49.7 um of linoxyn and pigment. Every dry-oil number is therefore `guessed`: the reaction and
  linoxyn are guesses, whatever the pigment.
- **Water with gum arabic** loses water to activity 0.5. The dry film keeps ~22 % water by volume (Flory-Huggins with
  the Hildebrand interaction). For gum arabic that is not absurd.
- **Negative result.** For particles in water (the acrylic latex, or pigment in water alone) the same mixing model
  keeps ~23 % water by volume at 50 % RH: `chemical_potential` gives a particle the entropy of a polymer chain of
  lattice sites, so water's activity is phi_w exp(1 - phi_w), as if dissolved in the particles. A real powder at 50 %
  RH holds only an adsorbed layer. The dry films of the acrylic and of search (b) are therefore wetter than they should
  be. Test: `test_particles_in_water_dry_to_a_powder_in_ordinary_air` (xfail). Owed to the substance owner.
- **No coalescence.** goo has no coalescence term: the latex stays particles, so the dry "acrylic" is a porous,
  scattering film (drying shifts of 9-15 dE00). This is what goo's declared world produces, not what an acrylic does.

## 3. The drawdown and the card (`goo.lab.drawdown`, `goo.lab.render`)

`drawdown(substance or Paint, binders, tints, illuminants, h_wet, env, bands)` makes, for each binder, the wet film
of each tint, dries it by §2, lays each over the black and the white stack, and looks (`goo.light.look` with
`goo.substance.maps`) under diffuse light; gloss is a second look under collimated light at 60 degrees over the white.
Readings are `goo.observe` (daylight and A; colour specular excluded). Bands: 380-740 nm in 20 nm, 1000 nm and 1500 nm
(100 nm wide).

Every number is `{"value", "unit", "evidence", "flags"[, "range"]}`. Evidence is the weakest of the map values the
look read (`maps.optical`, `surface`, `ground_optical`, `ground_surface` of every present film and the ground: their
`Derived` or `Bounded` evidence, which already carries the substances' provenance), the drying's (reactions and
mixing model) for a dry film, and the observer's. `flags` index the card's flag table: every flag a look or a drying
raised (out of domain with the map's reason, bounds collapsed, impossible), and one `impossible` flag per law an
impossible pigment fails, with its shortfall: the lab adds these itself, because a look flags only gain (K < 0) and
not, for example, an acausal eps. The card names, for each binder and state, what makes it as weak as it is
(`recipe.binders[].weakest`).

The image (PNG beside the JSON) draws the evidence: a tag on every swatch (C computed, K checked, A assumed,
G guessed; D out of domain, B bounds collapsed, X impossible), hatching and the word GUESS over every swatch of a
pigment with a guessed field, a red frame and the failed law for an impossible one, a crossed black box (never a
colour) where a look has no steady state, `!` where the colour is out of the sRGB gamut, and the flags' reasons in
the footer. Each swatch: top half over black, bottom half over white; the narrow strip is illuminant A.

## 4. The sheet (`goo.lab.sheet`)

Each number is measured by making films and looking:

| number | measurement |
|---|---|
| hiding thickness | the dry thickness where the masstone's contrast ratio reaches 0.98: a ladder of 36 wet thicknesses (1 um to 3 mm, log), each dried, over both grounds; log-linear interpolation; `null` with a note if never reached |
| tinting strength | (K/S)_tint - (K/S)_white at the band where the 1:10 tint's K/S is largest; K/S = (1 - R)^2 / 2R of an opaque (2 mm wet) dry film over white, specular excluded (Kubelka-Munk remission, `assumed`) |
| drying shift | CIEDE2000 between the wet and dry masstone over white, daylight |
| gloss | 60-degree gloss of the dry masstone |
| infrared | dry masstone reflectance over white per IR band, and white minus black (what an IR reflectogram sees through it) |

## 5. Admissibility, sensitivity, search

`admissible(s)`: the five laws, each with pass or fail, shortfall, unit, method (proved or measured) and the parts
measured; the failed laws; the fields that keep a possible substance from real.

`sensitivity(subject, card, sheet)`: the pigment's size distribution with its median diameter x0.5 and x2, and its
real index shifted by -0.05 and +0.05 at every wavelength (k kept; perturbation sizes `assumed`). Every card and sheet
number gets `range` = [min, max] over the nominal and the four variants; each swatch gets the largest CIEDE2000 any
variant moves it; the image prints the ranges.

`search`: a family (spheres, sigma_g 1.5, eps = eps_inf + one causal Gaussian band, or a lossless UV Lorentz for clear
particles; diameter; loading; white share), candidates `guessed`, made into films, dried and looked at exactly as a
drawdown; scipy's differential evolution, seeded (popsize 8; 10-12 generations; on 40 nm bands for (b) and (c), on the
lab's 20 nm bands for (a), because 40 nm bands misjudge a narrow absorption band by up to ~10 dE00; on 60 nm
visible-only bands for (b), whose L* objective needs no infrared), then Nelder-Mead on the lab's 20 nm bands; the
winner is re-measured through the maps.

**Speed.** `maps.optical` builds a Mie table on 81 host indices for every new substance (0.15-0.5 s), and
`optical_exact` builds it too. Candidates in a film without voids go instead through `search.FastOptical`, an exact
rearrangement of the same map: K = (h'/h) K' + phi_c K_c(n_host), with K' the maps' answer for the film without the
candidate and K_c by Mie at the one host index per band (tested equal to `optical_exact` to 1e-12 in K). A film with
voids goes to the maps whole. Per candidate on 20 nm bands: 0.58 s -> 0.09 s. Whole demonstrations, one core:
(a) ~10 min -> 66 s, (c) ~8 min -> 58 s, (b) ~10 min -> 318 s ((b)'s dry film has voids, a Bounded host, and must
use the maps' table). The real fix is in goo.substance: a way to evaluate `optical` for a substance at the host indices
a film actually has, without the 81-point table (or a lazily filled table). The impossible complement keeps the found n(lambda) and size and solves k band by band (a `Free`
dielectric) to hit a target spectrum: for a colour, the spectrum nearest the possible one with exactly the target
XYZ. `classify` names the laws it breaks. Results: `out/lab/search_{a,b,c}.json` and the substances' files.

## 6. The contribution format (`goo.lab.contribute`)

One JSON file per substance. Keys carry their SI unit (`rho_kg_m3`, `d_m`, `E0_J`, `lam_m`); a value outside the
plausible range of its unit is refused as a probable unit error. Fields: `format` ("goo.substance/1"), `name`,
`material`, `eps` (`oscillators` with `lorentz` {f, E0_J, gamma_J}, `drude` {Ep_J, gamma_J}, `gauss` {A, E0_J, s_J}
terms; or `tabulated` / `free` with `lam_m`, `parts` [{weight, n, k}], `lo_m`, `hi_m`, `citation`), `morph` (null, or
`sphere` / `coated` / `hollow` / `shape` with `d_m`, `sigma_g`, ...), `rho_kg_m3`, `thermo` (`M_kg_mol`, `p_sat_Pa`,
`dH_vap_J_mol`, `cohesion_J_m3`, `gamma_J_m2`, `gamma_d_J_m2`, `D_vapour_m2_s`), `rheo` (`eta_Pa_s`, `E_eta_J_mol`,
`separation_m`, `phi_percolation`, `coupling`), `reactions` [{stoich [[name, kg per kg]], `rate_1_s`, `E_a_J_mol`}],
`state` [{name, unit, lo, hi}], `reservoir_J_m3`, `prov` {field: {evidence, source, licence, note}}, and optionally
`class`, the class claimed.

The fields of `thermo`, `rheo`, `reactions` and `state` are **discovered** from the dataclasses (`Thermo`, `Rheo`,
`Reaction`, `StateVar`), not listed by hand: a field added to the description is written under `<name>_<SI unit>`
(or `<name>_SI` if its unit is not yet in `goo.lab.contribute.UNIT`, with a warning that its range is unchecked), and
read back. A top-level field of `Substance` the format does not know makes `to_json` refuse rather than drop it. Tests:
every field of Thermo and Rheo round-trips; a subclass with an extra field is carried or refused, never dropped.

`validate(file)` checks: the format tag and known keys; units by range; every given field has a provenance entry and
no entry lacks its field (keys `eps`, `morph`, `rho`, `reservoir`, `reactions`, `thermo.<p>`, `rheo.<p>`); every
evidence is a class of SPEC §2; **nothing is `computed` or `checked` without a source and a licence**; the description
is accepted by `Substance`; the five laws are evaluated, and a declared `class` must equal the computed one. All
errors are reported at once. Every library entry round-trips exactly (`to_json` then `validate` gives an equal
Substance; `tests/lab/test_lab_contribute.py`). Runtime: the standard library and numpy.

Worked example (a guessed violet; it validates as `possible`):

```json
{
 "format": "goo.substance/1",
 "name": "example_violet",
 "material": "",
 "eps": {"kind": "oscillators", "eps_inf": 2.4,
         "terms": [{"kind": "gauss", "A": 0.9, "E0_J": 3.4447e-19, "s_J": 2.243e-20}]},
 "morph": {"kind": "sphere", "d_m": 2e-07, "sigma_g": 1.5},
 "rho_kg_m3": 1500.0,
 "thermo": {},
 "rheo": {"separation_m": 1e-09, "phi_percolation": 0.15},
 "reactions": [],
 "reservoir_J_m3": 0.0,
 "prov": {
  "eps": {"evidence": "guessed", "source": "", "licence": "", "note": "one absorption band at 2.15 eV (577 nm), k ~0.3"},
  "morph": {"evidence": "assumed", "source": "", "licence": "", "note": "d50 0.2 um, sigma_g 1.5: a typical organic pigment"},
  "rho": {"evidence": "guessed", "source": "", "licence": "", "note": ""},
  "reservoir": {"evidence": "exact", "source": "", "licence": "", "note": "admissible: no reservoir"},
  "rheo.separation": {"evidence": "assumed", "source": "", "licence": "", "note": "1 nm adsorbed layer"},
  "rheo.phi_percolation": {"evidence": "assumed", "source": "", "licence": "", "note": "YODEL onset ~0.15"}
 },
 "class": "possible"
}
```

Energies are in J (2.15 eV = 3.4447e-19 J). Had `eps` said `"evidence": "computed"` with no source and licence,
`validate` would refuse it; had `d_m` been `0.2` (micrometres), it would be refused as a unit error; had `class` said
`"real"`, it would be refused with what blocks it (`material`, `eps guessed`, ...).

## 7. Command line

```text
python -m goo.lab card NAME [--fast] [--water]   NAME.png, NAME.json (card + sheet + admissible + sensitivity)
python -m goo.lab sheet NAME                     NAME.sheet.json
python -m goo.lab admissible NAME                NAME.admissible.json
python -m goo.lab validate FILE                  exit 1 if invalid
python -m goo.lab export NAME                    NAME.substance.json
python -m goo.lab search a|b|c [--quick]         search_X.json and the found substances' files
python -m goo.lab contact NAME ...               contact.png of cards already made
```

NAME is a library substance or a contribution file. Output goes to `out/lab/` (`--out` to change).

## 8. Known limits, and their causes

| limit | what it does to the cards | cause |
|---|---|---|
| hematite hides at ~1.2 um dry and is opaque at 1.5 um | its hiding and infrared numbers are not believable; its tints come out mauve-grey | Querry 1985's k in hematite's transparent range is too high (SUBSTANCES.md; classic's OPTICS-DATA check 6) |
| particle films keep ~23 % water at 50 % RH | the dry acrylic and the dry films of particles in water are wetter than real ones (`test_particles_in_water_dry_to_a_powder_in_ordinary_air`, xfail) | Flory-Huggins mixing in `maps.chemical_potential` gives a particle the entropy of a chain of lattice sites, so water's activity is phi_w exp(1 - phi_w) |
| latex does not coalesce | the dry "acrylic" is a porous scattering film; drying shifts of 6-15 dE00 | goo has no coalescence term |
| no gain saturation | gain_green has no steady state at drawdown loadings: crossed swatches and no numbers (value null, "no steady state") | SPEC has no saturation law; a look past threshold returns 0 and books the power to `unresolved` |
| the observer is unverified | every colour is `assumed` at best | the CIE 1931 analytic fit was not compared with the CIE table (not readable here) |
| the daylight is not D65 | colours under "daylight" are under a Planckian radiator at 6504 K | D65 is a CIE table, not shipped; the difference was not measured |
| every paint is out of domain | no card number is better than `assumed` | particle volume fraction above 0.1 (independent scattering) in every paint film |
| dry oil is always guessed | whatever the pigment | linseed oil's cure reaction and linoxyn are `guessed` |
