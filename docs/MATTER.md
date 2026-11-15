# goo.matter: substances as descriptions

`goo/matter/` is the kernel of the matter constitution: a substance is a point in a space of constitutive
descriptions (a dielectric function, a particle morphology, mechanics, changes); everything the engine uses is
derived from it by stated maps; every substance is labelled `real`, `possible` or `impossible` by named laws.
It changes nothing in the engine. It talks to the engine only through `PIGMENTS` (SPEC.md section 9), and
section 6 below lists, line by line, what the engine would have to read for the rest to take effect.

Run `PYTHONPATH=. python studies/matter.py` (seconds; `--redo` re-runs the searches, about a minute; `--fit`
re-fits the classic pigments, tens of minutes). It writes `out/matter-study.json` and `out/matter.png`.
Tests: `tests/test_matter.py`.

| module | holds |
|---|---|
| `dielectric` | `Oscillators` (Lorentz, Drude, causal Gaussian terms), `Tabulated` (measured n, k, cited), `Free` (anything); water, linseed, air |
| `morphology` | `LogNormal` sizes; `Sphere`, `Coated`, `Hollow`; `Shape` (held, not derived: raises `NotDerivable`) |
| `mie` | sphere (BHMIE) and coated sphere (BHCOAT), numpy, vectorised |
| `response` | Mie to Kubelka-Munk, goo's units and bands, `ResponseTable` over the surrounding index |
| `mechanics` | packing, intrinsic viscosity, Stokes-Einstein, Stokes, evaporation; goo's tick read in seconds |
| `substance` | `Substance`, `Liquid`, `Provenance`, `Change` |
| `laws` | the admissibility predicates, `classify`, gain and its debt, `ENERGY_LEDGER` |
| `tables` | `goo/data/optics.json` (when present) as `real` points |
| `classic` | the classic pigments as fitted points, with residuals |
| `phenotype` | data sheets made in the engine |
| `search` | the three inverse-design searches |
| `example` | one impossible substance carried through |
| `gaps` | what the engine cannot consume yet (section 6) |

## 1. Descriptions and maps

**Dielectric function** (convention exp(-i omega t): passive means Im eps >= 0, n + ik = sqrt(eps), k >= 0).

- `Oscillators`: eps(E) = eps_inf + sum of terms. Lorentz `f E0^2 / (E0^2 - E^2 - i gamma E)` and Drude
  `-Ep^2 / (E^2 + i gamma E)` (Bohren & Huffman 1983, ch. 9), and the causal Gaussian band
  `eps2 = A [exp(-((E-E0)/s)^2) - exp(-((E+E0)/s)^2)]`, `eps1 = (2A/sqrt(pi)) [F((E+E0)/s) - F((E-E0)/s)]`, F the
  Dawson function (De Sousa Meneses, Malki & Echegut 2006; Brendel & Bormann 1992). Every term is the response
  of damped oscillators, analytic in the upper half plane: causal by construction, passive when every strength
  and width is >= 0 and eps_inf >= 1. A negative strength is gain, and still causal (the laws are independent).
- `Tabulated(lam, n, k, citation)`: measured constants, linear interpolation, ends held 5% beyond the range
  (so a table ending at 1550 nm covers goo's N box), nothing further.
- `Free(lam, n, k)`: no claim; this is how impossible responses are written.
- Liquids: water by the Daimon & Masumura (2007) Sellmeier (written as lossless Lorentz terms), linseed as one
  UV oscillator at n_D 1.479 (dispersion plausible), air.

**Morphology.** Log-normal diameters by volume (volume-median d, sigma_g), integrated by Gauss-Hermite in ln d.
`Sphere`; `Coated` (core of the substance, shell of another dielectric, radius ratio); `Hollow` (the substance
as a shell around a void: opaque-polymer pigments); `Shape(name, params)` for anything else, held as data and
refused by every map (`NotDerivable`): goo says it does not model it rather than guess.

**Optics map** (`response.py`), each step with its source:

1. Mie efficiencies Q_abs, Q_sca, g in a non-absorbing medium n_s: m = (n + ik) / n_s, x = pi d n_s / lambda.
2. Per unit particle volume: cross-section per volume 3Q/(2d), averaged over the volume distribution:
   mu_a, mu_s (1/um per unit volume fraction). Independent scattering: no crowding (at paint loadings titania
   scatters up to ~2x less: Ross 1971), no aggregation. Linear in volume, which is goo's optics' assumption too.
3. Two-flux: **K = 2 mu_a, S = (3/4) mu_s (1 - g)** (van Gemert & Star 1987; Star, Marijnissen & van Gemert
   1988, the diffusion limit for diffuse light). Their `- mu_a/4` term in S is dropped: it can make S negative
   for strong absorbers, where the diffusion limit fails anyway. Valid for scattering-dominated, optically thick
   films; for strong absorbers K is right, S indicative. The same relation as the `optics.json` table.
4. goo's units: K, S per 0.025 mm of pure pigment (SPEC.md section 2): `K_goo = 25 um x K`.
5. Bands: means over goo's boxes R 600-700, G 500-600, B 400-500 nm and N 1500-1600 nm, sampled every 10 nm.
6. `ResponseTable`: K and S precomputed on n_s = 1.00, 1.05, ..., 1.60 and spline-interpolated per band
   (within 2% of exact Mie, tested); cached in memory and in `out/matter-cache/` by a digest of the description.
   `sub.ks(n_s)` takes a scalar or one index per band (a dispersive medium).

**Mechanics map** (`mechanics.py`).

| quantity | map | source |
|---|---|---|
| max packing phi_m (= CPVC) | 0.634 + 0.0658 delta + 0.0398 S delta^2 (delta, S: polydispersity, skewness of diameters) | Desmond & Weeks 2014 |
| intrinsic viscosity | 5/2 for rigid spheres of any interior | Einstein 1906/1911 |
| thickening | eta = eta_l (1 - phi/phi_m)^-([eta] phi_m) | Krieger & Dougherty 1959 |
| diffusivity | kT / (3 pi eta d) | Stokes-Einstein |
| settling | (rho_p - rho_f) g d^2 / (18 eta), hindered (1 - phi)^4.65 | Stokes; Richardson & Zaki 1954 |
| evaporation | J = D_v M p_sat (1 - RH) / (R T delta), recession J / rho | Fick, ideal vapour |

goo has no clock. Reading its water evaporation constant (0.15 um per tick) against water at 25 C, RH 50%,
a 5 mm still-air layer (0.058 um/s) gives **one tick ~ 2.6 s**; every per-tick number below uses that reading.

One microscopic fact moves everything: the size of a classic fit's particles sets its colour (Mie), its
hiding, its phi_m (through sigma_g), its Brownian D and its settling. See section 3 for what that implies.

## 2. Admissibility (`laws.py`)

Each law is a function of a description returning `Verdict(law, ok, amount, sentence)`.

| law | test | amount |
|---|---|---|
| passivity | Im eps >= 0 over the described range (and the shell's) | -min Im eps |
| causality | numerical Kramers-Kronig test (below) | rms remainder / max abs(eps - 1), fail above 2% |
| boundedness | eps_inf >= 1; f-sum weight (2/pi) int E eps2 dE <= (30 eV)^2; d in 2 nm-500 um; sigma_g in [1, 4]; ratios in [0, 1); density > 0 | worst excess |
| thermodynamic | viscosity, p_sat, surface tension, change rates >= 0; an irreversible change has no reverse rate | worst |
| conservation | each `Change`: product + sinks - sources = 1 per unit volume consumed | worst imbalance |
| reciprocity | isotropic scalar eps: by construction; a claim of non-reciprocity needs a declared bias field | 1 if unbiased claim |

**Causality on a finite range.** eps2 inside [E1, E2] gives, by the principal-value integral (Maclaurin's
alternate-point rule, Ohta & Ishida 1988), the in-range part of eps1 - 1. Any passive absorption outside the
range adds, inside it, a non-negative mixture of c/(W^2 - E^2) (W > E2: rising), -c/(E^2 - w^2) (w < E1:
falling) and a constant. These poles span *every* passive extension, so the test fits the residual by
non-negative least squares on a dense set of them and reports what is left. A large remainder proves (to the
discretisation) that no passive material has these n and k on this range. A small one does not prove that one
exists: it says the table is consistent with some passive extension. Noise, features narrower than the grid,
interpolated gaps and absorption tails at the edges limit any finite table. Random oscillator descriptions
pass with remainders below 1% (tested); absorption with no dispersion fails by an order of magnitude.

**Class.** `impossible` if any law fails, listing them; else `real` if its provenance ties it to a material
(`measured` or `computed`, with a citation), else `possible`. A fit to a table is not a material
(`Provenance("fit")` stays `possible`); a measurement does not excuse a violation.

**Gain and its debt.** `laws.km_any` continues the two-flux layer formulas to K < 0 (for -2S < K < 0 the
hyperbolic functions become circular) and detects the lasing threshold, where the denominator reaches zero
(for a weak absorber-free gain, |K| S ~ pi^2/2 in thickness-scaled units). `gain_debt` gives the spectral power
created per unit particle volume and unit irradiance (-mu_a, 1/um) and, per band, R + T - 1 of a film over
black: light out per unit light in beyond what came in. `ENERGY_LEDGER` states what the engine would have to
record: incident, reflected, absorbed (negative with gain), drawn per species against a declared finite
reservoir, and the rule when it is exhausted (K -> max(K, 0)), with an optics invariant beside the volume I1.

**Measured tables.** `tables.load()` holds `goo/data/optics.json` (when present; nothing depends on it) as
substances with `Tabulated` dielectrics and the table's own K, S kept beside them. A species whose n or k is
"guessed" at any wavelength (the table's per-wavelength provenance) is not `real`. Checked against the table
being built in parallel (schema goo.optics-data/1): my Mie from its n, k and sizes reproduces its K and S for
carbon, yellow (CdS) and ivory black to 4e-4 to 5e-3 (two independent Mie implementations); white and sienna
differ by 25% and 12% because that table averages uniaxial cross-sections (2/3 o + 1/3 e) while its n is the
ordinary ray. The causality test passes white, yellow, sienna and ivory black, and **fails vermilion (2.0%)**,
whose n below 620 nm is goo's guess (the table says so); carbon fails marginally (2.9%) over 380-1550 nm, which
rests on an extrapolated 380-430 nm and a linear 780-1550 nm gap, and passes (1.6%) on 380-780 nm alone.

## 3. The classic species as points (`classic.py`)

Family: eps_inf, two causal Gaussian bands, one Lorentz, all of non-negative strength; log-normal spheres.
Fitted in oil (n_s = 1.48) by least squares on log10(K + 0.01) and log10(S + 0.01) of the eight table numbers,
from starts chosen on a precomputed (n, d, sigma_g) grid of clear spheres. The descriptions are `possible`
(provenance: the table). Residual: rms and worst factor between derived and tabled; masstone and tint: CIEDE76
between the table's and the fit's data sheets, both made in the engine.

| pigment | rms factor | worst (entry) | n(550) | d (um) | masstone dE | tint dE |
|---|---|---|---|---|---|---|
| ivory_black | 1.000 | 1.000 (exact) | 1.41 | 16.7 | 0.0 | 0.0 |
| carbon | 1.025 | 1.04 (K_N) | 2.22 | 21.8 | 0.9 | 0.2 |
| green | 1.10 | 1.18 (S_N) | 1.57 | 19.0 | 3.5 | 0.4 |
| dioxazine | 1.10 | 1.23 (S_B) | 1.54 | 13.3 | 4.7 | 0.3 |
| white | 1.11 | 1.25 (S_B) | 1.64 | 0.26 | 1.8 | 0.2 |
| rose | 1.11 | 1.28 (S_B) | 1.54 | 17.7 | 5.1 | 0.2 |
| ultramarine | 1.14 | 1.33 (S_G) | 1.52 | 0.10 | 9.6 | 0.2 |
| sienna | 1.24 | 1.53 (S_R) | 1.61 | 13.7 | 9.2 | 0.5 |
| red | 1.30 | 1.92 (S_R) | 1.65 | 8.9 | 9.0 | 0.4 |
| yellow | 1.33 | 1.52 (S_N) | 1.65 | 6.4 | 8.1 | 0.2 |
| ochre | 1.38 | 2.04 (S_N) | 1.61 | 8.7 | 9.1 | 0.1 |

Best ivory black (exact), worst ochre. Tints in white agree within 0.5; dark masstones amplify small K/S errors.
Fits are not unique (12 parameters for 8 numbers): each is *a* point that reproduces the row, not *the*.

**Findings.**

- **goo's table is 50-150x weaker per unit pigment volume than fine real pigments.** Mie from measured constants
  at real sizes (the parallel `optics.json`): rutile S ~150-240 vs goo's 3.2; soot K ~450-560 vs 3.2; CdS K_B
  273 vs 2.8; hematite K_G 380 vs 1.25. The family reaches goo's magnitudes only with **coarse (6-22 um),
  low-contrast (n 1.5-1.65 in oil 1.48) particles**: nine of eleven fits sit there. As carbon black (20-100 nm,
  n ~2, k ~0.7) goo's carbon is unreachable; flat K across 400-1550 nm is the geometric saturation of grains
  ~20 um across, not a property of the material (soot's K falls as 1/lambda). Ivory black is reached exactly,
  but with n 1.41, below oil, which hydroxyapatite (1.63) is not. goo's white is reachable only as a low-index
  (n 1.64, 0.26 um) extender, i.e. chalk or barite, not titanium (2.7) or lead white (2.0).
- **White's S is inconsistent with itself.** S falls 7x from visible to 1.5 um (needs small particles, which
  scatter blue *more*) yet S_B < S_G: no sphere does both; best reach is 25% high at B.
- **Vermilion's S jump** (0.9 in R, 0.3 in G and B) is reached only to a factor 1.9; ochre's S_N, yellow's S_N
  to ~1.5-2. Plausible, not impossible: a sharper absorption edge or non-spherical grains might do it.
- **The coupling bites.** With the fitted (coarse) sizes, every classic pigment except white and ultramarine
  would settle out of a 25 um wash in under a tick (Stokes, in water: 0.05-0.3 mm/s); with fine sizes it would
  not, but its K and S would be ~100x the table's. goo's table and goo's watercolour cannot both be physical at
  the present scale. The honest exact limit is a composite: pigment diluted ~1:100 in a clear extender particle
  (a description goo does not yet have), or a rescaling of goo's `gain`.
- Brownian D of these particles is 1e-7 to 1e-4 cells^2/tick; goo's `D0 = 0.15` is 1e3-1e6x larger. goo's
  colour diffusion is a stand-in for shear mixing, not Stokes-Einstein (it keeps only the 1/eta scaling).
- goo's Krieger-Dougherty exponent is 2 for every pigment; spheres give [eta] phi_m = 1.6-1.8.

## 4. Phenotype and inverse design (`phenotype.py`, `search.py`)

A data sheet registers the substance's K, S (at a chosen n_s) in `PIGMENTS` under a temporary name, lays films
with `Canvas.add` on a 13 x 4 canvas (thicknesses 1 um-1 mm over gesso and over a fixed carbon stratum, a 1 mm
masstone, a 1:10 tint in white), reads `reflectance()` and removes the name. Under a second. It reports
masstone, undertone (10 um over gesso), tint and tinting strength (lightness lost), hiding (film at which
over-black and over-white differ by dE < 1), and 1.5 um reflectance and contrast. Searches use the engine's own
layer formula (`Canvas.km_layer`) for speed and are confirmed by a data sheet.

**(a) A light cyan.** Target L 75, C 22, h 210 (oil paint, 190 um over gesso). Best stock mix (goo.palette,
up to three pigments): white/ultramarine/green, dE 16.2. Nearest possible: n(550) 2.80, a red absorption band
at 1.61 eV (770 nm, s 0.34 eV), d 0.29 um; 3% of it in 97% white: **dE 1.2**. A strong, high-index tinter
(the role of phthalocyanines). Exact hit: the band K's change by (+4.5, -3.2, -5.6) of (246, 50, 38), all
positive: **no law need be broken**; goo's hole is in its palette, not in physics.

**(b) Dark in water, white in air.** Read as a 50 um film over black (the wet-dark mechanism: index matching).
Nearest possible: clear spheres with n_D 1.334 and water's dispersion to the third decimal, d 0.48 um, sigma_g
1.14: **L 0.1 in water, 98.3 in air** (S in water ~1e-4 against ~20 in air; cryolite, n 1.338, is the real
neighbour). The impossible complement: L = 100 in air over black needs R = 1 with T lost to the black, which
only gain gives: K = (-0.063, -0.053, -0.045) per 0.025 mm; it breaks passivity. L = 0 in water breaks
nothing. Over a white ground the target would need absorption that switches with the surroundings; no search
was run for that.

**(c) Colourless in the visible, absorbing at 1.5 um.** 25 um glaze in oil, judged against clear oil over
gesso and over black (the worse of the two). Nearest possible: a narrow band at 0.774 eV (1602 nm, s 0.04 eV),
eps_inf 2.21 so n matches oil, d 0.018 um: **dE 0.01, K_N 116** (goo's carbon: 3.0). Possible (as NIR
absorbers like doped-oxide nanoparticles are), because tiny particles scatter nothing and a narrow band's
Kramers-Kronig wiggle in n is confined to the infrared. The impossible complement: n = 1.48 flat, k a band at
1550 nm, dE 0.00, K_N 117: **fails causality (remainder 9%)**, passes everything else.

## 5. An impossible substance carried through (`example.py`)

"Green from nowhere": eps_inf 5, a passive UV band, and a causal Gaussian band of strength -0.02 at 2.3 eV
(540 nm); spheres d 0.3 um. Class **impossible**; fails **passivity** (Im eps = -0.020 at 539 nm), passes
causality (0.24%), boundedness, thermodynamic, conservation, reciprocity. In oil: K = (-0.014, -4.93, 0.35, 0),
S = (76, 90, 110, 15). Debt: **0.20 per um** of spectral power created per unit particle volume and irradiance at
540 nm; for films at phi 0.4 over black, R + T - 1 in the green is 0.050 (0.6 um film), 0.106 (1.2 um), 0.352
(3.1 um), and the film **lases above 6.0 um** (no steady two-flux solution). Over gesso the true green
reflectance is 1.02, 1.13, 2.65, then unbounded; the engine as it stands paints 0.97, 0.99, 1.00, 1.00: it
mis-computes negative K (`b = sqrt(max(a^2 - 1, 1e-14))`) and clips to 1, so the impossible paint looks like a
bright white with a faint green cast (masstone Lab 99.6, -2.4, 4.4), silently, with no debt. Section 6 row 14.

## 6. The gap list (`gaps.py`)

What goo.matter derives and goo's engine cannot consume. The line numbers refer to `goo/engine.py` as it was
when the kernel was written (551 lines); the engine has grown since, so read them as pointers, not addresses.

| # | parameter | engine now | line | the law would read |
|---|---|---|---|---|
| 1 | K(n_s), S(n_s) per pigment | fixed numbers, no surrounding | 29-41, 510-516 | each species' response at the cell's n_s per band |
| 2 | phi_m per particle species | `L["phi_m"] = 0.64` global | 177; 242, 244-245 | phi_m = sum x_s phi_m,s (first order) |
| 3 | [eta]_s, KD exponent | `** -2` for all | 242 | -[eta]_mix phi_m, [eta]_mix = sum x_s [eta]_s |
| 4 | liquid viscosity eta_l | `L["eta0"] = 0.12` for water and oil | 177; 242, 321, 431 | exp(sum y_l ln eta_l) |
| 5 | roles (solid / liquid) | by position: `CARRIERS`, `nc = 2`, `q[nc:]` | 26, 166-167, 239 | by description: particles vs continuous phases |
| 6 | D_s per pigment | `L["D0"] = 0.15`, one Dc for all | 177; 321, 431-437 | D_s ~ 1/(eta d_s); shear mixing as its own law |
| 7 | settling v_s | none (depth-averaged layer) | (SPEC 8) | a settled fraction per species, fed at v_s / h |
| 8 | density rho_s | none | - | needed by 7 |
| 9 | volatility per liquid | only `q[0]` evaporates at `evap` | 177; 444-450 | e_l from p_sat; per-species ledger entry |
| 10 | soak-in per liquid | only `q[0]`; `_store` is water | 178; 451-455; 170 | rate ~ gamma_l / eta_l; a store per liquid |
| 11 | cure (a Change) | only `q[1]` cures; dryness tests `q[0]`, `q[1]` | 179; 457-459; 475 | each species' declared Changes; dry = volatiles gone, binders past threshold |
| 12 | binder absorption | `vol[1] * [0.02, 0.03, 0.06, 0.01] * (1 + 2 kappa)` | 516 | each liquid's K from its own k(lambda) and state |
| 13 | liquid index | none in the engine | - | n_l per band from the description |
| 14 | gain and debt | negative K mis-computed, clipped | 522, 542 | `laws.km_any`, threshold test, energy ledger |
| 15 | yield onset, tau0 | `phi_y`, `tau0` global | 177 | not derived: needs an attraction parameter (Hamaker constant) in the description |

## 7. What two-flux optics cannot represent at all

- **Angle-dependent and interference colour** (thin films, opals, structural colour, goniochromism): K and S
  are angle-averaged diffuse quantities; a phase-coherent layer has no K or S. Needs a transfer-matrix model of
  coherent layers inside the stack and an angle-resolved light and view (a BRDF), i.e. more than two fluxes.
- **Flakes** (mica, metal, pearlescent): oriented platelets reflect specularly and by interference; their
  scattering is anisotropic and depends on orientation, which settles with flow. Needs `Shape("platelet")`
  optics (T-matrix or geometric-plus-thin-film), an orientation state per cell driven by the flow law, and a
  multi-flux or angle-resolved radiative transfer. Held as data today (`NotDerivable`).
- **Fluorescence**: absorbs at one wavelength and emits at another: K is not a loss but a transfer between
  bands. Needs a band-to-band redistribution matrix in the layer equations (two-flux per band coupled by a
  quantum-yield source term), and the energy ledger to show it creates no energy (Stokes shift).
- Also outside: dependent scattering at high PVC, aggregation, birefringence (cross-sections must be averaged,
  as the parallel table does), absorbing surroundings, and the surface (Fresnel, roughness), which belongs to
  the optics model, not to the substance.
