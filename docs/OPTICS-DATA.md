# goo's optical data

`goo/data/optics.json` gives, for every species goo uses: a material, a role, a refractive index spectrum, and
Kubelka-Munk absorption K and scattering S spectra in each surrounding (oil 1.48, water 1.33, air 1.00), plus S
and K against the surrounding's index. Every number at every wavelength is marked `computed`, `assumed` or
`guessed`. `goo.data` loads it (numpy only). `studies/optics_data.py` regenerates it from the source data and prints
the diff against the shipped table. Neither the engine nor an optics model reads it yet; `goo.matter.tables` reads
it as data. It is meant as input for the optics `v2` model (`SPEC.md` §6, §13).

## Policy

What ships is computed by us from public-domain (CC0) inputs, and the original measurement paper of every input is
cited. Anything with an ambiguous or restrictive licence (RIT, Golden, IFAC-CNR, Handprint, manufacturers, LBNL)
is only read and compared against, a few cited numbers at a time. It is never downloaded, copied or scraped. We
give attribution even where none is required, and every number carries its provenance.

## Sources and licences

| What | Licence | How used |
|---|---|---|
| refractiveindex.info database, M. N. Polyanskiy, *Sci. Data* **11**, 94 (2024), doi:10.1038/s41597-023-02898-2; github.com/polyanskiy/refractiveindex.info-database (commit recorded in the JSON as `database_commit`) | **CC0 1.0 Universal**. Its LICENSE file is the CC0 1.0 legal code, and its README says "The database is in the public domain ... CC0 1.0 Universal Public Domain Dedication" | read by the generator from a local clone (`REFRACTIVEINDEX_DB`). Not copied into the repo |
| miepython (S. Prahl), v3.3.1 | MIT | Mie efficiencies, generator only |
| PyYAML | MIT | reads the database, generator only |
| colour-science v0.4.7 | BSD-3-Clause | CIE 1931 observer, D65 and sRGB for the masstone check (`--report` only). Not shipped |

### Measurement papers behind the numbers used (each file's own reference, as the JSON's `sources` carries it)

- Rutile: J. R. Devore, Refractive indices of rutile and sphalerite, *J. Opt. Soc. Am.* **41**, 416 (1951). Files `main/TiO2/nk/Devore-o.yml` and `Devore-e.yml`, 0.43-1.53 µm.
- Soot (carbon, and the carbon in ivory black): W. H. Dalzell and A. F. Sarofim, Optical constants of soot and their application to heat-flux calculations, *J. Heat Transfer* **91**, 100 (1969). File `other/soots/propane soot/nk/Dalzell.yml`, 0.436-10 µm.
- Hematite: M. R. Querry, Optical constants, Contractor Report CRDC-CR-85034 (1985). Files `main/Fe2O3/nk/Querry-o.yml` and `Querry-e.yml`.
- Cadmium sulfide: R. E. Treharne et al., Optical design and fabrication of fully sputtered CdTe/CdS solar cells, *J. Phys. Conf. Ser.* **286**, 012038 (2011). File `main/CdS/nk/Treharne.yml`, 0.30-1.50 µm.
- Cinnabar: W. L. Bond, G. D. Boyd and H. L. Carter Jr., Refractive indices of HgS (cinnabar) between 0.62 and 11 µm, *J. Appl. Phys.* **38**, 4090 (1967), with the fit by G. C. Bhar. Files `main/HgS/nk/Bond-o.yml` and `Bond-e.yml`.
- Calcite: G. Ghosh, Dispersion-equation coefficients for the refractive index and birefringence of calcite and quartz crystals, *Opt. Commun.* **163**, 95 (1999). Files `main/CaCO3/nk/Ghosh-o.yml` and `Ghosh-e.yml`.
- Cellulose: C. Juntunen et al., Refractive index measurement of pharmaceutical powders in the short-wave infrared range ..., *Powder Technol.* **438**, 119621 (2024). File `organic/(C6H10O5)n - cellulose/nk/Juntunen.yml`, 1.10-1.65 µm.
- PMMA (the acrylic): X. Zhang et al., Complex refractive indices measurements of polymers in visible and near-infrared bands, *Appl. Opt.* **59**, 2337 (2020). File `organic/(C5H8O2)n - poly(methyl methacrylate)/nk/Zhang-Tomson.yml`.
- Water: G. M. Hale and M. R. Querry, Optical constants of water in the 200-nm to 200-µm wavelength region, *Appl. Opt.* **12**, 555 (1973). File `main/H2O/nk/Hale.yml`.
- Oleic acid (the dispersion shape assumed for linseed oil): S. H. Jones, M. D. King, A. D. Ward, *Chem. Commun.* **51**, 4914 (2015). File `organic/C18H34O2 - oleic acid/nk/Jones.yml`.
- Mie to Kubelka-Munk: M. J. C. van Gemert and W. M. Star, Relations between the Kubelka-Munk and the transport equation models for anisotropic scattering, *Lasers Life Sci.* **1**, 287 (1987).

## Method

1. **Constants.** n and k come from the files above, on 380-780 nm every 10 nm plus 1550 nm (42 samples).
   Outside a file's range, a formula is evaluated beyond it and a table is held at its end value. Those
   wavelengths are listed in `extrapolated_nm`: rutile 380-420 and 1550, soot 380-430, CdS 1550, PMMA 380-390,
   and cellulose everywhere in the visible (Juntunen's fit runs 1.10-1.65 µm). Where a source measured only a
   transparent range (rutile, cinnabar, calcite, cellulose), k = 0 is *assumed*, and the `k` provenance says so.
2. **Particles.** Particles are spheres with a log-normal size distribution by volume: median d50 and geometric
   standard deviation σg, 24 Gauss-Hermite nodes. All sizes are *assumed*: rutile 0.25 µm/1.35, carbon
   0.10/1.5, ivory black 2.0/1.8, hematite 0.30/1.6, CdS 0.40/1.5, HgS 1.0/1.6, calcite 3.0/1.8, cellulose
   20/1.5, latex 0.15/1.25. Uniaxial crystals average their cross-sections as 2/3 ordinary + 1/3 extraordinary.
   Ivory black (*assumed*) is a Maxwell Garnett mixture of 15 vol% soot in hydroxyapatite of index 1.63.
3. **Mie.** miepython gives Q_sca, Q_abs and g for relative index m = (n + ik)/n_medium in a non-absorbing
   medium, for each of 11 medium indices (1.00 to 1.60, `sweep_n_medium`). Per unit pigment volume, for a
   sphere of diameter d, μ_a = (3/2)⟨Q_abs/d⟩ and μ_s′ = (3/2)⟨Q_sca(1 − g)/d⟩.
4. **Kubelka-Munk.** K = 2 μ_a and S = (3/4) μ_s′. This is van Gemert and Star's diffusion-limit relation without
   its −μ_a/4 term, which would make S negative for blacks. **Limits:** the relation assumes μ_s′ ≫ μ_a (it
   fails for carbon and other strong absorbers). It assumes independent scattering, so there is no dependent
   scattering and no crowding at high pigment volume (TiO2's S per volume falls well below the dilute value
   above roughly 10-20% PVC). There is also no aggregation and no surface treatment. K and S are per unit
   pigment volume at infinite dilution.
5. **Continuous media.** Water, linseed oil and the acrylic's "bulk" film have K = 2α with α = 4πk/λ, and
   S = 0.
6. **Guessed values.** Where there are no CC0 constants, K and S in oil are goo's classic values, spread flat
   over the bands (B 380-490 nm, G 500-590 nm, R 600-780 nm, N at 1550) and divided by 0.025 mm. S in the
   other media is that value times the Mie ratio S(medium)/S(oil), computed with a *guessed* index (k = 0) and an
   assumed size. K is kept the same in every medium. Everything in this step is marked `guessed`.

**Units.** goo's `PIGMENTS` are "K and S per 0.025 mm of pure pigment" (SPEC.md §2). A volume of 1 is 0.25 mm of
pure pigment, and `reflectance(gain=10)` multiplies it by 10, so K_goo·(vol·10) = K_phys·(vol·0.25 mm). That gives
**K_goo = K_phys [mm⁻¹] × 0.025 mm**, and the same for S. The JSON is in mm⁻¹ for the pure pigment. Use
`data.GOO_UNIT_MM` or `Table.classic_table()` to convert.

## Provenance

| species | role | material | n | K | S | summary |
|---|---|---|---|---|---|---|
| white | pigment | rutile TiO2 (PW6) | computed | computed (k = 0 assumed) | computed | **computed** |
| carbon | pigment | soot / lamp black (PBk6/7) | computed | computed | computed | **computed** |
| sienna | pigment | hematite α-Fe2O3 (burnt sienna's chromophore) | computed | computed | computed | **computed** |
| yellow | pigment | CdS (PY37) | computed | computed | computed | **computed** |
| red | pigment | HgS vermilion (PR106) | computed ≥620 nm, guessed below | guessed <620, 0 above (k = 0 assumed) | computed ≥620 and 1550, guessed below | **mixed** |
| ivory_black | pigment | soot in hydroxyapatite (PBk9) | assumed | assumed | assumed | **assumed** |
| ochre | pigment | goethite (PY43) | guessed 2.35 | guessed | guessed | **guessed** |
| ultramarine | pigment | lazurite (PB29) | guessed 1.50 | guessed | guessed | **guessed** |
| rose | pigment | quinacridone / madder (PV19/PR83) | guessed 1.70 | guessed | guessed | **guessed** |
| green | pigment | viridian (PG18) | guessed 1.80 | guessed | guessed | **guessed** |
| dioxazine | pigment | PV23 | guessed 1.70 | guessed | guessed | **guessed** |
| gesso | ground | calcite (chalk) | computed | computed (0) | computed | **computed** |
| paper | ground | cellulose | computed (extrapolated) | computed (0) | computed | **computed** |
| acrylic | binder | PMMA stand-in | computed | computed | computed (as latex particles) | **computed** |
| water | carrier | water | computed | computed | 0 | **computed** |
| oil | carrier | linseed | assumed 1.48 | guessed (goo's yellowing) | 0 | **mixed** |

## Check 1: our K and S in oil against goo's current table

These are ours/goo in goo units. R, G and B are band means over 600-700, 500-600 and 400-500 nm; N is the
1550 nm value.

| species | K R | K G | K B | K N | S R | S G | S B | S N |
|---|---|---|---|---|---|---|---|---|
| white | 0 | 0 | 0 | 0 | 46 | 60 | 79 | 37 |
| carbon | 140 | 160 | 170 | 99 | 85 | 120 | 160 | 15 |
| sienna | 490 | 300 | 170 | 150 | 420 | 360 | 470 | 220 |
| yellow | ~0 | 300 | 96 | ~0 | 88 | 110 | 86 | 63 |
| red (mixed) | 0.15 | 0.95 | 1.0 | 0 | 39 | 1.1 | 1.0 | 160 |
| ivory_black (assumed) | 8.0 | 8.2 | 8.1 | 6.9 | 30 | 29 | 27 | 36 |

**Finding.** Wherever physics is available, goo's guesses are about **150 times too weak** (the geometric mean of
20 non-zero K and S ratios is 155, with a range of 46 to 490). The shape agrees roughly, but the magnitude does
not. At physical strength, 25 µm of paint at 20% PVC is fully hiding for rutile, carbon, hematite and CdS. At
goo's strength, the same film is a glaze. **We believe the computed values.** They rest on measured constants, and
for rutile they reproduce the coatings literature within 20% (check 2). The dilute-limit assumption overstates
S at high PVC, plausibly by a factor of 2 to 3 (crowding), not 100. Two consequences:

- goo's paintings are tuned to the weak table, so a `v2` model that adopts physical strengths must rescale volumes
  (or treat goo's "pigment" volume as pigment plus extender). That is a decision for the owner.
- The shipped `guessed` species keep goo's scale. In a mix with `computed` species they are about 150× too weak.
  A `v2` model should either scale guessed species by a common factor (155 is the measured mean, itself a guess)
  or keep both sets on goo's scale.

Other large disagreements:

- White K = 0. Ours has no UV-edge tail and no impurities. goo's 0.02 is a defensible floor.
- CdS K is zero above 520 nm and at 1550, as it must be for a 2.4 eV gap.
- Red's 0.15 K ratio in the R band is k = 0 assumed above 620 nm. Vermilion's absorption edge is near 600 nm.
- Hematite K at 1550 is 150× goo's. This is suspect: see check 6.

## Check 2: rutile against the coatings literature

- **Ours:** for monodisperse rutile in n = 1.5 at 550 nm (n_o 2.648, n_e 2.953), μ_s(1−g) per unit volume peaks
  at d = **0.20 µm**, at **12.5 µm⁻¹**. At 0.22 µm it is 11.6 µm⁻¹. The shipped log-normal (d50 0.25, oil 1.48)
  gives S = 7.57 µm⁻¹, which is μ_s′ = 10.1 µm⁻¹.
- **Reference:** the LBNL pigment database commentary on whites (coolcolors.lbl.gov, `LBNL-commentary/white.pdf`,
  read only) gives an optimum of "about 0.2 µm". For 0.22 µm in a binder of 1.5 at 550 nm it gives an
  angle-weighted Mie s of "about 12 µm⁻¹", and 10.4 µm⁻¹ from another author with other constants.
- **Agree** (within 5% monodisperse, and 15% for the shipped distribution). The test bounds it at a factor of 1.5.
- Size matters. At 550 nm, S_oil is 4.85, 7.57 and 3.68 µm⁻¹ for d50 × 0.5, × 1 and × 2. At 1550 nm it is 0.11,
  0.66 and 1.79 µm⁻¹. The 1.5 µm band depends strongly on the assumed size.

## Check 3: wet and dry

S(water)/S(oil) and S(air)/S(oil) are band means R G B, then the 1550 nm value. The last column is
S(air)/S(water) in G, which is the predicted drying shift of a watercolour wash's scattering.

| species | water/oil | air/oil | air/water (G) |
|---|---|---|---|
| white | 1.30 1.26 1.16 1.06 | 1.94 1.79 1.37 0.80 | 1.42 |
| carbon | 1.10 1.13 1.20 0.92 | 1.39 1.56 1.96 0.63 | 1.38 |
| sienna | 1.15 1.14 1.20 1.15 | 1.37 1.43 1.78 1.21 | 1.25 |
| yellow | 1.42 1.27 1.22 1.38 | 2.35 1.73 1.69 1.93 | 1.36 |
| red (mixed) | 1.09 1.08 1.13 1.17 | 1.27 1.32 1.38 1.44 | 1.22 |
| ivory_black (assumed) | 3.4 3.2 3.1 2.9 | 10.7 9.3 8.1 11 | 2.87 |
| gesso | 4.0 3.6 3.0 5.4 | 11.7 9.5 7.4 27 | 2.69 |
| paper | 4.2 3.4 2.8 12 | 11.9 9.4 7.5 31 | 2.81 |
| acrylic (latex) | 2700 457 127 240 | 2e4 3600 1100 1360 | 7.9 |
| ultramarine (guessed n 1.50) | 98 99 100 91 | 980 880 750 1180 | 8.9 |
| ochre / rose / green / dioxazine (guessed) | 1.3-2.9 | 1.8-9 | 1.5-2.7 |

Size sensitivity of air/oil and water/oil at 550 nm, for d50 × 0.5, × 1 and × 2:

| species | ×0.5 | ×1 | ×2 |
|---|---|---|---|
| white | 1.16, 1.14 | 1.84, 1.27 | 1.14, 1.06 |
| carbon | 0.98, 1.00 | 1.54, 1.13 | 2.52, 1.27 |
| sienna | 1.13, 1.10 | 1.49, 1.16 | 1.75, 1.18 |
| yellow | 2.2, 1.4 | 1.9, 1.3 | 1.2, 1.0 |
| gesso | 16, 4.4 | 10, 3.6 | 6.6, 2.9 |
| paper | 13, 5.3 | 9.5, 3.4 | 9.8, 3.3 |

The direction is robust. The size of the ratio depends on the assumed particle size within a factor of about 2.

What this predicts:

- High-index pigments (rutile, sulfides, hematite, carbon) scatter only **1.2-1.4×** more when a wash dries.
  Their drying shift is small, and comes mostly from the surface and from the paper under them.
- Low-index materials scatter **3-10×** more: ultramarine (index near oil), the grounds, and bone black's
  apatite. A wet paper ground scatters about 3× less than dry paper (R of the ground falls when wet). Gesso does
  too, by about 2.7×. The v2 design expected gesso's fall to be weak; it is not weak per unit scattering. A
  lean, unbound film (air in the pores) hides 1.4-1.9× better than the same pigment in oil in G for the
  high-index pigments, and much better for low-index ones.

Against Handprint (handprint.com/HP/WCL, read only):

| What Handprint reports | Verdict |
|---|---|
| Ultramarine PB29 lightens "by almost 30%" and drops "almost 20% in chroma" (`waterb.html`) | **agree** in direction (largest of all, ours ×8.9). The magnitude cannot be checked: our index is a guess, and the ratio goes roughly as (Δn)² near a match |
| Lamp black lightens by 180%, ivory black by "a mere 85%" (`waterw.html`) | **agree** in direction. **Disagree** in ranking: we predict ivory black's S rises more (×2.9 against ×1.4), and our ivory black is an assumed composite. Handprint's lightness change includes the wet surface, which is not in these numbers |
| Titanium white: "almost no drying shift, although they become much less opaque as they dry" (`waterw.html`) | a white over a white ground reads as less opaque when the paper brightens on drying (paper's S ×2.8), even though its own S rises ×1.4. **Agree** once the ground is included |
| Cadmium yellows "darken slightly", losing 2-15% saturation (`watery.html`) | **partly agree**: chroma loss yes (more S, paler blue absorption band). We predict slight lightening, not darkening |
| Iron oxides: "small to very large drying shift, losing saturation by more than 20%" (`watere.html`) | **agree**: modest (×1.25) |

Handprint's general drying-shift page (`pigmt8.html`) says loss of chroma is "consistently the primary change".
That is consistent with S rising in air.

**Finding (pointwise).** "Scatters more in air than water than oil" holds at every wavelength for non-absorbing
particles of index 1.5-2. It also holds in the R, G and B band means for every species. Pointwise it fails for
rutile at 380, 420 and 1550 nm and for carbon at 1550 nm. At a large relative index, a particle sized for a
binder is past its optimum in air, and the smaller size parameter in a low-index surrounding costs more than the
contrast gains. The tests check the claim only where it holds.

## Check 4: opacity in oil

Each film is at 20% pigment volume, with contrast ratio 0.98 (over black and over R = 0.80). The hiding thickness
is given at 550 nm and at the pigment's least-absorbing wavelength in 450-700 nm.

| species | hiding at 550 nm | hiding at least-absorbing λ | W&N oil rating |
|---|---|---|---|
| carbon | 0.95 µm | 0.85 µm | Lamp Black: "Opaque" |
| sienna (hematite 0.3 µm) | 0.88 µm | 1.9 µm (700 nm) | Burnt Sienna: "Transparent" |
| white | 7.8 µm | 5.9 µm | Titanium White: "Opaque" |
| yellow | 10.7 µm | 18 µm | Cadmium Yellow: "Opaque" |
| ivory_black (assumed) | 14 µm | 15 µm | Ivory Black: "Semi-Opaque" |
| red (mixed) | 145 µm | 41 µm | vermilion: Tr 0, "very opaque" (Handprint, watercolour) |
| ultramarine, rose, dioxazine, ochre, green (goo's scale) | 170-1000 µm | 550-2900 µm | French Ultramarine: "Transparent" |

The W&N ratings come from the Winsor & Newton Artists' Oil Colour product pages (uk.winsornewton.com), read only.

- **Agree:** carbon, white and cadmium yellow are opaque; ivory black is less opaque than lamp black.
- **Disagree:** burnt sienna. Pure hematite at 0.3 µm is a hiding red oxide. Hiding at 650 nm goes 10.8 µm →
  1.3 µm from 0.03 µm → 0.3 µm. A transparent burnt sienna needs nano-sized oxide or dilution by clay. Both are
  real, and both are outside this table: we cannot tell which from CC0 data.
- The guessed pigments' "transparency" reflects goo's 150× weaker scale more than the pigments themselves.

## Check 5: masstone in oil

R∞ from K/S, with the Saunderson correction (k1 = 0.04, k2 = 0.60), under D65 with the CIE 1931 2° observer, in
sRGB.

| species | sRGB | L* a* b* | in words |
|---|---|---|---|
| white | 250 250 250 | 98 0 0 | white (K = 0, so it is neutral: no cream) |
| carbon | 12 14 16 | 4 −1 −1 | black, very faintly cool |
| ivory_black (assumed) | 8 7 7 | 2 0 0 | black |
| sienna | 87 46 36 | 24 17 14 | dark red-brown |
| yellow | 255 192 0 | 84 20 114 | deep, warm cadmium yellow |
| red (mixed) | 219 0 26 | 44 75 50 | scarlet-red (vermilion-like) |
| gesso, paper | 250 250 250 | 98 0 0 | white |

- **Agree** qualitatively: hematite is a red-brown, CdS a yellow, rutile white, carbon black. Goethite (yellow-brown)
  could not be checked, because it is guessed (the guess gives L* 39 a* 6 b* 33, a dull yellow-brown).
- No freely readable, citable masstone coordinates in oil were found for these pigments. Quantitative comparison
  needs the RIT Munsell Color Science Laboratory's Artist Paint Spectral Database (Okumura 2005; Berns 2022) or
  Golden's drawdown spectra (see below).

## Check 6: the 1.5 µm band

The film is 50 µm at 30% PVC in oil, and R is given over black and over white (0.9). A pigment is "transparent"
when the two differ.

| species | R over black / over white | ours | reference: chsopensource.org (InGaAs vs ~1.1 µm camera, read only) | verdict |
|---|---|---|---|---|
| carbon | 0.002 / 0.002 | absorbing | carbon underdrawings are what IRR images | **agree** |
| ivory_black (assumed) | 0.011 / 0.011 | absorbing | ivory black: no transmittance gain with InGaAs | agree (weak test) |
| white (rutile) | 0.91 / 0.95 | scattering, S falls 11× from 550 nm | titanium white "more transparent with InGaAs" | **agree** (direction) |
| yellow (CdS) | 0.92 / 0.95 | non-absorbing, scattering | cadmium yellow: no gain | agree |
| red (HgS) | 0.97 / 0.97 | non-absorbing, strongly scattering | vermilion: "does not increase its transparency, at all" | agree |
| sienna (hematite) | 0.39 / 0.39 | **absorbing**, K/S 0.47 | red and yellow ochre "more transparent with InGaAs" | **disagree**. Querry's k ≈ 0.02-0.04 at 1.55 µm (from pellet reflectance) is very likely too high in hematite's transparent range. A transmission measurement of hematite's k at 1-2 µm would settle it |
| guessed pigments | — | goo's guesses | ultramarine and viridian: no gain | no test |

## What is still guessed, and what would replace it

| species | what is needed | possible sources and terms |
|---|---|---|
| ochre (goethite) | n, k of goethite 380-1550 nm | published goethite constants (e.g. from the atmospheric-dust literature) if released CC0. Otherwise, ask the authors for permission to deposit them in refractiveindex.info |
| ultramarine | n, k of lazurite, or K, S of PB29 in oil | the RIT Artist Paint Spectral Database (MCSL, RIT: request from Roy Berns' group); Golden's drawdowns (Golden Artist Colors, on request) |
| rose, dioxazine | K, S of PV19 and PV23 in oil (organic crystals: n, k are rarely published) | as above. Mie is the wrong tool for organic pigment aggregates; measured K, S are better |
| green (viridian) | n, k of Cr2O3·2H2O, or K, S | as above |
| red below 620 nm | k of HgS 400-620 nm | an ellipsometry or absorption study deposited CC0. RIT data on request for K, S |
| ivory black | n of hydroxyapatite (well known, about 1.63, but not in the CC0 database) and bone black's carbon fraction, or measured K, S | an apatite file in refractiveindex.info would make the host index `computed` |
| linseed oil | n(λ), k(λ) of linseed oil (yellowing) | refractometry plus transmission of cured films. No CC0 source was found |
| all pigments | measured masstone and tint spectra in oil and in gum arabic, to calibrate the Mie → K-M step (crowding, aggregation) | RIT database (Okumura 2005; Berns 2022), Golden drawdowns, IFAC-CNR FORS (fors.ifac.cnr.it). Each must be requested, and **we have not downloaded or copied any of them** |

## Not read or not downloaded, and why

- The RIT Artist Paint Spectral Database, Golden's spectra, and the IFAC-CNR FORS and watercolour databases.
  Each needs registration, a request or permission, so none of them was downloaded.
- The LBNL pigment database's spectra were not downloaded either. We read only its commentary PDF on whites.
- Handprint's pages and manufacturers' pages were read for a few cited statements. Nothing was copied or scraped.
- The Gamblin, Daler-Rowney and Golden charts did not render, and were not pursued.

## Regenerating

```
git clone --depth 1 https://github.com/polyanskiy/refractiveindex.info-database /somewhere/rii
pip install miepython pyyaml colour-science     # generator only
REFRACTIVEINDEX_DB=/somewhere/rii python studies/optics_data.py            # diff against the shipped table
REFRACTIVEINDEX_DB=/somewhere/rii python studies/optics_data.py --write    # rewrite goo/data/optics.json (~3 min)
REFRACTIVEINDEX_DB=/somewhere/rii python studies/optics_data.py --report   # the tables above
```

`tests/test_optics_data.py::test_regenerates_from_the_cc0_database` regenerates five species and requires no
difference. It is skipped when `REFRACTIVEINDEX_DB` or miepython is absent.

Packaging: `pyproject.toml` lists `goo.data` and `goo.matter` and ships `goo/data/*.json` as package data, so a
non-editable install includes the table.
