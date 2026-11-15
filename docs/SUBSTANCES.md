# goo.substance

What the kernel knows about matter (SPEC §4): descriptions, the admissibility laws, the derived maps that are
`goo.world`'s and `goo.light`'s only window onto matter, and the library. Owner: substance. Tests:
`tests/substance/`. `goo.substance` imports `goo.units` and `goo.evidence` only, and nothing of `goo.classic`.

| module | holds |
|---|---|
| `description` | `Substance`, `Thermo`, `Rheo`, `Reaction`, `StateVar` |
| `dielectric` | `Oscillators` (Lorentz, Drude, causal Gauss), `Tabulated`, `Free`; `eV`, `HC` |
| `morphology` | `LogNormal`, `Sphere`, `Coated`, `Hollow`, `Shape`; `NotDerivable` |
| `mie` | BHMIE and BHCOAT, vectorised, chunked |
| `response` | Mie to two-flux per unit volume; band sampling; the precomputed `OpticalTable` |
| `laws` | the five predicates, `classify` |
| `maps` | the maps of SPEC §4.3, with `reactions`, `ground_optical`, `ground_surface` |
| `library` | the substances |
| `_nk_data` | CC0 n, k on 350-1700 nm, generated; do not edit |
| `generate_nk.py` | its generator (a script; the package does not import it) |

## 1. The description

`Substance(name, eps, morph, rho, thermo, rheo, reactions, reservoir, prov, state, material)`: a frozen, hashable
dataclass. SI throughout; temperatures of reference values are `T_ref = goo.units.T_room` (20 C).

| field | unit | meaning |
|---|---|---|
| `name` | | a label; no code branches on it |
| `eps` | 1 | dielectric function of wavelength (m). `Oscillators(eps_inf, terms)`: energies in J (write `2.3 * eV`); `Tabulated(lam, parts, lo, hi, citation)`: n, k on a grid, weighted principal parts (uniaxial: 2/3 o + 1/3 e), described on [lo, hi]; `Free`: n, k with no claim |
| `morph` | m | `None` for a continuous phase; else `Sphere`, `Coated`, `Hollow` or `Shape` of a `LogNormal(d, sigma_g)`, d the volume-median diameter |
| `rho` | kg/m^3 | density (of the solid, for particles) |
| `thermo.M` | kg/mol | molar mass; sets the mixing volume M / (rho N_A). None: particles (mean particle volume) or a macromolecular phase (v = inf) |
| `thermo.p_sat` | Pa | saturation vapour pressure at T_ref; volatile iff > 0 |
| `thermo.dH_vap` | J/mol | enthalpy of vaporisation: Clausius-Clapeyron, and the DERIVED cohesive energy (dH_vap - R T) rho / M |
| `thermo.cohesion` | J/m^3 | cohesive energy density of a non-volatile substance (Hildebrand parameter squared). Refused when derivable |
| `thermo.gamma` | J/m^2 | surface tension against air (for a solid: its surface free energy) |
| `thermo.gamma_d` | J/m^2 | the dispersive part of gamma (Owens-Wendt); gamma - gamma_d is the polar part. Gives wetting and capillary suction by the geometric-mean rule |
| `thermo.D_vapour` | m^2/s | diffusivity of its vapour in air at T_ref |
| `rheo.eta` | Pa s | Newtonian viscosity at T_ref |
| `rheo.E_eta` | J/mol | viscous activation energy (Arrhenius) |
| `rheo.separation` | m | closest approach of two particles; sets the attraction that gives a yield stress |
| `rheo.phi_percolation` | 1 | volume fraction at which an attractive network first spans (YODEL's phi_0) |
| `rheo.coupling` | as declared | Onsager matrix among the rates of the declared state variables |
| `reactions` | kg/kg, 1/s, J/mol | `Reaction(stoich, rate, E_a)`: stoichiometry by mass (reactants negative), first-order rate at T_ref, activation energy |
| `reservoir` | J/m^3 | energy an impossible substance may draw; 0 for every admissible one |
| `prov` | | a `Provenance` for every field and every given parameter, keyed `eps`, `morph`, `rho`, `reservoir`, `reactions`, `thermo.<p>`, `rheo.<p>`; missing or extra keys are refused |
| `state` | | `StateVar(name, unit, lo, hi)` the substance declares (none in the library yet) |
| `material` | | the named material a "real" description is computed or checked against |

A parameter left `None` makes no claim and needs no provenance; maps that would need it mark the cells where the
substance is present out of domain (or say what they do instead).

The free-energy terms these give (per unit film volume): ideal mixing kT sum (phi_s / v_s) ln phi_s; Hildebrand-
Scatchard interaction (1/2) sum B_ij phi_i phi_j with B_ij = (sqrt c_i - sqrt c_j)^2; capillarity through gamma;
gravity through rho; exchange through p_sat. The dissipation terms: viscous (1/eta), yield (tau_y, flat Psi*),
vapour transport (D_vapour), reactions (rate), coupling.

## 2. Admissibility (`laws`)

Each predicate returns `Verdict(law, ok, shortfall, unit, detail, parts, method)`; `parts` lists every term measured.

| law | test | shortfall, unit |
|---|---|---|
| herglotz | oscillators of non-negative strength and width with eps_inf >= 1: proved. Else Im eps over the described range, and classic's finite-range Kramers-Kronig test (NNLS over every passive extension, remainder <= 2 %) | max(gain, remainder - 2 %), 1 |
| stable | rho > 0, gamma >= 0, cohesion >= 0, M > 0, p_sat >= 0, d > 0, sigma_g >= 1 | the worst term's, in its unit |
| dissipative | eta > 0, separation > 0, 0 <= phi_percolation < 1, D_vapour >= 0, rates >= 0, coupling's symmetric part PSD | the worst term's |
| reciprocal | coupling = coupling^T (isotropic eps and Hildebrand interaction: by construction) | max abs(L - L^T) / 2 |
| stoichiometric | each reaction's coefficients sum to 0 (1e-12); the substance is a reactant | abs(net), kg/kg |

`classify(s)` gives `Classification(klass, passed, failed, verdicts, blocking)`: impossible if any law fails; real if
possible, `material` is named, and every provenance is proved, exact, computed or checked; else possible, with
`blocking` listing what keeps it from real.

## 3. The maps (`maps`)

A film is any object with `.substances`, `.q`, `.h`, `.phi`, `.state` (`temperature` read when present); a ground
any object with `.solid`, `.porosity`, `.depth`, `.substances`, `.held`, `.z` (goo.film is not imported). Roles come
from descriptions: a morphology means a particle population. Voids: in a film V = max(0, P (1/phi_m - 1) - L); in a
ground, the pore volume its held matter leaves empty. Evidence: the weakest of the map's and of every field read
from the substances present; out of domain at best assumed.

A map that returns several quantities returns a named tuple in the table's order, each member its own Derived or
Bounded, so it unpacks by position: `K, S, n_host = optical(film, bands)`, `n_top, sigma = surface(film, bands)`,
`activity, activity_env, rate, p_sat = evaporation(film, env)[name]`.

**Why a cell is out of domain.** Every `in_domain` is false exactly where one of the map's named conditions fails,
and the Derived's (or Bounded's) `map` text names each failed condition with the number of points it covers, e.g.
`optical.K; out of domain: eps of rutile not described across bands 390, 410, 430 nm (described 430-1530 nm)
[3 of 18 points]`. The kinds of reason: a source's wavelength range exceeded at named bands; loading above the
independent-scattering bound; host index outside the precomputed grid; a particle shape without a law; liquids that
separate by the regular solution; jammed (phi >= phi_m); a powder (no liquid); a missing parameter (eta, gamma,
E_eta away from T_ref, dH_vap away from T_ref, D_vapour, phi_percolation); polydispersity beyond the packing fit;
particle Reynolds number too high; molecular Stokes-Einstein; voids (yield); an absorbing host; a reaction product
not in the film; a ground's overfull pores or grainless solid. Each kind is tested (`tests/substance/test_reasons.py`).
Empty cells return a stated convention and are in domain.

| map | returns | model | domain conditions | map evidence |
|---|---|---|---|---|
| `optical(film, bands)` | `Optics(K, S, host)` [B,H,W]; K, S in 1/m of h | sum phi_p K_p(n_h) + liquids' 2 (4 pi k / lambda); S = sum phi_p S_p(n_h); Mie per unit volume in a non-absorbing host, K = 2 mu_a, S = 3/4 mu_s (1-g), band means; host by Lorentz-Lorenz; with voids host, K, S `Bounded` (Wiener bounds) | eps described across the band; host 1.0-1.8; particle fraction <= 0.10; spheres; host absorption small; one liquid phase; packing fit | computed |
| `optical_exact` | the same, Mie at each cell's host index | | | |
| `ground_optical(ground, bands)` | `Optics(K, S, host)`; per unit depth | the same physics: the solid's particles at 1 - porosity, the held matter, air in the rest of the pores | as `optical`; solid with a morphology; pores not overfull | computed (at best assumed: grounds exceed 0.10) |
| `surface(film, bands)` | `Surface(index [B,H,W], roughness [H,W] m)` | overbound: the liquid's index, roughness 0; underbound: the liquid recedes from the top; roughness of a hexagonal sphere monolayer standing out of a flat liquid (computed geometry) at the empty fraction min(1, V / (0.395 d32)); index Wiener-bounded | spheres, one liquid, liquids' eps described | assumed |
| `ground_surface(ground, bands)` | `Surface(index, roughness)` | as `surface`, solid at 1 - porosity; the top is always a composite (Bounded); roughness 0 when the pores are full | solid with a morphology | assumed |
| `density(film)` | rho [H,W] kg/m^3 | sum phi_s rho_s (volumes add) | always; empty: 0 | assumed (SPEC §3) |
| `viscosity(film)` | eta [H,W] Pa s | liquids log-mixed, Arrhenius in T; Krieger-Dougherty with [eta] = 5/2 and phi_m from `packing`, phi = P / (P + L) | not jammed (else inf); a liquid (powder: inf); eta given; miscible; E_eta where T != T_ref | computed (assumed with two liquids) |
| `yield_stress(film)` | tau_y [H,W] Pa, `Bounded` | YODEL form, prefactor 1: G(phi) sum x_s A_s / (12 H_s^2 d32_s); Hamaker by Tabor-Winterton from eps at 589 nm, lo without and hi with the zero-frequency 3kT/4 | `yield_stress_domain(film)`: not jammed, a liquid, no voids, phi_percolation given | assumed |
| `packing(film)` | phi_m [H,W] | Desmond & Weeks on the number distribution of every population present | polydispersity <= 0.6, spheres; no particles: 0.634 | computed |
| `diffusivity(film)` | D [S,H,W] m^2/s | particles: kT E_n[1/d] / (3 pi eta_film); molecules: kT / (6 pi eta_l r_mol) | the viscosity's; molecules always out | computed |
| `settling(film)` | v [S,H,W] m/s along g at g_0 | Stokes with E_v[d^2], Richardson-Zaki (1-phi)^4.65; envelope density | Re < 0.1, a liquid, its viscosity | computed |
| `chemical_potential(film)` | mu [S,H,W] J/m^3 | Flory-Huggins plus Hildebrand-Scatchard; mu = d(h f)/dq | always (the model F itself); absent: -inf | assumed |
| `surface_tension(film)` | gamma [H,W] J/m^2 | liquids' gamma weighted by volume among liquids | gamma given, a liquid present | exact for one liquid, assumed for more |
| `evaporation(film, env)` | {name: `Exchange(activity, activity_env, rate, p_sat)`} | a = exp(mu v / kT); J = rate (a - a_env), rate = D_v M p_sat(T) / (R T delta); Clausius-Clapeyron; D_v ~ T^1.75 | dH_vap where T != T_ref, D_vapour given | assumed |
| `reactions(film, env)` | [`ReactionRate(substance, reaction, rate)`], rate [H,W] 1/s | Arrhenius, times env activities of reactants the film lacks | every product a substance of the film | assumed |

The environment is `goo.film.Environment` (read duck-typed: `.T`, `.activity`, `.boundary_layer`);
`maps.Atmosphere` is a stand-in with the same field names.

**goo.world's protocol** (goo/world/protocol.py). The world-facing maps return one Derived whose value is a named
tuple of plain arrays, the form goo.world reads (its `coeff` turns each member into an array; a tuple of Derived
would not unpack there).

| map | returns | model |
|---|---|---|
| `mixing(film, T)` | `MixingTerms(f, mu, hess, f_x, mu_x)`; J/m^2, J/m^3, J/m^4 | F_mix = h f_v: Flory-Huggins entropy (convex: the perspective of sum c x ln x) plus the Hildebrand interaction (1/2) q^T B q / h, split with beta = max eig B: convex part = entropy + (beta/2)(q^T q / h - h), concave part = (1/2) q^T (B - beta I) q / h + (beta/2) h. A pure substance has 0 in each; mu = -inf and hess = 0 where a substance is absent |
| `mobility(film)` | Lambda [S,H,W] m^2/(Pa s) | D v / (k_B T) from `diffusivity` (for spheres d^2 / (18 eta)) |
| `exchange(film, env)` | `ExchangeRates(k [S,H,W] m/(Pa s), mu_env [S] J/m^3, activity [S,H,W])` | mu_env = (k_B T / v) ln a_env for a volatile the environment names; k = (R / rho)(a - a_env)/(mu - mu_env), the secant of the boundary-layer law R = D_v M p_sat / (R_gas T delta) at the present state, so the flux there is exact; near equilibrium k = D_v M^2 p_sat a / (delta rho^2 (R_gas T)^2). a_env = 0 gives mu_env = -inf and k = 0, out of domain |
| `wetting(film, ground)` | `Wetting(S, ell)` [H,W] | S = 2 sqrt(gd_s gd_l) + 2 sqrt(gp_s gp_l) - 2 gamma_l (<= 0; complete wetting reported as 0); ell = sqrt(abs(A) / (12 pi abs(S))), A by Tabor-Winterton for solid / liquid / air, at least the molecular length |
| `sorption(film, ground)` | `Sorption(P, kappa)` [S,H,W] | per liquid substance: P = 2 gamma cos theta / r, cos theta from its own spreading coefficient on the ground's solid, 0 when non-wetting or the pores are full; kappa = porosity r^2 / (8 eta L), Lucas-Washburn with L = max(held / porosity, r) |
| `evaporation(film, env)` | {name: `Exchange(activity, activity_env, rate, p_sat)`} | the boundary-layer law in activities and kg/(m^2 s), as before (the former `exchange`) |

Every world-facing map is element-wise (tested on windows and on compacted [S, 1, n] films) and finite on every
cell, with two stated exceptions the protocol allows or the convention requires: mu = -inf at an absent substance,
and viscosity (and yield stress) = +inf where nothing can flow (no liquid, or jammed; out of domain except on empty
cells). An empty cell's surface tension is the mean gamma of the film's liquid substances (what may arrive).

**Identical cells are solved once** in `optical`, `surface`, `ground_optical` and `ground_surface`: cells are grouped
by exact equality of (q, T, voids) with `np.unique` on the values (never a hash), solved on the distinct cells and
scattered back; reason counts are weighted by multiplicity. A 400 x 300 painting-like film (801 distinct cells),
18 bands, optical + surface: 0.84 s before, 0.28 s after, results identical.



**Precomputed optics.** Per particle substance and band set, K and S are computed by Mie once on host indices
1.00-1.80 by 0.01 (band samples one per 10 nm, at most 8; 16 Gauss-Hermite size nodes) and spline-interpolated per
band; `optical` costs ~0.2 s on 256 x 256 cells, 19 bands, three substances. Interpolation error at the grid's
mid-points, as a fraction of the band's largest S (and K against the larger of the two), measured and tested:

| substance | error | substance | error |
|---|---|---|---|
| soot, rose, dioxazine, PMMA latex | <= 1e-7 | rutile | 4.0e-3 |
| hematite, gain_green | <= 2e-6 | ultramarine | 2.3e-3 |
| green | 6e-6 | cadmium sulfide | 9.3e-4 |
| ochre | 1.1e-4 | calcite (3 um) | 2.1e-2 |
| | | cellulose (20 um) | 3.4e-2 |

Large particles carry Mie's ripple in the host index; the grid would have to be finer for them. K >= 0 and S >= 0
are imposed after interpolation where they are theorems (passive particles; every S).

## 4. The library

Classes are computed by `classify`; `tests/substance/test_library.py` holds the expected table.

**Real** (2): `water` (Hale & Querry, CC0; every constant read and checked, below), `pmma` (a clear film: Zhang et
al., CC0; density checked against the infobox of its Wikipedia article).

**Checked means read.** A constant is `checked` only when its single value was read, on 2026-11-14, in the source
its provenance cites (page or table named; nothing copied); a value recalled from memory is `assumed`, with the
source it should be checked against.

| constant | source read | result |
|---|---|---|
| water rho, eta (20 C), eta (25 C) | NIST WebBook fluid isobar (IAPWS-95) | verified; 998.21 -> 998.207 kg/m^3, 0.8900 -> 0.890022 mPa s |
| water p_sat, dH_vap, gamma (20 C) | NIST WebBook saturation table, interpolated | verified (2.3394 kPa, 44.20 kJ/mol, 72.74 mN/m) |
| water M | NIST WebBook | 18.015 -> 18.0153 g/mol |
| water D_vapour | Marrero & Mason 1972, Table 13 (NIST SRD reprint) | verified (0.242 cm^2/s at 293 K) |
| ethanol M, p_sat, dH_vap | NIST WebBook (Antoine, Kretschmer & Wiebe) | verified; M 46.069 -> 46.0684 |
| ethanol rho | Wikipedia data page (Lange) | 789.3 -> 789.45 kg/m^3 |
| ethanol eta, E_eta, gamma | sources disagree or none at 20 C | downgraded to assumed |
| rutile, hematite, CdS, calcite, PMMA rho | Wikipedia infoboxes | hematite, calcite, PMMA verified; rutile 4250 -> 4230, CdS 4820 -> 4826 |

**Possible** (16):

| entry | eps | what blocks real |
|---|---|---|
| rutile | Devore 1951, o + e (CC0); k = 0 assumed | eps (k assumed), separation, phi_percolation |
| soot | Dalzell & Sarofim 1969 (CC0) | size, density, separation, phi_percolation |
| hematite | Querry 1985, o + e (CC0) | size, separation, phi_percolation |
| cadmium_sulfide | Treharne et al. 2011 (CC0) | size, separation, phi_percolation |
| calcite | Ghosh 1999, o + e (CC0); k = 0 assumed | eps, size, ... |
| cellulose | Juntunen et al. 2024 (CC0), 1.10-1.65 um only | eps, size, density, ... |
| pmma_latex | Zhang et al. 2020 (CC0) | size, ... |
| ethanol | Kedenburg et al. 2012 (CC0), 0.5-1.6 um | D_vapour, gamma, eta, E_eta (assumed) |
| linseed_oil | one UV oscillator for n_D 1.479 (assumed); cure reaction with oxygen (guessed) | everything assumed or guessed |
| linoxyn, gum_arabic | guessed | guessed |
| ochre, ultramarine, rose, green, dioxazine | guessed causal-Gaussian bands on the physical scale (k ~0.05-0.5) | guessed |

The CC0 entries cite the measurement and the database (refractiveindex.info, commit e6d15e364f70). Rutile's size is
checked against the LBNL white-pigment commentary as recorded in classic/docs/OPTICS-DATA.md (check 2); every other
size is assumed (SPEC §4.5). Kramers-Kronig remainders of the tabulated entries: soot 1.98 % (just inside the 2 %
tolerance), hematite 0.5 %, CdS 0.3 %, PMMA 0.07 %, water 0.06 %, rutile 0.01 %, calcite, cellulose, ethanol ~0.

**Impossible** (4): `gain_green` (a negative causal-Gaussian band at 540 nm: fails herglotz by 0.05, reservoir
1e5 J/m^3), `acausal_blue` (flat n under a k band: fails herglotz, KK remainder 16 %; cannot be paid for),
`negative_friction` (eta = -1 mPa s: fails dissipative; reservoir 1e3 J/m^3), `mass_from_nothing` (makes 1.5 kg per
kg: fails stoichiometric).

## 5. What was promoted from classic, what was left, and why

Promoted (copied and rewritten): the dielectric family (wavelengths in m, energies in J), the morphologies (sizes
in m), Mie (BHMIE and BHCOAT, unchanged in substance, with chunking), the Mie-to-two-flux map (SI, arbitrary bands,
host table on a finer grid), the mechanics maps (Desmond-Weeks, Einstein, Krieger-Dougherty, Stokes-Einstein, Stokes,
Richardson-Zaki, Fick evaporation), the Kramers-Kronig test, and the generator's refractiveindex.info reader (into
`generate_nk.py`, which now reads the database directly onto a 350-1700 nm grid instead of classic's 42 points).

Left behind: classic's 0.025 mm unit and its ~155x weak guessed table; `classic.py`, `phenotype.py`, `search.py`,
`example.py` (they drive the old engine; the lab is the second wave); `gaps.py` (a list of the old engine's lines);
the `boundedness` and `thermodynamic` laws (SPEC has five laws; their sign conditions moved into `stable` and
`dissipative`, the f-sum and size bounds were dropped); `km_any` and `ENERGY_LEDGER` (they belong to light and the
ledger); classic's `Change` (volume yields) for `Reaction` (mass, SPEC §3); vermilion and ivory black (their n or k
below 620 nm, or their composition, are guessed or assumed in classic's table; not promoted as real); the oil's
guessed yellowing K (no CC0 source); the disk cache.

Found while promoting: classic's `optics.json` holds the orientation-weighted n and k (2/3 o + 1/3 e) for the uniaxial
crystals, not the ordinary ray as classic/docs/MATTER.md says.

## 6. Assumed or guessed, and negative results

- Independent scattering above 10 % particle volume is out of domain: most paint cells' optics are `assumed`.
- Two-flux from Mie (van Gemert-Star without -mu_a/4); flat spectrum inside a band; uniaxial averaging.
- Host of a mixed-liquid cell: Lorentz-Lorenz; with voids only Wiener bounds.
- Surface roughness: a monolayer geometry and a receding flat front, assumed.
- Yield: YODEL's prefactor taken as 1; nu_e = 3e15 Hz; separations and percolation onsets assumed.
- Liquid viscosity mixing (Arrhenius), surface tension mixing (linear): assumed.
- Molecular Stokes-Einstein: out of domain by construction.
- **Negative:** the regular solution (Hildebrand) predicts water and ethanol separate (B = 4.8e8 J/m^3 against a
  critical 1.6e8); they mix in all proportions. Hydrogen bonding is not in the description. The test is `xfail`.
- **Negative:** under SPEC §4.2 as written, with sizes assumed (§4.5) and k = 0 assumed for transparent-range
  sources, no particle substance in the library is real.

## 7. In goo.world

`tests/substance/test_world_protocol.py` builds a 6 x 6 world (dx 0.1 mm) of 20 um of water on a cellulose ground
(porosity 0.5, 5 um pores, 100 um deep) at 50 % humidity with `goo.substance.maps` as its constitutive object, and
steps it: matter closes to ~1e-16, the energy identity to ~1e-12, every D_mech >= 0, water leaves to the air and into
the pores (Lucas-Washburn is fast: the film is in the pores within the first millisecond). Two shims remain in the
test until goo.world catches up, both chosen automatically: exchange's k and the environment's potentials for the
world that still reads `env.potentials`, and the yield stress's Bound collapsed at its low end. Wetting is left out
of that test: with a molecular wetting length (0.3 nm) the world's dF_wet multiplies exp(-h/ell) = 0 by
expm1(-dh/ell) = inf; a strict xfail keeps the case. Found on the way and fixed here: a pure substance's mu must be
exactly 0 (c phi, not c q / h), or the world's one-substance diffusion becomes singular; an empty cell needs a
surface tension, or the world's capillary stabilisation fails where liquid arrives; the suction must not depend on
whether the cell's film is present, or the lagged pore term changes when a film drains.
