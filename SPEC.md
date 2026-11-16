# goo: the kernel contract

**Status.** This is the contract for goo 0.2, the kernel that replaces goo classic. It says what must
be true of the code. §12 says what is true today; until a line there says otherwise, nothing here is
implemented. `docs/VISION.md` gives the reasons and is not a contract. The first engine lives in
`goo.classic`, deprecated; its own contract is `classic/SPEC.md`.

## 0. The break

1. Nothing in `goo` outside `goo.classic` imports `goo.classic`. `goo.classic` imports only itself.
2. Classic is frozen. It is changed only to keep it importable, and its goldens stay bit-identical.
3. Classic is an oracle, never a dependency. Tests under `tests/oracle/` may import both and compare
   them in a stated limit. Where the kernel disagrees with classic, the disagreement and its cause are
   written in `docs/NEGATIVE.md`. It is not tuned away. The kernel does not owe classic's pictures.
4. Code crosses from classic by promotion: copied deliberately, with its tests, and with whatever
   breaks this contract removed. Never by import.
5. Names never select algorithms. No branch on a substance's name, a medium's name or a position in
   an array. "Watercolour" names a region of behaviour.
6. No smuggling. An effect exists only as a consequence of descriptions acted on by common laws and
   read by declared observers. A flag, a per-medium factor or a colour typed into a table is
   smuggling. If an effect cannot yet be produced that way, the choices are to extend a law for every
   substance or to say goo does not model it.

## 1. Modules

```text
module          holds                                                  may import
goo.units       constants, unit helpers                                (nothing)
goo.evidence    evidence classes, Provenance, Derived, Bounded         (nothing)
goo.ledger      accounts, entries, receipts, residuals                 units, evidence
goo.substance   descriptions, admissibility, derived maps, data        units, evidence
goo.film        Film, Stack, Ground, Bands, Light, Environment: data   units, evidence, substance
goo.light       layer operators, star product, looks, light ledger     film, substance, ledger
goo.world       state, F, Psi*, the step, matter and energy ledgers    film, substance, ledger
goo.ports       bodies and reservoirs coupled to the world             world, film, substance, ledger
goo.journal     acts, replay, forks, clocks, looks as pictures         world, ports, light, observe
goo.observe     sensors: measurement models over looks                 light, film
goo.lab         drawdowns, data sheets, admissibility reports, search  substance, film, light, observe
goo.classic     the first engine, deprecated                           itself
```

Every module may import those to its right and what they import. `goo.lab` imports neither
`goo.world` nor `goo.ports`: the lab can be used without the painting engine. `goo.light` does not
import `goo.world`: light is solved on a `Stack`, whoever made it. The table is enforced by a test
that scans imports.

## 2. Units, numbers, evidence

**SI throughout.** Length m, time s, mass kg, energy J, temperature K. A film's amount is volume per
area, in m. Optical coefficients are in 1/m, viscosity in Pa s, surface tension in J/m², chemical
potential in J/m³ of the substance. `goo.units` gives `um`, `nm`, `mPa_s` and the like for writing
values. There are no goo units and no second scale. Arrays are float64.

**Resolution is not physics.** The cell side `dx` and the time step `dt` belong to the world, and
every law must give the same physical answer as they are refined, at a stated order. No constant is
tuned to a cell size or a tick.

**Evidence.** Every constitutive number carries a `Provenance(evidence, source, licence, note)` with
`evidence` one of, strongest first:

```text
proved     a theorem, with the proof or its reference
exact      an identity of the arithmetic, tested to rounding
computed   derived by a stated map from permissively licensed data
checked    computed or assumed, and compared against a restricted reference that is cited, not copied
assumed    a stated modelling choice
guessed    a number chosen because it looks right
```

A derived value is a `Derived(value, evidence, in_domain)`. Its evidence is the weakest of its inputs
and its map's; outside the map's domain of validity it is at best `assumed` and `in_domain` is false.
Where only bounds are known (an effective property of an unknown microstructure) the value is a
`Bounded(lo, hi)`, and a consumer either carries both ends or records which it chose and why.

**Data policy.** The repository ships only numbers computed from permissively licensed data (CC0 or
public domain) with attribution, and goo's own guesses marked `guessed`. Restricted references are
read and cited for checks. Nothing is copied from them. A number recalled and not looked up is
`assumed`, whatever source it names.

## 3. Three ledgers

Each ledger is double entry: an entry moves an amount from one account to another, so a ledger's sum
is zero by construction and what is tested is that the entries account for the change of state. That
difference is the **closure residual**, a number reported with every receipt.

**Matter**, in kg per substance. Accounts: `wet`, `stratum[k]`, `ground`, `body[name]`,
`reservoir[name]`. Substances keep their densities, volumes add (evidence `assumed`), and a reaction
moves mass between substances by its stoichiometry. Residual at most 1e-12 of the mass present.
(VISION §4 says volume. The contract books mass: a curing oil takes up oxygen and shrinks, so volume
is not conserved and mass is.)

**Energy**, in J. Accounts: `stored` (the free energy F of the live world), `frozen` (what the strata
hold), `work[body]`, `dissipated[mechanism]`, `debt[name]`. A receipt also gives F by term, before
and after. For every step and every act:

```text
ΔF  =  W_ports  −  D_mech  −  D_num
D_mech  ≥ 0     computed from the fluxes applied and the forces that drove them, never as a remainder
D_num   ≥ 0     the remainder: what the discretisation dissipates beyond the mechanisms
```

`D_num ≥ 0` must follow from the construction of the step for admissible substances, with the
argument written in `docs/DYNAMICS.md`. In arithmetic it is at least −1e-12 of the energy moved.
That is the `exact` solver, which factorises its systems and is the reference. A world may instead
step with the `iterative` solver, which solves the same systems to a stated tolerance by a
deterministic rule: O1, O2, O5 and O6 hold as before, the sign of `D_num` holds only to that
tolerance, every receipt reports the solver's residual, and a step whose `D_num` is negative beyond
rounding is flagged. Tests compare the two. It
is booked to `dissipated[numerical]` and reported per step in two parts: one from the time step,
which must fall at the scheme's stated order as `dt` is refined, and one from transport on the grid
(numerical mixing), which must fall as `dx` is refined. A coefficient of F that the step holds fixed
and then updates changes F without a flux; that change is booked to `work[constitutive]` and
flagged, and making it vanish is owed (§12). The environment
enters through F (§6), so a change of the environment's potentials is work done by a port.

**Light**, per band, as a fraction of the incident power. Accounts: `incident`,
`reflected.specular`, `reflected.diffuse`, `absorbed[k]` for each film, `absorbed.ground`,
`transmitted` (out of the back of a ground of finite depth), `debt[k]` (power a film with gain
supplied) and `unresolved`. Residual at most 1e-12 of the power entering, incident plus debt. A film
past its lasing threshold has no steady state: the look returns zero there, books the incident power
to `unresolved`, and says in which cells. With coupled bands (fluorescence) the ledger is in energy and the Stokes loss is
booked to the film that caused it.

**Receipts.** Every operation that changes or reads the world (a step, an act, a retirement, a look)
returns a `Receipt`: its entries in each ledger it touched, the three residuals, and its flags:
impossible substances present and the laws they fail, maps used outside their domain, bounds that
were collapsed, privileges used (§8).

## 4. Substances

**4.1 Description.** A substance is described where physics is simple:

```text
name
eps            dielectric function over wavelength              how it meets light
morph          morphology, or none for a continuous phase       size distribution, shape, structure
rho            density
thermo         its terms in the free energy                     volatility, surface tension, interactions, wetting
rheo           its terms in the dissipation potential           viscosity, packing, yield, mobility
reactions      what it becomes: stoichiometry by mass, a free energy of reaction, a rate   cure, coalescence, dissolution
reservoir      finite energy an impossible substance may draw   0 for every admissible substance
prov           provenance of each field
```

The engine reads no number about a substance that is not one of these or derived from them by a map
in §4.3. The order of substances in a world has no meaning.

**4.2 Admissibility.** Each law is a named predicate that returns pass or fail with a measured
shortfall.

```text
herglotz        eps is the boundary value of a Herglotz function        passive and causal
stable          its terms keep F bounded below
dissipative     its terms in Psi* are convex, non-negative, zero at 0   the second law, every process
reciprocal      linear cross-effects are symmetric                      Onsager
stoichiometric  its reactions conserve mass
```

A substance is **possible** if it passes all five, **real** if it is possible and every field is
`computed` or `checked` against a named material, and **impossible** if it fails a law. The rule is
strict on purpose: a pigment whose particle size is assumed is not real, and `classify` lists what
keeps each substance from being real. Impossible
substances are allowed and never silent. A failure that yields power (optical gain, negative
friction) draws on the substance's `reservoir`, the energy ledger books the draw to `debt[name]`,
and when the reservoir is empty the yield stops. A failure that cannot be paid for (an `eps` that is
no function's boundary value) is a flag on every receipt and every look the substance appears in.

**4.3 Derived maps.** `goo.substance.maps`: functions of a `Film` (§5), vectorised over its cells.
Each returns `Derived` (or `Bounded`) and states its domain.

```text
optical(film, bands)       K, S [B,H,W] in 1/m; host index [B,H,W]   Mie in the host the film provides; independent scattering
surface(film, bands)       index of the top interface [B,H,W]; rms roughness [H,W] in m   from binder against packing
ground_optical(ground, bands)   K, S, host index of the ground's body, to its depth   the same maps, on a porous solid
ground_surface(ground, bands)   index and roughness of the ground's top interface
density(film)              ρ [H,W], kg/m³                             volumes add
viscosity(film)            η [H,W], Pa s                              liquids by a stated mixing rule; particles by Krieger-Dougherty
yield_stress(film)         τ_y [H,W], Pa                              from loading against packing
packing(film)              φ_m [H,W]                                  from the morphologies present
diffusivity(film)          D [S,H,W], m²/s                            Stokes-Einstein in the film's viscosity
mobility(film)             Λ [S,H,W], m²/(Pa s)                       D v / (k_B T)
settling(film)             velocity [S,H,W], m/s                      Stokes; vertical settling is reported, not applied, on a depth-averaged film
mixing(film, T)            f [H,W] J/m²; μ [S,H,W] J/m³; Hessian [S,S,H,W] J/m⁴; a convex part and a concave part   ideal mixing plus interactions
surface_tension(film)      γ [H,W], J/m²                              from the liquids present
exchange(film, env)        k [S,H,W], m/(Pa s); the reservoir's potential [S], J/m³; the film's activity   against the environment
wetting(film, ground)      spreading coefficient [H,W], J/m²; wetting length [H,W], m
sorption(film, ground)     suction [S,H,W], Pa; rate coefficient [S,H,W], m/(Pa s)   into the ground's pores
reactions(film, env)       per reaction: rate of extent, 1/s           from the substances' reactions and the reservoirs
```

Every map is element-wise over cells (a cell's answer depends on that cell alone) and finite on
empty cells, so the world may call it on a window or on a compacted list of cells. A resistance
(viscosity, yield stress) may be +∞ where a cell holds nothing that can flow, meaning zero mobility.

`K` and `S` are per unit of the film's `h`, so `K h` is the optical depth whatever voids the film
has, and `S` is the two-flux scattering coefficient with the asymmetry already taken. A map that
returns several quantities returns a named tuple in the order given, each item its own `Derived` or
`Bounded`. A map outside its domain still answers, as
`assumed`, and the receipt carries the flag.

**4.4 Consumers.** `goo.world` and `goo.light` reach substances only through these maps, and take
them as an argument: each declares the functions of a `Film` it needs as a protocol
(`goo.world.Constitutive`, `goo.light.Optical`), and each is tested first against analytic
implementations of its own protocol (a constant viscosity, a grey absorber). `goo.substance.maps`
must satisfy both. Where a consumer needs a function this table lacks, it says so in its report and
the lead amends the table.

**4.5 Data.** One scale, the physical one. The substances promoted from classic's data table are
those computed by Mie from CC0 optical constants. Classic's five guessed pigments enter as
`possible` substances with `guessed` dielectric functions on the physical scale, and stay so until
measured data replaces them. Particle sizes are `assumed` until measured.

## 5. Films and stacks

`goo.film` is plain data with no laws:

```text
Film      substances, q[S,H,W] in m, state: named fields (cure, coalescence, alignment, temperature)
Ground    relief, a porous description (solid, porosity, pore size), what its pores hold
Stack     a ground and films, bottom first
Bands     centres and widths in m; no code assumes how many
Light     an illuminant's spectrum and a geometry (diffuse, or collimated at an angle)
```

## 6. Dynamics

The world's state is the wet film, the strata, the ground, the bodies and the environment. Matter
relaxes under one statement:

```text
∂_t x  =  ∂Ψ*( x ; −DF(x) )  +  ports
```

`F` is the sum of the terms the substances present contribute, and `Ψ*` the sum of their dissipation
potentials. A law is a term of one or the other. Anything that cannot be written so is phenomenology,
and is either derived or left out. The environment's reservoirs enter F through their potentials.

**First rung:** a depth-averaged thin film. Its small parameter is the surface slope, reported with
every step; above a stated bound the receipt carries a flag. Vertical settling, skinning from the
top and thick impasto need the next rung and are not faked on this one.

**First terms.** Gravity, with a vector that may have a component along the panel. Capillarity.
Mixing, ideal plus interaction. Wetting of the ground. Exchange of volatiles with the atmosphere.
Sorption into the ground's pores. Cure as a reaction with a reservoir. Dissolution of a stratum's
substances into a liquid that takes them. Viscous flow with a yield stress, which is a `Ψ*` flat near
zero force.

**Obligations on the step.** Each is a test.

```text
O1  conservation   every substance's mass, exactly, by flux form
O2  positivity     no amount goes negative
O3  energy         ΔF = W − D_mech − D_num with D_mech ≥ 0 and D_num ≥ 0   (§3)
O4  balance        a state with no driving force and no ports does not move: exactly where the force is zero in
                   arithmetic (a flat film, a paste below yield, a held clock), to rounding where it is zero only analytically
O5  locality       a windowed step equals the whole-panel step, bit for bit
O6  determinism    a fork continues identically; a replay reproduces every array, bit for bit
O7  refinement     convergence in dx and dt at the stated order
O8  exact limits   with a term's parameter at zero the result is bit-identical to the world without the term
```

The derivation comes before the code: `docs/DYNAMICS.md` writes F and Ψ* for the first terms, the
discrete step, and the argument for O1 to O4.

**Retirement.** Turning part of the wet film into a stratum retires its lateral degrees of freedom:
it can no longer flow. The matter moves from `wet` to `stratum[k]`, the free energy it still holds
moves from `stored` to `frozen`, and nothing is dropped. A stratum is not thermodynamically dead:
the potentials of the top stratum's substances stay in F, and whether it gives matter back to a
liquid above it is decided by the dissolution term, from descriptions.

## 7. Light

Light is solved, per band, on a `Stack`. A film is a passive operator on the fluxes entering it from
above and below, and films compose by the star product:

```text
R₁₂ = R₁ + T₁ R₂ T₁′ (1 − R₁′ R₂)⁻¹        T₁₂ = T₂ (1 − R₁′ R₂)⁻¹ T₁
```

A film's operator is derived from its composition through `optical` (§4.3) in the surroundings the
film actually provides, including air in the voids of an underbound film. A surface is a boundary
operator, Fresnel, with roughness dividing the reflected power between specular and diffuse. Between
two films the boundary is between the host index of the upper and the top-interface index of the
lower. The index of what lies above the painting (air, water, varnish to come) is an argument of the
look.

**First rung:** two fluxes, bands independent. Operators are arrays whose last two axes are the flux
channels, of size one on this rung, so that more channels (angles, polarisation, coupled bands) is a
change of size and not of interface.

```text
G1  passivity       for admissible substances, power out ≤ power in, every input
G2  monoid          the star product is associative to 1e-12; the empty film is its identity
G3  ledger          incident = reflected + absorbed per film + absorbed in the ground   (§3)
G4  scalar case     a homogeneous film reproduces the Kubelka-Munk closed form
G5  thick limit     an optically thick film shows its own reflectance whatever lies below
G6  matched index   a particle in a host of its own index scatters nothing
G7  order           a glaze over body colour differs from the reverse by the computed commutator
```

A look does not load the world. A coupling from light to matter (cure by light, heating, fading) is
a declared term that reads the light ledger's absorbed dose and nothing else. None exists yet.

## 8. Ports, bodies, clocks

A body couples to the world through ports of power-conjugate pairs:

```text
mechanical    velocity × traction
material      flux of each substance × its chemical potential
```

A body has state and a description, and its receipt books the matter that crossed and the work done.
First bodies: a source, which deposits without contact (a pour, a drop, a spray's cone); a rigid
blade; a porous brush, which gives or takes according to its retention curve; an absorbent pad; a
tape, which is a wall. The environment is a set of reservoirs at fixed potential: humidity, oxygen,
temperature, gravity, illumination. Changing one is an act with a receipt. A reservoir's activity is
positive: at activity zero it would take matter at unbounded gain of free energy. A substance the
atmosphere neither holds nor takes is left out, and its exchange coefficient is zero.

**Clocks.** The painter may scale the world's rates by a field `r(x, t)` that multiplies `Ψ*`.

```text
physical     the environment: humidity, temperature              through the ledgers like anything else
privileged   r ≥ 0: hold (0), slow, hasten, different by region  no law broken; booked as a privilege with ∫ r
impossible   r < 0: reverse                                      breaks `dissipative`; booked to debt[painter]
```

Scaling a convex Ψ* by `r ≥ 0` leaves O1 to O6 intact. A flux between two cells runs at the smaller
of their rates, so nothing crosses into or out of a held cell, and the receipt reports ∫ r over
cells and time. A reverse step is ill-posed, runs with a
stated regularisation, and flags every later receipt.

## 9. Journal and painters

A painting is `Φ_aₙ ∘ … ∘ Φ_a₁ (x₀)`: a pure function of its journal. An entry holds the act, its
parameters, its seeds and its clock, and a digest of its receipt. Replaying a journal reproduces
every array bit for bit (O6). An act that uses randomness takes its seed from the journal.

A painter is a controller under partial observation. It reads the world only through looks and
declared sensors, enforced by tests with the world behind glass. Rehearsal is a fork.

## 10. The lab

`goo.lab` answers what a substance would look like as paint, and how it is known.

```text
drawdown     masstone and a tint ladder, over black and over white, wet and dry, per binder, per illuminant, with gloss
sheet        the phenotype: hiding, tinting strength, drying shift, gloss, infrared response
admissible   the laws passed and failed, each with its shortfall
sensitivity  every result again with size halved and doubled and index perturbed
search       the nearest possible substance to a target; separately, the impossible one that hits it and the laws it breaks
contribute   one file per substance with sources and licence; `validate` checks it
```

The lab makes films by composition, not by flow. Its dried film must be the state the world's drying
converges to, which is tested from outside both (`tests/limits/`).

## 11. Claims

Mechanistic: this behaviour is generated by this declared world. Behavioural: it lies within a
stated distance of a named reference. Forbidden: that goo reproduces the chemistry of any product.
Negative results and disagreements with classic are kept in `docs/NEGATIVE.md`.

## 12. Status

```text
K0  units, evidence, ledger, film                                             done, tested (tests/test_shared.py, tests/test_layering.py)
    substances: description, five laws, maps, library on one scale            built; owner's tests pass (tests/substance/); not yet refereed
    substances into light, with no adapter                                    tested (tests/test_integration.py)
    substances into the world                                                 a 6 x 6 world of water on cellulose closes its ledgers through
                                                                              test-side shims (tests/substance/test_world_protocol.py); wetting left out
K1  docs/DYNAMICS.md; the step with O1 to O8 on the first terms              built; owner's tests pass against analytic laws (tests/world/); not yet refereed
    real substances through the step, with wetting and sorption, no shims     tested on a 6 x 6 world (tests/substance/test_world_protocol.py)
    the painter's cases (pigment in gum water drying, two paints meeting wet, a tilt, a pour over strata, a paste, a
    latex-thickened paint) step on 96 x 96 and 160 x 120 panels (tests/world/test_world_painting.py); a step is all or
    nothing; amounts below one molecule a cell are carried but do not diffuse (DYNAMICS.md, rule R7)
    owed: speed (a quiet step on 1024 x 768 takes 16 to 30 s; moving paint at 300,000 cells is out of reach); a
    gum-bound film never reaches a state `settle()` retires (its viscosity stays finite); pore water does not
    evaporate; cure (needs the reaction's free energy) and dissolution (xfail); surface tension is lagged
K2  light on the first rung with G1 to G7                                    built; owner's tests pass (tests/light/); not yet refereed
K3  ports, the first bodies, journal, replay, clocks                         built without a mechanical port: paints, vessels, pour, flood, drop, drip,
    spray, tape, tilt, air, wait, hold, settle; a journal with bit-identical replay; a command line; owner's tests pass
    (tests/ports/, tests/journal/); not yet refereed. Owed: a brush and a blade (the world has no wall riding on the
    film); a look of a whole 1024 x 768 painting runs out of memory unless taken tile by tile; substances beyond the
    library cannot be named in a session's description.
K4  the lab and observers                                                    built; owner's tests pass (tests/lab/, tests/observe/); not yet refereed
    owed: the observer's fit is unverified and the daylight is a 6504 K blackbody; no gain saturation; no coalescence
    term for latex; known wrong results are listed in docs/LAB.md §8
```

On the vision's ladder: L0 is to be re-earned by K0 and K1 (O1, O5, O6); L1 is K1; L2 is K0 with K1;
L3 is K2; L4 is the data in K0 with the lab's checks. Classic's standing on that ladder does not
carry over.

## 13. Ownership

Each area has one owner, who alone edits its files and its tests. A change to this contract is
proposed to the lead and made by the lead.

```text
substance                  goo/substance/, tests/substance/, docs/SUBSTANCES.md                 first wave
dynamics                   docs/DYNAMICS.md, goo/world/, tests/world/                           first wave
light                      goo/light/, tests/light/, docs/LIGHT.md                              first wave
referee                    tests/contract/, tests/limits/, tests/oracle/, docs/NEGATIVE.md      after the first wave
ports, journal, clocks     goo/ports/, goo/journal/, tests/ports/, tests/journal/, docs/PORTS.md, docs/JOURNAL.md   second wave
lab and observers          goo/lab/, goo/observe/, tests/lab/, tests/observe/, docs/LAB.md, docs/OBSERVE.md   second wave
performance                docs/PERFORMANCE.md, bench/ (prototypes and benchmarks; not part of the package)
lead                       SPEC.md, AGENTS.md, README.md, goo/units.py, goo/evidence.py, goo/ledger.py, goo/film.py,
                           tests/test_shared.py, tests/test_layering.py, integration
```

The referee writes its tests of §3, §6 and §7 from this contract alone, against the code the first
wave delivered, without reading the owners' tests.
