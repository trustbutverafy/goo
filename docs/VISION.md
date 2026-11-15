# goo: the maximal vision

**Status:** a vision, not a description. Section 13 says what is true today.
**Method:** take Ariadne's philosophy and none of its mathematics. Ariadne's world is conservative at
its core, with loss accounted; goo's is dissipative at its core, with work accounted. Waves, holonomy
and topological protection do not transfer. What transfers is the stance.

## 1. Thesis

A paint is a constitutive description. A painting is the history of accounted work done on one
persistent, stratified world. A picture is what a non-loading observer reads from that world.
Nothing else exists in goo: no effect, no medium, no style is anything but a consequence of
descriptions acted on by common laws and read by declared observers.

## 2. What is taken from Ariadne

1. One persistent world; names never select algorithms. "Watercolour" identifies a region of
   behaviour, not a branch of code.
2. Ledgers with receipts and closure residuals. A claim of conservation is a number.
3. Lawful interconnection: bodies couple through power-conjugate ports, and what crosses is booked.
4. Nonlinearity lives in state and potential, never in a memoryless map applied to the picture.
5. Anything active draws on a finite, receipted reservoir.
6. Observation does not load the world unless it declares that it does.
7. Exact limits: removing a mechanism recovers the simpler world exactly.
8. Behaviour is a phenotype; design is search under an admissibility predicate; mechanistic and
   behavioural claims are kept apart; some claims are forbidden.
9. A claim hierarchy, evidence classes, and a ledger of negative results, as part of the
   specification.

## 3. The world

```text
Σ            the space of substances (section 5)
Ω ⊂ R²       the panel
q : Ω → R^S≥0          wet layer: volume per area of each substance present, S ⊂ Σ finite
ξ : Ω → Ξ              internal state: cure, coalescence, alignment, temperature, ...
(q¹,ξ¹) … (qⁿ,ξⁿ)      strata: frozen layers, bottom first; the world's past, stored in space
g, w                   the ground: relief, and what its pores hold
tools                  bodies with their own state (load, configuration)
environment            reservoirs at fixed potential: humidity, oxygen, temperature, illumination
```

The world's state is all of this. Light is a fast part of the world, in equilibrium with the matter
at every instant (section 7). The strata make time visible: goo's memory is sediment.

## 4. Three ledgers

- **Matter.** Volume of each substance, including what it turns into. Exists today, closes to 1e-14.
- **Energy.** Free energy stored, work done by tools, dissipation, exchange with the environment.
  Does not exist today. It is what makes "the painter's work" and "an impossible substance's debt"
  meaningful.
- **Light.** For every band: reflected + absorbed in each layer + absorbed in the ground = incident.
  A look comes with a receipt of where the light went.

Every operation returns a receipt against all three, with a closure residual.

## 5. Substances

A substance σ ∈ Σ is described where physics is simple:

```text
ε_σ(ω)         dielectric function                      → how it meets light
μ_σ            morphology (size, shape, structure)      → particles
f_σ            its contribution to the free energy       → what it stores
Ψ*_σ           its contribution to the dissipation       → how it resists
N_σ            its reactions, with stoichiometry         → what it becomes
```

Everything the engine reads is derived from these by stated maps, each with a domain of validity:
scattering and absorption given the surroundings (Mie; independent scattering), viscosity and yield
given loading (Einstein, Krieger-Dougherty), diffusion and settling given size (Stokes-Einstein,
Stokes), film formation given composition. Outside a map's domain the value is marked `assumed`, and
where only bounds are known (an effective property of an unknown microstructure) **the engine carries
the bounds**, not a pretended value.

**Admissibility.** A substance is *possible* if:

```text
optics       ε_σ is a Herglotz function of frequency        (causal and passive in one condition)
storage      f_σ keeps the free energy bounded below         (stability)
dissipation  Ψ*_σ is convex, non-negative, zero at zero      (the second law, for every process)
coupling     linear cross-effects are symmetric              (Onsager reciprocity)
reaction     N_σ conserves volume by stoichiometry
```

It is *real* if in addition its description is tied to a measured material with provenance. It is
*impossible* if it fails a named condition. Impossible substances are permitted: a failure that
yields power (optical gain, negative friction) is supplied from a finite reservoir variable carried
with the substance, and the energy ledger shows the draw. A failure that cannot be paid for (a
dielectric function that is not the boundary value of any analytic function) is carried as a flag on
every picture it appears in.

The pair (f, Ψ*) is the definition of a material in the theory of generalised standard materials;
the Herglotz condition is the definition of a passive linear response. goo's substance space is the
product of the two, with morphology tying them together.

## 6. Dynamics

Matter relaxes. The law is one statement:

```text
F[q, ξ] = ∫_Ω  ½ G (h + z_b)²  +  ½ γ |∇(h + z_b)|²        levelling: gravity, capillarity
             +  f_mix(q)                                    mixing; non-ideal terms give immiscibility
             +  f_wet(q; ground)                            wetting and pinning
             +  f_chem(q, ξ)                                cure, coalescence
             −  Σ_σ μ_σ^env q_σ                             open to the atmosphere
             dx

∂_t x  =  ∂Ψ*( x ; −DF(x) )  +  ports                       x = (q, ξ)
```

with h = Σ_σ q_σ, z_b the ground plus the strata, and Ψ* the sum of the substances' dissipation
potentials evaluated on the thermodynamic forces. Each existing law is a term:

- leveling: force −∇(δF/δh); Newtonian mobility h³/3η gives a quadratic Ψ*; a yield stress gives a Ψ*
  that is flat near zero force, which is what "holds a brush mark" means;
- colour diffusion: the mixing term with a mobility from Stokes-Einstein;
- evaporation, absorption, cure: exchange with a reservoir, rate = ∂Ψ*/∂(affinity).

Then for every admissible substance set and every process:

```text
dF/dt  =  − ⟨ζ, ∂Ψ*(ζ)⟩  +  P_ports   ≤   P_ports ,     ζ = −DF(x)
F(t₁) − F(t₀)  =  W_tools  −  D ,       D ≥ 0                       (the energy ledger)
```

**Obligation on the code.** The discrete scheme must satisfy the discrete form of these as identities
or inequalities of the arithmetic, not as observations: conservation and positivity already do; the
dissipation inequality is to be had by writing each step as a minimising movement or an equivalent
structure-preserving update. Locality (windowed equals whole), determinism (a fork continues
identically) and replay are theorems of the same kind.

**Obligation on the laws.** Any present law that cannot be written as a term of F and Ψ* is
phenomenology and is either derived or marked. Candidates: the wet-edge evaporation boost (the real
cause is a diffusion-limited flux that diverges at a contact line), the pinning threshold, the rule
that species move in the donor cell's proportions.

**Change of rank.** `fix()` turns part of the wet layer into a stratum: degrees of freedom are
retired. What happens to their stored energy, their cure state and their residual water is a declared
policy with a receipt, as Ariadne declares what happens when a mode is retired.

## 7. Light

Light is solved, not simulated: stationary transport through the stack, per band.

```text
a layer ℓ      →  a passive scattering operator S_ℓ on the fluxes entering it from above and below
passivity      →  energy out ≤ energy in, for every input
composition    →  the star product:   R₁₂ = R₁ + T₁ R₂ T₁′ (1 − R₁′ R₂)⁻¹ ,   T₁₂ = T₂ (1 − R₁′ R₂)⁻¹ T₁
```

The star product is associative and not commutative. Passive operators are closed under it, and they
form a monoid whose identity is the empty layer. A painting's optics is an ordered product in that
monoid; Kubelka's layer formula is the scalar case. A glaze over body colour differs from body colour
over a glaze, and the difference is a commutator one can compute.

The same algebra holds at every rung of resolution, and each rung reduces to the one below exactly:

```text
two fluxes  ⊂  many fluxes (angles)  ⊂  polarised  ⊂  coherent amplitudes (interference pigments)
bands independent  ⊂  bands coupled by a kernel that never raises photon energy (fluorescence)
```

For intensities the operators are sub-stochastic kernels; for amplitudes, contractions. Surfaces are
boundary operators (Fresnel), with roughness dividing the reflected energy between the specular and
the diffuse channels. Gloss, sheen, colour and hiding are readings of one operator.

A layer's operator is derived from its composition: the substances' response functions evaluated in
the surroundings the film actually provides. That passive particles in a passive host give a passive
layer is a theorem to be proved for the maps used.

**The light ledger** gives absorbed energy per layer and band. By default light is a non-loading
observer. Declared couplings use exactly that ledger as their source: curing by light, heating,
fading in proportion to absorbed dose. A painting ages by what it has absorbed.

## 8. Tools and gestures

A tool is a body with state, coupled to the world through ports of power-conjugate pairs:

```text
mechanical     velocity  ×  traction
material       flux of each substance  ×  its chemical potential
```

A brush is a deformable porous body: it holds paint by capillarity. A loaded brush gives and a thirsty
one lifts because of where the potential is higher; today's "target thickness" rule becomes a
consequence of the brush's retention curve. A knife is a rigid boundary; a sponge a body of high
capillarity; tape a wall. Tools are points in their own constitutive space, with real, possible and
impossible members, under the same admissibility.

A gesture is a path in the tool's configuration space. Its receipt includes the painter's work,
∫ traction · velocity dt. A technique is a named family of gestures with a contract over observables:
what must be seen before (a damp sheen here), what will be seen after, what it costs.

Because the world is deterministic,

```text
painting  =  Φ_aₙ ∘ … ∘ Φ_a₁ (x₀)
```

a painting is a pure function of its journal. The journal is to goo what a score is to Ariadne, and a
painting can be checked, replayed and cited like a proof.

## 9. Observers and painters

Four domains, as now: the World (which includes its light), Sensors (an eye, a camera, a lidar, each
a measurement model with its own noise and memory), Presentation, and Painters.

A painter is a controller under partial observation: the state is hidden, the observations are looks,
the actions are gestures, the dynamics are the laws. The rule that a painter sees only what optics
show is the information constraint of that problem, enforced mechanically. Rehearsal on a fork is
planning with an exact model: imagination made precise.

The same structure read backwards is conservation science: from what sensors return, infer the
strata. goo owes **identifiability statements**: which hidden states are determined by which family
of looks (bands, lights, angles), and which are not, however good the sensor.

## 10. Scales and reductions

Every simplification is a reduction map with a small parameter, a stated error, and an exact limit:

```text
depth-averaged film  ⊂  layered film  ⊂  free-surface flow in depth        aspect ratio
stroke  ≪  levelling  ≪  drying  ≪  curing  ≪  ageing                       time-scale ratios
two-flux light  ⊂  …  ⊂  full transport                                     section 7
independent scattering  ⊂  dependent scattering                             pigment concentration
```

A phenomenon is assigned to the lowest rung that can hold it: a skin forming from the top needs
depth; craquelure needs elastic energy and a fracture term in F and Ψ*; an overhang needs the third
dimension. Adding a phenomenon means adding a term to F, to Ψ*, or a rung, for every substance.

## 11. Phenotype, design, claims

A phenotype is what tests in the world return: hiding power, tinting strength, drying shift, gloss,
levelling, open time, lifting, granulation, infrared response. Media are regions of phenotype space.
Three inverse problems share one statement, *find the cause of an observed or wanted behaviour, among
admissible causes*: formulate a paint; find gestures for a picture; infer a painting's strata.

Claims are kept apart. Mechanistic: this behaviour is generated by this declared world. Behavioural:
it lies within a stated distance of a named reference. Forbidden: that goo reproduces the chemistry of
any actual product. Every number carries an evidence class: proved, exact in arithmetic, computed from
permissive data, checked against a restricted reference, assumed, guessed. Negative results are kept.

## 12. Formal obligations

To be proved, the algebraic ones in a proof assistant:

1. Each species is conserved by the discrete update; volumes stay non-negative.
2. The discrete dissipation inequality, for convex Ψ*.
3. The star product is associative; passivity is closed under it; the light ledger closes.
4. The oscillator family lies in the Herglotz class; Herglotz implies passive and causal.
5. Each reduction map commutes with the dynamics in its limit (classic optics from v2; whole panel
   from windowed; original from fork).
6. For the admissible class, the second law holds for all processes (Coleman-Noll, for goo).

## 13. The ladder, and where goo stands

```text
L0  matter is conserved exactly; the world is deterministic; observers do not load   holds, tested
L1  thermodynamic: F and Ψ* exist for the laws as coded; energy ledger with receipts not begun
L2  constitutive: every law reads substances; today's paints are points in Σ         kernel in progress
L3  optical closure: passive layer algebra, light ledger, spectral, one surface      in progress
L4  grounded: real points computed from permissive data, checked against references  in progress
L5  predictive: phenotypes of real paint-outs reproduced within stated error         nothing yet
L6  inverse: formulation, gesture and stratigraphy inference with identifiability    nothing yet
```

L1 is the rung everything else stands on, and it is the one with no work behind it.

## 14. Risks

- The laws as coded may not be a gradient flow. Finding out is the first task, on paper, before code.
- Rigour in the structure does not make the data good; L4 and L5 can fail while L1 to L3 hold.
- Maximal scope invites smuggling under pressure. The anti-smuggling rule is the defence, and it costs
  features.
- A thin film in two and a half dimensions is the wrong home for thick paint, and saying so is part of
  the claim.
