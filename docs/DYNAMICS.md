# goo dynamics: the first rung

Owner: dynamics. This is the derivation SPEC §6 asks for: F and Ψ* for the first terms of a
depth-averaged film, the discrete step, and the argument that the step satisfies O1 to O4 as
identities or inequalities of its arithmetic. Code: `goo/world/`. Tests: `tests/world/`.
Everything is SI. A *reduction* is written **R** with its small parameter; a dropped law is written
**Dropped** with the reason.

## 1. State

On cells of side `dx` (index c, area dx²), faces f between neighbours L (left/top) and R:

```text
q[i,c]   m        volume per area of substance i in the wet film          h = Σ_i q_i,  m = Σ_i ρ_i q_i  (kg/m²)
z_b[c]   m        ground relief + the strata's thicknesses                 s = z_b + h  (the free surface)
w[i,c]   m        what the ground's pores hold;  θ = Σ_i w_i / C,  C = porosity × depth
env               T (K); gravity g = (g_x, g_y, g_n), g_n ≥ 0 into the panel; reservoir potentials μ°_i (J/m³)
```

φ_i = q_i/h, ρ̄ = m/h. A cell is *wet* if h > 0 and *bare* if h = 0. Positions x_c are absolute
panel coordinates ((index + ½) dx), never window coordinates.

**R1 (thin film).** Depth-averaged lubrication; small parameter the surface slope ε = |∇s|. Errors are
O(ε²) (the capillary area √(1+ε²) − 1 = ε²/2 + O(ε⁴); the lubrication pressure is hydrostatic to
O(ε²)). Every receipt reports max ε over faces touching wet cells and flags `slope` above the world's
bound (default 0.1, a reporting threshold, not physics).

**R2 (uniform through the depth).** Composition is uniform in z. Convection carries substances in the
proportions of the cell they leave. Small parameter: the Péclet number of depth diffusion inverted,
h²/(D t_flow). Settling through the depth, skinning, and stratification need the next rung.

**R3 (volumes add).** h = Σ q_i and each substance keeps its density (SPEC §3, `assumed`).

## 2. Free energy F (J)

F is the grand potential of the live world against the environment's reservoirs:

```text
F = F_grav + F_cap + F_wet + F_mix + F_exch + F_pore

F_grav = Σ_c dx² [ g_n m (z_b + h/2)  −  m (g_x x_c + g_y y_c) ]
F_cap  = Σ_f ½ γ_f [ (s_R − s_L)² − (z_R − z_L)² ]                          (½ ∫ γ (|∇s|² − |∇z_b|²))
F_wet  = Σ_c dx² (−S_c) (1 − exp(−h/ℓ_c))                                    S ≤ 0 spreading coefficient, ℓ > 0
F_mix  = Σ_c dx² f(q_c; T)                                                    from the protocol, J/m²
F_exch = −Σ_c dx² Σ_i μ°_i (q_i + w_i)                                       film and pores, against the reservoir
F_pore = Σ_c dx² [ −Σ_i P_i w_i + P_b C B(θ) ],   B(θ) = −θ − ln(1 − θ)     P_b = max_i P_i
```

- *Gravity*: a column of mass m whose centre is at z_b + h/2 under the normal component, plus the
  potential of the in-plane component. Exact for R2.
- *Capillarity*: the liquid surface's excess area, small slope (R1). The baseline −(Δz)² makes F_cap
  exactly zero on bare ground, so bare cells hold no energy and are rest states (O4, O5).
- *Wetting*: covering bare ground costs −S = γ(1 − cos θ_c) per area once the film is thicker than
  ℓ. Concave in h. With S = 0 it is absent exactly. Its increments are computed as
  sign(δh)(−S) e^{−min(h,h')/ℓ}(1 − e^{−|δh|/ℓ}), finite for a molecular ℓ against micrometre films.
- *Mixing*: f is the protocol's: ideal mixing plus interaction. It must be homogeneous of degree 1 in
  q (an intensive free energy per volume times h), as every mixing free energy is. The protocol gives
  f with its potentials μ_i = ∂f/∂q_i and Hessian, split into a convex part (implicit) and a concave
  part (explicit, may be zero).
- *Exchange*: a reservoir at fixed potential μ°_i (the atmosphere: μ° = (k_B T/v) ln a for a volatile
  at activity a > 0). The `exchange` map turns goo.film.Environment's activities into μ° (SPEC §4.3); a
  substance the environment does not name has no reservoir and no term. Grand potential form, as VISION
  §6 writes it. Every amount that can reach the reservoir is counted, the pores' as well as the film's,
  so sorption is a transfer inside F with its own D_mech (counting only the film would book nearly all
  of −μ° q of the water that soaks in to D_num). Strata are not counted: retired matter's free energy
  is in `frozen`. Changing μ° is work by a port.
- *Sorption*: liquid in the pores lowers F by its suction P_i (Laplace, 2γ cos θ / r, from the
  protocol) and a convex barrier keeps θ < 1. The barrier's form is `assumed`.

*Lagged coefficients.* γ_f, S, ℓ, P are evaluated by the protocol at the start of each step and held
fixed through it, so F is a fixed function during the step. If a later evaluation changes them (a
coefficient that depends on composition), the change of F at fixed state is booked as
`work[constitutive]` and flagged `out_of_domain` ("lagged coefficient"). For coefficients that do not
depend on composition this entry is exactly zero and is not posted. **Dropped**: Marangoni stress (a
gradient of γ along the surface); it needs γ(φ) inside F and the coupled step, the next rung.

## 3. Dissipation potentials Ψ* (W/m²) and the fluxes they give

Forces are minus derivatives of F; each flux is ∂Ψ*/∂(force). A clock rate r ≥ 0 (SPEC §8)
multiplies every Ψ*; on faces r_f = min(r_L, r_R).

**Flow.** Total volume flux U_f (m²/s) across a face. With R2 the force conjugate to U is the
mixture's pressure gradient G_f = −Σ_i φ_i^d (μ_i,R − μ_i,L)/dx, φ^d the donor's composition.

```text
Ψ*_flow(G) = ½ r M (|G| − Y)₊²       M = h³/(3η)   Y = τ_y / h
U = r M (|G| − Y)₊ sign G
```

Newtonian for τ_y = 0; flat for |G| ≤ Y, which is a yield stress: the wall shear stress h|G| must exceed
τ_y. **R4**: the exact depth-integrated Bingham flux (cubic in Y/h) is replaced by this convex law with
the same threshold and the same Newtonian limit (`assumed`). Mobility on a face between wet cells is
the mean of the two cells' h³/(3η); between a wet and a bare cell it is the wet cell's and the face is a
one-way valve (Ψ* = ½ r M (G − Y)₊² in the direction out of the wet cell only: a bare cell has nothing
to give). Between bare cells M = 0.

**Diffusion.** Relative volume fluxes j_i with Σ_i j_i = 0, driven by the full potentials:

```text
j_i = r Σ_k L_ik X_k,   X_k = −(μ_k,R − μ_k,L)/dx,   L = diag(ω) − ω ωᵀ / Σω,   ω_i = h_f Λ_i φ_i^up
Ψ*_diff = ½ Xᵀ L X  ≥ 0
```

Λ_i = D_i v_i/(k_B T) (m²/(Pa s)) is the protocol's `mobility`: by Stokes-Einstein Λ = 2a²/(9η), the
Stokes mobility. With ideal mixing (μ = (k_B T/v) ln φ + ρ ψ) this gives Fick's law with D_i *and*
lateral Stokes drift of a substance relative to the mixture under the in-plane gravity, so the lateral
part of settling comes from the same Ψ*. For any ω ≥ 0, L is symmetric positive semi-definite with
L·1 = 0 (XᵀLX = (1/2Σω) Σ_ik ω_i ω_k (X_i − X_k)²): reciprocity and the volume constraint. φ^up is the
volume fraction of substance i in the cell it leaves on that face (upwind), *at the new state*: so what
leaves a cell is proportional to what it holds, which is what keeps amounts positive without
cancellation (a frozen face mean of φ does not: a draining cell's gross fluxes would not shrink with its
content, and the flux-form update cancels to below rounding). Any ω ≥ 0 at any state keeps the energy
argument of §5, which needs only J = L X with L ⪰ 0.

**Exchange.** e_i (m/s) to the reservoir: Ψ* = ½ r k_i (μ_i − μ°_i)², e = r k (μ − μ°), k from the
protocol (`exchange`), 0 for non-volatiles. **Dropped**: condensation or absorption into a cell that
holds none of the substance (nucleation); exchange acts on substances present.

**Sorption.** a_i (m/s) into the pores: Ψ* = ½ r κ_i (μ_i − μ^p_i)₊², μ^p_i = ∂F_pore/∂w_i, κ from the
protocol. One-way. **Dropped**: desorption back into the film (it needs the pores' own mixing).

**Dropped**: the wet-edge evaporation boost (a diffusion-limited vapour field, not a term of these);
settling through the depth (R2); cure and dissolution are second priority and are xfail tests.

## 4. The discrete step

`step(dt)` is a Lie splitting of three substeps; each is a structure-preserving update with its own
energy inequality, so the sum has one. In each, the *implicit* force is the derivative of the convex
part of F at the new state and the *explicit* force the derivative of the concave part at the old one
(Eyre's convex splitting); mobilities are frozen at the old state.

**4.1 Flow** (cells' (h, m) move; substances in donor proportions φⁿ). Write the mechanical energy
E(h, m) = F_grav + F_cap + F_wet. Gravity's cross term is not convex when densities differ; split it
with a constant a = √(ρ_min ρ_max) over the world's substances:

```text
g_n m h/2 = (g_n/8a)(m + a h)²  −  (g_n/8a)(m − a h)²        convex − convex
```

Implicit: g_n m z_b, (g_n/8a)(m+ah)², F_cap (γ lagged), the in-plane term (linear). Explicit: −(g_n/8a)(m−ah)²
and F_wet (both concave). With one density, m = a h and the explicit gravity part is identically 0.
The unknown is δy = (δh, δm) on the *active cells* (cells touching a face with M > 0):

```text
δh_c = −(dt/dx) Σ_f n_cf U_f          δm_c = −(dt/dx) Σ_f n_cf ρ̄^d_f U_f            (n = +1 if U leaves c)
U_f  = r_f M_f (G_f − c_f Y_f),  c_f ∈ {+1, −1, 0}          G = G⁰ + (1/dx) Cᵀ H δy
⇒  (I + (dt/dx²) C r M Cᵀ H) δy = −(dt/dx) C r M (G⁰ − c Y)
```

H is the Hessian of the implicit part (per cell [[g_n a/4, g_n/4],[g_n/4, g_n/(4a)]] plus the capillary
Laplacian), PSD. *Orientation* (**R5**): each face may flow in one direction this step, the sign of a two-way Newtonian
predictor solved once with the old force's donors (and only out of a wet cell). This is a frozen
coefficient like the mobility (Ψ* of the face is the one-way valve in that direction), and it removes
the direction-dependence of the donor's density from the problem, which otherwise makes the step
non-convex across U = 0 when neighbouring columns have different densities (an active-set iteration
on it cycles; observed). A face whose flow would reverse within a step is stuck for that step.

*The convex problem.* With orientations fixed, the step minimises over u_f = o_f U_f ≥ 0 the convex
function Φ(u) = E(yⁿ − (dt/dx) C u) + dt dx² Σ_f (u_f²/(2 r M_f) + Y_f u_f) subject to each cell's outflow
(dt/dx) Σ_out u_f ≤ hⁿ_c (it cannot give what it does not hold). It is solved by a primal active-set
method: iterates stay feasible; each iteration solves the equality-constrained problem on the working
set (faces stuck; cells *saturated*, giving exactly hⁿ, with a multiplier λ_c, a reaction force against
their outflow) as one sparse LU of the block system; infeasible results are stepped back to the
boundary and the blocking face or cell joins the working set; when feasible, faces and cells with
violated multipliers are released (all at once, or one at a time once a working set repeats, which is
the textbook anti-cycling rule). Each iteration lowers Φ, so it terminates. A stuck face whose force
exceeds its threshold by less than 1e-12 of the largest force is at the threshold to rounding and is
not released. Where every donor in the solve has one density ρ, δm = ρ δh exactly and the system is solved in δh
alone (H = g_n(ρ + a)²/(4a) + the capillary Laplacian): the same equations, half the unknowns. An LU
is reused when a later solve has the same matrix bit for bit. Before the primal method, the working
set takes primal-dual moves (every violator joins at once), which settle a drying front in a few
iterations; only if a working set repeats does it switch to the primal method above.
Then each substance moves in flux form, q_i ← (1 − α_c) q_i + inflow_i with
α_c = outflow_c/hⁿ_c ≤ 1 (exactly 1 for a saturated cell).

*Quiet faces.* A face can move this step only if, at the old state, its force exceeds its threshold in a direction
it may flow (o G⁰_f > Y_f for o = +1 with donor L, or o = −1 with donor R; for Y = 0, o G⁰ > 0). If no face can,
the step returns the state bit for bit and nothing is assembled. Proof: for any orientation o, Φ is convex on the
feasible cone u ≥ 0 (with the outflow bounds), u = 0 is feasible, and at u = 0 the state is yⁿ, so the force is G⁰
and ∂Φ/∂u_f(0) = dt dx² (Y_f − o_f G⁰_f) ≥ 0 for every face. These are the KKT conditions of the cone at u = 0 (every
multiplier of u_f ≥ 0 is ∂Φ/∂u_f ≥ 0, every outflow bound is slack with multiplier 0), so u = 0 minimises Φ; the
minimiser of a convex function on a convex set is a minimiser, and Φ's quadratic part (dt dx² Σ u²/(2rM), M > 0 on
every face that could carry flow) makes it strictly convex in u, so u = 0 is *the* solution: what the full solve
would return, without solving. A paste below yield, a held clock, a level film: nothing is factorised. In a mixed
panel the orientation predictor is solved on the faces that can move only; a face below its threshold keeps the
old force's direction (R5: a frozen choice; any orientation keeps §5), and the solves of the active set are on the
cells of its free faces, so a stiff paste costs nothing next to a moving wash. A face with a threshold is *at* it to
rounding when its excess is below 1e-12 of max(the largest force, the size of the potentials it is a difference of,
(|A_L| + |A_R| + ρ^d(|B_L| + |B_R|))/dx): its force is not known better than that. The same bound decides release in
the active set, and a face released and then blocked again before a feasible point is reached, twice (or once with
no step to rounding, α ≤ 1e-12), stays stuck for the rest of the step: the minimiser over the smaller cone (u_f = 0)
is still a constrained minimiser of the same convex Φ, so §5 holds, and degenerate cycles of release and block end
(observed: 400 iterations on a water pour with an excess of 4e-12; 400 on a tapered flood on paper, α = 1e-19 and
8e-4 alternating).

*The constraint's share.* A saturated cell's multiplier λ_c ≥ 0 is the reaction against its outflow; its work
dt dx² Σ_f λ_donor |U_f| is inside D_num (§5, "Constraints") and is reported apart (`D_num constraint`), together
with the explicit wetting term's Bregman divergence −(ΔF_wet − ⟨∂F_wet(hⁿ), δh⟩) ≥ 0. Both are O(1) in dt where a
cell empties or crosses ℓ (a dewetting contact line moves one cell, whatever the step), so they are not a measure
of the time step's error and `advance` does not read them as one (§9a).

*Acceptance*: any substep that fails (the active set or a Newton iteration does not settle, an amount
would be negative) is redone as two half steps, deterministically and recursively, to a depth of 24, then
an error. The capillary stiffness is implicit and is not limited; nothing is limited by a Courant number.

**4.2 Diffusion** (h fixed, composition moves). Unknowns q_i on the active (cell, substance) pairs.
Implicit Euler in μ (convex mixing at the new state, concave at the old, gravity's ρ_i ψ exact because it
is linear in q at fixed h), with the upwind mobility at the new state (§3), solved by Newton in ln q
(the Jacobian includes the mobility's derivative; the step is du = dq/q from the same factors, a rise
capped at e² per iteration; a backtracking line search on the residual), with the factors kept while
the residual falls tenfold per iteration (chord), on the active pairs in canonical order, until the
upwind directions repeat, the update is below 1e-12 of each amount (plus 1e-14 of the cell's
thickness) and the flux-form update is non-negative; then applied in flux form. An upwind direction
changes only where the relative force exceeds the rounding of the potentials (1e-12 of their size);
where it does not, the flux is rounding too and either donor is an admissible mobility. An entering
amount whose iterate falls below e⁻³⁰ of the cell gets no inflow and is dropped exactly, its faces
carrying none of it (the entry rule, R6, applied during the solve).
Common parts of μ (capillary, wetting, the h-derivatives of gravity) cancel exactly under L·1 = 0.

*Sub-molecular amounts* (**R7**). A substance holding less than one of its molecules (or particles) in a cell,
0 < q_i < v_i/dx² (v_i its molecular volume; 0 for a phase without one, which never diffuses), is carried exactly
but does not diffuse this step: every face touching that cell has ω_i = 0. This is a statement about arithmetic,
not physics: a continuum flux of less than one molecule has no meaning, and its Newton iterate in ln q can fall
by hundreds of decades (observed: 1e-155 m at a thin front, where the mixing Hessian's c/q overflowed and the
factorisation was singular for every dt). O1 and O2 are untouched (the pair has no flux; its amount is carried
bit for bit), and §5 holds because it holds for any ω ≥ 0. Above the floor nothing changes (tested: a scene whose
amounts all exceed it gives the same arrays with the floor set to 0). Two further rules keep the iteration finite
without changing its answer: an iterate below the floor enters the Jacobian (only the path) at the floor, where the
Hessian is finite, and a pair's update is converged when below 1e-12 of its amount, 1e-14 of its cell's thickness
or 1e-12 of the gross amount the fluxes carry through it (the flux-form update is a sum of those terms and is not
known better; a 1e-21 m cell next to a 5e-6 m one could otherwise never converge). A Jacobian that is not finite,
and a factorisation that fails, reject the substep (it halves); they never escape as errors.

*The explicit potential of an entering substance.* A substance absent from a cell at the substep's start and allowed to
enter it has its μ_x masked to 0 together with the convex part (whose μ is −∞ there). With the substance maps' particle
thermodynamics (docs/SUBSTANCES.md §3a) a particle has no concave potential, so for an entering pigment the mask is
exact and D_num's time part is no longer negative where one paint's pigment enters the other's
(tests/world/test_world_painting.py, two paints meeting, no longer xfail). For an entering *liquid* with a concave
interaction part the mask is still not F's derivative: an open defect where two liquids that interact meet.

*Entry* (**R6**): a substance enters a neighbouring cell that lacks it only if, at e⁻³⁰ of the donor's
fraction, the relative force at the old state still drives it in. Where it would not, its equilibrium
share in that cell is below e⁻³⁰ of the donor's (uphill against strong drift, or against a solvent's
counter-flux), it is not representable against the fluxes that balance it, and that face carries none
of it this step. A numerical threshold on representability, not a physical constant.


*Particles in the Onsager matrix* (an open defect, found with the substance maps' particle thermodynamics).
L = diag(ω) − ωωᵀ/Σω gives each substance a friction against the ω-weighted mean motion. For a particle p in one
liquid w the pair's conductance is ω_p ω_w/(ω_p + ω_w), the smaller of the two: water's molecular mobility
(Λ_w = D_w v_w/k_B T ≈ 1e-17 m²/(Pa s)) and not the particle's Stokes mobility (d²/18η ≈ 5e-12 for 0.3 um rutile), so
the world interdiffuses particles and water at about 1e-16 m²/s where Stokes-Einstein (times the maps' thermodynamic
factor) gives about 1e-12. Water's self-diffusion is friction of water against water; a particle's motion through the
liquid is resisted by Stokes drag. The construction that says so is a network of pair conductances,
L = Σ_{i<k} c_ik (e_i − e_k)(e_i − e_k)ᵀ (symmetric, positive semi-definite and L·1 = 0 for any c ≥ 0, so §5 holds):
liquid with liquid c_lk = ω_l ω_k / W_L (W_L = Σ_liquids ω, the present rule among liquids), particle with liquid
c_pl = ω_p ω_l / W_L (the particle drifts through the liquid at its own mobility: j_p = ω_p (X_p − X̄_L)), particle with
particle 0 (no direct exchange). Without particles it is the present L. Not yet built: it needs the world to know which
substances are particles (a description field, morphology, read through the protocol) and a new Jacobian for the
Newton; until then particles in water diffuse too slowly, which is admissible (any L ⪰ 0 is) but wrong in rate.

**4.2a The ray** (how diffusion is solved first; the Newton above is its fallback). Diffusion at these resolutions is
not stiff (D dt/dx² ≈ 1e-4 to 0.06 for gum and water at dt of seconds and dx = 0.25 mm), so one explicit Onsager flux
does what the Newton does, and the energy inequality is kept by a search along that flux's ray instead of by
implicitness. At the old state, on the faces and substances of the solve (after R6 and R7, which act on ω as before):
X_i = −(μ_i,R − μ_i,L)/dx with the full potential (mixing, both parts, and ρ_i ψ); ω_i = κ_i φ_i(donor), with donors
that agree with the relative force, iterated at most four times from the centred guess (the Newton's rule, rounding
included); j = ω (X − X̄), with a relative force below the rounding of the potentials it is a difference of
(1e-12 of their size, the Newton's own rule) carrying none (on a face where one substance moves alone, X − X̄ is such
rounding: L is exactly 0 there); δ = −(dt/dx) div j. Along q(t) = qⁿ + t δ,

```text
Φ(t) = F_c(q(t)) + t dx² ⟨μ_x(qⁿ) + ρψ, δ⟩ + ½ t² D1,       D1 = dt dx² Σ_f Σ_i ω_i (X_i − X̄)²  ≥ 0
```

is convex (F_c convex, the rest linear and quadratic). Φ'(0) = −dt dx² Σ_f Σ_i j_i X_i (summation by parts, an
identity; −D1 when no force is masked), and it is computed so, face by face: summed over the cells, ⟨μ, δ⟩ cancels
potentials of 1e9 J/m³ and its rounding exceeded D1 on a megapixel paste (the search then returned t = 0). Φ'(t) − Φ'(0)
is computed as the sum of the per-cell differences μ_c(q(t)) − μ_c(qⁿ) against δ, each known to its own rounding. The
sign of Φ' is then decided on numbers of the size of the step's energy, not of F's. The step is the largest t in
[0, T], T = min(1, what keeps every amount positive), at which Φ'(t) ≤ 0 was *computed*: T itself when Φ'(T) ≤ 0,
else the lower end of a safeguarded Newton-bisection bracket on Φ' (each evaluation one mixing call). The flux applied
is t j and the force that drives it through the same L is t (X − X̄), so D_mech = t² D1, a sum of non-negative terms.
Then, with q' = q(t) and G = −F_x convex,

```text
−ΔF − D_mech ≥ −t ⟨∇F_c(q') + ∇F_x(qⁿ) + ρψ, δ⟩ dx² − t² D1 + Bregman_G(q', qⁿ)  = −t Φ'(t) + Bregman_G  ≥ 0,
```

(the argument uses only Φ'(t) ≤ 0 and D_mech = t² D1 with the D1 of Φ, whatever flux δ came from), the first
inequality being F_c(qⁿ) ≥ F_c(q') + ⟨∇F_c(q'), qⁿ − q'⟩ (convexity) and concavity of F_x, the last because
t ≥ 0 and Φ'(t) ≤ 0 as computed. So D_num ≥ 0 holds **by construction** for any t the search returns, however inexact,
not to a tolerance: an inexact search does less than the exact one, never something inadmissible. When t is interior,
Φ'(t) = 0 to rounding and D_num is the Bregman divergence of the step (O(dt²)): the time error, which `advance` reads.
At t = T = 1 with Φ'(1) < 0 the explicit step is a little short of the ray's optimum; −Φ'(1) is O(λ dt) D1, of the
order of the Bregman part, and is also time error. Positivity: what leaves a cell is proportional to what it holds
(upwind ω), and t ≤ (1 − 1e-9) of the bound that keeps every amount positive. O1: flux form. O4: zero forces give
D1 = 0 and δ = 0 exactly, and the state is returned bit for bit. O5: element-wise arithmetic plus sums over the
compacted active set in canonical order, the same in a window and on the panel.

The Newton is used instead (counted as `diffusion fallbacks` in the receipt) when a substance enters a cell that lacks
it (its μ is −∞ there and an explicit flux has no meaning), when a potential is not finite, or when t < 0.9 (the
explicit flux is too long for this dt: stiff gradients). The receipt reports `diffusion rays` and `max diffusion ray
shortfall` (1 − t). Measured (docs/PERFORMANCE.md §2.4): against 32 implicit substeps the ray's error is 3.0 % of the
substep's change against implicit Euler's 3.6 % at dt = 2 s, equal at dt = 30 s, worse at dt = 120 s (t* = 0.25, where
the fallback takes over). The open defect above (an entering substance's explicit potential) lies in the Newton path,
which the ray does not take where a substance enters, so the ray neither cures nor explains it.

**4.3 Exchange and sorption** (local). Per cell, unknowns e_i, a_i; Newton per cell, each cell
stopping on its own tolerance. The film's potential at the new state uses: convex mixing implicit,
concave mixing explicit, gravity split as in 4.1, and capillarity *stabilised*: μ_cap = (1/dx²)[∂F_cap/∂h_c(hⁿ) + σ_c δh_c]
with σ_c = Σ_{f∋c} γ_f, which is the convex splitting F_cap = (σ/2)|h|² − ((σ/2)|h|² − F_cap) and
needs σ ≥ ½ λ(γ L_graph) (Gershgorin). A single-substance cell whose implicit solution would have h < 0
is drained exactly: h' = 0, with e and a split so that both see one force ξ; this is the KKT point of
the constraint h ≥ 0, and ξ ≤ μ(h → 0⁺) is checked. The same holds for a cell of several substances in which one
substance i alone exchanges or is sorbed: with the others fixed its residual q_i' − q_i + e_i(q_i') + a_i(q_i') is
increasing in q_i', and when it is ≥ 0 at q_i' = 1e-12 q_i (the root lies within that of zero) the cell gives all of i,
exactly, split between e and a as above, the others untouched bit for bit, ξ ≤ μ_i(1e-12 q_i) checked. This is a
particle film's last water: a particle holds no water (its potential is bounded below by the menisci's −Π_e), so the
Newton, limited to 0.9 of what is held per iteration, only crept toward zero and the step failed. Candidates are found
with one map call: μ_i increases with q_i, so what the step could take at the old amount bounds it from above.

## 5. Ledger and the argument for O1 to O4

**D_mech** of each mechanism is computed from the fluxes applied and the forces that drove them:

```text
viscous     dt dx² Σ_f |U_f| (o_f G_f − Y_f − λ_d)₊   plastic   dt dx² Σ_f Y_f |U_f|
diffusion   dt dx² Σ_f Xᵀ r L X                  exchange  dx² Σ e²/(dt r k)       sorption  dx² Σ a²/(dt r κ)
```

each a sum of non-negative terms, so D_mech ≥ 0 in floating point, not only in exact arithmetic. The viscous term
is written as flux times force, which equals U²/(rM) at the solution: U²/M overflowed (2.5e88 J) where M ~ 0 in a
cell holding almost nothing at a running drop's edge, and D_num absorbed it with the opposite sign. A step whose
time part of D_num is negative beyond 1e-12 of the energy moved (and the rounding of the F values) carries a
`solver` flag under either solver; the transport part's sign is not established for a concave mixing (below). At
the solution of each substep these equal Σ flux · force (U(G − cY) = U²/(rM), and the rest likewise).

*Residual.* An operation's energy closure residual compares what it booked to `stored` with the change
of F recomputed on the region it touched, beyond the rounding of those F values (16 ulps of their
magnitude), over the energy it moved, floored at k_B T per cell of the region.

**D_num** is the remainder, −ΔF + W − D_mech, booked to `dissipated[numerical]`. ΔF of each substep is
computed from increments (quadratic terms exactly as ∇·δ + ½δ·Hδ, wetting with expm1), not as a
difference of totals, so its rounding is relative to the energy moved and not to F.

**O3 (energy).** Let F = F_c + F_x with F_c convex (implicit) and F_x concave (explicit), and let δ be a
substep's change of state with δ = −dt (div of fluxes) and fluxes J = ∂Ψ*(ξ), ξ = −(∇F_c(xⁿ⁺¹) + ∇F_x(xⁿ))
the force used. Summation by parts gives ⟨∇F_c(xⁿ⁺¹) + ∇F_x(xⁿ), −δ⟩ = dt ⟨J, ξ⟩ = D_mech. Then

```text
D_num = F(xⁿ) − F(xⁿ⁺¹) − D_mech
      = [F_c(xⁿ) − F_c(xⁿ⁺¹) − ⟨∇F_c(xⁿ⁺¹), xⁿ − xⁿ⁺¹⟩]  +  [G(xⁿ⁺¹) − G(xⁿ) − ⟨∇G(xⁿ), xⁿ⁺¹ − xⁿ⟩]
      = Bregman_{F_c}(xⁿ, xⁿ⁺¹) + Bregman_{G}(xⁿ⁺¹, xⁿ)  ≥ 0,          G = −F_x convex.
```

This holds for any frozen mobility ≥ 0, any rate r ≥ 0, and for the constrained (drained) solution,
whose force includes the constraint's reaction (the indicator of h ≥ 0 is convex). It is the
minimising-movement inequality written as an identity: D_num is the Bregman divergence, O(|δ|²) =
O(dt²) per step, so the total over a fixed interval falls as O(dt).

*Flow and mixing.* The flow force omits F_mix: in the continuum Σ_i q_i ∇μ_i^mix = 0 (Gibbs-Duhem,
f homogeneous of degree 1), so mixing drives no total flow. Discretely, moving volume in donor
proportions changes F_mix by ΔF_mix ≤ 0 whenever f is convex and homogeneous of degree 1 and every
cell's outflow is ≤ hⁿ (the acceptance condition): a cell that loses a fraction α ≤ 1 in its own
proportions has f((1−α)q) = (1−α) f(q), what it receives adds subadditively (f(a+b) ≤ f(a) + f(b)), and
what each donor loses equals what its receivers gain, f(φ_d δV) = δV f(q_d)/h_d. This −ΔF_mix ≥ 0 is
upwind transport's numerical mixing: it is booked to `dissipated[numerical]` with note `transport`,
separately from the time part (note `time`), and it falls as O(dx), not O(dt). For a mixing f with a
concave interaction part this argument fails and the sign of the transport part is not established
(immiscible liquids mixed by the upwind scheme can raise F): a stated gap.

**O1 (conservation).** Every substep moves matter between cells by fluxes whose amount leaves one cell
and enters its neighbour (flux form), and between accounts by the same numbers it books (e, a). Mass
per substance is conserved to the rounding of the per-cell updates, ≤ 1e-12 of the mass present.

*Constraints.* The flow step's cell-outflow bound and orientation are convex constraints that are
cones (t·u is feasible for t ∈ [0, 1] if u is). At the constrained minimiser u*, d/dt Φ(t u*) at t = 1 is
≤ 0, which reads ⟨∇E(y*), y* − yⁿ⟩ ≤ −D_mech with D_mech = dt dx² Σ (u²/(rM) + Y u); with convexity of
E, D_num ≥ Bregman ≥ 0 still. The reaction forces' share is inside D_num, not in D_mech.

**O2 (positivity).** Flow: every accepted step has every cell's outflow ≤ hⁿ (a constraint of the
problem, not an after-the-fact check), so every substance's new amount is (1 − α)qⁿ + inflow ≥ 0 by the
arithmetic of non-negative numbers; saturated cells are exactly 0 before inflow. (α = min(out/h, 1): the
bound holds in exact arithmetic and the min absorbs its last-bit rounding; a violation beyond 1e-12
rejects the step.) Diffusion: Newton keeps active amounts > 0 and the upwind form keeps the flux-form
update from cancelling. Local: the same, and drained cells are exactly 0. Pores: θ < 1 by the barrier, w
only increases. A remaining negative amount rejects the step (halving); no amount is clipped.

**O4 (balance).** A state with no driving force has G⁰ = 0 on every face, X = 0, and μ = μ° (or k = 0)
in every cell. Then the right-hand sides are exactly zero, every face is stuck or carries U = 0, the
linear solve of a zero right-hand side returns exactly zero, Newton's first residual is exactly zero
and it does not iterate: the state is returned bit for bit. Bare cells are rest states because every
mobility touching only bare cells is zero and F is zero there. A Bingham film whose stress is below
yield everywhere has every face stuck and is returned bit for bit. (A film whose surface is level over
relief has a driving force of rounding size, (h + z_b) − (h' + z_b') ≠ 0 in floating point; it is a
rest state to rounding, not exactly.)

## 6. Locality (O5) against the energy inequality (O3): how both hold

O3 is had from implicit, globally coupled solves. O5 asks that a windowed step equal the whole-panel
step bit for bit. Both hold here, by three facts:

1. *Rest states are fixed points* (O4): mobility is zero on every face that does not touch a wet cell,
   F is zero on bare cells, and the local step does nothing on bare cells. So the cells that can change
   in a step are the active cells: wet cells and bare cells sharing a face with one.
2. *The solve is restricted to the active set.* Every linear and nonlinear system is assembled on the
   active cells (or pairs) only, numbered in raster order, with faces in raster order (x faces then y
   faces). Raster order restricted to a sub-rectangle keeps relative order, so a window that contains
   the active set and its stencil (a margin of 3 cells) produces the *same matrix with the same numbering*
   as the panel, and SuperLU's factorisation and solve are deterministic functions of the matrix.
   Every iteration (active-set classes, Newton) stops on criteria computed over the active set alone;
   every reduction that enters the state (none does) or the ledger is a sum over the compacted active
   set in canonical order. Element-wise arithmetic is the same element for element.
3. *Absolute coordinates.* The in-plane potential uses panel coordinates, so a window reads the same
   numbers.

So a window equals the panel bit for bit, not to a tolerance, and the energy inequality is exact in
both. (Tested: identical arrays and identical receipt digests over several steps of a three-substance
scene with evaporation, sorption, wetting and a tilt, on a window about half the panel.) The cost is that a wet region is solved as a whole: windowing saves only the dry part of the
panel. A negative result in the other direction: an explicit local scheme would give O5 trivially but
O3 only under a step bound (explicit Euler on a convex F has D_num = −Bregman ≤ 0), and the capillary
term's stiffness, dt ≲ 3η dx⁴/(γ h³) ≈ 1e-6 s for water at 0.1 mm, makes it useless. Neither obligation
is relaxed.

*Cost proportional to the active region.* A step works on R, the window together with what changed in
the last step: coefficients are refreshed on R, F's totals are kept and changed by F(after) − F(before)
recomputed on R (outside it the state is untouched, so F's density is too; the window's three-cell
margin makes every face crossing R's edge join unchanged cells), and the slope is read on R. If a
halving's window reaches past R, R grows, and the cells added are saved as they were at the step's
start (they have not changed yet). Strata are stored on their bounding boxes and z_b is kept, not
recomputed. A fork shares every array and copies it on its first write (O6 tested). What remains
panel-sized per step is element-wise bookkeeping (the wet mask, the rate field, mass sums).

## 7. Clocks (SPEC §8)

`rate` multiplies every Ψ* (a field on cells; faces take the min of their cells). rate = 0 makes every
mobility zero: the step returns the state bit for bit (O4's argument) and the clock advances. Any
rate ≠ 1 flags `privilege` with amount ∫ r dt (the panel mean of r times dt). r ≥ 0 keeps every Ψ*
convex, so §5 holds unchanged. Negative rates are refused (reverse is a later wave).

## 8. Retirement, ports, forks

`retire(mask, substances)` moves those substances in those cells from `wet` to `stratum[k]` (bottom
first) and raises z_b by their thickness. The change of the live F is computed as the difference of F
before and after and moved from `stored` to `frozen`: nothing is dropped. `add(amounts)` is a source
port: matter from `origin`, energy F(after) − F(before) from `work[origin]`. Changing the environment
books its ΔF as `work[environment]`. `fork()` copies the state and both ledgers.

## 9a. Solvers, advance, settle

*Solvers* (SPEC §3). `exact` factorises every system (SuperLU) and is the reference: O3 by construction.
`iterative` solves the same systems with GMRES, preconditioned by an incomplete LU, to a relative residual
of 1e-10 within a fixed budget; a solve that misses it is factorised instead (counted as a fallback). The
systems are assembled on the active set in canonical order, so the iterates are the same in a window and
on the panel (O5, O6 hold, tested). D_num's sign holds to the tolerance; receipts report `max solver
residual`, `solver iterations`, `solver fallbacks`; a negative D_num beyond rounding is flagged. Where a
frozen choice of the step (a face's orientation, an entering amount kept or dropped) is decided on a number
the solver knows only to its tolerance, the two solvers can take different admissible steps, which differ
at O(dt), not at the tolerance (seen: 1.6e-4 of the film).

*Single-substance cells in the local step* are not solved by Newton: their residual q' − q₀ + e(q') + a(q')
is increasing in q' (the film's potential rises with q' under the convex split) and a(q') solves an
increasing equation at fixed q', so both are bracketed and bisected to the last bits; the drain (h' = 0) is
the case where the residual at 0 is still ≥ 0. (Newton oscillated on real water in cellulose, between filling
the pores past what the film held and the one-way valve shutting.)

*Newton's inner solves* (diffusion), under both solvers, are GMRES to a relative residual of 1e-10 preconditioned by
block Jacobi over each cell's unknowns, then (if that misses within 12 iterations) by aggregation multigrid on the
active set (`multigrid.py`: 2 x 2 aggregates in absolute panel coordinates, so a window builds the same hierarchy as
the panel; damped block-Jacobi smoothing; the coarsest level factorised), then a factorisation. The answer is
the nonlinear iterate, judged on its own residual (the convergence test adds a residual check so that a stalled
inner solve cannot pass as converged), so the inner tolerance sets the path and not the result: O3's argument
is unchanged, exactly as for the chord iteration, which already solves with a stale Jacobian. The system is
assembled on the substances that move on some face only (the others have ω = 0: no rows of L).

*advance(duration, rate)* covers world time with steps whose size follows the step's own error estimate,
e = D_num(time)/energy moved (the Bregman divergence, O(dt²) per step): double after e ≤ tol/4, halve after
e > tol (tol 1e-3), divide by the pieces after a halving. It reads only receipts, so it is the same in a
window and on the panel. What it reads: D_num time less its constraint share (above) and, when a cell drained dry
in the local step, less the local step's part; against the energy moved, floored at k_B T per cell of the region
and at half the recent power (the moved energy per second of the last steps, averaged) times dt, so a step in which
little happens is not held to a relative error on the little it moved. Only a full step (not the remainder that
lands on the end) sets the next size. *The noise floor and the bound on change.* e is a relative error of the energy
the step moved, which does not vanish as a decaying mode dies out (implicit Euler's D_num over the energy moved is about
λ dt / 2 at every amplitude), and near equilibrium both are rounding: before this rule a drying paste was held at
dt = 0.5 s for 2000 s of equilibrium (5516 steps for an hour of drying). The step reports `max relative change`, the
largest change of any amount in a cell over the step, relative to the cell's thickness (at least a tenth of the thickest
cell of the region, so a front entering bare cells is measured against the film); e is multiplied by
min(1, change / 0.1), an absolute tolerance on the state of 0.1 tol per step; a step whose change exceeds 0.25 halves
the next, and dt doubles only when the doubled step's change would stay below 0.25. Measured: the paste dries in
145 steps, and its water agrees with fixed 10 s steps to 1.2e-3 every 300 s (tests/world/test_world_painting.py). A halving that does not lower e by a quarter shows an error that is not the
time step's: the size returns to the one before and is held until e falls below tol/4. A step the world cannot
take (`StepFailed`, after its 24 halvings) leaves the world exactly as it was (the step is all or nothing: what its
accepted pieces committed is undone) and is retried at a quarter of its size; below 1e-9 of the duration advance
stops and returns what it did, with a `solver` flag saying where and why. The clock's privilege is summed over
the steps into one flag (equal flags of equal steps would otherwise merge when receipts add).

*The rest memo.* A step every substep of which left on an exit decided on the old state alone (flow: no face can move;
diffusion: no face with mobility, or every divergence exactly 0 at the old state; local: no driving reservoir and no
pores) returns the state bit for bit for *every* dt, so until the state, the environment, the clock's rate, the solver
or the window mode changes the next step is that same rest step. The world keeps the last such step's receipt keyed by
its state version (counted by every act: step, add, retire, settle, set_environment), the identities of q and held (a
fork's first write copies them), the rate (a scalar, or a digest of its field), the solver and the window, and returns
it with the new times and what depends on dt (the rate's integral, the privilege). Tested receipt by receipt against the
computed step and against each act's invalidation (tests/world/test_world_acts.py). Measured: a resting 192 x 256 paste,
231 ms a step computed, 0.2 ms recalled.

*Compiled kernels.* The local step's Newton update is compiled with numba (goo/world/kernels.py): per cell, no threads,
fastmath off, cached on disk; it gave the numpy version's bits on the painter's scenes. For one volatile and no pores
(K = 1) the residual is increasing in e, so each evaluation brackets the root and a Newton step that leaves the bracket
is replaced by its midpoint: where Newton stays inside the iterates are unchanged; at a vitrifying gum film (water's
potential bends sharply at the glass transition) it two-cycled and the step halved 516 times in 300 s of drying.
*Potentials without Hessians.* Where a substep needs μ and not the Hessian (the ray, the flow's transport term, the
local step's bracketing and drain tests, F's mixing term) the world calls the protocol's optional
`mixing_potentials` (the same values, no [S, S, n] array), and the local Newton reads only the Hessian's block on the
substances that exchange or sorb (`mixing_block`); a constitutive object without them is served by `mixing`. The ray's
search on Φ' is regula falsi (Illinois), with no Hessian, and ends on a point where Φ' ≤ 0 was computed; a flux whose
dissipation D1 is below k_B T per cell of the solve while Φ'(0) shows no descent to rounding is no flux (D_num = 0). The maps are not
compiled by the world (SPEC §4.4): they are most of the local step's time (measured: 57 % of it on a 197k-cell paste).

*settle()* retires every wet cell whose viscosity is +inf (no liquid left, or jammed particles): mobility is
exactly zero there, so retiring loses no freedom. A film that still holds a liquid is not touched.

*What the ground does, from the code.* Sorption acts in every wet cell with suction, whatever strata lie
between the film and the ground: strata do not block it (a gap: a stratum's permeability is not described).
Water held in the pores does not evaporate: exchange acts on the wet film only. There is no lateral transport
in the pores: `held` changes only by vertical sorption. Particles are not sorbed (the map gives them no
suction: filtered); as their liquid leaves, the film becomes particle-rich, its viscosity rises (Krieger) to
+inf at packing, and settle() retires it.

## 10. Cure and dissolution: not yet terms

*Cure* does not fit the gradient structure with the maps as they are. `reactions(film, env)` gives a
rate of extent (first order, 1/s) and no affinity, and a rate with no free energy is not a term of F or
Ψ*. With a free energy of reaction (g° per kg of each substance, or ΔG° per kg of extent, and the
reservoir species' potentials per kg), the term is F_rx = Σ dx² Σ_i ρ_i q_i g°_i, the reservoir's draw is
work by its port, the affinity is A = −Σ_i ν_i (g°_i + μ_i/ρ_i) − Σ_res ν_res μ_res, and Ψ* = k m A₊
(convex, zero at 0) gives the map's first-order rate wherever A > 0 and stops at equilibrium.

*Dissolution* needs a stratum to hold a live potential. Under the retirement rule its free energy is in
`frozen`, outside F, so no force can drive matter out of it. It needs the top stratum's mixing free
energy kept in F (frozen holding only what cannot move) and a rate-coefficient map for stratum-to-liquid
transfer.

## 9. Orders (O7) and what is verified

Space: second order in dx for smooth fields (centred mobility, centred differences; the upwind φ in
the diffusion mobility changes the flux at second order in the amplitude, not the linear rate); first
order at fronts (one-way faces, donor proportions). Time: first order (implicit Euler, Lie splitting, frozen
mobility). D_num: first order in dt (time part); the transport part is O(dx). Tests: Newtonian ripple
(rate (h³/3η)(ρ g k² + γ k⁴)), Bingham ripple below yield (held exactly), ideal-mixture ripple (rate
D k²), evaporation of a pure volatile into a fixed atmosphere (closed form, then exact drain).

Measured: the Newtonian ripple's rate converges at order 2.0 in dx and its amplitude at order 1.0 in dt;
D_num (time part) falls at order 1.0 in dt; the mixture ripple converges at order ≈ 2 in dx; a Bingham
ripple below yield is returned bit for bit, one above yield comes to rest exactly, with its largest wall
stress at τ_y (≤ τ_y(1 + 1e-9), > 0.9 τ_y), and keeps ≈ 0.38 of its amplitude (more than the 0.3 a pure
cosine would keep, because the crests stop before the flanks); evaporation of a pure volatile follows
the closed form of implicit Euler to 1e-10 and the exact solution to 1e-6, then drains to exactly zero.

Unverified: contact-line motion with partial wetting (ℓ is molecular and unresolved by 0.1 mm cells,
so it is not converged in dx); multi-component diffusion against Maxwell-Stefan; any real substance.
