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

*Entry* (**R6**): a substance enters a neighbouring cell that lacks it only if, at e⁻³⁰ of the donor's
fraction, the relative force at the old state still drives it in. Where it would not, its equilibrium
share in that cell is below e⁻³⁰ of the donor's (uphill against strong drift, or against a solvent's
counter-flux), it is not representable against the fluxes that balance it, and that face carries none
of it this step. A numerical threshold on representability, not a physical constant.

**4.3 Exchange and sorption** (local). Per cell, unknowns e_i, a_i; Newton per cell, each cell
stopping on its own tolerance. The film's potential at the new state uses: convex mixing implicit,
concave mixing explicit, gravity split as in 4.1, and capillarity *stabilised*: μ_cap = (1/dx²)[∂F_cap/∂h_c(hⁿ) + σ_c δh_c]
with σ_c = Σ_{f∋c} γ_f, which is the convex splitting F_cap = (σ/2)|h|² − ((σ/2)|h|² − F_cap) and
needs σ ≥ ½ λ(γ L_graph) (Gershgorin). A single-substance cell whose implicit solution would have h < 0
is drained exactly: h' = 0, with e and a split so that both see one force ξ; this is the KKT point of
the constraint h ≥ 0, and ξ ≤ μ(h → 0⁺) is checked.

## 5. Ledger and the argument for O1 to O4

**D_mech** of each mechanism is computed from the fluxes applied and the forces that drove them:

```text
viscous     dt dx² Σ_f U_f² / (r M_f)            plastic   dt dx² Σ_f Y_f |U_f|
diffusion   dt dx² Σ_f Xᵀ r L X                  exchange  dx² Σ e²/(dt r k)       sorption  dx² Σ a²/(dt r κ)
```

each a sum of non-negative terms, so D_mech ≥ 0 in floating point, not only in exact arithmetic. At
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

*advance(duration)* covers world time with steps whose size follows the step's own error estimate,
e = D_num(time)/energy moved (the Bregman divergence, O(dt²) per step): double after e ≤ tol/4, halve after
e > tol (tol 1e-3), divide by the pieces after a halving. It reads only receipts, so it is the same in a
window and on the panel.

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
