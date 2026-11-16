# goo performance: what makes the Python kernel fast enough to paint

Owner: performance. This is an analysis with working prototypes. It changes no file under `goo/` or `tests/`.
Prototypes live in `bench/`, and each runs as `PYTHONPATH=$PWD python bench/NAME.py ...` (each script's
docstring gives its arguments). Every number is marked **measured** (with what it was measured on), **computed**
(with what it was computed from) or **estimated** (with its assumptions).

**The machine of record.** Every measurement was taken on this one: "2 cores" of a 2.3 GHz Xeon (AVX-512),
7 GB RAM, numpy 2.5.3 with OpenBLAS on 1 thread, scipy 1.18, numba 0.68.
**Measured:** the second core gives nothing. Two processes doing the same pure-Python loop take 0.36 s against
0.17 s for one. A compute-bound numba `prange` loop runs in 0.64 s on 2 threads and 0.67 s on 1. In practice this
machine is one core. Single-thread memory bandwidth is **measured** at 7 GB/s for a write and 11 GB/s for a
read+write stream. First-touch writes to fresh pages run at 4 to 5 GB/s.

**The question, after the lead's redirect.** What makes the Python kernel (numpy, scipy, numba) fast enough to
paint a megapixel picture with real drying on this machine, and is that achievable? The lead's targets, and where
the evidence stands on each, is the table in §0. The GPU and compiled-language material is the short §8.

---

## 0. Summary

| target on this machine | present (measured) | with the prototypes | verdict |
|---|---|---|---|
| quiet step on 786,432 cells: "a few seconds or less" | 15.7 s, 3.0 GB | 4.4 s, the sum of measured parts; about 1.5 s **estimated** once fused | **supported**: 4.4 s measured, 1.5 s estimated |
| a full-panel paste dried to equilibrium in "a few minutes" | 5,516 steps for 3,600 s of world time (96 x 128, measured), so hours at megapixel | 120 fixed 30 s steps reach the same final state (measured on 96 x 128); 120 x 1.5 to 4.4 s = 3 to 9 min (**computed**) | **supported only after the step-size controller is fixed**; it is the larger factor |
| moving paint at about 50,000 active cells: "a few seconds a step" | 25,000 cells: 1.8 s a step at small dt (82% in SuperLU); about 7,000 cells at dt = 0.5 s: 82 to 106 s (405 factorisations) | multigrid flow on 25,000 cells: 0.18 s; on 395,000: 2.0 s (numpy, 8 V-cycles, measured) | **supported where cells do not empty in the step**; the outflow constraint is unsolved (§2.3) |
| a local passage of a few hundred to a few thousand cells: "well under a second a step, whatever the panel" | 46 ms a step on an empty megapixel panel; **2.0 s** a step when 900 other wet cells sit in a far corner | needs activity per cell instead of a bounding box (§3.1), not built | **not yet supported**: the cause is identified and measured, the fix is not built |
| a look at the whole panel: "tens of seconds, no running out of memory" | 1.66 us per cell-band-film, 78 to 94% of it in the maps; about 120 B per cell-band-film | not prototyped; cached strata plus maps per distinct composition, **estimated** at 10 to 30 s | **estimate only** |

**The four findings that matter.**

1. **The step size, not the step, decides the drying time.** `advance` holds dt at 0.5 s for 2,000 s of world time
   while the film sits at equilibrium. Its error estimate, D_num/(energy moved), becomes a ratio of rounding to
   rounding (**measured**: 5,516 steps to dry a paste, against 120 fixed steps that reach the same 4.104 um of
   water). No speed-up of the step makes up a factor of 46 in the number of steps.
2. **For moving paint, scipy's sparse direct factorisations are the cost, and the active set multiplies them.**
   82% of a 25,000-cell moving step is SuperLU's `gstrf` (**measured**). Where cells would give more than they
   hold in a step (a Courant number near 1), the primal active set refactorises once per blocked face or cell:
   339 factorisations for 1,664 cells (**measured**). This cost is already compiled C, so a straight port of the
   same algorithm (numba, or Rust later) gains at most the remaining 18% here.
3. **For quiet paint the cost is algorithm plus numpy, not physics.** The diffusion substep is not stiff at these
   resolutions (D dt/dx² is about 1e-4 to 0.06, **computed**) and changes the film by 2.5e-5 relative
   (**measured**). Yet it runs Newton with Krylov solves and Jacobian assembly: 70% of the quiet step. One explicit
   Onsager flux with an exact ray search does it in 1.41 s instead of 11.4 s at megapixel, keeps D_num ≥ 0 by
   construction, and is slightly more accurate (**measured**).
4. **Admissibility can be decoupled from solver accuracy.** For any flux field that is feasible and *optimal along
   its own ray*, D_num equals a Bregman divergence and is ≥ 0 by construction, and conservation is exact by flux
   form. This holds for the flow (multigrid) and diffusion (explicit flux) prototypes, whatever the solver's
   tolerance. The iterative solver's present weakness, "the sign of D_num holds only to that tolerance", goes away.

---

## 1. Cost model

### 1.1 What the mathematics requires (independent of the code)

Notation: N_a active cells, S substances in a cell, B bands, L films in a look.

**State, per cell (f64).** q [S], held [S], z_b, lagged coefficients (γ on 2 faces, S_w, ℓ, P [S], P_b): about
(3S + 6) x 8 B, so 120 B for S = 3 and 312 B for S = 11 (**computed**). The present code carries q and held densely
for all of the world's S substances. A painting has S = 11, but a cell holds about 4 (§3).

**Flow, per active cell, one density.** One SPD-structured linear system, in mixed form two 5-point operators:
about 20 flops to apply it. A multigrid V-cycle (2+2 red-black block Gauss-Seidel sweeps of about 30 flops, a
residual, transfers, times the 4/3 of the coarse levels) is about 200 flops and about 400 B of memory traffic per
cell (**computed** from the prototype's operations: about 10 arrays read and written per sweep). Eight cycles
give a residual of 1e-6: 1,600 flops and 3.2 KB per cell. That is 0.3 s of memory traffic at 786k cells at
10 GB/s (**computed**). Sparse direct factorisation has no place in this count. Its fill grows with N
(**measured**: from 6.7x to 18x nnz(A) between 1.6k and 99k unknowns; 1.6 s and 279 MB at 99k unknowns), and in
the constrained case it is multiplied by the number of active-set changes.

**Diffusion, per wet cell.** It is not stiff here (D dt/dx² for water at dt of seconds and dx = 0.25 mm is about
0.06, and for gum or pigment about 1e-4, **computed** from Stokes-Einstein mobilities). So the required work is
one explicit flux (2 faces x S substances x about 10 flops) plus 2 to 4 evaluations of the mixing potentials for
the ray search (S logs plus about 5S² flops each). That is about 400 flops per cell for S = 3 (**computed**).

**Local exchange, per wet cell.** Newton on one to three unknowns with 3 iterations (**measured**: 3 everywhere
in the quiet scene), each one mixing evaluation: about 300 flops (**computed**).

**Maps, per cell.** Viscosity (Krieger), yield, mobility, exchange, surface tension: tens to a few hundred flops
each, and they depend only on the cell's composition (§3.2).

**Quiet step total.** About 1,000 to 1,500 flops per cell for S = 3, about 1 GFLOP at megapixel, so 0.3 to 0.5 s
compiled at 2 to 3 GFLOP/s scalar with logs (**estimated**). Memory traffic is about 2 KB per cell, 1.6 GB, or
0.2 s at 7 to 10 GB/s (**estimated**).

**A look, per cell, band and film.** A Kubelka-Munk closed form (exp, sqrt, a division: about 50 flops) and one
star product (about 20 flops), about 100 flops in all. For 786k cells, 18 bands and 12 films that is 17 GFLOP
(**computed**). The optical coefficients depend on composition alone: K and S are per unit of h (SPEC §4.3), so a
film laid from one recipe has one composition however its thickness varies. Strata are immutable, so their
composed operator is a constant until the next retirement.

### 1.2 Where the present code spends its time and memory (measured, three sizes and more)

`bench/profile_step.py SCENE H W dt` (cProfile on a warm step; scenes from `bench/scenes.py`).

**Quiet scene** (paste with a yield stress, evaporating; `quiet.py`'s scene), dt = 2 s:

| cells | step | peak RSS | flow | diffuse | local | world bookkeeping |
|---|---|---|---|---|---|---|
| 12,288 (96 x 128) | 0.16 s | 127 MB | 0.01 | 0.10 | 0.04 | 0.01 |
| 49,152 (192 x 256) | 0.62 s | 267 MB | 0.02 | 0.39 | 0.16 | 0.05 |
| 196,608 (384 x 512) | 3.06 s | 827 MB | 0.10 | 2.03 | 0.73 | 0.23 |
| 786,432 (768 x 1024) | 15.7 s | 3.0 GB | 0.51 | 11.05 | 3.28 | about 1.0 |

The step is linear in cells at about 20 us per cell, and memory is about 3.8 KB per cell at peak. By module at
megapixel: `maps.py` 4.5 s of own time (the mixing map alone 4.3 s over 14 calls), diffusion's Python and numpy
3.7 s, scipy sparse assembly and matvecs about 2 s, the local substep 1.1 s. Nothing is factorised: the flow's
quiet-face rule already proves rest for this paste.

**Moving water** (a tilted 40 um disc on a sealed ground, `bench/scenes.py water`):

| wet cells | dt | step | factorisations | share in `gstrf` |
|---|---|---|---|---|
| 448 (24 x 32) | 0.5 s | 0.51 s | 68 | |
| 1,664 (48 x 64) | 0.5 s | 10.1 s | 339 | |
| 6,934 (96 x 128) | 0.5 s | 82 to 106 s | 405 | 83% |
| 24,564 (192 x 256) | 0.05 s | 1.78 s | 2 | 82% |

At dt = 0.5 s the disc's cells would give more than they hold (the outflow bound is a Courant condition of donor
transport; 1,114 of 1,536 cells overdraw unconstrained, **measured** in `bench/flow_mg.py`). The primal active
set then blocks them one or a few at a time, each block costing a refactorisation. The number of changes grows
like the number of constrained cells and each factorisation like N^1.3 to N^1.5 (**measured** fill growth above),
so the step grows faster than N². At small dt nothing binds and the cost is two factorisations: SuperLU's
diagonal-pivot attempt and then its fallback, 0.74 s each in the world (**measured**), against 0.2 s for the same
matrix factorised once outside it (**measured**, `bench/factor_scaling.py`).

**The step-size controller** (`advance`) on the quiet scene at 96 x 128 (**measured**): 3,600 s of world time took
5,516 steps and 454 s of wall time. dt fell to 0.512 s and stayed there from t = 900 s to t = 3,000 s, while the
water held at 4.104 um, its equilibrium with the air. Fixed steps (**measured**) do better: dt = 30 s, 120 steps,
13.8 s of wall time, a trajectory within 2% of `advance`'s and the same final 4.104 um; dt = 10 s, 34 s of wall
time, within 0.7%. Energy residual 0 and D_num ≥ 0 throughout.

**A lone drop** (317 cells, 30 degrees of tilt): 30 ms a step on 96 x 128 and 46 ms on 768 x 1024 (**measured**).
The same drop with 900 wet cells in the far corner of the megapixel panel: 2.0 s a step (**measured**), because
the window is the bounding box of every wet cell, here the whole panel. In the painting (repro/drips2.py), every
gum-bound stroke stays wet (SPEC §12: `settle()` never retires it), so every drop is solved together with the
whole panel. That explains 7 to 66 s for 0.5 s of world time.

**A panel at exact rest** (a uniform paste, no reservoir) on 768 x 1024: 3.9 to 4.6 s a step that returns the state
bit for bit (**measured**). With the clock held (rate 0) on the quiet scene: 0.93 s a step (**measured**).

**A look** (`bench/look_cost.py H W FILMS BANDS`, every cell different; **measured**):

| cells x films x bands | time | peak RSS | maps (`_inputs`) | layer solve |
|---|---|---|---|---|
| 7.5k x 6 x 16 | 4.8 s | 232 MB | 4.5 s | 0.04 s |
| 30k x 6 x 16 | 6.9 s | 455 MB | 5.9 s | 0.45 s |
| 120k x 6 x 16 | 19.2 s | 1.37 GB | 14.9 s | 0.84 s |
| 30k x 12 x 16 | 10.2 s | 734 MB | 8.4 s | 0.56 s |
| 30k x 6 x 32 | 11.0 s | 746 MB | 8.8 s | 0.07 s |

That is 1.66 us per cell-band-film at 120k cells, of which the layer algebra is 73 ns. Memory is about 120 B per
cell-band-film. The whole painting (786k x 12 x 18) would need about 20 GB (**computed**), hence the kill at
6 GB. Films laid from one recipe (one composition, varying thickness) are no cheaper: 14.0 s against 19.2 s
(**measured**). The maps' dedup (`_F.unique`) keys on q, not on composition, so every cell is distinct.

### 1.3 The gap and its parts

| workload | present | requirement (computed or estimated) | ratio | the gap is mostly |
|---|---|---|---|---|
| quiet step, 786k cells | 15.7 s, 3 GB | 0.3 to 0.5 s compiled; about 1.5 s Python plus numba (estimated); about 120 B per cell | 30 to 50x, and 30x in memory | algorithm (Newton plus Krylov where one explicit flux suffices: 11 s → 1.4 s measured), then numpy temporaries (maps 4.5 s, of which a fused kernel removes 7x, measured) |
| drying to equilibrium | 5,516 steps | about 120 steps | 46x | the controller |
| moving, 25k cells, small dt | 1.78 s | 0.18 s multigrid (numpy, measured); about 0.03 s bandwidth-bound (computed) | 10x by algorithm, 60x in all | algorithm (factorisation); Python overhead under 18% |
| moving, Courant near 1 | 10.8 s at 1.6k cells | 0.04 to 0.4 s (multigrid plus constraint handling, §2.3) | 25 to 250x | algorithm (active set x factorisation) |
| a drop beside a wet panel | 2.0 s | 46 ms (measured on an empty panel) | 40x | representation (bounding-box window) |
| look | about 1.6 us per cell-band-film, 120 B | about 50 ns, a few B (streamed) | 30x | maps on every cell instead of per composition, and no reuse of the strata (representation) |

**What a compiled implementation of the *same* algorithm would buy.** For moving paint, at most 18% (the
non-factorisation share). For the quiet step, perhaps 5 to 10x: the mixing kernel is **measured** at 7x over
`maps.mixing` with S = 3 and 5x with S = 11, and the local Newton at 4.7x (megapixel, inputs prepared by numpy).
But the quiet step's dominant term is diffusion's Newton and Krylov machinery, where changing the algorithm gave
8x in numpy before any compilation. For the look, about 1.3x on the layer algebra and much more on the maps.
**The lead's belief is confirmed for moving paint and refuted for quiet paint.** On quiet paint, compilation and
the algorithm each give roughly an order of magnitude, and they multiply.

---

## 2. The algorithm

Each substep minimises a convex function Φ (a minimising movement). The present code finds the minimiser exactly:
an active set with sparse direct solves for flow, Newton with Krylov solves for diffusion, Newton or bisection
per cell for the local step. The structure that fits the problem better rests on three observations.

### 2.1 Admissibility by construction from any feasible flux: the ray condition

Let J be any flux field (faces x substances) whose update y = yⁿ − (dt/dx) div J is feasible: amounts ≥ 0,
outflow ≤ content, valves respected. Let D_mech(J) = dt dx² Σ J·Ψ*'(J) be its dissipation computed from the
constitutive law, which is ≥ 0 term by term. Then choose t ≥ 0 to minimise Φ(tJ) on the feasible segment. Φ is
convex in t, and for the quadratic flow problem the minimiser is in closed form. At the minimiser,
t ∂_tΦ(tJ) ≤ 0, which reads ⟨∇F_c(y_t) + ∇F_x(yⁿ), y_t − yⁿ⟩ ≤ −D_mech(tJ). Convexity of F_c (and concavity of the
explicit part) then gives

    D_num = −ΔF − D_mech ≥ Bregman ≥ 0,   with equality to the Bregman divergence when t is interior.

This is the argument of DYNAMICS.md §5 ("Constraints"), applied to an *arbitrary* feasible direction instead of
the exact minimiser. Its consequences:

- **O1 (exact, by flux form) and O2 and O3 (by construction) hold for every iterate of every iterative method**,
  once a single scalar line search follows it. The solver's tolerance decides how close the step is to implicit
  Euler (its accuracy), never whether it is admissible. The present iterative path, "the sign of D_num holds only
  to that tolerance", becomes unnecessary.
- **Measured** (`bench/flow_mg.py`): with 2 V-cycles the ray gives t = 0.54 and D_num = Bregman = 4.4e-9 ≥ 0. With
  12 cycles, t = 1.000000 and the step equals the factorised one to 3e-13. D_num equals the present code's D_num
  to 4 digits (6.393e-9 in both).
- *Caveat.* D_num then contains the solver's shortfall: an inexact step dissipates less than it should, which is
  slower, not unphysical. It is no longer purely the time-discretisation error that `advance` reads. A receipt
  must therefore report the solver's residual and t, and the controller should read the Bregman divergence of
  the step, which is the time error alone.
- *Contract.* Nothing in SPEC §3 has to change for the `exact` solver. For the iterative one, the sentence "the
  sign of D_num holds only to that tolerance" can be struck. D_mech for an inexact flux is computed from the
  fluxes applied and the force the constitutive law assigns them, which SPEC §3 already allows ("computed from the
  fluxes applied and the forces that drove them").

### 2.2 The flow substep: matrix-free multigrid on the pressure (mixed) form

With one density the flow system is A δh = b, with A = I + (dt/dx²) K H, K = C M Cᵀ (mobility Laplacian) and
H = g'I + capillary Laplacian. It is fourth order. Written with the pressure increment p = H δh, it becomes two
second-order 5-point equations, δh + c K p = b and p − H δh = 0. Each cell holds two unknowns (δh, p), and a 2 x 2
block solve per cell is the classical point smoother for Cahn-Hilliard and thin-film systems. The prototype
(`bench/flow_mg.py`) uses red-black block Gauss-Seidel coloured by absolute (row + column) parity, bilinear
prolongation, averaging restriction, coarse levels rediscretised (coarse face mobility and γ the means of the two
fine faces on the coarse face), levels aligned at absolute multiples of 2^K with inactive padding, and the
coarsest level smoothed 60 times. Every operator is a stencil on [H, W] arrays with no matrix assembled.

- **Complexity.** O(N_a) per V-cycle, with an h-independent contraction of about 0.3 per cycle (**measured**:
  residual 9e-3 → 1e-6 in 8 cycles at 25k, 99k and 395k cells alike). At dt = 0.002 s the contraction is 0.08 and
  at dt = 0.5 s about 0.35: coarse rediscretisation degrades as the mobility jumps (M ∝ h³ at the rim). A Galerkin
  coarse operator or the cycle as a preconditioner for CG would restore it (not built).
- **Accuracy against the present step** (same inputs, captured from the world, 48 x 64 noisy disc, dt = 0.002 s,
  no constraint binding): |h_MG − h_present|/|δh_present| = 5e-3 after 2 cycles, 3e-5 after 4, 2e-9 after 8,
  3e-13 after 12 (**measured**). At dt = 0.05 s on 6,140 and 24,564 cells: 5e-6 and 3e-6 after 12 cycles
  (**measured**).
- **Time** (numpy, one core, **measured**): 25k active cells, 8 cycles: 0.18 s (present: 1.56 s for the substep).
  99k: 0.46 s. 395k on the 786k panel: 2.0 s, of which about 0.25 s per V-cycle over the full rectangle. Peak RSS
  478 MB at megapixel.
- **O5.** The hierarchy, colouring and padding are fixed by absolute coordinates, and a fixed number of V-cycles
  contains no reduction. Element-wise numpy arithmetic gives the same bits whatever the array's offset
  (**measured**: exp, log, expm1, log1p, sqrt and power on arrays shifted by 1 to 16 elements, 0 mismatches, and
  scalar against vector exp, 0 mismatches, on this AVX-512 build). So a window and the panel give the same bits,
  *provided* the window contains every coarse cell touching the active set at every level: the margin must be
  2^K cells, not 3. Only the ray search has reductions: two sums over the active cells, taken in canonical order
  over the compacted active set (identical arrays give identical pairwise sums), or exactly rounded with
  `math.fsum` (**measured**: 30 ms at 786k cells), which makes them independent of order altogether.
- **O6.** Deterministic: no data-dependent iteration count, no threads with dynamic schedules.
- **What would change.** The face-orientation predictor (R5) and the active set leave. Orientation matters only
  with several densities. For one density the problem is convex across U = 0, and two-way faces between wet cells
  are admissible. **Finding:** the present code freezes every face whose old force is exactly zero (it is not
  "oriented" and never becomes free). On a noiseless disc that is 21% of the step (**measured**: the present
  solution differs by 0.21 from the factorised solution of the same convex problem, which multigrid reproduces to
  1e-7). R5 states the freezing for faces whose flow would reverse; for faces whose force is exactly zero at the
  start it is a stronger frozen choice than DYNAMICS.md says.

### 2.3 The outflow constraint (a cell cannot give more than it holds)

This is the hard part, and the cause of the 339 factorisations. Three admissible treatments:

1. **Projection plus ray** (built): scale every outflowing face of an overdrawn cell to give exactly its content,
   then search the ray. This is admissible (§2.1) but far from the KKT point when much binds (**measured**: 34%
   deviation from the present step when 1,114 of 1,536 cells overdraw).
2. **Mobility limiting** (built): re-solve with the mobility of faces out of overdrawn cells scaled by
   content/outflow. Any frozen mobility ≥ 0 keeps the energy argument, and M ∝ h³ says a draining cell flows more
   slowly, so this is arguably more physical than "gives exactly all". **Measured:** 2, 4 and 8 re-solves give
   17%, 13% and 11% deviation from the present KKT step, in 0.13 to 0.38 s against the present 10.8 s. This is a
   *different constraint model*, not a faster solve of the same one.
3. **Semismooth Newton or primal-dual active set with multigrid inner solves** (not built): the same KKT point as
   the present code, with all constraint changes taken at once and each inner solve a few warm-started cycles.
   It is typically about 5 to 10 outer iterations (**estimated**, from the literature on obstacle problems, not
   measured here). The present code's primal-dual moves cycled on this scene, so this needs a globalisation (the
   ray search provides one).

The outflow bound is in effect a **Courant condition of donor-cell transport**: a cell cannot export more than its
content in a step. At dt = 0.5 s the tilted disc moves about 0.9 cell per step (**computed**:
U/h = h² ρ g sin θ/(3η) = 4.5e-4 m/s, times 0.5 s, over 0.25 mm). So steps of Courant number about 1 are where
the constraint binds everywhere. These steps also have D_num/ΔF = 94% (**measured**), so `advance` should not
choose them for accuracy anyway. Bounding the step by a Courant number (≤ 0.5, say) would make treatment 1 or 2
rare and treatment 3 unnecessary. **Recommendation:** add a Courant bound to `advance` (it reads receipts: the
receipt reports max outflow/content), use mobility limiting with the ray for the residual cases, and keep the
exact active set as the reference solver.

### 2.4 Diffusion: one explicit Onsager flux plus an exact ray search

At the old state, compute X, the upwind ω, the Onsager flux j = ω(X − X̄), and δ = −(dt/dx) div j. Then minimise
Φ(t) = F(q0 + tδ) + ½ t² D1 (convex part of mixing exact, concave part and gravity linear) by Newton on t, each
iteration one mixing evaluation. t is capped by positivity. D_mech = t² D1, and D_num ≥ 0 follows by §2.1.
`bench/diffuse_ray.py`:

- **Time** (quiet scene, **measured**): 49k cells, 0.10 s against present 0.44 s. 786k cells, 1.41 s against
  11.4 s. The mixing kernel is numba; faces are numpy (0.53 s of own time, fusable).
- **Accuracy** against a reference of 32 implicit substeps (49k cells, **measured**), as error over the substep's
  change: dt = 2 s, present implicit Euler 3.6%, ray 3.0%, ray in two half steps 1.5% (first order). dt = 8 s:
  11.5% against 10.1%. dt = 30 s: 23.2% against 23.2% (t* = 0.53). dt = 120 s: 27.5% against 35.4% (t* = 0.25: the
  stiff modes cap the ray). At dt = 30 s, which §1.2 shows is enough for drying, the two are equally accurate.
- **D_num**: 9.7e-11 against present 1.08e-10 (49k), 1.56e-9 against 1.74e-9 (786k). Both are ≥ 0, the ray's by
  construction.
- **O5.** Element-wise plus two global sums for the ray (canonical order or fsum). The ray's t is one scalar for
  the whole active set. A window gives the same t only if its active set is the panel's (true by construction of
  the window).
- **Not covered by the prototype:** R6 (entry into a cell that lacks a substance) and R7 (sub-molecular floor).
  Both are rules on ω that slot in unchanged, because any ω ≥ 0 is admissible.
- **Where it stops working.** When t* falls well below 1 (stiff composition gradients, large dt), the step does
  less than implicit Euler would. Remedies, in order: split dt into k ray steps (each admissible, cost k mixing
  evaluations), or fall back to the present Newton on that piece. The receipt reports t*, and `advance` can treat
  t* < 0.9 as an error signal.

### 2.5 The local step: per cell, compiled

It is already per cell and needs no algorithmic change. It needs no Python loop over Newton iterations with a
full-panel mixing call in each (the present code does 3 iterations x a mixing call on all live cells). See §5.3.

### 2.6 Rest, activity and local time stepping

- **Exact rest** (`bench/rest.py`) needs three predicates: flow quiet (the existing KKT rule), diffusion forces
  exactly zero (X − X̄ = 0 on every face with mobility), and no reservoir driving any present substance with no
  sorption. Detecting them costs one pass over the cells: 1.6 s at megapixel against 4.5 s for the present step
  (**measured**), the pass dominated by the viscosity, yield, mobility and exchange maps. **Rest is a property of
  (state, environment, rate)**, so once proved it can be memoised by the world's state version, and the next
  step costs 6 us (**measured**). A painting at rest between acts, or under a held clock, would cost nothing per
  step.
- **Activity per cell instead of per bounding box.** The present window is the bounding box of the wet cells. The
  right unit is the set of cells where some substep can change something. Flow: cells touching a movable face.
  Diffusion: wet cells with a relative force above rounding. Local: cells with a driving reservoir. These sets
  differ: in a drying painting almost every cell evaporates (local, cheap, per cell) while only a few hundred
  flow (global solve). Each substep should run on its own compacted set. The flow solve's O5 argument already
  works on a compacted active set in canonical order, and the multigrid needs only its aligned blocks.
- **Local time stepping** (different dt per region) breaks the single-clock structure of the receipts and of O5
  (a window would need the same subcycling decisions). With rest memoised and activity per substep, most of its
  benefit is already had. Not recommended now.
- **Explicit treatment of non-stiff terms.** Diffusion (above). Exchange and sorption are stiff per cell
  (implicit per cell is cheap and stays). Capillarity is the stiff term (dt ≲ 3η dx⁴/(γh³) ≈ 1e-6 s, DYNAMICS.md
  §6) and must stay implicit: that is what the multigrid is for.

### 2.7 Bit-identical windows with iterative methods: the rules

1. Iteration *counts* fixed (V-cycles, Newton steps on t, ray steps), or decided by a test computed over the
   active set alone, in canonical order. A test over the window (including inactive cells) can differ.
2. Multigrid hierarchies, colourings and paddings in absolute coordinates, with a window margin of 2^K.
3. Every reduction over the compacted active set in canonical order, or exactly rounded (fsum). Never `np.dot`
   through BLAS: OpenBLAS's kernel and its threading are outside goo's control.
4. numba kernels element-wise with no reductions across threads. **Measured:** 1 and 2 threads give
   bit-identical mixing Hessians. `fastmath=False`, so no reassociation and no fused multiply-add, which LLVM
   would otherwise contract.
5. A reimplementation is not bit-identical to the reference: the numba mixing agrees with `maps.mixing` to
   5e-13 relative in μ and 9e-12 in the Hessian, with 90% of Hessian entries bit-identical (**measured**).
   Replacing a map therefore changes the goldens. O6 (replay) holds within one implementation. Across
   implementations the obligation must be a stated tolerance.

---

## 3. The representation

### 3.1 Activity, windows and strata

- **Window by activity, not by bounding box** (§2.6). This is the cause of the 2.0 s against 46 ms drop
  (**measured**).
- **Panel-sized work per step.** `_window` sums S planes of q over the panel, the rate field is built as an
  [H, W] array, mass residuals sum S x N, and `stack()` expands every stratum to a full [S, H, W] array:
  830 MB for 12 strata at S = 11 (**computed**). The world should keep a wet mask and per-substance mass totals
  updated incrementally from each step's region, keep rate as a scalar unless it is a field, and hand light the
  strata on their boxes.
- **Strata as immutable shared layers.** A stratum never changes (SPEC §6). Its optical operator per band (R, T,
  R', T') and the *composition of all strata* below the wet film can be computed once at retirement and cached.
  A look after an act then needs the wet film's maps plus one star product per cell and band. The cached base
  costs 4 x B x N x 8 B = 453 MB at f64, or 226 MB at f32, for B = 18 at megapixel (**computed**). Invalidation:
  only a retirement (a new stratum) or a dissolution (a stratum giving matter back: not yet a term) changes it.

### 3.2 Composition: sparse palettes, maps per distinct composition, caching by content

- **Dense S planes.** With S = 11 and about 4 substances present per cell, every [S, ...] array is 2.75x larger
  than needed and every Hessian [S, S] is 7.6x (**computed**: 121 against 16 entries). **Measured:** mixing over
  786k cells with S = 11 (4 present) takes 5.1 s in `maps.mixing`, or 0.95 s in the numba kernel, against 0.49 s
  and 0.067 s with S = 3. A cell's *palette* (up to P = 4 to 6 substance indices plus amounts) keeps per-cell work
  at P² instead of S². A stroke's palette is known when it is laid. Mixing two wet strokes merges palettes, and
  in practice P stays small.
- **Maps per distinct composition.** Viscosity, yield, mobility, surface tension, optical K, S and host index all
  depend on the composition φ (and T), not on h: K and S are per unit h (SPEC §4.3), and viscosity and packing
  are intensive. The present dedup (`_F.unique`) keys on q, so a film whose thickness varies has as many distinct
  cells as cells (**measured**: palette films are no cheaper). Keying on φ (exactly, bit for bit, as now) would
  reduce a sprayed film to one evaluation. *Unverified:* that every map is intensive in this sense; the voids
  map (`_voids`) must be checked.
- **Caching keyed by content.** Rest memoised by state version (§2.6). Coefficients cached per distinct
  composition across steps (a dict from the φ bits to the coefficients, bounded LRU). Strata operators cached at
  retirement.
- **Don't materialise the [S, S, N] Hessian.** The numba mixing kernel is memory-bound on writing it
  (**measured**: 2 threads no faster than 1; 136 B written per cell). Consumers (the local Newton, the ray's
  second derivative δᵀHδ) should compute what they need inside a fused kernel.

### 3.3 Precision

- **f64, required.** q and held (the matter ledger closes to 1e-12 of the mass: flux-form updates of amounts
  whose sum must be exact), every ledger sum, F's increments, the ray search, and the mixing potentials, whose
  differences drive diffusion (X = −Δμ/dx is a difference of numbers of size kT/v·ln φ; in f32 the rounding of μ
  is about 6e-8|μ|, which DYNAMICS.md's rule "a direction changes only where the force exceeds 1e-12 of the
  potentials" could not even express).
- **f32, safe.** Optical K, S, host index and the layer algebra of a look: the data are known to a few percent,
  and the G3 closure of a look is per cell. At f32 the closure is about 1e-7 rather than 1e-16. That fails the
  contract's 1e-12 as written (§8), but it is far below any visible quantity. Viscosity and mobility as
  coefficients would also be safe, but they are cheap to keep in f64.
- **Memory per megapixel** (**computed**), as the state with S = 11 dense against a palette of P = 6: q, held and
  lagged fields 2 x 69 MB + 40 MB ≈ 180 MB dense, ≈ 100 MB with palettes. Each stratum: 69 MB dense, about 25 MB
  with palettes, less on its box. A look streamed in tiles of 16k cells: about 0.4 GB transient at the present
  120 B per cell-band-film (**computed**: 16k x 18 x 12 x 120 B), plus the 226 to 453 MB cached base.

---

## 4. The machine (this CPU)

- **Vectorised numpy** reaches about 0.25 s per V-cycle at 786k cells (**measured**), about 6 to 7x off the
  bandwidth bound of the same arithmetic (**computed** in §1.1). For element-wise maps, numpy's temporaries cost
  4 to 7x (**measured**: the fused mixing kernel against `maps.mixing`).
- **numba** (**measured** on this machine): mixing 7x at S = 3 and 5x at S = 11. Local Newton 4.7x on the Newton
  itself (0.53 s against 3.56 s at megapixel), with the numpy preparation of its inputs (1.0 s) now the larger part
  and to be fused too. Cold compile 1.8 s (mixing) and 3.6 s (local). With `cache=True`, 0.14 s and 7 ms on the
  next start. Bit-identical across thread counts (**measured**).
- **Two cores add nothing** on this VM (**measured**). Any plan that relies on threads for its speed-up is
  invisible here.
- **What numba predicts about a compiled language.** The per-cell speed of a scalar compiled kernel (both are
  LLVM): it should be close to what Rust would reach on the same loop, single-threaded. It does not predict
  explicit SIMD (numba auto-vectorises only simple loops; the mixing kernel with its branches mostly does not),
  multi-core scaling (none is measurable here), or the absence of Python orchestration between kernels.

---

## 5. Prototypes (what was built and measured)

All run on this machine. "Would replace" names the present path, and "to replace it" names the tests that would
have to pass.

### 5.1 (i) Matrix-free multigrid for the flow of one substance: `bench/flow_mg.py`

- Data layout: [H, W] arrays for cells, [H, W−1] and [H−1, W] for faces, which is what the world's windows
  already are. It reads its inputs from the same `Ctx` the present flow reads.
- Measured: §2.2 (equal to the present step to 3e-13 where no constraint binds; 0.18 s against 1.56 s at 25k
  cells; 2.0 s at 395k cells; scaling linear, contraction 0.3 per cycle at every size) and §2.3 (constraint
  handling: admissible, but 11 to 34% from the present KKT point when most cells overdraw).
- Keeps: O1 (flux form, mass change 0 or 2e-16 measured), O2 (projection; min h ≥ 0 measured), O3 by construction
  (D_num = Bregman, measured equal to the present D_num), O4 (b = 0 gives 0, and the quiet-face rule is
  unchanged), O5 and O6 under the rules of §2.7, O8 (a term at zero enters only as zero coefficients).
- Does not yet: several densities (needs the orientation or a two-unknown smoother), yield stress (Y enters the
  convex problem as a threshold, so it needs the active set on faces or a semismooth treatment), the KKT outflow
  constraint (§2.3).
- To replace the present path: `tests/world/` O1 to O8 tests with `solver="exact"` kept as the oracle, and a new
  test comparing the multigrid step to the factorised one to 1e-10 on scenes where no constraint binds. The
  painter's cases in `tests/world/test_world_painting.py` within their budgets. A window test with the 2^K margin.

### 5.2 (ii) Exact rest detection: `bench/rest.py`

- Measured: 1.6 s detection at megapixel against 4.5 s for the present step that returns the same state; 6 us
  memoised.
- Keeps every obligation: it returns the state only where the present step provably does (the same predicates).
- To replace the present path: O4 tests, plus a test that every scene detected at rest returns the same digest
  through the full step.

### 5.3 (iii) The per-cell local exchange solve, jitted: `bench/numba_kernels.py local`

- The same Newton, limiter and stopping rule as `local.py` for one exchanging substance. Measured: megapixel
  0.53 s against the present 3.56 s, plus 1.0 s of numpy for its inputs. 49k cells: 0.019 s against 0.157 s.
  Iterations 3 everywhere. e agrees to 2.6e-14 relative but is not bit-identical (0.4% of cells), because the
  mixing arithmetic is reordered.
- To replace the present path: the local-step tests in `tests/world/` (evaporation against the closed form to
  1e-10, the drain's KKT check), then regenerated goldens, since the bits change.

### 5.4 (iv) numba version of a hot kernel: `bench/numba_kernels.py mixing`

- Measured: §4. Agreement 5e-13 in μ and 9e-12 in the Hessian (cancellation in H_ent).
- To replace `maps.mixing`: `tests/substance/` mixing tests (homogeneity, Gibbs-Duhem, the convex split) at their
  tolerances. The evidence and provenance wrapper stays in Python around the kernel.

### 5.5 (v) Diffusion by one explicit flux plus the ray: `bench/diffuse_ray.py`

- Measured: §2.4. 8x at megapixel (1.41 s against 11.4 s), as accurate as implicit Euler up to dt = 30 s,
  D_num ≥ 0 by construction.
- To replace the present path: the mixture-ripple test (rate D k², order 2 in dx), O3 sign tests at every dt, R6
  and R7 tests (which need the rules ported), and a fallback to Newton when t* < 0.9, tested.

### 5.6 Supporting measurements

- `bench/profile_step.py` (profiles), `bench/factor_scaling.py` (SuperLU fill and time), `bench/look_cost.py`
  (looks).
- **Negative results.** The numba kernels gain nothing from a second thread here. The projection-only constraint
  treatment is far (34%) from the KKT point. The rediscretised multigrid slows from 0.08 to 0.35 per cycle as dt
  grows. The explicit diffusion ray loses to implicit Euler at dt = 120 s.

---

## 6. Staged plan (Python first)

**Stage 1. Make drying possible: fix the step count, and make the quiet step cheap.**
Changes:
- `advance`: a noise floor on its error estimate (treat D_num below the rounding of the energies involved, about
  16 eps x |F| on the region, as zero), plus a bound on the relative change of q per step, so that dt grows as
  the film approaches equilibrium.
- Rest memoised by state version (`bench/rest.py`).
- Diffusion by explicit flux plus ray (`bench/diffuse_ray.py`), with the present Newton as a fallback when
  t* < 0.9.
- The local Newton jitted, its inputs fused (`bench/numba_kernels.py`).
- `maps.mixing` jitted, without materialising the Hessian for consumers that need one entry.

Expected:
- Steps to dry: 5,516 → about 120 (**measured** with fixed steps on 96 x 128; that a fixed controller reaches it
  is an **estimate**).
- Quiet step at megapixel: 15.7 s → 4.4 s (**computed**, the sum of measured parts: flow quiet check 0.5,
  diffusion 1.4, local 0.53 + 1.0 inputs, bookkeeping about 1.0), and about 1.5 s once the numpy preparation is
  fused (**estimated**).
- Drying a full-panel paste: about 120 x 1.5 to 4.4 s = 3 to 9 min (**computed** from the above).

Risks:
- The ray's D_num now includes any shortfall from t* < 1 (O3 holds, but the time-error estimate is
  contaminated: the controller must read the Bregman part).
- Goldens change, because the numba bits differ.
- R6 and R7 must be carried into the explicit flux.

Tests: O1 to O8 unchanged and passing, plus new tests:
- a dried film's state against the present code's at dt = 10 s to 1e-3;
- the step count of a drying run;
- rest-memo invalidation on every act (add, retire, environment, rate).

This is **achievable on this CPU**, and it does not need a GPU: the evidence is measured parts plus a stated
fusion estimate.

**Stage 2. Activity per substep and the drop beside a wet panel.**
Changes:
- Each substep runs on its own compacted active set (flow: movable faces; diffusion: forces above rounding;
  local: cells with a driving reservoir) instead of the wet bounding box.
- Panel-sized bookkeeping is made incremental: wet mask, mass totals, rate as a scalar.
- Strata are handed to light on their boxes.

Expected: a drop of a few hundred cells beside a wet painting goes from 2.0 s to about 50 ms a step (**estimated**
from the 46 ms measured on an empty panel). The evaporating rest of the panel costs the local kernel only, about
0.5 s at megapixel or less with rest memoised (**measured** kernel time).

Risks: O5's argument must be restated per substep. Each substep's set is defined from the state alone, in
absolute coordinates, so the argument carries over, but it is a new test.

Tests: the O5 window test on scenes with distant wet regions; the drop scene's time budget.

**Stage 3. Matrix-free multigrid flow with the ray, and a Courant bound.**
Changes:
- `bench/flow_mg.py` moved into `goo/world`.
- The Courant bound read from receipts by `advance`.
- Mobility limiting for residual overdraws, and the exact active set kept as the `exact` reference.
- Later: two densities (the (δh, δm) unknowns per cell as a 3 x 3 block smoother) and yield stress
  (semismooth Newton on faces).

Expected (**measured** in numpy): 25k cells 0.18 s and 395k cells 2.0 s, at 8 cycles. With numba-fused smoothers,
about 3 to 6x more (**estimated** from the bandwidth bound of §1.1). That meets "about 50,000 active cells in a few
seconds" with margin, and makes 300,000 cells about 1 to 2 s.

Risks: the constraint model changes (mobility limiting against "gives exactly all") at steps of Courant number
near 1. Coarse-grid robustness at contact lines.

Tests: the multigrid step against the factorised one to 1e-10 where no constraint binds; the Newtonian ripple
rate and orders (O7); the Bingham ripple held bit for bit (O4); D_num ≥ 0 at every tolerance.

**Stage 4. Looks.**
Changes:
- Maps keyed on composition instead of q.
- Strata's composed operator cached at retirement.
- Tiles of 16k cells.
- Optionally f32 for the layer algebra, with a stated per-cell closure tolerance.

Expected (**estimated**):
- The base is cached, so the per-look work is the wet film's maps (a few distinct compositions per stroke) plus
  one star product per cell and band: 786k x 18 x about 100 ns ≈ 1.4 s.
- Memory under 1.5 GB.
- Without the cache, all 12 films: 786k x 18 x 12 x 73 ns ≈ 12 s of layer algebra (**computed** from the
  measured 73 ns), plus maps per distinct composition.

This is "tens of seconds" on the evidence, not proved.

Tests: G1 to G7 unchanged; a look with cached strata equal to the uncached look bit for bit (the same
element-wise arithmetic in the same order: the star product is associative only to 1e-12, so the cache must
compose in the order the uncached look does, bottom to top).

---

## 7. What I do not know

- Whether a fixed `advance` actually reaches about 120 steps for drying. Measured: fixed steps reach the
  equilibrium; the corrected controller is not built.
- Whether mobility limiting (§2.3) is an acceptable change of the constraint model to the owner. It is
  admissible, but it is not the present KKT step.
- How robust the rediscretised multigrid is on real paintings (rims, strata relief, partial wetting with a
  molecular ℓ). Tested only on smooth discs with 5% noise.
- Several densities and yield stress in the multigrid: not built.
- The diffusion ray with R6 and R7, and with S = 11 palettes: not built. Its t* in fronts of two paints meeting:
  not measured.
- That every map is intensive (depends on φ, not h), which keying on composition assumes. Unverified for `_voids`
  and `surface`.
- The fused cost of the numpy parts (the 1.5 s quiet step): an estimate, not measured.
- Megapixel present-code moving steps: not run (above 10 minutes by extrapolation).
- The look estimates: no look prototype was built.
- Whether the server's other hardware has a usable second core: this VM shows none.

---

## 8. The GPU and a later compiled port (short, all estimates unless marked)

**RTX 2070 Super.** Spec values, read on a third-party spec page (sysrqmts.com), not measured: 9.06 TFLOPS FP32,
283 GFLOPS FP64 (1:32), 8 GB, 448 GB/s.

- **Stencils and multigrid are bandwidth-bound.** A V-cycle's 400 B per cell (**computed**, §1.1) at about 300 GB/s
  achieved (assumed: two thirds of peak) is 1 ms per V-cycle at 786k cells, so 8 cycles about 10 ms in f32, and
  about 20 ms in f64 at twice the bytes (still bandwidth-bound: an FP64 rate of 283 GFLOPS covers 200 flops per
  cell at 786k cells in 0.6 ms).
- **Per-cell local solves and maps with logs** are compute-heavy, and FP64 at 1:32 hurts there. The quiet step's
  roughly 1 GFLOP (§1.1) takes about 4 ms at FP64 peak, or about 20 ms realistically (assumed: 20% of peak).
- **A look's 17 GFLOP:** about 10 ms in f32 at 20% of peak, or about 0.3 s in f64.
- **What fits:** state 100 to 200 MB, a look's cached base 0.2 to 0.5 GB, all comfortably inside 8 GB.
- **What does not map:** SuperLU, the active set with direct solves, Python orchestration per substep (kernel
  launches cost about 5 to 10 us each, so a step must be tens of fused kernels, not thousands of numpy calls).

**A compiled core on a modern 8-core desktop CPU** (**estimated**; assumed 40 to 60 GB/s memory bandwidth and
about 4 GFLOP/s per core scalar f64 with logs, about 8 with SIMD):
- quiet step at megapixel: 1.6 GB of traffic and 1 GFLOP, so 30 to 50 ms;
- multigrid flow on 400k cells at 8 V-cycles: 1.3 GB, about 30 ms;
- a full look (17 GFLOP): about 0.3 s.

That is f64 throughout, with the contract unchanged.

**Double precision and the contract.**
- *Verified from documentation read:* wgpu's `SHADER_F64` feature "enables 64-bit floating point types in SPIR-V
  shaders", is "a native only feature", and is listed for Vulkan only, with the warning that f64 is "frequently
  between 16 and 64 times slower"
  ([wgpu-types FeaturesWGPU](https://docs.rs/wgpu-types/latest/wgpu_types/struct.FeaturesWGPU.html)).
- *Verified from documentation read:* WGSL's core language has f32 and f16 (as an extension) but no f64. A 2022
  thread on the W3C GPU list says "Float64 is likely eventually ... There are no concrete plans", notes Metal's
  lack of f64 as a blocker, and warns that WGSL floats are IEEE in format "but not strictly speaking in
  accuracy/behavior" ([public-gpu, April 2022](https://lists.w3.org/Archives/Public/public-gpu/2022Apr/0005.html)).
  The WGSL specification page itself could not be fetched (the request for https://www.w3.org/TR/WGSL/ was not
  answered).
- *Verified from documentation read:* naga's validator requires a FLOAT64 capability for 64-bit literals
  ([naga changelog](https://rust-digger.code-maven.com/src/naga-23.0.0/CHANGELOG.md)).
- *Recalled, not verified:* whether naga's WGSL front end parses `f64` when the feature is on; CUDA has full IEEE
  double on all NVIDIA GPUs (at 1:32 on Turing consumer parts).

**What the contract means under f32.**
- The ledgers' 1e-12 closures cannot hold in f32 arithmetic (eps is 6e-8).
- The designs that keep the contract: f32 fields for coefficients and the layer algebra; f64 (on the GPU via CUDA,
  or on the CPU) for q, held, every ledger sum and the ray's sums; reductions exactly rounded or in fixed order
  (binned or integer accumulation is reproducible across devices); and per-backend tolerances stated in the
  contract wherever f32 enters (looks).
- Double-single arithmetic in shaders is possible but costs about 10 to 20 flops per operation (recalled).
- On a GPU, bit-for-bit means the same kernel with the same reduction order on the same device. GPU against CPU is
  a tolerance, and so is CUDA against wgpu: fused multiply-add contraction and transcendental implementations
  differ.

**Backend choice.** For Python-first, CuPy behind array-API-style code fails here: numpy temporaries are the
problem on the GPU too, at a launch per operation. numba's CUDA target or hand-written CUDA kernels called from
Python keep the fused-kernel structure that §5 shows the CPU needs anyway. JAX's and PyTorch's reductions are not
reproducible by default.

**Making a later Rust port easier, from a Python-first design.** The same choices that make the Python kernel
fast make the port straightforward:
- matrix-free stencil operators on [H, W] arrays, with no dependence on scipy's sparse direct solvers (keep
  SuperLU only as the `exact` oracle);
- per-cell kernels with explicit inputs and outputs (the numba kernels are already the shape of Rust functions);
- palette-sparse composition with fixed-size per-cell records;
- every reduction in a stated order;
- receipts computed from per-substep scalars.

What would make the port harder: business logic in numpy broadcasting idioms, and maps returning `Derived` wrappers
around full panels. The evidence wrapper should stay in Python around compiled kernels. Rust against Go: Rust has
LLVM codegen with explicit SIMD, rayon, PyO3 bindings for running the Python obligation tests against it, wgpu and
CUDA bindings, and no garbage collector. Go's compiler does little auto-vectorisation, has no portable SIMD, has
weaker GPU access, and its garbage collector and cgo overhead are poorly suited to millions of small per-cell
records. For this workload Rust is the better fit (an opinion from recalled language properties, not a
measurement).

---

## 9. Stage 1 as built (dynamics owner, 2026-11-15/16)

What changed in `goo/world` (the arguments are in docs/DYNAMICS.md §4.2a, §4.3 and §9a):

- **1a, advance.** The step reports `max relative change`. advance scales its error estimate by min(1, change/0.1),
  halves the next step above a change of 0.25, and doubles only below it.
- **1b, rest memo.** A step whose every substep left on an old-state-only exit is recalled until the state version,
  q's or held's identity, the rate, the solver or the window changes.
- **1c, diffusion by the ray.** D_num ≥ 0 by construction. Newton is the fallback on entry, on non-finite potentials,
  and when t < 0.9.
- **1d, local Newton update in numba** (`goo/world/kernels.py`). Per cell, no threads, fastmath off, cached. For one
  volatile it adds a bracket safeguard.

**After the substance maps' rework:**
- the drain rule extended to cells of several substances;
- `mixing_potentials` and `mixing_hvp` adopted where no Hessian is needed;
- `mixing_hvp` called in blocks of 32,768 cells, to avoid a defect in `goo.substance._freeenergy.hvp`: the name `sel`
  is reused inside its loop over chunks, giving a wrong index or a 42 GB allocation above 65,536 distinct
  compositions.

**Measured on 96 x 128 and 48 x 64** (this machine, old maps unless said):
- **Drying the paste for an hour:** 5,516 → 144 steps (96 x 128, old maps); 158 steps with the new maps
  (48 x 64, test). The new maps' film dries fast and then vitrifies; settle() retires all of it.
- **A resting paste:** 231 ms computed, 0.2 ms recalled (192 x 256).
- **Diffusion by the ray:** 0.44 → 0.10 s at 49k cells; it matches the fine reference better than implicit Euler
  (§2.4).
- **Local:** 0.86 → 0.77 s at 197k cells, bit-identical. The world's share is small: the maps are 57% of the local
  substep.

**Measured on 1024 x 768 (786,432 cells), library maps of 2026-11-15**, while another agent's tests held this
machine's effective single core at times, so the wall times are upper bounds:

| | before stage 1 | after stage 1 |
|---|---|---|
| quiet step (paste, evaporating) | 20.7 s, 2.98 GB | 13.7 to 14.4 s, 3.07 GB (the first step, from the initial condition, falls back to Newton: 41 s) |
| share in `goo.substance` maps (profile at 197k cells) | | 1.53 of 2.85 s (`_freeenergy.evaluate`), plus 0.26 s in exchange |
| old maps with particle mobility 0 (`bench/noparticle.py`) | 7.5 s | 6.3 to 6.9 s; an hour of drying in about 148 steps, about 985 s of wall time |
| drying a full-panel 30 um paste until settle() retires it | | see PROGRESS.md and the report (run in progress at the time of writing) |

**Reading.** With the real maps the world's own arithmetic is now the smaller part of a quiet step. What remains:
- the maps' evaluations: 7 full `mixing`, about 6 `mixing_potentials` and 3 `mixing_hvp` calls per step;
- the Newton fallback on a step whose ray falls short.

The next speed-up of the quiet step lies in the maps (a compiled per-cell kernel the world can call, signature in
the report) and in fewer map calls per step (the substeps re-evaluate at states another substep already evaluated).
