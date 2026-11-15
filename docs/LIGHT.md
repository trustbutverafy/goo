# goo.light: derivations, protocol, reservations

This is the light owner's account of SPEC §3 (the light ledger), §5 (what a look reads) and §7 (light).
The code is `goo/light/`; the tests are `tests/light/`. Every equation below is tested, and the test is
named in brackets.

## 1. A homogeneous film

A film of thickness `d`, absorption `K` and scattering `S` (both two-flux coefficients in 1/m, `S` with
the asymmetry already taken, SPEC §4.3) carries a diffuse flux `i` going down and `j` going up. With `z`
pointing down, Kubelka and Munk's equations are

```text
di/dz = −(K + S) i + S j            dj/dz = (K + S) j − S i
```

The film's operator is the response to unit flux entering from above with nothing from below:
`i(0) = 1`, `j(d) = 0`; then `R = j(0)` and `T = i(d)`. The film is the same seen from below, so
`R' = R`, `T' = T`. The system's matrix has eigenvalues `±α`, `α² = K(K + 2S)`. Writing `x = αd`,

```text
D = cosh x + (K + S) d shc x            shc x = sinh x / x
T = 1 / D                               R = S d shc x / D
```

`cosh x` and `shc x` are entire functions of `x²`, so these expressions are valid for every sign of
`x²`; `x² < 0` (cosines) occurs only with gain, `−2S < K < 0`. For `K ≥ 0` the code divides through by
`cosh x`, `u = d tanh(x)/x`:

```text
R = S u / (1 + (K + S) u)               T = sech x / (1 + (K + S) u)
```

which is finite and loses no digits in any limit [test_bulk.py::test_limits_are_exact_or_stable,
test_no_warnings_and_finite_over_extremes]:

- `d → 0`: `u = 0`, `R = 0`, `T = 1` exactly: the empty film is the identity.
- `S → 0`: `R = 0`, `T = sech x / (1 + tanh x) = e^{−Kd}`: Beer–Lambert.
- `K → 0`: `x → 0`, `tanh(x)/x` by its series below `x = 1e−4`; `R = Sd/(1 + Sd)`, `T = 1/(1 + Sd)`.
- `d → ∞`: `T = 0`, `R = R∞ = S / (K + S + α) = 1 + K/S − √((K/S)² + 2K/S)` (taken directly when
  `d = inf`, with `R∞ = 1` for `K = 0 < S` and the identity for `K = S = 0`).

This is Kubelka's closed form in other variables: with `a = 1 + K/S`, `b = √(a² − 1)`, `bSd = x`, and
`R = sinh(bSd) / (a sinh + b cosh)`, `T = b / (a sinh + b cosh)` (Kubelka 1948). G4 tests the code
against that form written independently, and against Kubelka's layer over a ground of reflectance `Rg`,
`R = (1 − Rg(a − b coth bSd)) / (a − Rg + b coth bSd)`, to 1e-13
[test_G4_scalar_case_reproduces_the_kubelka_munk_closed_form].

## 2. The star product

A layer maps the fluxes entering it (`a` from above, `b` from below) to the fluxes leaving it:

```text
up out of the top      = R a + T' b
down out of the bottom = T a + R' b
```

with `R, T, R', T'` of shape `(..., n, n)` (leading axes bands and cells, last two flux channels; `n = 1`
on this rung). Put layer 1 on layer 2 and call `d` and `u` the fluxes between them. With input `a`
from above, `d = T₁ a + R₁' u` and `u = R₂ d`, so `d = (1 − R₁' R₂)⁻¹ T₁ a`. Eliminating `d` and `u`,
and doing the same for input from below:

```text
X = (1 − R₁' R₂)⁻¹                      Y = (1 − R₂ R₁')⁻¹
R₁₂  = R₁ + T₁' R₂ X T₁                  T₁₂  = T₂ X T₁
R₁₂' = R₂' + T₂ R₁' Y T₂'                T₁₂' = T₁' Y T₂'
```

For scalars this is SPEC §7's `R₁₂ = R₁ + T₁ R₂ T₁′ (1 − R₁′ R₂)⁻¹`, `T₁₂ = T₂ (1 − R₁′ R₂)⁻¹ T₁`.
It is Redheffer's star product (1959).

**Monoid.** The empty film `E = (0, 1, 0, 1)` gives `X = Y = 1` and the products reduce to the other
factor without arithmetic, so `E ⋆ A = A ⋆ E = A` bit for bit. Associativity: `(A ⋆ B) ⋆ C` and
`A ⋆ (B ⋆ C)` are two orders of eliminating the internal fluxes of the same linear system (three layers,
two internal interfaces); where the system has a unique solution, both orders give it. In floating point
they agree to rounding: 1e-12 is required, 1e-13 is met for random passive layers of 1, 2 and 3 channels
and for films [test_G2_monoid_associative_with_the_empty_film_as_identity,
test_result_is_independent_of_bracketing]. The product is not commutative
[test_G2_monoid_is_not_commutative]; §G7 gives the commutator.

**Passive operators are closed under ⋆.** Call a layer passive if its entries are non-negative and each
column of `[R; T]` and of `[T'; R']` sums to at most 1: for every non-negative input the power out is at
most the power in. (For non-negative operators the worst input is one channel, so the column sums are
the whole condition.) If both factors are passive and the cavity is not a pair of lossless perfect
mirrors, the spectral radius of `R₁' R₂` is below 1 (`ρ ≤ ‖R₁'‖₁ ‖R₂‖₁ ≤ 1`, with equality only if both
reflect everything), so `X = Σₖ (R₁' R₂)ᵏ` converges and is non-negative: every product is a sum over
paths of products of non-negative entries, hence non-negative. Power: for unit input from above, what
the composite loses is `1 − Σ(R₁₂ + T₁₂) = A₁ + A₂`, where `Aᵢ` is the loss of layer `i` for the
(non-negative) fluxes that enter it; each is at least 0 because each layer is passive. The same holds
from below. So the composite is passive [test_G1_passivity_power_out_never_exceeds_power_in]. With
`R₁' = R₂ = 1` (two lossless mirrors facing each other) the product is undefined; no passive film reaches
it (internal surface reflectance is below 1 and a film with `R = 1` needs `K = 0` and infinite depth).

**A stack with its internal fluxes** (`solve`). For layers `L₀ … L_{N−1}` top first, the code forms the
products from below, `D_j = L_j ⋆ … ⋆ L_{N−1}`, and from above, `U_j = L₀ ⋆ … ⋆ L_{j−1}`. At interface
`j` the down flux is `d_j = (1 − U_j' R_{D_j})⁻¹ T_{U_j} a₀` and the up flux `u_j = R_{D_j} d_j`. Each
layer's loss is computed from its own operator and the fluxes entering it,
`A_j = (1 − colsum R_j − colsum T_j)·d_j + (1 − colsum R_j' − colsum T_j')·u_{j+1}`, and separately as
the difference of net flux across it; they agree to 1e-14
[test_solve_keeps_internal_fluxes_and_attributes_every_loss]. The ledger books `A_j`, not a remainder.

## 3. Fresnel and its hemispherical integrals

Light in a medium of index `n₁` meets one of `n₂ = m n₁` at incidence cosine `μ`. With
`μ_t = √(1 − (1 − μ²)/m²)`, unpolarised,

```text
F(m, μ) = ½ (r_s² + r_p²)        r_s = (μ − m μ_t)/(μ + m μ_t)        r_p = (m μ − μ_t)/(m μ + μ_t)
```

and `F = 1` past the critical angle (`m < 1`, `μ < μ_c = √(1 − m²)`). [test_fresnel_matches_snell_form]

**Diffuse (Lambertian) flux** arrives with power `2μ dμ`, so its reflectance is

```text
r(m) = ∫₀¹ F(m, μ) 2μ dμ.
```

For `m > 1` (from the lower index) this integral has a closed form (Walsh 1926; Stern 1964):

```text
r_e(n) = ½ + (n − 1)(3n + 1) / (6(n + 1)²) + n²(n² − 1)² / (n² + 1)³ · ln((n − 1)/(n + 1))
           − 2n³(n² + 2n − 1) / ((n² + 1)(n⁴ − 1)) + 8n⁴(n⁴ + 1) / ((n² + 1)(n⁴ − 1)²) · ln n
```

Its terms grow like `1/(n − 1)²` and cancel, so near `n = 1` it loses digits (1e-11 at `n − 1 = 1e−3`,
wrong at 1e-8). Below `n − 1 = 0.1` the code uses instead the series of the same function in `d = n − 1`,
`r_e = Σ_{k=1}^{18} (p_k + q_k ln d) dᵏ`, `r_e = d/3 + (19/24 − ln2/2 + ½ ln d) d² + …`, whose coefficients
(rational numbers and multiples of ln 2, in `goo/light/interface.py`) were obtained by expanding the closed
form symbolically. At the switch the two agree to 2e-16; the series' truncation is below 1e-18 there. Both
are compared with direct adaptive quadrature of Fresnel's equations written in angles, to 2e-13
[test_external_diffuse_is_the_hemispherical_integral, test_series_and_closed_form_agree_where_they_meet].

**From the denser side** (`m < 1`): across a lossless interface radiance divided by `n²` is invariant, so
for Lambertian flux `n₁² t₁₂ = n₂² t₂₁`, and

```text
r(m) = 1 − m² (1 − r_e(1/m))            (m < 1)
```

which contains total internal reflection. It is checked against quadrature with the critical angle as a
breakpoint, not assumed [test_internal_diffuse_with_total_internal_reflection_is_the_integral,
test_reciprocity_n2_t_is_symmetric]. `r(1) = 0` exactly, by branch on the value `m = 1` (an exact limit,
not a constant).

## 4. Surfaces

**A boundary is a layer.** Between `n_above` and `n_below`, for diffuse flux on both sides:

```text
R = r(n_below / n_above)    T = 1 − R    R' = r(n_above / n_below)    T' = 1 − R'
```

It is lossless (`R + T = 1` on each side) and composes by ⋆ like any film. Equal indices give the
identity exactly [test_boundary_is_lossless_and_equal_indices_are_the_identity].

**Entry of the illuminant.** The light's geometry decides only how the incident beam crosses the top
surface (`entry`). Diffuse light: `R = r(m)` with `m = n_surface/n_ambient`. Collimated at θ:
`R = F(m, cos θ)`. In both, `T = 1 − R`. What enters is diffuse inside (Saunderson's assumption, the
first rung's; §8). The top surface's `R'`, `T'` (diffuse light from inside) are the boundary's, whatever
the light.

**Roughness divides reflected power, it adds none.** A surface of rms height σ reflects specularly the
fraction `g(μ) = exp(−(4π n σ μ / λ)²)` of what it reflects at incidence cosine μ (Davies 1954; Bennett
and Porteus 1961: the total integrated scatter of a Gaussian surface, `n` the index of the incident
side, λ the band centre in vacuum); the rest is reflected diffusely. Collimated:
`specular = g(cos θ) F`. Diffuse:

```text
specular = ∫₀¹ F(m, μ) e^{−aμ²} 2μ dμ,      diffuse = r(m) − specular,      a = (4π n σ / λ)²
```

by 32-point Gauss–Legendre, with `μ − μ_c = (U − μ_c) t²` to remove the square root at the critical
angle, the part below `μ_c` (where `F = 1`) in closed form, and each part integrated from a non-negative
integrand: for `a ≤ 1` the diffuse part (`1 − e^{−aμ²}`), for `a > 1` the specular part, on
`[0, min(1, 6/√a)]`. Against adaptive quadrature: 1e-10 asserted, 1.5e-12 measured
[test_roughness_divides_the_reflected_power]. Total reflectance does not depend on σ
[test_roughness_moves_surface_reflection_from_specular_to_diffuse]. On this rung roughness acts only at
the top surface: inside, all flux is already diffuse.

## 5. A look

`look(stack, bands, light, optical)` builds the stack's operators (`operators`), top first:

```text
surface        ambient (n = 1 by default, `n_ambient`) | n_top of the topmost film present, else the ground
film[k]        boundary(n_above, n_top[k]) ⋆ bulk(K[k], S[k], h[k])          for each film, top film first
ground         boundary(n_above, n_top[ground]) ⋆ bulk(K_g, S_g, Ground.depth)
```

`n_above` starts at the surface's index and, going down, becomes the host index of each film present.
Where a film is empty (`h = 0`) its sublayer is the identity and `n_above` is unchanged, so an empty film
changes nothing, wherever it sits, and its optical values there are not read
[test_an_empty_film_is_the_identity_and_its_values_are_not_read]. The topmost present film's own
top boundary is then `n_top | n_top`: the identity, since the surface layer is that interface. The film's
top interface and bulk are one sublayer: the interface is lossless, so this does not change attribution.
The top layer is then replaced by its version for the light (§4), and `solve` gives the fluxes.

Outputs, as fractions of the incident power per band and cell: `reflectance [H, W, B] = specular +
diffuse`; `absorbed [K, H, W, B]`, bottom first like `stack.films`; `ground [H, W, B]`, what the ground's
body absorbs; `transmitted [H, W, B]`, what leaves through the back of a ground of finite depth (zero to
rounding for an opaque ground, exactly zero for `depth = inf`)
[test_the_ground_splits_what_it_takes_into_absorbed_and_transmitted].

**The light ledger** (SPEC §3), per band, in panel means: `incident → reflected.specular`,
`incident → reflected.diffuse`, `incident → absorbed[k]`, `incident → absorbed.ground`,
`incident → transmitted`; with gain, `debt[k] → incident` (§6); past a threshold,
`incident → unresolved` (§6). The closure residual is **relative to the power entering**, incident plus
debt: `max over cells and bands |(1 + debt) − (reflected + absorbed + transmitted + unresolved)| / (1 + debt)`,
each term computed as in §2 (the reflectance from the composite, the absorptions from each layer's own
operator), together with the ledger's own `|balance(incident) + 1| / (1 + mean debt)` per band. Without
gain it is the absolute residual in units of the incident power. It is at most 1e-12 for random admissible
stacks with indices,
roughness, empty cells, finite and semi-infinite grounds, under diffuse and collimated light; measured
≤ 8e-16 [test_G3_ledger_closes_for_random_admissible_stacks].

**A look loads nothing.** It reads `Film.h`, `Ground.depth`, `Ground.z.shape` and what `optical` returns,
and writes nothing: read-only inputs work and are unchanged after
[test_a_look_never_modifies_the_stack_and_takes_read_only_arrays]. The light enters only at the top
surface's entry; the operators do not depend on it, and the power spectrum does not enter the fractions
[test_a_different_light_changes_nothing_in_the_operators].

## 6. Impossible substances

`K < 0` is optical gain: the film is not passive (it fails `herglotz`, SPEC §4.2). Nothing is clipped.

**Below threshold** the film's operator is §1's (both signs of `x²`; for `K + S < 0`, `x > 1` the
denominator is formed as `D = (e^{−x}(α − K − S) − e^{x} S²/(α − K − S)) / (2α)`, using the identity
`1 + (K + S)/α = −S²/(α(α − K − S))`, so nothing cancels), tested against `expm` of the two-flux matrix
to 1e-10 [test_gain_below_threshold_solves_the_two_flux_equations]. The film's net absorption is negative;
`look` books the power it adds as `debt[k] → incident`, so the ledger closes as
`incident + debt = reflected + absorbed`, and the receipt carries an `impossible` flag naming the film
and the law [test_gain_below_threshold_books_its_excess_to_debt].

**Threshold.** A film's steady state exists while `D > 0` on `[0, d]`. The first zero of `D` is, for
`x² < 0` (`y = βd`, `β² = −K(K + 2S)`), at `y* = atan2(β, −(K + S)) ∈ (0, π)`; for `x² > 0` with
`K + S < 0`, at `x* = artanh(α / |K + S|)` (none for pure gain, `S = 0`: a lone amplifier, `T = e^{|K|d}`);
for `x² = 0`, at `(K + S) d = −1`. A semi-infinite gain medium never settles. Tested against a scan of
`D` [test_lasing_threshold_is_where_the_two_flux_denominator_first_vanishes]. A stack can also lase
without any film past its own threshold: an amplifier between a reflecting ground and the top surface's
internal reflectance is a cavity, and the product diverges where a round trip's gain reaches 1. The code
marks a cell and band **diverged** where any film is past threshold or any round-trip denominator met
in any product (`1 − R₁'R₂`, or `1 − ρ(R₁'R₂)` on n channels) is ≤ 0. Every partial cavity is checked:
with non-negative operators a sub-cavity's round trip bounds the whole one's from below.

**The defined value past threshold.** There is no steady state, so the look returns, in diverged cells
and bands only: reflectance, specular, diffuse, absorbed, ground and debt all 0; `unresolved = 1`;
`diverged = True`. The ledger books `incident → unresolved`, the closure holds, and the receipt carries
`impossible` on `stack` with the count. No NaN, no infinity
[test_past_the_lasing_threshold_a_look_is_defined_and_flagged]. A black pixel is not a claim about how a
laser looks: it says the question has no stationary answer in this model (a model with gain saturation
would have one; SPEC has no such law).

## G7. The commutator, in closed form

Two symmetric scalar layers (films) `A = (R_a, T_a)`, `B = (R_b, T_b)` over black:

```text
R(A over B) = R_a + T_a² R_b / (1 − R_a R_b)          R(B over A) = R_b + T_b² R_a / (1 − R_a R_b)
```

Over a common denominator, with `Δ_x = R_x² − T_x²`,

```text
[A, B] ≡ R(A over B) − R(B over A) = ( R_a (1 + R_b² − T_b²) − R_b (1 + R_a² − T_a²) ) / (1 − R_a R_b)
```

Consequences: a clear glaze (`R_a = 0`) gives `[A, B] = −R_b (1 − T_a²)`: body colour over a glaze shows
`R_b`, glaze over body colour `T_a² R_b`. Lossless layers (`R + T = 1`, so `1 + R² − T² = 2R`) commute:
order matters only through absorption. Over a ground of reflectance `R_g`: the two composites have the same
transmission `T_ab = T_a T_b / (1 − R_a R_b)`, and each one's reflectance from below is the other's from
above (`R'(A over B) = R(B over A)`). Then

```text
R(A over B over G) − R(B over A over G) = [A, B] · ( 1 − T_ab² R_g² / ((1 − R(A over B) R_g)(1 − R(B over A) R_g)) )
```

The test computes `R_a, T_a` by Beer–Lambert, `R_b, T_b` and `R_g = R∞` by Kubelka's hyperbolic form, the
formula above by hand, and compares a look of glaze over body colour with the reverse, over a white
ground, to 1e-14 [test_G7_order_a_glaze_over_body_colour_differs_from_the_reverse_by_the_commutator].

## 7. The Optical protocol

`goo.light.Optical`, passed to every entry point (`operators`, `look`). `goo.light` reads nothing else
about matter, except `Film.h` (thickness, m), `Ground.depth` (m) and the cell shape.

```text
optical(film, bands)          -> (K, S, n_host)   [B, H, W] each; K, S in 1/m (two-flux, S with asymmetry taken)
surface(film, bands)          -> (n_top, sigma)   n_top [B, H, W]; sigma [H, W] in m, rms height
ground_optical(ground, bands) -> (K, S, n_host)   [B, H, W]; the ground's body over its depth Ground.depth
ground_surface(ground, bands) -> (n_top, sigma)   the ground's top interface
```

Indices are real and relative to vacuum. Any item may be an array, a `Derived` or a `Bounded`, or the
whole return one `Derived` of the tuple; anything broadcastable to the shape is accepted. A `Derived` out of
its domain adds `out_of_domain` to the receipt; a `Bounded` is collapsed at its midpoint and adds
`bounds_collapsed` [test_derived_and_bounded_returns_are_unwrapped_and_flagged]. Values in cells where a
film is empty are not read. The analytic implementations in `tests/light/analytic.py` (a mixture rule that
gives grey absorbers, pure scatterers and gain media, and particles whose scattering vanishes at index
match) are fixtures, not physics.

G6 is a property of the map and the operator together: a map whose `S` vanishes at index match (any law
of the contrast) gives `S = 0`, and then a look shows no diffuse reflectance from the body at all: only
the smooth surface's `r_e(1.5)`, with Beer–Lambert absorption in the film, exactly
[test_G6_matched_index_a_particle_in_a_host_of_its_own_index_scatters_nothing].

## 8. Reserved, and how each would fit

The interface is the layer of `(..., n, n)` operators and ⋆; each extension is a change of `n` or of the
leading axes, and each must reduce to this rung exactly.

- **More flux channels (angles).** Discrete ordinates: `n` angular streams each way, `R, T` the n×n
  transfer between them, Fresnel per stream (no hemispherical integral), roughness as a kernel between
  streams. The collimated beam is then one stream and stays collimated inside (four-flux is `n = 2`:
  collimated and diffuse). `star` and `solve` already run for any `n`; two uncoupled identical channels give
  the one-channel result to 1e-14 [test_two_uncoupled_identical_channels_equal_one_channel,
  test_the_n_channel_path_in_a_look_equals_one_channel]. What is missing is the film operator for `n > 1`
  (the matrix exponential of the n-stream transport matrix) and its Optical inputs (a phase function, not
  only `S`).
- **Polarisation.** Channels `(s, p)` per stream: Fresnel's `r_s²`, `r_p²` instead of their mean; Stokes
  intensities need 4 per stream and Mueller blocks, still `(..., n, n)`.
- **Coherent thin films (interference pigments).** Amplitudes, not intensities: complex `r, t` per layer,
  the same star product (Redheffer's, for amplitude scattering matrices), passivity as contraction
  (`‖S‖₂ ≤ 1`) instead of sub-stochasticity, and an Optical that returns complex indices and thickness.
  Intensities are the incoherent limit (phase averaged), which is this rung.
- **Coupled bands (fluorescence).** Make the channel axis band × stream and let a film's operator carry a
  kernel from band `b` to band `b'` only for `λ_b' ≥ λ_b` (never raises photon energy). The ledger moves to
  energy and books the Stokes loss to the film (SPEC §3). Then `Light.power` enters: today it does not,
  since every output is a fraction of the incident power per band.

## 9. Assumptions (evidence `assumed`)

- Two-flux, diffuse inside: collimated light becomes diffuse on entering (Saunderson). The specular share is
  then only the top surface's.
- Each interface's index is the film's `n_top` (as the map gives it); the bulk below it is treated as
  being in that medium when diffuse light returns to that interface. A film whose host index differs from
  its top-interface index (an underbound film) is modelled only as far as the map's two numbers say.
- Interfaces between films use `n_host` of the film above and `n_top` of the film below. Inner interfaces
  are smooth in effect: their roughness has no channel to act on.
- Roughness by the scalar total-integrated-scatter law, valid for σ ≪ λ; for rougher surfaces it sends all
  reflected power to diffuse, which is its correct limit but not a model of shadowing or multiple scatter.
  Roughness does not change the Fresnel total.
- The ambient medium's index is 1 unless `n_ambient` says otherwise.
- What passes through `Ground.depth` leaves through the panel's back (booked to `transmitted`) and does
  not come back: the back is non-reflecting.
- Bands independent; real indices (absorption enters only through `K`).

## 10. Numbers

```text
G1   power out ≤ in: operators 1e-15 (single), 1e-14 (products of 6); look absorptions ≥ −1e-15
G2   associativity 1e-12 required, 1e-13 asserted; identity exact (bit for bit)
G3   closure 1e-12 required and asserted; measured ≤ 8e-16
G4   against Kubelka's hyperbolic form and Kubelka over a ground: 1e-13
G5   thick film over black, white, clear: equal to 1e-14; R∞ to rtol 1e-14
G6   S = 0 exactly; diffuse = 0 exactly; specular = r_e(1.5) exactly; Beer-Lambert to rtol 1e-13
G7   commutator, bare 1e-15; in a look over a ground 1e-14
```

How a look is computed. The protocol is called once per film and once for the ground. Cells whose inputs
(every value the protocol returned, and `h`, with empty cells' values replaced by fixed ones) are bit-identical
are solved once: a 64-bit hash of the inputs' bits groups them, and every input of every cell is then compared
exactly with its group's representative, so a collision disables the grouping and never merges different
cells [test_dedup_never_merges_different_cells]. Grouping is skipped when it would not remove 40% of the
cells. The distinct cells are solved in blocks of 32768 values (bands × cells), whose temporaries stay in
cache, on up to 2 threads, and everything elementwise afterwards is done on the distinct cells before each
output is expanded once. Each cell's arithmetic is the same whatever the grouping, block or thread, and the
panel means are taken in one fixed order, so the Look and the receipt's digest are bit for bit the same with
and without grouping, blocking or threads [test_dedup_blocks_and_threads_change_nothing_bit_for_bit].

Timing (`GOO_BENCH=1 python -m pytest -q -s tests/light/test_timing.py`, 2 cores, 400 × 300 cells,
16 bands, 6 films):

```text
                                         before      after
every cell distinct, diffuse light       5.8 s       2.9-3.3 s
every cell distinct, collimated light    4.0 s       2.3-2.5 s
painting-like (strokes of uniform paint) 3.3 s       0.6-0.7 s    (of which ~0.3 s is the test fixture's map)
peak memory                              2.2 GB      1.5 GB       (most of it the protocol's own arrays)
```

Not taken: the diffuse-light rough-surface quadrature is about a quarter of the all-distinct time. A table in
(m, a) with interpolation would remove most of it, but at a tested interpolation error of order 1e-8 to 1e-10,
not 1e-12, against the present results.
