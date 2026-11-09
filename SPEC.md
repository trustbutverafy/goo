# goo — specification

This is the contract between goo's parts. Code that breaks it is wrong even if the picture looks right.

## 1. Domains

| Domain | Module | Knows | May change |
|---|---|---|---|
| **World** | `goo.engine.Canvas` | everything: species volumes, strata, cure, the water in the ground, the ledger | its own state, only through the laws (§4) |
| **Tools** | `goo.brush.Brush`, `Stylus` | their own load and footprint; the wet layer they touch | the tool fields `U`, `V` (velocities) and the wet layer, only by exchange with their load |
| **Optics** | `Canvas.reflectance()` | species volumes per layer | nothing (pure function) |
| **Sensors** | `goo.sensors.*` | a `Surface`: height (mm), reflectance per band, roughness (µm), gloss | nothing in the World |
| **Painter** | `goo.painter`, `goo.palette` | colour seen on the canvas (`reflectance`), its motif, its paints | the World only through Tools, `fix`, `wait`, `step` |
| **Paintings, studies** | `paintings/`, `studies/` | whatever they are given | the World only through Tools, `Canvas.add`, `fix`, `wait`, `step` |

Two rules carry most of the weight:

- **No synthesis.** Every visible effect — a wet edge, a backrun, a bristle groove, a burr, a slump, a crack in a reflectance image — must come out of the laws acting on the state. A painter script may choose *where a brush goes and what it carries*; it may never write colour or height into the World directly. (`Canvas.add` exists for test setups and is booked like a brush load.)
- **Sensors see the surface, not the paint.** A sensor is constructed from a `Surface`. It must not read `q`, `strata`, `species` or the ledger. Ground truth for scoring a sensor is computed beside the measurement and never fed back into it (`tests/test_sensors.py::test_sensor_sees_only_the_surface`).

## 2. Units

| Quantity | Unit |
|---|---|
| Cell | 0.25 mm square (`goo.sensors.lidar.CELL`) |
| Volume per cell | thickness in units of 0.25 mm; a volume of 1 in one cell is 0.0156 µL |
| Time | one *tick* = one call to `Canvas.step()`; a brush touch is one tick |
| Bands | R, G, B and N (≈1.5 µm, where reflectography looks) |
| Optics gain | `reflectance(gain=10)`: K and S in `PIGMENTS` are per 0.025 mm of pure pigment |

Realistic quantities, for calibration: a paint layer is 0.1–0.4 units (25–100 µm); impasto 2–8 units (0.5–2 mm); a loaded filbert holds ~6000 units (0.1 mL); a wash mop ~12000.

## 3. Invariants (tested in `tests/test_engine.py`)

- **I1 Conservation.** For every species *s*: `loaded[s] = on_canvas[s] + in_brushes[s]`, plus, for water only, `+ evaporated + in_ground + ground_evaporated`. `on_canvas` counts the wet layer and every stratum. Recharging a brush moves its remainder to `rinsed` and out of `loaded`. Relative residual ≤ 1e-9 after any session.
- **I2 Positivity.** `q ≥ 0`, `store ≥ 0` everywhere, always. Positivity is guaranteed by construction (per-species outflow limiting, §4.6), not by clamping; the `np.maximum(q, 0)` after an update only removes round-off.
- **I3 Exact zero.** Every law has a switch in `Canvas.L` (`flow, drag, diffusion, evaporation, absorption, curing`). With all of them 0 and no tool touching, `step()` leaves `q` bit-for-bit unchanged.
- **I4 Determinism.** The same session (same seeds, same calls) gives the same arrays.
- **I5 Strata are immutable.** Only `fix()` creates a stratum; nothing modifies one afterwards. Tools act on the wet layer only.
- **I7 Two clocks.** Tools act every tick, in a window around the touch; the slow laws (§4.2, 4.4–4.6) run over the elapsed ticks every `L["lazy"]` ticks (`flush()`); `fix`, `wait` flush first. `lazy = 1` runs every law every tick. `flow_sub`, `diff_sub` cap substeps per tick for leveling and colour diffusion; both are stable at any cap (face caps, §4.2, 4.4), so the caps trade the speed of very fast leveling for time, never stability or conservation.
- **I6 Stability without loss.** `step()` chooses its own substeps from every process's stability limit; `capped` counts steps that hit `max_sub` and should stay 0 in normal use. A step may be slow; it may not blow up, go negative, or lose paint.

## 4. The World's laws

State: `q[species, H, W]` (wet layer; species = `CARRIERS` (`water`, `oil`) + the canvas's pigments), `strata` (list of frozen `q`-shaped layers, bottom first), ground relief `g`, base `zb = g + Σ strata`, oil cure `kappa ∈ [0, 1]`, water in the ground `store`.

Derived: thickness `h = Σ_s q_s`, solid fraction `φ = Σ_pigments q / h`.

1. **Rheology.** Krieger–Dougherty viscosity `η = η0 (1 − φ/φm)^-2 (1 + cure_stiff·κ)`; yield stress `τ = τ0 ((φ − φy)+ / (φm − φ))^1.5 (1 + cure_yield·κ²) + tau_cure·κ²`.
2. **Leveling.** Bingham thin-film flux down the gradient of `P = G(h + zb)`: `q_face = |∇P| Y²(3h − Y) / (6η)`, `Y = (h − τ/|∇P|)+`. Each face moves at most `0.2·|ΔP|/G` per substep (no overshoot: a discrete maximum principle). A contact line is pinned: onto a dry cell a film advances only by its thickness above `h_pin`. Species move in the donor cell's proportions.
3. **Tools.** Brushes write a drag velocity `U` (spread sideways by a screened-Poisson solve of range `drag_range`) and a plough velocity `V` (down the gradient of contact pressure, unspread). Paint is carried by `U + V` with a MUSCL / van Leer scheme, in the bounding box where the velocity matters, at its own CFL (operator split from 4.2, 4.4, 4.5).
4. **Colour.** Pigments exchange among the solids at `D = D0·η0/η` (Stokes–Einstein): thickness is unchanged by this law.
5. **Water.** Evaporates at `evap`, plus `edge_evap` at a wet edge; soaks into the ground at `absorb·porous` until `store_cap` (a dried stratum seals the ground: `porous = exp(−Σstrata/0.02)`); the ground dries at `store_evap`.
6. **Oil.** Cures at `cure_rate·(oil/h)·(1 − κ)/(0.3 + h)`: fastest in thin, oily films.
7. **Limiter.** All of 4.2 and 4.4 are summed into face fluxes per species, then each species' outflow from each cell is scaled to what the cell holds. Then the update is applied in conservative form.
8. **Strata.** `fix()` turns every dry cell (no water; lean or `κ ≥ cure_min`) into a new stratum. `wait(steps, evap_boost, cure_boost)` lets time pass with drying and curing sped up.

## 5. Tools

A `Brush`'s `exchange` is per pass: the fraction of the way to the target a cell gets as the whole (swept) footprint passes over it, spread over the touches that takes, so the paint laid doesn't depend on the brush's speed. A `Brush` has a width, bristles (an uneven, per-brush clumping profile), a load (volumes per species), `exchange`, `drag`, `plough`. `touch()`:

- writes `U` and `V` where it touches;
- exchanges paint with the wet layer toward `target·pressure·(0.55 + 0.45·bristle)`, at a rate set by contact `w`: gives from its load (in the load's proportions, at most a quarter of the load per touch), picks up the excess (in the canvas's proportions). Every unit given leaves the load; every unit taken enters it.
- A `Stylus` has no load and `exchange = 0`: it only drags and ploughs. Charcoal is a brush loaded with dry pigment and `tooth_only=True`.

## 6. Optics

`reflectance(gain=10, layers=None) → (H, W, 4)`: Kubelka–Munk per layer (`km_layer` → R, T), composed bottom-up over the ground (`GROUND_RGBN`) with `R = R1 + T1² Rb / (1 − R1 Rb)`. Linseed yellows slightly with cure. `layers=k` peels back to the first *k* strata.

## 7. Sensors

`Surface(Z_mm, nir, rough_um)`; gloss is a presentation input for renders, not a sensor input.

- `goo.sensors.lidar.Scanner`: the averaged-spot model. Its range precision is **optimistic** (it averages ranges over the spot); its intensity is physically a *large-aperture direct-detection* channel.
- `goo.sensors.coherent.Geometry`: an FMCW scanner whose spot is N scatterers summed coherently; range from the zoomed DTFT peak of the beat signal. Calibrated so a speckle-averaged white return at normal incidence gives 8 µm thermal precision at 100 GHz. Produces speckle, speckle range outliers, mixed pixels and reflectance-dependent precision without modelling any of them separately.

Measured (see `studies/spot.py`, `studies/reflectography.py`), at 40° on the pear: one coherent look 62 µm point RMS (17.5% of points off by > 50 µm); 4 intensity-weighted looks 15 µm; 0.06 mm spot 8.6 µm. Coherent intensity cannot do reflectography: the underdrawing's contrast-to-noise saturates at ~1.1 with wavelength looks (true value 2.7); a direct-detection channel gets 2.4.

## 8. Known limits

- The wet layer is depth-averaged: pigments mix sideways, never stratify within one wet layer. Glazes and layering need `fix()`.
- No surface tension beyond `h_pin`; no Marangoni; no gravity along the panel (it lies flat).
- No stress, no fracture (yet): drying shrinkage, craquelure, cupping are not modelled.
- Pigment K and S at 1.5 µm are plausible, not measured.
- The coherent sensor stops and stares; scanning-induced speckle noise (Baumann et al. 2014, second effect) is not modelled.

## 9. Extending

- **A new law:** a switch in `L` (default on); conservative form; contributes face fluxes to the limiter (§4.7) or is a local source with an explicit ledger entry; a test that it is exactly zero when off and does what its physics says when alone.
- **A new pigment:** `K`, `S` in four bands in `PIGMENTS`; nothing else.
- **A new tool:** writes only `U`, `V` and exchanges through a load; adds no species from nowhere.
- **A new sensor:** consumes a `Surface`; keeps truth beside its output for scoring; gets a calibration test.
