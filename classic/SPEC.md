> **Deprecated.** This is goo classic, the first engine, frozen at commit 6fcca08 (branch `goo-classic`). It is not developed further. The kernel that replaces it is specified in the repository's root `SPEC.md`.

# goo — specification

This is the contract between goo's parts. Code that breaks it is wrong even if the picture looks right. §15 says what is verified, what is opt-in, what is known wrong and what is unverified at this checkpoint; `docs/VISION.md` says where goo is meant to go.

## 1. Domains

| Domain | Module | Knows | May change |
|---|---|---|---|
| **World** | `goo.classic.engine.Canvas` | everything: species volumes, strata, cure, the water in the ground, the frisket, the ledger | its own state, only through the laws (§4) |
| **Tools** | `goo.classic.brush` (`Brush`, `Stylus`), `goo.classic.tools` (`Knife`, `Blotter`, `spatter`) | their own load and footprint; the wet layer they touch; the tape over it (`_open`); for charcoal, the surface it catches on | the tool fields `U`, `V` (velocities) and the wet layer, only by exchange with their load; the ledger's `loaded` and `rinsed` and the canvas's list of brushes when they are charged or rinsed |
| **Optics** | `goo.classic.optics` (the canvas's model, `cv.optics`) behind `Canvas.surface()`, `reflectance()`, `finish()`; `Canvas.sheen()`, `touch_dry()`; `goo.classic.surface.Surface.from_canvas()` | species volumes per layer, the cure of the wet layer (linseed yellows), the water in the ground (v2, `sheen`) | nothing (pure functions: no flush, no step; `touch_dry` with ticks pending flushes a scratch fork, not the canvas) |
| **Sensors** | `goo.classic.sensors.*` | a `Surface` (`goo.classic.surface`): height (mm), reflectance in the model's bands, roughness (µm), gloss (0..1), the cell size | nothing in the World |
| **Painter** | `goo.classic.painter`, `goo.classic.palette`, `goo.classic.detail`, `goo.classic.paints`, `goo.classic.studio`, `goo.classic.easel` | colour seen on the canvas (`reflectance`), the canvas's size, its motif (if any), its paints and tools (it may try them on scrap canvases of its own and measure those any way it likes); the studio also how wet the surface looks (`sheen`), the finger test (`touch_dry`) and `finish` | the World only through Tools, `fix`, `wait`, `step`; the studio also `mask` / `unmask` and `fork` (a rehearsal). What each may read is enforced (§10, §11) |
| **Paintings, studies** | `classic/paintings/`, `classic/studies/` | whatever they are given | the World only through Tools, `Canvas.add`, `fix`, `wait`, `step`, and the canvas's `laws` (its climate) |
| **Data** | `goo.classic.data` (`goo/classic/data/optics.json`) | optical constants and derived K, S spectra with provenance (§13) | nothing; it imports no other part of goo |
| **Matter** | `goo.classic.matter` | substance descriptions and the maps from them to K, S and mechanics (§14) | nothing in any painting's World. `goo.classic.matter.phenotype` makes scratch canvases with `Canvas.add` and registers a temporary entry in `PIGMENTS` for the length of one data sheet, then removes it |

Three rules carry most of the weight:

- **No synthesis.** Every visible effect — a wet edge, a backrun, a bristle groove, a burr, a slump, a crack in a reflectance image — must come out of the laws acting on the state. A painter script may choose *where a brush goes and what it carries*; it may never write colour or height into the World directly. (`Canvas.add` exists for test setups and is booked like a brush load.)
- **Sensors see the surface, not the paint.** A sensor is constructed from a `Surface`. It must not read `q`, `strata`, `species` or the ledger. The World is presented to sensors only by `Surface.from_canvas(cv)`, which goes through the three optics functions and reads no raw state. Ground truth for scoring a sensor is computed beside the measurement and never fed back into it (`classic/tests/test_sensors.py::test_sensor_sees_only_the_surface`, `::test_relight_sees_only_the_surface`).
- **A painter sees colour, not paint.** From its motif a painter may take everything; of the canvas it knows only what optics show (`reflectance()`, and for the studio `sheen()`, `touch_dry()`, `finish()`) and what `fix` returns (§10, §11). `classic/tests/test_detail.py::test_painter_stays_in_its_domain` and `classic/tests/test_studio.py::test_studio_stays_in_the_painters_domain` enforce it with a canvas behind glass.

## 2. Units

| Quantity | Unit |
|---|---|
| Cell | 0.25 mm square (`goo.classic.sensors.lidar.CELL`) |
| Volume per cell | thickness in units of 0.25 mm; a volume of 1 in one cell is 0.0156 µL |
| Time | one *tick* = one call to `Canvas.step()`; a brush touch is one tick. goo has no clock in seconds (§14 gives one reading) |
| Bands | classic: R, G, B and N (≈1.5 µm, 1550 nm, where reflectography looks and the lidar works); v2: 15 visible boxes of 20 nm and N (§6) |
| Optics gain | `reflectance(gain=10)`: K and S in `PIGMENTS` are per 0.025 mm of pure pigment. `goo.classic.data` holds K, S in mm⁻¹ of pure pigment: goo units = mm⁻¹ × 0.025 mm (`goo.classic.data.GOO_UNIT_MM`) |

Realistic quantities, for calibration: a paint layer is 0.1–0.4 units (25–100 µm); impasto 2–8 units (0.5–2 mm); a loaded filbert holds ~6000 units (0.1 mL); a wash mop ~12000.

The laws' constants are in cell units. A different cell size is a recalibration, not a parameter.

## 3. Invariants (tested in `classic/tests/test_engine.py` unless named)

- **I1 Conservation.** For every species *s*: `loaded[s] = on_canvas[s] + in_brushes[s]`, plus, for water only, `+ evaporated + in_ground + ground_evaporated`. `on_canvas` counts the wet layer and every stratum. Rinsing a brush (`rinse()`, `Knife.wipe()`, `Blotter.wring()`, and every recharge) moves its remainder to `led["rinsed"]` and out of `loaded`. Relative residual ≤ 1e-9 after any session.
- **I2 Positivity.** `q ≥ 0`, `store ≥ 0` everywhere, always. Positivity is guaranteed by construction (per-species outflow limiting, §4.7), not by clamping; the `np.maximum(q, 0)` after an update only removes round-off.
- **I3 Exact zero.** Every law has a switch in `Canvas.L` (`flow, drag, diffusion, evaporation, absorption, curing`). With all of them 0 and no tool touching, `step()` leaves `q` bit-for-bit unchanged.
- **I4 Determinism.** The same session (same seeds, same calls) gives the same arrays. A `fork()` continued exactly as the original continues gives bit-identical arrays, ledger and counters (`classic/tests/test_world_services.py::test_fork_continues_bit_identically`); the easel's undo replays acts on this guarantee (`classic/tests/test_easel.py::test_back_past_pruned_history_replays_exactly`).
- **I5 Strata are immutable.** Only `fix()` creates a stratum; nothing modifies one afterwards. Tools act on the wet layer only. Strata are stored compact (each cropped to its bounding box, with its offset and per-species totals). `Canvas.strata` reads as a sequence of dense (species, H, W) arrays, built on access and read-only; `strata.block(i)` and `strata.totals(i)` give the compact form. A fork shares the compact blocks rather than copying them (`test_fork_shares_strata_does_not_copy_them`).
- **I6 Stability without loss.** `step()` chooses its own substeps from every process's stability limit; `capped` counts steps that hit `max_sub` and should stay 0 in normal use. A step may be slow; it may not blow up, go negative, or lose paint. (Curing is the exception today: §8.)
- **I7 Two clocks.** Tools act every tick, in a window around the touch; the slow laws (§4.2, 4.4–4.6) run over the elapsed ticks every `L["lazy"]` ticks (`flush()`); `fix`, `wait`, `mask`, `unmask` flush first. `lazy = 1` runs every law every tick. `flow_sub`, `diff_sub` cap substeps per tick for leveling and colour diffusion; both are stable at any cap (face caps, §4.2, 4.4), so the caps trade the speed of very fast leveling for time, never stability or conservation.
- **I8 Windowed equals whole.** `L["windowed"]` (default 1) runs per-touch bookkeeping and the slow laws only where anything can change: cells holding paint, water in the ground (`store > 0`) or cure (`κ > 0`), plus a margin; live cells are never allowed within 2 cells of a window side that is not the panel's edge. `L["windowed"] = 0` runs everything over the whole panel. The two give bit-identical `q`, strata, `κ`, `store`, `zb`, ledger, `substeps` and `capped` (`test_windowed_equals_whole_panel*`). Code outside the engine that reads or assigns `q`, `U`, `V`, `store` or `kappa` makes the next windowed step or flush rescan the panel once; arrays held from before that are not watched. The identity is checked, not proved: it rests on how numpy and scipy sum and filter (§8).
- **I9 Tape is a wall.** Under the frisket (§4.9) no law and no tool changes the wet layer, the ground's water or the cure (`classic/tests/test_world_services.py::test_nothing_changes_under_the_mask`, `::test_tools_lay_nothing_under_the_mask`). A canvas that was masked and fully unmasked is bit-identical to one never masked (`::test_no_mask_is_todays_canvas`).

## 4. The World's laws

State: `q[species, H, W]` (wet layer; species = `CARRIERS` (`water`, `oil`) + the canvas's pigments), `strata` (a sequence of frozen `q`-shaped layers, bottom first, stored compact), ground relief `g`, base `zb = g + Σ strata`, oil cure `kappa ∈ [0, 1]`, water in the ground `store`, the frisket (an (H, W) bool or none).

Derived: thickness `h = Σ_s q_s`, solid fraction `φ = Σ_pigments q / h`. Every species after the carriers counts as a solid, the acrylic binder included.

1. **Rheology.** Krieger–Dougherty viscosity `η = η0 (1 − φ/φm)^-2 (1 + cure_stiff·κ)`; yield stress `τ = τ0 ((φ − φy)+ / (φm − φ))^1.5 (1 + cure_yield·κ²) + tau_cure·κ²`.
2. **Leveling.** Bingham thin-film flux down the gradient of `P = G(h + zb)`: `q_face = |∇P| Y²(3h − Y) / (6η)`, `Y = (h − τ/|∇P|)+`. Each face moves at most `0.2·|ΔP|/G` per substep (no overshoot: a discrete maximum principle). A contact line is pinned: onto a dry cell a film advances only by its thickness above `h_pin`. Species move in the donor cell's proportions.
3. **Tools.** Brushes write a drag velocity `U` (spread sideways by a screened-Poisson solve of range `drag_range`) and a plough velocity `V` (down the gradient of contact pressure, unspread). Paint is carried by `U + V` with a MUSCL / van Leer scheme, in the bounding box where the velocity matters, at its own CFL (operator split from 4.2, 4.4, 4.5).
4. **Colour.** Pigments exchange among the solids at `D = D0·η0/η` (Stokes–Einstein scaling only: §14 shows the magnitude is not Brownian): thickness is unchanged by this law.
5. **Water.** Evaporates at `evap`, plus `edge_evap` at a wet edge; soaks into the ground at `absorb·porous·(store_cap − store)/store_cap`, so absorption slows as the ground fills and stops at `store_cap` (a dried stratum seals the ground: `porous = exp(−Σstrata/0.02)`). The ground dries at `store_evap`, scaled by the `evaporation` switch, and only while the `absorption` law is on. Defaults: `absorb` 0.004 on gesso and 0.04 on any other ground, `store_cap` 0.08, `store_evap` 0.002 per tick.
6. **Oil.** Cures at `cure_rate·(oil/h)·(1 − κ)/(0.3 + h)`: fastest in thin, oily films. `cure_rate` is 0 by default: oil never cures unless the painting gives its canvas a cure rate (its climate).
7. **Limiter.** All of 4.2 and 4.4 are summed into face fluxes per species, then each species' outflow from each cell is scaled to what the cell holds. Then the update is applied in conservative form.
8. **Strata.** `fix(cure_min=0.8, region=None)` turns every dry cell into a new stratum and returns the fraction of the cells holding wet-layer paint that it fixed (0.0 when none holds paint, and also when none is dry). With `region` ((H, W) bool; another shape raises `ValueError`) only dry cells inside it are fixed and the fraction is of the region's cells holding paint (`test_fix_region_fixes_only_inside_it`). Dry means: water ≤ 1e-6 (`DRY_WATER`; that remainder is booked as evaporated), and either lean (oil ≤ 5% of `h`, `LEAN`) or `κ ≥ cure_min`. Fixed cells have their `κ` reset to 0. `wait(steps, evap_boost, cure_boost)` lets time pass with drying and curing sped up; a boost multiplies the canvas's own rate, so it does nothing to a rate of 0.
9. **World services.** Pure readings and boundary conditions the World offers to tools and painters (`classic/tests/test_world_services.py`).
   - `fork()`: an independent canvas in exactly this state. Mutable state is deep-copied (wet layer, cure, ground water, relief, tool fields, ledger, clocks, counters, windowed books, laws, frisket); strata blocks are shared (I5). Brushes are cloned in order; `Brush.twin(fork)` finds a brush's twin. Changes nothing in the original.
   - `sheen() → (H, W)` in 0..1, how wet the surface looks. Free liquid = water above `DRY_WATER` plus oil where the film without its water is not lean and `κ < CURE_TOUCH` (0.8). The pigment bed holds `bed = solids·(1 − φm)/φm` in its pores. Liquid standing above the bed glistens: `0.5 + 0.5·clip((free − bed)/GLISTEN, 0, 1)`, `GLISTEN = 0.02`. Otherwise `0.5·clip((free + store)/(bed + store_cap), 0, 1)`: satin while the pores or the ground hold liquid. Readers use `SHEEN_GLISTENING = 0.5` (above: glistening, a drop-in spreads freely) and `SHEEN_DAMP = 0.02` (above, up to 0.5: damp, satin, the moment for a backrun; at or below: dry-looking). Sheen never rises as water evaporates or soaks in, or as oil cures (`test_sheen_never_rises_as_water_leaves`). It reads the state directly and is the same under every optics model.
   - `touch_dry(cure_min=0.8) → (H, W) bool`: True where the wet layer holds nothing, or only what `fix(cure_min)` would fix now. Bare cells are touch-dry whatever water the ground holds (`sheen` shows that). With slow-law ticks pending it answers for the state `fix` would see, by flushing a scratch fork (`test_touch_dry_is_what_fix_would_fix`).
   - `mask(region)` / `unmask(region=None)`: the frisket. Both flush first. `mask` raises `ValueError` if any cell to be masked is not touch-dry. A masked cell is a wall for every law: no face flux (leveling, colour, tool advection) crosses a face touching it, and nothing evaporates, soaks in or cures under it. Tools multiply every contact by `_open` (1 open, 0 masked).
   - The acrylic binder: `PIGMENTS["acrylic"]` is a clear polymer solid (K 0.004–0.007, S 0.02 in every band; plausible, not measured). It is a species like a pigment to the laws; `goo.classic.paints` never offers it as a colour.

## 5. Tools

A `Brush`'s `exchange` is per pass: the fraction of the way to the target a cell gets as the whole (swept) footprint passes over it, spread over the touches that takes, so the paint laid doesn't depend on the brush's speed. A `Brush` has a width, bristles (an uneven, per-brush clumping profile), a load (volumes per species), `exchange`, `drag`, `plough`, and a shape. `touch()`:

- writes `U` and `V` where it touches;
- exchanges paint with the wet layer toward `target·pressure·(0.55 + 0.45·bristle)`, at a rate set by contact `w`: gives from its load (in the load's proportions, at most a quarter of the load per touch), picks up the excess (in the canvas's proportions, a quarter of it per touch). Every unit given leaves the load; every unit taken enters it.
- A `Stylus` has no load and `exchange = 0`: it only drags and ploughs. Charcoal is a brush loaded with dry pigment and `tooth_only=True`; it reads the surface it catches on.

Shapes. `shape="round"` (default): an oval footprint that follows the travel, bristles across it. `shape="flat"` (with `aspect ≥ 1`): the long axis is held at `heading` (radians, image frame: x right, y down; `None` holds it square to the travel), bristles along it, so a turning stroke goes thick and thin. A round brush given a heading raises `ValueError`.

One interface for every tool (`goo/classic/tools.py` docstring has the table): `charge({species: volume})` (the old load goes to the rinse jar first; booked); `stroke(path, speed=2.5, pressure=1.0, target, steps_after=0, heading=None) → Receipt`; `press(x, y, pressure=1.0, ticks=1, target) → Receipt` (held still, the canvas steps each tick); `rinse() → Volumes`; `twin(fork)` (the same tool on a `Canvas.fork()`; `ValueError` if the canvas is not a fork of the brush's own).

A `Receipt` (`goo.classic.brush.Receipt`, summable with `+`) has `species`, `given` and `taken` (per-species `Volumes`, `.volume` their total), `length` (cells), `ticks`, `cells` (distinct open cells touched), `bbox` and `mask(H, W)` of those cells. `touch()` called alone books no receipt.

`goo.classic.tools`:

- `Knife(cv, width)`: a rigid flat blade (`exchange` 1.0, `drag` 0, `plough` 2.0). Lays from its load toward a flat `target` (no bristle ribbons); takes all the wet paint above the target under the blade onto its load; in the last `edge` (1.5) cells at each end takes nothing and ploughs paint outward along the blade, leaving a ridge at each side. `wipe()` is `rinse()`.
- `Blotter(cv, width, capacity=None, exchange=0.5, target=0.0)`: a disc that takes and never gives and writes no velocity. All carrier is mobile; of the pigment, the fraction `(1 − φ/0.64)+`, so a dilute wash is taken nearly in proportion and a stiff paste keeps its pigment. Holds at most `capacity` (default 0.25 per cell of its area). `wring()` is `rinse()`.
- `spatter(brush, at, n, spread, size, rng) → Receipt`: `n` droplets from the brush's load, each of volume `size·Gamma(2, 1/2)` landing at `at + spread·N(0, 1)` as a dome 0.3 units thick (`DROP_FILM`), no velocity. Clipped to the load; what would land off the panel or on tape stays on the brush. The canvas steps once.
- `opening(cv, y0, y1, x0, x1)`: the tape over a window (1 open, 0 masked), or None.

Charging a brush books its load in the ledger and registers the brush with the canvas.

The engine's own tools use the tool channel: they write through `cv._q`, `cv._U`, `cv._V`, say where with `cv._mark_tool` (velocities) and `cv._mark` (paint), which keeps the windowed books (I8), and multiply their contact by `cv._open`. `classic/tests/test_tools.py::test_tools_stay_in_the_tool_channel` holds `goo.classic.tools` to that set. A tool that uses the public `q`, `U`, `V` still works and gives the same result, but makes the canvas rescan the panel.

## 6. Optics models

An optics model is an object the canvas holds: `Canvas(..., optics="classic" | "v2")`, `cv.optics` (`goo/classic/optics.py`). It has its bands (`cv.optics.bands`), its species table, and pure functions of the canvas state: the species volumes of the wet layer and of each stratum, the wet layer's cure `κ`, the water in the ground (`store / store_cap`) and the ground type. None of them flushes, steps or changes anything. `classic` is the default and reproduces goo's paintings bit for bit; paintings made by a looking painter depend on the optics, so changing the default is the owner's decision.

- `surface() → (H, W)`: the height of the top surface, `zb + h`, in volume units.
- `reflectance(gain=10, layers=None, light=None, adapt=False) → (H, W, nb)` in the model's bands: diffuse illumination, viewed along the normal, specular excluded. `layers=k` takes the first *k* entries of (strata, then the wet layer), so `k = len(strata)` peels back to the dried painting. `light` (a name in `goo.classic.optics.LIGHTS`: `"D65"` daylight, `"A"` tungsten, or a function of nm) returns instead the colour under that light, linear sRGB (H, W, 3); `adapt=True` as an eye adapted to it (Bradford, to D65).
- `finish() → (H, W)`: gloss 0..1, the smooth fraction of the same top surface: the specular part `reflectance()` leaves out.

**classic.** Kubelka–Munk per layer on pigment volumes in R, G, B, N (`km_layer` → R, T), composed bottom-up over the ground (`GROUND_RGBN`) with `R = R1 + T1² Rb / (1 − R1 Rb)`; linseed yellows with cure (`1 + 2κ` wet, 3 in a stratum). No surface, no surroundings. Finish of the *top film* (the wet layer where it holds paint, volume > 1e-9, else the topmost stratum with paint, else bare ground, 0): `gloss = clip(h/0.3, 0, 1) · clip((oil/h)/0.3, 0, 1)`. Under D65 its colour is R, G, B as they are; under another light each channel is scaled by that light's white (a camera-RGB approximation: classic has no spectrum). Water carries no K or S in classic, so a water-borne film looks the same wet and dry.

**v2** (opt-in). Everything below comes from composition and the species table; the code names no medium.

- *Bands*: 15 visible boxes of 20 nm, 400–700 nm (the observer folds 380–400 and 700–780 into the end boxes, as ASTM E308 does), plus 1.5 µm. A pigment's K and S are smooth curves `exp(c0 + c1 x + c2 x²)` whose means over 400–500, 500–600, 600–700 nm equal its B, G, R in `PIGMENTS` exactly; N is the table's. The ground's reflectance is a sigmoid-quadratic (Jakob & Hanika 2019) through `GROUND_RGBN`. Colour: CIE 1931 observer (the analytic fit of Wyman, Sloan & Shirley 2013), D65 (CIE table) or A (Planck, 2856 K), to linear sRGB (IEC 61966-2-1), white under D65 = (1, 1, 1).
- *Species table* `goo.classic.optics.SPECIES`: per species a role (`volatile`, `binder`, `pigment`, `void`), a refractive index `n`, K and S per 0.025 mm in the reference surrounding (oil; pigments take them from `PIGMENTS`), a `src` string saying what is measured and what is plausible, and optionally `dispersed_in` (an emulsion binder), `yellows`, and a measured `spectrum` that replaces the curves. Water 1.333, linseed 1.48, acrylic polymer 1.49, gum 1.47, air 1.000; pigment indices are plausible, not measured. Grounds in `goo.classic.optics.GROUNDS`: the dry reflectance as seen, the solid's index, and the fraction of pores bound by size or glue (paper: cellulose 1.53, 0.05; gesso: 2.0, 0.55).
- *Surroundings*. Per film: the continuous phase `L` is its carriers and binders, plus the coalesced part `clip((b/(b + w) − P)/(1 − P), 0, 1)` of a binder `b` dispersed in carrier `w`; its index `n_c` is the volume mean. The bed `B` is the pigments and the uncoalesced droplets; its void capacity `V = B (1 − P)/P`, `P = 0.64` (random close packing: the CPVC). The continuous phase fills the voids up to `V`; the rest is air, fraction `a = 1 − min(L, V)/V`. Scattering of particle *j*: `S_j = v_j S_ref,j [(1 − a) f(n_j/n_c) + a f(n_j)] / f(n_j/n_oil)`, `f(m) = ((m² − 1)/(m² + 2))²` (the Lorentz–Lorenz contrast factor: Rayleigh's index dependence, and the `(m − 1)²` law of diffuse scattering near index match). It carries no size or wavelength dependence of its own. Absorption `K_j = v_j K_ref,j`; linseed's grows with cure, `1 + 2 min(κ/0.8, 1)` wet and 3 in a stratum, so a film cured to `fix()`'s default looks the same before and after it.
- *Ground*: a semi-infinite KM scatterer whose dry body is the table's reflectance with its own rough surface taken off (inverse Saunderson). Its scattering scales with the contrast between its solid and what fills its pores: bound binder, air, and water as `store / store_cap` fills the free pores.
- *Surface*. Per film, closure `c = clip(L/V, 0, 1)`; index `n = c·n_c + (1 − c)·1` (an open, underbound bed has no interface: its particles' scattering in air is already in S); smoothness `clip((L − V)₊ / 0.1, 0, 1)` (0.1 units, 25 µm, of clear phase above the bed levels the tooth). Each film covers what is under it by `clip(h/0.02, 0, 1)`; the ground's surface is the index of its filled pores, rough.
- *Composition*: KM layers bottom-up over the ground body (associative: `goo.classic.optics.over`), then the surface (Saunderson 1942): `R = (1 − smooth)·F0 + (1 − F0)(1 − r_i) Rb / (1 − r_i Rb)`, `F0 = ((n − 1)/(n + 1))²`, `r_i = 1 − (1 − r_e)/n²`, `r_e` the Fresnel reflectance averaged over a cosine-weighted hemisphere. `finish()` = smoothness.
- Constants not from a named source are plausible, not measured: every pigment index, the polymer's and gum's K and S, the grounds' bound fractions and `n_solid`, `E_SMOOTH = 0.1`, `H_COVER = 0.02`, `KAPPA_FULL = 0.8`.
- v2 reads K and S from `PIGMENTS` (goo's classic scale). It does not read `goo/classic/data/optics.json` (§13).

**One code path.** `goo.classic.surface.Surface.from_canvas(cv)` is the one place a canvas is turned into what sensors see: `Z_mm = surface()·CELL`, `reflectance = reflectance()` with the model's `bands`, `gloss = finish()`, `rough_um = 1.5·(1 − gloss) + 0.3·gloss`. `Surface.rgb` is the colour under D65 (classic: the R, G, B bands), `nir` the 1.5 µm band, `colour(light)` the colour under another light, `under(light)` a Surface lit by it. Its arrays are read-only copies. `Palette(..., optics=cv.optics)` previews films with the model's `preview`, which is built from the same functions as `reflectance()`; a `Painter` sees through its palette's model and refuses one whose bands differ from the canvas's.

**Invariants** (`classic/tests/test_optics.py`): R + T + absorbed = 1 per layer and band, 0 ≤ R ≤ 1; no paint shows the ground (v2 exactly; classic to 1e-8, its S floor); an optically thick film shows its own R∞; a surrounding of the pigment's index makes it purely absorbing; v2 with every non-pigment index equal, no surface and classic bands is classic to 1e-6 (classic's S floor on empty layers); composition is associative; more absorption never brightens a band; more of a pure scatterer over black never darkens it; `fix()` does not change a cured film's look (1e-12) or a dried wash's (1e-5); drying changes colour continuously; the palette's preview equals the canvas in oil, watercolour, gum wash and acrylic, wet and dry (v2 1e-12; classic 1e-7); a made-up table shows every v2 effect from composition alone; the looking painter works under v2 (`test_one_code_path_the_looking_painter_works_under_v2`).

**Known limits of v2.** Measured with `classic/studies/optics.py` (ΔE there is CIE76, the mean over the patch). The table's K and S were tuned by eye for classic, which has no surface: through v2's surface oil paint reads darker in the mid-tones (the pear: ΔE 16.1 from classic). Classic's R, G, B are sRGB channels, not box means, so a masstone in v2's bands without a surface differs from classic's by up to ΔE 19.5 (rose 19.5, dioxazine 19.3, ultramarine 17.6); with the surface, up to ΔE 20.7 (green 20.7, red 20.6, rose 20.0). Wet/dry shifts are too large for a binderless wash and for acrylic: a 0.40 water + 0.03 ultramarine wash, wet at tick 40 against dry, ΔE 22.5 on paper and 24.2 on gesso (classic 0.25 and 0.37); an acrylic stroke ΔE 21.2 (classic 0.0); a wash with gum 0.03 ΔE 8.1. No interfaces between strata, no sinking-in, the ground is semi-infinite, water held in the ground does not absorb at 1.5 µm. The relight sensor keeps its own white specular with F0 = 0.04 whatever the model's surface index. `DetailPainter` passes colour, not band reflectance, to its palette, so a v2 palette refuses it (`ValueError` from `Palette._match_model`; read from the code, not tested). `goo.classic.paints` and `goo.classic.studio` use classic optics only (§11, §12).

## 7. Sensors

`Surface` (`goo.classic.surface`): `Z_mm`, `reflectance` (H, W, nb in the model's bands; classic 4), `rough_um`, `gloss`, `cell_mm`, `bands`; `nir` and `rgb` are views. Every sensor takes one. The older form `Surface(Z_mm, nir, rough_um)` still constructs a surface with no visible bands, which the visible-light sensor refuses.

- `goo.classic.sensors.lidar.Scanner`: the averaged-spot model. Its range precision is **optimistic** (it averages ranges over the spot); its intensity is physically a *large-aperture direct-detection* channel.
- `goo.classic.sensors.coherent.Geometry`: an FMCW scanner whose spot is N scatterers summed coherently; range from the zoomed DTFT peak of the beat signal. Calibrated so a speckle-averaged white return at normal incidence gives 8 µm thermal precision at 100 GHz. Produces speckle, speckle range outliers, mixed pixels and reflectance-dependent precision without modelling any of them separately.
- `goo.classic.sensors.relight`: a visible-light camera straight above the panel, one distant light of direction `L` (from the surface toward the light; image x right, y down, z toward the viewer) plus a white sky. Radiance `= I·V·(n·L)₊·[ρ(1 − gF) + gF·(s+2)/2·(r·L)₊^s] + ambient·AO·[ρ(1 − gF̄) + g⟨FΛ⟩]`, with `g` the gloss, Schlick `F0 = 0.04`, `s = 100`, `V` horizon-marched cast shadows with a ±1.5° penumbra, `AO` over 16 azimuths × 12 mm. The diffuse term gives up exactly what the skin reflects: directional albedo ≤ ρ + gF̄(1 − ρ) ≤ 1, proved for facets facing the viewer and measured ≤ 1 + 1e-3 for tilts to 60°. No exposure compensation. `prepare()` once, `render(L)` per direction; `relief_gain` scales the height before normals, shadows and occlusion (1 = the painting); `texels(image, n)` is the aperture of a texel: n × n cells averaged in linear radiance.

Measured (see `classic/studies/spot.py`, `classic/studies/reflectography.py`), at 40° on the pear: one coherent look 62 µm point RMS (17.5% of points off by > 50 µm); 4 intensity-weighted looks 15 µm; 0.06 mm spot 8.6 µm. Coherent intensity cannot do reflectography: the underdrawing's contrast-to-noise saturates at ~1.1 with wavelength looks (true value 2.7); a direct-detection channel gets 2.4. These figures predate `finish()`; the pear's oil is all wet, where the old and new gloss agree.

## 8. Known limits

- The wet layer is depth-averaged: pigments mix sideways, never stratify within one wet layer, and never settle (no granulation). Glazes and layering need `fix()`. The thin-film law assumes paint much thinner than it is wide; a ridge as tall as it is wide is outside it.
- No surface tension beyond `h_pin`; no Marangoni; no gravity along the panel (it lies flat).
- No stress, no fracture (yet): drying shrinkage, craquelure, cupping are not modelled.
- **Curing can overshoot.** The cure law is integrated explicitly over a whole flush, so with `lazy > 1` and a high rate (a large `cure_boost`) `κ` can leave [0, 1]. With `lazy = 90`, `cure_rate = 0.002`, `wait(900, cure_boost=12)` a 0.27-unit film of oil and white reaches `κ = 1.2`; with `lazy = 1` it stays below 1. Waiting in short spells is stable. Fixing it (the exact exponential update) changes existing paintings, so it is a change of law, not of bookkeeping.
- **Drying time is set by the ground's water store, not by `absorb`.** `store_cap` 0.08 emptied at `store_evap` 0.002 per tick takes 40 ticks; while the store is full, soaking-in is limited by how fast the ground dries, so paper's tenfold `absorb` buys almost nothing. Measured on this checkpoint: a 20 × 20-cell patch put down with `Canvas.add` (0.48 water + 0.02 ultramarine, a 0.5-unit wash of strength 0.04), default laws, `lazy = 1`: `touch_dry()` over the whole patch at tick 166 on paper and 179 on gesso; `sheen() ≤ SHEEN_DAMP` at tick 202 and 197. The merge reported ~195 (paper) and ~192 (gesso) for the same wash; its exact setup is not recorded here. In both measurements the two grounds differ by less than 10%.
- Pigment K and S at 1.5 µm (and, for `ivory_black` and `dioxazine`, in every band) are plausible, not measured. Where measured optical constants exist, the classic K and S are ~155× weaker than computed physical values (§13).
- The engine clips negative K silently: `km_layer` floors S and the root inside it, and `reflectance()` clips R to [0, 1]. A gain medium is mis-computed with no warning and no debt (§14).
- The engine's colour diffusion constant `D0` is 1e3–1e6 times the Brownian value for the particle sizes that reproduce the classic table (§14); it stands in for shear mixing.
- Finish is the top film's only: no sinking-in, no varnish, no dependence on cure or on the pigment beyond the oil fraction; any wet residue above 1e-9 hides the film beneath it.
- Relight: orthographic view from above only, one distant light, a sky without glints, occlusion measured from the panel plane. A light near the zenith hot-spots every glossy flat, which is physical.
- The lidar's glint lobe is narrow enough that gloss does not matter at the 40° stations.
- The coherent sensor stops and stares; scanning-induced speckle noise (Baumann et al. 2014, second effect) is not modelled.
- I8 was checked on one numpy and scipy version on one CPU. It relies on scipy's uniform filter being a running sum and on numpy's reductions grouping terms the same way for a window as for the panel; the ledger's evaporation sums and `fix()`'s totals stay panel-wide for that reason.
- `fix()` is destructive: every partial try leaves its own stratum (the sampler ends with 63). `touch_dry()` now answers what `fix()` would fix without fixing, and `fix(region=...)` fixes part of the panel; the studio uses both, `DetailPainter` still decides by trying `fix` (§10).

## 9. Extending

- **A new law:** a switch in `L` (default on); conservative form; contributes face fluxes to the limiter (§4.7) or is a local source with an explicit ledger entry; a test that it is exactly zero when off and does what its physics says when alone; bit-identical windowed and whole-panel (I8); it respects the frisket (I9).
- **A new pigment:** `K`, `S` in four bands in `PIGMENTS`; nothing else for classic. For v2, an entry in `goo.classic.optics.SPECIES` with its role, `n` and a `src` saying what is measured. Say whether the values are measured.
- **A new optical number** (in `PIGMENTS`, `SPECIES`, `GROUNDS` or `goo/classic/data/optics.json`): carries its provenance: computed (with its source and licence), measured (cited), assumed or guessed. `optics.json` is regenerated by `classic/studies/optics_data.py`, never edited by hand (§13).
- **A new tool:** writes only `U`, `V` and exchanges through a load; adds no species from nowhere; uses the tool channel and marks its window (§5); multiplies its contact by `_open`; returns a `Receipt`.
- **A new sensor:** is constructed from a `Surface` (from `Surface.from_canvas`, or synthetic) and reads nothing else; extends the source-inspection test to its module; keeps truth beside its output; gets a calibration test (a zero test, and a furnace or energy receipt where it applies).
- **A new painter:** a `Painter` subclass that reuses its seeing (`see`), mixing (`mix`, `swatch`) and stroke laying (`lay`); reads the World only through the Painter's interface (§1) and passes the proxy test in `classic/tests/test_detail.py`.
- **A new studio act:** a `Studio` method under `@act(example, kind)`, built from tools, `fix`, `wait`, `mask` / `unmask`; reads only what `classic/tests/test_studio.py` permits; raises `NothingLaid` when it lays (or takes) nothing.

## 10. Painters

`goo.classic.painter.Painter` looks: for each pass it compares the canvas to the motif simplified at the brush's own scale, puts strokes where the difference is largest, follows the motif's structure or a plan, and mixes each colour by inverting the optics over what the canvas shows along the stroke. Of the canvas, painter code (`goo.classic.painter`, `goo.classic.palette`, `goo.classic.detail`) may read only `reflectance`, `fix`, `wait`, `step`, `W`, `H` (`PAINTER_MAY` in `classic/tests/test_detail.py`).

`goo.classic.detail.DetailPainter` is a `Painter` for motifs with features a cell or a few cells wide (a line, a glyph, a texture): pixel art, a texture page, lettering.

- `split(motif, ids, width)` divides the motif, as a pure function of it, into its broad masses and its fine features: a cell is a feature if it fits in no (width+1)-square of its own colour and object. In the broad motif a feature takes the colour around it. Features are traced as 8-connected chains, each with its colour and width.
- Masses are painted from the broad motif, region by region, in turns that never put wet paint next to wet paint; a brush's rim stays inside its region except at the panel's edge. The painter then checks each region by what it sees and repeats coverage where the film does not hide.
- Features are traced last with brushes as wide as the feature, one colour at a time, light first, each dried and fixed before the next. The caller may name features to be laid as raised work, in short, loaded paint.
- Dryness is decided by trying: `wait` under a lamp, then `fix`; if `fix` returns less than 1, wait longer and try again. Never from cure, water or thickness. The canvas must have a cure rate for this to end (§4.6).

From its motif the painter may take everything: colour, placement, object ids, the geometry of its features. Tracing a motif is painting as long as every unit of paint goes through a tool and everything the painter knows of the canvas is colour.

`classic/paintings/sampler/` is its reference painting: a synthetic 64 × 64 motif at 4 cells per texel. Colour difference from the motif (CIEDE2000, per texel) 4.0 overall, 4.5 on the features; mass interiors 1.8. The stock pigments cannot mix three of its line colours, which is most of the feature error.

## 11. Studio and easel

`goo.classic.studio.Studio` is painter code without a motif: a canvas, a rack of brushes, named paints, a clock and a journal, painting in a manual's vocabulary. Each act chooses and sizes its brush from the rack (`WIDTHS`, 1.5–46 cells), charges it, lays strokes that run off the panel where the region meets the panel's edge and keep the rim inside the region elsewhere, rinses the brush, and returns a studio `Receipt`. An act that lays (or, for lift, blot, scrape, takes up) nothing raises `NothingLaid`; a request it cannot carry out raises `StudioError`. Both say what to change.

- `Studio.new(width, height, pigments, medium="watercolour", ground=None, seed=1, laws=None)`: watercolour and gouache on paper, acrylic and oil on gesso; `lazy = 3`; oil gets `cure_rate = 1e-4`; acrylic gets the binder species. The canvas uses classic optics (no optics argument).
- Acts (`goo.classic.studio.ACTS`): `mix, wet, wash, graded_wash, drop_in, backrun, glaze, dry_brush, scumble, line, dab, impasto, knife, scrape, lift, blot, spatter, reserve, release, wait, dry`. `studio.do({"do": ...})` performs one written as JSON; `vocabulary()` lists them with signatures and examples.
- Regions are JSON-able: `all, disc, band, rect, polygon, below, above`, combined with `or, and, minus, not, grow, shrink`.
- A reserve is the World's frisket (`Canvas.mask`); `release` lifts it.
- `dry(until="damp" | "touch" | "fixed", lamp, spell)` waits in short spells and looks after each: "damp" when at most 10% of the wet cells read above `SHEEN_GLISTENING`; "touch" when `touch_dry()` holds over them, then fixes if every paint laid since the last fix is permanent (oil, acrylic); "fixed" fixes whatever the paint. Judged by sheen and the finger test only.
- `rehearse()`: a studio on a `fork()` of the canvas holding the twins of the rack's brushes; `keep()` makes it the painting, leaving the block without it changes nothing. `choose(variants, by)` rehearses each variant and keeps the one whose photograph `by` scores highest.
- `sitting(name, fn)` checkpoints the whole studio to disk under a key made of the previous key, `name`, the source of `fn`, the values of the globals `fn` names directly and a hash of goo's own source. What those reach indirectly is not in the key.
- `look()` is an sRGB photograph (classic R, G, B); `look(raking=...)` goes through `goo.classic.sensors.relight`; `look_wet()` is `sheen()`; `wetness_picture()` marks glistening, damp and not-touch-dry cells.

Of the canvas, studio code may read only `reflectance, fix, wait, step, W, H` and the World services `sheen, touch_dry, finish, fork, mask, unmask` (`classic/tests/test_studio.py::test_studio_stays_in_the_painters_domain`, rehearsals included).

`goo.classic.easel` is the studio one command at a time with its state on disk, for look-act-look painting from a shell: `python -m goo.classic.easel SESSION new | do JSON | rehearse JSON | look [--wet] [--raking [AZ,EL]] [--region JSON] | back N | journal | help`. `rehearse` takes a list of alternatives (each an act or a list of acts), performs each on a fork and writes `rehearse/alt-K.png` and `sheet.png`; nothing changes. Every command writes `SESSION/look.png` (2× with a ruler in cells) and prints a JSON receipt; errors print `{"error", "fix"}` on stderr and exit 2. A failed `do` changes nothing (a list of acts is all or nothing). `back N` undoes acts from `history/NNNN.pkl`, replaying from an earlier checkpoint when pruned (the first and the last 12 are kept); replay is exact by I4. `classic/tests/test_easel.py` covers the round trip, replay past pruned history, and that `help` lists every act.

`classic/paintings/estuary/paint.py` is the studio's reference painting: a watercolour, 400 × 300 cells (100 × 75 mm) on paper, in four sittings (sky, hills, water, bank), written to `out/estuary/`. Its colour is not scored against anything.

## 12. Paints

`goo.classic.paints.Paint` is a recipe in goo's species: `parts` (fractions of the wet volume), `medium` (`watercolour, gouache, acrylic, oil, water`), `mix` (the colour pigments' shares), `permanent` (oil, acrylic), a default wet `thickness` and a `brushful`. Constructors: `watercolour(mix, strength=0.04)`, `gouache(mix, strength=0.35, opacity=0.05)` (opacified with white), `acrylic(mix, body=0.7, strength=0.5)` (`body` the binder's share of the solids), `oil(mix, fat=0.5)`, `water()`. `load(volume)` gives a `Brush.charge` load. `blend(other, t)` mixes by wet volume, same medium or toward clean water. `to_json()` / `from_json()` round-trip exactly (`test_a_paint_survives_json`). goo's watercolour has no gum: its dry film is pigment alone.

- `preview(paint, thickness, over)` → wet and dry sRGB and reflectance of one film over gesso, a reflectance or a patch, through the engine's own classic functions (`Canvas._KS`, `Canvas.km_layer`, the composition of `reflectance`); it equals a canvas (`test_preview_is_the_canvas_optics_exactly`). Classic only: under classic a water-borne film shows no drying shift.
- `mix_for(colour_srgb, medium, thickness, over, pigments, ...)` → (paint, predicted dry sRGB, ΔE): the mix of up to three pigments (and, for watercolour, the dilution) whose dry film is closest in CIEDE2000, searched on `Palette`'s grid and refined by Nelder–Mead. The error is returned, not hidden.
- `swatch_card(paints, path)` draws a manufacturer's chart (thin, medium, thick; wet and dry; over gesso and a carbon stripe).
- `goo.classic.palette.delta_e2000` is CIEDE2000 via scikit-image. `Palette.match` still ranks mixes by CIE76 distance.

## 13. Data and provenance

`goo.classic.data` loads `goo/classic/data/optics.json` (numpy only, no other part of goo). Per species (11 pigments, 2 grounds, the acrylic binder, water, oil): material, role, `n` and `k`, K and S spectra in mm⁻¹ of pure pigment in each surrounding (oil 1.48, water 1.33, air 1.00; "bulk" for a continuous medium), S and K against the surrounding's index (`sweep`, 11 indices 1.00–1.60), on 42 wavelengths (380–780 nm every 10 nm, and 1550 nm). Every number at every wavelength is marked `computed` (Mie theory from CC0 refractiveindex.info constants, each source's measurement paper cited), `assumed` (a stated physical assumption, e.g. every particle size) or `guessed` (goo's classic value, kept because nothing better was found). Species summaries: computed for white, carbon, sienna, yellow, gesso, paper, acrylic, water; mixed for red and oil; assumed for ivory black; guessed for ochre, ultramarine, rose, green, dioxazine. Method, sources, licences and checks: `classic/docs/OPTICS-DATA.md`. `classic/studies/optics_data.py` regenerates the file; `classic/tests/test_optics_data.py` checks shapes, units, provenance on every number, non-negativity, the qualitative claims, and (when `REFRACTIVEINDEX_DB` and miepython are present) that five species regenerate with no difference.

**Key finding.** Where measured constants exist, the classic table's K and S are ~155× weaker than the computed physical values (geometric mean of 20 non-zero ratios; range 46–490). At physical strength 25 µm of paint at 20% PVC is fully hiding for rutile, carbon, hematite and CdS; at goo's strength it is a glaze. The shipped table therefore mixes two scales (computed species at physical strength, guessed species at goo's), and is **not** wired into the engine or into optics v2. Adopting it means rescaling volumes or treating goo's pigment volume as pigment plus extender: the owner's decision. Only `goo.classic.matter.tables` reads it, as data.

Licence policy: shipped numbers are computed from CC0 inputs. Restricted references (RIT, Golden, IFAC-CNR, LBNL, Handprint, manufacturers' pages) were read for a few sanity checks only; nothing from them is included.

## 14. Matter

`goo.classic.matter` holds substances as descriptions (`classic/docs/MATTER.md`; the intent is `docs/MATTER-SPEC.md`): a dielectric function (`dielectric`: Lorentz, Drude and causal Gaussian oscillators; `Tabulated`; `Free`), a particle morphology (`morphology`: log-normal sizes; `Sphere`, `Coated`, `Hollow`; `Shape`, which every map refuses with `NotDerivable`), and stated maps: Mie (`mie`, checked against miepython to 1e-3) to K = 2μa, S = ¾μs(1 − g) (`response`), packing, intrinsic viscosity, Stokes–Einstein, Stokes settling and evaporation (`mechanics`). `laws` labels every substance `real`, `possible` or `impossible` by named predicates (passivity, causality by a numerical Kramers–Kronig test that fails above a 2% remainder, boundedness, thermodynamic sign, conservation, reciprocity), each returning the amount by which it is met or missed; `km_any` continues the two-flux formulas to K < 0 and finds the lasing threshold. `example.green_from_nowhere` is one impossible substance (optical gain at 540 nm) carried through.

The kernel modules (`dielectric, morphology, mie, response, mechanics, substance, laws, tables`) import nothing from the engine. `classic`, `phenotype`, `search` and `example` read `PIGMENTS`, `GROUND_RGBN`, `Canvas.km_layer` and `goo.classic.palette.lab`; `phenotype` makes scratch canvases. Nothing in `goo.classic.matter` changes a painting's World, and the engine reads nothing from `goo.classic.matter`. `classic/studies/matter.py` runs it end to end; `classic/tests/test_matter.py` tests it.

**Findings** (`classic/docs/MATTER.md` §3, §5):

- The classic pigments are reproduced only as coarse grains, 6–22 µm, with index near oil's (n 1.5–1.65 in oil 1.48): nine of eleven fits. White (0.26 µm, n 1.64: an extender, not titanium) and ultramarine (0.10 µm) are the exceptions. Fits are not unique (12 parameters for 8 numbers).
- At those sizes every classic pigment except white and ultramarine would settle out of a 25 µm wash in under a tick (Stokes, in water, reading one tick as ~2.6 s from the evaporation constant). goo's table and goo's watercolour cannot both be physical at the present scale.
- Brownian diffusivity of those particles is 1e-7 to 1e-4 cells²/tick; goo's `D0 = 0.15` is 1e3–1e6 times larger.
- goo's Krieger–Dougherty exponent is 2 for every pigment; spheres give 1.6–1.8.
- The engine clips negative K silently: the impossible green paints as a near-white (green reflectance 0.97–1.00 over gesso, where the two-flux solution gives 1.02, 1.13, 2.65, then no steady state above 6 µm), with no debt.

**Known gaps** (`goo/classic/matter/gaps.py`, the engine's side; its line numbers refer to an earlier, 551-line `goo/classic/engine.py`). The top rows:

| # | parameter | engine now | the law would read |
|---|---|---|---|
| 1 | K(n_s), S(n_s) per pigment, per band | fixed numbers, defined in no surrounding (in fact oil) | each species' response at the cell's surrounding index |
| 2 | φm (max packing, also CPVC) per particle species | `L["phi_m"] = 0.64` for every pigment | φm = Σ x_s φm,s (first order) |
| 3 | intrinsic viscosity, Krieger exponent | `** -2` for every pigment | −[η]_mix φm, [η]_mix = Σ x_s [η]_s |
| 4 | liquid viscosity | `L["eta0"] = 0.12` for water and oil alike | log-mixing of the liquids present |
| 5 | roles (solid, liquid) | by position: `CARRIERS`, `nc = 2`, `q[nc:]` | by description: particles vs continuous phases |
| 6 | colour diffusivity per pigment | `L["D0"] = 0.15`, one D for all | D_s ~ 1/(η d_s); shear mixing as its own law |
| 14 | gain (K < 0) and its debt | mis-computed and clipped | `laws.km_any`, a threshold test, an energy ledger |

The rest (settling, density, volatility and soak-in per liquid, cure as a declared change, the binder's absorption spectrum, liquid indices, yield onset) are in `gaps.py` and `classic/docs/MATTER.md` §6.

## 15. Status

At this checkpoint `python -m pytest classic/tests -q` gives 215 passed, 1 skipped (the regeneration test of §13, which needs `REFRACTIVEINDEX_DB`).

- **Verified by tests:** I1–I9; every law's exact zero; windowed equals whole; the world services (fork, sheen, touch_dry, frisket, fix by region); the tools' conservation, tape and tool channel; the painters' and the studio's domains (canvas behind glass); the easel's round trip and exact replay; the optics invariants of §6 for both models; `goo.classic.data`'s provenance and units; the matter kernel's Mie against miepython and its laws on constructed cases.
- **Verified by goldens (merge gate, not in pytest; not re-run for this edit):** under classic optics the still life, the pear and the merge's recorded sessions are bit-identical to the pristine engine.
- **Opt-in:** optics v2 (`Canvas(..., optics="v2")`). Classic is the default.
- **Known wrong:** v2 wet/dry shifts for a binderless wash and acrylic (ΔE 21–24, §6); v2 masstones up to ΔE 20.7 from classic; `DetailPainter` refuses v2; relight's own white specular; negative K clipped silently; `D0` not Brownian; classic K, S ~155× below physical values where constants exist; curing overshoot with `lazy > 1` (§8).
- **Not wired:** `goo/classic/data/optics.json` and `goo.classic.matter` feed nothing the engine or the optics models read.
- **Unverified:** every pigment index in `goo.classic.optics.SPECIES`; the classic and v2 colours against any measured paint-out; any reading of ticks as seconds; drying times as real times; hematite's computed K at 1.5 µm (suspect, `classic/docs/OPTICS-DATA.md` check 6); I8 beyond one numpy, scipy and CPU.

### 15.1 Ladder

Where this checkpoint stands on the ladder of `docs/VISION.md` §13:

| rung | | here |
|---|---|---|
| L0 | matter conserved exactly, deterministic, observers do not load | **holds, tested**: I1, I4 (and fork), the optics and world services are pure (`test_contract_models_are_pure_and_picklable`, `test_sheen_and_touch_dry_are_pure`), sensors see only a `Surface` |
| L1 | thermodynamic: a free energy and dissipation potential for the laws as coded, an energy ledger | **not begun**: no energy ledger, no F or Ψ* written for any law |
| L2 | constitutive: every law reads substances | **precursor only**: `goo.classic.matter` derives K, S and mechanics from descriptions, but the engine reads none of it; roles are by position (gaps 1–6) |
| L3 | optical closure | **precursor only**: optics v2 (spectral bands, surroundings, one surface, an associative layer composition) is opt-in and reads `PIGMENTS`, not substance descriptions; no light ledger |
| L4 | grounded in permissive data | **precursor only**: `goo/classic/data` is computed from CC0 constants with provenance and sanity-checked against a few restricted references; it is wired into nothing |
| L5 | predictive against real paint-outs | nothing yet |
| L6 | inverse with identifiability | nothing yet (`goo.classic.matter.search` runs three design searches on the two-flux formula; no identifiability statement exists) |
