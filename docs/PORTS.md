# goo.ports: bodies, sources, masks, the environment and clocks

Owner: ports, journal and clocks. Code: `goo/ports/`. Tests: `tests/ports/`. Contract: SPEC §3, §6, §8.
`goo.ports` imports `goo.world`, `goo.film`, `goo.substance`, `goo.ledger`, `goo.units`, `goo.evidence`.

A `Bench` holds a `World` and the bodies coupled to it. Every act is a method that returns one
`goo.ledger.Receipt`: the world's own receipts of the act (adds, steps, retirements, environment changes) summed,
the bench's own entries (vessels filled, matter caught by tape, overspray, spill), a matter closure over every
account the act could touch, and the act's privileges in one flag.

## 1. Paints and vessels

A `Paint` is a name and a recipe, `{substance: volume fraction}`, validated against the world's substances
(fractions >= 0, summing to 1 within 1e-9, then renormalised exactly). It names no medium; nothing branches on it.
Mixing paints by parts of volume makes a new recipe.

A `Vessel` holds a mass of each substance of one paint. Its matter account is `body[name]`, filled from `origin`.
Every source draws from a vessel and books what it lays from `body[name]`: to `wet` through `World.add` (with the
free energy the matter brings, from `work[body[name]]`), to a tape's `body[tape]`, or off the panel to
`reservoir[spill]` (pours, drops) or `reservoir[overspray]` (spray). A vessel that cannot give what is asked gives
what it holds; the receipt reports `shortfall m^3`, the summary says VESSEL RAN DRY, and an empty vessel lays
nothing. What is left after many draws below 1e-12 of what was put in is the rounding of the draws ("dust") and is
not laid.

## 2. Regions, paths, footprints (`goo.ports.geometry`)

Cell coordinates by default: x along columns, y down rows, cell (i, j) centred at (j + 1/2, i + 1/2); a cell is in
a shape when its centre is. `"units": "m"` in a description gives metres from the panel's top-left corner. Regions:
`"all"`, `rect [x0, y0, x1, y1]`, `disc [cx, cy, r]`, `polygon [[x, y], ...]`, `band {"rows"|"cols": [a, b]}`,
`union`, `intersection`, `difference` (the first minus the rest). Paths: a point `[x, y]`, a polyline, or
`{"points": ..., "smooth": n}` (Chaikin corner cutting, ends kept).

A footprint is flat-topped: every cell whose centre lies within the radius of the spout's segment gets the same
share. A tapering profile (a swept disc's chord, a paraboloid cap) was tried first and puts vanishingly thin films
on its edge cells, which today's world cannot step (§8); flat footprints have a step at their edge instead, which
the world's slope flag reports. The fraction of a footprint off the panel is computed on the extended grid and
booked to a reservoir.

## 3. Sources: deposition without contact

Each is resolved in time: matter arrives in pieces and the world runs between them.

```text
pour(vessel, path, volume, duration, width)   constant flow along the path at constant speed; pieces: the spout moves at most
                                              half its footprint per piece, at least min_pieces (4); each piece lays V/n, then
                                              the world runs duration/n. duration 0: laid, no time passes
flood(vessel, region, thickness, duration, pieces)   an even film: a pour whose spout covers the region
drop(vessel, at, volume, width)               one drop as a flat disc of diameter `width` (default: that of a hemisphere of
                                              its volume); no time passes; spreading is the world's
drip(vessel, path, drops, volume, interval)   drops evenly along the path, the world running `interval` between them
spray(vessel, path, flow, duration, cone_radius, droplet_size, seed)
```

Spray: monodisperse droplets of diameter `droplet_size`; their number is the volume over one droplet's, rounded,
each carrying an equal share. Each droplet's time is uniform over the act, its nozzle position the path's point
at that time, its landing point uniform in a disc of `cone_radius` about it, all from `numpy.random.default_rng(seed)`.
A droplet lands whole in the cell holding its centre; off the panel it is overspray. The seed is required and goes
in the journal (the session draws one if the painter gives none). **Not modelled**: evaporation in flight
(flash-off), so droplets arrive with the paint's composition; impact, splash and bounce; drift in air; a size
distribution; a droplet larger than a cell spreading over several cells on landing (the world spreads it).

## 4. Tape

`tape(name, region)` lays a mask on dry cells (refused over wet film). Every source's share for taped cells goes to
the tape's account `body[name]`; nothing is laid there. The tape is also a wall to flow, done by the clock: its
cells run at rate 0, and since a face runs at the smaller of its cells' rates, nothing crosses into or out of
them. **This is the privileged clock doing a wall's job until the world has real walls**, and every receipt of an
act during which a tape is on says so in a `privilege` flag. `peel(name)` frees the cells; what the tape caught
leaves with it (its account keeps the mass; the receipt reports `peeled kg`).

## 5. The environment

```text
tilt(angle, direction)    gravity (g0 sin a cos d, g0 sin a sin d, g0 cos a); d from +x (columns) toward +y (rows)
level()                   (0, 0, g0)
air(activity, T, boundary_layer)   the atmosphere's reservoirs; humidity is water's activity (the journal's "humidity")
```

Each is `World.set_environment`, whose change of F is booked as `work[environment]`. An activity of 0 is refused
(SPEC §8).

## 6. Clocks and settle

`wait(duration, rate=1, where=None)` lets time pass with the clock rate `rate` in `where` (default everywhere), 0
in held and taped cells, 1 elsewhere. With no hold, tape or rate the world runs unprivileged (`rate=None`). A rate
field != 1 anywhere makes the world flag each step; the bench sums those into one flag,
`privilege/painter`, whose amount is the integral of r over cells and time (m² s), and adds `privilege/clock`
flags saying why (held cells, tape as a wall, a local rate). A negative rate is refused with a clear error before
anything runs (reverse is not implemented). `hold(region)` and `release(region)` set and clear held cells.

When the world has `advance(duration, rate=...)`, wait and the sources' pieces call it. Until then the bench's
stepper takes `steps` implicit steps (default 10) whose sizes double, the last ending exactly at the duration's
sum; a source's piece is one step. Each step is taken on a fork and adopted only if the world accepts it: a step
the world gives up on (RuntimeError) has already committed part of itself, so it is never taken on the bench's
world. It is retried as three steps of a third, to a depth of 2, each retry counted in `world step retries`.

`settle(where, horizon, length, force)`: with `World.settle` (and no region or force) the world decides. Until then a
Look-free rule from the state and the constitutive maps: a wet cell can still flow if its viscosity is finite, its
yield stress (if any) is below the driving stress h(ρ|g_t| + (ρ g_n + γ k²) h k), and

```text
rate = h³/(3η) [ (ρ g_n + γ k²) k² + ρ |g_t| / (h ℓ) ],   k = 2π/ℓ,   rate × horizon ≥ 1
```

(levelling of a ripple of wavelength ℓ under gravity and capillarity, and drainage down a tilt over ℓ; defaults
horizon one day, ℓ = 1 mm). Cells that cannot flow are retired with `World.retire` (matter wet -> stratum[k], their
free energy stored -> frozen). `force=True` retires every wet cell in `where`: a privilege, flagged with the number
of cells that could still flow.

## 7. Receipts

Every act's receipt: the world's entries and the bench's, in the world's ledgers, in order; the residuals (matter:
the larger of the world's and the bench's closure, which compares every account it can — wet, ground, strata,
every vessel and tape, overspray and spill — with the state, over the mass held; energy: the world's); F by term
before and after; the flags (privileges consolidated, other flags deduplicated); and the report summed (the world's
dissipation by mechanism, D_num, max slope, rate integral) with the act's own numbers (laid m^3, shortfall,
droplets, seed, pieces, cells retired, retries).

## 8. What the world lacked, and how it is worked around

1. `World.advance`, `World.settle` and walls: absent; §6 and §4. `World.advance` is called with `rate=` if its
   signature has it, and refused if a hold or tape is on and it has not.
2. **Sorption on a window** (goo/world/world.py `World._coefficients`, and goo/world/local.py `local`): a windowed
   film is passed with the whole panel's `Ground` to `sorption`, and `maps.sorption` fails to broadcast
   `ground.held` (panel-shaped) to the window. Any `step(window="auto")` with a porous ground whose wet region is
   smaller than the panel raises ValueError. Worked around: the bench steps with `window="panel"` (the solves are
   on the active set either way; only element-wise work grows).
3. **Acceptance flips with the last bits of dt** (World.step): on a recorded state, dt = 0.2 s is accepted and
   dt = 0.20000000000000018 s fails after 24 halvings. Reproduction: `out/ports/repro/step_fails_one_ulp.py`.
   Worked around: fork-and-retry (§6), and step sizes are never computed as remainders.
4. **Singular diffusion Jacobian** (goo/world/diffuse.py `diffuse`, goo/substance/maps.py `mixing`): the Newton
   iteration in ln q drives a vanishing amount to 1e-155 m and below; `maps.mixing`'s Hessian (c/q on the diagonal,
   maps.py line ~966) overflows to inf and NaN; `splu` raises `RuntimeError: Factor is exactly singular`, which is
   not a `Reject`, so the world does not halve, and the step fails for every dt. It happened on most scenes of a
   pigment in gum water drying on paper. Reproduction: `out/ports/repro/diffusion_jacobian_inf.py`. Worked around
   in the journal's constitutive adapter (`goo.journal.session.Maps`): the Hessian, which enters only the Newton
   Jacobians, is evaluated with positive amounts raised to at least 1e-250 m; f and μ are the true ones. The
   bench itself does not apply the guard; the ports tests use scenes the world steps without it.
5. **Thin fronts**: flow puts films of 1e-13 m and less into cells next to a front in one step. Sources lay flat
   footprints (§2) so that they do not add to it.
6. **Thick pours are slow**: a pour of 20 uL along a 2 mm wide, 11 mm path (0.9 mm of film, slope 0.56) took 423 s
   on 48 x 64 cells; the same pour of 2 uL took 0.9 s. Pours of a few tens of micrometres are the first rung's
   domain anyway (the thin-film slope flag fires above 0.1).

## 9. What is privileged or faked, and unverified

Privileged and flagged: hold, local rates, tape as held cells, `settle(force)`. Not modelled: walls (tape is the
clock), spray flight and impact, any mechanical port (no brush, blade or pad this round: the world cannot take a
wall riding on the film). Unverified: the spray's coverage against a real gun; the footprints' shapes against
real spouts; the settle rule against any drying measurement (it is a statement about flow time scales, with a
painter's horizon, not a drying law).
