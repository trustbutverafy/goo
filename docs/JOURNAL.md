# goo.journal: the painting as a function of its journal

Owner: ports, journal and clocks. Code: `goo/journal/`. Tests: `tests/journal/`. Contract: SPEC §9.
`goo.journal` imports `goo.world`, `goo.ports`, `goo.light`, `goo.observe` and what they import.

## 1. Session

`Session(initial)` builds a World on a ground and owns its bodies (a `goo.ports.Bench`) and a list of entries.

```text
initial  {"size": [W, H] cells, "dx": "0.25 mm", "ground": "paper" | {"name": ..., "porosity": ..., "pore_radius": ...,
          "depth": ..., "solid": ...}, "substances": [library names], "air": {"humidity": 0.5, "temperature": "20 C",
          "activity": {...}}, "bands": {"step": ..., "ir": [...]}, "seed": 0}
```

The world's substances are fixed when the session starts (the World takes its substance list at construction):
list every substance any paint will use.

`do(act)` takes an act's JSON object, or a list of them done all or nothing. The act is done on a fork of the bench
and kept only if every act succeeds; otherwise `Refused` is raised with the reason and nothing has changed. Each act
appends an entry:

```text
n, act, params (JSON as given, regions and paths by description), seed (spray: drawn as initial seed + entry
number if the painter gave none, and written into params), clock [t0, t1], receipt (digest), world (digest of the
world after the act), summary (one line), wall (s; the only field replay does not reproduce)
```

`replay(entries, initial)` rebuilds the session by doing every act again and checks every receipt digest and every
world digest bit for bit (`ReplayError` otherwise). `back(n)` returns to the state after entry len - n, from a kept
fork (the last 32 are kept in memory) or by replay. `fork()` is an independent session for rehearsal: try acts on
it, look, drop it.

## 2. The vocabulary

Numbers are SI; strings carry units ("5 mL", "0.25 mm", "30 s", "2 min", "15 deg", "20 C", "0.5 uL/s"). Regions
and paths are in cells unless they say `"units": "m"` (docs/PORTS.md §2). `python -m goo.journal DIR vocab` prints
each act with an example; every example is tested to parse.

```text
paint   name, recipe {substance: fraction}            mix     name, parts {paint: parts}
fill    vessel, paint, volume                          pour    vessel, path, volume, duration, width [, min_pieces, edge]
flood   vessel, region, thickness [, duration, pieces, edge] drop    vessel, at, volume [, width]
drip    vessel, path, drops, volume, interval [, width]
spray   vessel, path, flow, duration, cone_radius, droplet_size [, seed, min_pieces, edge]
tape    name, region                                   peel    name
tilt    angle [, direction: down|up|left|right|angle]  level
air     [humidity, activity {name: a}, temperature, boundary_layer]
wait    duration [, rate, where, steps]                hold    region          release [region]
settle  [where, horizon, length, force]
```

`humidity` is the activity of the atmosphere's reservoir of the substance named `water`: the definition of
relative humidity, a key, not a branch.

Sizing a ground is an ordinary sequence: a paint of binder solution, `fill`, `flood` a thin film, `wait`, `settle`.

## 3. Looking

`look(light="daylight"|"A", exposure=1, region=None, gloss=False, infrared=False)` solves goo.light on the world's
stack (read-only views: `World.stack()`; a region crops to its bounding box) and returns a `Seen`: the Look, an sRGB
image (goo.observe colour, specular excluded, adapted to the light), and if asked the gloss reading (60 degrees, in
GU; grey image with 100 GU white) and the infrared reading (bands above 780 nm; grey image of the last band, linear).
**Exposure** is the camera's: linear sRGB is multiplied by it before the sRGB encoding and its clip, so a film with
gain, which can return more light than white, can be seen unclipped. A look changes nothing (tested by digest).
The default bands are 380-740 nm by 20 nm and one infrared band at 1000 nm (100 nm wide).

## 4. Behind glass

A painter reads the world only through looks and receipts. No public method of a Session returns a World, Film,
Stack, Ground, Bench, or an array that shares memory with the world's state (tested by walking every return value
and `np.shares_memory`); the session has no public attribute for the world; the command line uses only the
session's public interface (tested on its source). The vessels and tapes a painter holds (`vessels()`) are the
painter's own bodies, not the world.

## 5. The command line

```text
python -m goo.journal DIR new --size 160x120 --dx 0.25mm --ground paper --substances water,gum_arabic,hematite
                              [--porosity --pore-radius --depth --humidity 0.5 --temperature "20 C" --seed 0]
python -m goo.journal DIR do '<json act or list of acts>'      prints each entry's summary, world time, timings
python -m goo.journal DIR look [--region JSON] [--scale N] [--gloss] [--ir] [--exposure X] [--light A] [--name TAG]
python -m goo.journal DIR log | back N | fork NAME | use NAME | branches | vessels | vocab | replay
```

State in DIR: `current` (the branch), `branches/NAME/journal.json` (the journal: portable JSON, the truth),
`branches/NAME/state.pkz` (the session after the last entry), `ckpt/NNNN.pkz` (checkpoints at entry 0, every 10
entries with the 3 newest kept, and on request with `checkpoint`, never thinned), `lookcache.npz` (the picture
cache) and `looks/*.png`. `.pkz` files are pickles written by `goo.journal.store`: numpy arrays zlib-compressed,
the world's Grid rebuilt from its size, ledgers saved as balances (their entry lists are history the journal
keeps). They are a cache, **not portable** across goo, numpy or Python versions. Loading checks the state's last
world digest against the journal's. `back N` loads the newest checkpoint at or before the target and replays the
journal from it, checking every digest. The command line runs the session with no kept forks and acts done in
place (it saves only on success, so a refused act changes nothing on disk); this spares a copy of every panel
array per act. An act that fails prints `refused (nothing changed): why` and exits 1.

`look --scale` takes an integer (cells repeated) or a fraction 1/k (0.5, 0.25: k x k cells averaged in linear
sRGB, before the encoding); `--region` crops at the asked scale, full resolution by default.

On 1024 x 768 with 9 substances (gesso; paint, fill, hold all, flood 800 x 500 cells, forced settle, a drop):

```text
                         before          after
disk after 7 entries     2.0 GB          0.94 MB (state 0.5 MB, checkpoint 0.4 MB)
growth per entry         ~240 MB         what the act changed, compressed (a checkpoint every 10)
held flood (do)          4.7 s           2.1 s (1.0 s of it Python start, load, digest, save)
forced settle (do)       5.4 s           1.8 s
small act (do)           1.0-1.5 s       0.9-1.0 s
look, full panel, cold   30 s            13.5 s
look --scale 0.25, warm  (no fractions)  1.0 s
look after a small pour  30 s            4.0 s (only changed cells solved)
look --region, warm      30 s            0.9 s
```

## 6. Pictures and the optics cache

`Session.picture(...)` (and `look` on the command line) keeps, per branch, the linear sRGB (and gloss and infrared
when asked) of every cell with two 64-bit keys of that cell's inputs: the ground's z and held amounts and the
amounts of each film present in the cell, in order. A film absent from a cell is exactly the identity in
goo.light (tested), so a wet film retired into a stratum keeps its key. Cells whose keys are unchanged under the
same light and bands are not solved again; the rest are solved on a compacted stack ([S, 1, n]: light is
element-wise over cells) and scattered back. Tested bit for bit against a fresh look. The keys are a 128-bit
hash, not verified: a collision would show a stale cell.

Both looks and pictures read matter through `goo.journal.looking.Optics`: goo.substance.maps evaluated once per
distinct cell (grouped by the bits of every per-cell input, verified exactly) and gathered back; values are those
of maps, bit for bit; the counts in out-of-domain reasons' texts then count distinct compositions, not cells.

## 7. The constitutive adapter

The session's world reaches matter through `goo.journal.session.Maps`, which is `goo.substance.maps` by delegation
with `mixing` evaluated once per distinct cell (exact: mixing's text carries no counts; tested bit for bit),
and with one numerical guard on the mixing Hessian (docs/PORTS.md §8 item 4): the Hessian, used only in Newton
Jacobians, is evaluated with positive amounts raised to at least 1e-250 m. Without it most scenes of a pigment
drying on paper fail with a singular Jacobian. The guard belongs in goo.world (or goo.substance.maps) and should be
removed from here when they handle vanishing amounts.

## 8. Unverified

Replay across processes and machines (tested in one process and through the CLI's pickles on one machine); pickles
across versions; the infrared image's band choice (the last band) as a stand-in for a reflectogram's detector.
