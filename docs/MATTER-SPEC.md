# goo: the matter constitution

## Decision
goo is an arbitrary metamaterial. A species is not a name in a table. It is a point in a space of
constitutive descriptions, and the known paints (ultramarine in linseed, a gum-bound wash, an acrylic
emulsion) are particular points in it. The same laws act on every point. The space also contains
substances no one has made, and substances that cannot exist; the model must be able to hold all three
and must always know which is which.

## What a substance is
A substance is described at the scale where physics is simple, and everything the engine uses is
derived from that description by stated maps. Nothing the engine reads is typed in by hand when it
can be derived.

- **Matter**: a dielectric function over wavelength (how it bends and absorbs light), and for
  particles a morphology (size distribution, shape, internal structure such as a shell or a void).
- **Mechanics**: how it packs, how it thickens a liquid, how it yields; for a liquid, how it flows,
  evaporates, wets, soaks in, mixes with other liquids or refuses to.
- **Change**: what it turns into and when: drying, curing, coalescing, re-dissolving; and in general
  any internal state that evolves and feeds back on the other two.

From these, derived maps give the engine its numbers: scattering and absorption *as functions of what
surrounds the particle*; viscosity and yield as functions of loading; diffusion and settling from
size; evaporation, cure and absorption rates. One description, every law: change a particle's size
and its colour, its hiding power, its diffusion, its settling and its packing all move together,
lawfully. That coupling is what makes it a material and not a set of dials.

A substance therefore presents itself to the engine as **response functions of its environment**, not
as constants: K and S given the index around it, viscosity given the loading, and so on.

## Three kinds of point, always labelled
- **real**: tied to a material that exists, by measured or computed constants with provenance.
- **possible**: no known material, but it satisfies every admissibility law below.
- **impossible**: it violates named laws. Impossible substances are allowed. They are never silent:
  each violation is named, and where the violation amounts to getting something for nothing (light
  out with none in, flow without a force) the substance must draw on a declared, finite reservoir and
  the ledger records what was drawn. An impossibility is a debt with a receipt, not a cheat.

## Admissibility (the laws a possible substance obeys)
- Passivity: it absorbs or passes light, it does not amplify it. (Im of the dielectric function >= 0.)
- Causality: its refraction and absorption are two faces of one response (Kramers-Kronig consistent).
- Boundedness: indices, strengths and sizes within what the sum rules and the geometry allow.
- Thermodynamic sign: viscosity, diffusivity, yield, evaporation and cure rates are non-negative; time
  runs one way for irreversible changes.
- Conservation: volume in equals volume out, per species, including what it turns into.
- Reciprocity where claimed: a layer looks the same to light going either way unless it says otherwise.
Each is a predicate with a name. A substance carries the list it passes and the list it fails.

## The anti-smuggling rule
A new effect is admitted only as a consequence of a substance's description acting through the
common laws. A "pearlescent" flag, a "dries lighter" factor, a per-medium branch in the optics, a
colour typed into a table: these are smuggling. If an effect cannot yet be produced by description
plus law, the honest states are: extend a law (for every substance), or say goo does not model it.

## Phenotype
What a painter or a sensor would measure of a paint made from a substance: hiding power, tinting
strength, masstone and undertone, drying shift, gloss, levelling and brush-mark retention, open time,
lifting, granulation, what it does in the infrared. Measured by running standard tests in the engine
(a drawdown, a brush-out, a wash, a drying curve), never read off the description. A substance's data
sheet is generated, like a manufacturer's, by making the paint and looking.

## Inverse design
Given a phenotype wanted and the admissibility laws, find a description: a pigment that fills a hole
in the gamut, a paint that is black wet and white dry, an ink visible only at 1.5 um. The search
reports the nearest *possible* substance and, separately, what impossible substance would hit the
target and which laws it breaks. Mechanistic and behavioural claims stay separate: "this description
produces this behaviour in goo" is one claim; "a real material with this description exists" is
another, and only `real` makes it.

## Exact limits
The classic species are points in the space: with their descriptions, every derived number reproduces
today's table and today's paintings bit for bit. Removing a mechanism (no index contrast, no particles,
no volatility) recovers the simpler material exactly.

## What this asks of the engine (in order)
1. Optics reads response functions, not constants (the v2 optics model is the first consumer).
2. Roles by description, not by position: today `water` and `oil` are indices 0 and 1 with their laws
   hard-wired and every pigment thickens paint identically. Each law must read its parameters from the
   substances present and combine them by a stated mixing rule.
3. Interaction: liquids that do not mix; binders that re-dissolve or do not; internal state.
4. An energy ledger beside the volume ledger, so an impossible substance's debt has somewhere to go.
