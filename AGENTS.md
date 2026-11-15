# Working on goo

Read `SPEC.md` first. It is the contract; the code implements it. `docs/VISION.md` gives the reasons.

- **The break.** Nothing in `goo` outside `goo.classic` imports `goo.classic`. Classic is frozen: see
  `classic/AGENTS.md` before touching anything under `goo/classic/` or `classic/`. Code crosses from
  classic by copying it deliberately with its tests, never by import.
- **Ownership.** `SPEC.md` §13 says who owns which files. Edit only your own. If the contract is wrong
  or lacks something you need, say so in your report; the lead changes it.
- **Layering.** `SPEC.md` §1 says who may import whom. `tests/test_layering.py` enforces it.
- **Units.** SI, written with `goo.units` (`25 * um`). No constant tuned to a cell size or a time step.
- **Evidence.** Every constitutive number has a `Provenance`. Use `goo.evidence.derive` for derived
  values. Ship only numbers computed from CC0 or public-domain data, or guesses marked `guessed`.
  Restricted references may be read and cited for checks; copy nothing from them.
  A number recalled from memory is `assumed` until it has been looked up, whatever source it names.
- **No smuggling.** No branch on a substance's or medium's name or on a position in an array. No
  flag, factor or colour that stands in for a mechanism. If description plus law cannot produce an
  effect yet, extend a law for every substance or say goo does not model it.
- **Ledgers.** Every operation that changes or reads the world returns a `goo.ledger.Receipt` with
  its entries and closure residuals. A quantity that closes an identity by being its remainder is
  named as one (`dissipated[numerical]`) and its sign is tested.
- **Obligations are tests.** Each numbered obligation in `SPEC.md` (O1 to O8, G1 to G7) has a test
  named after it. A test that cannot pass yet is marked `xfail` with the reason, never weakened.
- **Negative results** go in `docs/NEGATIVE.md` with their cause. A disagreement with classic is
  recorded, not tuned away.
- Run `python -m pytest -q` before and after any change. Outputs go to `out/` (git-ignored).
- This repository is public. Test data is generated in code.
