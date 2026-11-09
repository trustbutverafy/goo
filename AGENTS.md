# Working on goo

Read `SPEC.md` first. It is the contract; the code implements it.

- Run `python -m pytest -q` before and after any change to `goo/`. All tests must pass.
- Never write colour or height into the World from a painter or study script. Move paint with tools; set up tests with `Canvas.add` (it books what it adds).
- A new law needs: a switch in `Canvas.L`, conservative form through the limiter (or an explicit ledger entry), and a test that it is exactly zero when off.
- Sensors take a `Surface` and nothing else from the World. Keep ground truth beside a measurement, never inside it.
- Units are in `SPEC.md` §2. Brush loads are in volume units (0.0156 µL each); a real brush holds thousands.
- Outputs go to `out/` (git-ignored). Figures worth keeping go to `docs/`.
