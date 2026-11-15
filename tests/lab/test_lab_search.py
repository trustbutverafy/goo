"""goo.lab.search and the command line, on small budgets. Owner: lab."""
import json, warnings
import numpy as np, pytest
from goo.substance import library as lib, classify, Oscillators, Gauss, eV
from goo import observe as O
from goo.lab import search as S, recipes as R
from goo.lab.__main__ import main

B = R.bands(step=40e-9)


def test_target_spectrum_has_exactly_the_target_colour():
    R_p = np.linspace(0.2, 0.6, B.B)
    t = (62.0, -38.0, -22.0)
    Rt = S.target_spectrum(R_p, B, t)
    lab_ = O.lab(O.xyz(Rt, B, O.DAYLIGHT), O.white_xyz(B, O.DAYLIGHT))
    assert O.ciede2000(lab_, t) < 1e-6 and np.all(Rt >= 0)


def test_solve_k_hits_a_passive_target_and_needs_gain_for_a_brighter_one():
    s = S.candidate("x", Oscillators(2.25, ()), 0.3e-6, "test")
    mk = lambda sub: R.stack(S.film_of(sub, R.OIL, 0.1, h=20e-6, state="wet"), "white")
    vis = np.asarray(B.centres) < 780e-9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = S._see(mk(s), B).diffuse[0, 0]
        dim, Rr = S.solve_k(s, mk, np.where(vis, 0.5 * base, base), B, vis, iters=30)
        assert np.allclose(Rr[vis], 0.5 * base[vis], rtol=2e-2)
        assert classify(dim).klass in ("possible", "impossible")
        bright, Rr = S.solve_k(s, mk, np.where(vis, np.minimum(1.0, 1.15 * base), base), B, vis, iters=30)
    k = np.imag(bright.eps.nk(np.asarray(B.centres)[vis]))
    assert np.all(k < 0)                                   # more light out than in: gain
    c = classify(bright)
    assert c.klass == "impossible" and "herglotz" in c.failed


def test_the_library_gamut_is_measured_by_making_films():
    lab_, mixes = S.library_gamut(palette=(lib.RUTILE, lib.CDS), steps=2, bands=B)
    assert len(mixes) == len(lab_) == 3
    names = [m[0] for m in mixes]
    i_white = names.index(("rutile",))
    assert lab_[i_white][0] > 90 and abs(lab_[i_white][1]) < 3


def test_a_searched_substance_is_a_guess_and_says_so():
    s = S.candidate("y", Oscillators(2.2, (Gauss(1.0, 2.0 * eV, 0.2 * eV),)), 0.2e-6, "a note")
    c = classify(s)
    assert c.klass == "possible"
    assert all(e == "guessed" for k, e in c.blocking if k in ("eps", "morph", "rho"))


def test_command_line_admissible_validate_export_and_fast_card(tmp_path, capsys):
    out = str(tmp_path)
    assert main(["admissible", "gain_green", "--out", out]) == 0
    assert "FAIL herglotz" in capsys.readouterr().out
    assert main(["export", "hematite", "--out", out]) == 0
    assert main(["validate", str(tmp_path / "hematite.substance.json")]) == 0
    d = json.loads((tmp_path / "hematite.substance.json").read_text()); d["morph"]["d_m"] = 300.0
    (tmp_path / "bad.json").write_text(json.dumps(d))
    assert main(["validate", str(tmp_path / "bad.json")]) == 1
    assert main(["card", "soot", "--fast", "--out", out]) == 0
    assert (tmp_path / "soot.png").exists() and json.loads((tmp_path / "soot.json").read_text())["sheet"]


def test_fast_optical_is_the_maps_rearranged_and_defers_to_them_when_voids_appear():
    from goo.substance import maps
    s = S.candidate("fast", Oscillators(2.6, (Gauss(1.2, 2.0 * eV, 0.15 * eV),)), 0.12e-6, "test")
    f = S.film_of(s, R.OIL, R.OIL.pvc, white_share=0.5)
    K1, S1, n1 = S.FastOptical(s).optical(f, B)
    K0, S0, n0 = (x.value for x in maps.optical_exact(f, B))
    assert np.max(np.abs(K1 - K0)) <= 1e-12 * np.max(K0) and np.max(np.abs(S1 - S0)) <= 1e-6 * np.max(S0)
    assert np.array_equal(n1, n0)
    dry_water = S.film_of(s, R.WATER, 0.5)                       # underbound: voids, so the maps answer it whole
    out = S.FastOptical(s).optical(dry_water, B)
    assert type(out.K).__name__ == "Bounded"
