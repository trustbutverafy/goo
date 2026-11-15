"""goo.classic.data: the optical data table (classic/docs/OPTICS-DATA.md). Loader, units, provenance, physical orderings, the rutile
check, and (when the CC0 database and miepython are present) regeneration from source."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from goo.classic import data

ROOT = Path(__file__).resolve().parents[2]
T = data.load()
W = T.wavelength_nm
PARTICLES = [n for n in T.names if T.species(n).sweep_S is not None]
KINDS = {"computed", "assumed", "guessed"}


def _generator():
    spec = importlib.util.spec_from_file_location("optics_data", ROOT / "classic" / "studies" / "optics_data.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


# ---- loader --------------------------------------------------------------------------------------------------
def test_grid_and_shapes():
    assert W.tolist() == list(range(380, 781, 10)) + [1550]
    for n in T.names:
        sp = T.species(n)
        assert sp.n.shape == sp.k.shape == W.shape
        for q in ("K", "S"):
            assert getattr(sp, q).keys() == sp.K.keys()
            for v in getattr(sp, q).values(): assert v.shape == W.shape
        if sp.role in ("pigment", "ground"): assert set(data.SURROUNDINGS) <= sp.K.keys()
        if sp.role == "carrier": assert set(sp.K) == {"bulk"}
        if sp.sweep_S is not None: assert sp.sweep_S.shape == (len(T.sweep_n_medium), len(W))
    assert T.species("white").n.flags.writeable is False


def test_round_trip(tmp_path):
    raw = json.loads(data.PATH.read_text())
    p = tmp_path / "copy.json"; p.write_text(json.dumps(raw))
    U = data.load(p)
    assert U.names == T.names
    for n in T.names:
        a, b = T.species(n), U.species(n)
        assert np.array_equal(a.n, b.n) and a.summary == b.summary and a.role == b.role
        for s in a.K: assert np.array_equal(a.K[s], b.K[s]) and np.array_equal(a.S[s], b.S[s])


def test_resample():
    const = np.full(W.shape, 3.0)
    assert np.allclose(data.resample(const, data.GOO_BANDS), 3.0)
    lin = W.astype(float).copy()                         # value = wavelength
    out = data.resample(lin, [(400, 500), (433, 467), 555.0, 550.0, 1550.0])
    assert np.allclose(out, [450, 450, 555, 550, 1550])
    two = np.stack([lin, 2 * lin])                       # leading axes are carried through
    assert data.resample(two, [(600, 700)]).shape == (2, 1)
    with pytest.raises(ValueError): data.resample(lin, [1000.0])
    with pytest.raises(ValueError): data.resample(lin, [(700, 900)])


def test_units_and_classic_shape():
    tab = T.classic_table("oil")
    assert set(tab) == set(T.by_role("pigment"))
    for v in tab.values(): assert len(v["K"]) == len(v["S"]) == 4
    w = T.species("white")
    assert np.isclose(tab["white"]["S"][3], w.S["oil"][-1] * 0.025)


def test_guessed_species_keep_goos_values_in_oil():
    """A guessed pigment in oil is goo's classic table, converted (x 1/0.025 mm), at the band centres."""
    goo = _generator().GOO_CLASSIC
    for n in T.names:
        sp = T.species(n)
        if sp.summary != "guessed" or sp.role != "pigment": continue
        for q in ("K", "S"):
            got = data.resample(sp.__getattribute__(q)["oil"], [650.0, 550.0, 450.0, 1550.0]) * data.GOO_UNIT_MM
            assert np.allclose(got, goo[n][q], rtol=1e-3), (n, q)


def test_no_runtime_dependency_on_mie():
    code = "import sys, goo.classic.data as d; d.load(); print('miepython' in sys.modules, 'yaml' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.stdout.split() == ["False", "False"], r.stderr
    for p in (ROOT / "goo" / "classic").rglob("*.py"): assert "miepython" not in p.read_text(), p


# ---- provenance ----------------------------------------------------------------------------------------------
def test_every_number_has_provenance():
    for n in T.names:
        sp = T.species(n)
        for q in ("n", "K", "S"):
            assert sp.provenance[q].shape == W.shape and set(sp.provenance[q]) <= KINDS, (n, q)
        assert sp.summary in KINDS | {"mixed"}
        kinds = set(sp.provenance["K"]) | set(sp.provenance["S"])
        if sp.summary == "computed": assert "guessed" not in kinds
        if sp.summary == "guessed": assert kinds == {"guessed"}
        if "computed" in kinds: assert sp.sources, n           # computed numbers carry their papers
        for s in sp.sources: assert s["reference"] and s["file"]


def test_which_species_stand_on_physics():
    summ = {n: T.species(n).summary for n in T.names}
    assert {n for n, s in summ.items() if s == "computed"} >= {"white", "carbon", "sienna", "yellow", "gesso", "paper", "water"}
    assert {n for n, s in summ.items() if s == "guessed"} == {"ultramarine", "ochre", "rose", "green", "dioxazine"}
    assert summ["ivory_black"] == "assumed" and summ["red"] == "mixed"


# ---- physics -------------------------------------------------------------------------------------------------
def test_K_and_S_non_negative_and_finite():
    for n in T.names:
        sp = T.species(n)
        for q in ("K", "S"):
            for v in getattr(sp, q).values(): assert np.all(np.isfinite(v)) and np.all(v >= 0), (n, q)
        if sp.sweep_S is not None: assert np.all(sp.sweep_S >= 0) and np.all(sp.sweep_K >= 0)


def _low_contrast(sp, i):
    """Index above all three surroundings, below 2, non-absorbing: the regime where S is set by index contrast."""
    return 1.50 < sp.n[i] < 2.0 and sp.k[i] < 1e-4


def test_scatters_more_in_air_than_water_than_oil():
    """Where a non-absorbing particle's index is above all three surroundings (and below 2), S(air) > S(water) >
    S(oil) at every wavelength; for every particulate species it holds for the R, G, B band means. (Pointwise it can
    fail for rutile at 380, 420 and 1550 nm and carbon at 1550 nm: at a large relative index, a particle sized for oil
    is past its optimum in air. classic/docs/OPTICS-DATA.md.)"""
    checked = 0
    for n in PARTICLES:
        sp = T.species(n)
        for i in range(len(W)):
            if not _low_contrast(sp, i): continue
            a, w, o = sp.S["air"][i], sp.S["water"][i], sp.S["oil"][i]
            assert a > w > o > 0, (n, W[i], a, w, o)
            checked += 1
        a, w, o = (data.resample(sp.S[s], data.GOO_BANDS[:3]) for s in ("air", "water", "oil"))
        assert np.all(a > w) and np.all(w > o), n
    assert checked > 150


def test_scattering_falls_toward_index_match():
    """S falls as the surrounding's index rises toward the particle's (flat to 3% for micron grains near n = 1, the
    geometric regime), and vanishes at a match."""
    checked = 0
    for n in PARTICLES:
        sp = T.species(n)
        for i in range(len(W)):
            if not _low_contrast(sp, i): continue
            s = sp.sweep_S[T.sweep_n_medium < sp.n[i] - 0.02, i]
            assert np.all(np.diff(s) < 0.03 * s[:-1]) and s[-1] < 0.5 * s[0], (n, W[i]); checked += 1   # flat to 3% where large
    assert checked > 150
    ac = T.species("acrylic"); i550 = int(np.where(W == 550)[0][0])
    j1, jm = list(T.sweep_n_medium).index(1.0), list(T.sweep_n_medium).index(1.49)
    assert abs(ac.n[i550] - 1.49) < 0.01
    assert ac.sweep_S[jm, i550] < 1e-3 * ac.sweep_S[j1, i550]


def test_index_match_is_purely_absorbing():
    pytest.importorskip("miepython")
    gen = _generator()
    K, S, _, _ = gen.mie_km(1.6 + 0.0j, 0.3, 1.5, 0.55, 1.6)
    assert S == 0 and K == 0
    prev = None
    for k in (0.05, 0.005, 0.0005):                          # real parts matched: S goes as k^2, K as k
        K, S, _, _ = gen.mie_km(1.6 + 1j * k, 0.3, 1.5, 0.55, 1.6)
        assert K > 0 and S < 0.01 * K
        if prev: assert S < 0.02 * prev[1] and K < 0.2 * prev[0]
        prev = (K, S)


def test_rutile_against_the_coatings_literature():
    """LBNL pigment commentary (white): Mie s ~ 12 um^-1 (another author 10.4) for 0.22 um rutile in a binder of 1.5
    at 550 nm. Ours: mu_s(1 - g) = S / 0.75 for d50 = 0.25 um in oil (1.48). Within a factor of 1.5."""
    w = T.species("white"); i = int(np.where(W == 550)[0][0])
    mus_r = w.S["oil"][i] / 0.75 / 1e3                    # um^-1
    assert 12.0 / 1.5 < mus_r < 12.0 * 1.5, mus_r
    assert w.S["oil"][i] > 3 * w.S["oil"][-1]               # scattering falls toward the infrared


def test_qualitative_colour_and_infrared():
    i = {x: int(np.where(W == x)[0][0]) for x in (450, 550, 650, 1550)}
    ks = lambda n, x: T.species(n).K["oil"][i[x]] / max(T.species(n).S["oil"][i[x]], 1e-12)
    assert ks("sienna", 650) < ks("sienna", 550) < ks("sienna", 450) * 1.5    # hematite: red-brown
    assert ks("yellow", 650) < 1e-3 and ks("yellow", 450) > 1                 # CdS: yellow (absorbs blue only)
    assert ks("carbon", 550) > 10 and ks("carbon", 1550) > 10                 # carbon: black, also at 1.5 um
    assert ks("white", 550) == 0 and ks("yellow", 1550) < 1e-6                # rutile, CdS: transparent at 1.5 um


# ---- regeneration --------------------------------------------------------------------------------------------
def test_regenerates_from_the_cc0_database():
    db = os.environ.get("REFRACTIVEINDEX_DB")
    if not db or not (Path(db) / "database" / "data").is_dir(): pytest.skip("REFRACTIVEINDEX_DB not set")
    pytest.importorskip("miepython"); pytest.importorskip("yaml")
    gen = _generator()
    only = ["white", "water", "acrylic", "sienna", "ultramarine"]
    new = gen.build(db, only=only)
    old = json.loads(data.PATH.read_text())
    old["species"] = {k: v for k, v in old["species"].items() if k in only}
    assert gen.diff(new, old) == []
