"""SPEC §4.5: the library. Every entry is classified (by the laws, never typed), every field it gives has a
Provenance, "real" entries meet the rule, admissible entries hold no reservoir, and the guessed pigments are on the
physical scale."""
import numpy as np
import pytest
from goo.evidence import EVIDENCE
from goo.film import Film, Bands
from goo.substance import classify, library as L, maps
from goo.substance.laws import STRONG
from goo.units import um, nm

EXPECTED = {
    "real": {"water", "pmma"},
    "possible": {"ethanol", "linseed_oil", "linoxyn", "gum_arabic", "rutile", "soot", "hematite", "cadmium_sulfide",
                 "calcite", "cellulose", "pmma_latex", "ochre", "ultramarine", "rose", "green", "dioxazine"},
    "impossible": {"gain_green", "acausal_blue", "negative_friction", "mass_from_nothing"},
}
FAILS = {"gain_green": ("herglotz",), "acausal_blue": ("herglotz",), "negative_friction": ("dissipative",),
         "mass_from_nothing": ("stoichiometric",)}


@pytest.mark.parametrize("s", L.ALL, ids=lambda s: s.name)
def test_every_entry_is_classified_as_expected(s):
    c = classify(s)
    want = [k for k, v in EXPECTED.items() if s.name in v]
    assert want == [c.klass], (s.name, str(c), c.blocking)
    if c.klass == "impossible":
        assert c.failed == FAILS[s.name]


def test_the_expected_table_covers_the_library():
    assert set().union(*EXPECTED.values()) == set(L.BY_NAME)


@pytest.mark.parametrize("s", L.ALL, ids=lambda s: s.name)
def test_every_field_has_a_provenance(s):
    keys = dict(s.prov)
    assert set(keys) == set(s.fields_present())
    for k, p in keys.items():
        assert p.evidence in EVIDENCE
        if p.evidence in ("computed", "checked"):
            assert p.source and p.licence


@pytest.mark.parametrize("s", [s for s in L.ALL if classify(s).klass == "real"], ids=lambda s: s.name)
def test_real_entries_meet_the_rule(s):
    assert s.material
    assert all(p.evidence in STRONG for _, p in s.prov)
    assert s.reservoir == 0.0


@pytest.mark.parametrize("s", L.ALL, ids=lambda s: s.name)
def test_reservoirs(s):
    c = classify(s)
    if c.klass != "impossible":
        assert s.reservoir == 0.0
    if "herglotz" in c.failed and s.name == "gain_green":       # a failure that yields power has something to draw
        assert s.reservoir > 0


def test_cc0_entries_cite_their_measurement_and_the_database():
    for s in (L.RUTILE, L.SOOT, L.HEMATITE, L.CDS, L.CALCITE, L.CELLULOSE, L.PMMA_LATEX, L.WATER, L.ETHANOL, L.PMMA):
        p = s.provenance("eps")
        assert p.licence == "CC0 1.0 Universal" and "refractiveindex.info" in p.source
        assert s.eps.citation
    assert "Devore" in L.RUTILE.provenance("eps").source
    assert "Hale" in L.WATER.provenance("eps").source
    # transparent-range sources leave k = 0 assumed, and the provenance says so
    assert L.RUTILE.provenance("eps").evidence == "assumed" and "k = 0 assumed" in L.RUTILE.provenance("eps").note
    assert L.SOOT.provenance("eps").evidence == "computed"


def test_promoted_constants_agree_with_classics_table():
    """The kernel regenerates n, k from the CC0 database; at the wavelengths both describe, they must agree with
    classic's optics.json to its rounding (classic's ordinary-ray n for the uniaxial entries)."""
    import json, pathlib
    d = json.load(open(pathlib.Path(__file__).resolve().parents[2] / "goo" / "classic" / "data" / "optics.json"))
    lam = np.asarray(d["wavelength_nm"]) * 1e-9
    pairs = [("white", L.RUTILE), ("carbon", L.SOOT), ("sienna", L.HEMATITE), ("yellow", L.CDS),
             ("gesso", L.CALCITE), ("acrylic", L.PMMA_LATEX), ("water", L.WATER)]
    for key, s in pairs:
        e = d["species"][key]
        ok = s.eps.in_range(lam) & (lam <= 780e-9)
        m = sum(w * m for w, m in s.eps.parts(lam[ok]))
        assert np.allclose(m.real, np.asarray(e["n"])[ok], rtol=2e-3, atol=2e-3), key
        assert np.allclose(m.imag, np.asarray(e["k"])[ok], rtol=2e-2, atol=2e-3), key


def test_guessed_pigments_are_on_the_physical_scale():
    """Classic's guessed table was ~155x too weak. At 20 % in linseed oil, each guessed pigment's peak K is of the
    order of the computed pigments' (hundreds per mm), not classic's few per mm."""
    b = Bands.uniform()
    for s in (L.OCHRE, L.ULTRAMARINE, L.ROSE, L.GREEN, L.DIOXAZINE):
        f = Film.uniform({L.LINSEED: 40 * um, s: 10 * um})
        K = maps.optical(f, b).K.value[:, 0, 0] / 0.2           # per unit pigment volume, 1/m
        assert 1e5 < K.max() < 1e8, (s.name, K.max())
        assert s.provenance("eps").evidence == "guessed"
