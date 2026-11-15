"""goo.lab.contribute: one file per substance, validate, round trip. Owner: lab."""
import json, copy, pathlib, re
import pytest
from goo.substance import library as lib
from goo.lab.contribute import to_json, from_json, validate, write, FORMAT

DOC = pathlib.Path(__file__).resolve().parents[2] / "docs" / "LAB.md"


@pytest.mark.parametrize("s", lib.ALL, ids=lambda s: s.name)
def test_every_library_entry_round_trips_exactly(s, tmp_path):
    p = write(s, tmp_path / f"{s.name}.json")
    v = validate(p)
    assert v.ok, v.text()
    assert v.substance == s
    assert from_json(json.loads(p.read_text())) == s


def test_the_worked_example_in_docs_validates():
    text = DOC.read_text()
    block = re.search(r"```json\n(.*?)\n```", text, re.S).group(1)
    v = validate(json.loads(block))
    assert v.ok, v.text()
    assert v.klass == "possible"


def _hem():
    return to_json(lib.HEMATITE)


def test_a_size_in_micrometres_is_refused_as_a_unit_error():
    d = _hem(); d["morph"]["d_m"] = 0.3
    v = validate(d)
    assert not v.ok and any("d_m" in e and "wrong unit" in e for e in v.errors)
    d = _hem(); d["eps"]["lam_m"] = [x * 1e9 for x in d["eps"]["lam_m"]]
    assert any("lam_m" in e for e in validate(d).errors)


def test_a_key_without_its_unit_is_refused():
    d = _hem(); d["rho"] = d.pop("rho_kg_m3")
    assert not validate(d).ok


def test_every_field_needs_provenance_and_no_more():
    d = _hem(); del d["prov"]["morph"]
    v = validate(d)
    assert not v.ok and any("no provenance" in e for e in v.errors)
    d = _hem(); d["prov"]["thermo.gamma"] = dict(evidence="guessed")
    assert not validate(d).ok


def test_nothing_is_computed_or_checked_without_a_source_and_licence():
    for ev in ("computed", "checked"):
        d = _hem(); d["prov"]["rho"] = dict(evidence=ev, source="a book", licence="")
        v = validate(d)
        assert not v.ok and any(f"marked {ev} without a source and a licence" in e for e in v.errors)
    d = _hem(); d["prov"]["rho"] = dict(evidence="measured", source="x", licence="y")
    assert not validate(d).ok


def test_a_declared_class_must_be_what_the_laws_give():
    d = _hem(); d["class"] = "real"
    v = validate(d)
    assert not v.ok and any("the laws give 'possible'" in e for e in v.errors)
    g = to_json(lib.GAIN_GREEN)
    v = validate(g)
    assert v.ok and v.klass == "impossible" and v.failed == ("herglotz",)
    g["class"] = "possible"
    assert not validate(g).ok


def test_errors_are_all_reported_at_once():
    d = _hem(); d["morph"]["d_m"] = 0.3; d["prov"]["eps"]["licence"] = ""; d["rho_kg_m3"] = -1
    assert len(validate(d).errors) >= 3


def test_format_tag_is_required():
    d = _hem(); d["format"] = "something else"
    assert not validate(d).ok and FORMAT == "goo.substance/1"


from dataclasses import dataclass, fields, replace
from goo.evidence import Provenance
from goo.substance import Thermo, Rheo, Reaction, StateVar
from goo.lab.contribute import _params_json, key_of, RANGE


@pytest.mark.parametrize("cls,attr", [(Thermo, "thermo"), (Rheo, "rheo")])
def test_every_field_of_thermo_and_rheo_is_carried(cls, attr):
    for f in fields(cls):
        lo, hi = RANGE.get(f.name, (0.0, 1.0))
        val = ((1.0,),) if f.name == "coupling" else lo + 0.25 * (hi - lo)       # a value inside its unit's range
        base = lib.LINOXYN
        s = replace(base, **{attr: cls(**{f.name: val})},
                    prov={k: p for k, p in base.prov if not k.startswith(attr + ".")}
                    | {f"{attr}.{f.name}": Provenance("guessed")})
        v = validate(json.loads(json.dumps(to_json(s, claim_class=False))))
        assert v.substance == s, f"{attr}.{f.name} is not carried: {v.text()}"


def test_reactions_and_state_are_carried():
    s = replace(lib.LINSEED, state=(StateVar("cure", "1", 0.0, 1.0),))
    v = validate(json.loads(json.dumps(to_json(s, claim_class=False))))
    assert v.substance == s, v.text()


def test_a_field_added_to_the_description_cannot_be_dropped_silently():
    @dataclass(frozen=True)
    class Thermo2(Thermo):
        new_term: float = None
    assert key_of("new_term") in _params_json(Thermo2(new_term=3.0))      # carried, under its SI-suffixed key

    @dataclass(frozen=True)
    class Substance2(type(lib.RUTILE)):
        new_field: float = 1.0
    s = Substance2(**{f.name: getattr(lib.RUTILE, f.name) for f in fields(lib.RUTILE)})
    with pytest.raises(TypeError, match="new_field"):
        to_json(s)
