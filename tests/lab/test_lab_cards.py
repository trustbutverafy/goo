"""goo.lab drawdown, sheet, admissible, sensitivity, render. Owner: lab."""
import json, pathlib
from dataclasses import replace
import numpy as np, pytest
from PIL import Image
from goo.substance import library as lib, classify
from goo.lab import recipes as R
from goo.lab.drawdown import drawdown
from goo.lab.sheet import sheet
from goo.lab.admissible import admissible
from goo.lab.sensitivity import sensitivity, variants, DN
from goo.lab.render import render_card, contact_sheet, tag_of

SMALL = dict(binders=(R.OIL, R.WATERCOLOUR), tints=(0.0, 0.9), bands=R.bands(step=40e-9))


@pytest.fixture(scope="module")
def hem():
    return drawdown(lib.HEMATITE, **SMALL)


def test_every_number_on_a_card_has_an_evidence_class_and_its_flags_exist(hem):
    ids = {f["id"] for f in hem.flags}
    nums = []
    for s in hem.data["swatches"]:
        nums += [c["lab"] for c in s["colour"].values()] + [c["XYZ"] for c in s["colour"].values()]
        nums += [s["infrared"]["R"], s["thickness"]]
    nums += [g["number"] for k in ("gloss", "hiding", "drying_shift") for g in hem.data[k]]
    for n in nums:
        assert n["evidence"] in ("proved", "exact", "computed", "checked", "assumed", "guessed")
        assert set(n["flags"]) <= ids
    assert len(hem.data["swatches"]) == 2 * 2 * 2 * 2          # binders x states x tints x grounds
    json.loads(hem.to_json())


def test_out_of_domain_maps_make_the_numbers_assumed_and_say_why(hem):
    s = hem.data["swatches"][0]
    assert s["evidence"] in ("assumed", "guessed")
    kinds = {hem.flags[i]["kind"] for i in s["flags"]}
    assert "out_of_domain" in kinds
    assert any("independent scattering" in f["subject"] for f in hem.flags)


def test_dry_oil_numbers_are_guessed_because_the_cure_and_linoxyn_are(hem):
    rec = next(r for r in hem.data["recipe"]["binders"] if r["binder"] == "linseed oil")
    assert rec["weakest"]["dry"]["evidence"] == "guessed"
    assert any("linoxyn" in w for w in rec["weakest"]["dry"]["because"])


def test_black_and_white_grounds_are_real_stacks_that_a_thin_glaze_tells_apart():
    thin = drawdown(R.Paint("glaze", ((lib.HEMATITE, 1.0),), R.OIL, 0.01), tints=(0.0,), bands=R.bands(step=40e-9),
                    h_wet=5e-6)
    sw = {s["ground"]: s for s in thin.data["swatches"] if s["state"] == "wet"}
    assert sw["black"]["colour"]["daylight"]["lab"]["value"][0] < sw["white"]["colour"]["daylight"]["lab"]["value"][0] - 20


def test_a_card_does_not_depend_on_a_substances_name():
    a = drawdown(lib.CDS, binders=(R.OIL,), tints=(0.0,), bands=R.bands(step=40e-9))
    b = drawdown(replace(lib.CDS, name="anything"), binders=(R.OIL,), tints=(0.0,), bands=R.bands(step=40e-9))
    for x, y in zip(a.data["swatches"], b.data["swatches"]):
        assert x["colour"]["daylight"]["lab"]["value"] == y["colour"]["daylight"]["lab"]["value"]


def test_an_impossible_substance_is_flagged_with_its_law_and_shortfall_on_every_swatch():
    c = drawdown(lib.GAIN_GREEN, binders=(R.OIL,), tints=(0.0,), bands=R.bands(step=40e-9))
    imp = [f for f in c.flags if f["kind"] == "impossible" and f["subject"] == "gain_green"]
    assert imp and "herglotz" in imp[0]["detail"] and imp[0]["amount"] == pytest.approx(0.05, rel=1e-4)
    for s in c.data["swatches"]:
        assert imp[0]["id"] in s["flags"]
    acausal = drawdown(lib.ACAUSAL_BLUE, binders=(R.OIL,), tints=(0.0,), bands=R.bands(step=40e-9))
    assert any(f["kind"] == "impossible" and f["subject"] == "acausal_blue" for f in acausal.flags)


def test_a_guessed_substance_looks_like_a_guess_on_the_image(tmp_path, hem):
    um_ = drawdown(lib.ULTRAMARINE, **SMALL)
    pg, ph = render_card(um_, tmp_path / "g.png"), render_card(hem, tmp_path / "h.png")
    g, h = np.asarray(Image.open(pg)).astype(float), np.asarray(Image.open(ph)).astype(float)
    # the wet oil masstone over white: hatched (not one colour) for the guess, flat for hematite
    box = (slice(248, 270), slice(160, 200))
    assert g[box].std(axis=(0, 1)).max() > 3.0
    assert h[box].std(axis=(0, 1)).max() < 0.5
    fid = {f["id"]: f for f in um_.flags}
    assert tag_of(um_.data["swatches"][0], fid).startswith("G")


def test_sheet_measures_by_looking_and_hiding_falls_with_loading():
    lo = sheet(R.Paint("lo", ((lib.RUTILE, 1.0),), R.OIL, 0.05), bands=R.bands(step=40e-9))
    hi = sheet(R.Paint("hi", ((lib.RUTILE, 1.0),), R.OIL, 0.20), bands=R.bands(step=40e-9))
    k = "linseed oil: hiding thickness (dry, CR 0.98)"
    assert hi[k]["value"] < lo[k]["value"]
    t = [k for k in hi if k.startswith("linseed oil: tinting")][0]
    assert hi[t]["evidence"] in ("assumed", "guessed")
    cds = sheet(lib.CDS, binders=(R.OIL,), bands=R.bands(step=40e-9))
    t = [k for k in cds if k.startswith("linseed oil: tinting")][0]
    band_nm = float(t.rsplit(" at ", 1)[1].split(" nm")[0])
    assert cds[t]["value"] > 0.1 and band_nm < 500                      # a yellow absorbs blue


def test_admissible_reports_every_law_with_its_shortfall():
    r = admissible(lib.GAIN_GREEN)
    assert r["klass"] == "impossible" and r["failed"] == ["herglotz"]
    h = next(v for v in r["laws"] if v["law"] == "herglotz")
    assert not h["ok"] and h["shortfall"] == pytest.approx(0.05, rel=1e-4)
    assert len(r["laws"]) == 5
    for s in lib.ALL:
        assert admissible(s)["klass"] == classify(s).klass
    assert any(b["field"] == "morph" for b in admissible(lib.HEMATITE)["blocking"])


def test_sensitivity_variants_change_size_and_index_and_ranges_contain_the_nominal():
    v = variants(lib.HEMATITE)
    assert v["d x0.5"].morph.size.d == pytest.approx(0.5 * lib.HEMATITE.morph.size.d)
    assert v["d x2"].morph.size.d == pytest.approx(2 * lib.HEMATITE.morph.size.d)
    lam = np.array([600e-9])
    n0 = np.real(lib.HEMATITE.eps.nk(lam))                  # uniaxial: each part's n is shifted by DN
    assert np.real(v["n +0.05"].eps.nk(lam)) == pytest.approx(n0 + DN, abs=2e-4)
    for s in (lib.CDS, lib.ULTRAMARINE):                    # one part: exactly n + DN, k kept
        w = variants(s)
        assert np.real(w["n +0.05"].eps.nk(lam)) == pytest.approx(np.real(s.eps.nk(lam)) + DN, abs=1e-12)
        assert np.imag(w["n -0.05"].eps.nk(lam)) == pytest.approx(np.imag(s.eps.nk(lam)), abs=1e-12)
    c = drawdown(lib.HEMATITE, binders=(R.OIL,), tints=(0.0,), bands=R.bands(step=40e-9))
    s = sheet(lib.HEMATITE, binders=(R.OIL,), bands=R.bands(step=40e-9))
    sensitivity(lib.HEMATITE, c, s, binders=(R.OIL,), bands=R.bands(step=40e-9))
    for g in c.data["gloss"] + c.data["hiding"]:
        lo, hi = g["number"]["range"]
        assert lo - 1e-12 <= g["number"]["value"] <= hi + 1e-12
    k = "linseed oil: hiding thickness (dry, CR 0.98)"
    assert s[k]["range"][0] <= s[k]["value"] <= s[k]["range"][1] and s[k]["range"][0] < s[k]["range"][1]


def test_contact_sheet_is_phone_sized(tmp_path, hem):
    p = contact_sheet([hem, hem], tmp_path / "c.png")
    w, h = Image.open(p).size
    assert w <= 1200 and h < 2400


def test_where_a_look_has_no_steady_state_the_card_gives_no_number_and_says_why():
    c = drawdown(lib.GAIN_GREEN, binders=(R.OIL,), tints=(0.0,), bands=R.bands(step=40e-9))
    lasing = [s for s in c.data["swatches"] if s["colour"]["daylight"]["unresolved"]]
    assert lasing, "the drawdown loading should put gain_green past its threshold"
    for s in lasing:
        assert s["colour"]["daylight"]["lab"]["value"] is None and "steady state" in s["colour"]["daylight"]["lab"]["note"]
    assert all(g["number"]["value"] is None for g in c.data["hiding"])
    sh = sheet(lib.GAIN_GREEN, binders=(R.OIL,), bands=R.bands(step=40e-9))
    assert sh["linseed oil: hiding thickness (dry, CR 0.98)"]["value"] is None
