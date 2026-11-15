"""goo.journal: the painting as a function of its journal (SPEC §9; docs/JOURNAL.md). Replay bit for bit, looks that
change nothing, the world behind glass, failing acts that change nothing, back, rehearsal, and the command line."""
import json
import pathlib
import subprocess
import sys
import warnings
import numpy as np
import pytest
from goo.ledger import Receipt
from goo.film import Film, Stack, Ground
from goo.world import World
from goo.journal import Session, replay, Refused, ReplayError, vocabulary, ACTS
from goo.journal import acts as A
from goo.journal.quantities import quantity, QuantityError
from goo.journal.cli import main
from goo.ports import Bench

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = pathlib.Path(__file__).resolve().parents[2]
SEALED = {"name": "calcite", "porosity": 0.0, "pore_radius": "1 um", "depth": "100 um"}


def initial(W=32, H=24, ground=SEALED, humidity=0.95):
    return dict(size=[W, H], substances=["water", "gum_arabic", "hematite"], ground=ground,
                air={"humidity": humidity})


BASE = [
    {"act": "paint", "name": "red", "recipe": {"water": 0.8, "gum_arabic": 0.15, "hematite": 0.05}},
    {"act": "paint", "name": "clear", "recipe": {"water": 0.85, "gum_arabic": 0.15}},
    {"act": "fill", "vessel": "cup", "paint": "red", "volume": "1 mL"},
    {"act": "fill", "vessel": "jar", "paint": "clear", "volume": "1 mL"},
]


def session(acts=(), **kw):
    s = Session(initial(**kw))
    for a in BASE + list(acts):
        s.do(a)
    return s


def test_quantities():
    assert quantity("5 mL", "volume") == pytest.approx(5e-6)
    assert quantity("0.25mm", "length") == pytest.approx(2.5e-4)
    assert quantity("2 min", "time") == 120.0
    assert quantity("90 deg", "angle") == pytest.approx(np.pi / 2)
    assert quantity("20 C", "temperature") == pytest.approx(293.15)
    assert quantity(0.3, "length") == 0.3
    with pytest.raises(QuantityError):
        quantity("5 parsecs", "length")
    with pytest.raises(QuantityError):
        quantity("5 s", "length")


def test_every_example_in_the_vocabulary_is_a_valid_act():
    for name, v in vocabulary().items():
        n, p = A.parse(v["example"])
        assert n == name and set(p) == set(ACTS[name]["spec"])
    with pytest.raises(A.ActError, match="unknown parameter"):
        A.parse({"act": "pour", "vessel": "cup", "path": [1, 1], "volume": 1e-9, "duration": 1, "width": 1e-3,
                 "brush": 1})
    with pytest.raises(A.ActError, match="required"):
        A.parse({"act": "pour", "vessel": "cup"})


def test_replay_is_bit_identical():
    acts = [
        {"act": "pour", "vessel": "cup", "path": [[4, 12], [16, 12]], "volume": "0.3 uL", "duration": "1 s",
         "width": "1.5 mm"},
        {"act": "spray", "vessel": "jar", "path": [[2, 3], [30, 3]], "flow": "0.1 uL/s", "duration": "1 s",
         "cone_radius": "1 mm", "droplet_size": "80 um"},
        {"act": "tilt", "angle": "10 deg", "direction": "right"},
        {"act": "hold", "region": {"rect": [0, 16, 32, 24]}},
        {"act": "wait", "duration": "2 s", "steps": 3},
        {"act": "release"},
        {"act": "settle", "force": True, "where": {"rect": [0, 0, 16, 24]}},
    ]
    s = session(acts)
    es = s.entries
    assert es[BASE.__len__() + 1]["seed"] is not None                    # spray drew a seed into the journal
    r = replay(es, s.initial)
    assert r.digest() == s.digest()
    assert [e["receipt"] for e in r.entries] == [e["receipt"] for e in es]
    bad = json.loads(json.dumps(es))
    bad[-3]["params"]["duration"] = "2.5 s"
    with pytest.raises(ReplayError):
        replay(bad, s.initial)


def test_each_entry_is_json_with_its_clock_seed_digest_and_summary():
    s = session([{"act": "drop", "vessel": "cup", "at": [10, 10], "volume": "0.05 uL"},
                 {"act": "wait", "duration": "1 s", "steps": 2}])
    e = s.entries[-1]
    json.dumps(s.journal())
    assert e["act"] == "wait" and e["clock"][1] - e["clock"][0] == pytest.approx(1.0)
    assert len(e["receipt"]) == 64 and len(e["world"]) == 64 and "residual" in e["summary"]
    assert "wall" in e["summary"] and e["wall"] >= 0


def test_a_look_changes_nothing_and_exposure_is_the_cameras():
    s = session([{"act": "pour", "vessel": "cup", "path": [[4, 12], [26, 12]], "volume": "0.4 uL", "duration": 0,
                  "width": "2 mm"}])
    d = s.digest()
    a = s.look(gloss=True, infrared=True)
    assert s.digest() == d
    b = s.look(exposure=2.0)
    assert s.digest() == d
    assert a.image.shape == (24, 32, 3) and a.image.dtype == np.uint8
    assert a.receipt.residual["light"] <= 1e-12
    assert np.all(b.image.astype(int) >= a.image.astype(int)) and b.image.mean() > a.image.mean()
    assert a.gloss.value.shape == (24, 32) and a.gloss.unit == "GU"
    wet = np.zeros((24, 32), bool); wet[10:14, 6:24] = True
    # gloss sees the wet film (water, n 1.33, over polished calcite, n 1.66, reads lower than bare calcite)
    assert abs(a.gloss.value[wet].mean() - a.gloss.value[0:3].mean()) > 1.0
    assert a.infrared.value.shape[:2] == (24, 32) and a.infrared_image.shape == (24, 32)
    part = s.look(region={"rect": [4, 8, 20, 16]})
    assert part.box == (8, 16, 4, 20) and part.image.shape == (8, 16, 3)
    assert np.array_equal(part.image, a.image[8:16, 4:20])


def _world_arrays(s):
    w = s._bench.world
    return [w.q, w.ground.held, w.ground.z, w.zb] + list(w.strata)


def _walk(x, seen=None):
    seen = set() if seen is None else seen
    if id(x) in seen:
        return
    seen.add(id(x))
    yield x
    if isinstance(x, dict):
        for v in x.values():
            yield from _walk(v, seen)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _walk(v, seen)
    elif hasattr(x, "__dataclass_fields__"):
        for k in x.__dataclass_fields__:
            yield from _walk(getattr(x, k), seen)


def test_the_world_is_behind_glass():
    """Nothing a painter gets from a session is the world, a film, a stack, a ground, the bench, or an array that
    shares memory with the world's state; a session has no public way to them; the command line uses only the
    session's public interface."""
    s = session()
    outs = [s.do({"act": "pour", "vessel": "cup", "path": [[4, 12], [26, 12]], "volume": "0.3 uL", "duration": "1 s",
                  "width": "2 mm"}),
            s.do([{"act": "wait", "duration": "1 s", "steps": 2}]),
            s.look(gloss=True, infrared=True), s.entries, s.log(), s.vessels(), s.paints(), s.initial, s.journal(),
            s.time, s.size, s.digest()]
    state = _world_arrays(s)
    for out in outs:
        for x in _walk(out):
            assert not isinstance(x, (World, Film, Stack, Ground, Bench)), type(x)
            if isinstance(x, np.ndarray):
                assert not any(np.shares_memory(x, a) for a in state)
    public = [n for n in dir(s) if not n.startswith("_")]
    assert not {"world", "bench", "stack", "q", "film"} & set(public)
    src = (ROOT / "goo" / "journal" / "cli.py").read_text()
    assert "._bench" not in src and ".world" not in src


def test_an_act_that_fails_changes_nothing_and_says_why():
    s = session()
    d, n = s.digest(), len(s.entries)
    for bad, why in [({"act": "pour", "vessel": "nope", "path": [1, 1], "volume": 1e-9, "duration": 1, "width": 1e-3},
                      "no vessel"),
                     ({"act": "wait", "duration": "1 s", "rate": -1}, "backwards"),
                     ({"act": "paint", "name": "x", "recipe": {"water": 0.5}}, "sum to"),
                     ({"act": "fly"}, "unknown act"),
                     ([{"act": "flood", "vessel": "jar", "region": "all", "thickness": "5 um"},
                       {"act": "tape", "name": "t", "region": "all"}], "wet")]:
        with pytest.raises(Refused, match=why):
            s.do(bad)
        assert s.digest() == d and len(s.entries) == n


def test_back_returns_to_an_earlier_entry_by_kept_forks_and_by_replay():
    s = session([{"act": "drop", "vessel": "cup", "at": [10, 10], "volume": "0.05 uL"}])
    mid = s.digest()
    s.do({"act": "wait", "duration": "1 s", "steps": 2})
    s.do({"act": "drop", "vessel": "cup", "at": [20, 10], "volume": "0.05 uL"})
    s.back(2)
    assert s.digest() == mid and len(s.entries) == len(BASE) + 1
    s.do({"act": "wait", "duration": "1 s", "steps": 2})
    t = s.fork()
    t._kept.clear()                                            # no kept forks: back must replay
    t.back(1)
    assert t.digest() == mid


def test_rehearsal_on_a_fork_leaves_the_session():
    s = session()
    d = s.digest()
    r = s.fork()
    r.do({"act": "flood", "vessel": "cup", "region": "all", "thickness": "5 um"})
    seen = r.look()
    assert seen.image.mean() != s.look().image.mean()
    assert s.digest() == d and len(r.entries) == len(s.entries) + 1
    assert s.vessels()[0]["cup"][1] > r.vessels()[0]["cup"][1]


def test_sizing_a_ground_is_an_ordinary_sequence_of_acts():
    s = Session(initial(ground={"name": "paper"}, humidity=0.5))
    s.do({"act": "paint", "name": "size", "recipe": {"water": 0.9, "gum_arabic": 0.1}})
    s.do({"act": "fill", "vessel": "jar", "paint": "size", "volume": "1 mL"})
    s.do({"act": "flood", "vessel": "jar", "region": "all", "thickness": "10 um"})
    s.do({"act": "wait", "duration": "5 min", "steps": 8})
    r = s.do({"act": "settle", "force": True})
    assert r.residual["matter"] <= 1e-12
    assert r.report["cells retired"] == 32 * 24 or r.report["cells retired"] + r.report["cells still wet"] == 32 * 24
    w = s._bench.world
    gi = [x.name for x in w.substances].index("gum_arabic")
    assert len(w.strata) == 1 and w.strata[0][gi].sum() > 0       # a gum stratum lies on the paper


@pytest.mark.xfail(reason="goo.substance.maps / goo.world: water in a gum solution sorbs into no pores: its mixing "
                          "potential (ideal mixing, ~ -1e7 J/m^3 at 10 % gum) is far below the paper's suction "
                          "(~1.5e4 Pa), so a size never wets the paper's pores; pure water does (docs/PORTS.md §8)",
                   strict=True)
def test_a_size_soaks_into_paper():
    s = Session(initial(W=8, H=8, ground={"name": "paper"}, humidity=0.5))
    s.do({"act": "paint", "name": "size", "recipe": {"water": 0.9, "gum_arabic": 0.1}})
    s.do({"act": "fill", "vessel": "jar", "paint": "size", "volume": "1 mL"})
    s.do({"act": "flood", "vessel": "jar", "region": "all", "thickness": "10 um"})
    r = s.do({"act": "wait", "duration": "1 s", "steps": 2})
    assert r.moved("matter", dst="ground").sum() > 0


def test_the_command_line_round_trips_a_small_session(tmp_path, capsys):
    d = str(tmp_path / "s")
    assert main([d, "new", "--size", "32x24", "--ground", "calcite", "--porosity", "0", "--pore-radius", "1um",
                 "--depth", "100um", "--substances", "water,gum_arabic,hematite", "--humidity", "0.95"]) == 0
    for a in BASE:
        assert main([d, "do", json.dumps(a)]) == 0
    assert main([d, "do", json.dumps([{"act": "pour", "vessel": "cup", "path": [[4, 12], [26, 12]],
                                       "volume": "0.3 uL", "duration": "1 s", "width": "2 mm"},
                                      {"act": "wait", "duration": "1 s", "steps": 2}])]) == 0
    out = capsys.readouterr().out
    assert "body[cup]->wet" in out and "residual" in out and "world time" in out
    assert main([d, "do", '{"act": "wait", "duration": "1 s", "rate": -2}']) == 1
    assert "refused" in capsys.readouterr().out
    assert main([d, "look", "--scale", "2", "--gloss", "--ir"]) == 0
    paths = [l for l in capsys.readouterr().out.splitlines() if l.endswith(".png")]
    assert len(paths) == 3 and all(pathlib.Path(p).exists() for p in paths)
    from PIL import Image
    assert Image.open(paths[0]).size == (64, 48)
    assert main([d, "log"]) == 0
    log = capsys.readouterr().out.splitlines()
    assert len(log) == len(BASE) + 2
    assert main([d, "fork", "alt"]) == 0 and main([d, "use", "alt"]) == 0
    assert main([d, "do", '{"act": "drop", "vessel": "cup", "at": [5, 5], "volume": "0.02 uL"}']) == 0
    assert main([d, "use", "main"]) == 0
    assert main([d, "back", "1"]) == 0
    assert main([d, "replay"]) == 0
    assert "bit for bit" in capsys.readouterr().out
    j = json.loads((tmp_path / "s" / "branches" / "main" / "journal.json").read_text())
    alt = json.loads((tmp_path / "s" / "branches" / "alt" / "journal.json").read_text())
    assert len(j["entries"]) == len(BASE) + 1 and len(alt["entries"]) == len(BASE) + 3
    r = subprocess.run([sys.executable, "-m", "goo.journal", d, "vocab"], capture_output=True, text=True,
                       cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0 and "spray" in r.stdout


# ------------------------------------------------------------------------------------------- size and pictures
HELD = [{"act": "hold", "region": "all"}]


def test_a_saved_session_is_the_size_of_its_arrays_and_continues_identically(tmp_path):
    from goo.journal import store
    s = session(HELD + [{"act": "flood", "vessel": "cup", "region": {"rect": [2, 2, 20, 12]}, "thickness": "8 um"},
                        {"act": "settle", "force": True}], W=96, H=64)
    blob = store.dumps(s)
    raw = sum(a.nbytes for a in _world_arrays(s))
    assert len(blob) < 0.2 * raw
    t = store.loads(blob)
    assert t.digest() == s.digest() and t.entries == s.entries
    more = {"act": "flood", "vessel": "jar", "region": {"disc": [60, 30, 8]}, "thickness": "5 um"}
    s.do(more); t.do(more)
    assert t.digest() == s.digest() and t.entries[-1]["receipt"] == s.entries[-1]["receipt"]


def test_the_command_line_keeps_few_checkpoints_and_goes_back_through_them(tmp_path, capsys):
    from goo.journal import cli
    d = str(tmp_path / "s")
    main([d, "new", "--size", "48x32", "--ground", "calcite", "--porosity", "0", "--pore-radius", "1um",
          "--depth", "100um", "--substances", "water,gum_arabic,hematite"])
    for a in BASE + HELD:
        assert main([d, "do", json.dumps(a)]) == 0
    for k in range(20):
        assert main([d, "do", json.dumps({"act": "drop", "vessel": "cup", "at": [2 + 2 * k, 10],
                                          "volume": "0.002 uL"})]) == 0
    bdir = tmp_path / "s" / "branches" / "main"
    ck = sorted(p.name for p in (bdir / "ckpt").iterdir())
    assert ck == ["0000.pkz", "0010.pkz", "0020.pkz"]
    j = json.loads((bdir / "journal.json").read_text())
    assert main([d, "checkpoint"]) == 0
    assert main([d, "back", "7"]) == 0                       # 25 -> 18: checkpoint 10, then 8 entries replayed
    s, _ = cli._load(bdir)
    assert len(s.entries) == 18 and s.digest() == j["entries"][17]["world"]
    assert "0025.keep.pkz" not in {p.name for p in (bdir / "ckpt").iterdir()}
    total = sum(p.stat().st_size for p in bdir.rglob("*") if p.is_file())
    assert total < 2_000_000
    capsys.readouterr()


def test_a_picture_reuses_unchanged_cells_exactly_and_scales_down_in_linear_light():
    s = session(HELD + [{"act": "flood", "vessel": "cup", "region": {"rect": [2, 2, 20, 12]}, "thickness": "8 um"},
                        {"act": "settle", "force": True},
                        {"act": "flood", "vessel": "jar", "region": {"disc": [24, 16, 5]}, "thickness": "5 um"}],
                W=40, H=28)
    cache = {}
    p = s.picture(cache=cache, gloss=True, infrared=True)
    L = s.look(gloss=True, infrared=True)
    assert p.solved == p.cells == 40 * 28
    assert np.array_equal(p.image, L.image) and np.array_equal(p.gloss_image, L.gloss_image)
    assert np.array_equal(p.infrared_image, L.infrared_image)
    s.do({"act": "drop", "vessel": "cup", "at": [35, 5], "volume": "0.01 uL"})
    q = s.picture(cache=cache, gloss=True, infrared=True)
    L2 = s.look(gloss=True, infrared=True)
    assert 0 < q.solved < 20 and np.array_equal(q.image, L2.image) and np.array_equal(q.gloss_image, L2.gloss_image)
    s.do({"act": "settle", "force": True})                   # wet film -> stratum: same optics, nothing to solve
    assert s.picture(cache=cache).solved == 0
    half = s.picture(cache=cache, scale=0.5)
    assert half.image.shape == (14, 20, 3) and half.solved == 0
    from goo.observe import colour as C
    rgb = cache["rgb"]
    want = C.srgb8(rgb.reshape(14, 2, 20, 2, 3).mean((1, 3)))
    assert np.array_equal(half.image, want)
    crop = s.picture(cache=cache, region={"rect": [10, 4, 30, 20]})
    assert crop.image.shape == (16, 20, 3) and np.array_equal(crop.image, s.look().image[4:20, 10:30])
    with pytest.raises(ValueError, match="fraction"):
        s.picture(scale=0.3)


def test_the_adapters_give_the_maps_answers_bit_for_bit():
    """Session.Maps.mixing and looking.Optics evaluate maps once per distinct cell and gather: exact."""
    from goo.substance import maps, library as L
    from goo.film import Film, Ground, Stack, Light
    from goo.light import look
    from goo.journal.session import Maps
    from goo.journal.looking import OPTICS, bands_of
    subs = (L.WATER, L.GUM_ARABIC, L.HEMATITE)
    rng = np.random.default_rng(1)
    q = np.zeros((3, 64, 80)); q[:, :, :40] = np.array([16e-6, 3e-6, 1e-6])[:, None, None]
    q[:, 5:15, 50:70] = rng.uniform(0, 5e-6, (3, 10, 20))
    a, b = maps.mixing(Film(subs, q)), Maps().mixing(Film(subs, q))
    assert all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a.value, b.value)) and a.map == b.map
    bands = bands_of({})
    g = Ground(np.zeros((64, 80)), L.CELLULOSE, 0.5, 5e-6, 2e-4, subs)
    st = Stack(g, [Film(subs, q), Film(subs, q[:, ::-1])], 1e-4)
    l1, l2 = look(st, bands, Light(bands, np.ones(bands.B)), maps), look(st, bands, Light(bands, np.ones(bands.B)), OPTICS)
    for k in ("reflectance", "specular", "diffuse", "absorbed", "ground", "transmitted"):
        assert np.array_equal(getattr(l1, k), getattr(l2, k)), k
