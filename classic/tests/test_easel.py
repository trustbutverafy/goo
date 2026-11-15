"""The easel (goo.classic.easel): one command at a time, all state in the session directory."""
import hashlib, io, json, os, subprocess, sys
from contextlib import redirect_stdout
import numpy as np
import pytest
from goo.classic import easel

SKY = {"do": "wash", "region": {"band": [0, 20]}, "paint": "ultramarine"}
SEA = {"do": "wash", "region": {"band": [20, 40]}, "paint": "2 ultramarine + rose"}


def run(*args, lines=False):
    buf = io.StringIO()
    with redirect_stdout(buf): code = easel.main(list(args))
    assert code == 0
    out = buf.getvalue().strip().splitlines()
    if lines: return out
    try: return json.loads(out[-1])
    except (json.JSONDecodeError, IndexError): return out


def digest(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_easel_round_trip(tmp_path):
    s = str(tmp_path / "session")
    r = run(s, "new", "--size", "48x40", "--pigments", "ultramarine,rose,ochre")
    assert r["ok"] and os.path.exists(os.path.join(s, "look.png"))
    r = run(s, "do", json.dumps(SKY))
    assert r["acts"] == 1 and r["receipts"][0]["given"]["ultramarine"] > 0
    after_sky = easel.Session(s).load().cv.reflectance()
    r = run(s, "do", json.dumps([{"do": "dry", "until": "touch"}, SEA]))
    assert r["acts"] == 3
    state = digest(os.path.join(s, "state.pkl"))
    r = run(s, "rehearse", json.dumps([{"do": "blot", "region": {"disc": [24, 30, 6]}}, [{"do": "wait", "ticks": 10}, {"do": "lift", "region": {"disc": [24, 30, 6]}}]]))
    assert r["changed"] is False and len(r["alternatives"]) == 2
    assert os.path.exists(os.path.join(s, "rehearse", "sheet.png")) and os.path.exists(os.path.join(s, "rehearse", "alt-2.png"))
    assert digest(os.path.join(s, "state.pkl")) == state                      # rehearsing changes nothing
    r = run(s, "look", "--wet"); assert "glistening" in r
    r = run(s, "look", "--raking", "30,20"); assert r["ok"]
    r = run(s, "back", "2")
    assert r["acts"] == 1 and len(r["undone"]) == 2
    assert np.array_equal(easel.Session(s).load().cv.reflectance(), after_sky)   # back to just after the sky, exactly
    lines = run(s, "journal", lines=True)
    acts = [json.loads(l)["act"] for l in lines]
    assert acts == ["wash", "dry", "wash", "back"]
    # an act that fails changes nothing, and says what to change
    state = digest(os.path.join(s, "state.pkl"))
    with pytest.raises(SystemExit) as e:
        easel.main([s, "do", json.dumps([SEA, {"do": "impasto", "region": {"band": [0, 9]}, "paint": "rose"}])])
    assert e.value.code == 2 and digest(os.path.join(s, "state.pkl")) == state
    assert json.loads(open(os.path.join(s, "acts.json")).read()) == [SKY]


def test_back_past_pruned_history_replays_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(easel, "KEEP_HISTORY", 1)
    s = str(tmp_path / "s")
    run(s, "new", "--size", "40x30", "--pigments", "ultramarine,rose")
    run(s, "do", json.dumps({"do": "wash", "region": {"band": [0, 10]}, "paint": "rose"}))
    one = easel.Session(s).load().cv.reflectance()
    for k in range(3): run(s, "do", json.dumps({"do": "dab", "at": [[5 + 10 * k, 20]], "paint": "ultramarine"}))
    assert sorted(os.listdir(os.path.join(s, "history"))) == ["0000.pkl", "0004.pkl"]
    r = run(s, "back", "3")
    assert r["replayed"] == 1 and np.array_equal(easel.Session(s).load().cv.reflectance(), one)


def test_help_is_the_whole_vocabulary():
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(easel.__file__)))))
    out = subprocess.run([sys.executable, "-m", "goo.classic.easel", "nowhere", "help"], capture_output=True, text=True, env=env, check=True).stdout
    from goo.classic.studio import ACTS
    for name, run_ in ACTS.items():
        assert f"\n{name}(" in out and json.dumps(run_._act["example"]) in out
