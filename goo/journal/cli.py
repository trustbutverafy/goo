"""Painting one command at a time (docs/JOURNAL.md §5).

    python -m goo.journal DIR new --size 160x120 --dx 0.25mm --ground paper --substances water,gum_arabic,hematite
    python -m goo.journal DIR do '<json act or list of acts>'
    python -m goo.journal DIR look [--region JSON] [--scale N] [--gloss] [--ir] [--exposure X] [--light daylight|A]
    python -m goo.journal DIR log | back N | fork NAME | use NAME | branches | vessels | vocab | replay

State lives in DIR: DIR/current names the branch; DIR/branches/NAME/journal.json is the journal (portable JSON,
the truth); DIR/branches/NAME/state.pkl and ckpt/NNNN.pkl are pickles of the session after the last entry and
after each entry (a cache: not portable across goo or numpy versions; replay rebuilds them from the journal).
This module uses only the Session's public interface."""
import argparse
import json
import pathlib
import pickle
import shutil
import sys
import time
from goo.journal.session import Session, Refused, replay, ReplayError
from goo.journal import acts as A
from goo.journal.looking import write_png


def _paths(d, branch=None):
    d = pathlib.Path(d)
    if branch is None:
        cur = d / "current"
        if not cur.exists():
            raise SystemExit(f"{d} holds no session (run 'new' first)")
        branch = cur.read_text().strip()
    b = d / "branches" / branch
    return d, b, branch


def _save(bdir, s):
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "ckpt").mkdir(exist_ok=True)
    j = s.journal()
    tmp = bdir / "journal.json.tmp"
    tmp.write_text(json.dumps(j, indent=1))
    tmp.replace(bdir / "journal.json")
    with open(bdir / "state.pkl.tmp", "wb") as f:
        pickle.dump(s, f, protocol=pickle.HIGHEST_PROTOCOL)
    (bdir / "state.pkl.tmp").replace(bdir / "state.pkl")
    shutil.copyfile(bdir / "state.pkl", bdir / "ckpt" / f"{len(j['entries']):04d}.pkl")


def _load(bdir):
    j = json.loads((bdir / "journal.json").read_text())
    p = bdir / "state.pkl"
    if p.exists():
        with open(p, "rb") as f:
            s = pickle.load(f)
        if s.entries and s.digest() != j["entries"][-1]["world"]:
            raise SystemExit("state.pkl does not match the journal; run 'replay'")
        return s, j
    return replay(j["entries"], j["initial"]), j


def _size(t):
    try:
        W, H = (int(v) for v in t.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("size is WxH, e.g. 160x120") from None
    return [W, H]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m goo.journal", description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="start a session")
    n.add_argument("--size", type=_size, required=True, help="W x H in cells, e.g. 160x120")
    n.add_argument("--dx", default="0.25 mm", help="cell side (default 0.25 mm)")
    n.add_argument("--ground", default="paper", help="gesso, paper, canvas, or a library substance")
    n.add_argument("--porosity"); n.add_argument("--pore-radius"); n.add_argument("--depth")
    n.add_argument("--substances", required=True, help="comma-separated library names, e.g. water,gum_arabic")
    n.add_argument("--humidity", default="0.5"); n.add_argument("--temperature", default="20 C")
    n.add_argument("--seed", type=int, default=0)
    n.add_argument("--force", action="store_true", help="replace a session already in DIR")
    d = sub.add_parser("do", help="do an act (JSON object) or a list of acts (all or nothing)")
    d.add_argument("act")
    lk = sub.add_parser("look", help="write a PNG of what the painting shows and print its path")
    lk.add_argument("--region", default=None, help="a region as JSON (its bounding box is shown)")
    lk.add_argument("--scale", type=int, default=1); lk.add_argument("--exposure", type=float, default=1.0)
    lk.add_argument("--gloss", action="store_true"); lk.add_argument("--ir", action="store_true")
    lk.add_argument("--light", default="daylight"); lk.add_argument("--name", default=None)
    sub.add_parser("log", help="the journal, one line an entry")
    bk = sub.add_parser("back", help="undo the last N acts"); bk.add_argument("n", type=int)
    fk = sub.add_parser("fork", help="copy this branch to NAME (stay on this one)"); fk.add_argument("name")
    us = sub.add_parser("use", help="switch to branch NAME"); us.add_argument("name")
    sub.add_parser("branches"); sub.add_parser("vessels"); sub.add_parser("vocab"); sub.add_parser("replay")
    a = ap.parse_args(argv)

    if a.cmd == "new":
        root = pathlib.Path(a.dir)
        if (root / "current").exists() and not a.force:
            raise SystemExit(f"{root} already holds a session (use --force to replace it)")
        if root.exists() and a.force:
            shutil.rmtree(root / "branches", ignore_errors=True)
        g = {"name": a.ground}
        for k in ("porosity", "pore_radius", "depth"):
            if getattr(a, k) is not None:
                g[k] = getattr(a, k)
        init = dict(size=a.size, dx=a.dx, ground=g, substances=a.substances.split(","),
                    air={"humidity": a.humidity, "temperature": a.temperature}, seed=a.seed)
        try:
            s = Session(init)
        except ValueError as e:
            raise SystemExit(f"refused: {e}")
        root.mkdir(parents=True, exist_ok=True)
        _save(root / "branches" / "main", s)
        (root / "current").write_text("main")
        W, H, dx = s.size
        print(f"new session in {root}: {W} x {H} cells of {dx * 1e3:g} mm, ground {g['name']}, "
              f"substances {', '.join(init['substances'])}; branch main")
        return 0

    root, bdir, branch = _paths(a.dir)
    if a.cmd == "use":
        if not (root / "branches" / a.name).exists():
            raise SystemExit(f"no branch {a.name!r}")
        (root / "current").write_text(a.name)
        print(f"on branch {a.name}")
        return 0
    if a.cmd == "fork":
        dst = root / "branches" / a.name
        if dst.exists():
            raise SystemExit(f"branch {a.name!r} exists")
        shutil.copytree(bdir, dst)
        print(f"branch {a.name} forked from {branch} (still on {branch})")
        return 0
    if a.cmd == "branches":
        for p in sorted((root / "branches").iterdir()):
            print(("* " if p.name == branch else "  ") + p.name)
        return 0
    if a.cmd == "vocab":
        for k, v in A.vocabulary().items():
            print(f"{k}: {v['doc']}\n    {json.dumps(v['example'])}")
        return 0

    t = time.perf_counter()
    s, j = _load(bdir)
    load = time.perf_counter() - t
    if a.cmd == "do":
        try:
            act = json.loads(a.act)
        except json.JSONDecodeError as e:
            print(f"refused: not JSON: {e}"); return 1
        try:
            recs = s.do(act)
        except Refused as e:
            print(f"refused (nothing changed): {e}"); return 1
        _save(bdir, s)
        k = len(recs) if isinstance(recs, list) else 1
        for e in s.entries[-k:]:
            print(f"{e['n']:4d}  {e['summary']}")
        print(f"[{branch}] world time {s.time:.6g} s; load {load:.2f} s, total {time.perf_counter() - t:.2f} s")
        return 0
    if a.cmd == "look":
        region = json.loads(a.region) if a.region else None
        try:
            seen = s.look(light=a.light, exposure=a.exposure, region=region, gloss=a.gloss, infrared=a.ir)
        except ValueError as e:
            print(f"refused: {e}"); return 1
        out = root / "looks"
        out.mkdir(exist_ok=True)
        tag = a.name or f"{branch}-{len(s.entries):04d}"
        paths = [write_png(seen.image, out / f"{tag}.png", a.scale)]
        if a.gloss:
            paths.append(write_png(seen.gloss_image, out / f"{tag}-gloss.png", a.scale))
        if a.ir:
            paths.append(write_png(seen.infrared_image, out / f"{tag}-ir.png", a.scale))
        fl = sorted({f.kind for f in seen.receipt.flags})
        print("\n".join(paths))
        print(f"look at t {s.time:.6g} s, light {seen.light}, exposure {seen.exposure:g}, box {seen.box}, "
              f"light residual {seen.receipt.residual['light']:.1e}, flags {', '.join(fl) or 'none'}; "
              f"{time.perf_counter() - t:.2f} s")
        return 0
    if a.cmd == "log":
        print("\n".join(s.log()) or "(no acts yet)")
        return 0
    if a.cmd == "vessels":
        v, tp = s.vessels()
        for k, (p, vol) in v.items():
            print(f"vessel {k}: {p}, {vol * 1e9:.4g} uL")
        for k, (on, kg) in tp.items():
            print(f"tape {k}: {'on' if on else 'peeled'}, caught {kg:.3g} kg")
        return 0
    if a.cmd == "back":
        ck = bdir / "ckpt" / f"{len(s.entries) - a.n:04d}.pkl"
        if 0 <= len(s.entries) - a.n < len(s.entries) and ck.exists():
            with open(ck, "rb") as f:
                s = pickle.load(f)
        else:
            try:
                s.back(a.n)
            except Refused as e:
                print(f"refused: {e}"); return 1
        for old in (bdir / "ckpt").glob("*.pkl"):
            if int(old.stem) > len(s.entries):
                old.unlink()
        _save(bdir, s)
        print(f"back to entry {len(s.entries)}, world time {s.time:.6g} s")
        return 0
    if a.cmd == "replay":
        try:
            r = replay(j["entries"], j["initial"])
        except ReplayError as e:
            print(f"REPLAY DIFFERS: {e}"); return 1
        _save(bdir, r)
        print(f"replayed {len(j['entries'])} entries bit for bit; world digest {r.digest()[:16]}; "
              f"{time.perf_counter() - t:.2f} s")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
