"""Painting one command at a time (docs/JOURNAL.md §5).

    python -m goo.journal DIR new --size 160x120 --dx 0.25mm --ground paper --substances water,gum_arabic,hematite
    python -m goo.journal DIR do '<json act or list of acts>'
    python -m goo.journal DIR look [--region JSON] [--scale N] [--gloss] [--ir] [--exposure X] [--light daylight|A]
    python -m goo.journal DIR log | back N | fork NAME | use NAME | branches | vessels | vocab | replay

State lives in DIR: DIR/current names the branch; DIR/branches/NAME/journal.json is the journal (portable JSON,
the truth); DIR/branches/NAME/state.pkz is the session after the last entry, and ckpt/NNNN.pkz checkpoints at entry
0, every CKPT_EVERY entries (the CKPT_RECENT newest kept) and on request (`checkpoint`, never thinned). They are
compressed pickles (goo.journal.store): a cache, not portable across goo or numpy versions; `back` loads the nearest
checkpoint and replays the journal from it, and `replay` rebuilds everything from the journal.
This module uses only the Session's public interface."""
import argparse
import json
import pathlib
import shutil
import sys
import time
from goo.journal.session import Session, Refused, replay, ReplayError
from goo.journal import acts as A
from goo.journal.looking import write_png, load_cache, save_cache
from goo.journal import store


def _paths(d, branch=None):
    d = pathlib.Path(d)
    if branch is None:
        cur = d / "current"
        if not cur.exists():
            raise SystemExit(f"{d} holds no session (run 'new' first)")
        branch = cur.read_text().strip()
    b = d / "branches" / branch
    return d, b, branch


CKPT_EVERY = 10          # a checkpoint every this many entries
CKPT_RECENT = 3          # periodic checkpoints kept besides entry 0 and those asked for


def _ckpts(bdir):
    """{entry number: path} of the checkpoints on disk."""
    out = {}
    for p in (bdir / "ckpt").glob("*.pkz"):
        out[int(p.name.split(".")[0])] = p
    return out


def _save(bdir, s, checkpoint=False):
    """Write the journal and the session (compressed: goo.journal.store); write a checkpoint at entry 0, every
    CKPT_EVERY entries and when asked, and thin the periodic ones to the CKPT_RECENT newest."""
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "ckpt").mkdir(exist_ok=True)
    j = s.journal()
    tmp = bdir / "journal.json.tmp"
    tmp.write_text(json.dumps(j, indent=1))
    tmp.replace(bdir / "journal.json")
    blob = store.dumps(s)
    (bdir / "state.pkz.tmp").write_bytes(blob)
    (bdir / "state.pkz.tmp").replace(bdir / "state.pkz")
    n = len(j["entries"])
    if checkpoint or n == 0 or n % CKPT_EVERY == 0:
        name = f"{n:04d}.keep.pkz" if checkpoint else f"{n:04d}.pkz"
        (bdir / "ckpt" / name).write_bytes(blob)
    periodic = sorted(k for k, p in _ckpts(bdir).items() if k > 0 and ".keep." not in p.name)
    for k in periodic[:-CKPT_RECENT]:
        _ckpts(bdir)[k].unlink()


def _load(bdir):
    j = json.loads((bdir / "journal.json").read_text())
    p = bdir / "state.pkz"
    if p.exists():
        s = store.load(p)
        es = s.entries
        if len(es) != len(j["entries"]) or (es and es[-1]["world"] != j["entries"][-1]["world"]):
            raise SystemExit("state.pkz does not match the journal; run 'replay'")
        s.configure(keep=0, atomic=False)
        return s, j
    return replay(j["entries"], j["initial"], keep=0, atomic=False), j


def _back(bdir, j, k):
    """The session after entry k: the newest checkpoint at or before k, then the journal's entries replayed."""
    cks = {n: p for n, p in _ckpts(bdir).items() if n <= k}
    if cks:
        s = store.load(cks[max(cks)])
        s.configure(keep=0, atomic=False)
        if s.entries != j["entries"][:len(s.entries)]:
            cks = {}
    if not cks:
        s = Session(j["initial"], keep=0, atomic=False)
    return s.resume(j["entries"][len(s.entries):k])


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
    lk.add_argument("--scale", type=float, default=1, help="integer, or a fraction 1/k (0.5, 0.25): k x k cells averaged in linear light"); lk.add_argument("--exposure", type=float, default=1.0)
    lk.add_argument("--gloss", action="store_true"); lk.add_argument("--ir", action="store_true")
    lk.add_argument("--light", default="daylight"); lk.add_argument("--name", default=None)
    sub.add_parser("log", help="the journal, one line an entry")
    bk = sub.add_parser("back", help="undo the last N acts"); bk.add_argument("n", type=int)
    fk = sub.add_parser("fork", help="copy this branch to NAME (stay on this one)"); fk.add_argument("name")
    us = sub.add_parser("use", help="switch to branch NAME"); us.add_argument("name")
    sub.add_parser("checkpoint", help="keep a checkpoint of the current state (never thinned)")
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
            s = Session(init, keep=0, atomic=False)
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
        cpath = bdir / "lookcache.npz"
        cache = load_cache(cpath) if cpath.exists() else {}
        try:
            pic = s.picture(light=a.light, exposure=a.exposure, region=region, scale=a.scale, gloss=a.gloss,
                            infrared=a.ir, cache=cache)
        except ValueError as e:
            print(f"refused: {e}"); return 1
        tl = time.perf_counter()
        save_cache(cache, cpath)
        out = root / "looks"
        out.mkdir(exist_ok=True)
        tag = a.name or f"{branch}-{len(s.entries):04d}"
        paths = [write_png(pic.image, out / f"{tag}.png")]
        if a.gloss:
            paths.append(write_png(pic.gloss_image, out / f"{tag}-gloss.png"))
        if a.ir:
            paths.append(write_png(pic.infrared_image, out / f"{tag}-ir.png"))
        fl = sorted({f.kind for f in pic.receipt.flags}) if pic.receipt else []
        print("\n".join(paths))
        print(f"look at t {s.time:.6g} s, light {pic.light}, exposure {pic.exposure:g}, scale {pic.scale:g}, box "
              f"{pic.box}, image {pic.image.shape[1]} x {pic.image.shape[0]}; optics solved for {pic.solved} of "
              f"{pic.cells} cells (the rest reused)"
              + (f", light residual {pic.receipt.residual['light']:.1e}, flags {', '.join(fl) or 'none'}"
                 if pic.receipt else "")
              + f"; {time.perf_counter() - t:.2f} s (cache save {time.perf_counter() - tl:.2f} s)")
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
        k = len(s.entries) - a.n
        if not 0 <= k <= len(s.entries):
            print(f"refused: back 0 to {len(s.entries)}"); return 1
        s = _back(bdir, j, k)
        for n, p in _ckpts(bdir).items():
            if n > k:
                p.unlink()
        _save(bdir, s)
        print(f"back to entry {len(s.entries)}, world time {s.time:.6g} s; {time.perf_counter() - t:.2f} s")
        return 0
    if a.cmd == "checkpoint":
        _save(bdir, s, checkpoint=True)
        print(f"checkpoint kept at entry {len(s.entries)}")
        return 0
    if a.cmd == "replay":
        try:
            r = replay(j["entries"], j["initial"], keep=0, atomic=False)
        except ReplayError as e:
            print(f"REPLAY DIFFERS: {e}"); return 1
        _save(bdir, r)
        print(f"replayed {len(j['entries'])} entries bit for bit; world digest {r.digest()[:16]}; "
              f"{time.perf_counter() - t:.2f} s")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
