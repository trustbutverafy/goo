"""goo.easel: the studio one command at a time, state on disk, so a painter at a shell (or an agent with tools)
works look-act-look.

    python -m goo.easel SESSION new --size 400x300 --pigments ultramarine,rose,ochre,sienna --medium watercolour
    python -m goo.easel SESSION do '{"do": "wet", "region": {"band": [0, 172]}}'
    python -m goo.easel SESSION do '[{"do": "graded_wash", "region": {"band": [0, 172]}, "from": "ultramarine", "to": "ochre"},
                                     {"do": "dry", "until": "damp"}]'
    python -m goo.easel SESSION rehearse '[{"do": "backrun", ...}, {"do": "blot", ...}]'   # one picture each; nothing changes
    python -m goo.easel SESSION look [--wet] [--raking] [--region JSON]
    python -m goo.easel SESSION back 2                                                     # undo the last two acts
    python -m goo.easel SESSION journal
    python -m goo.easel SESSION help                                                       # every act, with an example

Every command writes SESSION/look.png (the painting at 2x with a ruler in cells along the edges) and prints a
small JSON receipt on stdout. Errors print {"error": ..., "fix": ...} on stderr and exit with status 2; a `do`
that fails changes nothing (a list of acts is all or nothing).

Files: state.pkl (the studio), acts.json (the live acts, in order), journal.json (everything, including what was
undone), history/NNNN.pkl (the studio after the Nth act, at command boundaries; old ones are pruned and
recovered by replaying acts, which is exact: I4), rehearse/alt-K.png and sheet.png.
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, time
import numpy as np

KEEP_HISTORY = 12


def _out(obj):
    print(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else repr(o)))


def _fail(msg, fix=None, code=2):
    sys.stderr.write(json.dumps({"error": msg, **({"fix": fix} if fix else {})}) + "\n")
    sys.exit(code)


class Session:
    def __init__(self, path):
        self.path = path

    def p(self, *a): return os.path.join(self.path, *a)

    def load(self):
        from goo.studio import Studio
        if not os.path.exists(self.p("state.pkl")):
            _fail(f"no session at {self.path}", f"start one: python -m goo.easel {self.path} new --size 400x300 --pigments ultramarine,rose,ochre")
        with open(self.p("state.pkl"), "rb") as f: state = pickle.load(f)
        st = Studio.__new__(Studio); st._restore(state); st._rec = None
        return st

    def save(self, st, n_acts):
        os.makedirs(self.p("history"), exist_ok=True)
        data = pickle.dumps(st._state(), protocol=5)
        for name in ("state.pkl", os.path.join("history", f"{n_acts:04d}.pkl")):
            tmp = self.p(name + ".tmp")
            with open(tmp, "wb") as f: f.write(data)
            os.replace(tmp, self.p(name))
        hist = sorted(int(f[:4]) for f in os.listdir(self.p("history")) if f.endswith(".pkl"))
        for k in hist[1:-KEEP_HISTORY]: os.remove(self.p("history", f"{k:04d}.pkl"))   # keep the first and the last few

    def acts(self):
        return json.load(open(self.p("acts.json"))) if os.path.exists(self.p("acts.json")) else []

    def write_acts(self, acts): json.dump(acts, open(self.p("acts.json"), "w"), indent=1)

    def journal(self):
        return json.load(open(self.p("journal.json"))) if os.path.exists(self.p("journal.json")) else []

    def write_journal(self, j): json.dump(j, open(self.p("journal.json"), "w"), indent=1)


def picture(st, path, img=None, scale=2, ruler=True):
    """The painting at `scale`, with a ruler in cells (ticks every 10, numbers every 50) on the top and left."""
    from PIL import Image, ImageDraw
    img = st.look() if img is None else img
    im = Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    if not ruler: im.save(path); return path
    m = 22
    can = Image.new("RGB", (im.width + m, im.height + m), (250, 250, 248)); can.paste(im, (m, m))
    d = ImageDraw.Draw(can)
    for x in range(0, img.shape[1] + 1, 10):
        X = m + x * scale; L = 8 if x % 50 == 0 else 4; d.line([(X, m - L), (X, m - 1)], fill=(90, 90, 90))
        if x % 50 == 0: d.text((X + 2, 1), str(x), fill=(40, 40, 40))
    for y in range(0, img.shape[0] + 1, 10):
        Y = m + y * scale; L = 8 if y % 50 == 0 else 4; d.line([(m - L, Y), (m - 1, Y)], fill=(90, 90, 90))
        if y % 50 == 0: d.text((1, Y + 2), str(y), fill=(40, 40, 40))
    can.save(path); return path


def sheet(images, labels, path, scale=1):
    """Pictures side by side (two per row when there are more than three), each labelled."""
    from PIL import Image, ImageDraw
    h, w = images[0].shape[:2]; n = len(images); cols = n if n <= 3 else 2; rows = (n + cols - 1) // cols
    pad, top = 8, 16
    can = Image.new("RGB", (cols * (w * scale + pad) + pad, rows * (h * scale + pad + top) + pad), (250, 250, 248))
    d = ImageDraw.Draw(can)
    for k, (im, lab) in enumerate(zip(images, labels)):
        r, c = divmod(k, cols); x = pad + c * (w * scale + pad); y = pad + r * (h * scale + pad + top)
        can.paste(Image.fromarray(im).resize((w * scale, h * scale), Image.NEAREST), (x, y + top))
        d.text((x, y + 2), lab[:60], fill=(30, 30, 30))
    can.save(path); return path


def _replay(sess, target):
    """The studio after the first `target` live acts: from the newest checkpoint at or before it, replaying the rest."""
    hist = sorted(int(f[:4]) for f in os.listdir(sess.p("history")) if f.endswith(".pkl"))
    base = max(k for k in hist if k <= target)
    from goo.studio import Studio
    with open(sess.p("history", f"{base:04d}.pkl"), "rb") as f: state = pickle.load(f)
    st = Studio.__new__(Studio); st._restore(state); st._rec = None
    for a in sess.acts()[base:target]: st.do(a)
    return st, base


def _parse_json(text, what):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        _fail(f"{what} is not JSON: {e}", 'write it like \'{"do": "wash", "region": {"band": [0, 100]}, "paint": "ultramarine"}\' (double quotes)')


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m goo.easel", description="goo's studio, one command at a time")
    ap.add_argument("session"); ap.add_argument("command", choices=["new", "do", "rehearse", "look", "back", "journal", "help"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--size", default="400x300"); ap.add_argument("--pigments", default="ultramarine,rose,ochre,sienna,carbon,yellow")
    ap.add_argument("--medium", default="watercolour"); ap.add_argument("--ground"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wet", action="store_true", help="look: show wetness (blue glistening, amber damp, hatched not touch-dry)")
    ap.add_argument("--raking", nargs="?", const="20,18", help="look: low light, AZIMUTH,ELEVATION in degrees")
    ap.add_argument("--region", help="look: crop to a region (JSON)")
    ap.add_argument("--scale", type=int, default=2)
    a = ap.parse_args(argv)
    t0 = time.time()
    from goo import studio as S
    sess = Session(a.session)

    if a.command == "help":
        print(__doc__.split("Every command")[0].strip() + "\n\nACTS (the JSON keys are the parameter names; `from` for from_)\n\n"
              + S.vocabulary() + "\n\nREGIONS\n    " + "\n    ".join(S.REGION_KINDS.values())
              + "\n\nPAINTS\n    " + " ".join(S.Studio.paint.__doc__.split()))
        return 0

    try:
        if a.command == "new":
            if os.path.exists(sess.p("state.pkl")): _fail(f"{a.session} already holds a session", "pick another directory, or remove this one")
            os.makedirs(a.session, exist_ok=True)
            try:
                W, H = map(int, a.size.lower().split("x"))
            except ValueError:
                _fail(f"--size {a.size!r}", "write WIDTHxHEIGHT in cells (0.25 mm each), e.g. 400x300")
            st = S.Studio.new(W, H, a.pigments.split(","), medium=a.medium, ground=a.ground, seed=a.seed, workdir=a.session)
            sess.save(st, 0); sess.write_acts([]); sess.write_journal([])
            picture(st, sess.p("look.png"), scale=a.scale)
            _out(dict(ok=True, session=a.session, size=[W, H], medium=a.medium, pigments=st.pigments, look=sess.p("look.png"), seconds=round(time.time() - t0, 2)))
            return 0

        st = sess.load()
        if a.command == "do":
            acts = _parse_json(a.arg or "", "the act")
            acts = acts if isinstance(acts, list) else [acts]
            j0 = len(st.journal); receipts = []
            for k, x in enumerate(acts):
                try:
                    r = st.do(x)
                except (S.NothingLaid, S.StudioError, NotImplementedError) as e:
                    _fail(f"act {k + 1} of {len(acts)} ({x.get('do') if isinstance(x, dict) else x}): {e}",
                          "nothing was changed; correct that act and send the list again" if len(acts) > 1 else "nothing was changed")
                receipts.append(r.to_json() if hasattr(r, "to_json") else r)
            live = sess.acts() + acts
            sess.save(st, len(live)); sess.write_acts(live)
            sess.write_journal(sess.journal() + st.journal[j0:])
            picture(st, sess.p("look.png"), scale=a.scale)
            _out(dict(ok=True, acts=len(live), clock=st.clock, receipts=receipts, look=sess.p("look.png"), seconds=round(time.time() - t0, 2)))
            return 0

        if a.command == "rehearse":
            alts = _parse_json(a.arg or "", "the alternatives")
            if not isinstance(alts, list) or not alts:
                _fail("rehearse takes a JSON list of alternatives, each an act or a list of acts", "e.g. '[{\"do\": \"blot\", ...}, [{\"do\": \"dry\", \"until\": \"damp\"}, {\"do\": \"backrun\", ...}]]'")
            os.makedirs(sess.p("rehearse"), exist_ok=True)
            for f in os.listdir(sess.p("rehearse")): os.remove(sess.p("rehearse", f))
            imgs, labels, out = [st.look()], ["now"], []
            for k, alt in enumerate(alts):
                with st.rehearse() as r:
                    try:
                        rec = r.do(alt); err = None
                    except (S.NothingLaid, S.StudioError, NotImplementedError) as e:
                        rec, err = None, str(e)
                    im = r.look()
                path = picture(r, sess.p("rehearse", f"alt-{k + 1}.png"), img=im, scale=a.scale)
                imgs.append(im); acts = alt if isinstance(alt, list) else [alt]
                labels.append(f"{k + 1}: " + " + ".join(str(x.get("do")) for x in acts) + (" (failed)" if err else ""))
                out.append(dict(alt=k + 1, picture=path, **({"error": err} if err else {"receipts": [x.to_json() for x in (rec if isinstance(rec, list) else [rec])]})))
            sheet(imgs, labels, sess.p("rehearse", "sheet.png"), scale=a.scale)
            picture(st, sess.p("look.png"), scale=a.scale)
            _out(dict(ok=True, changed=False, alternatives=out, sheet=sess.p("rehearse", "sheet.png"),
                      keep="send the one you like with `do`", seconds=round(time.time() - t0, 2)))
            return 0

        if a.command == "look":
            if a.wet: img = st.wetness_picture()
            else:
                raking = None
                if a.raking:
                    az, el = map(float, a.raking.split(",")); raking = dict(azimuth=az, elevation=el)
                img = st.look(raking=raking)
            if a.region:
                m = S.coerce_region(_parse_json(a.region, "the region")).mask(st.H, st.W); ys, xs = np.nonzero(m)
                if ys.size: img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            picture(st, sess.p("look.png"), img=img, scale=a.scale, ruler=not a.region)
            s = st.look_wet()
            _out(dict(ok=True, look=sess.p("look.png"), clock=st.clock, acts=len(sess.acts()),
                      glistening=round(float((s > S.GLISTEN).mean()), 3), damp=round(float(((s > S.DAMP) & (s <= S.GLISTEN)).mean()), 3),
                      reserved=round(float(st.reserved.mean()), 3), seconds=round(time.time() - t0, 2)))
            return 0

        if a.command == "back":
            try:
                n = int(a.arg or 1)
            except ValueError:
                _fail(f"back {a.arg!r}", "give a number of acts to undo, e.g. back 1")
            live = sess.acts()
            if not 1 <= n <= len(live): _fail(f"cannot go back {n}: the session has {len(live)} acts", f"use 1..{len(live)}")
            target = len(live) - n
            st, base = _replay(sess, target)
            sess.write_acts(live[:target]); sess.save(st, target)
            sess.write_journal(sess.journal() + [dict(act="back", params=dict(n=n), receipt=dict(to_act=target, replayed=target - base, clock=st.clock))])
            picture(st, sess.p("look.png"), scale=a.scale)
            _out(dict(ok=True, acts=target, undone=live[target:], replayed=target - base, look=sess.p("look.png"), seconds=round(time.time() - t0, 2)))
            return 0

        if a.command == "journal":
            picture(st, sess.p("look.png"), scale=a.scale)
            for e in sess.journal():
                r = e.get("receipt", {})
                print(json.dumps(dict(act=e["act"], **({"params": e["params"]} if e.get("params") else {}),
                                      **{k: r[k] for k in ("tick", "ticks", "given", "taken", "notes", "to_act", "fixed") if k in r},
                                      **({"failed": e["failed"]} if e.get("failed") else {}))))
            return 0
    except (S.StudioError, S.NothingLaid) as e:
        _fail(str(e))


if __name__ == "__main__":
    sys.exit(main())
