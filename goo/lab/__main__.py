"""python -m goo.lab COMMAND ...   (docs/LAB.md §7). Everything is written under --out (default out/lab).

    card NAME [--fast] [--water] [--pvc X]  drawdown card with sheet, admissibility and sensitivity: NAME.png and NAME.json
    sheet NAME                the phenotype: NAME.sheet.json (and printed)
    admissible NAME           the five laws with shortfalls (printed and NAME.admissible.json)
    validate FILE             check a contribution file (exit status 1 if invalid)
    export NAME               write NAME.substance.json in the contribution format
    search a|b|c [--quick]    the inverse-design demonstrations: search_X.json and the found substances' files
    contact NAME...           a contact sheet of cards already made (their JSON), contact.png
    render NAME...            redraw cards already made from their JSON

NAME is a library substance or a contribution file (*.json).
"""
import argparse, json, pathlib, sys, warnings


def resolve(name):
    from goo.substance import library as lib
    from goo.lab.contribute import validate
    p = pathlib.Path(name)
    if p.suffix == ".json" and p.exists():
        v = validate(p)
        if v.substance is None:
            raise SystemExit(v.text())
        return v.substance
    try:
        return lib.get(name)
    except KeyError:
        raise SystemExit(f"{name!r} is neither a library substance ({', '.join(s.name for s in lib.ALL)}) nor a file")


def make_card(sub, out, fast=False, binders=None, pvc=None):
    from goo.lab.drawdown import drawdown
    from goo.lab.sheet import sheet
    from goo.lab.sensitivity import sensitivity
    from goo.lab.admissible import admissible
    from goo.lab.render import render_card
    from goo.lab import recipes as R
    kw = {} if binders is None else dict(binders=binders)
    subject = sub if pvc is None else R.Paint(sub.name, ((sub, 1.0),), None, pvc)
    card = drawdown(subject, **kw)
    sh = sheet(subject, **kw)
    if not fast:
        card.data["sensitivity"] = sensitivity(subject, card, sh, **kw)
    card.data["sheet"] = sh
    card.data["admissible"] = admissible(sub)
    out.mkdir(parents=True, exist_ok=True)
    png = render_card(card, out / f"{sub.name}.png")
    (out / f"{sub.name}.json").write_text(card.to_json())
    return card, png


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m goo.lab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["card", "sheet", "admissible", "validate", "export", "search", "contact",
                                        "render"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--out", default="out/lab")
    ap.add_argument("--fast", action="store_true", help="card without sensitivity")
    ap.add_argument("--quick", action="store_true", help="search with a tiny budget")
    ap.add_argument("--water", action="store_true", help="card: add the 'water' recipe (particles in water alone)")
    ap.add_argument("--pvc", type=float, default=None, help="card: pigment volume fraction of the wet paint, every binder")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    warnings.simplefilter("ignore")
    if a.command == "card":
        from goo.lab import recipes as R
        for n in a.args:
            _, png = make_card(resolve(n), out, a.fast, (R.WATER,) + R.BINDERS if a.water else None, a.pvc)
            print(png)
    elif a.command == "sheet":
        from goo.lab.sheet import sheet
        for n in a.args:
            s = resolve(n)
            sh = sheet(s)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{s.name}.sheet.json").write_text(json.dumps(sh, indent=1, default=float))
            for k, v in sh.items():
                if isinstance(v, dict) and "value" in v:
                    print(f"{k}: {v['value']} {v['unit']} [{v['evidence']}] {v.get('note', '')}")
    elif a.command == "admissible":
        from goo.lab.admissible import admissible, text
        for n in a.args:
            s = resolve(n)
            r = admissible(s)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{s.name}.admissible.json").write_text(json.dumps(r, indent=1))
            print(text(r))
    elif a.command == "validate":
        from goo.lab.contribute import validate
        bad = 0
        for f in a.args:
            v = validate(pathlib.Path(f))
            print(f"{f}: {v.text()}")
            bad += not v.ok
        return 1 if bad else 0
    elif a.command == "export":
        from goo.lab.contribute import write
        out.mkdir(parents=True, exist_ok=True)
        for n in a.args:
            print(write(resolve(n), out / f"{n}.substance.json"))
    elif a.command == "search":
        from goo.lab import search as S
        from goo.lab.contribute import write
        out.mkdir(parents=True, exist_ok=True)
        budget = dict(maxiter=1, popsize=4) if a.quick else {}
        for which in a.args:
            res = dict(a=S.demo_a, b=S.demo_b, c=S.demo_c)[which](**budget)
            for part in ("possible", "impossible"):
                write(res[part]["substance"], out / f"{res[part]['substance'].name}.substance.json")
            (out / f"search_{which}.json").write_text(json.dumps(S.summary(res), indent=1))
            print(out / f"search_{which}.json")
    elif a.command == "render":
        from goo.lab.drawdown import Card
        from goo.lab.render import render_card
        for n in a.args:
            d = json.loads((out / f"{n}.json").read_text())
            print(render_card(Card(d.pop("subject"), d, d.pop("flags")), out / f"{n}.png"))
    elif a.command == "contact":
        from goo.lab.drawdown import Card
        from goo.lab.render import contact_sheet
        cards = []
        for n in a.args:
            d = json.loads((out / f"{n}.json").read_text())
            cards.append(Card(d.pop("subject"), d, d.pop("flags")))
        print(contact_sheet(cards, out / "contact.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
