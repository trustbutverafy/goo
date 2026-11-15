"""Generator of goo/substance/_nk_data.py from a local clone of the refractiveindex.info database (CC0 1.0).

Not imported by goo.substance. Run it to regenerate the shipped constants and diff them against the file:

    REFRACTIVEINDEX_DB=/path/to/refractiveindex.info-database python goo/substance/generate_nk.py [--write]

What it does, for each source file listed in SOURCES: reads the file's n (formula 1-5 or a table) and k (a table, or
k = 0 where the source measured only a transparent range: `k="zero"`, which the shipped record says is assumed),
evaluates them on GRID (350-1700 nm every 10 nm), and records the range where the file's own data hold (the
intersection of its n and k ranges). Outside that range a formula is evaluated and a table is held at its end
value; the description marks those wavelengths as out of its domain. Nothing else is computed here.
Requires PyYAML (generator only)."""
import os, sys
from pathlib import Path
import numpy as np

GRID_NM = np.arange(350.0, 1700.0 + 1e-9, 10.0)
OUT = Path(__file__).with_name("_nk_data.py")

# key: (database file, k source).  "file": k from the file; "zero": transparent-range data, k = 0 assumed.
SOURCES = {
    "rutile_o": ("main/TiO2/nk/Devore-o.yml", "zero"),
    "rutile_e": ("main/TiO2/nk/Devore-e.yml", "zero"),
    "soot": ("other/soots/propane soot/nk/Dalzell.yml", "file"),
    "hematite_o": ("main/Fe2O3/nk/Querry-o.yml", "file"),
    "hematite_e": ("main/Fe2O3/nk/Querry-e.yml", "file"),
    "cds": ("main/CdS/nk/Treharne.yml", "file"),
    "calcite_o": ("main/CaCO3/nk/Ghosh-o.yml", "zero"),
    "calcite_e": ("main/CaCO3/nk/Ghosh-e.yml", "zero"),
    "cellulose": ("organic/(C6H10O5)n - cellulose/nk/Juntunen.yml", "zero"),
    "pmma": ("organic/(C5H8O2)n - poly(methyl methacrylate)/nk/Zhang-Tomson.yml", "file"),
    "water": ("main/H2O/nk/Hale.yml", "file"),
    "ethanol": ("organic/C2H6O - ethanol/nk/Kedenburg.yml", "file"),
}


def _floats(s): return [float(x) for x in str(s).split()]


def _formula(f, c, x):
    """refractiveindex.info dispersion formulas 1-5 (database/doc), x in um. Promoted from classic's generator."""
    c = list(c) + [0.0] * 20; x2 = x * x
    if f == 1:
        return np.sqrt(1 + c[0] + sum(c[i] * x2 / (x2 - c[i + 1] ** 2) for i in range(1, 17, 2)))
    if f == 2:
        return np.sqrt(1 + c[0] + sum(c[i] * x2 / (x2 - c[i + 1]) for i in range(1, 17, 2)))
    if f == 3:
        return np.sqrt(c[0] + sum(c[i] * x ** c[i + 1] for i in range(1, 17, 2)))
    if f == 4:
        return np.sqrt(c[0] + c[1] * x ** c[2] / (x2 - c[3] ** c[4]) + c[5] * x ** c[6] / (x2 - c[7] ** c[8])
                       + sum(c[i] * x ** c[i + 1] for i in range(9, 17, 2)))
    if f == 5:
        return c[0] + sum(c[i] * x ** c[i + 1] for i in range(1, 17, 2))
    raise ValueError(f"formula {f} not supported")


def read(db, rel):
    import yaml, re
    d = yaml.safe_load(open(Path(db) / "database" / "data" / rel, encoding="utf-8"))
    n = k = None; nr = kr = None
    for blk in d["DATA"]:
        t = blk["type"]
        if t.startswith("tabulated"):
            a = np.array([_floats(l) for l in blk["data"].strip().splitlines()])
            w = a[:, 0]; r = (w.min(), w.max())
            if t in ("tabulated nk", "tabulated n"):
                n = (lambda x, w=w, v=a[:, 1]: np.interp(x, w, v)); nr = r
            if t == "tabulated nk":
                k = (lambda x, w=w, v=a[:, 2]: np.interp(x, w, v)); kr = r
            if t == "tabulated k":
                k = (lambda x, w=w, v=a[:, 1]: np.interp(x, w, v)); kr = r
        elif t.startswith("formula"):
            c = _floats(blk["coefficients"]); r = _floats(blk["wavelength_range"]); f = int(t.split()[1])
            n = (lambda x, f=f, c=c, r=r: _formula(f, c, np.asarray(x, float))); nr = (r[0], r[1])
    clean = lambda s: re.sub(r"<[^>]+>", "", " ".join(str(s).split())).replace("&amp;", "&").strip()
    return n, k, nr, kr, clean(d.get("REFERENCES", "")), clean(d.get("COMMENTS", ""))


def commit(db):
    head = (Path(db) / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = Path(db) / ".git" / head[5:]
        if ref.exists(): return ref.read_text().strip()
        for line in (Path(db) / ".git" / "packed-refs").read_text().splitlines():
            if line.endswith(head[5:]): return line.split()[0]
    return head


def build(db):
    lam_um = GRID_NM / 1e3
    out = {}
    for key, (rel, ksrc) in SOURCES.items():
        nf, kf, nr, kr, ref, com = read(db, rel)
        lo, hi = nr
        if ksrc == "file":
            lo, hi = max(lo, kr[0]), min(hi, kr[1])
        n = np.asarray(nf(lam_um), float)
        if not np.all(np.isfinite(n)):                  # a formula that fails outside its range is held flat
            n = np.asarray(nf(np.clip(lam_um, nr[0], nr[1])), float)
        k = np.zeros_like(n) if ksrc == "zero" else np.maximum(np.asarray(kf(lam_um), float), 0.0)
        out[key] = dict(file=rel, reference=ref, comment=com, k=ksrc, range_m=(float(f"{lo * 1e-6:.6g}"), float(f"{hi * 1e-6:.6g}")),
                        n=tuple(float(f"{v:.6g}") for v in n), kk=tuple(float(f"{v:.6g}") for v in k))
    return out


def dumps(table, db_commit):
    lines = ['"""Optical constants n, k on a wavelength grid, generated by goo/substance/generate_nk.py from the',
             'refractiveindex.info database (M. N. Polyanskiy, Sci. Data 11, 94 (2024), doi:10.1038/s41597-023-02898-2),',
             'CC0 1.0 Universal. Do not edit by hand: regenerate. Each entry gives its file, the measurement it reports',
             '(reference), the range where the file\'s data hold (range_m, m), and whether k came from the file or is',
             'k = 0 assumed in a transparent range (k = "zero")."""',
             f"DATABASE = {'refractiveindex.info database, github.com/polyanskiy/refractiveindex.info-database'!r}",
             f"COMMIT = {db_commit!r}", "LICENCE = 'CC0 1.0 Universal'",
             f"GRID_M = tuple(x * 1e-9 for x in range({int(GRID_NM[0])}, {int(GRID_NM[-1]) + 1}, {int(GRID_NM[1] - GRID_NM[0])}))",
             "SOURCES = {"]
    for key, e in table.items():
        lines.append(f"    {key!r}: dict(")
        for f in ("file", "reference", "comment", "k", "range_m"):
            lines.append(f"        {f}={e[f]!r},")
        lines.append(f"        n={e['n']!r},")
        lines.append(f"        kk={e['kk']!r}),")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv):
    db = os.environ.get("REFRACTIVEINDEX_DB")
    if not db: sys.exit("set REFRACTIVEINDEX_DB to a clone of the refractiveindex.info database")
    text = dumps(build(db), commit(db))
    if "--write" in argv:
        OUT.write_text(text); print(f"wrote {OUT}")
    else:
        old = OUT.read_text() if OUT.exists() else ""
        print("identical" if old == text else "DIFFERS from the shipped file")


if __name__ == "__main__":
    main(sys.argv[1:])
