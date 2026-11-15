"""The clean break: `goo` is the new kernel and knows nothing of goo classic; goo classic imports only itself."""
import ast, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOO = ROOT / "goo"
CLASSIC = GOO / "classic"


def _run(code):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_import_goo_does_not_load_classic():
    out = _run("import sys, goo; assert goo.__file__.startswith(%r), goo.__file__; "
               "print(sorted(m for m in sys.modules if m == 'goo.classic' or m.startswith('goo.classic.'))); "
               "print(goo.__version__)" % str(GOO))
    mods, version = out.splitlines()
    assert mods == "[]"
    assert version == "0.2.0.dev0"


def test_import_goo_classic_warns_once():
    out = _run("import warnings\n"
               "with warnings.catch_warnings(record=True) as w:\n"
               "    warnings.simplefilter('always')\n"
               "    import goo.classic\n"
               "dep = [x for x in w if issubclass(x.category, DeprecationWarning)]\n"
               "print(len(dep)); print(dep[0].message if dep else '')")
    n, msg = out.splitlines()
    assert n == "1"
    assert "goo.classic is goo's first engine, frozen at 6fcca08" in msg


def _imports(path):
    """(module, level) for every import in a file; `from . import x` gives ('', 1) and names separately."""
    pkg = ".".join(path.relative_to(ROOT).with_suffix("").parts[:-1])     # the package the file is in
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
        if isinstance(node, ast.Import):
            for a in node.names: yield a.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[:len(base) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module
            yield mod
            for a in node.names: yield f"{mod}.{a.name}"


def _under(mod, pkg):
    return mod == pkg or mod.startswith(pkg + ".")


def test_kernel_never_imports_classic():
    files = [p for p in GOO.rglob("*.py") if CLASSIC not in p.parents]
    assert GOO / "__init__.py" in files
    bad = [(str(p), m) for p in files for m in _imports(p) if _under(m, "goo.classic")]
    assert not bad, bad


def test_classic_imports_only_itself():
    files = list(CLASSIC.rglob("*.py"))
    assert len(files) > 10
    bad = [(str(p), m) for p in files for m in _imports(p) if _under(m, "goo") and not _under(m, "goo.classic")]
    assert not bad, bad
