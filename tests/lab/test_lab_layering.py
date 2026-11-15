"""The lab works without the painting engine; nothing here reaches classic. Owner: lab and observers."""
import ast, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FORBIDDEN = {"lab": {"world", "ports", "journal", "classic"}, "observe": {"world", "ports", "journal", "classic", "lab"}}


def _imports(p):
    out = set()
    for node in ast.walk(ast.parse(p.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            if node.module == "goo":
                out |= {f"goo.{a.name}" for a in node.names}
    return out


def test_lab_and_observe_import_neither_world_ports_nor_classic():
    for pkg, bad in FORBIDDEN.items():
        for p in (ROOT / "goo" / pkg).rglob("*.py"):
            used = {m.split(".")[1] for m in _imports(p) if m.startswith("goo.") and len(m.split(".")) > 1}
            assert not used & bad, f"{p}: imports {used & bad}"


def test_these_tests_import_no_classic_world_or_ports():
    for d in ("lab", "observe"):
        for p in (ROOT / "tests" / d).rglob("*.py"):
            used = {m.split(".")[1] for m in _imports(p) if m.startswith("goo.") and len(m.split(".")) > 1}
            assert not used & {"classic", "world", "ports"}, p


def test_importing_the_lab_loads_no_world():
    code = "import sys, goo.lab, goo.lab.search, goo.lab.__main__; " \
           "print(sorted(m for m in sys.modules if m.split('.')[:2] in (['goo','world'], ['goo','ports'], ['goo','classic'])))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT), "PATH": ""})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"
