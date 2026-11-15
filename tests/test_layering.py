"""SPEC §1: who may import whom, enforced by scanning imports. Owner: the lead."""
import ast, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "goo"
DIRECT = {
    "units": [], "evidence": [],
    "ledger": ["units", "evidence"],
    "substance": ["units", "evidence"],
    "film": ["units", "evidence", "substance"],
    "light": ["film", "substance", "ledger"],
    "world": ["film", "substance", "ledger"],
    "ports": ["world", "film", "substance", "ledger"],
    "journal": ["world", "ports", "light", "observe"],
    "observe": ["light", "film"],
    "lab": ["substance", "film", "light", "observe"],
}


def closure(m, seen=None):
    seen = set() if seen is None else seen
    for d in DIRECT[m]:
        if d not in seen:
            seen.add(d); closure(d, seen)
    return seen


def module_of(path):
    rel = path.relative_to(ROOT).parts
    return rel[0][:-3] if len(rel) == 1 else rel[0]


def goo_imports(path):
    """Top-level goo modules a file imports, absolute or relative."""
    pkg = ("goo",) + path.relative_to(ROOT).parts[:-1]
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[:len(pkg) - node.level + 1]
                mod = ".".join(base + ((node.module,) if node.module else ()))
                names = [mod] if node.module or len(base) > 1 else [f"goo.{a.name}" for a in node.names]
            else:
                names = [node.module or ""]
                if node.module == "goo":
                    names = [f"goo.{a.name}" for a in node.names]
        else:
            continue
        for n in names:
            parts = n.split(".")
            if parts[0] == "goo" and len(parts) > 1:
                out.add(parts[1])
    return out


def kernel_files():
    """Every .py of the kernel: under goo/, outside goo/classic/, except the root __init__.py."""
    return [p for p in ROOT.rglob("*.py")
            if "classic" not in p.relative_to(ROOT).parts and p != ROOT / "__init__.py"]


def test_every_kernel_module_is_in_the_table():
    for p in kernel_files():
        assert module_of(p) in DIRECT, f"{p}: goo.{module_of(p)} is not a module of SPEC §1"


def test_imports_follow_the_table():
    bad = []
    for p in kernel_files():
        m = module_of(p)
        if m not in DIRECT:
            continue
        for used in goo_imports(p) - {m}:
            if used not in closure(m):
                bad.append(f"{p.relative_to(ROOT.parent)}: goo.{m} imports goo.{used}")
    assert not bad, "\n".join(bad)
