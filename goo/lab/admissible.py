"""admissible: the five laws passed and failed, each with its shortfall, and what keeps a substance from being real."""
from goo.substance import classify


def admissible(substance):
    """{name, klass, laws: [{law, ok, shortfall, unit, detail, method, parts}], failed, blocking: [{field, evidence}]}"""
    c = classify(substance)
    return dict(
        name=substance.name, klass=c.klass, material=substance.material or None,
        laws=[dict(law=v.law, ok=v.ok, shortfall=v.shortfall, unit=v.unit, detail=v.detail, method=v.method,
                   parts=[dict(term=t, shortfall=s, unit=u) for t, s, u in v.parts]) for v in c.verdicts],
        failed=list(c.failed), passed=list(c.passed),
        blocking=[dict(field=k, evidence=e) for k, e in c.blocking],
        reservoir_J_m3=substance.reservoir,
        note=("impossible: fails " + ", ".join(c.failed)) if c.failed else
             ("real" if c.klass == "real" else "possible; not real because of the fields in 'blocking'"))


def text(report):
    lines = [f"{report['name']}: {report['klass'].upper()}" + (f" ({report['material']})" if report["material"] else "")]
    for v in report["laws"]:
        lines.append(f"  {'pass' if v['ok'] else 'FAIL'} {v['law']:<15} shortfall {v['shortfall']:.3g} {v['unit']}"
                     f"  [{v['method']}] {v['detail']}")
    if report["blocking"]:
        lines.append("  not real because: " + ", ".join(f"{b['field']} {b['evidence']}" for b in report["blocking"]))
    if report["reservoir_J_m3"]:
        lines.append(f"  reservoir {report['reservoir_J_m3']:g} J/m^3")
    return "\n".join(lines)
