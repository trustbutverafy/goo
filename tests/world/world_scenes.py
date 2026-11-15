"""Scenes for the world's tests: physical magnitudes, generated in code."""
import numpy as np
from goo.units import um, mm, mPa_s, k_B, T_room
from goo.world import World, Environment
from world_analytic import Sub, Analytic, flat_ground, KT_V_SMALL

WATER, GUM, OCHRE = Sub("water", 998.0), Sub("gum", 1350.0), Sub("ochre", 3500.0)
SUBS3 = (WATER, GUM, OCHRE)
KV3 = {"water": KT_V_SMALL, "gum": KT_V_SMALL / 50, "ochre": k_B * T_room / 1e-18}
LAM3 = {"water": 2e-9 / KT_V_SMALL, "gum": 1e-10 / (KT_V_SMALL / 50), "ochre": 2 * (0.5e-6) ** 2 / (9 * 2 * mPa_s)}
HUMID = Environment(activity={"water": 0.5})


def blobs(H, W, centres, radius, amounts):
    yy, xx = np.mgrid[:H, :W]
    q = np.zeros((len(amounts[0]), H, W))
    for (cy, cx), am in zip(centres, amounts):
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        for i, a in enumerate(am):
            q[i][m] += a
    return q


def watercolour(H=24, W=28, tilt=3.0, dx=0.25 * mm, sorb=True, wet=True, tau=None, k=1e-15):
    """Two touching blobs of a gum-water-ochre paint and a clear-water blob on a tilted, porous ground, drying in
    half-humid air."""
    g = flat_ground(H, W, porosity=0.3, depth=80 * um, z=2 * um * np.sin(np.arange(W) / 3.0)[None, :] * np.ones((H, 1)))
    q = blobs(H, W, [(9, 9), (12, 15), (17, 20)], 4,
              [(40 * um, 6 * um, 3 * um), (35 * um, 4 * um, 0.5 * um), (30 * um, 0.0, 0.0)])
    con = Analytic(SUBS3, eta=2 * mPa_s, tau=tau, kT_v=KV3, Lam=LAM3, k={"water": k, "gum": 0.0, "ochre": 0.0},
                   wet=(-3e-3, 0.5 * um) if wet else None,
                   sorb=({"water": 3e4, "gum": 0.0, "ochre": 0.0}, {"water": 2e-15, "gum": 0.0, "ochre": 0.0}) if sorb else None)
    env = Environment(gravity=(0.5, tilt, 9.0), activity={"water": 0.5})
    return World(con, SUBS3, g, dx, env, q=q)


def ledger_identity(r):
    """(residual of dF = W - D_mech - D_num against the state's F, energy moved)."""
    into = sum(float(e.amount) for e in r.entries if e.ledger == "energy" and e.dst == "stored")
    out = sum(float(e.amount) for e in r.entries if e.ledger == "energy" and e.src == "stored")
    dF = sum(a for (_, a) in r.F.values()) - sum(b for (b, _) in r.F.values())
    scale = sum(abs(a) for (_, a) in r.F.values()) + r.report["energy moved"] + 1e-300
    return abs((into - out) - dF) / scale, r.report["energy moved"]


def check_energy(r, tol=1e-12):
    res, moved = ledger_identity(r)
    assert res <= tol, f"energy identity residual {res:.3g}"
    for e in r.entries:
        if e.ledger == "energy" and e.dst.startswith("dissipated["):
            if e.dst == "dissipated[numerical]":
                assert float(e.amount) >= -1e-12 * moved, (e.note, float(e.amount), moved)
            else:
                assert float(e.amount) >= 0.0, (e.dst, float(e.amount))
    assert r.residual["energy"] <= 1e-12 and r.residual["matter"] <= 1e-12, r.residual
