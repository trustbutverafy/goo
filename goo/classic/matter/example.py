"""goo.classic.matter.example: one deliberately impossible substance carried all the way through.

"Green from nowhere": a high-index scatterer (eps_inf 5, a UV band) with a causal Gaussian band of *negative*
strength at 2.3 eV (540 nm): it amplifies green light. Written as oscillators, it is Kramers-Kronig consistent
(gain is causal too); it fails passivity only. Its paint is a debt: below the lasing threshold of a film it
returns more green than it receives, and the reservoir pays the difference; above it (a thick enough, scattering
enough film: |K| S >~ pi^2 / 2 in thickness-scaled units) the two-flux solution has no steady state.
"""
from __future__ import annotations
import numpy as np
from goo.classic.engine import Canvas, GROUND_RGBN
from goo.classic.matter.dielectric import Oscillators, Gauss
from goo.classic.matter.morphology import LogNormal, Sphere
from goo.classic.matter.substance import Substance
from goo.classic.matter import laws, phenotype


def green_from_nowhere(A=-0.02):
    return Substance("green from nowhere", Oscillators(5.0, (Gauss(A, 2.3, 0.12), Gauss(2.0, 4.6, 0.6))),
                     Sphere(LogNormal(0.3, 1.4)), density=4.0, note="optical gain in the green: impossible by design")


def carry_through(sub=None, vols=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05), phi=0.4):
    """Class, failed laws, debt per film, and what the engine paints (through PIGMENTS) beside what the
    two-flux equations actually give (laws.km_any, composed over gesso)."""
    sub = sub or green_from_nowhere()
    rep = laws.report(sub)
    K, S = sub.ks(1.48, exact=True)
    rows = []
    for v in vols:
        oil = v * (1 - phi) / phi
        Kt = K * v * 10 + oil * 10 * phenotype.OIL_K; St = S * v * 10
        R, T = laws.km_any(Kt, St)
        with np.errstate(invalid="ignore"):
            Rg = np.where(np.isfinite(R), R + T * T * GROUND_RGBN / (1 - R * GROUND_RGBN), np.inf)
        Re, Te = Canvas.km_layer(Kt, St)
        Reng = np.clip(Re + Te * Te * GROUND_RGBN / (1 - Re * GROUND_RGBN), 0, 1)
        rows.append(dict(vol=v, film_um=v / phi * 250, R_true=Rg, R_engine=Reng, layer_R=R, layer_T=T,
                         debt=np.maximum(R + T - 1, 0)))     # drawn per unit incident light (the layer alone)
    # lasing threshold in the green: the film pigment volume at which the layer's denominator reaches zero
    vv = np.geomspace(1e-4, 0.2, 400)
    thr = None
    for v in vv:
        R, _ = laws.km_any(K[1:2] * v * 10, S[1:2] * v * 10)
        if not np.isfinite(R[0]): thr = float(v); break
    sheet = phenotype.datasheet(sub)              # the engine, as it stands (it clips: see gaps.py)
    debt = laws.gain_debt(sub, pigment_units=0.05)
    return dict(sub=sub, report=rep, K=K, S=S, rows=rows, threshold_vol=thr,
                threshold_film_um=None if thr is None else thr / phi * 250, sheet=sheet, debt=debt,
                n_540=sub.dielectric.nk(np.array([540.0]))[0])
