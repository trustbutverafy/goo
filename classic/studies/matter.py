"""The matter kernel, end to end: Mie validation, the classic pigments as points, data sheets made in the
engine, the three inverse-design searches, and one impossible substance carried through. Writes
out/matter-study.json and one figure, out/matter.png (copy to classic/docs/ if worth keeping).

    PYTHONPATH=. python classic/studies/matter.py            # uses cached search results in out/ if present
    PYTHONPATH=. python classic/studies/matter.py --redo     # re-run the searches (a few minutes)
    PYTHONPATH=. python classic/studies/matter.py --fit      # re-fit the classic pigments (slow: ~10 min a pigment)
"""
import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from goo.classic.engine import PIGMENTS, GROUND_RGBN
from goo.classic.palette import linear_to_srgb, lab
from goo.classic.matter import mie, laws, classic, phenotype, search, example, mechanics, gaps
from goo.classic.matter import WATER, LINSEED, Liquid, response

OUT = Path(__file__).resolve().parents[2] / "out"; OUT.mkdir(exist_ok=True)
CACHE = OUT / "matter-searches.json"
J = lambda a: np.asarray(a).tolist() if isinstance(a, np.ndarray) else a


def mie_validation():
    rng = np.random.default_rng(0)
    x = 10 ** rng.uniform(-2, 2.3, 400); m = rng.uniform(1.0, 3.0, 400) + 1j * 10 ** rng.uniform(-4, 0.5, 400); m[:50] = m[:50].real
    qe, qs, qa, g = mie.sphere(m, x)
    out = dict(cases=400, x_range=[0.01, 200], m_range="n 1-3, k 0-3")
    try:
        import miepython as mp
        ref = np.array([mp.efficiencies_mx(np.conj(mi), xi)[:4] for mi, xi in zip(m, x)])
        out.update(miepython=mp.__version__, max_rel_qext=float(np.abs(qe / ref[:, 0] - 1).max()),
                   max_rel_qsca=float(np.abs(qs / ref[:, 1] - 1).max()), max_abs_g=float(np.abs(g - ref[:, 3]).max()),
                   median_rel_qsca=float(np.median(np.abs(qs / ref[:, 1] - 1))))
    except ImportError: out["miepython"] = None
    xs = np.array([0.01]); r = mie.sphere(1.5 + 0.1j, xs); rr = mie.rayleigh(1.5 + 0.1j, xs)
    out["rayleigh_x0.01"] = dict(qsca_ratio=float(r[1][0] / rr[0][0]), qabs_ratio=float(r[2][0] / rr[1][0]))
    out["geometric_qext"] = {str(int(v)): float(mie.sphere(1.5 + 0.5j, v)[0]) for v in (10, 100, 1000)}
    return out


def classic_points():
    rows = {}
    for name in PIGMENTS:
        if name not in classic.FITS: continue
        sub = classic.substance(name); r = classic.residual(name, sub)
        st = phenotype.sheet_from_ks(r["Kt"], r["St"], name); sf = phenotype.sheet_from_ks(r["K"], r["S"], name + " (fit)")
        diel = sub.dielectric
        rows[name] = dict(rms=r["rms"], worst=r["worst"], worst_entry=r["worst_entry"], K=J(r["K"]), S=J(r["S"]),
                          Kt=J(r["Kt"]), St=J(r["St"]), d_um=sub.morphology.size.d_um, sigma_g=sub.morphology.size.sigma_g,
                          n_550=float(diel.nk(np.array([550.0]))[0].real), k_550=float(diel.nk(np.array([550.0]))[0].imag),
                          masstone_dE=float(np.linalg.norm(st["masstone_lab"] - sf["masstone_lab"])),
                          tint_dE=float(np.linalg.norm(st["tint_lab"] - sf["tint_lab"])),
                          masstone_table=J(st["masstone_rgb"]), masstone_fit=J(sf["masstone_rgb"]),
                          klass=sub.klass, sheet=phenotype.format_sheet(dict(sf, klass=sub.klass)),
                          mechanics_oil=sub.mechanics(OIL), mechanics_water=sub.mechanics(H2O))
    return rows


H2O = Liquid("water", WATER, None, density=0.997, viscosity=0.89e-3, p_sat=3169.0, molar_mass=0.018015, surface_tension=0.072)
OIL = Liquid("linseed", LINSEED, None, density=0.93, viscosity=0.05, p_sat=0.0, molar_mass=0.87, surface_tension=0.034)


def run_searches():
    tgt = np.array([75.0, 22 * np.cos(np.radians(210)), 22 * np.sin(np.radians(210))])
    stock = search.stock_gap(tgt)[0]
    t = time.time(); a = search.search_cyan(tgt, maxiter=30); ta = time.time() - t
    sa = phenotype.datasheet(a["sub"])
    t = time.time(); b = search.search_wetdark(maxiter=15); tb = time.time() - t
    Kw, Sw = b["K_wet"], b["S_wet"]; Ka, Sa = b["K_dry"], b["S_dry"]
    sbw = phenotype.sheet_from_ks(Kw, Sw, "wet-dark, in water", carrier="water")
    sba = phenotype.sheet_from_ks(Ka, Sa, "wet-dark, in air", carrier="water")       # water carries nothing optical
    t = time.time(); c = search.search_nir(maxiter=40); tc = time.time() - t
    sc = phenotype.datasheet(c["sub"])
    ns = response.NS_GRID
    S_curve = np.array([b["sub"].ks(n)[1] for n in ns])
    lam = np.r_[np.linspace(400, 700, 61), np.linspace(1000, 2000, 51)]
    return dict(
        a=dict(target_lab=J(tgt), stock_mix=stock[0], stock_dE=stock[2], stock_rgb=J(stock[1]), dE=a["dE"],
               white_share=a["white_share"], lab=J(a["lab"]), K=J(a["K"]), S=J(a["S"]), p=J(a["p"]), seconds=ta,
               klass=a["sub"].klass, desc=a["sub"].describe(), n550=J(a["sub"].dielectric.nk(np.array([550.0]))[0].real),
               impossible=dict(K=J(a["impossible"]["K"]), dK=J(a["impossible"]["dK"]), gain=a["impossible"]["gain"]),
               sheet=phenotype.format_sheet(sa), tint_rgb=J(phenotype.lab_to_linear(a["lab"]))),
        b=dict(L_wet=b["L_wet"], L_dry=b["L_dry"], n_D=b["n_D"], p=J(b["p"]), K_wet=J(Kw), S_wet=J(Sw), K_dry=J(Ka),
               S_dry=J(Sa), klass=b["sub"].klass, sheet_klass=b["sub"].klass, d_um=b["sub"].morphology.size.d_um, sigma_g=b["sub"].morphology.size.sigma_g,
               impossible=dict(K_for_R1=J(b["impossible"]["K_for_R1"]), note=b["impossible"]["note"]),
               sheet_wet=phenotype.format_sheet(sbw), sheet_dry=phenotype.format_sheet(sba), seconds=tb,
               S_vs_ns=J(S_curve), ns=J(ns), over_black_wet=J(sbw["drawdown_black"][-3]), over_black_dry=J(sba["drawdown_black"][-3])),
        c=dict(dE=c["dE"], K=J(c["K"]), S=J(c["S"]), p=J(c["p"]), klass=c["sub"].klass, seconds=tc,
               d_um=c["sub"].morphology.size.d_um, sheet=phenotype.format_sheet(sc),
               k_spec=J(c["sub"].dielectric.nk(lam).imag), n_spec=J(c["sub"].dielectric.nk(lam).real), lam=J(lam),
               impossible=dict(dE=c["impossible"]["dE"], K=J(c["impossible"]["K"]), S=J(c["impossible"]["S"]),
                               report=c["impossible"]["report"],
                               k_spec=J(c["impossible"]["sub"].dielectric.nk(lam).imag),
                               n_spec=J(c["impossible"]["sub"].dielectric.nk(lam).real),
                               kk=laws.kk_residual(c["impossible"]["sub"].dielectric)[0])))


def impossible_example():
    r = example.carry_through()
    rel, E, e1, kk, ext = laws.kk_residual(r["sub"].dielectric)
    return dict(report=r["report"], K=J(r["K"]), S=J(r["S"]), threshold_vol=r["threshold_vol"],
                threshold_film_um=r["threshold_film_um"], n_540=[r["n_540"].real, r["n_540"].imag],
                rows=[dict(vol=x["vol"], film_um=x["film_um"], R_true=J(x["R_true"]), R_engine=J(x["R_engine"]),
                           debt=J(x["debt"])) for x in r["rows"]],
                spectral_debt_per_um=r["debt"]["spectral_debt_per_um"], worst_nm=r["debt"]["worst_nm"],
                sheet=phenotype.format_sheet(r["sheet"]), ledger=laws.ENERGY_LEDGER,
                eps2=J(r["sub"].dielectric.eps(np.linspace(400, 700, 151)).imag))


def _sw(ax, x, y, rgb, w=0.8, h=0.8, label=None, **kw):
    c = linear_to_srgb(np.clip(np.asarray(rgb, float)[:3], 0, 1))
    ax.add_patch(plt.Rectangle((x, y), w, h, color=c, ec="#52514e", lw=0.6))
    if label: ax.text(x + w / 2, y - 0.08, label, ha="center", va="top", fontsize=7, color="#52514e", **kw)


def figure(res, path):
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb"})
    C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)
    # A: classic fits
    ax = fig.add_subplot(gs[0, 0]); cl = res["classic"]; names = list(cl)
    y = np.arange(len(names))
    ax.barh(y, [10 ** cl[n]["rms"] for n in names], color=C1, height=0.55, label="rms factor (8 numbers)")
    ax.scatter([10 ** cl[n]["worst"] for n in names], y, color=C2, s=22, zorder=3, label="worst entry factor")
    for i, n in enumerate(names):
        ax.text(max(10 ** cl[n]["worst"], 10 ** cl[n]["rms"]) * 1.05, i, cl[n]["worst_entry"], va="center", fontsize=7, color="#52514e")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis(); ax.axvline(1, color="#b5b4ae", lw=0.8)
    ax.set_xlabel("derived / table (factor, log; floor 0.01)"); ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("A  classic pigments as points: derived (Mie, in oil) vs table", loc="left", fontsize=9)
    # B: masstones table vs fit, and search (a)
    ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(0, 12); ax.set_ylim(-1.3, 4.2); ax.axis("off")
    for i, n in enumerate(names):
        _sw(ax, i, 3.0, cl[n]["masstone_table"], 0.9, 0.5); _sw(ax, i, 2.45, cl[n]["masstone_fit"], 0.9, 0.5)
        ax.text(i + 0.45, 3.6, n[:7], ha="center", fontsize=6.5, rotation=30, color="#52514e")
    ax.text(11.2, 3.25, "table", fontsize=7, color="#52514e"); ax.text(11.2, 2.7, "derived", fontsize=7, color="#52514e")
    a = res["searches"]["a"]
    _sw(ax, 0.3, 0.5, phenotype.lab_to_linear(a["target_lab"]), 2.2, 1.2, "target\nL 75, C 22, h 210")
    _sw(ax, 3.4, 0.5, a["stock_rgb"], 2.2, 1.2, f"best stock mix\ndE {a['stock_dE']:.1f}")
    _sw(ax, 6.5, 0.5, a["tint_rgb"], 2.2, 1.2, f"nearest possible\ndE {a['dE']:.1f}")
    ax.text(0.3, -1.05, f"possible: n(550) {a['n550']:.2f}, absorbs red, 3% in white.\nExact hit: band K change "
            f"{np.round(a['impossible']['dK'][:3], 1).tolist()}, all K >= 0: no law broken", fontsize=7, color="#0b0b0b")
    ax.set_title("B  masstones in the engine; search (a): a missing cyan", loc="left", fontsize=9)
    # C: search (b): S vs surrounding index, and swatches over black
    ax = fig.add_subplot(gs[0, 2]); b = res["searches"]["b"]
    S = np.array(b["S_vs_ns"])
    for j, (lab_, col) in enumerate((("R", C2), ("G", C3), ("B", C1))):
        ax.plot(b["ns"], S[:, j], color=col, lw=2, label=lab_)
    ax.axvline(1.333, color="#b5b4ae", lw=0.8, ls="--"); ax.text(1.34, S.max() * 0.9, "water", fontsize=7, color="#52514e")
    ax.axvline(1.48, color="#b5b4ae", lw=0.8, ls="--"); ax.text(1.49, S.max() * 0.9, "oil", fontsize=7, color="#52514e")
    ax.set_xlabel("index around the particles"); ax.set_ylabel("S (goo units per 0.025 mm pigment)")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.12, 1.0))
    ins = ax.inset_axes([0.5, 0.5, 0.48, 0.3]); ins.set_xlim(0, 2.2); ins.set_ylim(-0.4, 1); ins.axis("off")
    _sw(ins, 0, 0.1, b["over_black_wet"], 0.9, 0.8, f"50 um over black\nwater: L {b['L_wet']:.1f}"); _sw(ins, 1.2, 0.1, b["over_black_dry"], 0.9, 0.8, f"\nair: L {b['L_dry']:.1f}")
    ax.set_title(f"C  search (b): clear spheres n_D {b['n_D']:.3f}, d {b['d_um']:.2f} um", loc="left", fontsize=9)
    # D: search (c): k spectra
    ax = fig.add_subplot(gs[1, 0]); c = res["searches"]["c"]; lam = np.array(c["lam"])
    v = lam <= 700; nv = lam >= 1000
    for m, (lab_, col, ls) in enumerate(((f"possible (dE {c['dE']:.2f}, K_N {c['K'][3]:.0f})", C1, "-"),
                                         (f"impossible (dE {c['impossible']['dE']:.2f}, KK {c['impossible']['kk']:.0%})", C2, "--"))):
        k = np.array(c["k_spec"] if m == 0 else c["impossible"]["k_spec"])
        ax.plot(lam[v], k[v], color=col, lw=2, ls=ls, label=lab_); ax.plot(lam[nv], k[nv], color=col, lw=2, ls=ls)
    ax.set_yscale("symlog", linthresh=1e-4); ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("k (absorption index)")
    ax.axvspan(1500, 1600, color="#eeeeea"); ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("D  search (c): colourless in the visible, absorbing at 1.5 um", loc="left", fontsize=9)
    # E: impossible example
    ax = fig.add_subplot(gs[1, 1]); im = res["impossible"]
    vols = [r["film_um"] for r in im["rows"]]
    Rt = np.array([r["R_true"] for r in im["rows"]]); Re = np.array([r["R_engine"] for r in im["rows"]])
    Rt_plot = np.where(np.isfinite(Rt[:, 1]), Rt[:, 1], np.nan)
    ax.plot(vols, Rt_plot, "-o", color=C3, lw=2, ms=5, label="green R, two-flux with gain (true)")
    ax.plot(vols, Re[:, 1], "-s", color=C2, lw=2, ms=5, label="green R, engine as it stands (clipped)")
    ax.axhline(1, color="#b5b4ae", lw=0.8); ax.set_ylim(0.9, 3.0)
    if im["threshold_film_um"]: ax.axvline(im["threshold_film_um"], color="#52514e", lw=0.8, ls=":")
    if im["threshold_film_um"]: ax.text(im["threshold_film_um"] * 1.05, 2.0, "lasing\nthreshold", fontsize=7, color="#52514e")
    ax.set_xscale("log"); ax.set_xlabel("film over gesso (um, phi 0.4 in oil)"); ax.set_ylabel("reflectance, G band")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title(f"E  impossible: 'green from nowhere' fails {', '.join(im['report']['fails'])}", loc="left", fontsize=9)
    # F: KK test on the impossible NIR ink (eps1 vs what eps2 + a passive extension allow)
    ax = fig.add_subplot(gs[1, 2])
    from goo.classic.matter.dielectric import Free
    lamc = np.array(c["lam"])
    rel, E, e1, kk, ext = laws.kk_residual(Free(np.linspace(250, 2500, 451), np.interp(np.linspace(250, 2500, 451), lamc, c["impossible"]["n_spec"]),
                                                np.interp(np.linspace(250, 2500, 451), lamc, c["impossible"]["k_spec"])))
    ax.plot(E, e1, color=C1, lw=2, label="eps1 - 1 as written")
    ax.plot(E, kk + ext, color=C2, lw=2, ls="--", label="KK of eps2 + best passive extension")
    ax.set_xlabel("photon energy (eV)"); ax.set_ylabel("eps1 - 1"); ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"F  causality test, impossible NIR ink: remainder {rel:.0%}", loc="left", fontsize=9)
    fig.suptitle("goo.classic.matter: substances as descriptions; derived K, S; searches; an impossible substance", x=0.01, ha="left", fontsize=11)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    if "--fit" in sys.argv:
        for name in PIGMENTS: classic.FITS[name] = classic.fit_pigment(name).tolist(); print(name, classic.residual(name)["rms"])
        print(json.dumps(classic.FITS))
    res = dict(mie=mie_validation(), classic=classic_points())
    if CACHE.exists() and "--redo" not in sys.argv: res["searches"] = json.load(open(CACHE))
    else:
        res["searches"] = run_searches(); json.dump(res["searches"], open(CACHE, "w"), default=J, indent=1)
    res["impossible"] = impossible_example()
    res["liquids"] = dict(water=H2O.mechanics(), linseed=OIL.mechanics(), tick_s=mechanics.TICK_S)
    res["gaps"] = gaps.GAPS
    json.dump(res, open(OUT / "matter-study.json", "w"), default=J, indent=1)
    figure(res, OUT / "matter.png")
    print(json.dumps(dict(mie=res["mie"], classic={k: (round(v["rms"], 3), round(v["worst"], 3), v["worst_entry"]) for k, v in res["classic"].items()}), indent=1))


if __name__ == "__main__":
    main()
