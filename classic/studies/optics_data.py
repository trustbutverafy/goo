"""classic/studies/optics_data.py: regenerate goo's optical data table (goo/classic/data/optics.json) from CC0 optical constants.

    REFRACTIVEINDEX_DB=/path/to/refractiveindex.info-database python classic/studies/optics_data.py            # diff only
    REFRACTIVEINDEX_DB=... python classic/studies/optics_data.py --write     # rewrite goo/classic/data/optics.json
    REFRACTIVEINDEX_DB=... python classic/studies/optics_data.py --report    # the tables in classic/docs/OPTICS-DATA.md

REFRACTIVEINDEX_DB is a clone of https://github.com/polyanskiy/refractiveindex.info-database (CC0 1.0). It is read,
never copied: what ships is what this script computes from it. Method (classic/docs/OPTICS-DATA.md):

  optical constants n + ik (refractiveindex.info, each file's own measurement paper cited)
  -> Mie theory (miepython, MIT) for a log-normal particle size distribution (assumed), per surrounding index
  -> transport coefficients per unit pigment volume: mu_a = (3/2) <Q_abs/d>, mu_s' = (3/2) <Q_sca (1 - g)/d>
  -> Kubelka-Munk K = 2 mu_a, S = (3/4) mu_s'   (van Gemert & Star 1987, diffusion limit, mu_a/4 term dropped)

Units: K and S in mm^-1 for the pure pigment (pigment volume fraction 1), at independent scattering. goo's PIGMENTS
are the same quantity times 0.025 mm (classic/SPEC.md section 2: per 0.025 mm of pure pigment, reflectance(gain=10)).

miepython (and, for --report only, colour-science) are dependencies of this script, never of goo at run time.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "goo" / "classic" / "data" / "optics.json"

GRID_NM = np.r_[np.arange(380, 781, 10), 1550].astype(float)          # 41 visible samples + the 1.5 um band
SURROUND = {"oil": 1.48, "water": 1.33, "air": 1.00}                    # OPTICS-SPEC: linseed, water, air
SWEEP_N = [1.00, 1.10, 1.20, 1.33, 1.40, 1.45, 1.48, 1.49, 1.52, 1.56, 1.60]   # medium indices for S(n_medium)
GOO_UNIT_MM = 0.025                                                     # goo K, S are per 0.025 mm of pure pigment
BANDS_NM = {"B": (400, 500), "G": (500, 600), "R": (600, 700)}          # rough bands for comparing with goo's table
SIG = 4                                                                  # significant figures shipped

# goo's PIGMENTS (goo/classic/engine.py), frozen here as of 2026-11 so the shipped "guessed" values do not move when the
# engine's table does. (R, G, B, N) per 0.025 mm of pure pigment. The linseed line is engine._KS's yellowing term
# per unit volume of oil (uncured).
GOO_CLASSIC = {
    "carbon":      dict(K=[3.2, 3.2, 3.2, 3.0], S=[0.10, 0.10, 0.10, 0.10]),
    "white":       dict(K=[0.02, 0.02, 0.03, 0.01], S=[3.2, 3.2, 3.0, 0.45]),
    "ultramarine": dict(K=[2.3, 1.7, 0.20, 0.04], S=[0.05, 0.06, 0.30, 0.05]),
    "ochre":       dict(K=[0.25, 0.55, 1.70, 0.05], S=[0.45, 0.40, 0.25, 0.10]),
    "sienna":      dict(K=[0.30, 1.25, 2.40, 0.08], S=[0.30, 0.20, 0.10, 0.12]),
    "rose":        dict(K=[0.10, 2.30, 0.85, 0.02], S=[0.03, 0.02, 0.04, 0.02]),
    "green":       dict(K=[1.60, 0.30, 1.90, 0.10], S=[0.06, 0.08, 0.04, 0.05]),
    "yellow":      dict(K=[0.04, 0.12, 2.80, 0.02], S=[0.90, 0.85, 0.35, 0.30]),
    "red":         dict(K=[0.06, 2.60, 2.20, 0.03], S=[0.90, 0.30, 0.25, 0.30]),
    "ivory_black": dict(K=[4.6, 4.8, 5.0, 4.5], S=[0.016, 0.016, 0.016, 0.02]),
    "dioxazine":   dict(K=[1.1, 3.4, 0.55, 0.03], S=[0.04, 0.03, 0.06, 0.03]),
    "oil":         dict(K=[0.02, 0.03, 0.06, 0.01], S=[0, 0, 0, 0]),
}

DB_CITE = ("M. N. Polyanskiy, Refractiveindex.info database of optical constants, Sci. Data 11, 94 (2024), "
           "https://doi.org/10.1038/s41597-023-02898-2; database https://github.com/polyanskiy/refractiveindex.info-database, "
           "CC0 1.0 Universal")
MIE_CITE = "miepython (S. Prahl), https://github.com/scottprahl/miepython, MIT licence"
KM_CITE = ("M. J. C. van Gemert and W. M. Star, Relations between the Kubelka-Munk and the transport equation models "
           "for anisotropic scattering, Lasers Life Sci. 1, 287-298 (1987)")

# ---------------------------------------------------------------------------------------------------------------
# Species -> material. "sources": refractiveindex.info files, each with an orientation weight (uniaxial crystals:
# 2/3 ordinary + 1/3 extraordinary, averaged over cross-sections, an approximation for randomly oriented grains).
# "range_nm": where the file's own data or formula holds; outside it the constants are extrapolated (a formula is
# evaluated, a table is held at its end value) and flagged. "k": "file" or "zero" (transparent-range data, k = 0
# assumed). "guess": for species without CC0 constants, a guessed index used only for the medium-dependence of S.
# Sizes: log-normal by volume, median diameter d50 (um) and geometric standard deviation; all assumed.
# ---------------------------------------------------------------------------------------------------------------
SPECIES = {
    "white": dict(role="pigment", material="titanium dioxide, rutile (PW6)",
                  sources=[("main/TiO2/nk/Devore-o.yml", 2 / 3), ("main/TiO2/nk/Devore-e.yml", 1 / 3)],
                  k="zero", size=(0.25, 1.35),
                  note="Devore measured n by prism in rutile's transparent range (0.43-1.53 um); k = 0 is assumed, so "
                       "the absorption edge near 410 nm (rutile's creamy undertone) is missing and 380-420 nm n is the "
                       "formula extrapolated. Pigment surface treatments (alumina, silica) are not modelled."),
    "carbon": dict(role="pigment", material="amorphous carbon: lamp/carbon black, as soot (PBk6/PBk7)",
                   sources=[("other/soots/propane soot/nk/Dalzell.yml", 1.0)], k="file", size=(0.10, 1.5),
                   note="Soot constants (Dalzell & Sarofim) for lamp black; carbon black is fractal aggregates of 20-80 nm "
                        "spheres, here a 0.1 um sphere. Absorption per volume barely depends on size in this regime; "
                        "scattering does strongly (see sensitivity). 380-430 nm holds the 436 nm value."),
    "ivory_black": dict(role="pigment", material="bone/ivory black (PBk9): carbon in hydroxyapatite",
                        sources=[("other/soots/propane soot/nk/Dalzell.yml", 1.0)], k="file", size=(2.0, 1.8),
                        composite=dict(host_n=1.63, f_carbon=0.15),
                        note="ASSUMED composite: Maxwell Garnett mixture of 15 vol% soot (Dalzell & Sarofim constants) in "
                             "hydroxyapatite of index 1.63 (assumed, k = 0; apatite is not in the CC0 database), 2 um "
                             "grains. Carbon fraction and host index are assumptions; a measurement of bone black's own "
                             "n, k (or its K, S) would replace them."),
    "sienna": dict(role="pigment", material="hematite, alpha-Fe2O3 (burnt sienna's chromophore, PBr7)",
                   sources=[("main/Fe2O3/nk/Querry-o.yml", 2 / 3), ("main/Fe2O3/nk/Querry-e.yml", 1 / 3)],
                   k="file", size=(0.30, 1.6),
                   note="Per unit volume of hematite. A natural burnt sienna is hematite with clay and silica (extenders "
                        "of index near oil), which dilute K and S per unit paint volume but not per unit hematite. "
                        "Querry's k in hematite's transparent range (k ~ 0.02-0.04 at 1.55 um) comes from pellet "
                        "reflectance and is probably too high there."),
    "yellow": dict(role="pigment", material="cadmium sulfide, CdS (cadmium yellow, PY37)",
                   sources=[("main/CdS/nk/Treharne.yml", 1.0)], k="file", size=(0.40, 1.5),
                   note="Treharne et al.: an RF-sputtered polycrystalline film, ellipsometry, 0.30-1.50 um; 1550 nm holds "
                        "the 1498 nm value. Pigment CdS (often with ZnS) may differ in the band-edge position."),
    "red": dict(role="pigment", material="mercury sulfide, HgS (vermilion, PR106)",
                sources=[("main/HgS/nk/Bond-o.yml", 2 / 3), ("main/HgS/nk/Bond-e.yml", 1 / 3)], k="zero",
                size=(1.0, 1.6), range_only=True, guess_n=2.9,
                note="Bond, Boyd & Carter measured n of cinnabar from 0.62 um, in its transparent range: S above 620 nm "
                     "and at 1550 nm is computed (k = 0 assumed there); below 620 nm, where vermilion absorbs, there are "
                     "no CC0 constants and K, S are goo's guess. A measured k of HgS across 400-620 nm would fix it."),
    "ochre": dict(role="pigment", material="goethite, alpha-FeOOH (yellow ochre, PY43)", sources=[], guess_n=2.35,
                  size=(0.5, 1.6),
                  note="No goethite constants in the CC0 database. Needs n, k of goethite 380-1550 nm."),
    "ultramarine": dict(role="pigment", material="ultramarine, synthetic lazurite (PB29)", sources=[], guess_n=1.50,
                        size=(1.5, 1.6),
                        note="No lazurite/ultramarine constants in the CC0 database. Needs n, k 380-1550 nm, or measured "
                             "K, S of the pigment in oil."),
    "rose": dict(role="pigment", material="quinacridone or madder lake (PV19 / PR83); organic", sources=[],
                 guess_n=1.70, size=(0.12, 1.5),
                 note="Organic pigment: no CC0 constants. Needs n, k of the pigment crystal or measured K, S."),
    "green": dict(role="pigment", material="viridian, hydrated chromium oxide (PG18), a transparent green", sources=[],
                  guess_n=1.80, size=(1.0, 1.6),
                  note="No Cr2O3 or viridian constants in the CC0 database. Needs n, k 380-1550 nm."),
    "dioxazine": dict(role="pigment", material="dioxazine violet (PV23); organic", sources=[], guess_n=1.70,
                      size=(0.10, 1.5),
                      note="Organic pigment: no CC0 constants. Needs n, k or measured K, S."),
    "gesso": dict(role="ground", material="calcite, CaCO3 (chalk gesso)",
                  sources=[("main/CaCO3/nk/Ghosh-o.yml", 2 / 3), ("main/CaCO3/nk/Ghosh-e.yml", 1 / 3)], k="zero",
                  size=(3.0, 1.8),
                  note="Chalk ground; k = 0 assumed (Ghosh's dispersion fit is for the transparent range). A gypsum "
                       "(Italian) or acrylic (calcite + rutile) gesso would differ; gypsum is not in the CC0 database."),
    "paper": dict(role="ground", material="cellulose (paper fibres)",
                  sources=[("organic/(C6H10O5)n - cellulose/nk/Juntunen.yml", 1.0)], k="zero", size=(20.0, 1.5),
                  note="Juntunen et al. measured microcrystalline cellulose at 1.10-1.65 um; the visible is the Cauchy fit "
                       "extrapolated (n 1.56-1.60). Fibres (ribbons ~20 um wide) are treated as spheres: the absolute "
                       "S is rough, the dependence on the filling index is the point. k = 0 assumed (no lignin, no sizing)."),
    "acrylic": dict(role="binder", material="acrylic polymer (PMMA as the stand-in for an acrylate copolymer)",
                    sources=[("organic/(C5H8O2)n - poly(methyl methacrylate)/nk/Zhang-Tomson.yml", 1.0)], k="file",
                    size=(0.15, 1.25), bulk=True,
                    note="Bulk K is the clear film (S = 0). The per-surrounding K, S are the polymer as 0.15 um latex "
                         "particles: in water, the wet emulsion; in air, an uncoalesced dried latex; in oil, nearly "
                         "index-matched. 380-390 nm hold the 400 nm value."),
    "water": dict(role="carrier", material="liquid water, 25 C", sources=[("main/H2O/nk/Hale.yml", 1.0)], k="file",
                  bulk=True, note="KM K = 2 alpha with alpha = 4 pi k / lambda (diffuse light); S = 0."),
    "oil": dict(role="carrier", material="linseed oil", sources=[], bulk=True, assumed_n=1.48,
                note="No linseed constants in the CC0 database. n = 1.48 at 589 nm (OPTICS-SPEC) is assumed, with the "
                     "dispersion of oleic acid (Jones et al., Cauchy B = 4.565e-3 um^2). K is goo's yellowing guess."),
}
OLEIC = "organic/C18H34O2 - oleic acid/nk/Jones.yml"


# ---- refractiveindex.info reader ------------------------------------------------------------------------------
def _floats(s): return [float(x) for x in str(s).split()]


def read_rii(db, rel):
    """Return (n(lam_um), k(lam_um), (lo_um, hi_um), reference, comment) for one refractiveindex.info file."""
    import yaml
    d = yaml.safe_load(open(Path(db) / "database" / "data" / rel, encoding="utf-8"))
    nfun = kfun = None; lo, hi = np.inf, -np.inf
    for blk in d["DATA"]:
        t = blk["type"]
        if t.startswith("tabulated"):
            a = np.array([_floats(l) for l in blk["data"].strip().splitlines()])
            w = a[:, 0]; lo, hi = min(lo, w.min()), max(hi, w.max())
            if t == "tabulated nk":
                nfun = (lambda x, w=w, v=a[:, 1]: np.interp(x, w, v)); kfun = (lambda x, w=w, v=a[:, 2]: np.interp(x, w, v))
            elif t == "tabulated n": nfun = (lambda x, w=w, v=a[:, 1]: np.interp(x, w, v))
            elif t == "tabulated k": kfun = (lambda x, w=w, v=a[:, 1]: np.interp(x, w, v))
        elif t.startswith("formula"):
            c = _floats(blk["coefficients"]); r = _floats(blk["wavelength_range"]); lo, hi = min(lo, r[0]), max(hi, r[1])
            nfun = (lambda x, f=int(t.split()[1]), c=c: _formula(f, c, np.asarray(x, float)))
    ref = " ".join(str(d.get("REFERENCES", "")).split()); com = " ".join(str(d.get("COMMENTS", "")).split())
    return nfun, kfun, (lo, hi), ref, com


def _formula(f, c, x):
    """refractiveindex.info dispersion formulas 1-5 (database/doc/Dispersion formulas.pdf); x in um."""
    c = list(c) + [0.0] * 20; x2 = x * x
    if f == 1:
        n2 = 1 + c[0] + sum(c[i] * x2 / (x2 - c[i + 1] ** 2) for i in range(1, 17, 2)); return np.sqrt(n2)
    if f == 2:
        n2 = 1 + c[0] + sum(c[i] * x2 / (x2 - c[i + 1]) for i in range(1, 17, 2)); return np.sqrt(n2)
    if f == 3:
        n2 = c[0] + sum(c[i] * x ** c[i + 1] for i in range(1, 17, 2)); return np.sqrt(n2)
    if f == 4:
        n2 = (c[0] + c[1] * x ** c[2] / (x2 - c[3] ** c[4]) + c[5] * x ** c[6] / (x2 - c[7] ** c[8])
              + sum(c[i] * x ** c[i + 1] for i in range(9, 17, 2)))
        return np.sqrt(n2)
    if f == 5:
        return c[0] + sum(c[i] * x ** c[i + 1] for i in range(1, 17, 2))
    raise ValueError(f"formula {f} not supported")


def _clean_ref(ref):
    import re
    return re.sub(r"<[^>]+>", "", ref).replace("&amp;", "&").strip()


# ---- Mie -> Kubelka-Munk ---------------------------------------------------------------------------------------
def lognormal_nodes(d50, sg, n=24):
    """Gauss-Hermite nodes and weights for a log-normal volume distribution (median d50, geometric sd sg)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return d50 * np.exp(np.log(sg) * x), w / w.sum()


def mie_km(m_particle, d50, sg, lam_um, n_medium):
    """K, S (mm^-1, per unit pigment volume fraction) and g for a log-normal (by volume) polydispersion of spheres of
    complex index m_particle in a non-absorbing medium. K = 2 mu_a, S = (3/4) mu_s (1 - g)."""
    import miepython
    d, w = lognormal_nodes(d50, sg)
    m = complex(np.real(m_particle), -abs(np.imag(m_particle)))
    qext, qsca, qback, g = miepython.efficiencies(m, d, lam_um, n_medium)
    qabs = np.maximum(np.asarray(qext) - np.asarray(qsca), 0.0)
    mua = 1.5 * np.sum(w * qabs / d) * 1e3                  # per um -> per mm
    mus_r = 1.5 * np.sum(w * np.asarray(qsca) * (1 - np.asarray(g)) / d) * 1e3
    mus = 1.5 * np.sum(w * np.asarray(qsca) / d) * 1e3
    return 2 * mua, 0.75 * mus_r, mus, mus_r


def maxwell_garnett(m_incl, m_host, f):
    ei, eh = m_incl ** 2, m_host ** 2
    return np.sqrt(eh * (ei + 2 * eh + 2 * f * (ei - eh)) / (ei + 2 * eh - f * (ei - eh)))


# ---- building the table ----------------------------------------------------------------------------------------
def optical_constants(db, name, spec):
    """Per orientation: list of (weight, n(42), k(42)); plus provenance of the constants per wavelength and citations."""
    lam_um = GRID_NM / 1e3
    parts, cites, extrap = [], [], np.zeros(len(GRID_NM), bool)
    inrange = np.ones(len(GRID_NM), bool)
    for rel, wgt in spec.get("sources", []):
        nf, kf, (lo, hi), ref, com = read_rii(db, rel)
        out = (lam_um < lo - 1e-9) | (lam_um > hi + 1e-9)
        x = lam_um.copy()
        if spec.get("range_only"):
            inrange &= ~(lam_um < lo - 1e-9)                  # below the measured range: no constants at all
            x = np.maximum(x, lo)
        n = np.asarray(nf(x), float)
        k = np.zeros_like(n) if (spec.get("k") == "zero" or kf is None) else np.maximum(np.asarray(kf(x), float), 0.0)
        if spec.get("range_only"): out &= lam_um > hi
        extrap |= out
        parts.append((wgt, n, k))
        cites.append(dict(file=rel, reference=_clean_ref(ref), comment=_clean_ref(com)[:240],
                          range_um=[round(lo, 4), round(hi, 4)]))
    return parts, cites, extrap, inrange


def _db_commit(db):
    import subprocess
    try: return subprocess.run(["git", "-C", str(db), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError: return None


def build(db, only=None):
    """The whole table, or (only=[names]) just those species (for the regeneration test)."""
    import miepython
    lam_um = GRID_NM / 1e3
    out = dict(
        schema="goo.optics-data/1",
        wavelength_nm=GRID_NM.tolist(),
        surroundings=SURROUND,
        sweep_n_medium=SWEEP_N,
        units=dict(K="mm^-1 for pure pigment (pigment volume fraction 1); goo units = value * 0.025 mm",
                   S="mm^-1 for pure pigment; goo units = value * 0.025 mm", n="real refractive index",
                   k="extinction coefficient (imaginary index)"),
        method=dict(
            mie=MIE_CITE + f" (version {miepython.__version__})",
            km_relation="K = 2 mu_a, S = (3/4) mu_s (1 - g); " + KM_CITE + "; diffusion limit, the -mu_a/4 term dropped",
            size_distribution="log-normal by volume (d50, geometric sd), Gauss-Hermite, 24 nodes; sizes are assumed",
            birefringence="uniaxial crystals: cross-sections averaged 2/3 ordinary + 1/3 extraordinary",
            limits="independent scattering only: no dependent scattering, no crowding at high pigment volume, "
                   "no aggregation; spheres; non-absorbing surroundings"),
        database=DB_CITE,
        database_commit=_db_commit(db),
        species={})
    oleic_n = None
    for name, spec in SPECIES.items():
        if only is not None and name not in only: continue
        rec = dict(role=spec["role"], material=spec["material"], note=spec["note"])
        parts, cites, extrap, inrange = optical_constants(db, name, spec)
        if name == "oil":
            nf, _, _, ref, _ = read_rii(db, OLEIC)
            B = 4.565e-3; C = 1e-4                               # Jones et al. Cauchy terms (um^2, um^4)
            A = 1.48 - B / 0.589 ** 2 - C / 0.589 ** 4
            n = A + B / lam_um ** 2 + C / lam_um ** 4
            parts = [(1.0, n, np.zeros_like(n))]
            cites = [dict(file=OLEIC, reference=_clean_ref(ref), used_for="dispersion shape only")]
        if spec.get("composite"):
            c = spec["composite"]
            wgt, n, k = parts[0]
            meff = maxwell_garnett(n + 1j * k, c["host_n"] + 0j, c["f_carbon"])
            parts = [(1.0, meff.real, meff.imag)]
        guessed = not parts
        prov = dict(n=[], K=[], S=[])
        if guessed:
            n = np.full(len(GRID_NM), spec["guess_n"]); k = np.zeros_like(n)
            rec["n"] = n.tolist(); rec["k"] = k.tolist()
            prov = {q: ["guessed"] * len(GRID_NM) for q in ("n", "k", "K", "S")}
        else:
            n = sum(w * p for w, p, _ in parts); k = sum(w * q for w, _, q in parts)
            if not inrange.all():
                n = np.where(inrange, n, spec["guess_n"]); k = np.where(inrange, k, 0.0)
            rec["n"] = n.tolist(); rec["k"] = k.tolist()
            base = "assumed" if (spec.get("composite") or name == "oil") else "computed"
            prov["n"] = [base if ok else "guessed" for ok in inrange]
            prov["k"] = [("assumed" if spec.get("k") == "zero" or name == "oil" else base) if ok else "guessed"
                         for ok in inrange]
        rec["sources"] = cites
        rec["extrapolated_nm"] = [int(x) for x in GRID_NM[extrap]]

        if spec.get("bulk"):                                    # a continuous medium: K = 2 alpha, S = 0
            kk = np.asarray(rec["k"]); alpha = 4 * np.pi * kk / (lam_um * 1e-3)
            Kb = 2 * alpha
            if name == "oil":                                   # goo's yellowing guess per unit oil volume
                Kb = _spread(GOO_CLASSIC["oil"]["K"]) / GOO_UNIT_MM
                prov["K"] = ["guessed"] * len(GRID_NM)
            else:
                prov["K"] = ["computed" if p == "computed" else p for p in prov["k"]]
            rec["K"] = dict(bulk=Kb.tolist()); rec["S"] = dict(bulk=np.zeros(len(GRID_NM)).tolist())
            prov["S"] = prov["S"] or ["computed"] * len(GRID_NM)
        if spec["role"] in ("pigment", "ground") or name == "acrylic":
            d50, sg = spec["size"]
            rec["size"] = dict(d50_um=d50, sigma_g=sg, distribution="log-normal by volume", provenance="assumed")
            K = {}; S = {}
            sweepK = np.zeros((len(SWEEP_N), len(GRID_NM))); sweepS = np.zeros_like(sweepK)
            mpart = n + 1j * k
            for j, nm in enumerate(SWEEP_N):
                for i, lam in enumerate(lam_um):
                    kk_, ss_ = 0.0, 0.0
                    if guessed or (not inrange[i]):
                        _, ss_, _, _ = mie_km(complex(n[i], 0.0), d50, sg, lam, nm)
                    else:
                        for w, pn, pk in parts:
                            a, b, _, _ = mie_km(complex(pn[i], pk[i]), d50, sg, lam, nm); kk_ += w * a; ss_ += w * b
                    sweepK[j, i] = kk_; sweepS[j, i] = ss_
            # guessed wavelengths: K and S in oil from goo's table; other media scale S by the Mie ratio; K kept
            gmask = np.array([guessed or not ok for ok in inrange])
            if gmask.any():
                gk = _spread(GOO_CLASSIC[name]["K"]) / GOO_UNIT_MM; gs = _spread(GOO_CLASSIC[name]["S"]) / GOO_UNIT_MM
                j_oil = SWEEP_N.index(SURROUND["oil"])
                ratio = sweepS / np.maximum(sweepS[j_oil], 1e-30)
                sweepS = np.where(gmask, gs * ratio, sweepS)
                sweepK = np.where(gmask, gk, sweepK)
            for s, nm in SURROUND.items():
                j = SWEEP_N.index(nm); K[s] = sweepK[j].tolist(); S[s] = sweepS[j].tolist()
            if spec.get("bulk"):
                K.update(rec["K"]); S.update(rec["S"])
            rec["K"] = K; rec["S"] = S
            rec["sweep"] = dict(K=sweepK.tolist(), S=sweepS.tolist())
            kprov = []
            for i in range(len(GRID_NM)):
                if gmask[i]: kprov.append("guessed")
                elif spec.get("composite"): kprov.append("assumed")
                else: kprov.append("computed")
            prov["K"] = kprov; prov["S"] = list(kprov)
            rec["sensitivity"] = sensitivity(spec, parts, guessed, inrange)
        rec["provenance"] = dict(by_wavelength=prov, summary=_summary(prov))
        out["species"][name] = rec
    return _round(out)


def sensitivity(spec, parts, guessed, inrange):
    """K and S in oil at 550 and 1550 nm for d50 x 0.5, x 1, x 2: how much the assumed size matters."""
    d50, sg = spec["size"]; res = {}
    for f in (0.5, 1.0, 2.0):
        row = {}
        for lam in (550.0, 1550.0):
            i = int(np.where(GRID_NM == lam)[0][0])
            if guessed or not inrange[i]:
                row[f"S_oil_{int(lam)}"] = None; row[f"K_oil_{int(lam)}"] = None; continue
            kk = ss = 0.0
            for w, pn, pk in parts:
                a, b, _, _ = mie_km(complex(pn[i], pk[i]), d50 * f, sg, lam / 1e3, SURROUND["oil"]); kk += w * a; ss += w * b
            row[f"K_oil_{int(lam)}"] = kk; row[f"S_oil_{int(lam)}"] = ss
        res[f"d50_x{f:g}"] = row
    return res


def _summary(prov):
    kinds = set(prov["K"]) | set(prov["S"]) | set(prov["n"])
    if kinds == {"computed"} or kinds == {"computed", "assumed"} and set(prov["K"]) | set(prov["S"]) == {"computed"}:
        return "computed"
    if len(kinds) == 1: return kinds.pop()
    if "guessed" in kinds and "computed" in kinds: return "mixed"
    return "assumed" if "guessed" not in kinds else "guessed"


def _spread(v4):
    """goo's (R, G, B, N) band values onto the grid: B for 380-490 nm, G 500-590, R 600-780, N at 1550 (flat)."""
    R, G, B, N = v4
    return np.array([B if l < 500 else G if l < 600 else R if l <= 780 else N for l in GRID_NM], float)


def _round(o):
    if isinstance(o, dict): return {k: _round(v) for k, v in o.items()}
    if isinstance(o, list): return [_round(v) for v in o]
    if isinstance(o, float):
        return 0.0 if o == 0 else float(f"{o:.{SIG}g}")
    if isinstance(o, (np.floating,)): return _round(float(o))
    return o


def dumps(table):
    """JSON with one array per line (diffable, compact)."""
    s = json.dumps(table, indent=1, ensure_ascii=False)
    import re
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", s) + "\n"


def diff(new, old, rtol=1e-3):
    """Lines describing where two tables differ by more than rtol (relative)."""
    lines = []
    for name in sorted(set(new["species"]) | set(old.get("species", {}))):
        a, b = new["species"].get(name), old.get("species", {}).get(name)
        if a is None or b is None: lines.append(f"{name}: {'added' if b is None else 'removed'}"); continue
        for q in ("n", "k"):
            _cmp(lines, f"{name}.{q}", a.get(q), b.get(q), rtol)
        for q in ("K", "S"):
            for s in sorted(set(a.get(q, {})) | set(b.get(q, {}))):
                _cmp(lines, f"{name}.{q}.{s}", a.get(q, {}).get(s), b.get(q, {}).get(s), rtol)
        if a["provenance"]["summary"] != b["provenance"]["summary"]:
            lines.append(f"{name}: provenance {b['provenance']['summary']} -> {a['provenance']['summary']}")
    return lines


def _cmp(lines, label, x, y, rtol):
    if x is None or y is None:
        if x is not y: lines.append(f"{label}: present in only one table")
        return
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape: lines.append(f"{label}: shape {y.shape} -> {x.shape}"); return
    r = np.abs(x - y) / np.maximum(np.maximum(np.abs(x), np.abs(y)), 1e-12)
    if r.max() > rtol:
        i = int(r.argmax()); lines.append(f"{label}: max rel diff {r.max():.3g} at {GRID_NM[i]:.0f} nm ({y[i]:.5g} -> {x[i]:.5g})")


# ---- report (the tables in classic/docs/OPTICS-DATA.md) -----------------------------------------------------------------
def report(table):
    import miepython
    W = np.asarray(table["wavelength_nm"]); sp = table["species"]; vis = W <= 780

    sys.path.insert(0, str(HERE.parents[1]))
    from goo.classic.data import resample

    def band(v, lo, hi):                                  # the loader's band mean (trapezoid), or the 1550 nm value
        return float(resample(v, [lo if lo == hi else (lo, hi)], W)[0])

    print("## Provenance\n\n| species | role | material | n | K | S | summary |\n|---|---|---|---|---|---|---|")
    for name, r in sp.items():
        p = r["provenance"]["by_wavelength"]
        f = lambda q: "/".join(sorted(set(p.get(q, [])))) or "-"
        print(f"| {name} | {r['role']} | {r['material']} | {f('n')} | {f('K')} | {f('S')} | {r['provenance']['summary']} |")

    print("\n## Ratio to goo's current table (ours in oil / goo), band means B 400-500, G 500-600, R 600-700 nm, N 1550\n")
    print("| species | K: R | G | B | N | S: R | G | B | N |\n|---|---|---|---|---|---|---|---|---|")
    for name in GOO_CLASSIC:
        if name == "oil" or sp[name]["provenance"]["summary"] == "guessed": continue
        r = sp[name]; K = np.asarray(r["K"]["oil"]) * GOO_UNIT_MM; S = np.asarray(r["S"]["oil"]) * GOO_UNIT_MM
        g = GOO_CLASSIC[name]; cells = []
        for q, arr in (("K", K), ("S", S)):
            for j, (b, (lo, hi)) in enumerate(list(BANDS_NM.items())[::-1] + [("N", (1550, 1550))]):
                ours = band(arr, lo, hi); ref = g[q][j]
                cells.append(f"{ours:.3g}/{ref:g} = {ours / ref:.2g}" if ref > 0 else f"{ours:.3g}/0")
        print(f"| {name} | " + " | ".join(cells) + " |")

    print("\n## Size sensitivity (oil; K, S in mm^-1 at 550 and 1550 nm; d50 x 0.5 / x 1 / x 2)\n")
    for name, r in sp.items():
        s = r.get("sensitivity")
        if not s or s["d50_x1"]["S_oil_550"] is None and s["d50_x1"]["S_oil_1550"] is None: continue
        row = []
        for key in ("S_oil_550", "S_oil_1550", "K_oil_550"):
            vals = [s[f"d50_x{f}"][key] for f in ("0.5", "1", "2")]
            row.append(key + " " + " / ".join("-" if v is None else f"{v:.3g}" for v in vals))
        print(f"- {name} (d50 {r['size']['d50_um']} um): " + "; ".join(row))

    print("\n## Rutile check: monodisperse spheres, binder n = 1.5, 550 nm\n")
    nf_o, _, _, _, _ = read_rii(os.environ["REFRACTIVEINDEX_DB"], SPECIES["white"]["sources"][0][0])
    nf_e, _, _, _, _ = read_rii(os.environ["REFRACTIVEINDEX_DB"], SPECIES["white"]["sources"][1][0])
    no, ne = float(nf_o(0.55)), float(nf_e(0.55))
    ds = np.linspace(0.05, 0.8, 76); best = []
    for d in ds:
        tot = 0.0; totmus = 0.0
        for nn, w in ((no, 2 / 3), (ne, 1 / 3)):
            qe, qs, qb, g = miepython.efficiencies(complex(nn, 0), d, 0.55, 1.5)
            tot += w * 1.5 * qs * (1 - g) / d; totmus += w * 1.5 * qs / d
        best.append((tot, d, totmus))
    t, dopt, mus = max(best)
    at22 = [b for b in best if abs(b[1] - 0.22) < 1e-6][0]
    print(f"- n_o {no:.3f}, n_e {ne:.3f}; mu_s(1-g)/phi peaks at d = {dopt:.2f} um: {t:.2f} um^-1 (mu_s/phi {mus:.2f} um^-1)")
    print(f"- at d = 0.22 um: mu_s(1-g)/phi = {at22[0]:.2f} um^-1, mu_s/phi = {at22[2]:.2f} um^-1, KM S/phi = {0.75 * at22[0]:.2f} um^-1")
    i550 = int(np.where(W == 550)[0][0])
    print(f"- shipped white, log-normal d50 0.25: S_oil(550) = {sp['white']['S']['oil'][i550] / 1e3:.2f} um^-1")

    print("\n## Wet and dry: S in water / S in oil and S in air / S in oil, band means (and 1550 nm)\n")
    print("| species | water/oil R G B N | air/oil R G B N | air/water G |\n|---|---|---|---|")
    for name, r in sp.items():
        if "oil" not in r["S"] or r["role"] == "carrier": continue
        S = {s: np.asarray(r["S"][s]) for s in SURROUND}
        def rat(a, b): return [band(S[a], lo, hi) / max(band(S[b], lo, hi), 1e-30) for lo, hi in
                               [BANDS_NM["R"], BANDS_NM["G"], BANDS_NM["B"], (1550, 1550)]]
        w, a = rat("water", "oil"), rat("air", "oil")
        print(f"| {name} ({r['provenance']['summary']}) | " + " ".join(f"{x:.3g}" for x in w) + " | "
              + " ".join(f"{x:.3g}" for x in a) + f" | {a[1] / w[1]:.3g} |")

    print("\n## Size sensitivity of the wet/dry ratios at 550 nm (computed species): air/oil and water/oil for d50 x 0.5 / x 1 / x 2\n")
    db = os.environ["REFRACTIVEINDEX_DB"]
    for name, spec in SPECIES.items():
        r = sp[name]
        if "size" not in spec or r["provenance"]["by_wavelength"]["S"][i550] == "guessed": continue
        parts, _, _, _ = optical_constants(db, name, spec)
        if spec.get("composite"):
            _, n0, k0 = parts[0]; c = spec["composite"]; me = maxwell_garnett(n0 + 1j * k0, c["host_n"] + 0j, c["f_carbon"])
            parts = [(1.0, me.real, me.imag)]
        cells = []
        for f in (0.5, 1.0, 2.0):
            Sm = {}
            for s_, nm in SURROUND.items():
                Sm[s_] = sum(w * mie_km(complex(pn[i550], pk[i550]), spec["size"][0] * f, spec["size"][1], 0.55, nm)[1]
                             for w, pn, pk in parts)
            cells.append(f"{Sm['air'] / Sm['oil']:.3g}, {Sm['water'] / Sm['oil']:.3g}")
        print(f"- {name}: " + " / ".join(cells))

    geo = []
    for name in ("white", "carbon", "sienna", "yellow"):
        for q in ("K", "S"):
            for j, (lo, hi) in enumerate([BANDS_NM["R"], BANDS_NM["G"], BANDS_NM["B"]]):
                ours = band(np.asarray(sp[name][q]["oil"]) * GOO_UNIT_MM, lo, hi); ref = GOO_CLASSIC[name][q][j]
                if ours > 1e-3 * ref and ref > 0: geo.append(ours / ref)
    print(f"\n- geometric mean of ours/goo over computed pigments' visible K and S (non-zero): {np.exp(np.mean(np.log(geo))):.3g} "
          f"(range {min(geo):.2g}..{max(geo):.2g}, {len(geo)} values)")

    print("\n## Hiding in oil: film thickness (um) at pigment volume 0.20 for contrast ratio 0.98 (over black / over "
          "R = 0.80), at 550 nm and at the pigment's least-absorbing wavelength in 450-700 nm (where a glaze shows "
          "what is under it)\n")
    m = (W >= 450) & (W <= 700)
    for name, r in sp.items():
        if r["role"] != "pigment": continue
        K = np.asarray(r["K"]["oil"]) * 0.2; S = np.asarray(r["S"]["oil"]) * 0.2
        ks = K / np.maximum(S, 1e-12); j = np.where(m)[0][int(np.argmin(ks[m]))]
        print(f"- {name} ({r['provenance']['summary']}): K/S(550) = {ks[i550]:.3g}, hiding(550) = {hiding(K[i550], S[i550]):.3g}; "
              f"least-absorbing {W[j]:.0f} nm: K/S = {ks[j]:.3g}, hiding = {hiding(K[j], S[j]):.3g}")

    print("\n## Masstone in oil: R_inf from K/S, Saunderson (k1 = 0.04, k2 = 0.60), D65, CIE 1931 2 deg, sRGB\n")
    for name, r in sp.items():
        if r["role"] not in ("pigment", "ground"): continue
        rgb, lab, Rinf = masstone(W[vis], np.asarray(r["K"]["oil"])[vis], np.asarray(r["S"]["oil"])[vis])
        print(f"- {name} ({r['provenance']['summary']}): sRGB {tuple(int(round(255 * c)) for c in rgb)}, "
              f"L* {lab[0]:.0f} a* {lab[1]:.0f} b* {lab[2]:.0f}; R_inf 450/550/650 = "
              + "/".join(f"{np.interp(x, W[vis], Rinf):.3f}" for x in (450, 550, 650)))

    print("\n## 1550 nm: K, S in oil (mm^-1), K/S, and R of a 50 um film at pigment volume 0.3 over black / over white (0.9)\n")
    i = len(W) - 1
    for name, r in sp.items():
        if r["role"] != "pigment": continue
        K = r["K"]["oil"][i]; S = r["S"]["oil"][i]
        X = 0.05 * 0.3
        Rb = layer_R(K * X, S * X, 0.0); Rw = layer_R(K * X, S * X, 0.9)
        print(f"- {name} ({r['provenance']['by_wavelength']['S'][i]}): K {K:.3g}, S {S:.3g}, K/S {K / max(S, 1e-12):.3g}, "
              f"over black {Rb:.3f}, over white {Rw:.3f}")


def layer_R(KX, SX, Rg):
    S = max(SX, 1e-9); a = 1 + KX / S; b = np.sqrt(max(a * a - 1, 1e-14)); bs = min(b * S, 30)
    sh, ch = np.sinh(bs), np.cosh(bs); R = sh / (a * sh + b * ch); T = b / (a * sh + b * ch)
    return R + T * T * Rg / (1 - R * Rg)


def hiding(K, S, Rw=0.8, cr=0.98):
    """Film thickness (um) for contrast ratio R(black)/R(white backing) = cr; K, S in mm^-1 of the film."""
    for X in np.geomspace(1e-4, 50, 4000):              # mm
        if layer_R(K * X, S * X, 0.0) / layer_R(K * X, S * X, Rw) >= cr: return X * 1e3
    return np.inf


def masstone(W, K, S, k1=0.04, k2=0.60):
    import colour
    ks = K / np.maximum(S, 1e-12)
    Rinf = 1 + ks - np.sqrt(ks * ks + 2 * ks)
    R = (1 - k1) * (1 - k2) * Rinf / (1 - k2 * Rinf)
    sd = colour.SpectralDistribution(dict(zip(W.astype(int), R)))
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ill = colour.SDS_ILLUMINANTS["D65"]
    XYZ = colour.sd_to_XYZ(sd, cmfs, ill) / 100
    rgb = np.clip(colour.XYZ_to_sRGB(XYZ), 0, 1)
    lab = colour.XYZ_to_Lab(XYZ, colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
    return rgb, lab, Rinf


def main(argv):
    db = os.environ.get("REFRACTIVEINDEX_DB")
    if not db or not (Path(db) / "database" / "data").is_dir():
        sys.exit("set REFRACTIVEINDEX_DB to a clone of https://github.com/polyanskiy/refractiveindex.info-database")
    new = build(db)
    if "--report" in argv:
        report(new); return
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    lines = diff(new, old)
    print("\n".join(lines) if lines else "no differences from the shipped table (rtol 1e-3)")
    if "--write" in argv:
        OUT.write_text(dumps(new)); print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv[1:])
