"""goo.matter.gaps: every constitutive parameter goo.matter derives that goo's engine cannot consume today,
where the engine holds it instead (a global constant, or a law hard-wired to a species index), and what the
law would read instead. This is the specification for the next change to the engine. Line numbers refer to
goo/engine.py as of this kernel (551 lines; `Canvas.L` defaults at lines 177-181).

Mixing rules: x_s is the volume fraction of solid species s among the solids, y_l of liquid l among the
liquids; a cell's surrounding index n_s is what fills the space around the particles (v2 optics: the liquids'
volume-weighted index up to the bed's void capacity, air beyond).
"""

GAPS = [
    dict(param="K(n_s), S(n_s) per pigment, per band", derived="Mie + van Gemert-Star (response.py)",
         line="29-41 (PIGMENTS constants), 510-516 (_KS reads PIGMENTS[s]['K'], ['S'] as numbers)",
         now="K, S fixed numbers, defined in no surrounding (in fact: oil)",
         instead="_KS evaluates each species' response function at the cell's n_s per band: "
                 "K = sum_s v_s K_s(n_s), S = sum_s v_s S_s(n_s)"),
    dict(param="phi_m (max packing, also CPVC) per particle species", derived="Desmond-Weeks RCP of the size distribution",
         line="177 (L['phi_m'] = 0.64 global); used at 242 (_eta) and 244-245 (_tau)",
         now="one phi_m for every pigment and mixture",
         instead="phi_m(cell) = sum_s x_s phi_m,s (first-order rule; bimodal mixtures pack denser, Farr & Groot 2009)"),
    dict(param="intrinsic viscosity [eta]_s and Krieger exponent [eta] phi_m", derived="Einstein 5/2 for spheres; Krieger-Dougherty",
         line="242 (exponent -2 hard-coded in _eta)",
         now="exponent 2 for every pigment (equals [eta] = 3.125 at phi_m 0.64, not the sphere's 2.5, which gives 1.6)",
         instead="eta = eta_liq (1 - phi/phi_m)^(-[eta]_mix phi_m), [eta]_mix = sum_s x_s [eta]_s"),
    dict(param="liquid viscosity eta_l", derived="Liquid.viscosity (water 0.89 mPa s, linseed ~50 mPa s)",
         line="177 (L['eta0'] = 0.12 global), used at 242, 321, 431",
         now="one carrier viscosity for water and oil alike",
         instead="eta_liq(cell) = exp(sum_l y_l ln eta_l) (Arrhenius log-mixing of the liquids present)"),
    dict(param="which species are solids, which liquids (roles)", derived="morphology is None or not",
         line="26 (CARRIERS = ['water', 'oil']), 166-167 (species = CARRIERS + pigments, nc = 2), 239 (phi = q[nc:] / h)",
         now="role by position in the species list",
         instead="role by description: particles are solids, continuous phases are liquids; phi = sum over particle species"),
    dict(param="colour diffusivity D_s per pigment", derived="Stokes-Einstein kT / (3 pi eta d_s)",
         line="177 (L['D0'] = 0.15 global), 321 (_nsub), 431-437 (one Dc for all pigments)",
         now="one D for every pigment, scaled by eta0/eta; magnitude 1e3-1e6 x Brownian (docs/MATTER.md) (a stand-in for shear mixing)",
         instead="per-species D_s = D_ref (d_ref / d_s)(eta_ref / eta); Brownian part from the derived D_s, and "
                 "any shear mixing declared as a separate law"),
    dict(param="settling speed v_s per pigment", derived="Stokes (rho_p - rho_f) g d^2 / 18 eta, Richardson-Zaki (1-phi)^4.65",
         line="none: the wet layer is depth-averaged (SPEC 8); no density anywhere",
         now="no settling, no granulation", instead="a settled-fraction state per species (or two sublayers), fed at v_s / h"),
    dict(param="density rho_s per species", derived="Substance.density; envelope density for hollow/coated",
         line="none", now="species have no mass", instead="needed by settling and by any gravity-driven law"),
    dict(param="volatility (evaporation rate) per liquid", derived="diffusion-limited J = D_v M p_sat (1-RH) / (R T delta)",
         line="177 (L['evap'], L['edge_evap']); 444-450 (only q[0] evaporates)",
         now="water (index 0) is the only volatile, at a global rate",
         instead="each liquid l evaporates at its own rate e_l from p_sat; evaporated volume booked per species in the ledger"),
    dict(param="absorption into the ground per liquid", derived="viscosity, surface tension (Washburn: rate ~ gamma r / 4 eta)",
         line="178 (L['absorb'], L['store_cap']); 451-455 (only q[0] soaks in); 170 (_store is water only)",
         now="only water soaks in", instead="per-liquid soak rate ~ gamma_l / eta_l, and a store per liquid"),
    dict(param="cure (a Change of the binder) and its rate", derived="Change(product, rate, yields)",
         line="179 (L['cure_rate']); 457-459 (q[1] cures); 475 (fix: lean = q[1] <= 5% of h)",
         now="oil (index 1) is the only curing species; dryness tests q[0] and q[1] by position",
         instead="each species' declared Changes advance its own state; dry = no volatile left and every binder "
                 "past its cure threshold"),
    dict(param="binder absorption (yellowing) spectrum", derived="the binder's Im eps (dielectric) and its change state",
         line="516 (K += vol[1] * [0.02, 0.03, 0.06, 0.01] * (1 + 2 kappa))",
         now="hard-wired to species 1 with typed numbers",
         instead="K of each liquid from its own dielectric (4 pi k / lambda, x2 for diffuse light), a function of its state"),
    dict(param="refractive index of each liquid (and of air in the voids)", derived="Liquid.dielectric (water Daimon-Masumura; linseed)",
         line="none in engine.py (the v2 optics model holds n per species in its own table)",
         now="no index: what surrounds a particle does not matter", instead="n_l per band from the description"),
    dict(param="gain (K < 0) and its debt", derived="laws.passivity, laws.gain_debt, laws.km_any",
         line="522 (km_layer: S clamped >= 1e-7, b = sqrt(max(a^2 - 1, 1e-14))), 542 (np.clip(Rb, 0, 1))",
         now="a negative K is silently mis-computed and clipped: no debt, no threshold",
         instead="the analytic continuation for a < 1 (laws.km_any), a lasing-threshold test, and an energy ledger "
                 "(laws.ENERGY_LEDGER) drawing on a declared reservoir"),
    dict(param="yield onset phi_y and tau0", derived="not derived (floc structure: not in the description yet)",
         line="177 (L['phi_y'] = 0.2, L['tau0'] = 0.05 global)", now="one yield law for all",
         instead="needs an interparticle-attraction parameter in the description (Hamaker constant, surface treatment)"),
]
