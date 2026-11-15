"""goo.classic.matter.mechanics: how a substance packs, thickens, diffuses, settles, flows and evaporates, derived from
its description. Every map is stated with its source; SI units unless said.

Particles
  packing      random close packing of a size distribution of spheres: phi_m = 0.634 + 0.0658 delta
               + 0.0398 S delta^2 (delta = polydispersity, S = skewness of the number distribution of
               diameters): Desmond & Weeks (2014), Phys. Rev. E 90, 022204, fitted for delta <= ~0.6 (clipped
               there). phi_m is also the critical pigment volume concentration (CPVC) in the binder-fills-the-
               voids sense. Hollow and coated spheres pack by their envelope.
  thickening   intrinsic viscosity [eta] = 5/2 for rigid spheres (Einstein 1906, 1911), whatever their inside;
               Krieger & Dougherty (1959): eta = eta_liquid (1 - phi/phi_m)^(-[eta] phi_m).
  diffusion    Stokes-Einstein: D = k_B T / (3 pi eta d).
  settling     Stokes: v = (rho_p - rho_f) g d^2 / (18 eta) (dilute; negative = creaming); hindered by
               (1 - phi)^4.65 (Richardson & Zaki 1954) in a loaded film.
Liquids
  viscosity    a number with its source (eta at 20-25 C).
  evaporation  diffusion-limited through a still-air boundary layer delta: J = D_v M p_sat (1 - RH) / (R T delta)
               (kg m^-2 s^-1; Fick's law with an ideal vapour); surface recession J / rho.
Shapes that are not spheres raise NotDerivable: goo does not guess their packing or [eta].

goo's engine has no physical clock. To compare, a tick is read from its water evaporation constant
(`evap` = 0.0006 volume units = 0.15 um per tick) against water at 25 C, RH 50 %, delta = 5 mm (0.058 um/s
by the map above): TICK_S ~ 2.6 s. This is a reading, not a calibration goo makes.
"""
from __future__ import annotations
import numpy as np
from goo.classic.matter.morphology import NotDerivable

KB = 1.380649e-23; RGAS = 8.314462618; G = 9.81
T0 = 298.15
CELL_M = 0.25e-3                                   # goo's cell, m
EVAP_UNITS = 0.0006                                 # goo's Canvas.L["evap"], volume units (0.25 mm) per tick


def phi_max(morph):
    if morph is None: raise NotDerivable("a liquid does not pack")
    if morph.kind not in ("sphere", "coated", "hollow"):
        raise NotDerivable(f"packing of {getattr(morph, 'name', morph.kind)!r} particles is not derived")
    dl = min(morph.size.polydispersity(), 0.6); S = morph.size.skewness() if dl > 0 else 0.0
    return float(min(0.634 + 0.0658 * dl + 0.0398 * S * dl * dl, 0.87))


def intrinsic_viscosity(morph):
    if morph is None: raise NotDerivable("a liquid has no intrinsic viscosity of its own")
    if morph.kind not in ("sphere", "coated", "hollow"):
        raise NotDerivable(f"[eta] of {getattr(morph, 'name', morph.kind)!r} particles is not derived")
    return 2.5


def krieger_exponent(morph): return intrinsic_viscosity(morph) * phi_max(morph)


def stokes_einstein(d_um, eta, T=T0): return KB * T / (3 * np.pi * eta * np.asarray(d_um) * 1e-6)


def stokes_settling(d_um, rho_p, rho_f, eta):
    """m/s, positive downward (rho in g/cm^3)."""
    return (rho_p - rho_f) * 1000.0 * G * (np.asarray(d_um) * 1e-6) ** 2 / (18 * eta)


def evaporation(p_sat, M, rho, D_v=2.5e-5, RH=0.5, T=T0, delta=5e-3):
    """Recession speed (m/s) of a free liquid surface (p_sat Pa, M kg/mol, rho g/cm^3)."""
    J = D_v * M * p_sat * (1 - RH) / (RGAS * T * delta)
    return J / (rho * 1000.0)


TICK_S = EVAP_UNITS * CELL_M / evaporation(3169.0, 0.018015, 0.997)   # ~2.6 s: see module docstring


def particle_mechanics(morph, density, liquid):
    """Every mechanical number derivable for a particle population in `liquid` (a goo.classic.matter.Liquid)."""
    d = morph.size.d_um
    rho_eff = density * morph.solid_fraction() + (1 - morph.solid_fraction()) * 0.0012   # void: air
    if morph.kind == "coated":
        rc = morph.ratio ** 3; rho_eff = density * rc + morph.shell_density * (1 - rc)
    out = dict(rho_envelope=rho_eff, d_um=d)
    try:
        out.update(phi_m=phi_max(morph), intrinsic_viscosity=intrinsic_viscosity(morph),
                   krieger_exponent=krieger_exponent(morph))
    except NotDerivable as e: out["not_derived"] = str(e)
    eta = liquid.viscosity
    out["D_m2_s"] = float(stokes_einstein(d, eta))
    out["v_settle_m_s"] = float(stokes_settling(d, rho_eff, liquid.density, eta))
    # in goo's units: diffusivity in cells^2 per tick, settling through a 0.1-unit (25 um) film, in ticks
    out["D_cells2_tick"] = out["D_m2_s"] * TICK_S / CELL_M ** 2
    out["settle_25um_ticks"] = (25e-6 / abs(out["v_settle_m_s"]) / TICK_S) if out["v_settle_m_s"] else np.inf
    return out
