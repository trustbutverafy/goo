"""Analytic implementations of goo.world.Constitutive for the world's tests (SPEC §4.4): a Newtonian liquid of
constant viscosity, a Bingham paste, an ideal (Flory-Huggins) mixture, one volatile. Numbers are physical
magnitudes chosen for the tests (`guessed`), not data about any material."""
import numpy as np
from goo.evidence import derive
from goo.units import mPa_s, mN, m, k_B, T_room
from goo.world import Mixing


class Sub:
    def __init__(self, name, rho):
        self.name, self.rho = name, rho


def water_like(name="water", rho=998.0):
    return Sub(name, rho)


def _d(v, name):
    return derive(v, "exact", "guessed", map=name)


class Analytic:
    """Constant coefficients, one per substance where it matters. A parameter given as None makes that function
    absent (the term is absent). kT_v: k_B T / v_i in J/m^3 per substance (ideal mixing); Lam: Stokes mobility
    m^2/(Pa s); k: exchange rate m/(Pa s); wet: (S, ell); sorb: (P per substance, kappa per substance)."""

    def __init__(self, substances, eta=1 * mPa_s, tau=None, gamma=72 * mN / m, kT_v=None, Lam=None, k=None,
                 wet=None, sorb=None):
        self.subs = tuple(substances)
        self.eta, self.tau_, self.gamma = eta, tau, gamma
        self.kT_v, self.Lam, self.k, self.wet, self.sorb = kT_v, Lam, k, wet, sorb
        for name, val in (("yield_stress", tau), ("surface_tension", gamma), ("mixing", kT_v), ("mobility", Lam),
                          ("exchange", k), ("wetting", wet), ("sorption", sorb)):
            if val is None:
                setattr(self, name, None)

    def _vec(self, v, film):
        return np.asarray([v[s.name] if isinstance(v, dict) else v for s in film.substances], np.float64)

    def viscosity(self, film):
        return _d(np.full(film.shape, float(self.eta)), "constant viscosity")

    def yield_stress(self, film):
        return _d(np.full(film.shape, float(self.tau_)), "constant yield stress")

    def surface_tension(self, film):
        return _d(np.full(film.shape, float(self.gamma)), "constant surface tension")

    def mixing(self, film, T):
        c = self._vec(self.kT_v, film)[:, None, None]
        q = film.q
        h = q.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = np.where(h > 0, q / h, 0.0)
            lnphi = np.where(q > 0, np.log(np.where(q > 0, phi, 1.0)), -np.inf)
            f = np.sum(np.where(q > 0, c * q * lnphi, 0.0), 0)
            cphi = np.sum(c * phi, 0)
            mu = c * (lnphi + 1.0) - cphi
            S = q.shape[0]
            eye = np.eye(S)[:, :, None, None]
            hess = (eye * (c / q)[:, None] - c[:, None] / h - c[None, :] / h + cphi / h)
        return _d(Mixing(f, mu, hess), "ideal Flory-Huggins mixing")

    def mobility(self, film):
        return _d(np.broadcast_to(self._vec(self.Lam, film)[:, None, None], film.q.shape).copy(), "constant mobility")

    def exchange(self, film, env):
        return _d(np.broadcast_to(self._vec(self.k, film)[:, None, None], film.q.shape).copy(), "constant exchange")

    def wetting(self, film, ground):
        S, l = self.wet
        return _d((np.full(film.shape, float(S)), np.full(film.shape, float(l))), "constant wetting")

    def sorption(self, film, ground):
        P, kap = self.sorb
        sh = film.q.shape
        return _d((np.broadcast_to(self._vec(P, film)[:, None, None], sh).copy(),
                   np.broadcast_to(self._vec(kap, film)[:, None, None], sh).copy()), "constant sorption")


def flat_ground(H, W, porosity=0.0, depth=0.0, z=None):
    from goo.film import Ground
    return Ground(np.zeros((H, W)) if z is None else z, Sub("gesso", 2300.0), porosity, 5e-6, depth)


KT_V_SMALL = k_B * T_room / (3e-29)        # a small molecule's k_B T / v, J/m^3
