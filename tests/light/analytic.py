"""Analytic implementations of goo.light.Optical, for testing light without real substances.

A test substance carries its own two-flux coefficients per unit volume fraction (k, s in 1/m; a number or
one value per band), an index n and an rms roughness sigma (m). These are test fixtures, not physics."""
import numpy as np
from goo.film import Bands, Light, Film, Ground, Stack
from goo.units import um, nm


class Sub:
    def __init__(self, name, k=0.0, s=0.0, n=1.5, sigma=0.0, rho=1000.0):
        self.name, self.k, self.s, self.n, self.sigma, self.rho = name, k, s, n, sigma, rho


def _per_band(v, B):
    v = np.asarray(v, np.float64)
    return np.broadcast_to(v, (B,))[:, None, None]


class Mixture:
    """K = sum phi_i k_i, S = sum phi_i s_i, host and top index sum phi_i n_i, roughness sum phi_i sigma_i.
    Covers a grey absorber (s = 0), a pure scatterer (k = 0) and a gain medium (k < 0). Calls are counted."""

    def __init__(self):
        self.calls = 0

    def _mix(self, film, attr, B):
        phi = film.phi
        return sum(_per_band(getattr(s, attr), B) * phi[i] for i, s in enumerate(film.substances))

    def optical(self, film, bands):
        self.calls += 1
        B = bands.B
        return self._mix(film, "k", B), self._mix(film, "s", B), self._mix(film, "n", B)

    def surface(self, film, bands):
        phi = film.phi
        sig = sum(s.sigma * phi[i] for i, s in enumerate(film.substances))
        return self._mix(film, "n", bands.B), sig

    def ground_optical(self, ground, bands):
        s, B = ground.solid, bands.B
        shp = (B,) + ground.z.shape
        return (np.broadcast_to(_per_band(s.k, B), shp), np.broadcast_to(_per_band(s.s, B), shp),
                np.broadcast_to(_per_band(s.n, B), shp))

    def ground_surface(self, ground, bands):
        s = ground.solid
        return np.broadcast_to(_per_band(s.n, bands.B), (bands.B,) + ground.z.shape), np.full(ground.z.shape, s.sigma)


class Particles(Mixture):
    """Particles in a host: the first substance is the host (index n_h), the others particles of index n_p and
    absorption k. Scattering per volume fraction s0 ((m^2 - 1)/(m^2 + 2))^2 with m = n_p / n_h (the
    Lorentz-Lorenz contrast; any law that vanishes at m = 1 would do). Host index = the host's."""

    def __init__(self, s0=1e5):
        super().__init__()
        self.s0 = s0

    def optical(self, film, bands):
        B, phi = bands.B, film.phi
        host = film.substances[0]
        nh = _per_band(host.n, B) * np.ones(film.shape)
        K = _per_band(host.k, B) * phi[0]
        S = np.zeros((B,) + film.shape)
        for i, p in enumerate(film.substances[1:], 1):
            m2 = (_per_band(p.n, B) / nh) ** 2
            S = S + self.s0 * ((m2 - 1) / (m2 + 2)) ** 2 * phi[i]
            K = K + _per_band(p.k, B) * phi[i]
        return K, S, nh

    def surface(self, film, bands):
        host = film.substances[0]
        return _per_band(host.n, bands.B) * np.ones(film.shape), np.full(film.shape, host.sigma)


def bands(B=4):
    return Bands(np.linspace(420, 680, B) * nm, np.full(B, 260 * nm / B))


def diffuse(b):
    return Light(b, np.ones(b.B), "diffuse")


def ground(H, W, solid, depth=np.inf):
    return Ground(np.zeros((H, W)), solid, 0.3, 1 * um, depth)


def random_stack(rng, H=3, W=4, nfilms=3, B=4, gain=False, index=True):
    """A random admissible stack (or with gain, if asked): each film one or two substances, random amounts
    (some cells empty), random coefficients over 0 .. 1e6 1/m, indices 1 .. 2.5, roughness 0 .. 0.3 um."""
    films = []
    for j in range(nfilms):
        subs = []
        for i in range(rng.integers(1, 3)):
            k = rng.uniform(0, 1, B) * 10 ** rng.uniform(1, 6)
            s = rng.uniform(0, 1, B) * 10 ** rng.uniform(1, 6)
            if gain and i == 0:
                k = -k * 0.2
            subs.append(Sub(f"f{j}s{i}", k, s, rng.uniform(1.0, 2.5) if index else 1.0, rng.uniform(0, 0.3) * um))
        q = rng.uniform(0, 30, (len(subs), H, W)) * um * (rng.random((len(subs), H, W)) > 0.2)
        films.append(Film(tuple(subs), q))
    solid = Sub("solid", rng.uniform(0, 1, B) * 1e4, rng.uniform(0.5, 1, B) * 1e5,
                rng.uniform(1.2, 2.0) if index else 1.0, rng.uniform(0, 0.5) * um)
    return Stack(ground(H, W, solid, rng.choice([np.inf, 300 * um])), films, dx=1e-3)
