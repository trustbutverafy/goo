"""The derived maps (SPEC §4.3): functions of a film (or a ground), vectorised over its cells, returning
goo.evidence.Derived, or Bounded where only bounds are known. The only window goo.world and goo.light have onto
matter. A map that returns several quantities returns a named tuple in the table's order, so it unpacks by position:
`K, S, n_host = optical(film, bands)`.

A film here is any object with `.substances` (a tuple of goo.substance.Substance), `.q` [S, H, W] (m), `.h` [H, W]
(m), `.phi` [S, H, W] and `.state` (a dict of [H, W] fields; `temperature` in K is read when present, else
goo.units.T_room). A ground: any object with `.solid` (a Substance), `.porosity`, `.depth` (m), `.substances` and
`.held` [S, H, W] (m held in its pores) and `.z` [H, W]. Bands: any object with `.centres`, `.widths` (m) and `.B`.
An environment: any object with `.temperature` (K), `.activity` (reservoir name -> activity: {"water": 0.5} for
50 % RH, {"oxygen": 1.0} for air) and `.boundary_layer` (m); `Atmosphere` is one.

Roles are read from descriptions, never from names or positions: a substance with a morphology is a particle
population, one without a continuous phase ("liquid", even when it is glassy, such as cured oil).

Volumes add (SPEC §3): phi = q / h. When the liquids cannot fill the space between packed particles the film holds
voids V = max(0, P (1 / phi_m - 1) - L) (m), with P the particles' and L the liquids' volume per area and phi_m the
packing fraction; in a ground the voids are the pore volume its held matter leaves empty. The host around particles
is then liquid and air in an unknown microstructure, and the maps that depend on it return bounds.

Domain. Every Derived's `in_domain` is false exactly where one of the map's stated conditions fails, and its `map`
text names each failed condition and how many points it covers ("optical.K; out of domain: eps of rutile not
described across bands 390, 410 nm (described 430-1530 nm) [36 of 114 points]; ..."). A Bounded carries the same
text. Empty cells (h = 0) hold nothing: a map returns a stated convention there and counts them in its domain.

Evidence: each output's evidence is the weakest of its map's and of the provenance of every field it read, from the
substances present (q > 0 somewhere); outside the map's domain it is at best `assumed` (goo.evidence.derive).
"""
from dataclasses import dataclass, field
from typing import NamedTuple
import numpy as np
from goo.units import T_room, R_gas, k_B, g_0
from goo.evidence import derive, Bounded, Derived, weakest
from goo.substance import response
from goo.substance import _freeenergy as _fe
from goo.substance.dielectric import h_planck
from goo.substance.morphology import SPHERICAL, envelope_rho

PHI_INDEPENDENT = 0.10        # particle volume fraction up to which independent scattering is taken to hold (assumed)
NU_E = 3.0e15                 # Hz: main electronic absorption frequency in the Tabor-Winterton Hamaker constant (assumed)
RZ_N = 4.65                   # Richardson-Zaki hindered-settling exponent, low Reynolds number (Richardson & Zaki 1954)
RE_MAX = 0.1                  # particle Reynolds number below which Stokes holds
DW_DELTA_MAX = 0.6            # polydispersity up to which Desmond & Weeks' packing fit holds
INTRINSIC_SPHERE = 2.5        # Einstein (1906, 1911): [eta] of rigid spheres, whatever their interior (proved)
D_LINE = 589.3e-9             # m: where a static (visible) index is read
HOST_ABSORPTION_MAX = 0.1     # 4 pi k_host d / lambda below which Mie in a non-absorbing host is taken to hold


@dataclass(frozen=True)
class Atmosphere:
    """A stand-in with goo.film.Environment's field names (T, gravity, activity, boundary_layer), for callers that
    have no Film module at hand; the maps read any object with .T, .activity and .boundary_layer."""
    T: float = T_room
    gravity: tuple = (0.0, 0.0, g_0)
    activity: dict = field(default_factory=dict)
    boundary_layer: float = 5e-3


def _env_T(env):
    return float(getattr(env, "T", T_room))


class MixingTerms(NamedTuple):
    """goo.world's Mixing: f [H,W] J/m^2 (convex part), mu [S,H,W] J/m^3, hess [S,S,H,W] J/m^4, and the concave
    part f_x [H,W], mu_x [S,H,W]."""
    f: np.ndarray
    mu: np.ndarray
    hess: np.ndarray
    f_x: np.ndarray
    mu_x: np.ndarray


class ExchangeRates(NamedTuple):
    k: np.ndarray             # [S,H,W] m/(Pa s): volume flux out per area = k (mu - mu_env)
    mu_env: np.ndarray        # [S] J/m^3: the reservoir's potential (0 where a substance has none)
    activity: np.ndarray      # [S,H,W] the film's activity of each substance


class Wetting(NamedTuple):
    S: np.ndarray             # [H,W] J/m^2, spreading coefficient <= 0
    ell: np.ndarray           # [H,W] m, wetting length > 0


class Sorption(NamedTuple):
    P: np.ndarray             # [S,H,W] Pa, capillary suction of the pores on each substance
    kappa: np.ndarray         # [S,H,W] m/(Pa s), rate: volume flux into the pores per area = kappa (mu - mu_pore)


class Optics(NamedTuple):
    K: object                 # [B, H, W] 1/m of h (or of a ground's depth)
    S: object                 # [B, H, W] 1/m
    host: object              # [B, H, W] real index around the particles


class Surface(NamedTuple):
    index: object             # [B, H, W] index of the top interface
    roughness: object         # [H, W] rms roughness, m


class Exchange(NamedTuple):
    activity: Derived         # [H, W] activity of the volatile in the film (pure liquid = 1)
    activity_env: float       # the environment reservoir's activity: the film's equilibrium activity
    rate: Derived             # [H, W] kg/(m^2 s) per unit activity difference: J = rate (activity - activity_env)
    p_sat: Derived            # [H, W] Pa at the film's temperature


class ReactionRate(NamedTuple):
    substance: str
    reaction: object          # the goo.substance.Reaction
    rate: Derived             # [H, W] 1/s: d(extent)/dt = rate * (mass of the substance)
    affinity: Derived         # [H, W] J per kg of extent: -dG - sum_i nu_i mu_i / rho_i - sum_res nu_res mu_res


# ------------------------------------------------------------------------------------------------------- helpers
class _Why:
    """The reasons a map's cells are out of its domain: (sentence, mask) pairs."""

    def __init__(self, shape, empty, mult=None):
        self.shape, self.items = shape, []
        self.empty = np.broadcast_to(empty, shape)
        self.mult = np.ones(shape) if mult is None else np.broadcast_to(mult, shape)

    def add(self, text, mask):
        m = np.broadcast_to(np.asarray(mask, bool), self.shape) & ~self.empty
        if m.any():
            self.items.append((text, m.copy()))

    def merge(self, other):
        for t, m in other.items:
            self.add(t, np.broadcast_to(m, self.shape) if m.ndim <= len(self.shape) else m.any(0))

    @property
    def domain(self):
        bad = np.zeros(self.shape, bool)
        for _, m in self.items:
            bad |= m
        return ~bad

    def text(self, name):
        if not self.items:
            return name
        n = int(self.mult.sum())
        return name + "; out of domain: " + "; ".join(f"{t} [{int(self.mult[m].sum())} of {n} points]"
                                                      for t, m in self.items)


def _nm(x):
    return ", ".join(f"{v * 1e9:.0f}" for v in np.atleast_1d(x))


class _F:
    """What every map reads of a film (or of a ground made to look like one), computed once."""

    def __init__(self, subs, q, h, T, voids=None, mult=None):
        self.subs = tuple(subs)
        self.q = np.asarray(q, float)
        self.h = np.asarray(h, float)
        self.phi = np.divide(self.q, self.h, out=np.zeros_like(self.q), where=self.h > 0)
        self.shape = self.q.shape[1:]
        self.T = np.broadcast_to(np.asarray(T, float), self.shape)
        self.part = np.array([s.morph is not None for s in self.subs], bool)
        self.present = np.array([bool(np.any(self.q[i] > 0)) for i in range(len(self.subs))], bool)
        self.P = self.q[self.part].sum(0) if self.part.any() else np.zeros(self.shape)
        self.L = self.q[~self.part].sum(0) if (~self.part).any() else np.zeros(self.shape)
        self.empty = self.h <= 0
        self.fixed_voids = voids
        self.mult = mult                                         # [H, W] how many cells each point stands for

    @classmethod
    def lite(cls, film, T=None):
        """What the mixing entry points read (substances, q, T, which substances are present), without the arrays the
        other maps need (phi, P, L): the same values of those fields as of() gives."""
        st = getattr(film, "state", {}) or {}
        if T is None:
            T = st.get("temperature", T_room)
        F = cls.__new__(cls)
        F.subs = tuple(film.substances)
        F.q = np.asarray(film.q, float)
        F.shape = F.q.shape[1:]
        F.T = np.broadcast_to(np.asarray(T, float), F.shape)
        F.part = np.array([s.morph is not None for s in F.subs], bool)
        S = len(F.subs)
        F.present = (F.q.reshape(S, -1) > 0).any(1) if F.q.size else np.zeros(S, bool)
        return F

    @classmethod
    def of(cls, film, T=None):
        st = getattr(film, "state", {}) or {}
        if T is None:
            T = st.get("temperature", T_room)
        q = np.asarray(film.q, float)
        return cls(film.substances, q, q.sum(0), T)

    def unique(self):
        """(compacted _F over the distinct cells, inverse index): cells with identical (q, T, voids) are solved once.
        The comparison is exact (np.unique on the values themselves), never a hash alone."""
        S = len(self.subs)
        cols = [self.q.reshape(S, -1), self.T.reshape(1, -1)]
        if self.fixed_voids is not None:
            cols.append(np.asarray(self.fixed_voids, float).reshape(1, -1))
        keys = np.concatenate(cols, 0).T
        u, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        n = u.shape[0]
        qu = u[:, :S].T.reshape(S, 1, n)
        V = None if self.fixed_voids is None else u[:, S + 1].reshape(1, n)
        hu = self.h.reshape(-1)[np.unique(inv.ravel(), return_index=True)[1]].reshape(1, n)
        return _F(self.subs, qu, hu, u[:, S].reshape(1, n), V, cnt.reshape(1, n).astype(float)), inv.ravel()

    @classmethod
    def ground(cls, g):
        """The ground as a film of depth `depth`: its solid at (1 - porosity) depth, its held matter, its voids."""
        held = np.asarray(g.held, float)
        shape = np.shape(g.z)
        solid = np.full((1,) + shape, (1 - g.porosity) * g.depth)
        q = np.concatenate([solid, held.reshape((-1,) + shape)], 0) if held.size else solid
        voids = np.maximum(g.porosity * g.depth - held.sum(0) if held.size else g.porosity * g.depth, 0.0)
        voids = np.broadcast_to(voids, shape)
        st = getattr(g, "state", {}) or {}
        return cls((g.solid,) + tuple(g.substances), q, np.full(shape, float(g.depth)),
                   st.get("temperature", T_room), voids)

    def idx(self, particles):
        return [i for i in range(len(self.subs)) if self.part[i] == particles]

    def ev(self, keys_of, only=None):
        out = [s.evidence(*keys_of(s)) for i, s in enumerate(self.subs)
               if self.present[i] and (only is None or i in only)]
        return weakest(*out) if out else "exact"

    def y(self):
        li = self.idx(False)
        if not li:
            return li, np.zeros((0,) + self.shape)
        return li, np.divide(self.q[li], self.L, out=np.zeros((len(li),) + self.shape), where=self.L > 0)

    def mixed_liquids(self):
        li = self.idx(False)
        return np.sum(self.q[li] > 0, axis=0) >= 2 if li else np.zeros(self.shape, bool)

    def why(self, lead=()):
        return _Why(tuple(lead) + self.shape, self.empty, self.mult)

    def shapes(self, why):
        for i in self.idx(True):
            if self.subs[i].morph.kind not in SPHERICAL:
                why.add(f"{self.subs[i].name}'s particles are of shape {self.subs[i].morph.kind!r}, for which goo has "
                        "no law", self.q[i] > 0)


def _miscible(si, sj, T):
    """Flory-Huggins critical test for one pair of liquids with the interaction of goo.substance._freeenergy
    (Hansen where both give their parts, else Hildebrand): B < (k T / 2)(1/sqrt(v_i) + 1/sqrt(v_j))^2."""
    B, rule = _fe.pair_interaction(si, sj)
    if rule is None:
        return np.ones(np.shape(T), bool)
    vi, vj = si.molecular_volume, sj.molecular_volume
    return B < 0.5 * k_B * T * (1 / np.sqrt(vi) + 1 / np.sqrt(vj)) ** 2


def _liquid_phase(F, why):
    """Add the reasons a cell's liquids are not one mixed phase."""
    li = F.idx(False)
    for a in range(len(li)):
        for b in range(a + 1, len(li)):
            i, j = li[a], li[b]
            both = (F.q[i] > 0) & (F.q[j] > 0)
            if both.any():
                why.add(f"{F.subs[i].name} and {F.subs[j].name} separate by the regular solution: not one liquid phase",
                        both & ~_miscible(F.subs[i], F.subs[j], F.T))


def _packing(F):
    """phi_m [H, W], its _Why, by Desmond & Weeks (2014) on the number distribution of every spherical population
    present in the cell; the monodisperse value 0.634 where there are no particles."""
    why = F.why(); F.shapes(why)
    pi_ = [i for i in F.idx(True) if F.subs[i].morph.kind in SPHERICAL]
    if not pi_:
        return np.full(F.shape, 0.634), why
    m = [np.zeros(F.shape) for _ in range(4)]
    for i in pi_:
        z = F.subs[i].morph.size
        n = F.q[i] / z.particle_volume()
        for k in range(4):
            m[k] = m[k] + n * z.number_moment(k)
    has = m[0] > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        m1, m2, m3 = (np.where(has, m[k] / np.where(has, m[0], 1), 0.0) for k in (1, 2, 3))
        sd = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
        delta = np.where(has, sd / np.where(has, m1, 1), 0.0)
        skew = np.where(has & (sd > 0), (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / np.where(sd > 0, sd, 1) ** 3, 0.0)
    dl = np.minimum(delta, DW_DELTA_MAX)
    phim = np.minimum(0.634 + 0.0658 * dl + 0.0398 * skew * dl * dl, 0.87)
    why.add(f"polydispersity above {DW_DELTA_MAX} (Desmond & Weeks' fit; clipped there)", delta > DW_DELTA_MAX + 1e-12)
    return np.where(has, phim, 0.634), why


def _voids(F, phim):
    if F.fixed_voids is not None:
        return F.fixed_voids
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.maximum(F.P * (1.0 / phim - 1.0) - F.L, 0.0)


def _suspension_phi(F):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(F.P + F.L > 0, F.P / np.where(F.P + F.L > 0, F.P + F.L, 1), 0.0)


# ------------------------------------------------------------------------------- per-substance optical constants
_LIQ = {}


def _liquid_bands(s, bands):
    """Band means of n and k of a continuous phase, the band centres' wavelengths, and per-band in_range."""
    key = (s, response.bands_key(bands))
    if key not in _LIQ:
        lam, own = response.band_samples(bands)
        m = s.eps.nk(lam)
        _LIQ[key] = (response.band_mean(m.real, own, bands.B), response.band_mean(m.imag, own, bands.B),
                     response.band_mean(lam, own, bands.B),
                     np.array([s.eps.in_range(lam)[own == b].all() for b in range(bands.B)]))
    return _LIQ[key]


def _range_text(s, d=None):
    d = d or s.eps
    if hasattr(d, "lo"):
        return f"described {d.lo * 1e9:.0f}-{d.hi * 1e9:.0f} nm"
    return "not described there"


def _eps_why(F, bands, why, inr_of):
    """Add, per present substance, the bands its eps does not describe."""
    for i, s in enumerate(F.subs):
        if not F.present[i]:
            continue
        inr = inr_of(s)
        if not inr.all():
            bad = ~inr
            why.add(f"eps of {s.name} not described across bands {_nm(np.asarray(bands.centres)[bad])} nm "
                    f"({_range_text(s)})", bad[:, None, None] & (F.q[i] > 0)[None])


def _host_eps(F, bands):
    """Lorentz-Lorenz mixed eps of the liquids [B, H, W] (real part), their absorption sum phi_l 4 pi k_l / lam
    [B, H, W] (1/m), and the volume-weighted k of the liquids [B, H, W]."""
    B = bands.B
    li, y = F.y()
    ll = np.zeros((B,) + F.shape); alpha = np.zeros((B,) + F.shape); kmix = np.zeros((B,) + F.shape)
    for a, i in enumerate(li):
        n, k, lam, _ = _liquid_bands(F.subs[i], bands)
        e = n ** 2 - k ** 2
        ll += y[a][None] * ((e - 1) / (e + 2))[:, None, None]
        alpha += F.phi[i][None] * (4 * np.pi * k / lam)[:, None, None]
        kmix += y[a][None] * k[:, None, None]
    eps_l = (1 + 2 * ll) / (1 - ll)
    return np.where(F.L > 0, eps_l, 1.0), alpha, kmix


def _wiener(eps_l, f_l):
    """Wiener bounds on the eps of a liquid-air mixture with liquid volume fraction f_l (any microstructure); exact
    (lo = hi) for a single phase."""
    lo, hi = 1.0 / (f_l / eps_l + (1 - f_l)), f_l * eps_l + (1 - f_l)
    one = (f_l >= 1) | (f_l <= 0)
    return np.where(one, hi, np.minimum(lo, hi)), hi


# ------------------------------------------------------------------------------------------------------- optical
def optical(film, bands):
    """(K, S, host): K, S [B, H, W] in 1/m of film thickness h, and the host index [B, H, W].

    Model: K = sum_p phi_p K_p(n_h) + sum_l phi_l 2 (4 pi k_l / lambda); S = sum_p phi_p S_p(n_h), with K_p, S_p the
    two-flux coefficients per unit particle volume of goo.substance.response (Mie in a non-absorbing host,
    independent scattering, K = 2 mu_a, S = 3/4 mu_s (1 - g), band means), precomputed per substance and band on
    the host-index grid and spline-interpolated. The host is the liquids mixed by Lorentz-Lorenz; where the film
    holds voids it is liquid and air in an unknown microstructure, bounded by the Wiener bounds, and then host, K and S
    are Bounded (K and S taken at the lower bound, the mid-point and the upper bound of the host index; their min
    and max). K h and S h are the optical depths; phi = q / h.

    Domain, each a named reason: every present substance's eps described across the band; host index inside the
    table's grid (1.0-1.8); particle volume fraction (voids included) <= PHI_INDEPENDENT (independent scattering);
    spherical particles; polydispersity within the packing fit; host absorption 4 pi k_l d / lambda <
    HOST_ABSORPTION_MAX; one liquid phase. Empty cells: K = S = 0, host 1.
    Exact limits: no particles gives S = 0 exactly; a particle in a host of its own index scatters nothing (Mie,
    tested by `optical_exact`); K, S are linear in each particle's volume (independent scattering)."""
    F0 = _F.of(film)
    F, inv = F0.unique()
    phim, pw = _packing(F)
    return _expand(_optical(F, bands, _voids(F, phim), pw, False, "optical"), inv, F0.shape)


def _expand(out, inv, shape):
    """Scatter a map's answer on distinct cells [..., 1, n] back to the film's cells [..., H, W]."""
    def e(a):
        a = np.asarray(a)
        return a[..., 0, inv].reshape(a.shape[:-2] + tuple(shape)) if a.ndim >= 2 else a

    def one(x):
        if isinstance(x, Bounded):
            return Bounded(e(x.lo), e(x.hi), x.evidence, x.map)
        dom = x.in_domain if np.ndim(x.in_domain) == 0 else e(x.in_domain)
        return Derived(e(x.value), x.evidence, dom, x.map)
    return type(out)(*(one(x) for x in out))


def optical_exact(film, bands):
    """`optical` with Mie computed directly at each cell's host index instead of interpolated (slow; for checks)."""
    F = _F.of(film)
    phim, pw = _packing(F)
    return _optical(F, bands, _voids(F, phim), pw, True, "optical")


def ground_optical(ground, bands):
    """(K, S, host) of a porous ground, [B, H, W], K and S in 1/m of its depth (K depth is its optical depth).

    The same physics as `optical`, on the ground's own composition: its solid (a particle population: its
    morphology) at volume fraction 1 - porosity of the depth, the matter its pores hold (`held`, any substances) and
    air in the rest of the pores. So a liquid soaked into the pores lowers the solid's scattering by index matching,
    and a ground differs from another only through its solid's description. With air in the pores and liquid
    present the host is Bounded (Wiener). Domain as `optical`'s (a ground is far above the independent-scattering
    bound, so its values are at best assumed), plus: the solid has a morphology; the pores do not overflow."""
    F0 = _F.ground(ground)
    F, inv = F0.unique()
    return _expand(_optical(F, bands, F.fixed_voids, _ground_why(F, ground, "the scattering of its pores is not derived"),
                            False, "ground_optical"), inv, F0.shape)


def _ground_why(F, ground, what):
    why = F.why()
    if ground.solid.morph is None:
        why.add(f"the ground's solid {ground.solid.name} has no morphology: {what}", np.ones(F.shape, bool))
    held = F.q[1:].sum(0) if F.q.shape[0] > 1 else np.zeros(F.shape)
    why.add("its pores hold more than their volume", held > ground.porosity * ground.depth * (1 + 1e-12))
    return why


def _optical(F, bands, V, pre, exact, name):
    B = bands.B; shp = (B,) + F.shape
    why = F.why((B,)); why.merge(pre)
    eps_l, alpha, kmix = _host_eps(F, bands)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_l = np.where(F.L + V > 0, F.L / np.where(F.L + V > 0, F.L + V, 1), 1.0)
    lo, hi = _wiener(eps_l, f_l[None])
    n_lo, n_hi = np.sqrt(lo), np.sqrt(hi)
    hosts = [n_lo] if not np.any(n_hi > n_lo + 1e-12) else [n_lo, 0.5 * (n_lo + n_hi), n_hi]
    Ks, Ss = [], []
    pi_ = F.idx(True)
    outside = np.zeros(shp, bool)
    tables = {}
    for i in pi_:
        s = F.subs[i]
        if F.present[i] and s.morph.kind in SPHERICAL:
            tables[i] = response.table(s, bands)
    for n_h in hosts:
        K = 2 * alpha.copy(); S = np.zeros(shp)
        for i, t in tables.items():
            s = F.subs[i]
            if exact:
                Kp, Sp, ins = _exact_at(s, bands, n_h)
            else:
                Kp, Sp, ins = t(n_h)
            if t.passive:
                Kp = np.maximum(Kp, 0.0)                         # a theorem the spline may undershoot
            Sp = np.maximum(Sp, 0.0)
            K += F.phi[i][None] * Kp; S += F.phi[i][None] * Sp
            outside |= (F.q[i] > 0)[None] & ~ins
        Ks.append(K); Ss.append(S)
    _eps_why(F, bands, why, lambda s: tables[F.subs.index(s)].in_range if F.subs.index(s) in tables
             else (_liquid_bands(s, bands)[3] if s.morph is None else np.ones(B, bool)))
    why.add(f"host index outside the precomputed {response.HOST_GRID[0]:.2f}-{response.HOST_GRID[-1]:.2f} "
            "(clipped to it)", outside)
    with np.errstate(divide="ignore", invalid="ignore"):
        tot = F.q.sum(0) + V
        phi_p = np.where(tot > 0, F.P / np.where(tot > 0, tot, 1), 0.0)
    why.add(f"particle volume fraction above {PHI_INDEPENDENT} (independent scattering)", (phi_p > PHI_INDEPENDENT)[None])
    F.shapes(why)
    _liquid_phase(F, why)
    dmax = np.zeros(F.shape)                                     # per cell: the largest particles it holds
    for i in tables:
        dmax = np.where(F.q[i] > 0, np.maximum(dmax, F.subs[i].morph.size.d), dmax)
    lamc = np.asarray(bands.centres)[:, None, None]
    why.add(f"absorbing host: 4 pi k d / lambda >= {HOST_ABSORPTION_MAX} (Mie assumes a clear host)",
            4 * np.pi * kmix * dmax / lamc >= HOST_ABSORPTION_MAX)
    dom = why.domain
    ev = weakest("computed", F.ev(lambda s: ("eps", "morph") if s.morph is not None else ("eps",)))
    if F.mixed_liquids().any():
        ev = weakest(ev, "assumed")                              # Lorentz-Lorenz mixing of liquids
    for A in Ks + Ss:
        A[:, F.empty] = 0.0
    n_lo = np.where(F.empty[None], 1.0, n_lo); n_hi = np.where(F.empty[None], 1.0, n_hi)
    if len(hosts) == 1:
        return Optics(derive(Ks[0], ev, in_domain=dom, map=why.text(f"{name}.K")),
                      derive(Ss[0], ev, in_domain=dom, map=why.text(f"{name}.S")),
                      derive(n_lo, ev, in_domain=dom, map=why.text(f"{name}.host")))
    evb = weakest(ev, "assumed")
    note = "; bounded: liquid and air in an unknown microstructure (Wiener bounds of the host)"
    return Optics(Bounded(np.minimum.reduce(Ks), np.maximum.reduce(Ks), evb, why.text(f"{name}.K") + note),
                  Bounded(np.minimum.reduce(Ss), np.maximum.reduce(Ss), evb, why.text(f"{name}.S") + note),
                  Bounded(n_lo, n_hi, evb, why.text(f"{name}.host") + note))


def _exact_at(s, bands, n_h):
    lam, own = response.band_samples(bands)
    K = np.zeros(n_h.shape); S = np.zeros(n_h.shape)
    for b in range(bands.B):
        u, inv = np.unique(n_h[b], return_inverse=True)
        Kb, Sb = response.two_flux(s.eps, s.morph, lam[own == b], u)
        K[b] = Kb.mean(1)[inv].reshape(n_h[b].shape); S[b] = Sb.mean(1)[inv].reshape(n_h[b].shape)
    ins = (n_h >= response.HOST_GRID[0]) & (n_h <= response.HOST_GRID[-1])
    return K, S, ins


# ------------------------------------------------------------------------------------------------------- surface
def _monolayer_rms():
    """rms height of the top of a hexagonal monolayer of equal spheres on a plane whose interstices hold liquid up to
    a flat level, against the empty fraction delta of the interstices; per unit sphere diameter."""
    nx, ny = 60, 104
    xs = (np.arange(nx) + 0.5) / nx; ys = (np.arange(ny) + 0.5) / ny * np.sqrt(3)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    rho2 = np.full(X.shape, np.inf)
    for cx, cy in [(0, 0), (0.5, np.sqrt(3) / 2)]:
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                rho2 = np.minimum(rho2, (X - cx - ox) ** 2 + (Y - cy - oy * np.sqrt(3)) ** 2)
    inside = rho2 < 0.25
    c = np.sqrt(np.maximum(0.25 - rho2, 0.0))
    zlo = np.where(inside, 0.5 - c, 1.0); zhi = np.where(inside, 0.5 + c, 0.0)
    vol, rms = [], []
    for l in np.linspace(0.0, 1.0, 201):
        solid = np.where(inside, np.clip(np.minimum(zhi, l) - np.minimum(zlo, l), 0, None), 0.0)
        vol.append(np.mean(l - solid)); rms.append(np.std(np.maximum(zhi, l)))
    vol = np.array(vol); delta = 1 - vol / vol[-1]
    o = np.argsort(delta)
    return delta[o], np.array(rms)[o], float(vol[-1])


_DELTA, _RMS, _INTERSTICE = _monolayer_rms()      # _INTERSTICE: interstitial volume of the monolayer per area / d


def surface(film, bands):
    """(index, roughness): index of the top interface [B, H, W] and rms roughness [H, W] (m).

    Model: where the liquids fill the packed particles' interstices (no voids) the top is the liquid surface: its
    index is the liquids' (Lorentz-Lorenz), roughness 0 on this rung (relief belongs to the world). Where they do not,
    the liquid recedes from the top: the top monolayer of particles (Sauter diameter d32 of the cell's mixture) is
    empty to the fraction delta = min(1, V / (0.395 d32)) of its interstices, its roughness is that of a hexagonal
    monolayer of spheres standing out of a flat liquid level (computed geometry, `_monolayer_rms`), and the top's
    index is Bounded by the Wiener bounds of the top monolayer's composition (particles at phi_m, liquid, air),
    with each particle's eps taken as its real part at the band (>= 1). Empty cells: index 1, roughness 0.
    Domain: spherical particles; one liquid phase; the liquids' eps described. Evidence: at best assumed."""
    F0 = _F.of(film)
    F, inv = F0.unique()
    phim, pw = _packing(F)
    V = _voids(F, phim)
    return _expand(_surface(F, bands, V, phim, liquid_on_top=(V <= 0), pre=pw, name="surface"), inv, F0.shape)


def ground_surface(ground, bands):
    """(index, roughness) of a ground's top, [B, H, W] and [H, W] (m): `surface`'s model with the solid at volume
    fraction 1 - porosity and the pores' contents. Pores never overflow onto the top, so the top is always a
    composite of solid, held matter and air: its index is Bounded (Wiener bounds), its roughness that of the top
    monolayer of solid particles emptied to the fraction its pores' voids leave (0 when the pores are full). The
    ground's relief z belongs to the world. Domain: the solid has a spherical morphology; one liquid phase."""
    F0 = _F.ground(ground)
    F, inv = F0.unique()
    return _expand(_surface(F, bands, F.fixed_voids, np.full(F.shape, 1.0 - ground.porosity),
                            liquid_on_top=np.zeros(F.shape, bool),
                            pre=_ground_why(F, ground, "no grain to make a roughness"), name="ground_surface"),
                   inv, F0.shape)


def _surface(F, bands, V, fsolid, liquid_on_top, pre, name):
    B = bands.B
    why = F.why((B,)); why.merge(pre)
    rwhy = F.why(); rwhy.merge(pre)
    eps_l = _host_eps(F, bands)[0]
    pi_ = [i for i in F.idx(True) if F.subs[i].morph.kind in SPHERICAL]
    n3 = np.zeros(F.shape); n2 = np.zeros(F.shape); epsp = np.zeros((B,) + F.shape)
    lam, own = response.band_samples(bands)
    for i in pi_:
        z = F.subs[i].morph.size
        n = F.q[i] / z.particle_volume()
        n3 += n * z.number_moment(3); n2 += n * z.number_moment(2)
        e = response.band_mean(np.real(F.subs[i].eps.eps(lam)), own, B)
        epsp += F.q[i][None] * np.maximum(e, 1.0)[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        d32 = np.where(n2 > 0, n3 / np.where(n2 > 0, n2, 1), 0.0)
        epsp = np.where(F.P > 0, epsp / np.where(F.P > 0, F.P, 1), 1.0)
        delta = np.where(d32 > 0, np.minimum(1.0, V / np.where(d32 > 0, _INTERSTICE * d32, 1)), 0.0)
    rough = np.where(V > 0, np.interp(delta, _DELTA, _RMS) * d32, 0.0)
    fp = np.where(F.P > 0, fsolid, 0.0); rest = 1 - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        fill = np.where(F.L + V > 0, F.L / np.where(F.L + V > 0, F.L + V, 1), 0.0)
    fl = rest * np.where(liquid_on_top, 1.0, np.where(V > 0, 1 - delta, fill)); fa = rest - fl
    hi = fp * epsp + fl * eps_l + fa
    lo = 1.0 / (fp / epsp + fl / eps_l + fa)
    top_l = np.where(F.L > 0, np.sqrt(eps_l), 1.0)
    idx_lo = np.where(liquid_on_top[None], top_l, np.sqrt(lo)); idx_hi = np.where(liquid_on_top[None], top_l, np.sqrt(hi))
    idx_lo = np.where(F.empty[None], 1.0, idx_lo); idx_hi = np.where(F.empty[None], 1.0, idx_hi)
    F.shapes(why); F.shapes(rwhy)
    _liquid_phase(F, why); _liquid_phase(F, rwhy)
    _eps_why(F, bands, why, lambda s: _liquid_bands(s, bands)[3] if s.morph is None else np.ones(B, bool))
    ev = weakest("assumed", F.ev(lambda s: ("eps", "morph") if s.morph is not None else ("eps",)))
    R = derive(rough, ev, in_domain=rwhy.domain, map=rwhy.text(f"{name}.roughness"))
    if np.all(idx_lo == idx_hi):
        ev_i = weakest("computed", F.ev(lambda s: ("eps",), only=F.idx(False)))
        if F.mixed_liquids().any():
            ev_i = weakest(ev_i, "assumed")
        return Surface(derive(idx_lo, ev_i, in_domain=why.domain, map=why.text(f"{name}.index")), R)
    return Surface(Bounded(np.minimum(idx_lo, idx_hi), np.maximum(idx_lo, idx_hi), ev,
                           why.text(f"{name}.index") + "; bounded: Wiener bounds of the top's composition"), R)


# ------------------------------------------------------------------------------------------- mechanics and mixing
def density(film):
    """rho [H, W] kg/m^3 = sum_s phi_s rho_s: volumes add (SPEC §3, evidence assumed). rho h is the mass per area
    exactly. Empty cells: 0. Voids, where present, are not counted (they are not part of h). Always in domain."""
    F = _F.of(film)
    rho = np.array([s.rho for s in F.subs], float)
    val = np.tensordot(rho, F.phi, 1) if len(F.subs) else np.zeros(F.shape)
    return derive(val, "assumed", F.ev(lambda s: ("rho",)), map="density")


def packing(film):
    """phi_m [H, W]: random close packing of the cell's particles, by Desmond & Weeks (2014, Phys. Rev. E 90,
    022204): 0.634 + 0.0658 delta + 0.0398 S delta^2, delta and S the polydispersity and skewness of the number
    distribution of diameters of every spherical population present (their mixture). Domain: delta <= 0.6 (clipped
    there beyond), spherical particles. Cells without particles: 0.634 (the monodisperse limit)."""
    F = _F.of(film)
    phim, why = _packing(F)
    return derive(phim, "computed", F.ev(lambda s: ("morph",), only=F.idx(True)), in_domain=why.domain,
                  map=why.text("packing"))


def _eta_liquids(F, why):
    """Arrhenius log-mixing of the liquids' viscosities at the cell temperature, [H, W]; reasons added to why."""
    li, y = F.y()
    ln = np.zeros(F.shape); lin = np.zeros(F.shape); neg = np.zeros(F.shape, bool)
    for a, i in enumerate(li):
        s = F.subs[i]; here = F.q[i] > 0
        if s.rheo.eta is None:
            why.add(f"{s.name} has no viscosity", here)
            continue
        fac = np.ones(F.shape)
        if s.rheo.E_eta is not None:
            fac = np.exp(s.rheo.E_eta / R_gas * (1 / F.T - 1 / T_room))
        else:
            why.add(f"{s.name} has no E_eta and the cell is not at T_ref", here & (np.abs(F.T - T_room) > 1e-9))
        e = s.rheo.eta * fac
        lin += y[a] * e
        if s.rheo.eta > 0:
            ln += y[a] * np.log(e)
        else:
            neg |= here
    why.add("a liquid with eta <= 0 (impossible): mixed linearly", neg)
    # vitrification: a binder whose solvent (every other liquid) is at or below its glass composition s_g is a glass;
    # above it the log-mixed viscosity carries the free-volume factor exp(y_b s_g / (s - s_g)) (Doolittle's form
    # with the free volume taken proportional to the solvent in excess of the glass composition: negligible in the
    # wet range, divergent at s_g)
    glass = np.zeros(F.shape, bool)
    for a, i in enumerate(li):
        sb = F.subs[i]
        if sb.rheo.glass is None:
            continue
        here = F.q[i] > 0
        sv = 1.0 - y[a]
        gl = here & (sv <= sb.rheo.glass)
        with np.errstate(divide="ignore", invalid="ignore"):
            ln = ln + np.where(here & ~gl, y[a] * sb.rheo.glass / np.where(gl, 1.0, sv - sb.rheo.glass), 0.0)
        glass |= gl
    why.add("the liquid is a glass (a binder's solvent at or below its glass composition): it does not flow", glass)
    _liquid_phase(F, why)
    return np.where(F.L > 0, np.where(glass, np.inf, np.where(neg, lin, np.exp(ln))), np.inf)


def viscosity(film):
    """eta [H, W] Pa s. Liquids: Arrhenius log-mixing eta_l = exp(sum_l y_l ln eta_l(T)) over their volume fractions
    y_l among the liquids (assumed when two or more are present; a liquid with eta <= 0, an impossible one, is mixed
    linearly), each eta_l(T) = eta_l exp(E_eta / R (1/T - 1/T_ref)). Particles: Krieger & Dougherty (1959)
    eta = eta_l (1 - phi/phi_m)^(-[eta] phi_m), phi = P / (P + L) the particles' fraction of the suspension, phi_m from
    `packing`, [eta] = 5/2 (Einstein, rigid spheres). Dilute limit: eta_l (1 + 5/2 phi).
    Vitrification: a binder b with a glass composition s_g (rheo.glass) whose solvent fraction s_b = 1 - y_b (every
    other liquid) is at or below s_g makes the liquid a glass: +inf (it no longer flows; goo.world's settle() retires
    it). Above, eta_l carries exp(y_b s_g / (s_b - s_g)) (Doolittle's free-volume form, the free volume taken
    proportional to the solvent in excess of the glass composition, coefficient 1: assumed): negligible when wet,
    divergent at s_g, monotone in s_b.
    Domain, each a named reason: phi < phi_m (beyond: inf, jammed); not a glass (else inf); a liquid present (a
    powder: inf); every liquid with a viscosity; miscible liquids; E_eta given where T != T_ref; the packing's.
    Empty cells: inf, in domain."""
    F = _F.of(film)
    phim, why = _packing(F)
    eta_l = _eta_liquids(F, why)
    phi = _suspension_phi(F)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.clip(1 - phi / phim, 0.0, None)
        eta = np.where(r > 0, eta_l * r ** (-INTRINSIC_SPHERE * phim), np.inf)
    eta = np.where(F.P > 0, eta, eta_l)
    why.add("no liquid: a powder does not flow", (F.L <= 0) & (F.P > 0))
    why.add("particles at or above their packing fraction: jammed", (phi >= phim) & (F.P > 0) & (F.L > 0))
    ev = weakest("computed", F.ev(lambda s: ("rheo.eta", "rheo.E_eta"), only=F.idx(False)),
                 F.ev(lambda s: ("morph",), only=F.idx(True)))
    if F.mixed_liquids().any():
        ev = weakest(ev, "assumed")
    return derive(eta, ev, in_domain=why.domain, map=why.text("viscosity"))


def _hamaker_n(s, lam=D_LINE):
    return float(np.mean([np.real(m[0]) for _, m in s.eps.parts(np.array([lam]))]))


class _OneBand:
    """The sodium D line as a band, for maps that read a static index."""
    centres = np.array([D_LINE]); widths = np.array([1e-9]); B = 1


def _yield(F):
    phim, why = _packing(F)
    n3 = np.sqrt(_host_eps(F, _OneBand())[0][0])
    phi = _suspension_phi(F)
    m_lo = np.zeros(F.shape); m_hi = np.zeros(F.shape); phi0 = np.zeros(F.shape)
    for i in F.idx(True):
        s = F.subs[i]; here = F.q[i] > 0
        if s.morph.kind not in SPHERICAL or s.rheo.separation is None:
            continue
        x = np.where(F.P > 0, F.q[i] / np.where(F.P > 0, F.P, 1), 0.0)
        n1 = _hamaker_n(s)
        A = 3 * h_planck * NU_E * (n1 ** 2 - n3 ** 2) ** 2 / (16 * np.sqrt(2) * (n1 ** 2 + n3 ** 2) ** 1.5)
        d32 = s.morph.size.sauter(); H = s.rheo.separation
        m_lo += x * A / (12 * H ** 2 * d32)
        m_hi += x * (A + 0.75 * k_B * F.T) / (12 * H ** 2 * d32)
        if s.rheo.phi_percolation is None:
            why.add(f"{s.name} attracts but has no phi_percolation", here)
        else:
            phi0 += x * s.rheo.phi_percolation
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(phi > phi0, phi * (phi - phi0) ** 2 / (phim * np.maximum(phim - phi, 0.0)), 0.0)
    G = np.where(phi >= phim, np.inf, G)
    G = np.where(F.P > 0, G, 0.0)
    why.add("no liquid: a powder's cohesion is not modelled", (F.L <= 0) & (F.P > 0))
    why.add("voids in the film: capillary bridges are not modelled", (_voids(F, phim) > 0) & (F.P > 0) & (F.L > 0))
    why.add("particles at or above their packing fraction: jammed", (phi >= phim) & (F.P > 0) & (F.L > 0))
    lo = np.where(G == np.inf, np.inf, np.nan_to_num(G * m_lo)); hi = np.where(G == np.inf, np.inf, np.nan_to_num(G * m_hi))
    return lo, hi, why


def yield_stress(film):
    """tau_y [H, W] Pa, Bounded. Model (the YODEL form, Flatt & Bowen 2006, J. Am. Ceram. Soc. 89, 1244, with its
    order-one prefactor taken as 1): tau_y = G(phi) sum_s x_s A_s / (12 H_s^2 d32_s), G = phi (phi - phi_0)^2 /
    (phi_m (phi_m - phi)) for phi_0 < phi < phi_m, 0 below phi_0; x_s the volume fraction of population s among the
    particles, phi = P / (P + L), phi_0 = sum_s x_s phi_percolation_s, H_s = rheo.separation, d32 the Sauter
    diameter. A_s by Tabor-Winterton from the indices at 589 nm: A = 3 h nu_e (n1^2 - n3^2)^2 / (16 sqrt(2)
    (n1^2 + n3^2)^(3/2)), nu_e = NU_E (assumed); its zero-frequency part (at most 3 kT / 4, unknown without a static
    permittivity) is what the bounds carry: lo without it, hi with it. A population without `separation` is taken as
    non-attracting (hard spheres): it adds no yield. Cells without particles: 0.
    Domain (a Bounded has no in_domain: `yield_stress_domain` gives it, and the Bounded's map text names the reasons):
    phi < phi_m (beyond: inf, jammed), a liquid present, no voids, every attracting population with
    phi_percolation, spherical particles, the packing's."""
    F = _F.of(film)
    lo, hi, why = _yield(F)
    ev = weakest("assumed", F.ev(lambda s: ("eps", "morph", "rheo.separation", "rheo.phi_percolation"),
                                 only=F.idx(True)), F.ev(lambda s: ("eps",), only=F.idx(False)))
    return Bounded(lo, hi, ev, why.text("yield_stress") + "; bounded: the Hamaker constant's zero-frequency part")


def yield_stress_domain(film):
    """in_domain [H, W] of `yield_stress`; its reasons are in the Bounded's map text."""
    return _yield(_F.of(film))[2].domain


def diffusivity(film):
    """D [S, H, W] m^2/s. Particles: Stokes-Einstein in the film's viscosity (SPEC §4.3), D = k T E_n[1/d] /
    (3 pi eta) with E_n the number mean over the size distribution and eta from `viscosity` (a generalised
    Stokes-Einstein: long-time self-diffusion in the suspension, assumed). Molecular substances: Stokes-Einstein in
    the liquids' viscosity with the radius of a sphere of the molecular volume, D = k T / (6 pi eta_l r) (out of
    domain: Stokes-Einstein for molecules is good to a factor of order 2); macromolecular phases (no M): 0.
    Where the viscosity is infinite, 0. Domain: the viscosity's, and spherical particles."""
    F = _F.of(film)
    eta = viscosity(film)
    lw = F.why(); eta_l = _eta_liquids(F, lw)
    S = len(F.subs)
    why = F.why((S,))
    vt = eta.map.split("; out of domain: ", 1)
    why.add("the viscosity's domain" + (f" ({vt[1]})" if len(vt) > 1 else ""), ~np.asarray(eta.in_domain, bool)[None])
    D = np.zeros(F.q.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, s in enumerate(F.subs):
            one = np.zeros((S, 1, 1), bool); one[i] = True
            if s.morph is not None:
                if s.morph.kind not in SPHERICAL:
                    why.add(f"{s.name}'s particles are of shape {s.morph.kind!r}", one & (F.q[i] > 0)[None])
                D[i] = k_B * F.T * s.morph.size.number_moment(-1) / (3 * np.pi * eta.value)
            elif np.isfinite(s.molecular_volume):
                r = (3 * s.molecular_volume / (4 * np.pi)) ** (1 / 3)
                D[i] = k_B * F.T / (6 * np.pi * eta_l * r)
                why.add(f"Stokes-Einstein for the molecules of {s.name} (good to a factor of order 2)",
                        one & (F.q[i] > 0)[None])
    D = np.where(np.isfinite(D) & (D > 0), D, 0.0)
    ev = weakest("computed", eta.evidence, F.ev(lambda s: ("morph",) if s.morph is not None else ("thermo.M", "rho")))
    return derive(D, ev, in_domain=why.domain, map=why.text("diffusivity"))


def settling(film):
    """velocity [S, H, W] m/s, positive along gravity, at standard gravity g_0 (scale by the world's |g| / g_0):
    Stokes with Richardson-Zaki hindrance, v = (rho_env - rho_l) g_0 E_v[d^2] / (18 eta_l) (1 - phi)^4.65, rho_env
    the particle envelope's density, rho_l and eta_l the liquids' (volume-weighted density, log-mixed viscosity),
    E_v the volume-weighted mean, phi = P / (P + L). Negative: creaming. Continuous phases: 0. Reported, not applied,
    on a depth-averaged film (SPEC §4.3). Domain: particle Reynolds number rho_l |v| d / eta_l < RE_MAX, a liquid
    present, the liquids' viscosity, spherical particles. Cells without liquid: 0."""
    F = _F.of(film)
    S = len(F.subs)
    lw = F.why(); eta_l = _eta_liquids(F, lw)
    li, y = F.y()
    rho_l = sum((y[a] * F.subs[i].rho for a, i in enumerate(li)), np.zeros(F.shape))
    phi = _suspension_phi(F)
    why = F.why((S,))
    v = np.zeros(F.q.shape)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i, s in enumerate(F.subs):
            if s.morph is None:
                continue
            here = F.q[i] > 0
            one = np.zeros((S, 1, 1), bool); one[i] = True
            for t, m in lw.items:
                why.add(f"for {s.name}: {t}", one & (m & here)[None])
            why.add(f"no liquid for {s.name} to settle in", one & ((F.L <= 0) & here)[None])
            if s.morph.kind not in SPHERICAL:
                why.add(f"{s.name}'s particles are of shape {s.morph.kind!r}", one & here[None])
            z = s.morph.size
            vi = (envelope_rho(s.morph, s.rho) - rho_l) * g_0 * z.volume_moment(2) / (18 * eta_l) * (1 - phi) ** RZ_N
            vi = np.where((F.L > 0) & np.isfinite(vi), vi, 0.0)
            Re = rho_l * np.abs(vi) * z.d / np.where(np.isfinite(eta_l), eta_l, np.inf)
            why.add(f"particle Reynolds number of {s.name} >= {RE_MAX} (Stokes fails)", one & (here & (Re >= RE_MAX))[None])
            v[i] = vi
    ev = weakest("computed", F.ev(lambda s: ("rho", "rheo.eta", "rheo.E_eta") if s.morph is None else ("rho", "morph")))
    return derive(v, ev, in_domain=why.domain, map=why.text("settling"))


def _mix_evidence(F):
    def keys(s):
        k = s.molecular_volume_keys()
        if s.morph is None and s.cohesion is not None:
            k = k + (s.cohesion_parts_keys() if s.cohesion_parts is not None else s.cohesion_keys())
        if s.morph is not None:
            k = k + ("morph",)
        return k
    ev = F.ev(keys)
    if any(F.present[i] for i in F.idx(True)) and any(F.present[i] for i in F.idx(False)):
        ev = weakest(ev, F.ev(lambda s: ("thermo.gamma",), only=F.idx(False)))
    return weakest("assumed", ev)


def chemical_potential(film):
    """mu [S, H, W] J/m^3 of each substance: d(F_mix)/dq_s of the mixing free energy of `mixing` (both parts):
    the liquids' solution among the liquids alone (Flory-Huggins with the Hansen or Hildebrand interaction), the
    particles' ideal entropy and their excluded volume (goo.substance._freeenergy). A liquid's mu includes minus
    the particles' osmotic pressure; a particle holds no solvent. The pure substance has mu = 0. A substance absent
    from a cell has mu = -inf there when it has a mixing unit (any amount entering lowers F without bound); a
    macromolecular phase (v = inf) has no entropy of its own and stays finite. Empty cells: -inf for every molecular
    or particle substance, 0 for the others. Always in domain: this is the model F. Evidence: at best assumed."""
    F = _F.of(film)
    S = len(F.subs)
    ms = _fe.MixSet.of(F.subs)
    f, mu, _, fx, mux = _fe.evaluate(ms, F.q.reshape(S, -1), F.T.reshape(-1), need_hess=False)
    with np.errstate(invalid="ignore"):
        tot = mu + mux
    return derive(tot.reshape(F.q.shape), _mix_evidence(F), map="chemical_potential")


def surface_tension(film):
    """gamma [H, W] J/m^2: the liquids' surface tensions weighted by their volume fractions among the liquids
    (assumed when two or more are present; exact for one). Particles do not change it on this rung. Domain: every
    liquid present has gamma; a liquid present. A powder: 0, out of domain. An empty cell: the mean gamma of the
    film's liquid substances (what may arrive), in domain."""
    F = _F.of(film)
    li, y = F.y()
    why = F.why()
    g = np.zeros(F.shape)
    for a, i in enumerate(li):
        gm = F.subs[i].thermo.gamma
        if gm is None:
            why.add(f"{F.subs[i].name} has no surface tension", F.q[i] > 0)
        else:
            g += y[a] * gm
    why.add("no liquid: no liquid surface", F.L <= 0)
    # an empty cell holds nothing, but the world uses its gamma for matter that may arrive: the mean gamma of the
    # film's liquid substances (a convention that depends on the substances, not on any cell, so it is element-wise)
    gs = [F.subs[i].thermo.gamma for i in li if F.subs[i].thermo.gamma is not None]
    g = np.where(F.empty, float(np.mean(gs)) if gs else 0.0, g)
    ev = weakest("exact", F.ev(lambda s: ("thermo.gamma",), only=li))
    if F.mixed_liquids().any():
        ev = weakest(ev, "assumed")
    return derive(g, ev, in_domain=why.domain, map=why.text("surface_tension"))


def _p_sat(s, T):
    t = s.thermo
    if t.dH_vap is None:
        return np.full(np.shape(T), t.p_sat)
    return t.p_sat * np.exp(-t.dH_vap / R_gas * (1 / T - 1 / T_room))


def evaporation(film, env):
    """{name: Exchange(activity, activity_env, rate, p_sat)} for every volatile substance (p_sat > 0) of the film:
    the boundary-layer law in its own variables (activities, kg/(m^2 s)). `exchange` gives the same law in the
    world's variables (potentials, m/(Pa s)).

    activity: a = exp(mu v / (k T)) from `chemical_potential` (1 for the pure liquid, 0 where absent).
    activity_env: env.activity[name] (0 when the environment does not name it): the film's equilibrium activity.
    p_sat(T): Clausius-Clapeyron from p_sat and dH_vap.
    rate: diffusion through a still-air boundary layer delta (Fick, ideal vapour): J = rate (a - a_env) kg/(m^2 s),
    rate = D_v(T) M p_sat(T) / (R T delta), D_v(T) = D_vapour (T / T_ref)^1.75 (Fuller's exponent, assumed).
    Positive J leaves the film. Domain: dH_vap given where T != T_ref; D_vapour given."""
    F = _F.of(film)
    mu = chemical_potential(film)
    T = F.T
    out = {}
    for i, s in enumerate(F.subs):
        if not s.volatile:
            continue
        t = s.thermo
        with np.errstate(over="ignore", invalid="ignore"):
            a = np.exp(mu.value[i] * s.molecular_volume / (k_B * T))
        a = np.where(np.isnan(a), 0.0, a)
        ps = _p_sat(s, T)
        wp = F.why(); wp.add(f"{s.name} has no dH_vap and the cell is not at T_ref: p_sat(T) unknown",
                             (t.dH_vap is None) & (np.abs(T - T_room) > 1e-9))
        wr = F.why(); wr.merge(wp)
        wr.add(f"{s.name} has no vapour diffusivity", np.full(F.shape, t.D_vapour is None))
        Dv = t.D_vapour * (T / T_room) ** 1.75 if t.D_vapour is not None else np.zeros(F.shape)
        rate = Dv * t.M * ps / (R_gas * T * env.boundary_layer)
        ev_ps = weakest("computed", s.evidence("thermo.p_sat", "thermo.dH_vap"))
        ev_r = weakest("assumed", ev_ps, s.evidence("thermo.D_vapour", "thermo.M"))
        out[s.name] = Exchange(derive(a, mu.evidence, map=f"exchange.activity[{s.name}]"),
                               float(dict(env.activity).get(s.name, 0.0)),
                               derive(rate, ev_r, in_domain=wr.domain, map=wr.text(f"exchange.rate[{s.name}]")),
                               derive(ps, ev_ps, in_domain=wp.domain, map=wp.text(f"exchange.p_sat[{s.name}]")))
    return out


def reactions(film, env):
    """[ReactionRate(substance, reaction, rate, affinity)] for every reaction of every substance present.

    rate [H, W] 1/s = k exp(-E_a / R (1/T - 1/T_ref)) times the activities of the environment reservoirs it consumes
    (every reactant that is not a substance of the film is drawn from env.activity; 0 when not named). The extent
    advances at rate x (mass of the reacting substance per area); the stoichiometry moves mass.

    affinity [H, W] J per kg of extent, the force conjugate to the extent (positive: the reaction runs downhill):
        A = -dG - sum_i nu_i mu_i / rho_i - sum_res nu_res (R T / M_res) ln a_res
    nu the stoichiometry (kg per kg of extent), dG the description's free energy of reaction (film substances pure,
    reservoirs at activity 1, T_ref), mu_i the film's mixing potential (J/m^3, `mixing`, both parts) of each film
    substance, a_res the environment's activity of each reservoir, M_res its molar mass (Reaction.M_reservoir).
    A first-order rate r with this affinity is the dissipation potential Psi* = r m A_+ per area (convex in A, zero at
    A <= 0, m the reacting mass per area): the world writes cure as a term of F (sum_i rho_i q_i g_i) and this Psi*.
    Conventions: a film substance absent from the cell has mu = -inf (its first molecule makes A = +inf); its term
    is left out, so A is then a finite lower bound (any A > 0 gives the same Psi* rate). A reservoir at activity 0:
    its term is left out likewise (the rate is 0 there). No dG: A = 0, out of domain. dG is at T_ref: away from it,
    out of domain. Domain: every product is a substance of the film (else its mass has nowhere to go); dG and the
    reservoirs' molar masses given; T = T_ref for the affinity."""
    F = _F.of(film)
    names = {s.name: i for i, s in enumerate(F.subs)}
    envs = dict(env.activity)
    out = []
    mx = None
    for i, s in enumerate(F.subs):
        if not F.present[i]:
            continue
        for rx in s.reactions:
            k = rx.rate * np.exp(-rx.E_a / R_gas * (1 / F.T - 1 / T_room))
            why = F.why()
            wa = F.why()
            here = F.q[i] > 0
            A = np.zeros(F.shape)
            Mres = dict(rx.M_reservoir)
            if rx.dG is None:
                wa.add(f"{s.name}'s reaction has no free energy of reaction (dG)", here)
            else:
                if mx is None:
                    mx = mixing_potentials(film).value
                A = A - rx.dG
                wa.add("dG is at T_ref; the cell is not", here & (np.abs(F.T - T_room) > 1e-9))
            for n, c in rx.stoich:
                if n in names:
                    j = names[n]
                    if rx.dG is not None:
                        with np.errstate(invalid="ignore"):
                            mu = mx.mu[j] + mx.mu_x[j]
                        A = A - np.where(np.isfinite(mu), c * mu / F.subs[j].rho, 0.0)
                    continue
                if c < 0:
                    a_res = float(envs.get(n, 0.0))
                    k = k * a_res
                    if rx.dG is not None:
                        if n not in Mres:
                            wa.add(f"no molar mass for the reservoir {n}: its potential is left out", here)
                        elif a_res > 0:
                            A = A - c * R_gas * F.T / Mres[n] * np.log(a_res)
                else:
                    why.add(f"its product {n} is not a substance of the film", here)
            wa.merge(why)
            ev = weakest("assumed", s.evidence("reactions"))
            eva = weakest(ev, s.evidence("reactions.dG"), _mix_evidence(F)) if rx.dG is not None else "assumed"
            out.append(ReactionRate(s.name, rx, derive(k, ev, in_domain=why.domain, map=why.text(f"reactions[{s.name}]")),
                                    derive(np.where(F.empty, 0.0, A), eva, in_domain=wa.domain,
                                           map=wa.text(f"reactions.affinity[{s.name}]"))))
    return out


# ------------------------------------------------------------------------------------------- the world's protocol
def mixing(film, T=None):
    """MixingTerms(f, mu, hess, f_x, mu_x) of goo.world's protocol: the mixing free energy per area of each cell,
    homogeneous of degree 1 in q, with its derivatives with respect to q, split into a convex and a concave part.

    F_mix (goo.substance._freeenergy, whose docstring gives every formula):
      liquids   Flory-Huggins among the liquids alone, k T sum_l c_l q_l ln(q_l / L), with the interaction
                (1/2) q_l^T B q_l / L: B the Hansen distance where both liquids give their polar and hydrogen-bonding
                parts (so water and ethanol mix), else Hildebrand-Scatchard
      particles their ideal entropy k T sum_p c_p q_p ln(q_p / h) (c_p = 1 / particle volume: negligible per volume)
                and their excluded volume, a hard-sphere excess per population against the liquid it sits in whose
                osmotic pressure diverges at the population's packing fraction (from its own size distribution),
                followed until it reaches the capillary entry pressure of the pores; beyond, voids open and the
                term is linear (the liquid's potential is minus that pressure). Particles hold no solvent.
    Split: convex f = the entropies + excluded volume + (beta / 2)(q_l^T q_l / L - L); concave f_x = (1/2) q_l^T
    (B - beta I) q_l / L + (beta / 2) L, beta = max(0, largest eigenvalue of B over the film's liquids) (one number
    per set of substances, so the split is the same in every cell and every window). f + f_x is the free energy;
    a pure substance (a dry powder too) has f = f_x = 0.
    T: the temperature (K), a number or [H, W]; None: the film's state, else T_ref.
    At an absent substance with a mixing unit mu is -inf, its hess entries are 0; on an empty cell f = f_x = 0,
    mu_x = 0, hess = 0. Cost: what the cells' compositions cost: each distinct (volume fractions, T) is solved once,
    on the substances present in it. Evidence: at best assumed. See also `mixing_potentials`, `mixing_hvp`."""
    F = _F.lite(film, T)
    S = len(F.subs)
    ms = _fe.MixSet.of(F.subs)
    f, mu, H, fx, mux = _fe.evaluate(ms, F.q.reshape(S, -1), F.T.reshape(-1), need_hess=True)
    shp = F.shape
    out = MixingTerms(f.reshape(shp), mu.reshape((S,) + shp), H.reshape((S, S) + shp), fx.reshape(shp),
                      mux.reshape((S,) + shp))
    return derive(out, _mix_evidence(F), map=_mix_text(ms))


def _mix_text(ms):
    return (f"mixing (liquids: Flory-Huggins + Hansen/Hildebrand; particles: ideal + hard-sphere excess to the "
            f"capillary entry pressure; convex split beta = {ms.beta:.3g} J/m^3)")


def mixing_potentials(film, T=None):
    """`mixing` without its Hessian: MixingTerms(f, mu, None, f_x, mu_x), the same arrays bit for bit, at a
    fraction of the cost (no [S, S, H, W] array). For callers that need potentials only."""
    F = _F.lite(film, T)
    S = len(F.subs)
    ms = _fe.MixSet.of(F.subs)
    f, mu, _, fx, mux = _fe.evaluate(ms, F.q.reshape(S, -1), F.T.reshape(-1), need_hess=False)
    shp = F.shape
    out = MixingTerms(f.reshape(shp), mu.reshape((S,) + shp), None, fx.reshape(shp), mux.reshape((S,) + shp))
    return derive(out, _mix_evidence(F), map=_mix_text(ms))


def mixing_hvp(film, dq, T=None):
    """The convex part's Hessian applied to dq [S, H, W] (m), per cell: [S, H, W] J/m^3, equal to
    einsum('ijc,jc->ic', mixing(film).hess, dq) without materialising the Hessian (each cell's block is formed on
    its present substances only and discarded). Entries of absent substances are 0, as in `mixing`."""
    F = _F.lite(film, T)
    S = len(F.subs)
    ms = _fe.MixSet.of(F.subs)
    q = F.q.reshape(S, -1); dqf = np.asarray(dq, float).reshape(S, -1)
    out = _fe.hvp(ms, q, F.T.reshape(-1), dqf)
    return derive(out.reshape(F.q.shape), _mix_evidence(F), map="mixing_hvp")


def mixing_block(film, rows, T=None):
    """`mixing` with only the block of its Hessian on the substances `rows` (indices into film.substances):
    MixingTerms(f, mu, hess [R, R, H, W], f_x, mu_x), the block equal to mixing(film).hess[rows][:, rows]. For
    callers that need a few rows of the Hessian (goo.world's local step: the substances that exchange or sorb)."""
    F = _F.lite(film, T)
    S = len(F.subs)
    ms = _fe.MixSet.of(F.subs)
    rows = np.asarray(rows, np.int64)
    f, mu, Hb, fx, mux = _fe.evaluate_block(ms, F.q.reshape(S, -1), F.T.reshape(-1), rows)
    shp = F.shape
    out = MixingTerms(f.reshape(shp), mu.reshape((S,) + shp), Hb.reshape((rows.size, rows.size) + shp), fx.reshape(shp),
                      mux.reshape((S,) + shp))
    return derive(out, _mix_evidence(F), map=_mix_text(ms))


def mobility(film):
    """Lambda [S, H, W] m^2/(Pa s) = D v / (k_B T): `diffusivity` times the mixing volume over k_B T (for a
    sphere in Stokes-Einstein, d^2 / (18 eta) per particle diameter: the Stokes mobility). A macromolecular phase
    (v = inf, D = 0) has 0. Domain and its reasons: `diffusivity`'s. Finite everywhere (0 where eta is infinite)."""
    F = _F.of(film)
    D = diffusivity(film)
    v = np.array([s.molecular_volume for s in F.subs])[:, None, None]
    with np.errstate(invalid="ignore"):
        Lam = np.where((D.value > 0) & np.isfinite(v), D.value * v / (k_B * F.T), 0.0)
    return Derived(Lam, D.evidence, D.in_domain, D.map.replace("diffusivity", "mobility", 1))


def exchange(film, env):
    """ExchangeRates(k, mu_env, activity) of goo.world's protocol: the film's exchange of volatiles with the
    environment's reservoirs, in potentials. Volume flux out of the film per area: e_i = k_i (mu_i - mu_env_i).

    mu_env_i = (k_B T_env / v_i) ln a_i for a volatile substance (p_sat > 0) that env.activity names; a substance
    the environment does not name, or that is not volatile, has no reservoir: k = 0, mu_env = 0.
    k is the boundary-layer law of `evaporation` (diffusion of an ideal vapour through still air of thickness
    delta: mass flux R (a - a_env), R = D_v M p_sat(T) / (R_gas T delta)) written in potentials by its secant at the
    film's present state: k_i = (R_i / rho_i) (a_i - a_env) / (mu_i - mu_env_i), mu_i the film's mixing potential
    (`mixing`, both parts) and a_i = exp(v_i mu_i / (k_B T)). So the flux at the state where k is evaluated is the
    boundary-layer flux exactly, and near equilibrium k -> R a v / (rho k_B T) = D_v M^2 p_sat a / (delta rho^2
    (R_gas T)^2). The world freezes k over a step (DYNAMICS.md §3), so over a step the law is linear in potential.
    k >= 0 always. An environment activity of 0 (perfectly dry) puts mu_env at -inf, which a quadratic Psi* in
    (mu - mu_env) cannot hold: k is then 0 and the cells are out of domain (give a small activity instead).
    activity: [S, H, W] the film's a_i (1 for a pure liquid, 0 where absent). Domain: `evaporation`'s, and a_env > 0."""
    F = _F.of(film)
    S = len(F.subs)
    Tenv = _env_T(env)
    mx = mixing_potentials(film).value
    with np.errstate(invalid="ignore"):
        mu = mx.mu + mx.mu_x
    kT = k_B * F.T
    v = np.array([s.molecular_volume for s in F.subs])
    acts = dict(env.activity)
    k = np.zeros(F.q.shape); mu_env = np.zeros(S); act = np.zeros(F.q.shape)
    why = F.why((S,))
    evs = []
    det = evaporation(film, env)
    for i, s in enumerate(F.subs):
        with np.errstate(over="ignore", invalid="ignore"):
            if np.isfinite(v[i]):
                act[i] = np.where(F.q[i] > 0, np.exp(v[i] * mu[i] / kT), 0.0)
        if not s.volatile or s.name not in acts:
            continue
        one = np.zeros((S, 1, 1), bool); one[i] = True
        a_env = float(acts[s.name])
        e = det[s.name]
        R_i = e.rate.value
        for t, m in _reasons(e.rate, F):
            why.add(t, one & m[None])
        if a_env <= 0:
            mu_env[i] = -np.inf
            why.add(f"the environment's activity of {s.name} is 0: its potential is -inf", one & (F.q[i] > 0)[None])
            continue
        mu_env[i] = k_B * Tenv / v[i] * np.log(a_env)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            x = v[i] * (mu[i] - mu_env[i]) / kT
            sec = np.where(np.abs(x) < 1e-8, 1.0 + 0.5 * x, np.expm1(x) / np.where(x == 0, 1.0, x))
            sec = np.where(np.isneginf(x), 0.0, sec)
            ki = R_i / s.rho * (v[i] / kT) * a_env * sec
        k[i] = np.where(np.isfinite(ki) & (F.q[i] > 0), ki, 0.0)
        evs.append(e.rate.evidence)
    ev = weakest("assumed", *evs) if evs else "assumed"
    return derive(ExchangeRates(k, mu_env, act), ev, in_domain=why.domain, map=why.text("exchange"))


def _reasons(d, F):
    """The (sentence, mask) reasons of a Derived made here, recomputed as masks of its in_domain (one reason)."""
    dom = np.broadcast_to(np.asarray(d.in_domain, bool), F.shape)
    if dom.all():
        return []
    return [(d.map.split("; out of domain: ", 1)[-1].split(" [")[0], ~dom)]


def _liquid_surface(F):
    """Volume-weighted gamma and gamma_d of the cell's liquids, [H, W], and which cells lack them."""
    li, y = F.y()
    g = np.zeros(F.shape); gd = np.zeros(F.shape); missing = np.zeros(F.shape, bool)
    for a, i in enumerate(li):
        t = F.subs[i].thermo
        here = F.q[i] > 0
        if t.gamma is None or t.gamma_d is None:
            missing |= here
            continue
        g += y[a] * t.gamma; gd += y[a] * t.gamma_d
    return g, gd, missing


def _spreading(F, solid, why):
    """Spreading coefficient of the cell's liquid on `solid` by the geometric-mean rule (Owens & Wendt 1969):
    S = 2 sqrt(gd_s gd_l) + 2 sqrt(gp_s gp_l) - 2 gamma_l, unclipped; and gamma_l."""
    g, gd, missing = _liquid_surface(F)
    ts = solid.thermo
    if ts.gamma is None or ts.gamma_d is None:
        why.add(f"the ground's solid {solid.name} has no surface energy (gamma, gamma_d)", F.L > 0)
        return np.zeros(F.shape), g, np.zeros(F.shape, bool)
    why.add("a liquid without gamma and gamma_d", missing)
    gp = np.maximum(g - gd, 0.0)
    S = 2 * np.sqrt(ts.gamma_d * gd) + 2 * np.sqrt(max(ts.gamma - ts.gamma_d, 0.0) * gp) - 2 * g
    ok = (F.L > 0) & ~missing
    return np.where(ok, S, 0.0), g, ok


def wetting(film, ground):
    """Wetting(S, ell) [H, W] of goo.world's protocol: the spreading coefficient S (J/m^2, <= 0) of the cell's
    liquid on the ground's solid and a wetting length ell (m).

    S by the geometric-mean rule from the surface energies in the descriptions (thermo.gamma, thermo.gamma_d of the
    liquids, volume-weighted, and of the ground's solid): S = 2 sqrt(gd_s gd_l) + 2 sqrt(gp_s gp_l) - 2 gamma_l,
    i.e. -S = gamma_l (1 - cos theta) by Young. S > 0 (complete wetting) is returned as 0 with a reason: F_wet's form
    needs S <= 0. ell = sqrt(|A| / (12 pi |S|)), the thickness at which the van der Waals energy A / (12 pi h^2) of
    the film equals |S| (de Gennes 1985), A the Tabor-Winterton Hamaker constant of solid / liquid / air from the
    indices at 589 nm; at least the liquid's molecular length v^(1/3). Cells without liquid: S = 0, ell = 1 nm.
    Particles in the film do not enter on this rung. Evidence: at best assumed."""
    F = _F.of(film)
    why = F.why()
    S, g, ok = _spreading(F, ground.solid, why)
    why.add("complete wetting (S > 0): taken as S = 0", S > 0)
    S = np.minimum(S, 0.0)
    n1 = _hamaker_n(ground.solid) if hasattr(ground.solid, "eps") else 1.5
    n3 = np.sqrt(_host_eps(F, _OneBand())[0][0])
    n2 = 1.0
    a1, a2 = np.sqrt(n1 ** 2 + n3 ** 2), np.sqrt(n2 ** 2 + n3 ** 2)
    A = 3 * h_planck * NU_E / (8 * np.sqrt(2)) * (n1 ** 2 - n3 ** 2) * (n2 ** 2 - n3 ** 2) / (a1 * a2 * (a1 + a2))
    li, y = F.y()
    vmix = sum((y[a] * (F.subs[i].molecular_volume if np.isfinite(F.subs[i].molecular_volume) else 0.0)
                for a, i in enumerate(li)), np.zeros(F.shape))
    lmol = np.where(vmix > 0, np.cbrt(vmix), 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        ell = np.where(S < 0, np.sqrt(np.abs(A) / (12 * np.pi * np.abs(np.where(S < 0, S, 1.0)))), 0.0)
    ell = np.where((F.L > 0) & (S < 0), np.maximum(ell, lmol), 1e-9)
    ev = weakest("assumed", F.ev(lambda s: ("thermo.gamma", "thermo.gamma_d", "eps"), only=F.idx(False)),
                 ground.solid.evidence("thermo.gamma", "thermo.gamma_d", "eps"))
    return derive(Wetting(S, ell), ev, in_domain=why.domain, map=why.text("wetting"))


def sorption(film, ground):
    """Sorption(P, kappa) [S, H, W] of goo.world's protocol: the capillary suction of the ground's pores on each
    liquid (Pa) and a rate coefficient (m/(Pa s)); volume flux into the pores per area = kappa (mu - mu_pore).

    Per liquid substance i, from its own description and the ground's (so a cell's answer does not change when its
    film comes or goes, and an empty cell has the values for what may arrive): P_i = 2 gamma_i cos theta_i / r
    (Laplace, cylindrical pores of radius r = pore_radius), cos theta_i = 1 + S_i / gamma_i by Young, S_i the
    geometric-mean spreading coefficient of i on the ground's solid (`wetting`'s rule), capped at 1.
    Non-wetting (cos theta <= 0): 0, with a reason. Particles: 0 (filtered by the pores on this rung). Full pores: 0.
    kappa_i = porosity r^2 / (8 eta L), Lucas-Washburn: Poiseuille flow in a bundle of straight pores (tortuosity 1,
    assumed) through the wetted depth L = max(held / porosity, r); eta the cell's liquids' viscosity where it holds
    liquid, else the substance's own. Evidence: at best assumed."""
    F = _F.of(film)
    why = F.why()
    r, por, depth = float(ground.pore_radius), float(ground.porosity), float(ground.depth)
    held = np.asarray(ground.held, float)
    w = np.broadcast_to(held.sum(0) if held.size else np.zeros(F.shape), F.shape)
    full = w >= por * depth * (1 - 1e-12)
    ts = ground.solid.thermo
    lw = F.why()
    eta_mix = _eta_liquids(F, lw)
    L = np.maximum(w / por if por > 0 else np.zeros(F.shape), r)
    P = np.zeros(F.q.shape); K = np.zeros(F.q.shape)
    solid_ok = ts.gamma is not None and ts.gamma_d is not None
    if not solid_ok:
        why.add(f"the ground's solid {ground.solid.name} has no surface energy (gamma, gamma_d)", F.L > 0)
    for i in F.idx(False):
        s = F.subs[i]; t = s.thermo; here = F.q[i] > 0
        if t.gamma is None or t.gamma_d is None or not solid_ok:
            if solid_ok:
                why.add(f"{s.name} has no gamma and gamma_d", here)
            continue
        Si = (2 * np.sqrt(ts.gamma_d * t.gamma_d) + 2 * np.sqrt(max(ts.gamma - ts.gamma_d, 0.0) * max(t.gamma - t.gamma_d, 0.0))
              - 2 * t.gamma)
        cos = min(1.0 + Si / t.gamma, 1.0) if t.gamma > 0 else 0.0
        if cos <= 0:
            why.add(f"{s.name} does not wet the pores (cos theta <= 0): no suction", here)
            continue
        P[i] = np.where(full, 0.0, 2 * t.gamma * cos / r) if r > 0 else 0.0
        eta = np.where(F.L > 0, eta_mix, s.rheo.eta if s.rheo.eta is not None else np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            K[i] = np.where(np.isfinite(eta) & (eta > 0), por * r ** 2 / (8 * eta * L), 0.0)
    for t_, m in lw.items:
        why.add(t_, m)
    ev = weakest("assumed", F.ev(lambda s: ("thermo.gamma", "thermo.gamma_d", "rheo.eta"), only=F.idx(False)),
                 ground.solid.evidence("thermo.gamma", "thermo.gamma_d"))
    return derive(Sorption(P, K), ev, in_domain=why.domain, map=why.text("sorption"))
