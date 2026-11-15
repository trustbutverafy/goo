"""Lorenz-Mie scattering by a homogeneous sphere and by a coated (core-shell) sphere, numpy, vectorised over arrays
of (relative index, size parameter). Promoted unchanged in substance from goo/classic/matter/mie.py; `efficiencies`
and `coated_efficiencies` add chunking so that large particles do not exhaust memory.

Algorithms are those of Bohren & Huffman (1983), "Absorption and Scattering of Light by Small Particles",
appendices A (BHMIE) and B (BHCOAT): the logarithmic derivative D_n(mx) by downward recurrence, the Riccati-
Bessel functions psi_n, chi_n of the real size parameter by upward recurrence, terms to n_stop = x + 4 x^(1/3)
+ 2 (Wiscombe 1980). Each element of an array stops at its own n_stop. Convention: m = n + ik with k >= 0 absorbing
(exp(-i omega t)); k < 0 (gain) is accepted and computed by the same recurrences.

Returns efficiencies (cross-section / geometric cross-section pi r^2): Q_ext, Q_sca, Q_abs = Q_ext - Q_sca, and the
asymmetry parameter g = <cos theta>. Checked against miepython (tests/substance/test_optics.py, when installed).

Limits: BHCOAT's upward recurrences in the shell lose accuracy for strongly absorbing shells of large optical
thickness (|Im m2| y >> 1).
"""
import numpy as np


def _nstop(x): return np.floor(x + 4.0 * np.cbrt(x) + 2.0).astype(int)


def _sums(an_bn_iter, x, N):
    """Accumulate Q_ext, Q_sca, g from an iterator of (n, a_n, b_n, active)."""
    qe = np.zeros(x.shape); qs = np.zeros(x.shape); gq = np.zeros(x.shape)
    ap = np.zeros(x.shape, complex); bp = np.zeros(x.shape, complex)
    for n, an, bn, act in an_bn_iter:
        an = np.where(act, an, 0); bn = np.where(act, bn, 0)
        qe += (2 * n + 1) * (an.real + bn.real)
        qs += (2 * n + 1) * (np.abs(an) ** 2 + np.abs(bn) ** 2)
        if n > 1:
            gq += (n - 1) * (n + 1) / n * (ap * np.conj(an) + bp * np.conj(bn)).real
        gq += (2 * n + 1) / (n * (n + 1)) * (an * np.conj(bn)).real
        ap, bp = an, bn
    qe *= 2 / x ** 2; qs *= 2 / x ** 2
    g = np.where(qs > 0, 4 / x ** 2 * gq / np.maximum(qs, 1e-300), 0.0)
    return qe, qs, g


def sphere(m, x):
    """Homogeneous sphere: relative complex index m (particle / medium), size parameter x = 2 pi r n_med / lam.
    Broadcasts m and x. Returns (Q_ext, Q_sca, Q_abs, g)."""
    m, x = np.broadcast_arrays(np.asarray(m, complex), np.asarray(x, float))
    shape = x.shape; m = m.ravel(); x = x.ravel()
    ns = _nstop(x); N = int(ns.max())
    mx = m * x
    nmx = int(max(N, np.abs(mx).max())) + 16
    D = np.zeros((nmx + 1, x.size), complex)                  # downward: D_{n-1} = n/z - 1/(D_n + n/z)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / mx - 1 / (D[n] + n / mx)

    def it():
        psi0, psi1 = np.cos(x), np.sin(x)
        chi0, chi1 = -np.sin(x), np.cos(x)
        xi1 = psi1 - 1j * chi1
        for n in range(1, N + 1):
            act = n <= ns
            psi = np.where(act, (2 * n - 1) * psi1 / x - psi0, psi1)
            chi = np.where(act, (2 * n - 1) * chi1 / x - chi0, chi1)
            xi = psi - 1j * chi
            ta = D[n] / m + n / x; tb = m * D[n] + n / x
            an = (ta * psi - psi1) / (ta * xi - xi1)
            bn = (tb * psi - psi1) / (tb * xi - xi1)
            yield n, an, bn, act
            psi0, psi1 = np.where(act, psi1, psi0), psi
            chi0, chi1 = np.where(act, chi1, chi0), chi
            xi1 = psi1 - 1j * chi1

    with np.errstate(all="ignore"):
        qe, qs, g = _sums(it(), x, N)
    return tuple(a.reshape(shape) for a in (qe, qs, qe - qs, g))


def coated(m_core, m_shell, x_core, x_outer):
    """Coated sphere (BHCOAT): core of relative index m_core and size parameter x_core = 2 pi a n_med / lam,
    shell m_shell out to x_outer >= x_core. Efficiencies refer to the outer radius. Returns (Q_ext, Q_sca,
    Q_abs, g). x_core = 0 is a homogeneous shell sphere; a hollow sphere has m_core = 1 / n_med."""
    m1, m2, x, y = np.broadcast_arrays(np.asarray(m_core, complex), np.asarray(m_shell, complex),
                                       np.asarray(x_core, float), np.asarray(x_outer, float))
    shape = y.shape; m1, m2, x, y = (a.ravel() for a in (m1, m2, x, y))
    tiny = x < 1e-8
    if tiny.all():
        return sphere(m2.reshape(shape), y.reshape(shape))
    xs = np.where(tiny, 1e-8, x)
    ns = _nstop(y); N = int(ns.max())
    x1, x2, y2 = m1 * xs, m2 * xs, m2 * y

    def it():
        d0x1 = np.cos(x1) / np.sin(x1); d0x2 = np.cos(x2) / np.sin(x2); d0y2 = np.cos(y2) / np.sin(y2)
        psi0y, psi1y = np.cos(y), np.sin(y)
        chi0y, chi1y = -np.sin(y), np.cos(y)
        xi1y = psi1y - 1j * chi1y
        chi0y2, chi1y2 = -np.sin(y2), np.cos(y2)
        chi0x2, chi1x2 = -np.sin(x2), np.cos(x2)
        core = ~tiny                                          # the core still matters for this element
        for n in range(1, N + 1):
            act = n <= ns
            psiy = (2 * n - 1) * psi1y / y - psi0y
            chiy = (2 * n - 1) * chi1y / y - chi0y
            xiy = psiy - 1j * chiy
            d1y2 = 1 / (n / y2 - d0y2) - n / y2
            d1x1 = 1 / (n / x1 - d0x1) - n / x1
            d1x2 = 1 / (n / x2 - d0x2) - n / x2
            chix2 = (2 * n - 1) * chi1x2 / x2 - chi0x2
            chiy2 = (2 * n - 1) * chi1y2 / y2 - chi0y2
            chipx2 = chi1x2 - n * chix2 / x2
            chipy2 = chi1y2 - n * chiy2 / y2
            anc = (m2 * d1x1 - m1 * d1x2) / (m2 * d1x1 * chix2 - m1 * chipx2) / (chix2 * d1x2 - chipx2)
            bnc = (m2 * d1x2 - m1 * d1x1) / (m2 * chipx2 - m1 * d1x1 * chix2) / (chix2 * d1x2 - chipx2)
            brack = anc * (chiy2 * d1y2 - chipy2)
            crack = bnc * (chiy2 * d1y2 - chipy2)
            small = np.maximum.reduce([np.abs(brack * chipy2), np.abs(brack * chiy2), np.abs(crack * chipy2),
                                       np.abs(crack * chiy2)]) <= 1e-12 * np.abs(d1y2)
            core = core & ~(small | ~np.isfinite(brack) | ~np.isfinite(crack))
            brack = np.where(core, brack, 0); crack = np.where(core, crack, 0)
            dnbar = (d1y2 - brack * chipy2) / (1 - brack * chiy2)
            gnbar = (d1y2 - crack * chipy2) / (1 - crack * chiy2)
            ta = dnbar / m2 + n / y; tb = m2 * gnbar + n / y
            an = (ta * psiy - psi1y) / (ta * xiy - xi1y)
            bn = (tb * psiy - psi1y) / (tb * xiy - xi1y)
            yield n, an, bn, act
            psi0y, psi1y = psi1y, psiy; chi0y, chi1y = chi1y, chiy; xi1y = psi1y - 1j * chi1y
            chi0x2, chi1x2 = chi1x2, chix2; chi0y2, chi1y2 = chi1y2, chiy2
            d0x1, d0x2, d0y2 = d1x1, d1x2, d1y2

    with np.errstate(all="ignore"):
        qe, qs, g = _sums(it(), y, N)
    return tuple(a.reshape(shape) for a in (qe, qs, qe - qs, g))


def rayleigh(m, x):
    """Small-particle limit (x << 1, |m| x << 1): Q_sca = 8/3 x^4 |L|^2, Q_abs = 4 x Im L, L = (m^2-1)/(m^2+2)."""
    m = np.asarray(m, complex); x = np.asarray(x, float)
    L = (m * m - 1) / (m * m + 2)
    return 8 / 3 * x ** 4 * np.abs(L) ** 2, 4 * x * L.imag


_BUDGET = 4_000_000          # complex elements of the D_n table per chunk (~64 MB)


def _chunked(fn, size_key, arrays):
    """Apply fn to arrays (broadcast, raveled) in chunks sorted by size_key so each chunk's recursion is short."""
    arrays = np.broadcast_arrays(*arrays)
    shape = arrays[0].shape
    flat = [a.ravel() for a in arrays]
    key = size_key(*flat)
    order = np.argsort(key, kind="stable")
    out = [np.empty(key.size) for _ in range(4)]
    i = 0
    while i < key.size:
        j = i + 1
        lim = key[order[i]]
        # grow the chunk while the longest recursion times the chunk size stays in budget
        step = max(1, int(_BUDGET // (lim + 32)))
        j = min(key.size, i + step)
        while j > i + 1 and (key[order[j - 1]] + 32) * (j - i) > _BUDGET:
            j = i + max(1, (j - i) // 2)
        sel = order[i:j]
        res = fn(*[a[sel] for a in flat])
        for o, r in zip(out, res):
            o[sel] = r
        i = j
    return tuple(o.reshape(shape) for o in out)


def efficiencies(m, x):
    """sphere(m, x) over arrays of any size, in memory-bounded chunks."""
    m = np.asarray(m, complex); x = np.asarray(x, float)
    return _chunked(sphere, lambda m, x: np.maximum(x, np.abs(m * x)) + 4 * np.cbrt(x) + 2, (m, x))


def coated_efficiencies(m_core, m_shell, x_core, x_outer):
    """coated(...) over arrays of any size, in memory-bounded chunks."""
    return _chunked(coated, lambda a, b, c, y: y + 4 * np.cbrt(y) + 2,
                    (np.asarray(m_core, complex), np.asarray(m_shell, complex), np.asarray(x_core, float),
                     np.asarray(x_outer, float)))
