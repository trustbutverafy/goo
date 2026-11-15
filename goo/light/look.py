"""A look: light solved on a Stack, per band, with its ledger (SPEC §3, §7). docs/LIGHT.md §5-§6, §10.

The stack is read, never written: every array a look touches is read through the Optical protocol or as
Film.h and Ground.depth, and the results are new arrays.

How a look is computed (docs/LIGHT.md §10): the protocol is called once per film and once for the ground;
cells whose inputs are bit-identical are solved once (exactly: keys are verified, not trusted); the distinct
cells are solved in blocks small enough to stay in cache, on a few threads. Every cell's arithmetic is the
same elementwise arithmetic whatever the block, so the result does not depend on blocking or threads.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import numpy as np
from goo.ledger import Ledger, Receipt, Flag
from goo.light.operator import Layer, star, solve, channels as _channels
from goo.light.bulk import homogeneous
from goo.light.interface import boundary, entry
from goo.light.optical import unwrap

BLOCK = 32768          # values per block (bands x cells): a block's temporaries stay in cache
THREADS = min(2, os.cpu_count() or 1)


@dataclass(frozen=True)
class Operators:
    """The stack's light-independent operators, top first: the top surface (for diffuse flux on both sides),
    one sublayer per film (its top interface star its bulk; bottom-most film last), and the ground (its top
    interface star its body). Leading shape [B, H, W], one channel.

    n_surface, sigma: the index and roughness of the top surface in each cell (that of the topmost film
    present, else the ground's), which the light's entry reads. ok[j]: where layer j has a steady state."""
    layers: tuple
    names: tuple
    n_surface: np.ndarray
    sigma: np.ndarray
    ok: np.ndarray
    flags: tuple


def _inputs(stack, bands, optical):
    """Read matter through the protocol, once. Returns (films, ground, flags): films bottom first, each a dict
    of K, S, nh, nt [B, H, W], sig [H, W], h [H, W] with every value in an empty cell set to a fixed one (0
    for K, S, h, sig; 1 for indices), so that empty cells neither read nor distinguish what the map said."""
    B = bands.B
    H, W = stack.ground.z.shape
    lead = (B, H, W)
    flags = []
    gK, gS, gn = unwrap(optical.ground_optical(stack.ground, bands), 3, "ground_optical", [lead] * 3, flags)
    gnt, gsig = unwrap(optical.ground_surface(stack.ground, bands), 2, "ground_surface", [lead, (H, W)], flags)
    for name, arr in [("ground index", gnt), ("ground host index", gn)]:
        if np.any(arr <= 0):
            raise ValueError(f"{name} must be positive")
    if np.any(gK < 0):
        flags.append(Flag("impossible", "ground", "K < 0: optical gain, not passive; booked to debt.ground",
                          float(-np.min(gK))))
    if np.any(gS < 0):
        flags.append(Flag("impossible", "ground", "S < 0: negative scattering coefficient", float(-np.min(gS))))
    ground = dict(K=gK, S=gS, nt=gnt, sig=gsig)

    films = []
    for k, f in enumerate(stack.films):
        K, S, nh = unwrap(optical.optical(f, bands), 3, f"optical(film[{k}])", [lead] * 3, flags)
        nt, sig = unwrap(optical.surface(f, bands), 2, f"surface(film[{k}])", [lead, (H, W)], flags)
        h = np.asarray(f.h, np.float64)
        p = h > 0
        if np.any(p & ((nt <= 0) | (nh <= 0))):
            raise ValueError(f"film[{k}]: indices must be positive")
        if np.any(p & (K < 0)):
            flags.append(Flag("impossible", f"film[{k}]", "K < 0: optical gain, not passive (fails herglotz); "
                              f"the power it adds is booked to debt[{k}]", float(-np.min(np.where(p, K, 0.0)))))
        if np.any(p & (S < 0)):
            flags.append(Flag("impossible", f"film[{k}]", "S < 0: negative scattering coefficient",
                              float(-np.min(np.where(p, S, 0.0)))))
        films.append(dict(K=np.where(p, K, 0.0), S=np.where(p, S, 0.0), nh=np.where(p, nh, 1.0),
                          nt=np.where(p, nt, 1.0), sig=np.where(p, sig, 0.0), h=np.where(p, h, 0.0)))
    return films, ground, flags


def _build(films, ground, depth, n_ambient):
    """The operators from the inputs, on whatever cell shape they have ([B, *cells], sig and h [*cells])."""
    n_s, sig_s = ground["nt"], ground["sig"]
    for f in films:                                        # bottom first: higher films overwrite
        p = f["h"] > 0
        n_s = np.where(p, f["nt"], n_s)
        sig_s = np.where(p, f["sig"], sig_s)
    lead = n_s.shape
    layers, names, oks = [], [], []
    layers.append(boundary(np.broadcast_to(np.asarray(n_ambient, np.float64), lead), n_s))
    names.append("surface"); oks.append(np.ones(lead, bool))
    n_above = n_s
    for k in range(len(films) - 1, -1, -1):                # top film first
        f = films[k]
        p = f["h"] > 0
        bnd = boundary(n_above, np.where(p, f["nt"], n_above))
        R, T, ok = homogeneous(f["K"], f["S"], f["h"])
        R, T = R[..., None, None], T[..., None, None]
        layers.append(star(bnd, Layer(R, T, R, T))); names.append(f"film[{k}]"); oks.append(ok)
        n_above = np.where(p, f["nh"], n_above)
    bnd = boundary(n_above, ground["nt"])
    R, T, ok = homogeneous(ground["K"], ground["S"], depth)
    R, T = R[..., None, None], T[..., None, None]
    layers.append(star(bnd, Layer(R, T, R, T))); names.append("ground"); oks.append(ok)
    return layers, names, n_s, sig_s, np.array(oks)


def operators(stack, bands, optical, n_ambient=1.0):
    """Build the stack's operators from what `optical` (a goo.light.Optical) returns. Films are entered from the
    top; a film with h = 0 in a cell is absent there: its sublayer is the identity and the index above the
    next interface down is unchanged."""
    if np.any(np.asarray(n_ambient) <= 0):
        raise ValueError("ambient index must be positive")
    films, ground, flags = _inputs(stack, bands, optical)
    layers, names, n_s, sig_s, oks = _build(films, ground, stack.ground.depth, n_ambient)
    return Operators(tuple(layers), tuple(names), n_s, sig_s, oks, tuple(flags))


def _distinct_cells(films, ground, ncell):
    """(representatives, inverse) of cells whose inputs are bit-identical, or None if dedup would not pay.
    A 64-bit FNV-style hash of every input's bits per cell, then an exact check of every input against its
    representative: a hash collision makes dedup refuse, never merges different cells."""
    cols = []
    for d in films + [ground]:
        for key, a in d.items():
            a = np.ascontiguousarray(a.reshape(-1, ncell) if a.ndim == 3 else a.reshape(1, ncell))
            cols.append(a)
    h = np.full(ncell, np.uint64(1469598103934665603))
    prime = np.uint64(1099511628211)
    for a in cols:
        for row in a.view(np.uint64):
            h ^= row
            h *= prime
    _, rep, inv = np.unique(h, return_index=True, return_inverse=True)
    if rep.size > 0.6 * ncell:
        return None
    inv = inv.ravel()
    back = rep[inv]                                        # each cell's representative
    for a in cols:
        for row in a.view(np.uint64):
            if not np.array_equal(row[back], row):
                return None
    return rep, inv


def _compress(d, idx, ncell):
    return {k: (v.reshape(v.shape[0], ncell)[:, idx] if v.ndim == 3 else v.reshape(ncell)[idx]) for k, v in d.items()}


@dataclass(frozen=True)
class Look:
    """What a stack shows under a light, as fractions of the incident power per band and cell.

    reflectance = specular + diffuse                      [H, W, B]
    absorbed[k]   net power absorbed by stack.films[k]    [K, H, W, B]   (bottom first; negative for gain)
    ground        absorbed by the ground's body           [H, W, B]
    transmitted   leaving through the back of a ground of finite depth   [H, W, B]  (0 for an opaque ground)
    debt[k], debt_ground   power added by gain, = max(-absorbed, 0)       (booked to debt[...])
    diverged      cells and bands past a lasing threshold: no steady state           [H, W, B] bool
    unresolved    1 where diverged, else 0; every other output is 0 there (a defined value, never NaN)

    reflectance + sum(absorbed) + ground + transmitted + unresolved = 1 to rounding, and the receipt's light
    ledger books the panel means: incident + debt = reflected + absorbed + transmitted (+ unresolved)."""
    bands: object
    reflectance: np.ndarray
    specular: np.ndarray
    diffuse: np.ndarray
    absorbed: np.ndarray
    ground: np.ndarray
    transmitted: np.ndarray
    debt: np.ndarray
    debt_ground: np.ndarray
    diverged: np.ndarray
    unresolved: np.ndarray
    receipt: Receipt


def look(stack, bands, light, optical, *, n_ambient=1.0, channels=1, t=0.0, dedup=True, threads=None):
    """Solve light on `stack` in `bands` under `light`, reading matter only through `optical` (goo.light.Optical).

    channels > 1 runs the same first-rung physics on that many uncoupled identical channels (the incident
    power split equally): a check that the n-channel path is the one-channel path. dedup and threads change
    how the work is done, never the result (tested bit for bit). Returns a Look."""
    if light.bands.B != bands.B or not np.array_equal(light.bands.centres, bands.centres):
        raise ValueError("look: the light's bands are not the bands asked for")
    if np.any(np.asarray(n_ambient) <= 0):
        raise ValueError("ambient index must be positive")
    B = bands.B
    H, W = stack.ground.z.shape
    ncell = H * W
    nf = len(stack.films)
    films, ground, flags = _inputs(stack, bands, optical)
    depth = stack.ground.depth

    dd = _distinct_cells(films, ground, ncell) if dedup and ncell > 1 else None
    if dd is not None:
        rep, inv = dd
        cf = [_compress(f, rep, ncell) for f in films]
        cg = _compress(ground, rep, ncell)
    else:
        rep, inv = None, None
        cf = [_compress(f, slice(None), ncell) for f in films]
        cg = _compress(ground, slice(None), ncell)
    C = cg["sig"].shape[0]
    N = nf + 2
    lam = bands.centres[:, None]
    out_R = np.empty((B, C)); out_spec = np.empty((B, C)); out_tr = np.empty((B, C))
    out_loss = np.empty((N, B, C)); out_div = np.empty((B, C), bool)

    def run(sl):
        bf = [{k: (v[:, sl] if v.ndim == 2 else v[sl]) for k, v in f.items()} for f in cf]
        bg = {k: (v[:, sl] if v.ndim == 2 else v[sl]) for k, v in cg.items()}
        layers, _, n_s, sig_s, oks = _build(bf, bg, depth, n_ambient)
        R_in, T_in, spec = entry(n_ambient, n_s, sig_s[None], lam, light.geometry)
        top = layers[0]
        layers = [Layer(R_in[..., None, None], T_in[..., None, None], top.Rp, top.Tp)] + layers[1:]
        if channels > 1:
            layers = [_channels(L, channels) for L in layers]
        sol = solve(layers)
        with np.errstate(invalid="ignore"):
            div = ~np.all(oks, axis=0) | ~(sol.margin > 0)
            div |= ~np.isfinite(sol.reflected) | ~np.all(np.isfinite(sol.loss), axis=0)
            div |= ~np.isfinite(sol.transmitted)
        out_R[:, sl] = sol.reflected
        out_spec[:, sl] = spec
        out_tr[:, sl] = sol.transmitted
        out_loss[:, :, sl] = sol.loss
        out_div[:, sl] = div

    step = max(1, BLOCK // max(B, 1))
    blocks = [slice(i, min(i + step, C)) for i in range(0, C, step)]
    nthreads = THREADS if threads is None else threads
    if nthreads > 1 and len(blocks) > 1:
        with ThreadPoolExecutor(nthreads) as ex:
            list(ex.map(run, blocks))
    else:
        for sl in blocks:
            run(sl)

    # everything elementwise is done on the distinct cells, in [..., C, B] layout; each output is expanded once
    cb = lambda a: np.ascontiguousarray(np.swapaxes(a, -1, -2))
    div = cb(out_div)
    live = ~div
    z = lambda a: np.where(live, a, 0.0)
    R_c = z(cb(out_R))
    spec_c = z(cb(out_spec))
    diff_c = z(R_c - spec_c)
    loss = cb(out_loss)                                    # [N, C, B]: surface, films top first, ground
    abs_c = z(loss[1:1 + nf][::-1])                        # bottom first, like stack.films
    gnd_c = z(loss[-1])
    tr_c = z(cb(out_tr))
    unres_c = div.astype(np.float64)
    debt_c = np.maximum(-abs_c, 0.0)
    debtg_c = np.maximum(-gnd_c, 0.0)
    absp_c = np.maximum(abs_c, 0.0)
    gndp_c = np.maximum(gnd_c, 0.0)

    # closure, per cell and band, relative to the power entering (incident + debt):
    # incident + debt = reflected + absorbed + transmitted (+ unresolved)
    entering = 1.0 + debt_c.sum(0) + debtg_c
    taken = spec_c + diff_c + absp_c.sum(0) + gndp_c + tr_c + unres_c
    residual = float(np.max(np.abs(entering - taken) / entering)) if R_c.size else 0.0

    def expand(a):                                         # [..., C, B] -> [..., H, W, B]
        if inv is not None:
            a = a[..., inv, :]
        return a.reshape(a.shape[:-2] + (H, W, B))

    reflectance, specular, diffuse = expand(R_c), expand(spec_c), expand(diff_c)
    absorbed, ground_abs, transmitted = expand(abs_c), expand(gnd_c), expand(tr_c)
    debt, debt_g, diverged, unresolved = expand(debt_c), expand(debtg_c), expand(div), expand(unres_c)

    L = Ledger("light", (B,), tuple(f"{c_ * 1e9:.6g} nm" for c_ in bands.centres))
    # panel means by pairwise summation along a contiguous axis (accurate; the same order on either path)
    mean = lambda a: np.ascontiguousarray(a.reshape(H * W, B).T).mean(-1)
    L.post("incident", "reflected.specular", mean(specular))
    L.post("incident", "reflected.diffuse", mean(diffuse))
    for k in range(nf):
        L.post("incident", f"absorbed[{k}]", mean(np.maximum(absorbed[k], 0.0)))
        if np.any(debt_c[k] > 0):
            L.post(f"debt[{k}]", "incident", mean(debt[k]), "power added by optical gain")
    L.post("incident", "absorbed.ground", mean(np.maximum(ground_abs, 0.0)))
    if np.any(debtg_c > 0):
        L.post("debt.ground", "incident", mean(debt_g), "power added by optical gain")
    L.post("incident", "transmitted", mean(transmitted), "through the back of the ground")
    if np.any(div):
        L.post("incident", "unresolved", mean(unresolved), "past a lasing threshold: no steady state")
    if B:
        residual = max(residual, float(np.max(np.abs(L.balance("incident") + 1.0) / mean(expand(entering)))))

    flags = list(flags)
    if np.any(diverged):
        flags.append(Flag("impossible", "stack", "past a lasing threshold in "
                          f"{int(diverged.sum())} of {diverged.size} cell-bands: no steady state; reflectance and "
                          "absorption are 0 there and the incident power is booked to 'unresolved'",
                          float(diverged.mean())))
    receipt = Receipt("look", float(t), float(t), L.since(0), {"light": residual}, {}, tuple(flags))
    return Look(bands, reflectance, specular, diffuse, absorbed, ground_abs, transmitted, debt, debt_g, diverged,
                unresolved, receipt)
