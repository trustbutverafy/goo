"""Double-entry ledgers and receipts (SPEC §3). An entry moves an amount from one account to another, so a
ledger sums to zero by construction; what is tested is the closure residual: the difference between what the
entries say an account holds and what the state holds."""
from dataclasses import dataclass, field
import hashlib
import numpy as np

KINDS = ("matter", "energy", "light")
FLAGS = ("impossible", "out_of_domain", "bounds_collapsed", "privilege", "reverse", "slope", "solver")


@dataclass(frozen=True)
class Entry:
    ledger: str
    src: str
    dst: str
    amount: np.ndarray        # matter: kg per substance; energy: J; light: fraction of incident per band (or per cell and band)
    note: str = ""


@dataclass(frozen=True)
class Flag:
    kind: str                 # one of FLAGS
    subject: str              # the substance, map, body or painter concerned
    detail: str = ""
    amount: float = 0.0

    def __post_init__(self):
        if self.kind not in FLAGS:
            raise ValueError(f"not a flag kind: {self.kind!r} (one of {FLAGS})")


class Ledger:
    """One ledger. `shape` is the shape of every amount: (S,) for matter, () for energy, (B,) or (H, W, B) for
    light. Accounts are created by their first entry. Matter enters the world from the account 'origin'."""

    def __init__(self, kind, shape=(), labels=None):
        if kind not in KINDS:
            raise ValueError(f"not a ledger kind: {kind!r}")
        self.kind, self.shape, self.labels = kind, tuple(shape), tuple(labels) if labels is not None else None
        self._bal, self._entries = {}, []

    def post(self, src, dst, amount, note=""):
        if src == dst:
            raise ValueError(f"entry from an account to itself: {src!r}")
        a = np.array(np.broadcast_to(np.asarray(amount, np.float64), self.shape))
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{self.kind} ledger: non-finite amount {src!r} -> {dst!r}")
        zero = np.zeros(self.shape)
        self._bal[src] = self._bal.get(src, zero) - a
        self._bal[dst] = self._bal.get(dst, zero) + a
        e = Entry(self.kind, src, dst, a, note)
        self._entries.append(e)
        return e

    def balance(self, account):
        return np.array(self._bal.get(account, np.zeros(self.shape)))

    def accounts(self, prefix=""):
        return sorted(a for a in self._bal if a.startswith(prefix))

    def held(self, prefix):
        """Sum of the balances of every account whose name starts with `prefix`."""
        return sum((self._bal[a] for a in self.accounts(prefix)), np.zeros(self.shape))

    def imbalance(self):
        """Largest magnitude of the sum over all accounts: zero to rounding, by double entry."""
        return float(np.max(np.abs(sum(self._bal.values(), np.zeros(self.shape))))) if self._bal else 0.0

    def residual(self, account, actual, scale=None):
        """Closure residual of one account (or every account under a prefix ending in '*') against the state:
        max |entries - actual| / scale. scale defaults to the largest magnitude in `actual`, or 1 if that is 0."""
        said = self.held(account[:-1]) if account.endswith("*") else self.balance(account)
        actual = np.asarray(actual, np.float64)
        s = scale if scale is not None else (float(np.max(np.abs(actual))) or 1.0)
        return float(np.max(np.abs(said - actual))) / s

    def mark(self):
        return len(self._entries)

    def since(self, mark=0):
        return tuple(self._entries[mark:])

    def fork(self):
        new = Ledger(self.kind, self.shape, self.labels)
        new._bal = {k: v.copy() for k, v in self._bal.items()}
        new._entries = list(self._entries)
        return new

    def digest(self):
        h = hashlib.sha256()
        for k in sorted(self._bal):
            h.update(k.encode()); h.update(np.ascontiguousarray(self._bal[k]).tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class Receipt:
    """What an operation did, against every ledger it touched (SPEC §3). `F` gives the free energy by term
    before and after, when the operation changed the world. `report` holds named numbers the operation
    states about itself (the parts of D_num, the largest slope, the integral of a clock rate). Receipts
    add: an act is the sum of its steps; a reported number adds, except one whose name starts with
    'max ' which takes the larger."""
    op: str
    t0: float
    t1: float
    entries: tuple = ()
    residual: dict = field(default_factory=dict)       # ledger kind -> closure residual
    F: dict = field(default_factory=dict)              # term -> (before, after), J
    flags: tuple = ()
    report: dict = field(default_factory=dict)         # name -> float

    def moved(self, ledger, src=None, dst=None):
        """Total amount this receipt's entries moved in one ledger, optionally from accounts starting with
        `src` and/or to accounts starting with `dst`. 0.0 if nothing matched."""
        hits = [e.amount for e in self.entries if e.ledger == ledger
                and (src is None or e.src.startswith(src)) and (dst is None or e.dst.startswith(dst))]
        return sum(hits[1:], hits[0].copy()) if hits else 0.0

    def flagged(self, kind):
        return tuple(f for f in self.flags if f.kind == kind)

    def __add__(self, other):
        if not isinstance(other, Receipt):
            return NotImplemented
        res = dict(self.residual)
        for k, v in other.residual.items():
            res[k] = max(res.get(k, 0.0), v)
        F = dict(self.F)
        for k, (b, a) in other.F.items():
            F[k] = (F[k][0], a) if k in F else (b, a)
        flags = self.flags + tuple(f for f in other.flags if f not in self.flags)
        report = dict(self.report)
        for k, v in other.report.items():
            report[k] = v if k not in report else (max(report[k], v) if k.startswith("max ") else report[k] + v)
        return Receipt(self.op, min(self.t0, other.t0), max(self.t1, other.t1), self.entries + other.entries,
                       res, F, flags, report)

    def digest(self):
        h = hashlib.sha256(f"{self.op}|{self.t0!r}|{self.t1!r}".encode())
        for e in self.entries:
            h.update(f"{e.ledger}|{e.src}|{e.dst}".encode()); h.update(np.ascontiguousarray(e.amount).tobytes())
        for f in self.flags:
            h.update(f"{f.kind}|{f.subject}|{f.detail}|{f.amount!r}".encode())
        for k in sorted(self.report):
            h.update(f"{k}|{float(self.report[k])!r}".encode())
        return h.hexdigest()
