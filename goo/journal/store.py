"""Saving a session at the size of what it holds (docs/JOURNAL.md §5).

A session is pickled with three substitutions, made by a pickler's reducer_override so that no other module has to
know about them:

  numpy arrays     zlib-compressed bytes (level 1): an empty panel or a stratum's zeros cost almost nothing, a wet
                   film costs what its numbers do
  goo.world.grid.Grid   rebuilt from (H, W, dx, i0, j0) on load: its index arrays are a function of those
  goo.ledger.Ledger     balances only. The entries list is what receipts are cut from during an act (mark/since);
                   it is history, and the journal (the receipts' digests) keeps that. The world's digest reads only
                   the balances, so a loaded world has the same digest and continues identically.

Files are not portable across goo, numpy or Python versions (pickle); the journal is."""
import io
import pickle
import zlib
import numpy as np
from goo.ledger import Ledger
from goo.world.grid import Grid

MIN_ARRAY = 256          # bytes; smaller arrays are pickled as they are


def _array(blob, dtype, shape):
    return np.frombuffer(zlib.decompress(blob), dtype=np.dtype(dtype)).reshape(shape).copy()


def _grid(H, W, dx, i0, j0):
    return Grid(H, W, dx, i0, j0)


def _ledger(kind, shape, labels, balances):
    L = Ledger(kind, shape, labels)
    L._bal = {k: np.array(v) for k, v in balances.items()}
    return L


class _Pickler(pickle.Pickler):
    def reducer_override(self, obj):
        if type(obj) is np.ndarray and obj.nbytes >= MIN_ARRAY and obj.dtype != object:
            a = np.ascontiguousarray(obj)
            return _array, (zlib.compress(a.tobytes(), 1), a.dtype.str, a.shape)
        if type(obj) is Grid:
            i0 = int(round(obj.Y[0] / obj.dx - 0.5)) if obj.N else 0
            j0 = int(round(obj.X[0] / obj.dx - 0.5)) if obj.N else 0
            return _grid, (obj.H, obj.W, obj.dx, i0, j0)
        if type(obj) is Ledger:
            return _ledger, (obj.kind, obj.shape, obj.labels, dict(obj._bal))
        return NotImplemented


def dumps(obj):
    buf = io.BytesIO()
    _Pickler(buf, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)
    return buf.getvalue()


def loads(b):
    return pickle.loads(b)


def save(obj, path):
    path = str(path)
    with open(path + ".tmp", "wb") as f:
        f.write(dumps(obj))
    import os
    os.replace(path + ".tmp", path)


def load(path):
    with open(path, "rb") as f:
        return loads(f.read())
