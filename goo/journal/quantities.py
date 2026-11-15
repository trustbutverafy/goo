"""Quantities in acts' JSON: a number is SI; a string is a number and a unit ("5 mL", "0.25mm", "20 deg", "2 min",
"20 C"). Units are goo.units' and a few painter's words; nothing else is accepted."""
import re
import numpy as np
from goo import units as U


class QuantityError(ValueError):
    pass


UNITS = {
    "length": {"m": U.m, "mm": U.mm, "um": U.um, "µm": U.um, "nm": U.nm, "cm": 1e-2},
    "time": {"s": U.s, "ms": U.ms, "min": U.minute, "minute": U.minute, "minutes": U.minute, "h": U.hour,
             "hour": U.hour, "hours": U.hour, "day": U.day, "days": U.day},
    "volume": {"m3": 1.0, "m^3": 1.0, "L": U.litre, "l": U.litre, "mL": U.mL, "ml": U.mL, "uL": U.uL, "ul": U.uL,
               "µL": U.uL},
    "flow": {"m3/s": 1.0, "m^3/s": 1.0, "mL/s": U.mL, "ml/s": U.mL, "uL/s": U.uL, "ul/s": U.uL, "mL/min": U.mL / 60,
             "L/s": U.litre},
    "angle": {"rad": 1.0, "deg": np.pi / 180, "°": np.pi / 180},
    "temperature": {"K": 1.0},
    "number": {"": 1.0},
}
_NUM = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(\S*)\s*$")


def quantity(v, kind, what="value"):
    """SI float of a number or a string with a unit of the given kind."""
    if isinstance(v, bool):
        raise QuantityError(f"{what}: a number, not {v!r}")
    if isinstance(v, (int, float)):
        x = float(v)
    elif isinstance(v, str):
        m = _NUM.match(v)
        if not m:
            raise QuantityError(f"{what}: cannot read {v!r} as a number with a unit")
        num, unit = float(m.group(1)), m.group(2)
        if kind == "temperature" and unit in ("C", "°C", "degC"):
            x = U.celsius(num)
        elif unit == "" :
            x = num
        else:
            table = UNITS[kind]
            if unit not in table:
                raise QuantityError(f"{what}: unit {unit!r} is not a {kind} unit ({', '.join(u for u in table if u)})")
            x = num * table[unit]
    else:
        raise QuantityError(f"{what}: a number or a string with a unit, not {type(v).__name__}")
    if not np.isfinite(x):
        raise QuantityError(f"{what}: not finite")
    return x
