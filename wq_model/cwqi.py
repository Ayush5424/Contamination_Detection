"""Cooling-Water Quality Index (CWQI) - paper Section III-B, equations (1) and (2).

Each reading is turned into a quality rating q_i that is 0 at the ideal value and
100 at the limit on the side of the deviation (1); the ratings are combined with
the weights w_i into a single index (2).  Lower CWQI = better water.
"""
import numpy as np

SENSORS = ["temp", "ph", "cond", "do", "frc", "turb", "inhib"]

# Table II of the paper: ideal value, low limit, high limit (None = one-sided), weight
SPEC = {
    "temp":  dict(name="Temperature",           unit="°C",    ideal=29.0, low=None, high=38.0, w=0.12),
    "ph":    dict(name="pH",                    unit="",           ideal=7.8,  low=6.8,  high=9.0,  w=0.18),
    "cond":  dict(name="Conductivity",          unit="µS/cm", ideal=1500, low=None, high=3000, w=0.14),
    "do":    dict(name="Dissolved oxygen",      unit="mg/L",       ideal=8.0,  low=4.0,  high=None, w=0.14),
    "frc":   dict(name="Free residual chlorine", unit="mg/L",      ideal=0.5,  low=0.1,  high=1.0,  w=0.14),
    "turb":  dict(name="Turbidity",             unit="NTU",        ideal=1.0,  low=None, high=10.0, w=0.14),
    "inhib": dict(name="Corrosion inhibitor",   unit="mg/L",       ideal=10.0, low=4.0,  high=None, w=0.14),
}

SAFE_LIMIT, ALERT_LIMIT = 50.0, 75.0          # Safe < 50 <= Warning < 75 <= Alert
LEVELS = ["Safe", "Warning", "Alert"]


def sub_index(sensor, value):
    """Equation (1): quality rating of one parameter, 0 (ideal) to 100 (at the limit)."""
    s = SPEC[sensor]
    v = np.asarray(value, dtype=float)
    q = np.zeros_like(v)
    if s["high"] is not None:
        above = v > s["ideal"]
        q[above] = (v[above] - s["ideal"]) / (s["high"] - s["ideal"])
    if s["low"] is not None:
        below = v < s["ideal"]
        q[below] = (s["ideal"] - v[below]) / (s["ideal"] - s["low"])
    return 100.0 * q


def weighted_contributions(readings):
    """w_i * q_i for every sensor; columns follow SENSORS."""
    return np.column_stack([SPEC[s]["w"] * sub_index(s, readings[s]) for s in SENSORS])


def cwqi(readings):
    """Equation (2): the weighted sum of the ratings.

    `readings` is any mapping/DataFrame with the seven sensor keys.
    """
    return weighted_contributions(readings).sum(axis=1)


def risk_level(index_value):
    """Map a CWQI value to 0 = Safe, 1 = Warning, 2 = Alert."""
    v = np.asarray(index_value)
    return np.where(v >= ALERT_LIMIT, 2, np.where(v >= SAFE_LIMIT, 1, 0))


def risk_name(index_value):
    return LEVELS[int(risk_level(index_value))]
