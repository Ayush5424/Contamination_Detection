"""Pre-processing - paper Section III-C.

Short gaps are filled, and for every sensor the feature vector x_t holds the latest
reading plus its causal 30-min rolling median and standard deviation (21 features).
The index of the rolling-median readings is the physics-based base value W^b_t.
"""
import numpy as np
import pandas as pd

from .cwqi import SENSORS, cwqi, weighted_contributions

WINDOW = 6                                   # 6 samples x 5 min = 30 min


def fill_gaps(measured: pd.DataFrame) -> pd.DataFrame:
    """Interpolate gaps of up to 30 min, then carry the last value."""
    return measured.interpolate(limit=WINDOW, limit_direction="both").ffill().bfill()


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for s in SENSORS:
        cols[s] = clean[s]
        cols[s + "_med"] = clean[s].rolling(WINDOW, min_periods=1).median()
        cols[s + "_std"] = clean[s].rolling(WINDOW, min_periods=2).std().fillna(0)
    return pd.DataFrame(cols, index=clean.index)


FEATURE_COLUMNS = [c for s in SENSORS for c in (s, s + "_med", s + "_std")]


def smoothed(features: pd.DataFrame) -> pd.DataFrame:
    m = features[[s + "_med" for s in SENSORS]].copy()
    m.columns = SENSORS
    return m


def base_index(features: pd.DataFrame) -> np.ndarray:
    """W^b_t: equations (1)-(2) applied to the rolling-median readings."""
    return cwqi(smoothed(features))


def base_contributions(features: pd.DataFrame) -> np.ndarray:
    """w_s q_s of the smoothed readings, one column per sensor (used by the explainer)."""
    return weighted_contributions(smoothed(features))


def anomaly_features(clean: pd.DataFrame) -> np.ndarray:
    """Isolation Forest input (Section III-F): value, first difference, 30-min rolling std."""
    values = clean[SENSORS].values
    diff = np.vstack([np.zeros((1, len(SENSORS))), np.diff(values, axis=0)])
    roll_std = clean[SENSORS].rolling(WINDOW, min_periods=2).std().fillna(0).values
    return np.hstack([values, diff, roll_std])
