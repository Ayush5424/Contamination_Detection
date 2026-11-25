"""XGBoost forecaster - the lag-feature alternative of paper Section V (Table IV).

Inputs per reading: the 21 features x_t, the base index W^b_t, and for each lag of
15, 30 and 60 min the smoothed readings and their change since then (64 inputs).
One XGBoost model per horizon learns W_{t+h} - W^b_t, exactly like the LSTM target,
so the two forecasters are interchangeable in the fusion step.
"""
import numpy as np
from xgboost import XGBRegressor

from .cwqi import SENSORS
from .features import FEATURE_COLUMNS
from .forecaster import HORIZONS

LAGS = [3, 6, 12]                            # 15, 30, 60 min
MIN_HISTORY = max(LAGS) + 1


def lag_features(features, base):
    S = features[[s + "_med" for s in SENSORS]].values
    parts = [features[FEATURE_COLUMNS].values, np.asarray(base)[:, None]]
    for lag in LAGS:
        shifted = np.vstack([np.repeat(S[:1], lag, 0), S[:-lag]])
        parts += [shifted, S - shifted]
    return np.hstack(parts)


class XGBForecaster:
    def __init__(self, n_estimators=400):
        self.models = [XGBRegressor(n_estimators=n_estimators, max_depth=6, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, n_jobs=4, random_state=0) for _ in HORIZONS]

    def fit(self, features, base, target, train_idx):
        LF = lag_features(features, base)
        n = len(base)
        idx = np.asarray(train_idx)
        for model, h in zip(self.models, HORIZONS):
            rows = idx[idx < n - h]
            model.fit(LF[rows], target[rows + h] - base[rows])
        return self

    def predict(self, features, base, rows):
        """Forecasts (n_rows x 3 horizons) for the given row positions."""
        LF = lag_features(features, base)[np.asarray(rows)]
        b = np.asarray(base)[np.asarray(rows)]
        return np.column_stack([b + m.predict(LF) for m in self.models])
