"""End-to-end monitor: loads the trained models and turns sensor readings into a risk level.

Mirrors Fig. 1 of the paper:
sensors -> pre-processing -> {hybrid CWQI estimate + SHAP, forecast, anomaly} -> fusion -> level
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch

from .cwqi import SENSORS, SPEC, cwqi
from .features import (FEATURE_COLUMNS, anomaly_features, base_contributions, base_index,
                       build_features, fill_gaps, smoothed)
from .forecaster import WINDOW, CWQIForecaster, HORIZON_MIN
from .anomaly import AnomalyDetector
from .fusion import classify, risk_score
from .xgb_forecaster import MIN_HISTORY as XGB_MIN_HISTORY

FORECASTERS = {"lstm": "LSTM", "xgb": "XGBoost"}

HISTORY = WINDOW + 12                        # readings kept in memory (enough for every model)


class WaterQualityMonitor:
    def __init__(self, model_dir):
        self.estimator = joblib.load(os.path.join(model_dir, "estimator.joblib"))
        self.anomaly: AnomalyDetector = joblib.load(os.path.join(model_dir, "anomaly.joblib"))
        self.forecaster = CWQIForecaster()
        self.forecaster.net.load_state_dict(torch.load(os.path.join(model_dir, "forecaster.pt")))
        with open(os.path.join(model_dir, "forecaster_stats.json")) as f:
            stats = json.load(f)
        self.forecaster.stats = {k: np.array(v) if isinstance(v, list) else v for k, v in stats.items()}
        xgb_path = os.path.join(model_dir, "xgb_forecaster.joblib")
        self.xgb_forecaster = joblib.load(xgb_path) if os.path.exists(xgb_path) else None
        self.flag_history = []

    def assess(self, history: pd.DataFrame, step_index: np.ndarray, forecaster="lstm"):
        """Assess the newest reading.

        history: the last HISTORY measured readings (columns = 7 sensors, NaN allowed);
        step_index: the absolute 5-min step number of each row (for the time-of-day input);
        forecaster: "lstm" (Section III-E) or "xgb" (the XGBoost alternative of Section V).
        """
        if forecaster == "xgb" and self.xgb_forecaster is None:
            raise ValueError("XGBoost forecaster not trained. Run: python train_models.py --xgb-forecaster-only")
        clean = fill_gaps(history[SENSORS])
        feats = build_features(clean)
        base = base_index(feats)
        contrib = base_contributions(feats)
        X = feats[FEATURE_COLUMNS].values
        last = len(history) - 1

        estimate = float(self.estimator.predict(X[last:], base[last:])[0])            # eq. (3)
        if forecaster == "xgb" and len(history) >= XGB_MIN_HISTORY:
            forecasts = self.xgb_forecaster.predict(feats, base, [last])[0]
        elif forecaster == "lstm" and len(history) >= WINDOW:
            forecasts = self.forecaster.predict(smoothed(feats), base, step_index, [last])[0]
        else:
            forecasts = np.array([estimate] * 3)
        forecast_max = float(np.max(forecasts))

        A = anomaly_features(clean)
        score = float(self.anomaly.score(A[last:])[0])
        flagged = bool(score >= self.anomaly.tau)
        self.flag_history = (self.flag_history + [flagged])[-6:]
        evidence = AnomalyDetector.evidence(self.flag_history)                        # eq. (5)

        r = risk_score(estimate, forecast_max, evidence)                               # eq. (6)
        attribution = self.estimator.attribute(X[last:], contrib[last:])[0]            # eq. (4)
        ranked = sorted(zip(SENSORS, attribution), key=lambda t: -t[1])

        return {
            "raw_cwqi": float(cwqi(clean.iloc[last:])[0]),
            "cwqi": estimate,
            "forecast": {f"{m} min": float(v) for m, v in zip(HORIZON_MIN, forecasts)},
            "forecast_max": forecast_max,
            "forecaster": FORECASTERS[forecaster],
            "risk_score": r,
            "level": classify(r),
            "current_level": classify(estimate),
            "anomaly_score": score,
            "anomaly_flag": flagged,
            "anomaly_evidence": evidence,
            "attribution": {s: float(v) for s, v in zip(SENSORS, attribution)},
            "root_cause": ranked[0][0],
            "root_cause_name": SPEC[ranked[0][0]]["name"],
        }
