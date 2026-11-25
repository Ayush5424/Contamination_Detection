"""Train every model of the system and save it to models/.

    python train_models.py            # compact demo models, about 10 minutes on a laptop
    python train_models.py --paper    # the exact settings of the paper, about 1.5 hours
    python train_models.py --xgb-forecaster-only   # add only the XGBoost forecaster (about 2 minutes)

Steps follow Section IV of the paper: simulate 240 days of data, split it 60/15/25 in time
order, train the hybrid stacked ensemble, the LSTM forecaster and the Isolation Forest,
then evaluate everything on the unseen 60-day test segment.
"""
import argparse
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

from wq_model.cwqi import SENSORS, cwqi, risk_level
from wq_model.features import (FEATURE_COLUMNS, anomaly_features, base_contributions, base_index,
                               build_features, fill_gaps, smoothed)
from wq_model.estimator import HybridCWQIEstimator
from wq_model.forecaster import HORIZONS, WINDOW, CWQIForecaster
from wq_model.anomaly import AnomalyDetector
from wq_model.xgb_forecaster import XGBForecaster
from wq_model.simulator import EVENT_ROOT, STEPS_PER_HOUR, simulate

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def add_xgb_forecaster(feats, base, y, tr, te):
    """Train only the XGBoost forecaster and add its test result to metrics.json."""
    log("Training the XGBoost lag-feature forecaster ...")
    xfc = XGBForecaster().fit(feats, base, y, np.where(tr)[0])
    joblib.dump(xfc, os.path.join(MODELS, "xgb_forecaster.joblib"), compress=3)
    n = len(y)
    ok = np.zeros(n, bool)
    ok[WINDOW - 1:n - max(HORIZONS)] = True
    rows = np.where(te & ok)[0]
    pred = xfc.predict(feats, base, rows)[:, 2]
    rmse = float(np.sqrt(mean_squared_error(y[rows + 12], pred)))
    path = os.path.join(MODELS, "metrics.json")
    with open(path) as f:
        metrics = json.load(f)
    metrics["forecast_60min_rmse"]["XGBoost (lag features)"] = rmse
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"  XGBoost 60-min forecast RMSE on the test segment: {rmse:.2f}")


def main(paper_size, xgb_only=False):
    os.makedirs(MODELS, exist_ok=True)
    np.random.seed(0)
    torch.manual_seed(0)
    t0 = time.time()

    # ------------------------------------------------------------------ data (Section IV-A)
    log("Simulating 240 days of cooling-loop data ...")
    true, measured, labels, events, faults = simulate()
    n = len(true)
    a, b = int(0.60 * n), int(0.75 * n)
    split = np.array(["train"] * a + ["val"] * (b - a) + ["test"] * (n - b))
    tr, va, te = split == "train", split == "val", split == "test"
    y = labels["cwqi_true"].values
    log(f"  {n:,} readings, {len(events)} contamination events, {len(faults)} sensor faults")

    # ------------------------------------------------------------------ pre-processing (III-C)
    clean = fill_gaps(measured)
    feats = build_features(clean)
    X = feats[FEATURE_COLUMNS].values
    base = base_index(feats)
    contrib = base_contributions(feats)

    # ------------------------------------------------------------------ hybrid estimator (III-C, III-D)
    if xgb_only:
        add_xgb_forecaster(feats, base, y, tr, te)
        return
    log("Training the hybrid stacked ensemble (6 learners + linear meta-learner) ...")
    fit_rows = tr | va
    est = HybridCWQIEstimator(size="paper" if paper_size else "demo")
    est.fit(X[fit_rows], base[fit_rows], y[fit_rows], contrib[fit_rows])
    cwqi_hat = est.predict(X, base)
    joblib.dump(est, os.path.join(MODELS, "estimator.joblib"), compress=3)
    log(f"  meta-learner weights: {est.meta_weights}")

    # ------------------------------------------------------------------ LSTM forecaster (III-E)
    log("Training the LSTM forecaster (2 x 64 units) ...")
    fc = CWQIForecaster()
    fc.fit(smoothed(feats), base, y, np.where(tr)[0], np.where(va)[0], log=log)
    torch.save(fc.net.state_dict(), os.path.join(MODELS, "forecaster.pt"))
    with open(os.path.join(MODELS, "forecaster_stats.json"), "w") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fc.stats.items()}, f)

    # ------------------------------------------------------------------ XGBoost forecaster (Section V)
    log("Training the XGBoost lag-feature forecaster ...")
    xfc = XGBForecaster().fit(feats, base, y, np.where(tr)[0])
    joblib.dump(xfc, os.path.join(MODELS, "xgb_forecaster.joblib"), compress=3)

    # ------------------------------------------------------------------ Isolation Forest (III-F)
    log("Training the Isolation Forest and tuning its threshold on validation data ...")
    A = anomaly_features(clean)
    det = AnomalyDetector().fit(A[tr])
    is_anomaly = labels["event"].values | labels["fault"].values
    det.tune_threshold(A[va], is_anomaly[va])
    joblib.dump(det, os.path.join(MODELS, "anomaly.joblib"), compress=3)

    # ------------------------------------------------------------------ evaluation on the test segment
    log("Evaluating on the 60-day test segment ...")
    ok = np.zeros(n, bool)
    ok[WINDOW - 1:n - max(HORIZONS)] = True
    rows = np.where(te & ok)[0]
    forecasts = np.full((n, 3), np.nan)
    forecasts[rows] = fc.predict(smoothed(feats), base, np.arange(n), rows)
    fmax = np.where(np.isnan(forecasts).any(1), cwqi_hat, np.nanmax(forecasts, 1))
    xgb_forecasts = np.full((n, 3), np.nan)
    xgb_forecasts[rows] = xfc.predict(feats, base, rows)

    hi = te & (y >= 50)
    rmse = lambda t, p: float(np.sqrt(mean_squared_error(t, p)))
    future60 = np.r_[y[12:], np.full(12, np.nan)]
    flags = det.flags(A)

    worst_next_hour = pd.Series(y[::-1]).rolling(13, min_periods=1).max().values[::-1]
    true_level = risk_level(worst_next_hour)
    pred_level = risk_level(np.maximum(cwqi_hat, fmax))
    m = te & ok

    root_hits, root_n = 0, 0
    for e in events:
        if split[e["start"]] == "train":
            continue
        peak = e["start"] + int(np.argmax(y[e["start"]:e["end"]]))
        att = est.attribute(X[peak:peak + 1], contrib[peak:peak + 1])[0]
        root_hits += SENSORS[int(np.argmax(att))] in EVENT_ROOT[e["type"]]
        root_n += 1

    metrics = {
        "trained_with": "paper settings" if paper_size else "compact demo settings",
        "test_days": int(te.sum() // (24 * STEPS_PER_HOUR)),
        "estimation": {
            "Index formula on raw readings": dict(r2=float(r2_score(y[te], cwqi(clean[te]))),
                                                  rmse=rmse(y[te], cwqi(clean[te]))),
            "Hybrid stacked ensemble": dict(r2=float(r2_score(y[te], cwqi_hat[te])), rmse=rmse(y[te], cwqi_hat[te]),
                                            mae=float(mean_absolute_error(y[te], cwqi_hat[te])),
                                            rmse_contaminated=rmse(y[hi], cwqi_hat[hi]),
                                            bias_contaminated=float(np.mean(cwqi_hat[hi] - y[hi]))),
        },
        "forecast_60min_rmse": {
            "Persistence (no change)": rmse(future60[rows], cwqi_hat[rows]),
            "LSTM": rmse(future60[rows], forecasts[rows, 2]),
            "XGBoost (lag features)": rmse(future60[rows], xgb_forecasts[rows, 2]),
        },
        "anomaly": {"threshold_tau": det.tau,
                    "f1": float(f1_score(is_anomaly[te], flags[te])),
                    "events_detected": f"{sum(flags[e['start']:e['end']].any() for e in events if split[e['start']] == 'test')}"
                                       f"/{sum(split[e['start']] == 'test' for e in events)}"},
        "early_warning": {"macro_f1": float(f1_score(true_level[m], pred_level[m], average="macro")),
                          "false_alarm_rate": float((pred_level[m][true_level[m] == 0] >= 1).mean())},
        "root_cause_top1": f"{root_hits}/{root_n}",
        "meta_weights": est.meta_weights,
        "training_minutes": round((time.time() - t0) / 60, 1),
    }
    with open(os.path.join(MODELS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------------ demo stream for the dashboard
    stream = measured[te].copy()
    stream["cwqi_true"] = y[te]
    stream["event_type"] = labels["event_type"].values[te]
    stream["fault"] = labels["fault"].values[te]
    stream["step"] = np.where(te)[0]
    stream.to_csv(os.path.join(MODELS, "demo_stream.csv"))

    log(json.dumps(metrics, indent=2))
    log(f"Done in {metrics['training_minutes']} min. Models saved to {MODELS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true", help="use the full settings of the paper (slow)")
    ap.add_argument("--xgb-forecaster-only", action="store_true",
                    help="keep the trained models and add only the XGBoost forecaster")
    args = ap.parse_args()
    main(args.paper, args.xgb_forecaster_only)
