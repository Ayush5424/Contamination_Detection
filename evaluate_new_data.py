"""Does the system still give the right answer on new data?

    python evaluate_new_data.py              # 3 new data sets + 2 harder conditions (about 3 minutes)
    python evaluate_new_data.py --days 30    # shorter data sets

The trained models in models/ are NOT retrained. They are run on data they have never seen:
  * new simulations with other random seeds than the training data (seed 42): new events,
    new noise, new sensor faults;
  * the same new data with twice the sensor noise;
  * the same new data with a pH probe that reads 0.3 too high (calibration drift).
The same metrics as train_models.py are reported, plus the early-warning lead time.
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
from sklearn.metrics import f1_score, mean_squared_error, r2_score

from wq_model.anomaly import AnomalyDetector
from wq_model.cwqi import SENSORS, cwqi, risk_level
from wq_model.features import (FEATURE_COLUMNS, anomaly_features, base_contributions, base_index,
                               build_features, fill_gaps, smoothed)
from wq_model.forecaster import HORIZONS, WINDOW, CWQIForecaster
from wq_model.simulator import EVENT_ROOT, NOISE, STEPS_PER_HOUR, simulate
from wq_system.batch import lead_times

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
HOUR = STEPS_PER_HOUR


def load_models():
    est = joblib.load(os.path.join(MODELS, "estimator.joblib"))
    det: AnomalyDetector = joblib.load(os.path.join(MODELS, "anomaly.joblib"))
    fc = CWQIForecaster()
    fc.net.load_state_dict(torch.load(os.path.join(MODELS, "forecaster.pt")))
    with open(os.path.join(MODELS, "forecaster_stats.json")) as f:
        fc.stats = {k: np.array(v) if isinstance(v, list) else v for k, v in json.load(f).items()}
    xfc = joblib.load(os.path.join(MODELS, "xgb_forecaster.joblib"))
    return est, det, fc, xfc


def evaluate(models, true, measured, labels, events):
    est, det, fc, xfc = models
    n = len(true)
    y = labels["cwqi_true"].values
    clean = fill_gaps(measured)
    feats = build_features(clean)
    X = feats[FEATURE_COLUMNS].values
    base = base_index(feats)
    contrib = base_contributions(feats)

    cwqi_hat = est.predict(X, base)
    raw = cwqi(clean)

    rows = np.arange(WINDOW - 1, n - max(HORIZONS))
    lstm = fc.predict(smoothed(feats), base, np.arange(n), rows)
    xgb = xfc.predict(feats, base, rows)
    future60 = y[rows + 12]

    worst_next_hour = pd.Series(y[::-1]).rolling(13, min_periods=1).max().values[::-1]
    true_level = risk_level(worst_next_hour)[rows]
    fused = {}
    for name, f in (("LSTM", lstm), ("XGBoost", xgb)):
        pred = risk_level(np.maximum(cwqi_hat[rows], f.max(1)))
        full = np.zeros(n, int)
        full[rows] = pred
        warn_leads, warn_missed = lead_times(y, full, 50, 1)
        alert_leads, alert_missed = lead_times(y, full, 75, 2)
        fused[name] = dict(
            macro_f1=f1_score(true_level, pred, average="macro"),
            false_alarm=(pred[true_level == 0] >= 1).mean(),
            alert_recall=(pred[true_level == 2] == 2).mean() if (true_level == 2).any() else np.nan,
            warn_lead=np.mean(warn_leads) if warn_leads else np.nan, warn_n=len(warn_leads), warn_missed=warn_missed,
            alert_lead=np.mean(alert_leads) if alert_leads else np.nan, alert_n=len(alert_leads),
            alert_missed=alert_missed)

    is_anomaly = labels["event"].values | labels["fault"].values
    flags = det.flags(anomaly_features(clean))

    hits = 0
    for e in events:
        peak = e["start"] + int(np.argmax(y[e["start"]:e["end"]]))
        att = est.attribute(X[peak:peak + 1], contrib[peak:peak + 1])[0]
        hits += SENSORS[int(np.argmax(att))] in EVENT_ROOT[e["type"]]

    hi = y >= 50
    rmse = lambda t, p: float(np.sqrt(mean_squared_error(t, p)))
    return dict(
        r2=r2_score(y, cwqi_hat), rmse=rmse(y, cwqi_hat), rmse_raw=rmse(y, raw),
        rmse_hi=rmse(y[hi], cwqi_hat[hi]) if hi.any() else np.nan,
        bias_hi=float(np.mean(cwqi_hat[hi] - y[hi])) if hi.any() else np.nan,
        fc_persist=rmse(future60, cwqi_hat[rows]), fc_lstm=rmse(future60, lstm[:, 2]), fc_xgb=rmse(future60, xgb[:, 2]),
        anomaly_f1=f1_score(is_anomaly, flags),
        events_caught=f"{sum(flags[e['start']:e['end']].any() for e in events)}/{len(events)}",
        root_cause=f"{hits}/{len(events)}", fused=fused, n_events=len(events))


def with_noise(true, measured, factor, seed):
    """Extra Gaussian noise so the total sensor noise is `factor` times the original."""
    rng = np.random.default_rng(seed)
    out = measured.copy()
    for s in SENSORS:
        out[s] = out[s] + rng.normal(0, NOISE[s] * np.sqrt(factor ** 2 - 1), len(out))
    return out


def main(days, seeds):
    t0 = time.time()
    print("Loading the trained models (trained on seed 42; they are not retrained here) ...", flush=True)
    models = load_models()

    results = []
    for seed in seeds:
        print(f"Simulating {days} new days with seed {seed} ...", flush=True)
        true, measured, labels, events, _ = simulate(days=days, seed=seed)
        results.append((f"New data, seed {seed}", evaluate(models, true, measured, labels, events)))
        if seed == seeds[0]:
            noisy = with_noise(true, measured, 2.0, seed + 1)
            results.append((f"Seed {seed}, sensor noise x2", evaluate(models, true, noisy, labels, events)))
            drift = measured.copy()
            drift["ph"] = drift["ph"] + 0.3
            results.append((f"Seed {seed}, pH probe +0.3 drift", evaluate(models, true, drift, labels, events)))

    with open(os.path.join(MODELS, "metrics.json")) as f:
        m = json.load(f)
    ref = dict(r2=m["estimation"]["Hybrid stacked ensemble"]["r2"], rmse=m["estimation"]["Hybrid stacked ensemble"]["rmse"],
               rmse_hi=m["estimation"]["Hybrid stacked ensemble"]["rmse_contaminated"],
               fc_lstm=m["forecast_60min_rmse"]["LSTM"], fc_xgb=m["forecast_60min_rmse"]["XGBoost (lag features)"],
               anomaly_f1=m["anomaly"]["f1"], macro_f1=m["early_warning"]["macro_f1"],
               false_alarm=m["early_warning"]["false_alarm_rate"], root=m["root_cause_top1"])

    f = lambda v, d=3: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"
    print()
    print("=" * 108)
    print(f"{'Data set':34s} {'R2':>7s} {'RMSE':>6s} {'RMSE>=50':>8s} {'LSTM60':>7s} {'XGB60':>6s} "
          f"{'AnomF1':>6s} {'macroF1':>7s} {'FalseAl':>7s} {'Root':>7s}")
    print(f"{'Original test segment (reference)':34s} {f(ref['r2'], 4):>7s} {f(ref['rmse'], 2):>6s} "
          f"{f(ref['rmse_hi'], 2):>8s} {f(ref['fc_lstm'], 2):>7s} {f(ref['fc_xgb'], 2):>6s} {f(ref['anomaly_f1'], 2):>6s} "
          f"{f(ref['macro_f1']):>7s} {f(100 * ref['false_alarm'], 2) + '%':>7s} {ref['root']:>7s}")
    for name, r in results:
        u = r["fused"]["LSTM"]
        print(f"{name:34s} {f(r['r2'], 4):>7s} {f(r['rmse'], 2):>6s} {f(r['rmse_hi'], 2):>8s} {f(r['fc_lstm'], 2):>7s} "
              f"{f(r['fc_xgb'], 2):>6s} {f(r['anomaly_f1'], 2):>6s} {f(u['macro_f1']):>7s} "
              f"{f(100 * u['false_alarm'], 2) + '%':>7s} {r['root_cause']:>7s}")
    print()
    print("Early warning (fusion), mean minutes before the true crossing; (missed onsets):")
    print(f"{'Data set':34s} {'LSTM: Warning':>15s} {'LSTM: Alert':>13s} {'XGB: Warning':>14s} {'XGB: Alert':>12s}")
    for name, r in results:
        cells = []
        for fcn in ("LSTM", "XGBoost"):
            u = r["fused"][fcn]
            cells += [f"{f(u['warn_lead'], 0)} ({u['warn_missed']}/{u['warn_n'] + u['warn_missed']})",
                      f"{f(u['alert_lead'], 0)} ({u['alert_missed']}/{u['alert_n'] + u['alert_missed']})"]
        print(f"{name:34s} {cells[0]:>15s} {cells[1]:>13s} {cells[2]:>14s} {cells[3]:>12s}")
    print("=" * 108)
    print(f"Done in {(time.time() - t0) / 60:.1f} min.")

    out = {name: r for name, r in results}
    with open(os.path.join(MODELS, "new_data_results.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=lambda v: None if isinstance(v, float) and np.isnan(v) else float(v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60, help="days per new data set")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 123, 2027], help="random seeds (training used 42)")
    args = ap.parse_args()
    main(args.days, args.seeds)
