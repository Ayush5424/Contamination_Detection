"""Runs the saved models on a whole new data set at once (the "Test on new data" tab).

The data can be a CSV file the user uploads (any plant or lab log with the seven sensors), or a
new simulation with another seed than training (42), optionally with extra sensor noise or a
drifting pH probe. Nothing is retrained. Every reading gets the same assessment as in the live
monitor (hybrid CWQI, forecast, fused level, anomaly flag); every Warning/Alert episode gets a
SHAP root cause. If the file has the true index (column cwqi_true, as in simulated data), the
accuracy is reported too, with the same metrics as evaluate_new_data.py.
"""
import io
import re
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error, r2_score

from wq_model.cwqi import LEVELS, SENSORS, SPEC, cwqi, risk_level
from wq_model.features import (FEATURE_COLUMNS, anomaly_features, base_contributions, base_index,
                               build_features, fill_gaps, smoothed)
from wq_model.forecaster import HORIZON_MIN, WINDOW
from wq_model.simulator import EVENT_LABEL, EVENT_ROOT, NOISE, STEPS_PER_HOUR, simulate
from wq_model.xgb_forecaster import MIN_HISTORY as XGB_MIN_HISTORY

from .processor import step_numbers

HOUR = STEPS_PER_HOUR
MIN_ROWS = WINDOW                                  # 3 h: the LSTM needs this much history
MAX_ROWS = 120 * 24 * HOUR                         # 120 days
CHART_POINTS = 1500

# accepted column names (lower case, spaces/brackets removed) for each sensor
ALIASES = {
    "temp": ["temp", "temperature", "watertemperature", "t"],
    "ph": ["ph"],
    "cond": ["cond", "conductivity", "ec", "specificconductance"],
    "do": ["do", "dissolvedoxygen", "oxygen"],
    "frc": ["frc", "freeresidualchlorine", "chlorine", "freechlorine", "cl2"],
    "turb": ["turb", "turbidity"],
    "inhib": ["inhib", "inhibitor", "corrosioninhibitor"],
}
TIME_NAMES = ["ts", "time", "timestamp", "datetime", "date", "unnamed:0", ""]


def _key(name):
    name = re.sub(r"\(.*?\)|\[.*?\]", "", str(name).lower())
    return re.sub(r"[^a-z0-9:]", "", name)


def parse_csv(text):
    """CSV text -> (readings with a 5-min DatetimeIndex, cwqi_true or None, event_type or None, notes)."""
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise ValueError(f"could not read the CSV file ({exc})")
    keys = {_key(c): c for c in df.columns}
    cols, notes = {}, []
    for s, names in ALIASES.items():
        found = next((keys[n] for n in names if n in keys), None)
        if found is not None:
            cols[s] = found
    missing = [s for s in SENSORS if s not in cols]
    if len(missing) == len(SENSORS):
        raise ValueError("no sensor columns found. Expected: ts, temp, ph, cond, do, frc, turb, inhib "
                         "(download the sample CSV to see the format)")
    readings = pd.DataFrame({s: pd.to_numeric(df[cols[s]], errors="coerce") if s in cols else np.nan
                             for s in SENSORS})
    for s in missing:                              # a sensor that is not installed: assume its ideal value
        readings[s] = SPEC[s]["ideal"]
        notes.append(f"No {SPEC[s]['name']} column: its ideal value {SPEC[s]['ideal']} was assumed.")

    tcol = next((keys[n] for n in TIME_NAMES if n in keys), None)
    ts = pd.to_datetime(df[tcol], errors="coerce") if tcol is not None else None
    if ts is None or ts.isna().mean() > 0.5:
        ts = pd.date_range("2026-01-01", periods=len(df), freq="5min")
        notes.append("No usable time column: readings were taken as 5 minutes apart.")
    readings.index = pd.DatetimeIndex(ts)
    truth = pd.to_numeric(df["cwqi_true"], errors="coerce").values if "cwqi_true" in df else None
    events = df["event_type"].fillna("none").astype(str).values if "event_type" in df else None

    keep = ~readings.index.isna()
    readings, truth = readings[keep], truth[keep] if truth is not None else None
    events = events[keep] if events is not None else None
    if not readings.index.is_monotonic_increasing:
        order = np.argsort(readings.index.values, kind="stable")
        readings = readings.iloc[order]
        truth = truth[order] if truth is not None else None
        events = events[order] if events is not None else None
        notes.append("Rows were sorted by time.")

    step = readings.index.to_series().diff().median()
    if pd.notna(step) and step != pd.Timedelta("5min"):
        # the models were trained on 5-min readings, so other intervals are averaged / spread to 5 min
        grid = readings.index.floor("5min")
        frame = readings.assign(_t=grid)
        if truth is not None:
            frame["_truth"] = truth
        frame = frame.groupby("_t").mean(numeric_only=True)
        frame = frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq="5min"))
        readings = frame[SENSORS]
        truth = frame["_truth"].values if truth is not None else None
        events = None
        notes.append(f"Readings were {step} apart; they were converted to the 5-minute interval the models use.")
    if len(readings) < MIN_ROWS:
        raise ValueError(f"need at least {MIN_ROWS} readings (3 hours at 5-min intervals); the file has {len(readings)}")
    if len(readings) > MAX_ROWS:
        raise ValueError(f"at most {MAX_ROWS} readings (120 days) per file; the file has {len(readings)}")
    gaps = float(readings.isna().mean().mean())
    if gaps > 0:
        notes.append(f"{100 * gaps:.1f}% of the values are missing; short gaps were filled as in the live system.")
    return readings, truth, events, notes


def simulated(days, seed, noise=1.0, ph_drift=0.0):
    """A new simulated data set (training used seed 42), as (readings, truth, events, notes)."""
    true, measured, labels, _, _ = simulate(days=days, seed=seed)
    notes = [f"New simulation: {days} days, seed {seed} (the models were trained on seed 42)."]
    if noise > 1:
        rng = np.random.default_rng(seed + 1)
        for s in SENSORS:
            measured[s] = measured[s] + rng.normal(0, NOISE[s] * np.sqrt(noise ** 2 - 1), len(measured))
        notes.append(f"Sensor noise raised to {noise:g}x.")
    if ph_drift:
        measured["ph"] = measured["ph"] + ph_drift
        notes.append(f"pH probe reads {ph_drift:+g} off (calibration drift).")
    return measured, labels["cwqi_true"].values, labels["event_type"].values, notes


def lead_times(y, pred_level, threshold, level):
    """Minutes between the first warning and the true crossing of `threshold`, for every onset.

    An onset = the true index crosses the threshold after staying below it for at least 3 h
    (as in the paper). Missed = no warning from 3 h before to 6 h after the crossing.
    """
    leads, missed = [], 0
    above = y >= threshold
    for t in np.where(above[1:] & ~above[:-1])[0] + 1:
        if t < 3 * HOUR or above[t - 3 * HOUR:t].any():
            continue
        lo, hi = t - 3 * HOUR, min(len(y), t + 6 * HOUR)
        hits = np.where(pred_level[lo:hi] >= level)[0]
        if len(hits):
            leads.append((t - (lo + hits[0])) * 5)
        else:
            missed += 1
    return leads, missed


def _runs(mask):
    """(start, end) of each run of True, end exclusive."""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def _num(v):
    return None if v is None or not np.isfinite(v) else float(v)


def analyse(monitor, readings, forecaster="lstm", truth=None, events=None):
    """Assess every reading of a data set. Returns (report for the page, per-reading result table)."""
    if forecaster == "xgb" and monitor.xgb_forecaster is None:
        raise ValueError("XGBoost forecaster not trained")
    n = len(readings)
    clean = fill_gaps(readings[SENSORS].astype(float))
    if clean.isna().any().any():
        raise ValueError("a sensor column has no numeric values at all")
    feats = build_features(clean)
    X = feats[FEATURE_COLUMNS].values
    base = base_index(feats)
    contrib = base_contributions(feats)

    estimate = monitor.estimator.predict(X, base)                                  # eq. (3)
    first = (XGB_MIN_HISTORY if forecaster == "xgb" else WINDOW) - 1
    forecast = np.repeat(estimate[:, None], len(HORIZON_MIN), axis=1)              # too little history: no change
    rows = np.arange(first, n)
    if forecaster == "xgb":
        forecast[rows] = monitor.xgb_forecaster.predict(feats, base, rows)
    else:
        forecast[rows] = monitor.forecaster.predict(smoothed(feats), base, step_numbers(readings.index), rows)
    forecast_max = forecast.max(1)
    risk = np.maximum(estimate, forecast_max)                                      # eq. (6), lambda = 0
    level = risk_level(risk)
    current = risk_level(estimate)

    score = monitor.anomaly.score(anomaly_features(clean))
    flags = score >= monitor.anomaly.tau
    evidence = pd.Series(flags.astype(float)).rolling(6, min_periods=1).mean().values    # eq. (5)

    # every Warning/Alert episode, with the SHAP root cause at its peak (eq. 4)
    episodes = []
    runs = []
    for a, b in _runs(level >= 1):                 # Safe for less than 30 min in between = same episode
        if runs and a - runs[-1][1] < 6:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    peaks = [a + int(np.argmax(risk[a:b])) for a, b in runs]
    if peaks:
        att = monitor.estimator.attribute(X[peaks], contrib[peaks])
    for k, ((a, b), p) in enumerate(zip(runs, peaks)):
        ranked = np.argsort(-att[k])
        cause = SENSORS[ranked[0]]
        crossed = np.where(current[a:b] >= 1)[0]
        ep = {
            "start": readings.index[a], "end": readings.index[b - 1], "minutes": int((b - a) * 5),
            "level": LEVELS[int(level[a:b].max())], "peak_cwqi": float(estimate[a:b].max()),
            "peak_time": readings.index[p], "root_cause": cause, "root_cause_name": SPEC[cause]["name"],
            "contributors": [SPEC[SENSORS[j]]["name"] for j in ranked[:3] if att[k][j] > 0.5],
            # how long the forecast-based warning came before the estimate itself reached 50
            "early_minutes": int(crossed[0] * 5) if len(crossed) else None,
        }
        if events is not None:
            kinds = [e for e in pd.unique(events[a:b]) if e != "none"]
            ep["true_event"] = EVENT_LABEL.get(kinds[0], kinds[0]) if kinds else None
        episodes.append(ep)

    table = pd.DataFrame({
        "cwqi_estimate": estimate, "cwqi_formula_raw": cwqi(clean),
        **{f"forecast_{m}min": forecast[:, j] for j, m in enumerate(HORIZON_MIN)},
        "risk_score": risk, "level": np.array(LEVELS)[level], "current_level": np.array(LEVELS)[current],
        "anomaly_score": score, "anomaly_flag": flags, "anomaly_evidence": evidence,
    }, index=readings.index)
    table.index.name = "ts"
    table = pd.concat([readings[SENSORS], table], axis=1)
    if truth is not None:
        table["cwqi_true"] = truth

    counts = {name: int((level == i).sum()) for i, name in enumerate(LEVELS)}
    report = {
        "rows": n, "start": readings.index[0], "end": readings.index[-1], "days": round(n / (24 * HOUR), 1),
        "forecaster": forecaster, "levels": counts,
        "level_share": {k: v / n for k, v in counts.items()},
        "anomalies": int(flags.sum()), "episodes": episodes[::-1][:200], "n_episodes": len(episodes),
        "root_causes": pd.Series([e["root_cause_name"] for e in episodes]).value_counts().to_dict(),
        "chart": _chart(readings.index, estimate, forecast_max, truth, flags),
        "metrics": _metrics(truth, events, estimate, cwqi(clean), forecast, level, X, contrib, monitor)
        if truth is not None and np.isfinite(truth).sum() > MIN_ROWS else None,
    }
    return report, table


def _chart(index, estimate, forecast_max, truth, flags):
    """Down-sampled series for the page: the highest value in each bucket, so peaks stay visible."""
    n = len(index)
    size = max(1, int(np.ceil(n / CHART_POINTS)))
    out = []
    for a in range(0, n, size):
        b = min(n, a + size)
        p = {"t": index[a], "est": float(estimate[a:b].max()), "fc": float(forecast_max[a:b].max()),
             "anomaly": bool(flags[a:b].any())}
        if truth is not None:
            p["truth"] = _num(np.nanmax(truth[a:b])) if np.isfinite(truth[a:b]).any() else None
        out.append(p)
    return out


def _metrics(truth, events, estimate, raw, forecast, level, X, contrib, monitor):
    """Accuracy against the true index, as in evaluate_new_data.py."""
    y = np.asarray(truth, dtype=float)
    ok = np.isfinite(y)
    n = len(y)
    rmse = lambda t, p: float(np.sqrt(mean_squared_error(t, p)))
    hi = ok & (y >= 50)
    out = {
        "r2": float(r2_score(y[ok], estimate[ok])), "rmse": rmse(y[ok], estimate[ok]),
        "rmse_raw_formula": rmse(y[ok], raw[ok]),
        "rmse_contaminated": rmse(y[hi], estimate[hi]) if hi.any() else None,
    }
    rows = np.arange(WINDOW - 1, n - 12)
    rows = rows[ok[rows] & ok[rows + 12]]
    if len(rows):
        out["forecast_60min_rmse"] = rmse(y[rows + 12], forecast[rows, 2])
        out["persistence_60min_rmse"] = rmse(y[rows + 12], estimate[rows])
        worst = pd.Series(y[::-1]).rolling(13, min_periods=1).max().values[::-1]
        true_level = risk_level(worst)[rows]
        pred = level[rows]
        out["macro_f1"] = float(f1_score(true_level, pred, average="macro"))
        out["false_alarm_rate"] = float((pred[true_level == 0] >= 1).mean()) if (true_level == 0).any() else None
    yy = np.where(ok, y, 0)
    for name, thr, lv in (("warning", 50, 1), ("alert", 75, 2)):
        leads, missed = lead_times(yy, level, thr, lv)
        out[f"{name}_lead_min"] = float(np.mean(leads)) if leads else None
        out[f"{name}_onsets"] = len(leads) + missed
        out[f"{name}_missed"] = missed
    if events is not None:                         # root cause at the peak of every true event
        spans = [(a, b, events[a]) for a, b in _runs(np.asarray(events) != "none")]
        spans = [(a, b, e) for a, b, e in spans if e in EVENT_ROOT and ok[a:b].any()]
        if spans:
            peaks = [a + int(np.nanargmax(y[a:b])) for a, b, _ in spans]
            att = monitor.estimator.attribute(X[peaks], contrib[peaks])
            hits = sum(SENSORS[int(np.argmax(att[k]))] in EVENT_ROOT[e] for k, (_, _, e) in enumerate(spans))
            out["root_cause"] = f"{hits}/{len(spans)}"
    return out


def sample_csv():
    """A small example file: 4 simulated days (seed 99) with the optional truth columns."""
    measured, truth, events, _ = simulated(4, 99)
    df = measured.round(3)
    df.index.name = "ts"
    df["cwqi_true"] = np.round(truth, 2)
    df["event_type"] = events
    return df.to_csv()


class ResultStore:
    """Keeps the last few result tables so the page can download them as CSV."""

    def __init__(self, size=5):
        self.size, self.items = size, {}

    def add(self, table):
        rid = uuid.uuid4().hex[:12]
        self.items[rid] = table
        while len(self.items) > self.size:
            self.items.pop(next(iter(self.items)))
        return rid

    def csv(self, rid):
        table = self.items.get(rid)
        return None if table is None else table.round(4).to_csv()
