"""One new reading -> stored, assessed by every model, stored again, alerts sent.

Used by both ingestion paths: the MQTT subscriber (ingest_service.py) and the REST API
(POST /readings in api.py).
"""
import threading

import numpy as np
import pandas as pd

from wq_model.cwqi import SENSORS, SPEC
from wq_model.pipeline import FORECASTERS, HISTORY, WaterQualityMonitor

from . import config
from .notifier import AlertPolicy, Notifier
from .storage import Store


def step_numbers(ts):
    """Absolute 5-min step of each timestamp; only the time of day (step mod 288) is used by the LSTM."""
    # as_unit: pandas 3 stores times in microseconds by default, asi8 must be nanoseconds here
    return (pd.DatetimeIndex(ts).as_unit("ns").asi8 // (300 * 10 ** 9)).astype(np.int64)


def parse_reading(payload):
    """Validates a reading message: {"ts": ISO time (optional), "temp": .., ..., "inhib": ..}."""
    if all(payload.get(s) is None for s in SENSORS):
        raise ValueError(f"reading has no value for any of the sensors {SENSORS}")
    ts = pd.Timestamp(payload.get("ts") or pd.Timestamp.now()).tz_localize(None).floor("s")
    values = {}
    for s in SENSORS:
        v = payload.get(s)
        values[s] = float(v) if v is not None else np.nan
    return ts, values


class Processor:
    def __init__(self, store=None, monitor=None, notifier=None, forecaster=config.FORECASTER):
        self.store = store or Store()
        self.monitor = monitor or WaterQualityMonitor(config.MODEL_DIR)
        self.notifier = notifier or Notifier()
        self.policy = AlertPolicy()
        self.forecaster = forecaster
        self.flags = {}                                          # anomaly flag history per device
        self.lock = threading.Lock()

    def set_forecaster(self, name):
        if name not in FORECASTERS:
            raise ValueError(f"forecaster must be one of {list(FORECASTERS)}")
        if name == "xgb" and self.monitor.xgb_forecaster is None:
            raise ValueError("XGBoost forecaster not trained")
        self.forecaster = name

    def process(self, payload, device=config.DEVICE):
        ts, values = parse_reading(payload)
        with self.lock:
            rid = self.store.add_reading(device, ts, values)
            hist = self.store.recent_readings(device, HISTORY)
            # only this run's recent past: after a restart or a gap the database still holds older
            # (or, after a faster demo run, later) readings, which must not be mixed into the window
            hist = hist[(hist["ts"] <= ts) & (hist["ts"] > ts - pd.Timedelta(minutes=5 * HISTORY))]
            values = hist[SENSORS].astype(float)
            for s in SENSORS:                                    # a sensor with no reading yet: assume its ideal value
                if values[s].isna().all():
                    values[s] = SPEC[s]["ideal"]
            self.monitor.flag_history = self.flags.get(device, [])
            res = self.monitor.assess(values, step_numbers(hist["ts"]), self.forecaster)
            self.flags[device] = self.monitor.flag_history
            self.store.add_assessment(rid, device, ts, res, payload.get("cwqi_true"))
            if payload.get("backfill"):                          # backdated history sent at start-up:
                self.policy.check(device, ts, res)               # the policy learns the state, no alert is sent
                return res
            for alert in self.policy.check(device, ts, res):
                alert["channels"] = self.notifier.send(alert)
                self.store.add_alert(alert)
        return res
