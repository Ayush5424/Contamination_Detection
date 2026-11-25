"""Endpoints used by the React dashboard (frontend/), mounted under /dashboard by api.py.

Two data sources:
  * replay - the API replays the unseen 60-day test segment through the models itself
             (one shared session; the page calls /dashboard/replay/step on a timer);
  * live   - reads what the ingestion worker stores in the database, and sends commands
             to the sensor feeder agent over MQTT (inject, clear, pause, interval, auto_events,
             forecaster).  The API starts the feeder and the ingestion worker itself (wq_system/live.py).
"""
import json
import os
import threading
import time
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from wq_model.cwqi import ALERT_LIMIT, SAFE_LIMIT, SENSORS, SPEC
from wq_model.guidance import ACTIONS
from wq_model.pipeline import FORECASTERS, HISTORY
from wq_model.simulator import EVENT_LABEL, EVENT_TYPES, apply_event

from . import batch, config
from .storage import to_result

LOG_LENGTH = 288                                  # 24 h of 5-min readings on the chart


def clean(obj):
    """Makes numpy / NaN / timestamps JSON friendly."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def load_stream():
    df = pd.read_csv(os.path.join(config.MODEL_DIR, "demo_stream.csv"), index_col=0, parse_dates=True)
    df["event_type"] = df["event_type"].fillna("none")
    return df


def list_events(df):
    out, prev = [], "none"
    for i, t in enumerate(df["event_type"].values):
        if t != "none" and prev == "none":
            out.append((i, t))
        prev = t
    return out


def point(r, reading=None):
    """The few fields the charts need for one reading (CWQI chart and sensor trend lines)."""
    reading = r.get("reading") if reading is None else reading
    return {"time": r["time"], "cwqi": r["cwqi"], "true_cwqi": r["true_cwqi"], "level": r["level"],
            "reading": {s: reading.get(s) for s in SENSORS} if reading else None}


class ReplaySession:
    """Built-in replay of the test segment, assessed by the API's own models."""

    def __init__(self, proc, stream):
        self.proc, self.stream = proc, stream
        self.events = list_events(stream)
        self.lock = threading.Lock()
        self.alert_id = 0                         # keeps counting across resets, so the page sees new alerts
        self.reset()

    def reset(self, position=HISTORY):
        self.pos = max(HISTORY, min(position, len(self.stream) - 1))
        self.log, self.alerts, self.injection, self.last = [], [], None, None
        self.flags = []

    def reading_at(self, i):
        row = self.stream.iloc[i][SENSORS].to_dict()
        inj = self.injection
        if inj and i >= inj["start"]:
            ramp = min(1.0, (i - inj["start"] + 1) / inj["ramp"])
            row = apply_event(row, inj["type"], inj["severity"] * ramp)
        return row

    def step_once(self, forecaster):
        i = self.pos
        if i >= len(self.stream):
            return False
        lo = i - HISTORY + 1
        history = pd.DataFrame([self.reading_at(j) for j in range(lo, i + 1)], index=self.stream.index[lo:i + 1])
        monitor = self.proc.monitor
        with self.proc.lock:                      # the monitor is shared with POST /readings
            saved, monitor.flag_history = monitor.flag_history, self.flags
            try:
                res = monitor.assess(history, self.stream["step"].values[lo:i + 1], forecaster)
            finally:
                self.flags, monitor.flag_history = monitor.flag_history, saved
        res["time"] = self.stream.index[i]
        res["true_cwqi"] = float(self.stream["cwqi_true"].iloc[i]) if self.injection is None else np.nan
        res["reading"] = history.iloc[-1].to_dict()
        prev = self.last["level"] if self.last else "Safe"
        self.last = res
        self.log = (self.log + [point(res)])[-LOG_LENGTH:]
        if res["level"] != prev:
            self.alert_id += 1
            cause = res["root_cause_name"] if res["level"] != "Safe" else None
            self.alerts.insert(0, {
                "id": self.alert_id, "time": res["time"], "kind": "level", "level": res["level"],
                "title": f"{res['level'].upper()} at {res['time']:%H:%M}" + (f": {cause}" if cause else ""),
                "cwqi": res["cwqi"], "forecast_max": res["forecast_max"], "root_cause_name": cause})
            self.alerts = self.alerts[:50]
        self.pos = i + 1
        return True

    def state(self):
        return {"source": "replay", "last": self.last, "log": self.log, "alerts": self.alerts[:8],
                "injection": self.injection, "position": self.pos, "total": len(self.stream),
                "finished": self.pos >= len(self.stream)}


class GatewayLink:
    """MQTT client that sends commands to the sensor gateway and keeps its latest status."""

    def __init__(self):
        import paho.mqtt.client as mqtt
        self.status, self.connected, self.error = None, False, None
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"dashboard-api-{os.getpid()}")

        def on_connect(c, *_):
            self.connected = True
            c.subscribe(config.topic("status"), qos=1)

        def on_disconnect(*_):
            self.connected = False

        def on_message(_, __, msg):
            if msg.topic != config.topic("status"):
                return
            try:
                self.status = json.loads(msg.payload)
            except ValueError:
                pass                              # never let a bad message stop the MQTT thread

        self.client.on_connect, self.client.on_disconnect = on_connect, on_disconnect
        self.client.on_message = on_message
        try:
            self.client.connect(config.MQTT_HOST, config.MQTT_PORT, keepalive=30)
            self.client.loop_start()
        except OSError as exc:
            self.error = f"MQTT broker not reachable at {config.MQTT_HOST}:{config.MQTT_PORT} ({exc})"

    def send(self, cmd):
        if self.error:
            raise HTTPException(503, self.error)
        self.client.publish(config.topic("control"), json.dumps(cmd), qos=1)

    def info(self):
        return {"connected": self.connected, "error": self.error, "status": self.status}

    def close(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


class Injection(BaseModel):
    type: Literal[tuple(EVENT_TYPES)]
    severity: float = Field(1.5, ge=0.5, le=2.0)
    ramp_hours: float = Field(3, ge=1, le=6)


class GatewayCommand(BaseModel):
    cmd: Literal["inject", "clear", "pause", "resume", "interval", "auto_events", "forecaster"]
    type: Optional[str] = None
    severity: Optional[float] = None
    ramp_hours: Optional[float] = None
    seconds: Optional[float] = None
    on: Optional[bool] = None
    name: Optional[Literal["lstm", "xgb"]] = None


class NewDataFile(BaseModel):
    csv: str = Field(..., description="CSV text: a time column and the seven sensor columns")
    forecaster: Literal["lstm", "xgb"] = "lstm"


class NewSimulation(BaseModel):
    days: int = Field(30, ge=1, le=120)
    seed: int = Field(7, ge=0, le=1_000_000)
    noise: float = Field(1.0, ge=1.0, le=4.0, description="sensor noise, times the normal level")
    ph_drift: float = Field(0.0, ge=-1.0, le=1.0, description="pH probe calibration offset")
    forecaster: Literal["lstm", "xgb"] = "lstm"


def make_router(proc, live=None):
    router = APIRouter(prefix="/dashboard", tags=["dashboard"])
    stream = load_stream()
    replay = ReplaySession(proc, stream)
    link = {"gateway": None}

    def gateway():
        if link["gateway"] is None or link["gateway"].error:
            if link["gateway"]:
                link["gateway"].close()
            link["gateway"] = GatewayLink()
        return link["gateway"]

    def check_forecaster(name):
        if name == "xgb" and proc.monitor.xgb_forecaster is None:
            raise HTTPException(400, "XGBoost forecaster not trained")

    @router.get("/config")
    def dashboard_config():
        """Everything static the page needs: parameters, limits, guidance, events, forecasters."""
        return clean({
            "parameters": {s: SPEC[s] for s in SENSORS}, "sensors": SENSORS,
            "limits": {"warning": SAFE_LIMIT, "alert": ALERT_LIMIT},
            "actions": ACTIONS,
            "event_types": [{"key": t, "label": EVENT_LABEL[t]} for t in EVENT_TYPES],
            "events": [{"number": k + 1, "type": t, "label": EVENT_LABEL[t], "time": stream.index[i]}
                       for k, (i, t) in enumerate(replay.events)],
            "forecasters": [{"key": k, "label": v} for k, v in FORECASTERS.items()
                            if k == "lstm" or proc.monitor.xgb_forecaster is not None],
            "default_forecaster": proc.forecaster,
            "live_agent": live is not None,
        })

    # ------------------------------------------------------------ replay
    @router.get("/replay")
    def replay_state():
        with replay.lock:
            if replay.last is None:
                replay.step_once(proc.forecaster)
            return clean(replay.state())

    @router.post("/replay/step")
    def replay_step(n: int = 1, forecaster: Literal["lstm", "xgb"] = "lstm"):
        """Advance the replay by n readings (1 = 5 simulated minutes)."""
        check_forecaster(forecaster)
        with replay.lock:
            for _ in range(max(1, min(n, 24))):
                if not replay.step_once(forecaster):
                    break
            return clean(replay.state())

    @router.post("/replay/reset")
    def replay_reset():
        with replay.lock:
            replay.reset()
            replay.step_once(proc.forecaster)
            return clean(replay.state())

    @router.post("/replay/jump/{event}")
    def replay_jump(event: int, forecaster: Literal["lstm", "xgb"] = "lstm"):
        """Go to 6 h before a recorded test-period event (numbered from 1)."""
        if not 1 <= event <= len(replay.events):
            raise HTTPException(404, f"event must be 1-{len(replay.events)}")
        check_forecaster(forecaster)
        with replay.lock:
            replay.reset(replay.events[event - 1][0] - 72)
            replay.step_once(forecaster)
            return clean(replay.state())

    @router.post("/replay/inject")
    def replay_inject(inj: Injection):
        with replay.lock:
            replay.injection = dict(type=inj.type, severity=inj.severity,
                                    ramp=max(1, int(inj.ramp_hours * 12)), start=replay.pos)
            return clean(replay.state())

    @router.post("/replay/clear")
    def replay_clear():
        with replay.lock:
            replay.injection = None
            return clean(replay.state())

    # ------------------------------------------------------------ live
    @router.get("/live")
    def live_state(device: str = config.DEVICE):
        """Latest assessments and alerts of one device (default: the feeder's), plus the feeder status."""
        a = proc.store.table("assessments", LOG_LENGTH, device)
        rd = proc.store.table("readings", LOG_LENGTH + 10, device).set_index("id")
        al = proc.store.table("alerts", 8, device)
        alerts = [{"id": int(x["id"]), "time": pd.Timestamp(x["ts"]), "kind": x["kind"], "level": x["level"],
                   "title": x["title"], "cwqi": x["cwqi"], "forecast_max": x["forecast_max"],
                   "root_cause_name": x["root_cause_name"], "channels": x["channels"]}
                  for _, x in al.iloc[::-1].iterrows()]
        return clean({
            "source": "live",
            "last": proc.store.latest(device) if not a.empty else None,
            "log": [point(to_result(row), rd.loc[int(row["reading_id"]), SENSORS].to_dict()
                          if int(row["reading_id"]) in rd.index else {})
                    for _, row in a.iterrows()],
            "alerts": alerts,
            "gateway": gateway().info(),
            "agent": live.info() if live else {"enabled": False},
            "server_time": time.time(),
            "database": {"backend": proc.store.engine.url.get_backend_name(), "counts": proc.store.counts()},
        })

    @router.post("/gateway")
    def gateway_command(cmd: GatewayCommand):
        """Sends a command to the sensor feeder agent (or, for "forecaster", to the ingestion worker)."""
        if cmd.cmd == "inject" and cmd.type not in EVENT_TYPES:
            raise HTTPException(422, f"type must be one of {EVENT_TYPES}")
        if cmd.cmd == "forecaster":
            check_forecaster(cmd.name)
        gateway().send(cmd.model_dump(exclude_none=True))
        return {"sent": cmd.model_dump(exclude_none=True)}

    # ------------------------------------------------------------ new data (batch test)
    results = batch.ResultStore()

    def run_batch(data, forecaster, source):
        check_forecaster(forecaster)
        readings, truth, events, notes = data
        try:
            report, table = batch.analyse(proc.monitor, readings, forecaster, truth, events)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        report.update(source=source, notes=notes, result_id=results.add(table))
        return clean(report)

    @router.post("/newdata/csv")
    def newdata_csv(req: NewDataFile):
        """Runs the saved models (no retraining) on every reading of an uploaded CSV file."""
        try:
            data = batch.parse_csv(req.csv)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return run_batch(data, req.forecaster, "file")

    @router.post("/newdata/simulate")
    def newdata_simulate(req: NewSimulation):
        """Runs the saved models on a new simulation with another seed than training (42)."""
        data = batch.simulated(req.days, req.seed, req.noise, req.ph_drift)
        return run_batch(data, req.forecaster, "simulation")

    @router.get("/newdata/sample.csv")
    def newdata_sample():
        return Response(batch.sample_csv(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="sample_new_data.csv"'})

    @router.get("/newdata/result/{rid}.csv")
    def newdata_result(rid: str):
        text = results.csv(rid)
        if text is None:
            raise HTTPException(404, "result expired; run the test again")
        return Response(text, media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="results_{rid}.csv"'})

    @router.delete("/database")
    def clear_database():
        proc.store.clear()
        return {"counts": proc.store.counts()}

    return router
