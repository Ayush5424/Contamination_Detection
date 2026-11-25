"""REST API of the system (FastAPI).

    uvicorn api:app --port 8765          # dashboard at http://127.0.0.1:8765, documentation at /docs

The React dashboard (frontend/, built into frontend/dist) is served at / and uses the /dashboard endpoints.

POST /predict      assess a window of readings without storing anything (for other software)
POST /readings     ingest one reading over HTTP instead of MQTT (stored, assessed, alerts sent)
GET  /latest       latest assessment          GET /readings, /assessments, /alerts   history
GET  /metrics      test-set results            GET /parameters                      Table II
"""
import json
import os
from datetime import datetime
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from wq_model.cwqi import SENSORS, SPEC
from wq_model.pipeline import HISTORY
from wq_system import config
from wq_system.dashboard import make_router
from wq_system.live import LiveSystem
from wq_system.processor import Processor, step_numbers
from wq_system.storage import to_result

app = FastAPI(title="Cooling-Water Quality Monitoring API",
              description="AI-based water quality monitoring and contamination risk prediction for data "
                          "center cooling systems. Hybrid stacked ensemble + SHAP, LSTM/XGBoost forecast, "
                          "Isolation Forest and Safe/Warning/Alert risk fusion.",
              version="1.0")
proc = Processor()
# built-in live system (MQTT broker if needed + sensor feeder agent + ingestion worker), LIVE_AGENT=1 by default
live = LiveSystem(proc) if config.LIVE_AGENT else None
app.include_router(make_router(proc, live))


@app.on_event("startup")
def start_live_system():
    if live:
        live.start()


@app.on_event("shutdown")
def stop_live_system():
    if live:
        live.stop()
FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")


class Reading(BaseModel):
    ts: Optional[datetime] = Field(None, description="time of the reading; default: now")
    temp: Optional[float] = Field(None, description="Water temperature, degC")
    ph: Optional[float] = Field(None, description="pH")
    cond: Optional[float] = Field(None, description="Conductivity, uS/cm")
    do: Optional[float] = Field(None, description="Dissolved oxygen, mg/L")
    frc: Optional[float] = Field(None, description="Free residual chlorine, mg/L")
    turb: Optional[float] = Field(None, description="Turbidity, NTU")
    inhib: Optional[float] = Field(None, description="Corrosion inhibitor, mg/L")

    model_config = {"json_schema_extra": {"example": {
        "ts": "2026-06-30T00:05:00", "temp": 29.9, "ph": 7.99, "cond": 1868, "do": 7.4,
        "frc": 0.62, "turb": 1.1, "inhib": 4.8}}}


class PredictRequest(BaseModel):
    readings: list[Reading] = Field(..., min_length=1, max_length=HISTORY,
                                    description=f"the most recent readings, oldest first, 5 min apart "
                                                f"(up to {HISTORY}; at least 36 for the LSTM forecast)")
    forecaster: Literal["lstm", "xgb"] = "lstm"


def _clean(obj):
    """Makes numpy / NaN values JSON friendly."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def _records(df):
    return _clean(df.replace({np.nan: None}).to_dict(orient="records"))


@app.get("/info", tags=["info"])
def info():
    return {"service": app.title, "docs": "/docs", "database": proc.store.engine.url.get_backend_name(),
            "forecaster": proc.forecaster, "notification_channels": proc.notifier.channels}


@app.get("/health", tags=["info"])
def health():
    return {"status": "ok", "counts": proc.store.counts()}


@app.post("/predict", tags=["models"])
def predict(req: PredictRequest):
    """Runs every model on the newest reading of the window. Nothing is stored and no alert is sent."""
    if req.forecaster == "xgb" and proc.monitor.xgb_forecaster is None:
        raise HTTPException(400, "XGBoost forecaster not trained")
    rows = [r.model_dump() for r in req.readings]
    if all(r["ts"] is None for r in rows):
        end = pd.Timestamp.now().floor("5min")
        ts = pd.date_range(end=end, periods=len(rows), freq="5min")
    else:
        ts = pd.to_datetime([r["ts"] for r in rows])
    hist = pd.DataFrame([{s: r[s] for s in SENSORS} for r in rows], dtype=float)
    if hist.isna().all().any():
        missing = [s for s in SENSORS if hist[s].isna().all()]
        raise HTTPException(422, f"no values for {missing}")
    proc.monitor.flag_history = []
    res = proc.monitor.assess(hist, step_numbers(ts), req.forecaster)
    return _clean(res)


@app.post("/readings", tags=["ingestion"])
def ingest(reading: Reading, device: str = config.DEVICE):
    """Stores one reading, assesses it with the device's recent history and sends alerts if needed."""
    payload = reading.model_dump()
    payload["ts"] = payload["ts"].isoformat() if payload["ts"] else None
    try:
        res = proc.process(payload, device)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return _clean(res)


@app.get("/latest", tags=["history"])
def latest(device: Optional[str] = None):
    res = proc.store.latest(device)
    if res is None:
        raise HTTPException(404, "no assessments yet")
    return _clean(res)


@app.get("/readings", tags=["history"])
def get_readings(limit: int = Query(100, le=5000), device: Optional[str] = None):
    return _records(proc.store.table("readings", limit, device))


@app.get("/assessments", tags=["history"])
def get_assessments(limit: int = Query(100, le=5000), device: Optional[str] = None):
    df = proc.store.table("assessments", limit, device)
    return [_clean(to_result(a)) | {"reading_id": int(a["reading_id"])} for _, a in df.iterrows()]


@app.get("/alerts", tags=["history"])
def get_alerts(limit: int = Query(50, le=1000), device: Optional[str] = None):
    return _records(proc.store.table("alerts", limit, device))


@app.put("/forecaster/{name}", tags=["models"])
def set_forecaster(name: Literal["lstm", "xgb"]):
    """Forecaster used by POST /readings (the MQTT service has its own setting)."""
    try:
        proc.set_forecaster(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"forecaster": proc.forecaster}


@app.get("/metrics", tags=["info"])
def metrics():
    with open(os.path.join(config.MODEL_DIR, "metrics.json")) as f:
        return json.load(f)


@app.get("/parameters", tags=["info"])
def parameters():
    """The seven monitored parameters with their ideal values, limits and weights (Table II)."""
    return SPEC


class DashboardFiles(StaticFiles):
    """Built React app. index.html is re-checked on every visit, so a rebuilt dashboard shows at once;
    the hashed files in assets/ never change and can be cached."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if not path.startswith("assets/"):
            response.headers["Cache-Control"] = "no-cache"
        return response


# the React dashboard; registered last so every API route above takes precedence
if os.path.isdir(FRONTEND):
    app.mount("/", DashboardFiles(directory=FRONTEND, html=True), name="dashboard")
else:
    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    def no_frontend():
        return ("<h3>Dashboard not built yet.</h3><p>Run <code>npm install</code> and <code>npm run build</code> "
                "in the <code>frontend</code> folder, then restart the API. API documentation: "
                "<a href='/docs'>/docs</a></p>")
