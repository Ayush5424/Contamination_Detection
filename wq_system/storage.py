"""Storage layer: sensor readings, model assessments and alerts.

SQL tables through SQLAlchemy, so the same code runs on SQLite (default, no install)
and MySQL (set DATABASE_URL).  If INFLUX_URL is set, every reading and assessment is
also written to InfluxDB as a time series.
"""
import json

import pandas as pd
from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, Text,
                        create_engine, delete, event, func, insert, select)

from wq_model.cwqi import SENSORS

from . import config

meta = MetaData()

readings = Table(
    "readings", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("device", String(64), index=True),
    Column("ts", DateTime, index=True),
    *[Column(s, Float, nullable=True) for s in SENSORS],
    Column("received_at", DateTime, server_default=func.now()),
)

assessments = Table(
    "assessments", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("reading_id", Integer, index=True),
    Column("device", String(64), index=True),
    Column("ts", DateTime, index=True),
    Column("cwqi", Float), Column("raw_cwqi", Float),
    Column("forecast_15", Float), Column("forecast_30", Float), Column("forecast_60", Float),
    Column("forecast_max", Float), Column("forecaster", String(16)),
    Column("risk_score", Float), Column("level", String(16)), Column("current_level", String(16)),
    Column("anomaly_score", Float), Column("anomaly_flag", Boolean), Column("anomaly_evidence", Float),
    Column("root_cause", String(16)), Column("root_cause_name", String(64)),
    Column("attribution", Text),
    Column("true_cwqi", Float, nullable=True),                  # simulation only: the ground truth
)

alerts = Table(
    "alerts", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("device", String(64), index=True),
    Column("ts", DateTime),
    Column("kind", String(16)),                                  # level or anomaly
    Column("level", String(16)),
    Column("title", String(200)),
    Column("message", Text),
    Column("cwqi", Float), Column("forecast_max", Float),
    Column("root_cause_name", String(64)),
    Column("channels", String(200)),
    Column("created_at", DateTime, server_default=func.now()),
)


class Store:
    def __init__(self, url=config.DATABASE_URL):
        self.engine = create_engine(url, pool_pre_ping=True)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _wal(conn, _):                                   # readers and one writer at the same time
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
        meta.create_all(self.engine)
        self.influx = InfluxMirror() if config.INFLUX_URL else None

    # ---------------------------------------------------------------- writes
    def add_reading(self, device, ts, values):
        row = {"device": device, "ts": ts, **{s: _num(values.get(s)) for s in SENSORS}}
        with self.engine.begin() as c:
            rid = c.execute(insert(readings).values(**row)).inserted_primary_key[0]
        if self.influx:
            self.influx.write("reading", device, ts, {s: row[s] for s in SENSORS if row[s] is not None})
        return rid

    def add_assessment(self, reading_id, device, ts, res, true_cwqi=None):
        fc = list(res["forecast"].values())
        row = dict(reading_id=reading_id, device=device, ts=ts, cwqi=res["cwqi"], raw_cwqi=res["raw_cwqi"],
                   forecast_15=fc[0], forecast_30=fc[1], forecast_60=fc[2], forecast_max=res["forecast_max"],
                   forecaster=res["forecaster"], risk_score=res["risk_score"], level=res["level"],
                   current_level=res["current_level"], anomaly_score=res["anomaly_score"],
                   anomaly_flag=res["anomaly_flag"], anomaly_evidence=res["anomaly_evidence"],
                   root_cause=res["root_cause"], root_cause_name=res["root_cause_name"],
                   attribution=json.dumps(res["attribution"]), true_cwqi=_num(true_cwqi))
        with self.engine.begin() as c:
            aid = c.execute(insert(assessments).values(**row)).inserted_primary_key[0]
        if self.influx:
            self.influx.write("assessment", device, ts,
                              {k: float(row[k]) for k in ("cwqi", "forecast_max", "risk_score", "anomaly_evidence")},
                              tags={"level": row["level"]})
        return aid

    def add_alert(self, alert):
        with self.engine.begin() as c:
            return c.execute(insert(alerts).values(**alert)).inserted_primary_key[0]

    def clear(self):
        with self.engine.begin() as c:
            for t in (alerts, assessments, readings):
                c.execute(delete(t))

    # ---------------------------------------------------------------- reads
    def recent_readings(self, device, n):
        q = select(readings).where(readings.c.device == device).order_by(readings.c.id.desc()).limit(n)
        df = pd.read_sql(q, self.engine)
        return df.iloc[::-1].reset_index(drop=True)

    def table(self, name, limit=100, device=None, after_id=None):
        t = meta.tables[name]
        q = select(t)
        if device:
            q = q.where(t.c.device == device)
        if after_id is not None:
            q = q.where(t.c.id > after_id)
        df = pd.read_sql(q.order_by(t.c.id.desc()).limit(limit), self.engine)
        return df.iloc[::-1].reset_index(drop=True)

    def latest(self, device=None):
        """Latest assessment joined with its reading, in the same shape as WaterQualityMonitor.assess()."""
        a = self.table("assessments", 1, device)
        if a.empty:
            return None
        a = a.iloc[0]
        with self.engine.connect() as c:
            r = c.execute(select(readings).where(readings.c.id == int(a["reading_id"]))).mappings().first()
        return to_result(a, r)

    def counts(self):
        with self.engine.connect() as c:
            return {name: c.execute(select(func.count()).select_from(meta.tables[name])).scalar()
                    for name in ("readings", "assessments", "alerts")}


def to_result(a, reading=None):
    return {
        "id": int(a["id"]), "device": a["device"], "time": pd.Timestamp(a["ts"]),
        "cwqi": a["cwqi"], "raw_cwqi": a["raw_cwqi"],
        "forecast": {"15 min": a["forecast_15"], "30 min": a["forecast_30"], "60 min": a["forecast_60"]},
        "forecast_max": a["forecast_max"], "forecaster": a["forecaster"],
        "risk_score": a["risk_score"], "level": a["level"], "current_level": a["current_level"],
        "anomaly_score": a["anomaly_score"], "anomaly_flag": bool(a["anomaly_flag"]),
        "anomaly_evidence": a["anomaly_evidence"],
        "root_cause": a["root_cause"], "root_cause_name": a["root_cause_name"],
        "attribution": json.loads(a["attribution"]),
        "true_cwqi": a["true_cwqi"] if pd.notna(a["true_cwqi"]) else float("nan"),
        "reading": {s: (reading[s] if reading is not None and reading[s] is not None else float("nan"))
                    for s in SENSORS},
    }


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else v                                 # NaN -> NULL


class InfluxMirror:
    """Writes points to InfluxDB 2.x (measurement = reading or assessment, tag = device)."""

    def __init__(self):
        from influxdb_client import InfluxDBClient
        from influxdb_client.client.write_api import SYNCHRONOUS
        self.client = InfluxDBClient(url=config.INFLUX_URL, token=config.INFLUX_TOKEN, org=config.INFLUX_ORG)
        self.api = self.client.write_api(write_options=SYNCHRONOUS)

    def write(self, measurement, device, ts, fields, tags=None):
        from influxdb_client import Point
        p = Point(measurement).tag("device", device).time(pd.Timestamp(ts).to_pydatetime())
        for k, v in (tags or {}).items():
            p = p.tag(k, v)
        for k, v in fields.items():
            p = p.field(k, float(v))
        try:
            self.api.write(bucket=config.INFLUX_BUCKET, record=p)
        except Exception as exc:                                 # the SQL store stays the source of truth
            print("InfluxDB write failed:", exc)
