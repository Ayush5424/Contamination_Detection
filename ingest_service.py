"""Ingestion service: MQTT subscriber -> database -> models -> alerts, as its own process.

    python ingest_service.py

Every reading published on datacenter/cooling/<device>/readings is stored, assessed by
all models (hybrid CWQI + SHAP, forecast, Isolation Forest, risk fusion), and the result
is stored and published on datacenter/cooling/<device>/assessment.  A level change or a
burst of anomalies sends a notification on every configured channel.

Send {"cmd": "forecaster", "name": "xgb"} (or "lstm") on the control topic to switch forecaster.

The API runs the same worker in a thread by default (LIVE_AGENT=1); run this separate service
only with LIVE_AGENT=0 for the API, otherwise every reading would be stored twice.
"""
from wq_system import config
from wq_system.ingest import IngestWorker
from wq_system.processor import Processor


def main():
    print("Loading the trained models ...", flush=True)
    proc = Processor()
    print(f"Database: {proc.store.engine.url.render_as_string(hide_password=True)}"
          f"{'  + InfluxDB ' + config.INFLUX_URL if config.INFLUX_URL else ''}", flush=True)
    print(f"Notification channels: {', '.join(proc.notifier.channels)}", flush=True)
    print(f"Forecaster: {proc.forecaster}", flush=True)
    IngestWorker(proc, log=lambda m: print(m, flush=True)).run()


if __name__ == "__main__":
    main()
