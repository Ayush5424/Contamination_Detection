"""Ingestion worker: MQTT subscriber -> database -> models -> alerts.

Used by ingest_service.py (its own process) and by the API's built-in live system (a thread
that shares the API's already loaded models).  Every reading on datacenter/cooling/<device>/readings
is stored, assessed by all models, stored again, and the result is published on
datacenter/cooling/<device>/assessment.  {"cmd": "forecaster", "name": "xgb"} on a control topic
switches the forecaster.
"""
import json
import queue
import threading
import time

import paho.mqtt.client as mqtt

from . import config

WILDCARD = f"{config.TOPIC_ROOT}/+/"


class IngestWorker:
    def __init__(self, proc, client_id="ingest-service", log=print):
        self.proc, self.log = proc, log
        self.inbox = queue.Queue()
        self.processed, self.rejected, self.last_error = 0, 0, None
        self.stop_event = threading.Event()
        self.thread = None
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_message = lambda c, u, msg: self.inbox.put((msg.topic, msg.payload))
        # the feeder waits for "online" before its warm-up burst, so no reading is lost while models load
        self.client.will_set(config.INGEST_STATUS_TOPIC, "offline", qos=1, retain=True)

    def _on_connect(self, client, *_):
        client.subscribe(WILDCARD + "readings", qos=1)
        client.subscribe(WILDCARD + "control", qos=1)
        client.publish(config.INGEST_STATUS_TOPIC, "online", qos=1, retain=True)
        self.log(f"ingest: subscribed to {WILDCARD}readings")

    def connect(self):
        while not self.stop_event.is_set():
            try:
                self.client.connect(config.MQTT_HOST, config.MQTT_PORT, keepalive=30)
                break
            except OSError:
                self.log(f"ingest: waiting for the MQTT broker at {config.MQTT_HOST}:{config.MQTT_PORT} ...")
                self.stop_event.wait(2)
        self.client.loop_start()

    def handle(self, topic, payload):
        device, kind = topic.split("/")[-2:]
        try:
            data = json.loads(payload)
            if kind == "control":
                if data.get("cmd") == "forecaster":
                    self.proc.set_forecaster(data["name"])
                    self.log(f"ingest: forecaster switched to {self.proc.forecaster}")
                return
            res = self.proc.process(data, device)
        except Exception as exc:
            self.rejected += 1
            self.last_error = f"{topic}: {exc}"
            self.log(f"ingest: rejected message on {topic}: {exc}")
            return
        out = {k: v for k, v in res.items() if k != "attribution"}
        out.update(ts=data.get("ts"), attribution=res["attribution"])
        self.client.publish(config.topic("assessment", device), json.dumps(out, default=float), qos=0)
        self.processed += 1
        if self.processed % 50 == 0:
            self.log(f"ingest: {self.processed} readings processed, latest level {res['level']}")

    def run(self):
        self.connect()
        while not self.stop_event.is_set():
            try:
                topic, payload = self.inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            self.handle(topic, payload)
        self.client.publish(config.INGEST_STATUS_TOPIC, "offline", qos=1, retain=True)
        self.client.loop_stop()
        self.client.disconnect()

    def start(self):
        self.thread = threading.Thread(target=self.run, name="ingest-worker", daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()

    def info(self):
        return {"processed": self.processed, "rejected": self.rejected, "backlog": self.inbox.qsize(),
                "last_error": self.last_error}
