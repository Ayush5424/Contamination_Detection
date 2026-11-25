"""Sensor feeder agent: keeps publishing fresh cooling-water readings over MQTT, forever.

No CSV file: every reading is generated on the fly by wq_model.live_stream.PlantStream (the
simulator's process equations, stepped one 5-min reading at a time).  A real plant replaces this
agent with its sensor gateway, which publishes the same JSON message to the same topic:
    {"ts": "2026-11-09T12:05:00", "device": "loop1", "temp": 29.9, "ph": 7.99, "cond": 1868,
     "do": 7.4, "frc": 0.62, "turb": 1.1, "inhib": 4.8}

Plant time: every reading is 5 minutes of plant time (the sampling interval the models were
trained on).  The clock starts at the current time.  With --interval 300 one reading arrives
every 5 real minutes, so plant time = wall-clock time; a shorter interval plays the plant faster
for a demo.  On start the agent first sends the last 4 h (backdated, marked "backfill": stored and
assessed but no alerts), so the models have a full history window at once.

Commands on the control topic (the dashboard sends them):
    {"cmd": "inject", "type": "biofouling", "severity": 1.5, "ramp_hours": 3}
    {"cmd": "clear"} | {"cmd": "pause"} | {"cmd": "resume"} | {"cmd": "interval", "seconds": 0.5}
    {"cmd": "auto_events", "on": false}
"""
import json
import threading
import time

import pandas as pd
import paho.mqtt.client as mqtt

from wq_model.live_stream import PlantStream
from wq_model.pipeline import HISTORY
from wq_model.simulator import DT_MIN, EVENT_TYPES

from . import config

WARMUP = HISTORY                                  # backdated readings sent first (fills the model window)
INGEST_WAIT = 180                                 # max seconds to wait for the ingestion worker before the warm-up


class FeederAgent:
    def __init__(self, interval=1.0, device=config.DEVICE, seed=None, warmup=WARMUP, log=print):
        self.interval, self.device, self.log = max(0.05, float(interval)), device, log
        start = pd.Timestamp.now().floor(f"{DT_MIN}min") - pd.Timedelta(minutes=DT_MIN * (warmup - 1))
        self.stream = PlantStream(seed=seed, start=start)
        self.warmup_left = warmup
        self.sent, self.paused, self.last_ts, self.last_sent = 0, False, None, None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sensor-feeder-{device}")
        self.client.on_connect = self._on_connect
        self.client.message_callback_add(config.topic("control", device), self._on_control)
        self.ingest_online = threading.Event()
        self.client.message_callback_add(config.INGEST_STATUS_TOPIC, self._on_ingest_status)

    # ------------------------------------------------------------ MQTT
    def _on_connect(self, client, *_):
        client.subscribe(config.topic("control", self.device), qos=1)
        client.subscribe(config.INGEST_STATUS_TOPIC, qos=1)
        self.publish_status()

    def _on_ingest_status(self, client, _, msg):
        if msg.payload == b"online":
            self.ingest_online.set()
        else:
            self.ingest_online.clear()

    def _on_control(self, client, _, msg):
        try:
            cmd = json.loads(msg.payload)
            with self.lock:
                c = cmd.get("cmd")
                if c == "inject" and cmd.get("type") in EVENT_TYPES:
                    self.stream.inject(cmd["type"], float(cmd.get("severity", 1.5)), float(cmd.get("ramp_hours", 3)))
                elif c == "clear":
                    self.stream.clear()
                elif c in ("pause", "resume"):
                    self.paused = c == "pause"
                elif c == "interval":
                    self.interval = max(0.05, float(cmd["seconds"]))
                elif c == "auto_events":
                    self.stream.auto_events = bool(cmd.get("on", True))
                else:
                    return                            # e.g. "forecaster" is for the ingestion service
            self.log(f"feeder control: {cmd}")
            self.publish_status()
        except Exception as exc:
            self.log(f"feeder: bad control message: {exc}")

    def status(self):
        manual = self.stream.manual
        return dict(mode="live", agent="feeder", device=self.device, sent=self.sent, paused=self.paused,
                    interval=self.interval, warming_up=self.warmup_left > 0,
                    plant_time=self.last_ts.isoformat() if self.last_ts is not None else None,
                    last_sent=self.last_sent, auto_events=self.stream.auto_events,
                    injection=(dict(type=manual["type"], severity=manual["severity"],
                                    clearing=manual["clearing"] is not None) if manual else None),
                    events=self.stream.active_event())

    def publish_status(self):
        self.client.publish(config.topic("status", self.device), json.dumps(self.status()), qos=1, retain=True)

    # ------------------------------------------------------------ loop
    def next_message(self):
        with self.lock:
            ts, values, truth, _ = self.stream.next()
        msg = {"ts": ts.isoformat(), "device": self.device}
        msg.update({s: (None if v != v else round(float(v), 4)) for s, v in values.items()})
        msg["cwqi_true"] = round(truth, 4)                  # simulation only: draws the grey ground-truth line
        return ts, msg

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.client.connect(config.MQTT_HOST, config.MQTT_PORT, keepalive=30)
                break
            except OSError:
                self.log(f"feeder: waiting for the MQTT broker at {config.MQTT_HOST}:{config.MQTT_PORT} ...")
                self.stop_event.wait(2)
        self.client.loop_start()
        if not self.ingest_online.wait(INGEST_WAIT) and not self.stop_event.is_set():
            self.log("feeder: no ingestion service reported online; publishing anyway")
        self.log(f"feeder: publishing to {config.topic('readings', self.device)}, "
                 f"{self.warmup_left} warm-up readings, then one every {self.interval} s")
        while not self.stop_event.is_set():
            if self.paused and self.warmup_left <= 0:
                self.stop_event.wait(0.2)
                continue
            ts, msg = self.next_message()
            if self.warmup_left > 0:
                msg["backfill"] = True                   # past readings: stored and assessed, but no alerts
            self.client.publish(config.topic("readings", self.device), json.dumps(msg), qos=1)
            self.sent += 1
            self.last_ts, self.last_sent = ts, time.time()
            if self.warmup_left > 0:
                self.warmup_left -= 1
                if self.warmup_left == 0:
                    self.publish_status()
                self.stop_event.wait(0.02)
                continue
            if self.sent % 5 == 0:
                self.publish_status()
            self.stop_event.wait(self.interval)
        self.client.loop_stop()
        self.client.disconnect()

    def start(self):
        self.thread = threading.Thread(target=self.run, name="sensor-feeder", daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
