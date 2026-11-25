"""Built-in live system, started by the API (LIVE_AGENT=1, the default): one command is enough.

    MQTT broker (only if none is running) -> sensor feeder agent -> ingestion worker -> database
                                                                                   -> React live monitor

The parts still talk over real MQTT, exactly as when they run as separate processes
(broker.py, sensor_publisher.py, ingest_service.py), and the ingestion worker shares the API's
already loaded models.  Set LIVE_AGENT=0 when those separate processes (or Docker Compose) are used,
otherwise each reading would be ingested twice.
"""
import asyncio
import logging
import socket
import threading
import time

from . import config
from .feeder import FeederAgent
from .ingest import IngestWorker

log = logging.getLogger("wq.live")


def broker_running(host=config.MQTT_HOST, port=config.MQTT_PORT):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


class BrokerThread:
    """The amqtt broker of broker.py, in a background thread with its own event loop."""

    def __init__(self):
        self.error, self.ready = None, threading.Event()
        self.thread = threading.Thread(target=self._run, name="mqtt-broker", daemon=True)

    def _run(self):
        from amqtt.broker import Broker
        from broker import CONFIG
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            broker = Broker(CONFIG, loop=loop)
            loop.run_until_complete(broker.start())
        except Exception as exc:
            self.error = str(exc)
            self.ready.set()
            return
        self.ready.set()
        loop.run_forever()

    def start(self):
        self.thread.start()
        self.ready.wait(15)
        return self


class LiveSystem:
    def __init__(self, proc, interval=None):
        self.proc = proc
        self.interval = float(interval or config.FEEDER_INTERVAL)
        self.broker = self.feeder = self.ingest = None
        self.broker_mode, self.started_at = None, None
        self.messages = []

    def _log(self, msg):
        self.messages = (self.messages + [f"{time.strftime('%H:%M:%S')} {msg}"])[-20:]
        print(msg, flush=True)

    def start(self):
        local = config.MQTT_HOST in ("localhost", "127.0.0.1")
        if broker_running():
            self.broker_mode = "existing"
        elif local:
            self.broker = BrokerThread().start()
            self.broker_mode = "built-in" if not self.broker.error else "failed"
            if self.broker.error:
                self._log(f"live: could not start the MQTT broker: {self.broker.error}")
        else:
            self.broker_mode = "waiting"                 # remote broker not up yet: the clients keep retrying
        self._log(f"live: MQTT broker on {config.MQTT_HOST}:{config.MQTT_PORT} ({self.broker_mode})")
        self.ingest = IngestWorker(self.proc, client_id="api-ingest-worker", log=self._log).start()
        self.feeder = FeederAgent(self.interval, config.DEVICE, log=self._log).start()
        self.started_at = time.time()
        return self

    def stop(self):
        for part in (self.feeder, self.ingest):
            if part:
                part.stop()

    def info(self):
        return {"enabled": True, "broker": self.broker_mode,
                "feeder": self.feeder.status() if self.feeder else None,
                "ingest": self.ingest.info() if self.ingest else None,
                "log": self.messages[-5:]}
