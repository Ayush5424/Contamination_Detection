"""Local MQTT broker for running the live system without Docker.

    python broker.py              # listens on port 1883

In Docker Compose the Eclipse Mosquitto broker is used instead; both speak standard MQTT,
so real sensor gateways can publish to either one.
"""
import asyncio
import logging

from amqtt.broker import Broker

from wq_system import config

CONFIG = {
    "listeners": {"default": {"type": "tcp", "bind": f"0.0.0.0:{config.MQTT_PORT}"}},
    "plugins": {"amqtt.plugins.authentication.AnonymousAuthPlugin": {"allow_anonymous": True}},
}


async def main():
    broker = Broker(CONFIG)
    await broker.start()
    print(f"MQTT broker running on port {config.MQTT_PORT}. Press Ctrl+C to stop.", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
