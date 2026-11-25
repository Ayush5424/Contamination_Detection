"""Settings of the live system.

Values come from environment variables; any variable not set there is read from
settings.env in the project folder (KEY=value lines).  Docker Compose sets them directly.
"""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env_file(path=os.path.join(HERE, "settings.env")):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


def get(key, default=""):
    return os.environ.get(key, default)


MODEL_DIR = get("MODEL_DIR", os.path.join(HERE, "models"))
DATA_DIR = get("DATA_DIR", os.path.join(HERE, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# database: SQLite by default; MySQL with e.g. mysql+pymysql://user:password@host/dbname
DATABASE_URL = get("DATABASE_URL", "sqlite:///" + os.path.join(DATA_DIR, "monitor.db").replace("\\", "/"))

# optional time-series mirror (InfluxDB 2.x)
INFLUX_URL = get("INFLUX_URL")
INFLUX_TOKEN = get("INFLUX_TOKEN")
INFLUX_ORG = get("INFLUX_ORG", "jims")
INFLUX_BUCKET = get("INFLUX_BUCKET", "cooling_water")

# MQTT
MQTT_HOST = get("MQTT_HOST", "localhost")
MQTT_PORT = int(get("MQTT_PORT", "1883"))
DEVICE = get("DEVICE_ID", "loop1")
TOPIC_ROOT = get("MQTT_TOPIC_ROOT", "datacenter/cooling")


def topic(kind, device=DEVICE):
    """kind: readings, control, assessment, alerts or status."""
    return f"{TOPIC_ROOT}/{device}/{kind}"


INGEST_STATUS_TOPIC = f"{TOPIC_ROOT}-services/ingest"      # retained "online"/"offline" of the ingestion worker


FORECASTER = get("FORECASTER", "lstm")                       # lstm or xgb

# built-in live system of the API: MQTT broker (if none runs) + sensor feeder agent + ingestion worker.
# Set 0 when broker.py / sensor_publisher.py / ingest_service.py run as separate processes (or in Docker).
LIVE_AGENT = get("LIVE_AGENT", "1") == "1"
FEEDER_INTERVAL = float(get("FEEDER_INTERVAL", "1"))           # real seconds per 5-min reading (300 = real time)

# notifications
DESKTOP_NOTIFY = get("DESKTOP_NOTIFY", "1") == "1"
SMTP_HOST = get("SMTP_HOST")
SMTP_PORT = int(get("SMTP_PORT", "587"))
SMTP_USER = get("SMTP_USER")
SMTP_PASSWORD = get("SMTP_PASSWORD")
SMTP_TLS = get("SMTP_TLS", "1") == "1"
ALERT_EMAIL_FROM = get("ALERT_EMAIL_FROM", SMTP_USER)
ALERT_EMAIL_TO = get("ALERT_EMAIL_TO")
NTFY_SERVER = get("NTFY_SERVER", "https://ntfy.sh")
NTFY_TOPIC = get("NTFY_TOPIC")
WEBHOOK_URL = get("WEBHOOK_URL")
