"""Alerts and notifications (the "Alerts and notification" block of Fig. 1).

AlertPolicy decides when to notify:
  * the risk level rises (Safe -> Warning -> Alert) - at once - or falls and stays lower for
    30 min, so a value hovering around a limit does not send a message every 5 minutes, and
  * a new burst of anomalies starts: the evidence a_t rises to 0.5 or more (a separate channel,
    because lambda = 0), at most once per hour.
Notifier sends each alert on every configured channel, in a background thread:
  log file (always), desktop pop-up, e-mail (SMTP), phone push (ntfy.sh) and a generic webhook.
"""
import json
import os
import platform
import smtplib
import threading
import urllib.request
from email.message import EmailMessage

from wq_model.guidance import ACTIONS

from . import config

RANK = {"Safe": 0, "Warning": 1, "Alert": 2}
CLEAR_AFTER = 6                                # readings (30 min) at a lower level before "it is lower now"
ANOMALY_EVIDENCE = 0.5
ANOMALY_COOLDOWN = 12                          # readings (1 h) between two anomaly notifications


class AlertPolicy:
    def __init__(self):
        self.level = {}
        self.lower_for = {}
        self.evidence = {}
        self.since_anomaly = {}

    def check(self, device, ts, res):
        out = []
        prev = self.level.get(device, "Safe")
        lower = RANK[res["level"]] < RANK[prev]
        self.lower_for[device] = self.lower_for.get(device, 0) + 1 if lower else 0
        if RANK[res["level"]] > RANK[prev] or self.lower_for[device] >= CLEAR_AFTER:
            out.append(level_alert(device, ts, res, prev))
            self.level[device] = res["level"]
            self.lower_for[device] = 0

        wait = self.since_anomaly.get(device, ANOMALY_COOLDOWN)
        rising = self.evidence.get(device, 0.0) < ANOMALY_EVIDENCE <= res["anomaly_evidence"]
        self.evidence[device] = res["anomaly_evidence"]
        if rising and wait >= ANOMALY_COOLDOWN:
            out.append(anomaly_alert(device, ts, res))
            wait = 0
        self.since_anomaly[device] = wait + 1
        return out


def level_alert(device, ts, res, prev):
    level = res["level"]
    if level == "Safe":
        title = f"[{device}] Water quality back to SAFE"
        msg = f"CWQI {res['cwqi']:.1f}; the risk has cleared (was {prev})."
    else:
        early = res["current_level"] == "Safe"
        title = f"[{device}] {level.upper()}: {'early warning' if early else 'cooling-water risk'}"
        msg = (f"CWQI now {res['cwqi']:.1f}, worst forecast for the next 60 min {res['forecast_max']:.1f} "
               f"({res['forecaster']}). Risk score {res['risk_score']:.1f}.\n"
               + ("The water is still acceptable, but the forecast predicts it will cross the limit "
                  "within the hour.\n" if early else "")
               + f"Most likely cause: {res['root_cause_name']}. {ACTIONS[res['root_cause']]}")
    return _alert(device, ts, "level", level, title, msg, res)


def anomaly_alert(device, ts, res):
    title = f"[{device}] Anomaly detected"
    msg = (f"The Isolation Forest flagged {res['anomaly_evidence'] * 100:.0f}% of the last 30 min of readings: "
           f"an abrupt change or a possible sensor fault. Largest contribution: {res['root_cause_name']}. "
           f"CWQI {res['cwqi']:.1f} ({res['level']}).")
    return _alert(device, ts, "anomaly", res["level"], title, msg, res)


def _alert(device, ts, kind, level, title, msg, res):
    return dict(device=device, ts=ts, kind=kind, level=level, title=title, message=msg, cwqi=res["cwqi"],
                forecast_max=res["forecast_max"], root_cause_name=res["root_cause_name"])


class Notifier:
    def __init__(self, log_path=None):
        self.log_path = log_path or f"{config.DATA_DIR}/alerts.log"
        self.channels = ["log"]
        if config.DESKTOP_NOTIFY and _desktop_available():
            self.channels.append("desktop")
        if config.SMTP_HOST and config.ALERT_EMAIL_TO:
            self.channels.append("email")
        if config.NTFY_TOPIC:
            self.channels.append("ntfy")
        if config.WEBHOOK_URL:
            self.channels.append("webhook")

    def send(self, alert):
        """Returns the channel list at once; the slow channels are sent in the background."""
        self._log(alert)
        threading.Thread(target=self._send_remote, args=(alert,), daemon=True).start()
        return ",".join(self.channels)

    def _log(self, alert):
        line = f"{alert['ts']}  {alert['title']}  |  {alert['message']}".replace("\n", " ")
        print("ALERT", line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _send_remote(self, alert):
        for name in self.channels[1:]:
            try:
                getattr(self, "_" + name)(alert)
            except Exception as exc:                             # one failed channel must not stop the others
                print(f"Notification via {name} failed: {exc}", flush=True)

    def _desktop(self, alert):
        from plyer import notification
        notification.notify(title=alert["title"][:63], message=alert["message"][:255],
                            app_name="Cooling-Water Monitor", timeout=10)

    def _email(self, alert):
        m = EmailMessage()
        m["Subject"], m["From"], m["To"] = alert["title"], config.ALERT_EMAIL_FROM, config.ALERT_EMAIL_TO
        m.set_content(f"{alert['message']}\n\nTime: {alert['ts']}\nDevice: {alert['device']}\n")
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=15) as s:
            if config.SMTP_TLS:
                s.starttls()
            if config.SMTP_USER:
                s.login(config.SMTP_USER, config.SMTP_PASSWORD)
            s.send_message(m)

    def _ntfy(self, alert):
        priority = {"Alert": "urgent", "Warning": "high"}.get(alert["level"], "default")
        req = urllib.request.Request(f"{config.NTFY_SERVER}/{config.NTFY_TOPIC}", data=alert["message"].encode(),
                                     headers={"Title": alert["title"].encode("ascii", "ignore").decode(),
                                              "Priority": priority, "Tags": "droplet"})
        urllib.request.urlopen(req, timeout=10).close()

    def _webhook(self, alert):
        body = json.dumps({**alert, "ts": str(alert["ts"]), "text": f"{alert['title']}\n{alert['message']}"})
        req = urllib.request.Request(config.WEBHOOK_URL, data=body.encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).close()


def _desktop_available():
    try:
        from plyer import notification  # noqa: F401
    except ImportError:
        return False
    return platform.system() == "Windows" or bool(os.environ.get("DISPLAY"))
