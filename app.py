"""Live monitoring dashboard.

    streamlit run app.py

Two data sources:
  * Built-in replay - replays the unseen 60-day test segment reading by reading through the
    trained models, with no other program needed.
  * Live (MQTT + database) - shows what the ingestion service stores in the database while the
    sensor gateway publishes over MQTT; the sidebar sends commands to the gateway over MQTT.
In both modes you can inject a contamination event and watch the system raise a Warning and an Alert.
"""
import base64
import io
import json
import os
import time
import wave

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from wq_model.cwqi import ALERT_LIMIT, SAFE_LIMIT, SENSORS, SPEC, sub_index
from wq_model.guidance import ACTIONS
from wq_model.pipeline import FORECASTERS, HISTORY, WaterQualityMonitor
from wq_model.simulator import EVENT_LABEL, EVENT_TYPES, apply_event

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

COLORS = {"Safe": "#1e8e3e", "Warning": "#e37400", "Alert": "#d93025"}
REPLAY, LIVE = "Built-in replay", "Live: MQTT sensors + database"

st.set_page_config(page_title="Cooling-Water Quality Monitor", page_icon="\U0001F4A7", layout="wide")


# ---------------------------------------------------------------- resources
@st.cache_resource
def load_monitor():
    return WaterQualityMonitor(MODELS)


@st.cache_data
def load_stream():
    df = pd.read_csv(os.path.join(MODELS, "demo_stream.csv"), index_col=0, parse_dates=True)
    df["event_type"] = df["event_type"].fillna("none")
    return df


@st.cache_data
def load_metrics():
    with open(os.path.join(MODELS, "metrics.json")) as f:
        return json.load(f)


@st.cache_resource
def load_store():
    from wq_system.storage import Store
    return Store()


@st.cache_resource
def mqtt_link():
    """MQTT client for sending commands to the gateway; keeps the gateway's latest status."""
    import paho.mqtt.client as mqtt
    from wq_system import config
    link = {"status": None, "connected": False, "error": None}
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"dashboard-{os.getpid()}")

    def on_connect(c, *_):
        link["connected"] = True
        c.subscribe(config.topic("status"), qos=1)

    def on_disconnect(*_):
        link["connected"] = False

    def on_message(_, __, msg):
        link["status"] = json.loads(msg.payload)

    client.on_connect, client.on_disconnect, client.on_message = on_connect, on_disconnect, on_message
    try:
        client.connect(config.MQTT_HOST, config.MQTT_PORT, keepalive=30)
        client.loop_start()
    except OSError as exc:
        link["error"] = f"MQTT broker not reachable at {config.MQTT_HOST}:{config.MQTT_PORT} ({exc})"
    link["client"] = client
    link["publish"] = lambda cmd: client.publish(config.topic("control"), json.dumps(cmd), qos=1)
    return link


def beep_wav():
    rate, t = 22050, np.arange(int(22050 * 0.35)) / 22050
    tone = (0.4 * np.sin(2 * np.pi * 880 * t) * np.minimum(1, (0.35 - t) * 20)).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((tone * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


BEEP = base64.b64encode(beep_wav()).decode()


def notify_in_app(title, level):
    """Pop-up in the page plus a short sound (the browser must have been clicked once)."""
    icon = {"Alert": "\U0001F6A8", "Warning": "⚠️", "Safe": "✅"}.get(level, "\U0001F514")
    st.toast(title, icon=icon)
    if st.session_state.get("sound", True) and level != "Safe":
        SOUND.markdown(f"<audio autoplay src='data:audio/wav;base64,{BEEP}'></audio>", unsafe_allow_html=True)


if not os.path.exists(os.path.join(MODELS, "estimator.joblib")):
    st.error("Trained models not found. Run `python train_models.py` first (about 10 minutes).")
    st.stop()

monitor = load_monitor()
stream = load_stream()


def list_events(df):
    out, prev = [], "none"
    for i, t in enumerate(df["event_type"].values):
        if t != "none" and prev == "none":
            out.append((i, t))
        prev = t
    return out


EVENTS = list_events(stream)
EVENT_LABELS = [f"{k + 1}. {EVENT_LABEL[t]}  ({stream.index[i]:%d %b %H:%M})" for k, (i, t) in enumerate(EVENTS)]


# ---------------------------------------------------------------- replay state
def reset(position=HISTORY):
    st.session_state.pos = max(HISTORY, position)
    st.session_state.log = []
    st.session_state.alerts = []
    st.session_state.injection = None
    st.session_state.last = None
    monitor.flag_history = []


if "pos" not in st.session_state:
    reset()
    st.session_state.running = False
    st.session_state.seen_alert_id = None


def reading_at(i):
    row = stream.iloc[i][SENSORS].to_dict()
    inj = st.session_state.injection
    if inj and i >= inj["start"]:
        ramp = min(1.0, (i - inj["start"] + 1) / inj["ramp"])
        row = apply_event(row, inj["type"], inj["severity"] * ramp)
    return row


def step_once():
    i = st.session_state.pos
    if i >= len(stream):
        st.session_state.running = False
        return
    lo = i - HISTORY + 1
    history = pd.DataFrame([reading_at(j) for j in range(lo, i + 1)], index=stream.index[lo:i + 1])
    res = monitor.assess(history, stream["step"].values[lo:i + 1], st.session_state.get("forecaster", "lstm"))
    res["time"] = stream.index[i]
    res["true_cwqi"] = float(stream["cwqi_true"].iloc[i]) if st.session_state.injection is None else np.nan
    res["reading"] = history.iloc[-1].to_dict()
    st.session_state.last = res
    st.session_state.log = (st.session_state.log + [res])[-288:]              # keep 24 h
    prev = st.session_state.log[-2]["level"] if len(st.session_state.log) > 1 else "Safe"
    if res["level"] != prev:
        st.session_state.alerts.insert(0, {
            "Time": res["time"].strftime("%d %b %H:%M"), "Level": res["level"],
            "CWQI now": round(res["cwqi"], 1), "Forecast (max 60 min)": round(res["forecast_max"], 1),
            "Main cause": res["root_cause_name"] if res["level"] != "Safe" else "-"})
        st.session_state.new_toast = (f"{res['level'].upper()} at {res['time']:%H:%M}"
                                      + (f": {res['root_cause_name']}" if res["level"] != "Safe" else ""),
                                      res["level"])
    st.session_state.pos = i + 1


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    source = st.radio("Data source", [REPLAY, LIVE], key="source",
                      help="Replay runs everything inside this page. Live needs the broker, the ingestion "
                           "service and the sensor gateway (Start Live System.bat or Docker Compose).")
    fc_options = [k for k in FORECASTERS if k == "lstm" or monitor.xgb_forecaster is not None]
    forecaster = st.selectbox("Forecaster", fc_options, format_func=lambda k: FORECASTERS[k], key="forecaster",
                              help="LSTM (Section III-E) or the XGBoost lag-feature forecaster (Section V).")
    st.toggle("Pop-up and sound on alerts", value=True, key="sound")
    st.divider()

    if source == REPLAY:
        st.header("Controls")
        c1, c2 = st.columns(2)

        def toggle_run():
            st.session_state.running = not st.session_state.running

        c1.button("▶ Start" if not st.session_state.running else "⏸ Pause", use_container_width=True,
                  on_click=toggle_run)
        c2.button("Step ⏭", use_container_width=True, on_click=step_once)
        speed = st.select_slider("Speed (simulated time per update)", ["5 min", "15 min", "30 min", "1 h"], "15 min")
        steps_per_tick = {"5 min": 1, "15 min": 3, "30 min": 6, "1 h": 12}[speed]
        show_truth = st.toggle("Show ground-truth CWQI", value=True,
                               help="The true index of the simulated water, which the models never see.")

        st.divider()
        st.subheader("Jump to a recorded event")
        st.selectbox("Contamination events in the test period", range(len(EVENTS)),
                     format_func=lambda k: EVENT_LABELS[k], key="event_choice")

        def jump():
            reset(EVENTS[st.session_state.event_choice][0] - 72)
            st.session_state.running = True

        st.button("Go to 6 h before this event", use_container_width=True, on_click=jump)

        st.divider()
        st.subheader("Inject a contamination event")
        st.selectbox("Fault type", EVENT_TYPES, format_func=lambda t: EVENT_LABEL[t], key="inj_type")
        st.slider("Severity", 0.5, 2.0, 1.5, 0.1, key="inj_sev")
        st.slider("Develops over (hours)", 1, 6, 3, key="inj_ramp")
        c3, c4 = st.columns(2)

        def inject():
            st.session_state.injection = dict(type=st.session_state.inj_type, severity=st.session_state.inj_sev,
                                              ramp=st.session_state.inj_ramp * 12, start=st.session_state.pos)
            st.session_state.running = True

        def clear_fault():
            st.session_state.injection = None

        c3.button("Inject", type="primary", use_container_width=True, on_click=inject)
        c4.button("Clear fault", use_container_width=True, on_click=clear_fault)
        if st.session_state.injection:
            inj = st.session_state.injection
            st.info(f"Injected: {EVENT_LABEL[inj['type']]}, severity {inj['severity']}")

        st.divider()

        def reset_all():
            reset()
            st.session_state.running = False

        st.button("Reset to start of test period", use_container_width=True, on_click=reset_all)

    else:
        link = mqtt_link()
        store = load_store()
        st.header("Sensor gateway (MQTT)")
        status = link["status"]
        if link["error"]:
            st.error(link["error"])
            st.button("Retry connection", on_click=mqtt_link.clear)
        elif not link["connected"]:
            st.warning("Connecting to the MQTT broker ...")
        elif status is None:
            st.warning("Connected to the broker, but the sensor gateway has not reported yet. "
                       "Start `python sensor_publisher.py`.")
        else:
            st.success(f"Gateway online - {'paused' if status['paused'] else 'publishing'} one reading every "
                       f"{status['interval']:g} s")
        send = link["publish"]

        c1, c2 = st.columns(2)
        paused = bool(status and status["paused"])
        c1.button("▶ Resume" if paused else "⏸ Pause", use_container_width=True,
                  on_click=lambda: send({"cmd": "resume" if paused else "pause"}))
        c2.button("⟳ Refresh", use_container_width=True)
        st.select_slider("Seconds between readings", [0.1, 0.25, 0.5, 1.0, 2.0, 5.0], 1.0, key="live_interval",
                         on_change=lambda: send({"cmd": "interval", "seconds": st.session_state.live_interval}))
        show_truth = st.toggle("Show ground-truth CWQI", value=True,
                               help="Sent by the simulated gateway only; a real plant has no ground truth.")
        auto = st.toggle("Auto-refresh", value=True)

        def send_forecaster():
            send({"cmd": "forecaster", "name": st.session_state.forecaster})

        st.button(f"Use {FORECASTERS[forecaster]} in the ingestion service", use_container_width=True,
                  on_click=send_forecaster)

        st.divider()
        st.subheader("Jump to a recorded event")
        st.selectbox("Contamination events in the test period", range(len(EVENTS)),
                     format_func=lambda k: EVENT_LABELS[k], key="event_choice")
        st.button("Go to 6 h before this event", use_container_width=True,
                  on_click=lambda: send({"cmd": "jump", "event": st.session_state.event_choice + 1}))

        st.divider()
        st.subheader("Inject a contamination event")
        st.selectbox("Fault type", EVENT_TYPES, format_func=lambda t: EVENT_LABEL[t], key="inj_type")
        st.slider("Severity", 0.5, 2.0, 1.5, 0.1, key="inj_sev")
        st.slider("Develops over (hours)", 1, 6, 3, key="inj_ramp")
        c3, c4 = st.columns(2)
        c3.button("Inject", type="primary", use_container_width=True,
                  on_click=lambda: send({"cmd": "inject", "type": st.session_state.inj_type,
                                         "severity": st.session_state.inj_sev,
                                         "ramp_hours": st.session_state.inj_ramp}))
        c4.button("Clear fault", use_container_width=True, on_click=lambda: send({"cmd": "clear"}))
        if status and status.get("injection"):
            inj = status["injection"]
            st.info(f"Injected: {EVENT_LABEL[inj['type']]}, severity {inj['severity']}")

        st.divider()
        counts = store.counts()
        st.caption(f"Database `{store.engine.url.get_backend_name()}`: {counts['readings']:,} readings, "
                   f"{counts['assessments']:,} assessments, {counts['alerts']:,} alerts")
        st.button("Clear database", use_container_width=True, on_click=store.clear)


# ---------------------------------------------------------------- page
SOUND = st.empty()                               # fixed slot, so a sound never shifts the page layout
st.title("\U0001F4A7 AI-Based Water Quality Monitoring for Data Center Cooling")
st.caption(f"Hybrid stacked ensemble + SHAP  •  {FORECASTERS[forecaster]} forecast  •  Isolation Forest  •  "
           f"Safe / Warning / Alert early warning  •  Source: {source}")

tab_live, tab_perf, tab_how = st.tabs(["Live monitor", "Model performance", "How it works"])


def render_live(r, log, alerts, alerts_caption):
    level = r["level"]
    st.markdown(
        f"<div style='background:{COLORS[level]};color:white;padding:14px 20px;border-radius:10px;"
        f"font-size:26px;font-weight:700'>{level.upper()}"
        f"<span style='font-size:16px;font-weight:400;margin-left:18px'>risk score R = {r['risk_score']:.1f}"
        f"  (Safe &lt; 50 ≤ Warning &lt; 75 ≤ Alert)</span></div>", unsafe_allow_html=True)
    if level != "Safe" and r["current_level"] == "Safe":
        st.warning("Early warning: the water is still acceptable now, but the forecast predicts it will "
                   "cross the limit within the next hour.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CWQI now (hybrid estimate)", f"{r['cwqi']:.1f}")
    k2.metric(f"Worst forecast, next 60 min ({r.get('forecaster', 'LSTM')})", f"{r['forecast_max']:.1f}",
              delta=f"{r['forecast_max'] - r['cwqi']:+.1f}", delta_color="inverse")
    k3.metric("Anomaly detector", "Flagged" if r["anomaly_flag"] else "Normal",
              f"evidence {r['anomaly_evidence']:.2f}", delta_color="off")
    k4.metric("Reading time", f"{r['time']:%d %b %Y, %H:%M}")

    # CWQI chart: history + forecast + thresholds
    hist = pd.DataFrame({"time": [x["time"] for x in log],
                         "Hybrid estimate": [x["cwqi"] for x in log],
                         "Ground truth": [x["true_cwqi"] for x in log]})
    series = ["Hybrid estimate"] + (["Ground truth"] if show_truth else [])
    long = hist.melt("time", series, "Series", "CWQI").dropna()
    fc_points = pd.DataFrame({"time": [r["time"] + pd.Timedelta(minutes=m) for m in (15, 30, 60)],
                              "CWQI": list(r["forecast"].values()), "Series": "Forecast"})
    long = pd.concat([long, fc_points])
    y_max = max(100, float(long["CWQI"].max()) + 5)
    base = alt.Chart(long).encode(
        x=alt.X("time:T", title=None),
        y=alt.Y("CWQI:Q", scale=alt.Scale(domain=[0, y_max]), title="CWQI"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Hybrid estimate", "Ground truth", "Forecast"],
                                                     range=["#1a73e8", "#5f6368", "#d93025"])))
    lines = base.transform_filter(alt.datum.Series != "Forecast").mark_line()
    points = base.transform_filter(alt.datum.Series == "Forecast").mark_line(point=True, strokeDash=[4, 3])
    rules = alt.Chart(pd.DataFrame({"y": [SAFE_LIMIT, ALERT_LIMIT], "label": ["Warning 50", "Alert 75"]})) \
        .mark_rule(strokeDash=[2, 2], color="#9aa0a6").encode(y=alt.Y("y:Q", title="CWQI"))
    st.altair_chart((lines + points + rules).properties(height=300), use_container_width=True)

    st.subheader("Sensor readings")
    cols = st.columns(len(SENSORS))
    for col, s in zip(cols, SENSORS):
        v = r["reading"][s]
        q = float(sub_index(s, [v])[0]) if pd.notna(v) else np.nan
        status = "✅ in range" if q < 100 else ("⚠️ beyond limit" if pd.notna(v) else "no reading")
        col.metric(f"{SPEC[s]['name']}", f"{v:.2f} {SPEC[s]['unit']}" if pd.notna(v) else "missing",
                   status, delta_color="off")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Why? Root-cause attribution (SHAP)")
        att = pd.DataFrame({"Parameter": [SPEC[s]["name"] for s in SENSORS],
                            "Contribution (index points)": [r["attribution"][s] for s in SENSORS]})
        bar = alt.Chart(att).mark_bar().encode(
            x=alt.X("Contribution (index points):Q"),
            y=alt.Y("Parameter:N", sort="-x"),
            color=alt.condition(alt.datum["Contribution (index points)"] > 0,
                                alt.value("#d93025"), alt.value("#1e8e3e")))
        st.altair_chart(bar.properties(height=230), use_container_width=True)
    with right:
        st.subheader("Operator guidance")
        if level == "Safe":
            st.success("Water quality is acceptable. No action needed.")
        else:
            st.error(f"**Most likely cause: {r['root_cause_name']}**\n\n{ACTIONS[r['root_cause']]}")
        if r["anomaly_flag"]:
            st.info("The anomaly detector flagged this reading: an abrupt change or a possible probe fault.")
        st.subheader("Alert log")
        if len(alerts):
            st.dataframe(pd.DataFrame(alerts), hide_index=True, use_container_width=True)
        else:
            st.caption("No alerts yet.")
        st.caption(alerts_caption)


with tab_live:
    if source == REPLAY:
        if st.session_state.last is None:
            step_once()
        render_live(st.session_state.last, st.session_state.log, st.session_state.alerts[:8],
                    "Level changes in this replay (pop-up and sound in the page).")
        toast = st.session_state.pop("new_toast", None)
        if toast:
            notify_in_app(*toast)
    else:
        from wq_system.storage import to_result
        a = store.table("assessments", 288)
        if a.empty:
            st.info("Waiting for data. Start the live system:\n\n"
                    "1. `python broker.py`  (MQTT broker)\n"
                    "2. `python ingest_service.py`  (MQTT → database → models → alerts)\n"
                    "3. `python sensor_publisher.py`  (simulated sensor gateway)\n\n"
                    "or double-click **Start Live System.bat**, or run `docker compose up`.")
        else:
            r = store.latest()
            log = [to_result(row) for _, row in a.iterrows()]
            al = store.table("alerts", 8)
            table = [{"Time": pd.Timestamp(x["ts"]).strftime("%d %b %H:%M"), "Type": x["kind"], "Level": x["level"],
                      "Message": x["title"], "Sent via": x["channels"]} for _, x in al.iloc[::-1].iterrows()]
            render_live(r, log, table, "Alerts stored in the database by the ingestion service, which also sends "
                                       "them as desktop / e-mail / phone notifications if configured.")
            if st.session_state.seen_alert_id is not None:
                for _, x in al[al["id"] > st.session_state.seen_alert_id].iterrows():
                    notify_in_app(x["title"], x["level"])
            st.session_state.seen_alert_id = int(al["id"].max()) if len(al) else 0

with tab_perf:
    m = load_metrics()
    st.subheader("Results on the unseen 60-day test segment")
    st.caption(f"Models trained with {m['trained_with']} in {m['training_minutes']} min. "
               "The paper's tables were produced with the full settings (`python train_models.py --paper`).")
    est = m["estimation"]["Hybrid stacked ensemble"]
    raw = m["estimation"]["Index formula on raw readings"]
    a, b, c, d = st.columns(4)
    a.metric("R² (hybrid estimate)", f"{est['r2']:.4f}")
    b.metric("RMSE", f"{est['rmse']:.2f}", f"raw readings: {raw['rmse']:.2f}", delta_color="off")
    c.metric("RMSE when contaminated (CWQI ≥ 50)", f"{est['rmse_contaminated']:.2f}",
             f"bias {est['bias_contaminated']:+.2f}", delta_color="off")
    d.metric("Root cause identified", m["root_cause_top1"] + " events")
    e, f, g, h = st.columns(4)
    fc = m["forecast_60min_rmse"]
    e.metric("60-min forecast RMSE (LSTM)", f"{fc['LSTM']:.2f}",
             f"persistence: {fc['Persistence (no change)']:.2f}", delta_color="off")
    if "XGBoost (lag features)" in fc:
        e.metric("60-min forecast RMSE (XGBoost)", f"{fc['XGBoost (lag features)']:.2f}")
    f.metric("Anomaly detection F1", f"{m['anomaly']['f1']:.2f}", f"events caught {m['anomaly']['events_detected']}",
             delta_color="off")
    g.metric("Early-warning macro-F1", f"{m['early_warning']['macro_f1']:.3f}")
    h.metric("False-alarm rate", f"{m['early_warning']['false_alarm_rate'] * 100:.2f}%")
    st.subheader("Meta-learner weights of the stacked ensemble")
    st.bar_chart(pd.Series(m["meta_weights"], name="weight"))

with tab_how:
    st.markdown("""
**Pipeline (Fig. 1 of the paper)**

1. **Sensors** measure 7 parameters every 5 minutes: temperature, pH, conductivity, dissolved oxygen,
   free residual chlorine, turbidity and corrosion-inhibitor concentration.
2. **Ingestion**: the sensor gateway publishes each reading over **MQTT**; the ingestion service stores it in
   the **database** (SQLite or MySQL, optionally mirrored to the InfluxDB time-series database).
3. **Pre-processing** fills short gaps and computes a 30-min rolling median and standard deviation (21 features).
4. **CWQI** (eq. 1-2): each reading gets a rating from 0 (ideal) to 100 (at its limit); the weighted sum is the index.
5. **Hybrid estimator** (eq. 3): the index of the smoothed readings plus a correction learned by a stacked
   ensemble of XGBoost, CatBoost, Random Forest, Gradient Boosting, Extra Trees and AdaBoost.
6. **SHAP** (eq. 4): splits the estimate into one contribution per sensor, which names the root cause.
7. **Forecast**: LSTM or XGBoost predicts the index 15, 30 and 60 min ahead.
8. **Isolation Forest** (eq. 5): flags abrupt anomalies and probe faults.
9. **Risk fusion** (eq. 6): R = max(current index, worst forecast); Safe < 50, Warning 50-75, Alert ≥ 75.
10. **Alerts and notification**: level changes and anomaly bursts go to the dashboard (pop-up and sound),
    the alert log, desktop notifications, e-mail and phone push (ntfy), as configured in `settings.env`.
11. **REST API** (FastAPI, http://localhost:8000/docs) lets other software send readings and read results.

The whole system runs with one command in **Docker** (`docker compose up`): Mosquitto MQTT broker, MySQL,
InfluxDB, ingestion service, sensor gateway, API and this dashboard.

**Data:** a physics-informed simulation of an open recirculating cooling tower (240 days, 5-min readings,
81 contamination events, 50 sensor faults). The live monitor replays the last 60 days, which the models
never saw during training.
""")
    fig = os.path.join(HERE, "assets", "fig1_architecture.png")
    if os.path.exists(fig):
        st.image(fig, caption="Architecture of the proposed system (Fig. 1)", width=420)

# ---------------------------------------------------------------- auto-play / auto-refresh
if source == REPLAY and st.session_state.running:
    for _ in range(steps_per_tick):
        step_once()
    time.sleep(0.3)
    st.rerun()
elif source == LIVE and auto:
    time.sleep(1.0)
    st.rerun()
