import {
  Bell, Boxes, Cpu, Database, Filter, Gauge, Layers, Network, Radar, Search, ShieldAlert, TrendingUp, Workflow,
} from "lucide-react";

const STEPS = [
  [Cpu, "Sensors", "7 parameters every 5 min: temperature, pH, conductivity, dissolved oxygen, free residual chlorine, turbidity, corrosion inhibitor."],
  [Network, "Ingestion (MQTT)", "The sensor gateway publishes each reading over MQTT; the ingestion service stores it in SQLite / MySQL and InfluxDB."],
  [Filter, "Pre-processing", "Short gaps are filled; a 30-min rolling median and standard deviation give 21 features."],
  [Gauge, "CWQI (eq. 1–2)", "Each reading gets a rating from 0 (ideal) to 100 (at its limit); the weighted sum is the index."],
  [Layers, "Hybrid estimator (eq. 3)", "Index of the smoothed readings plus a correction learned by a stacked ensemble of 6 tree models."],
  [Search, "SHAP (eq. 4)", "Splits the estimate into one contribution per sensor, which names the root cause."],
  [TrendingUp, "Forecast", "LSTM or XGBoost predicts the index 15, 30 and 60 min ahead."],
  [Radar, "Isolation Forest (eq. 5)", "Flags abrupt anomalies and probe faults as a separate alarm channel."],
  [ShieldAlert, "Risk fusion (eq. 6)", "R = max(current index, worst forecast): Safe < 50, Warning 50–75, Alert ≥ 75."],
  [Bell, "Alerts", "Dashboard pop-up and sound, alert log, desktop, e-mail, phone push (ntfy) and webhook."],
  [Database, "REST API", "FastAPI serves this React dashboard and lets other software send readings and read results (/docs)."],
  [Boxes, "Docker", "docker compose up starts Mosquitto, MySQL, InfluxDB, ingestion, gateway and the API."],
];

export default function HowItWorks() {
  return (
    <div>
      <div className="card-head" style={{ marginBottom: 16 }}>
        <div>
          <div className="card-title" style={{ fontSize: "1.1rem" }}><Workflow size={18} />Pipeline (Fig. 1 of the paper)</div>
          <div className="card-sub">From a sensor reading to a Safe / Warning / Alert decision in about 0.2 s on a laptop CPU</div>
        </div>
      </div>
      <div className="steps">
        {STEPS.map(([Icon, title, text], i) => (
          <div className="card step" key={title}>
            <div className="step-num"><Icon size={17} /></div>
            <div>
              <h3><span>{String(i + 1).padStart(2, "0")}</span>{title}</h3>
              <p>{text}</p>
            </div>
          </div>
        ))}
      </div>
      <div className="how-bottom">
        <figure className="card figure" style={{ margin: 0 }}>
          <div className="card-title" style={{ marginBottom: 12 }}>Architecture (Fig. 1)</div>
          <img src="/fig1_architecture.png" alt="Architecture of the proposed system" />
        </figure>
        <div className="card">
          <div className="card-title">Data</div>
          <p className="muted" style={{ marginBottom: 0 }}>
            No public data set of AI data center cooling-water chemistry exists, so a physics-informed simulator of an open
            recirculating cooling tower generates the data. The live monitor replays the last 60 days, which the models
            never saw during training.
          </p>
          <div className="facts">
            <div className="fact"><b>240</b><span>days of 5-min readings</span></div>
            <div className="fact"><b>69,120</b><span>readings</span></div>
            <div className="fact"><b>81</b><span>contamination events, 5 types</span></div>
            <div className="fact"><b>50</b><span>sensor faults (spikes, stuck probes)</span></div>
            <div className="fact"><b>60 / 15 / 25 %</b><span>train / validation / test, by time</span></div>
            <div className="fact"><b>1 %</b><span>missing readings</span></div>
          </div>
        </div>
      </div>
    </div>
  );
}
