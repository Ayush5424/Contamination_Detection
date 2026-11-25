import { useRef, useState } from "react";
import {
  CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  Bell, Crosshair, Download, FileUp, FlaskConical, Gauge, Info, LineChart as LineIcon, ListChecks, Play, Radar,
  Search, ShieldCheck, Target, Timer, TrendingUp,
} from "lucide-react";
import { api } from "../api.js";
import { LEVEL_CLASS, fmtDay, fmtDayTime, num, toMs } from "../format.js";

const DAYS = [
  [7, "7 days"],
  [30, "30 days"],
  [60, "60 days"],
];
const NOISE = [
  [1, "normal"],
  [2, "2× noise"],
];
const DRIFT = [
  [0, "none"],
  [0.3, "pH +0.3"],
];
const MAX_FILE_MB = 25;

function Segmented({ options, value, onChange }) {
  return (
    <div className="segmented" role="radiogroup">
      {options.map(([v, label]) => (
        <button key={v} role="radio" aria-checked={value === v} className={value === v ? "seg active" : "seg"}
                onClick={() => onChange(v)}>
          {label}
        </button>
      ))}
    </div>
  );
}

function Tile({ icon: Icon, label, value, sub, tone }) {
  return (
    <div className="card kpi">
      <div className="kpi-head">
        <span className="kpi-icon" style={tone ? { color: `var(--${tone})` } : undefined}>
          <Icon size={16} />
        </span>
        {label}
      </div>
      <div className="kpi-value">{value}</div>
      {sub && <div className="kpi-foot" style={{ minHeight: 0 }}>{sub}</div>}
    </div>
  );
}

// ------------------------------------------------------------------ inputs
function UploadCard({ busy, onRun }) {
  const input = useRef(null);
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const pick = (f) => {
    setError(null);
    if (f && f.size > MAX_FILE_MB * 1e6) {
      setError(`File is larger than ${MAX_FILE_MB} MB.`);
      setFile(null);
      return;
    }
    setFile(f ?? null);
  };
  return (
    <div className="card">
      <div className="card-head">
        <div>
          <div className="card-title"><FileUp size={17} />Upload your own readings (CSV)</div>
          <div className="card-sub">A plant log, a lab sheet or any sensor export</div>
        </div>
      </div>
      <div className="nd-format">
        Columns: <code>ts, temp, ph, cond, do, frc, turb, inhib</code>. Full names such as <i>Temperature</i> or{" "}
        <i>Conductivity</i> also work. Optional: <code>cwqi_true</code> and <code>event_type</code> to score the accuracy.
        Readings 5 min apart; other intervals are converted. At least 36 rows (3 h).
      </div>
      <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={(e) => pick(e.target.files[0])} />
      <div className="row">
        <button className="btn" onClick={() => input.current.click()} disabled={busy}>
          <FileUp size={15} />
          Choose CSV file
        </button>
        <a className="btn ghost" href="/dashboard/newdata/sample.csv" download>
          <Download size={15} />
          Sample CSV
        </a>
      </div>
      {file && <div className="nd-file">{file.name} · {(file.size / 1024).toFixed(0)} KB</div>}
      {error && <div className="callout alert">{error}</div>}
      <button className="btn primary wide" disabled={!file || busy} onClick={async () => onRun(await file.text(), file.name)}>
        <Play size={15} />
        Run the models on this file
      </button>
    </div>
  );
}

function SimulateCard({ busy, onRun }) {
  const [seed, setSeed] = useState(7);
  const [days, setDays] = useState(30);
  const [noise, setNoise] = useState(1);
  const [drift, setDrift] = useState(0);
  return (
    <div className="card">
      <div className="card-head">
        <div>
          <div className="card-title"><FlaskConical size={17} />Generate new simulated data</div>
          <div className="card-sub">New events, noise and sensor faults the models have never seen</div>
        </div>
      </div>
      <label className="field" style={{ marginTop: 0 }}>
        <span className="field-row">
          Random seed <span className="muted">training used 42</span>
        </span>
        <div className="row" style={{ marginTop: 0 }}>
          <input type="number" min="0" max="1000000" value={seed} className="nd-seed"
                 onChange={(e) => setSeed(Math.max(0, Math.floor(Number(e.target.value) || 0)))} />
          <button className="btn" onClick={() => setSeed(Math.floor(Math.random() * 100000))}>Random</button>
        </div>
      </label>
      {seed === 42 && <div className="callout warn">Seed 42 is the training data. Pick another seed for a fair test.</div>}
      <label className="field">
        Length
        <Segmented options={DAYS} value={days} onChange={setDays} />
      </label>
      <label className="field">
        Sensor noise
        <Segmented options={NOISE} value={noise} onChange={setNoise} />
      </label>
      <label className="field">
        Probe calibration drift
        <Segmented options={DRIFT} value={drift} onChange={setDrift} />
      </label>
      <button className="btn primary wide" disabled={busy}
              onClick={() => onRun({ seed, days, noise, ph_drift: drift })}>
        <Play size={15} />
        Generate and test
      </button>
    </div>
  );
}

// ------------------------------------------------------------------ results
function ResultChart({ chart, hasTruth, limits }) {
  const data = chart.map((p) => ({ t: toMs(p.t), est: p.est, fc: p.fc, truth: p.truth ?? null }));
  const yMax = Math.max(100, Math.ceil((Math.max(...data.flatMap((d) => [d.est, d.fc, d.truth ?? 0])) + 5) / 10) * 10);
  const short = data.length > 1 && data[data.length - 1].t - data[0].t < 5 * 86400000; // under 5 days: show hours too
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 10, right: 12, bottom: 0, left: -12 }}>
        <ReferenceArea y1={0} y2={limits.warning} fill="var(--safe)" fillOpacity={0.05} ifOverflow="hidden" />
        <ReferenceArea y1={limits.warning} y2={limits.alert} fill="var(--warn)" fillOpacity={0.07} ifOverflow="hidden" />
        <ReferenceArea y1={limits.alert} y2={yMax} fill="var(--alert)" fillOpacity={0.06} ifOverflow="hidden" />
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={short ? fmtDayTime : fmtDay}
               tick={{ fontSize: 11, fill: "var(--muted)" }} tickLine={false} axisLine={{ stroke: "var(--border)" }} minTickGap={60} />
        <YAxis domain={[0, yMax]} ticks={[0, 25, limits.warning, limits.alert, 100].filter((t) => t <= yMax)}
               tick={{ fontSize: 11, fill: "var(--muted)" }} tickLine={false} axisLine={false} />
        <Tooltip labelFormatter={fmtDayTime} formatter={(v, k) => [num(v), { est: "Hybrid estimate", fc: "Worst forecast (60 min)", truth: "Ground truth" }[k]]}
                 contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10 }} />
        <ReferenceLine y={limits.warning} stroke="var(--warn)" strokeDasharray="4 4" />
        <ReferenceLine y={limits.alert} stroke="var(--alert)" strokeDasharray="4 4" />
        <Line dataKey="fc" stroke="var(--forecast)" strokeWidth={1.2} strokeOpacity={0.6} dot={false} isAnimationActive={false} />
        {hasTruth && <Line dataKey="truth" stroke="var(--truth)" strokeWidth={1.4} dot={false} isAnimationActive={false} />}
        <Line dataKey="est" stroke="var(--blue)" strokeWidth={1.8} dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function lead(v, onsets) {
  if (!onsets) return "no onsets";
  return v === null ? "–" : `${num(v, 0)} min`;
}

function Accuracy({ m }) {
  return (
    <>
      <div className="section-title">
        Accuracy against the ground truth <span className="muted">the file had a cwqi_true column</span>
      </div>
      <div className="perf-kpis">
        <Tile icon={Target} label="R² · hybrid CWQI estimate" value={num(m.r2, 4)}
              sub={`RMSE ${num(m.rmse, 2)} · raw formula ${num(m.rmse_raw_formula, 2)}`} />
        {m.forecast_60min_rmse !== undefined && (
          <Tile icon={TrendingUp} label="60-min forecast RMSE" value={num(m.forecast_60min_rmse, 2)}
                sub={`persistence (no change) ${num(m.persistence_60min_rmse, 2)}`} />
        )}
        {m.macro_f1 !== undefined && (
          <Tile icon={Bell} label="Early-warning macro-F1" value={num(m.macro_f1, 3)}
                sub={`false alarms ${m.false_alarm_rate === null ? "–" : `${num(100 * m.false_alarm_rate, 2)}%`} of Safe instants`} />
        )}
        <Tile icon={Timer} label="Warning before the true crossing" value={lead(m.warning_lead_min, m.warning_onsets)}
              sub={`${m.warning_onsets} onsets · ${m.warning_missed} missed · Alert: ${lead(m.alert_lead_min, m.alert_onsets)} (${m.alert_onsets} onsets, ${m.alert_missed} missed)`} />
        {m.root_cause && (
          <Tile icon={Search} label="Root cause identified (SHAP)" value={m.root_cause} sub="true events, top-1 parameter" tone="safe" />
        )}
      </div>
    </>
  );
}

function Episodes({ rep }) {
  const showTruth = rep.episodes.some((e) => "true_event" in e);
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div className="card-head">
        <div>
          <div className="card-title"><ListChecks size={17} />Warning and Alert episodes</div>
          <div className="card-sub">
            {rep.n_episodes} episodes{rep.n_episodes > rep.episodes.length ? `, newest ${rep.episodes.length} shown` : ""} ·
            root cause by SHAP at the peak · "early" = how long the forecast warned before the estimate itself reached 50
          </div>
        </div>
      </div>
      {rep.episodes.length ? (
        <div className="nd-table-wrap">
          <table className="nd-table">
            <thead>
              <tr>
                <th>Start</th>
                <th>Duration</th>
                <th>Level</th>
                <th>Peak CWQI</th>
                <th>Root cause</th>
                <th>Early</th>
                {showTruth && <th>True event</th>}
              </tr>
            </thead>
            <tbody>
              {rep.episodes.map((e) => (
                <tr key={e.start}>
                  <td className="num">{fmtDayTime(e.start)}</td>
                  <td className="num">{e.minutes >= 120 ? `${num(e.minutes / 60, 1)} h` : `${e.minutes} min`}</td>
                  <td><span className={`pill ${LEVEL_CLASS[e.level]}`}>{e.level}</span></td>
                  <td className="num">{num(e.peak_cwqi)}</td>
                  <td title={e.contributors.join(", ")}>{e.root_cause_name}</td>
                  <td className="num">{e.early_minutes === null ? "forecast only" : e.early_minutes ? `${e.early_minutes} min` : "–"}</td>
                  {showTruth && <td className="muted">{e.true_event ?? "none (false alarm or sensor fault)"}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="callout safe">
          <ShieldCheck size={18} />
          <div>The water stayed Safe for the whole data set: no Warning or Alert.</div>
        </div>
      )}
    </div>
  );
}

function Results({ rep, cfg }) {
  const pct = (k) => `${num(100 * rep.level_share[k], 1)}%`;
  const causes = Object.entries(rep.root_causes);
  return (
    <div style={{ marginTop: 20 }}>
      <div className="card-head" style={{ marginBottom: 12 }}>
        <div>
          <div className="card-title" style={{ fontSize: "1.1rem" }}>
            Results · {rep.source === "file" ? rep.name : "new simulation"}
          </div>
          <div className="card-sub">
            {rep.rows.toLocaleString()} readings · {fmtDayTime(rep.start)} to {fmtDayTime(rep.end)} ({rep.days} days) ·
            forecaster {rep.forecaster === "xgb" ? "XGBoost" : "LSTM"} · saved models, not retrained
          </div>
        </div>
        <a className="btn nd-download" href={`/dashboard/newdata/result/${rep.result_id}.csv`} download>
          <Download size={15} />
          Download results (CSV)
        </a>
      </div>
      {rep.notes.length > 0 && (
        <div className="callout info" style={{ marginTop: 0, marginBottom: 16 }}>
          <Info size={18} />
          <div>{rep.notes.map((n) => <div key={n}>{n}</div>)}</div>
        </div>
      )}
      <div className="perf-kpis">
        <Tile icon={ShieldCheck} label="Safe" value={pct("Safe")} sub={`${rep.levels.Safe.toLocaleString()} readings`} tone="safe" />
        <Tile icon={Bell} label="Warning" value={pct("Warning")} sub={`${rep.levels.Warning.toLocaleString()} readings`} tone="warn" />
        <Tile icon={Crosshair} label="Alert" value={pct("Alert")} sub={`${rep.levels.Alert.toLocaleString()} readings`} tone="alert" />
        <Tile icon={Gauge} label="Warning / Alert episodes" value={rep.n_episodes}
              sub={causes.length ? `most often: ${causes[0][0]}` : "none"} />
        <Tile icon={Radar} label="Anomaly flags" value={rep.anomalies.toLocaleString()}
              sub={`${num((100 * rep.anomalies) / rep.rows, 1)}% of readings · Isolation Forest`} />
      </div>
      {rep.metrics && <Accuracy m={rep.metrics} />}
      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-head">
          <div>
            <div className="card-title"><LineIcon size={17} />CWQI over the whole data set</div>
            <div className="card-sub">Highest value per {rep.rows > 1500 ? "time bucket" : "reading"} · lower is better</div>
          </div>
          <div className="legend">
            <span><i className="swatch" style={{ background: "var(--blue)" }} />Hybrid estimate</span>
            {rep.metrics && <span><i className="swatch" style={{ background: "var(--truth)" }} />Ground truth</span>}
            <span><i className="swatch" style={{ background: "var(--forecast)", opacity: 0.6 }} />Worst 60-min forecast</span>
          </div>
        </div>
        <ResultChart chart={rep.chart} hasTruth={Boolean(rep.metrics)} limits={cfg.limits} />
      </div>
      {causes.length > 0 && (
        <div className="chips" style={{ marginTop: 16 }}>
          <span className="eyebrow" style={{ alignSelf: "center" }}>Root causes:</span>
          {causes.map(([name, n]) => (
            <span key={name} className="pill alert">{name} × {n}</span>
          ))}
        </div>
      )}
      <Episodes rep={rep} />
    </div>
  );
}

export default function NewData({ cfg, forecaster }) {
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [rep, setRep] = useState(null);
  const fcLabel = cfg.forecasters.find((f) => f.key === forecaster)?.label ?? forecaster;

  const go = async (label, fn) => {
    setBusy(label);
    setError(null);
    try {
      setRep(await fn());
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div>
      <div className="card-head" style={{ marginBottom: 16 }}>
        <div>
          <div className="card-title" style={{ fontSize: "1.1rem" }}>Test the trained models on new data</div>
          <div className="card-sub">
            Every reading goes through the same models as the live monitor (hybrid CWQI, {fcLabel} forecast, Isolation
            Forest, risk fusion, SHAP). Nothing is retrained. Change the forecaster in the control panel.
          </div>
        </div>
      </div>
      <div className="nd-inputs">
        <UploadCard busy={Boolean(busy)} onRun={(csv, name) =>
          go(`Running the models on ${name}…`, async () => ({ ...(await api.newDataCsv(csv, forecaster)), name }))} />
        <SimulateCard busy={Boolean(busy)} onRun={(sim) =>
          go(`Simulating ${sim.days} days (seed ${sim.seed}) and running the models…`, () => api.newDataSimulate({ ...sim, forecaster }))} />
      </div>
      {busy && (
        <div className="card nd-busy">
          <div className="spinner" />
          {busy}
        </div>
      )}
      {error && <div className="callout alert" style={{ marginTop: 16 }}>{error}</div>}
      {rep && !busy && <Results rep={rep} cfg={cfg} />}
    </div>
  );
}
