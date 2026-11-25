import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, LineChart, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  Activity, AlertTriangle, BellRing, CheckCircle2, Clock, Droplets, FlaskConical, Gauge, LineChart as LineIcon,
  Lightbulb, Radar, Search, ShieldAlert, ShieldCheck, Siren, Thermometer, TrendingUp, Waves, Wind, Zap, Shield, Inbox,
} from "lucide-react";
import { COLORS, LEVEL_CLASS, continuous, fmtDay, fmtDayTime, fmtTime, levelOf, levelVars, num, subIndex, toMs } from "../format.js";

const HORIZONS = [
  ["15 min", 15],
  ["30 min", 30],
  ["60 min", 60],
];
const SENSOR_ICON = { temp: Thermometer, ph: FlaskConical, cond: Zap, do: Wind, frc: Droplets, turb: Waves, inhib: Shield };
const LEVEL_ICON = { Safe: ShieldCheck, Warning: AlertTriangle, Alert: Siren };

// ------------------------------------------------------------------ risk gauge (eq. 6 on the CWQI scale)
function polar(cx, cy, r, v) {
  const a = Math.PI * (1 - v / 100);
  return [cx + r * Math.cos(a), cy - r * Math.sin(a)];
}

function arc(cx, cy, r, from, to) {
  const [x1, y1] = polar(cx, cy, r, from);
  const [x2, y2] = polar(cx, cy, r, to);
  return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
}

function RiskGauge({ value, limits }) {
  const v = Math.max(0, Math.min(100, value));
  const [mx, my] = polar(100, 100, 80, v);
  const level = levelOf(value, limits);
  const zones = [
    [0, limits.warning, "var(--safe)"],
    [limits.warning, limits.alert, "var(--warn)"],
    [limits.alert, 100, "var(--alert)"],
  ];
  return (
    <svg className="gauge" viewBox="-6 -4 212 126" role="img" aria-label={`Risk score ${num(value)}`}>
      {zones.map(([a, b, c]) => (
        <path key={a} d={arc(100, 100, 80, a + 0.5, b - 0.5)} stroke={c} strokeWidth="12" fill="none" opacity="0.28" />
      ))}
      <path d={arc(100, 100, 80, 0, Math.max(0.5, v))} stroke={COLORS[level]} strokeWidth="12" fill="none"
            strokeLinecap="round" style={{ transition: "d 0.4s" }} />
      {[0, limits.warning, limits.alert, 100].map((t) => {
        const [x, y] = polar(100, 100, 98, t);
        return (
          <text key={t} x={x} y={y + 3} textAnchor="middle" className="gauge-tick">
            {t}
          </text>
        );
      })}
      <circle cx={mx} cy={my} r="8" fill="var(--surface)" stroke={COLORS[level]} strokeWidth="4" />
      <text x="100" y="92" textAnchor="middle" className="gauge-value num">
        {num(value)}
      </text>
      <text x="100" y="112" textAnchor="middle" className="gauge-label">
        RISK SCORE R
      </text>
    </svg>
  );
}

function StatusCard({ r, cfg }) {
  const level = r.level;
  const Icon = LEVEL_ICON[level];
  const early = level !== "Safe" && r.current_level === "Safe";
  let msg;
  if (level === "Safe") msg = "Water quality is within limits now and over the next 60 minutes.";
  else if (early) msg = "Water is acceptable now, but the forecast crosses the limit within the next hour.";
  else if (level === "Warning") msg = "Water quality is poor. Act before it reaches the Alert level.";
  else msg = "Water quality is unacceptable. Immediate corrective action is needed.";
  return (
    <div className="card status-card" style={levelVars(level)}>
      <div className="status-top">
        <span className="eyebrow">System status</span>
        <span className={`pill ${LEVEL_CLASS[level]}`}>
          Safe &lt; {cfg.limits.warning} ≤ Warning &lt; {cfg.limits.alert} ≤ Alert
        </span>
      </div>
      <div className="level-word">
        <Icon size={30} strokeWidth={2.2} />
        {level.toUpperCase()}
      </div>
      <div className="status-msg">{msg}</div>
      <RiskGauge value={r.risk_score} limits={cfg.limits} />
      {early && (
        <div className="callout warn">
          <TrendingUp size={16} />
          <div>
            <b>Early warning</b> – forecast peak {num(r.forecast_max)} vs. {num(r.cwqi)} now.
          </div>
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ KPI tiles
function Spark({ data, color }) {
  return (
    <div style={{ width: 110, height: 34 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 3, bottom: 3, left: 0, right: 0 }}>
          <YAxis hide domain={["dataMin - 1", "dataMax + 1"]} />
          <Line dataKey="v" stroke={color} strokeWidth={1.8} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function Kpis({ r, log, view, cfg }) {
  const delta = r.forecast_max - r.cwqi;
  const trend = continuous(log).slice(-72).map((p) => ({ v: p.cwqi }));
  const fcLevel = levelOf(r.forecast_max, cfg.limits);
  return (
    <div className="kpis">
      <div className="card kpi">
        <div className="kpi-head">
          <span className="kpi-icon"><Gauge size={16} /></span>
          CWQI now · hybrid estimate
        </div>
        <div className="kpi-value">
          {num(r.cwqi)}
          <small>/ 100</small>
        </div>
        <div className="kpi-foot">
          <span className={`pill ${LEVEL_CLASS[r.current_level]}`}>{r.current_level}</span>
          <Spark data={trend} color="var(--blue)" />
        </div>
      </div>
      <div className="card kpi">
        <div className="kpi-head">
          <span className="kpi-icon"><TrendingUp size={16} /></span>
          Worst forecast · next 60 min ({r.forecaster})
        </div>
        <div className="kpi-value">{num(r.forecast_max)}</div>
        <div className="kpi-foot">
          <span className={delta > 0 ? "delta up" : "delta down"}>
            {delta > 0 ? "▲" : "▼"} {num(Math.abs(delta))} vs now
          </span>
          <span className="num">
            {HORIZONS.map(([k]) => num(r.forecast[k])).join(" → ")}
          </span>
        </div>
        <span className={`pill ${LEVEL_CLASS[fcLevel]}`} style={{ alignSelf: "flex-start" }}>
          forecast level: {fcLevel}
        </span>
      </div>
      <div className="card kpi">
        <div className="kpi-head">
          <span className="kpi-icon"><Radar size={16} /></span>
          Anomaly detector · Isolation Forest
        </div>
        <div className="kpi-value" style={{ color: r.anomaly_flag ? "var(--alert)" : "var(--safe)" }}>
          {r.anomaly_flag ? "Flagged" : "Normal"}
        </div>
        <div className="kpi-foot">
          <span>evidence (last 30 min)</span>
          <span className="num"><b>{num(r.anomaly_evidence, 2)}</b></span>
        </div>
        <div className="bar">
          <div style={{ width: `${100 * r.anomaly_evidence}%`, background: r.anomaly_evidence >= 0.5 ? "var(--alert)" : "var(--accent)" }} />
        </div>
      </div>
      <div className="card kpi">
        <div className="kpi-head">
          <span className="kpi-icon"><Clock size={16} /></span>
          Latest reading
        </div>
        <div className="kpi-value num">{fmtTime(r.time)}</div>
        <div className="kpi-foot">
          <span>{fmtDay(r.time)}</span>
          <span>{view.source === "replay" ? "5-min sampling" : `${r.device ?? "loop1"} · plant time`}</span>
        </div>
        {view.source === "replay" ? (
          <div title="Position in the unseen 60-day test segment">
            <div className="bar">
              <div style={{ width: `${(100 * view.position) / view.total}%`, background: "var(--accent)" }} />
            </div>
          </div>
        ) : (
          <div className="kpi-foot" style={{ minHeight: 0 }}>
            <span>{view.database.counts.readings.toLocaleString()} readings stored</span>
          </div>
        )}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ CWQI chart
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const names = { est: "Hybrid estimate", truth: "Ground truth", fc: "Forecast" };
  const seen = new Set();
  return (
    <div className="tooltip">
      <div className="muted" style={{ marginBottom: 4 }}>{fmtDayTime(label)}</div>
      {payload
        .filter((p) => p.value !== null && p.value !== undefined && names[p.dataKey] && !seen.has(p.dataKey) && seen.add(p.dataKey))
        .map((p) => (
          <div className="tooltip-row" key={p.dataKey}>
            <span style={{ color: p.color }}>{names[p.dataKey]}</span>
            <b className="num">{num(p.value)}</b>
          </div>
        ))}
    </div>
  );
}

function CwqiChart({ r, log, showTruth, limits }) {
  const data = continuous(log).map((p) => ({ t: toMs(p.time), est: p.cwqi, truth: showTruth ? p.true_cwqi : null }));
  const now = toMs(r.time);
  // the forecast line starts at the current estimate and goes through the 15/30/60-min forecasts
  if (data.length) data[data.length - 1].fc = r.cwqi;
  for (const [key, m] of HORIZONS) data.push({ t: now + m * 60000, fc: r.forecast[key] });
  const values = data.flatMap((d) => [d.est, d.truth, d.fc]).filter((v) => v !== null && v !== undefined);
  const yMax = Math.max(100, Math.ceil((Math.max(...values) + 5) / 10) * 10);
  return (
    <ResponsiveContainer width="100%" height={320}>
      <ComposedChart data={data} margin={{ top: 10, right: 12, bottom: 0, left: -12 }}>
        <ReferenceArea y1={0} y2={limits.warning} fill="var(--safe)" fillOpacity={0.05} ifOverflow="hidden" />
        <ReferenceArea y1={limits.warning} y2={limits.alert} fill="var(--warn)" fillOpacity={0.07} ifOverflow="hidden" />
        <ReferenceArea y1={limits.alert} y2={yMax} fill="var(--alert)" fillOpacity={0.06} ifOverflow="hidden" />
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={fmtTime}
               tick={{ fontSize: 11, fill: "var(--muted)" }} tickLine={false} axisLine={{ stroke: "var(--border)" }} minTickGap={48} />
        <YAxis domain={[0, yMax]} ticks={[0, 25, limits.warning, limits.alert, 100].filter((t) => t <= yMax)}
               tick={{ fontSize: 11, fill: "var(--muted)" }} tickLine={false} axisLine={false} />
        <Tooltip content={<ChartTooltip />} />
        <ReferenceLine y={limits.warning} stroke="var(--warn)" strokeDasharray="4 4"
                       label={{ value: "Warning", position: "insideTopLeft", fontSize: 10, fill: "var(--warn)" }} />
        <ReferenceLine y={limits.alert} stroke="var(--alert)" strokeDasharray="4 4"
                       label={{ value: "Alert", position: "insideTopLeft", fontSize: 10, fill: "var(--alert)" }} />
        <ReferenceLine x={now} stroke="var(--faint)" strokeDasharray="2 3"
                       label={{ value: "now", position: "insideTopRight", fontSize: 10, fill: "var(--muted)" }} />
        {showTruth && (
          <Line dataKey="truth" stroke="var(--truth)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
        )}
        <Line dataKey="est" stroke="var(--blue)" strokeWidth={2} dot={false} isAnimationActive={false} />
        <Line dataKey="fc" stroke="var(--forecast)" strokeWidth={2} strokeDasharray="5 4"
              dot={{ r: 3.5, fill: "var(--surface)", strokeWidth: 2 }} connectNulls isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function ChartCard({ r, log, showTruth, cfg }) {
  const hours = Math.round((continuous(log).length * 5) / 60);
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div className="card-head">
        <div>
          <div className="card-title"><LineIcon size={17} />Cooling-Water Quality Index (CWQI)</div>
          <div className="card-sub">Last {hours || 1} h and the 60-min forecast · lower is better</div>
        </div>
        <div className="legend">
          <span><i className="swatch" style={{ background: "var(--blue)" }} />Hybrid estimate</span>
          {showTruth && <span><i className="swatch" style={{ background: "var(--truth)" }} />Ground truth</span>}
          <span><i className="swatch dashed" />Forecast</span>
          <span><i className="zone" style={{ background: "var(--safe)", opacity: 0.35 }} />Safe</span>
          <span><i className="zone" style={{ background: "var(--warn)", opacity: 0.45 }} />Warning</span>
          <span><i className="zone" style={{ background: "var(--alert)", opacity: 0.4 }} />Alert</span>
        </div>
      </div>
      <CwqiChart r={r} log={log} showTruth={showTruth} limits={cfg.limits} />
    </div>
  );
}

// ------------------------------------------------------------------ sensors
function SensorCard({ s, r, log, cfg }) {
  const spec = cfg.parameters[s];
  const Icon = SENSOR_ICON[s] ?? Activity;
  const v = r.reading[s];
  const missing = v === null || v === undefined;
  const q = subIndex(spec, v);
  const status = missing ? "no reading" : q >= 100 ? "beyond limit" : q >= 50 ? "drifting" : "in range";
  const cls = missing ? "muted" : q >= 100 ? "alert" : q >= 50 ? "warn" : "safe";
  const color = `var(--${cls === "muted" ? "faint" : cls})`;
  const trend = continuous(log)
    .slice(-72)
    .map((p) => ({ v: p.reading?.[s] ?? null }));
  const limit = [spec.low !== null && `≥ ${spec.low}`, spec.high !== null && `≤ ${spec.high}`].filter(Boolean).join(" · ");
  const digits = s === "cond" ? 0 : s === "frc" ? 2 : 1;
  return (
    <div className={`card sensor ${cls === "alert" ? "bad" : ""}`}>
      <div className="sensor-head">
        <Icon size={15} />
        {spec.name}
      </div>
      <div className="sensor-value">
        {missing ? "—" : num(v, digits)}
        <small>{missing ? "missing" : spec.unit}</small>
      </div>
      <div style={{ height: 30 }}>
        {trend.length > 2 && (
          <ResponsiveContainer>
            <LineChart data={trend} margin={{ top: 3, bottom: 3, left: 0, right: 0 }}>
              <YAxis hide domain={["auto", "auto"]} />
              <Line dataKey="v" stroke={color} strokeWidth={1.6} dot={false} connectNulls isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
      <div className="sensor-foot" title={`rating q = ${num(q, 0)} (0 = ideal, 100 = at the limit)`}>
        <div className="bar">
          <div style={{ width: `${Math.min(100, q ?? 0)}%`, background: color }} />
        </div>
        <span className={`pill ${cls}`}>{status}</span>
      </div>
      <div className="sensor-foot">
        ideal {spec.ideal} · limit {limit}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ insights
function AttributionCard({ r, cfg }) {
  const data = cfg.sensors
    .map((s) => ({ name: cfg.parameters[s].name, v: r.attribution[s] }))
    .sort((a, b) => b.v - a.v);
  return (
    <div className="card">
      <div className="card-head">
        <div>
          <div className="card-title"><Search size={17} />Why? Root-cause attribution</div>
          <div className="card-sub">SHAP contribution of each sensor, in index points (eq. 4)</div>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 0 }} barCategoryGap={6}>
          <CartesianGrid stroke="var(--grid)" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 11, fill: "var(--muted)" }} tickLine={false} axisLine={false} />
          <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 12, fill: "var(--text)" }} tickLine={false} axisLine={false} />
          <Tooltip cursor={{ fill: "var(--grid)" }} formatter={(v) => [`${num(v, 2)} index points`, "Contribution"]}
                   contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10 }} />
          <ReferenceLine x={0} stroke="var(--border-strong)" />
          <Bar dataKey="v" radius={[0, 4, 4, 0]} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.name} fill={d.v > 0 ? "var(--alert)" : "var(--safe)"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function GuidanceCard({ r, cfg }) {
  const level = r.level;
  const beyond = cfg.sensors.filter((s) => (subIndex(cfg.parameters[s], r.reading[s]) ?? 0) >= 100);
  const contributors = cfg.sensors
    .filter((s) => r.attribution[s] > 0.5)
    .sort((a, b) => r.attribution[b] - r.attribution[a])
    .slice(0, 3);
  return (
    <div className="card">
      <div className="card-head">
        <div>
          <div className="card-title"><Lightbulb size={17} />Operator guidance</div>
          <div className="card-sub">What to check, based on the root cause</div>
        </div>
      </div>
      {level === "Safe" ? (
        <div className="callout safe">
          <CheckCircle2 size={18} />
          <div>
            <b>No action needed</b>
            <p>
              {beyond.length
                ? `${beyond.length} parameter(s) beyond limit, but the overall index is acceptable.`
                : "All seven parameters are within their limits."}
            </p>
          </div>
        </div>
      ) : (
        <>
          <div className="eyebrow">Most likely cause</div>
          <div className="cause-name" style={{ color: COLORS[level] }}>{r.root_cause_name}</div>
          <div className={`callout ${LEVEL_CLASS[level]}`}>
            <ShieldAlert size={18} />
            <div style={{ color: "var(--text)" }}>{cfg.actions[r.root_cause]}</div>
          </div>
          {contributors.length > 1 && (
            <>
              <div className="eyebrow" style={{ marginTop: 14 }}>Contributing parameters</div>
              <div className="chips">
                {contributors.map((s) => (
                  <span key={s} className="pill alert">
                    {cfg.parameters[s].name} +{num(r.attribution[s])}
                  </span>
                ))}
              </div>
            </>
          )}
        </>
      )}
      {r.anomaly_flag && (
        <div className="callout info">
          <Radar size={18} />
          <div>The anomaly detector flagged this reading: an abrupt change or a possible probe fault.</div>
        </div>
      )}
    </div>
  );
}

function AlertCard({ alerts, source }) {
  return (
    <div className="card">
      <div className="card-head">
        <div>
          <div className="card-title"><BellRing size={17} />Alert log</div>
          <div className="card-sub">
            {source === "replay"
              ? "Level changes in this replay"
              : "Stored by the ingestion service and sent on every configured channel"}
          </div>
        </div>
      </div>
      {alerts.length ? (
        <ul className="timeline">
          {alerts.map((a) => (
            <li key={a.id}>
              <span className="tl-dot" style={{ background: a.kind === "anomaly" ? "var(--info)" : COLORS[a.level] }} />
              <div>
                <div className="tl-title">
                  {a.kind === "anomaly" ? "Anomaly burst" : a.level}
                  {a.root_cause_name && a.level !== "Safe" ? ` · ${a.root_cause_name}` : ""}
                </div>
                <div className="tl-meta num">
                  {fmtDayTime(a.time)} · CWQI {num(a.cwqi)} · forecast {num(a.forecast_max)}
                  {source === "live" && a.channels ? ` · via ${a.channels}` : ""}
                </div>
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <div className="empty">
          <Inbox size={26} />
          No alerts yet. Inject a fault to see one.
        </div>
      )}
    </div>
  );
}

function Waiting({ view }) {
  const agent = view?.agent;
  return (
    <div className="card">
      <div className="card-title" style={{ marginBottom: 8 }}><Clock size={17} />Waiting for live data</div>
      {agent?.enabled ? (
        <p className="muted" style={{ marginTop: 0 }}>
          The built-in sensor feeder agent is starting (MQTT broker: {agent.broker ?? "…"}). The first readings
          appear in a few seconds.
        </p>
      ) : (
        <>
          <p className="muted" style={{ marginTop: 0 }}>
            The API's built-in feeder agent is switched off (<code>LIVE_AGENT=0</code>). Start the separate parts:
          </p>
          <ol>
            <li><code>python broker.py</code> – MQTT broker</li>
            <li><code>python ingest_service.py</code> – MQTT → database → models → alerts</li>
            <li><code>python sensor_publisher.py</code> – sensor feeder agent</li>
          </ol>
          <p className="muted">
            or double-click <b>Start Live System.bat</b>, or run <code>docker compose up</code>.
          </p>
        </>
      )}
    </div>
  );
}

function Loading() {
  return (
    <div className="splash" style={{ minHeight: 300 }}>
      <div className="spinner" />
    </div>
  );
}

export default function LiveMonitor({ cfg, view, source, showTruth }) {
  if (!view || view.source !== source) return <Loading />;
  const r = view.last;
  if (!r) return source === "live" ? <Waiting view={view} /> : <Loading />;
  return (
    <div>
      <div className="hero">
        <StatusCard r={r} cfg={cfg} />
        <Kpis r={r} log={view.log} view={view} cfg={cfg} />
      </div>
      <ChartCard r={r} log={view.log} showTruth={showTruth} cfg={cfg} />
      <div className="section-title">
        Sensor readings <span className="muted">7 online parameters · every 5 min · bar = rating q (100 = at the limit)</span>
      </div>
      <div className="sensors">
        {cfg.sensors.map((s) => (
          <SensorCard key={s} s={s} r={r} log={view.log} cfg={cfg} />
        ))}
      </div>
      <div className="insights">
        <AttributionCard r={r} cfg={cfg} />
        <GuidanceCard r={r} cfg={cfg} />
        <AlertCard alerts={view.alerts} source={source} />
      </div>
    </div>
  );
}
