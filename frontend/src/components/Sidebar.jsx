import { useState } from "react";
import {
  AlertOctagon, CheckCircle2, Database, FlaskRound, Info, MapPin, Pause, Play, Radio, RefreshCw, RotateCcw,
  SkipForward, SlidersHorizontal, Trash2, X, XCircle,
} from "lucide-react";
import { api } from "../api.js";
import { fmtDayTime } from "../format.js";

const SPEEDS = [
  [1, "5 min"],
  [3, "15 min"],
  [6, "30 min"],
  [12, "1 h"],
];
const INTERVALS = [0.25, 0.5, 1, 2, 5, 300];

function Toggle({ label, checked, onChange, title }) {
  return (
    <label className="toggle" title={title}>
      {label}
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="switch" />
    </label>
  );
}

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

function Note({ kind, children }) {
  const Icon = { ok: CheckCircle2, warn: Info, err: XCircle, info: Info }[kind];
  return (
    <div className={`note ${kind}`}>
      <Icon size={15} />
      <div>{children}</div>
    </div>
  );
}

function EventJump({ cfg, onJump }) {
  const [event, setEvent] = useState(1);
  return (
    <div className="panel">
      <div className="panel-title"><MapPin size={14} />Recorded events</div>
      <label className="field" style={{ marginTop: 0 }}>
        Contamination events in the test period
        <select value={event} onChange={(e) => setEvent(Number(e.target.value))}>
          {cfg.events.map((ev) => (
            <option key={ev.number} value={ev.number}>
              {ev.number}. {ev.label} – {fmtDayTime(ev.time)}
            </option>
          ))}
        </select>
      </label>
      <button className="btn wide" onClick={() => onJump(event)}>
        <SkipForward size={15} />
        Go to 6 h before this event
      </button>
    </div>
  );
}

function InjectForm({ cfg, injection, onInject, onClear }) {
  const [type, setType] = useState(cfg.event_types[0].key);
  const [severity, setSeverity] = useState(1.5);
  const [ramp, setRamp] = useState(3);
  const label = (k) => cfg.event_types.find((t) => t.key === k)?.label ?? k;
  return (
    <div className="panel">
      <div className="panel-title"><FlaskRound size={14} />Inject a contamination event</div>
      <label className="field" style={{ marginTop: 0 }}>
        Fault type
        <select value={type} onChange={(e) => setType(e.target.value)}>
          {cfg.event_types.map((t) => (
            <option key={t.key} value={t.key}>
              {t.label}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        <span className="field-row">
          Severity <span className="badge">{severity.toFixed(1)}×</span>
        </span>
        <input type="range" min="0.5" max="2" step="0.1" value={severity}
               onChange={(e) => setSeverity(Number(e.target.value))} />
      </label>
      <label className="field">
        <span className="field-row">
          Develops over <span className="badge">{ramp} h</span>
        </span>
        <input type="range" min="1" max="6" step="1" value={ramp} onChange={(e) => setRamp(Number(e.target.value))} />
      </label>
      <div className="row">
        <button className="btn danger" onClick={() => onInject({ type, severity, ramp_hours: ramp })}>
          <AlertOctagon size={15} />
          Inject
        </button>
        <button className="btn" onClick={onClear} disabled={!injection}>
          Clear fault
        </button>
      </div>
      {injection && (
        <Note kind="err">
          Active fault: <b>{label(injection.type)}</b>, severity {injection.severity}×
        </Note>
      )}
    </div>
  );
}

function ReplayControls({ cfg, view, forecaster, running, setRunning, speed, setSpeed, showTruth, setShowTruth, run }) {
  const pos = view?.source === "replay" ? view : null;
  return (
    <>
      <div className="panel">
        <div className="panel-title"><Play size={14} />Playback</div>
        <div className="row" style={{ marginTop: 0 }}>
          <button className="btn primary" onClick={() => setRunning(!running)}>
            {running ? <Pause size={15} /> : <Play size={15} />}
            {running ? "Pause" : "Start"}
          </button>
          <button className="btn" disabled={running} onClick={() => run(() => api.replayStep(1, forecaster))}>
            <SkipForward size={15} />
            Step
          </button>
        </div>
        <label className="field">
          Simulated time per update
          <Segmented options={SPEEDS} value={speed} onChange={setSpeed} />
        </label>
        <Toggle label="Show ground-truth CWQI" checked={showTruth} onChange={setShowTruth}
                title="The true index of the simulated water, which the models never see." />
        {pos && (
          <>
            <div className="progress-label">
              <span>Unseen 60-day test segment</span>
              <span className="num">{Math.round((100 * pos.position) / pos.total)}%</span>
            </div>
            <div className="bar">
              <div style={{ width: `${(100 * pos.position) / pos.total}%`, background: "var(--accent)" }} />
            </div>
          </>
        )}
      </div>
      <EventJump cfg={cfg} onJump={async (ev) => {
        const s = await run(() => api.replayJump(ev, forecaster));
        if (s) setRunning(true);
      }} />
      <InjectForm
        cfg={cfg}
        injection={view?.source === "replay" ? view.injection : null}
        onInject={async (inj) => {
          const s = await run(() => api.replayInject(inj));
          if (s) setRunning(true);
        }}
        onClear={() => run(api.replayClear)}
      />
      <div className="panel">
        <button className="btn ghost wide" style={{ marginTop: 0 }} onClick={() => {
          setRunning(false);
          run(api.replayReset);
        }}>
          <RotateCcw size={15} />
          Reset to start of test period
        </button>
      </div>
    </>
  );
}

function LiveControls({ cfg, view, forecaster, showTruth, setShowTruth, autoRefresh, setAutoRefresh, run }) {
  const live = view?.source === "live" ? view : null;
  const gw = live?.gateway;
  const agent = live?.agent;
  // the API's own feeder agent reports directly; a separate sensor_publisher.py reports over MQTT
  const status = agent?.feeder ?? gw?.status;
  const send = (cmd) => run(() => api.gateway(cmd));
  const fcLabel = cfg.forecasters.find((f) => f.key === forecaster)?.label ?? forecaster;
  const age = status?.last_sent && live?.server_time ? Math.max(0, live.server_time - status.last_sent) : null;
  const label = (k) => cfg.event_types.find((t) => t.key === k)?.label ?? k;
  const natural = (status?.events ?? []).filter((e) => e.source === "natural");

  let banner;
  if (!live) banner = <Note kind="info">Connecting…</Note>;
  else if (gw?.error && !agent?.enabled) banner = <Note kind="err">{gw.error}</Note>;
  else if (!status)
    banner = (
      <Note kind="warn">
        No sensor feeder is publishing yet. Start the API with <code>LIVE_AGENT=1</code> (the default) or run{" "}
        <code>python sensor_publisher.py</code>.
      </Note>
    );
  else if (status.warming_up) banner = <Note kind="info">Feeder agent starting: sending the last 4 h of readings…</Note>;
  else
    banner = (
      <Note kind={status.paused ? "warn" : "ok"}>
        {agent?.enabled ? "Built-in feeder agent" : "Sensor gateway"} {status.paused ? "paused" : "publishing"} –
        one reading (5 plant minutes) every {status.interval >= 300 ? "5 min (real time)" : `${status.interval} s`}
        {age !== null && !status.paused ? ` · last ${age < 2 ? "just now" : `${Math.round(age)} s ago`}` : ""}
      </Note>
    );

  const paused = Boolean(status?.paused);
  const counts = live ? live.database.counts : null;
  const backlog = agent?.ingest?.backlog ?? 0;
  return (
    <>
      <div className="panel">
        <div className="panel-title"><Radio size={14} />Sensor feeder agent (MQTT)</div>
        {banner}
        <div className="row">
          <button className="btn primary" onClick={() => send({ cmd: paused ? "resume" : "pause" })} disabled={!status}>
            {paused ? <Play size={15} /> : <Pause size={15} />}
            {paused ? "Resume" : "Pause"}
          </button>
          <button className="btn" onClick={() => run(api.live)}>
            <RefreshCw size={15} />
            Refresh
          </button>
        </div>
        <label className="field" title="Every reading is 5 minutes of plant time. 'Real' sends one every 5 real minutes.">
          Seconds between readings
          <Segmented options={INTERVALS.map((s) => [s, s >= 300 ? "Real" : String(s)])} value={status?.interval}
                     onChange={(s) => send({ cmd: "interval", seconds: s })} />
        </label>
        {backlog > 5 && <Note kind="info">Ingestion is catching up: {backlog} readings waiting.</Note>}
        <Toggle label="Random contamination events" checked={status?.auto_events ?? true}
                onChange={(on) => send({ cmd: "auto_events", on })}
                title="The simulated plant starts a natural event now and then, like the training data." />
        <Toggle label="Show ground-truth CWQI" checked={showTruth} onChange={setShowTruth}
                title="Sent by the simulated feeder only; a real plant has no ground truth." />
        {showTruth && natural.length > 0 && (
          <Note kind="info">Simulation truth: a natural <b>{label(natural[0].type)}</b> event is developing.</Note>
        )}
        <Toggle label="Auto-refresh" checked={autoRefresh} onChange={setAutoRefresh} />
        <button className="btn wide" onClick={() => send({ cmd: "forecaster", name: forecaster })}>
          Use {fcLabel} in the ingestion service
        </button>
      </div>
      <InjectForm
        cfg={cfg}
        injection={status?.injection && !status.injection.clearing ? status.injection : null}
        onInject={(inj) => send({ cmd: "inject", ...inj })}
        onClear={() => send({ cmd: "clear" })}
      />
      <div className="panel">
        <div className="panel-title"><Database size={14} />Database</div>
        {counts && (
          <div className="db-line">
            <code>{live.database.backend}</code> · {counts.readings.toLocaleString()} readings ·{" "}
            {counts.assessments.toLocaleString()} assessments · {counts.alerts.toLocaleString()} alerts
          </div>
        )}
        <button className="btn ghost wide" onClick={async () => {
          if (!window.confirm("Delete all readings, assessments and alerts from the database?")) return;
          await run(api.clearDatabase);
          run(api.live);
        }}>
          <Trash2 size={15} />
          Clear database
        </button>
      </div>
    </>
  );
}

export default function Sidebar(props) {
  const { open, onClose, cfg, source, setSource, forecaster, setForecaster, sound, setSound } = props;
  return (
    <>
      <div className={open ? "scrim show" : "scrim"} onClick={onClose} />
      <aside className={open ? "sidebar open" : "sidebar"}>
        <div className="sidebar-head">
          <SlidersHorizontal size={17} />
          Control panel
          <button className="icon-btn close-btn" onClick={onClose} aria-label="Close control panel">
            <X size={16} />
          </button>
        </div>
        <div className="panel">
          <label className="field" style={{ marginTop: 0 }}>
            Data source
            <Segmented options={[["replay", "Built-in replay"], ["live", "Live (MQTT)"]]} value={source} onChange={setSource} />
          </label>
          <label className="field" title="LSTM (Section III-E) or the XGBoost lag-feature forecaster (Section V).">
            Forecaster
            <Segmented options={cfg.forecasters.map((f) => [f.key, f.label])} value={forecaster} onChange={setForecaster} />
          </label>
          <Toggle label="Pop-up and sound on alerts" checked={sound} onChange={setSound} />
        </div>
        {source === "replay" ? <ReplayControls {...props} /> : <LiveControls {...props} />}
      </aside>
    </>
  );
}
