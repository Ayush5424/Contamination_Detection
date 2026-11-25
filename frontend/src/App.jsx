import { useCallback, useEffect, useRef, useState } from "react";
import { Activity, BarChart3, Clock, Cpu, Droplet, FileSearch, Menu, Moon, Sun, Workflow } from "lucide-react";
import { api } from "./api.js";
import { beep } from "./notify.js";
import { fmtDayTime } from "./format.js";
import Sidebar from "./components/Sidebar.jsx";
import LiveMonitor from "./components/LiveMonitor.jsx";
import Performance from "./components/Performance.jsx";
import NewData from "./components/NewData.jsx";
import HowItWorks from "./components/HowItWorks.jsx";
import Toasts from "./components/Toasts.jsx";

const TABS = [
  ["live", "Live monitor", Activity],
  ["newdata", "Test on new data", FileSearch],
  ["perf", "Model performance", BarChart3],
  ["how", "How it works", Workflow],
];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function storedTheme() {
  try {
    return localStorage.getItem("wq-theme") || "light";
  } catch {
    return "light";
  }
}

export default function App() {
  const [cfg, setCfg] = useState(null);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState("live");
  const [theme, setTheme] = useState(storedTheme);
  const [source, setSource] = useState("live"); // live (sensor feeder agent over MQTT) | replay
  const [forecaster, setForecaster] = useState("lstm");
  const [sound, setSound] = useState(true);
  const [showTruth, setShowTruth] = useState(true);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(3); // readings per update: 1 = 5 simulated minutes
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [view, setView] = useState(null); // state returned by /dashboard/replay or /dashboard/live
  const [toasts, setToasts] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const seenAlert = useRef({});
  const dismissToast = useCallback((key) => setToasts((t) => t.filter((x) => x.key !== key)), []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("wq-theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);

  useEffect(() => {
    api
      .config()
      .then((c) => {
        setCfg(c);
        setForecaster(c.default_forecaster);
      })
      .catch((e) => setError(`Cannot reach the API: ${e.message}`));
  }, []);

  // pop-up + beep for alerts that are newer than the last one seen in this data source
  useEffect(() => {
    if (!view || view.source !== source) return;
    const ids = view.alerts.map((a) => a.id);
    const maxId = ids.length ? Math.max(...ids) : 0;
    const seen = seenAlert.current[source];
    seenAlert.current[source] = Math.max(seen ?? 0, maxId);
    if (seen === undefined) return; // first load: don't replay old alerts
    const fresh = view.alerts.filter((a) => a.id > seen).reverse();
    if (!fresh.length) return;
    setToasts((t) => [...t, ...fresh.map((a) => ({ key: `${source}-${a.id}`, title: a.title, level: a.level }))].slice(-4));
    if (sound && fresh.some((a) => a.level !== "Safe")) beep();
  }, [view, source, sound]);

  const run = useCallback(async (fn) => {
    try {
      const s = await fn();
      if (s && s.source) setView(s);
      setError(null);
      return s;
    } catch (e) {
      setError(e.message);
    }
  }, []);

  // load the chosen source once
  useEffect(() => {
    if (!cfg) return;
    setView(null);
    run(source === "replay" ? api.replay : api.live);
  }, [cfg, source, run]);

  // replay clock: advance `speed` readings every ~0.3 s while running
  useEffect(() => {
    if (!cfg || source !== "replay" || !running) return;
    let stop = false;
    (async () => {
      while (!stop) {
        try {
          const s = await api.replayStep(speed, forecaster);
          if (stop) break;
          setView(s);
          setError(null);
          if (s.finished) {
            setRunning(false);
            break;
          }
        } catch (e) {
          setError(e.message);
          setRunning(false);
          break;
        }
        await sleep(300);
      }
    })();
    return () => {
      stop = true;
    };
  }, [cfg, source, running, speed, forecaster]);

  // live: poll the database every second
  useEffect(() => {
    if (!cfg || source !== "live" || !autoRefresh) return;
    let stop = false;
    (async () => {
      while (!stop) {
        await sleep(1000);
        if (stop) break;
        try {
          const s = await api.live();
          if (!stop) {
            setView(s);
            setError(null);
          }
        } catch (e) {
          if (!stop) setError(e.message);
        }
      }
    })();
    return () => {
      stop = true;
    };
  }, [cfg, source, autoRefresh]);

  if (!cfg) {
    return (
      <div className="splash">
        {error ? (
          <div className="callout alert" style={{ maxWidth: 520 }}>
            <div>
              <b>{error}</b>
              <p>
                Start the API first: <code>python -m uvicorn api:app --port 8765</code>. Loading the models takes about a
                minute.
              </p>
            </div>
          </div>
        ) : (
          <>
            <div className="spinner" />
            Loading the models…
          </>
        )}
      </div>
    );
  }

  const fcLabel = cfg.forecasters.find((f) => f.key === forecaster)?.label ?? forecaster;
  const isLive = source === "live";
  const feederStatus = view?.source === "live" ? view.agent?.feeder ?? view.gateway?.status : null;
  const active = isLive ? Boolean(feederStatus && !feederStatus.paused && (view.agent?.feeder || view.gateway?.connected)) : running;
  const simTime = view?.source === source && view.last ? view.last.time : null;

  return (
    <div className="layout">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        cfg={cfg}
        view={view}
        source={source}
        setSource={(s) => {
          setRunning(false);
          setSource(s);
        }}
        forecaster={forecaster}
        setForecaster={setForecaster}
        sound={sound}
        setSound={setSound}
        showTruth={showTruth}
        setShowTruth={setShowTruth}
        running={running}
        setRunning={setRunning}
        speed={speed}
        setSpeed={setSpeed}
        autoRefresh={autoRefresh}
        setAutoRefresh={setAutoRefresh}
        run={run}
      />
      <div className="main">
        <header className="topbar">
          <button className="icon-btn menu-btn" onClick={() => setSidebarOpen(true)} aria-label="Open control panel">
            <Menu size={18} />
          </button>
          <div className="brand">
            <div className="logo">
              <Droplet size={20} strokeWidth={2.4} />
            </div>
            <div style={{ minWidth: 0 }}>
              <div className="brand-title">Cooling-Water Quality Monitor</div>
              <div className="brand-sub">AI-based contamination risk prediction for data center cooling systems</div>
            </div>
          </div>
          <div className="topbar-right">
            <span className="chip" title="Data source">
              <span className={active ? "dot on" : "dot"} />
              <b>{isLive ? "Live · MQTT" : "Replay"}</b>
              {active ? "running" : "paused"}
            </span>
            <span className="chip hide-sm" title="Forecaster">
              <Cpu size={13} />
              <b>{fcLabel}</b>
            </span>
            {simTime && (
              <span className="chip hide-sm" title="Time of the latest reading (simulated plant time)">
                <Clock size={13} />
                <b className="num">{fmtDayTime(simTime)}</b>
              </span>
            )}
            <button className="icon-btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                    aria-label="Switch light / dark theme" title="Switch light / dark theme">
              {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
            </button>
          </div>
        </header>
        <div className="content">
          <nav className="tabs">
            {TABS.map(([k, label, Icon]) => (
              <button key={k} className={tab === k ? "tab active" : "tab"} onClick={() => setTab(k)}>
                <Icon size={15} />
                {label}
              </button>
            ))}
          </nav>
          {error && (
            <div className="callout alert" style={{ marginTop: 0, marginBottom: 16 }}>
              {error}
            </div>
          )}
          {tab === "live" && <LiveMonitor cfg={cfg} view={view} source={source} showTruth={showTruth} />}
          {tab === "newdata" && <NewData cfg={cfg} forecaster={forecaster} />}
          {tab === "perf" && <Performance />}
          {tab === "how" && <HowItWorks />}
        </div>
      </div>
      <Toasts toasts={toasts} dismiss={dismissToast} />
    </div>
  );
}
