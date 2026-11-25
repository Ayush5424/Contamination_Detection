import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Bell, Crosshair, Gauge, Radar, Search, ShieldCheck, Target, TrendingUp } from "lucide-react";
import { api } from "../api.js";
import { num } from "../format.js";

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

const tooltipStyle = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10 };

function ChartCard({ icon: Icon, title, sub, data, highlight, digits = 2, height = 230 }) {
  return (
    <div className="card">
      <div className="card-head">
        <div>
          <div className="card-title"><Icon size={17} />{title}</div>
          <div className="card-sub">{sub}</div>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 18, right: 8, bottom: 0, left: -18 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 11, fill: "var(--text)" }} tickLine={false} axisLine={{ stroke: "var(--border)" }} interval={0} />
          <YAxis tick={{ fontSize: 11, fill: "var(--muted)" }} tickLine={false} axisLine={false} />
          <Tooltip cursor={{ fill: "var(--grid)" }} formatter={(v) => num(v, digits)} contentStyle={tooltipStyle} />
          <ReferenceLine y={0} stroke="var(--border-strong)" />
          <Bar dataKey="v" radius={[5, 5, 0, 0]} maxBarSize={56} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.name} fill={highlight(d) ? "var(--accent)" : "var(--faint)"} fillOpacity={highlight(d) ? 0.95 : 0.55} />
            ))}
            <LabelList dataKey="v" position="top" formatter={(v) => num(v, digits)} style={{ fontSize: 11, fill: "var(--text)" }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Performance() {
  const [m, setM] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    api.metrics().then(setM).catch((e) => setError(e.message));
  }, []);
  if (error) return <div className="callout alert">{error}</div>;
  if (!m) return <div className="splash" style={{ minHeight: 300 }}><div className="spinner" /></div>;

  const est = m.estimation["Hybrid stacked ensemble"];
  const raw = m.estimation["Index formula on raw readings"];
  const fc = m.forecast_60min_rmse;
  const fcData = [
    { name: "Persistence", v: fc["Persistence (no change)"] },
    { name: "LSTM", v: fc.LSTM },
    ...("XGBoost (lag features)" in fc ? [{ name: "XGBoost", v: fc["XGBoost (lag features)"] }] : []),
  ];
  const bestFc = Math.min(...fcData.map((d) => d.v));
  const estData = [
    { name: "Formula, raw", v: raw.rmse },
    { name: "Hybrid ensemble", v: est.rmse },
  ];
  const short = { XGBoost: "XGB", CatBoost: "CatB", "Random Forest": "RF", "Gradient Boosting": "GB", "Extra Trees": "ET",
                  AdaBoost: "Ada" };
  const weights = Object.entries(m.meta_weights).map(([name, v]) => ({ name: short[name] ?? name, v }));
  const errCut = Math.round(100 * (1 - est.rmse / raw.rmse));

  return (
    <div>
      <div className="card-head" style={{ marginBottom: 16 }}>
        <div>
          <div className="card-title" style={{ fontSize: "1.1rem" }}>Results on the unseen {m.test_days}-day test segment</div>
          <div className="card-sub">
            Models trained with {m.trained_with} in {m.training_minutes} min. The paper's tables used the full settings
            (<code>python train_models.py --paper</code>).
          </div>
        </div>
      </div>
      <div className="perf-kpis">
        <Tile icon={Target} label="R² · hybrid CWQI estimate" value={num(est.r2, 4)} sub={`RMSE ${num(est.rmse, 2)} · MAE ${num(est.mae, 2)}`} />
        <Tile icon={Crosshair} label="RMSE when contaminated (CWQI ≥ 50)" value={num(est.rmse_contaminated, 2)}
              sub={`bias ${est.bias_contaminated >= 0 ? "+" : ""}${num(est.bias_contaminated, 2)} – no under-estimation`} />
        <Tile icon={Search} label="Root cause identified (SHAP)" value={m.root_cause_top1} sub="events, top-1 parameter" tone="safe" />
        <Tile icon={Radar} label="Anomaly detection F1" value={num(m.anomaly.f1, 2)} sub={`events caught ${m.anomaly.events_detected}`} />
        <Tile icon={Bell} label="Early-warning macro-F1" value={num(m.early_warning.macro_f1, 3)} sub="Safe / Warning / Alert, 60-min horizon" />
        <Tile icon={ShieldCheck} label="False-alarm rate" value={`${num(m.early_warning.false_alarm_rate * 100, 2)}%`} sub="of Safe instants" tone="safe" />
      </div>
      <div className="perf-charts">
        <ChartCard icon={TrendingUp} title="60-min forecast error" sub="RMSE in index points · lower is better"
                   data={fcData} highlight={(d) => d.v === bestFc} />
        <ChartCard icon={Gauge} title="CWQI estimation error" sub={`RMSE · hybrid cuts the error by ${errCut}% vs. the raw formula`}
                   data={estData} highlight={(d) => d.name === "Hybrid ensemble"} />
        <ChartCard icon={Target} title="Meta-learner weights β" sub="Trust in each learner (eq. 3): XGB XGBoost · CatB CatBoost · RF Random Forest · GB Gradient Boosting · ET Extra Trees · Ada AdaBoost"
                   data={weights} highlight={(d) => d.v > 0} digits={2} />
      </div>
    </div>
  );
}
