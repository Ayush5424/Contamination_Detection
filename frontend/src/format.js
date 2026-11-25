// Level colours come from CSS variables, so they follow the light / dark theme.
export const COLORS = { Safe: "var(--safe)", Warning: "var(--warn)", Alert: "var(--alert)" };
export const LEVEL_CLASS = { Safe: "safe", Warning: "warn", Alert: "alert" };
export const levelVars = (level) => {
  const k = LEVEL_CLASS[level] ?? "safe";
  return { "--lvl": `var(--${k})`, "--lvl-soft": `var(--${k}-soft)`, "--lvl-line": `var(--${k}-line)` };
};

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const pad = (n) => String(n).padStart(2, "0");

// API times are naive ISO strings (simulated plant time); parse them as local time.
export const toMs = (iso) => new Date(iso).getTime();

export function fmtTime(iso) {
  const d = new Date(iso);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function fmtDay(iso) {
  const d = new Date(iso);
  return `${pad(d.getDate())} ${MONTHS[d.getMonth()]} ${d.getFullYear()}`;
}

export function fmtDayTime(iso) {
  const d = new Date(iso);
  return `${pad(d.getDate())} ${MONTHS[d.getMonth()]} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export const num = (v, digits = 1) => (v === null || v === undefined ? "–" : Number(v).toFixed(digits));

export const levelOf = (v, limits) => (v >= limits.alert ? "Alert" : v >= limits.warning ? "Warning" : "Safe");

// Equation (1) of the paper: rating 0 at the ideal value, 100 at the limit on that side.
export function subIndex(spec, v) {
  if (v === null || v === undefined) return null;
  if (spec.high !== null && v > spec.ideal) return (100 * (v - spec.ideal)) / (spec.high - spec.ideal);
  if (spec.low !== null && v < spec.ideal) return (100 * (spec.ideal - v)) / (spec.ideal - spec.low);
  return 0;
}

// Only the latest continuous stretch of readings: after a jump (or old data in the database)
// the stored times are not continuous, and joining them would draw misleading lines.
export function continuous(log) {
  let start = log.length - 1;
  while (start > 0) {
    const gap = toMs(log[start].time) - toMs(log[start - 1].time);
    if (gap <= 0 || gap > 3600000) break;
    start -= 1;
  }
  return log.slice(Math.max(0, start));
}
