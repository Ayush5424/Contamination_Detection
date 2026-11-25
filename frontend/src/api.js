// Thin wrapper around the FastAPI endpoints (same origin; Vite proxies them in development).

async function request(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

const get = (path) => request("GET", path);
const post = (path, body) => request("POST", path, body);

export const api = {
  config: () => get("/dashboard/config"),
  metrics: () => get("/metrics"),

  replay: () => get("/dashboard/replay"),
  replayStep: (n, forecaster) => post(`/dashboard/replay/step?n=${n}&forecaster=${forecaster}`),
  replayReset: () => post("/dashboard/replay/reset"),
  replayJump: (event, forecaster) => post(`/dashboard/replay/jump/${event}?forecaster=${forecaster}`),
  replayInject: (inj) => post("/dashboard/replay/inject", inj),
  replayClear: () => post("/dashboard/replay/clear"),

  newDataCsv: (csv, forecaster) => post("/dashboard/newdata/csv", { csv, forecaster }),
  newDataSimulate: (sim) => post("/dashboard/newdata/simulate", sim),

  live: () => get("/dashboard/live"),
  gateway: (cmd) => post("/dashboard/gateway", cmd),
  clearDatabase: () => request("DELETE", "/dashboard/database"),
};
