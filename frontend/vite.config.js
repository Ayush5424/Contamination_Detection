import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the page on :5173 and forwards API calls to the FastAPI server on :8765.
// `npm run build` writes frontend/dist, which api.py serves at http://127.0.0.1:8765/.
const API = process.env.API_URL || "http://127.0.0.1:8765";
const proxied = ["/dashboard", "/metrics", "/parameters", "/info", "/health", "/latest",
                 "/readings", "/assessments", "/alerts", "/predict", "/forecaster", "/docs", "/openapi.json"];

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: Object.fromEntries(proxied.map((p) => [p, API])) },
});
