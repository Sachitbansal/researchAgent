// Vite config for the browser UI.
// In dev (`npm run dev`) the page is served by Vite and every /api request is proxied to
// the Python server, so the browser sees one origin and the backend needs no CORS.
// `npm run build` writes frontend/dist/, which web/api.py serves directly.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Declared here rather than pulling in @types/node for one variable.
declare const process: { env: Record<string, string | undefined> };
const backend = process.env.SCIAGENT_BACKEND ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: backend, changeOrigin: false } },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
