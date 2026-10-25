// Vite config for the browser UI.
// In dev (`npm run dev`) the page is served by Vite and every /api request is proxied to
// the Python server, so the browser sees one origin and the backend needs no CORS.
// `npm run build` writes frontend/dist/, which web/api.py serves directly. Tailwind v4
// runs as a Vite plugin, reading its tokens from src/styles.css (docs/DECISIONS.md, D8).
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Declared here rather than pulling in @types/node for one variable.
declare const process: { env: Record<string, string | undefined> };
const backend = process.env.SCIAGENT_BACKEND ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { "/api": { target: backend, changeOrigin: false } },
  },
  // KaTeX is about 270 kB of the ~570 kB bundle (docs/DECISIONS.md, D7); one local page,
  // so one chunk is fine and the default 500 kB warning would fire on every build.
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 700 },
});
