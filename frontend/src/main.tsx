/* Entry point: mounts the app into #root.
   In:  index.html's root element. Out: the React tree, with KaTeX's stylesheet (math in
        answers) and ours (Tailwind, src/styles.css) loaded. */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "katex/dist/katex.min.css";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
