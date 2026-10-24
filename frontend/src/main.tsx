/* Entry point: mounts the app into #root.
   In:  index.html's root element. Out: the React tree, with the stylesheet loaded. */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
