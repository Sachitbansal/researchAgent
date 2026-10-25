/* Small, pure formatting helpers shared by the panes.
   In:  papers, steps, numbers and timestamps. Out: short human-readable strings — who
        wrote a paper, what a tool call returned, what the agent is doing right now.
   Nothing here knows any topic; every label is built from data the server sent. */

import type { Paper, Step } from "./types";

export const plural = (n: number, noun: string, many = `${noun}s`) => `${n} ${n === 1 ? noun : many}`;

export function who(p: Pick<Paper, "authors" | "year">): string {
  const names = p.authors || [];
  const lead = names.length > 2 ? `${names[0]} and ${names.length - 1} others` : names.join(" and ");
  return [lead, p.year].filter(Boolean).join(", ");
}

export const secs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`);

export const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

/* check_evidence_consistency is handed the whole draft answer, so every value is clipped. */
export const argLine = (args?: Record<string, unknown>, width = 60) =>
  Object.entries(args || {})
    .map(([k, v]) => `${k}=${clip(typeof v === "string" ? JSON.stringify(v) : JSON.stringify(v) ?? "", width)}`)
    .join("  ");

/* Each tool returns a different dict, so read whichever count field it has. */
export function outcome(result?: Record<string, unknown>): string {
  if (!result || typeof result !== "object") return "done";
  const r = result as Record<string, any>;
  if (r.error) return `failed: ${r.error}${r.detail ? `, ${clip(String(r.detail), 90)}` : ""}`;
  if (Array.isArray(r.chunks)) {
    return plural(r.chunks.length, "passage") + (r.sufficient_evidence === false ? ", not enough to answer" : "");
  }
  if (Array.isArray(r.papers_added)) {
    // The admission gate's verdict is part of the outcome: what it turned away, and when
    // it turned away everything, that arXiv does not cover the subject well (D16).
    const rejected = Array.isArray(r.papers_rejected) ? r.papers_rejected.length : 0;
    const parts = [`${plural(r.papers_added.length, "paper")} added`, plural(r.chunks_added || 0, "chunk")];
    if (rejected) parts.push(`${rejected} rejected by the admission gate`);
    if (r.arxiv_coverage === "poor") parts.push("arXiv covers this poorly");
    return parts.join(", ");
  }
  if (Array.isArray(r.claims)) {
    return `${plural(r.claims.length, "claim")} checked, ${Math.round((r.grounded_ratio || 0) * 100)}% grounded`;
  }
  if ("found" in r) return r.found ? "found a disagreement to look at" : "no disagreement found";
  if (r.summary) return clip(String(r.summary), 110);
  if (r.image_path) return "figure opened";
  return "done";
}

/* "iteration_cap" -> "iteration cap". Generic on purpose: new stop reasons just work. */
export const humanise = (s?: string | null) => (s ? s.replace(/_/g, " ") : "");

export function relTime(ts: number): string {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 86400 * 7) return `${Math.floor(s / 86400)} d ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

/* What the agent is doing right now, read off the last step it announced. */
export function liveStatus(steps: Step[]): string {
  const last = steps[steps.length - 1];
  if (!last) return "Starting";
  const turn = typeof last.iteration === "number" ? ` (turn ${last.iteration + 1})` : "";
  switch (last.kind) {
    case "thinking":
      return `Thinking${turn}`;
    case "tool_start":
      return `Calling ${last.tool_name}`;
    case "tool_end":
      return `${last.tool_name} returned ${outcome(last.result)}`;
    case "image":
      return "Looking at a figure";
    case "skipped":
      return `Skipped ${last.tool_name || "a call"}: ${last.reason}`;
    case "cap":
      return last.reason ? `Stopped calling tools: ${last.reason}` : "Out of steps, answering with what it has";
    case "answering":
      return "Writing the answer";
    case "memory":
      return `Recalled ${plural(last.memories?.length ?? 0, "finding")} from earlier chats`;
    default:
      return humanise(last.kind);
  }
}

/* chunk_id -> the excerpt retrieval returned for it, from this answer's own steps. A
   `memory` step counts too: its passages were in the agent's view, so they are citable
   (D19), and `tool` says "memory" so the panes can say where the passage came from. */
export function excerpts(steps: Step[]): Map<string, { text: string; score?: number; tool: string }> {
  const out = new Map<string, { text: string; score?: number; tool: string }>();
  for (const s of steps) {
    const chunks = (s.result as any)?.chunks;
    if ((s.kind !== "tool_end" && s.kind !== "memory") || !Array.isArray(chunks)) continue;
    for (const c of chunks) {
      if (c?.chunk_id && !out.has(c.chunk_id)) {
        out.set(c.chunk_id, { text: String(c.text ?? ""), score: c.score, tool: s.kind === "memory" ? "memory" : s.tool_name || "" });
      }
    }
  }
  return out;
}

/* The figures this answer looked at, with the caption inspect_figure returned. */
export function figures(steps: Step[]): { url: string | null; path?: string; caption?: string; id?: string }[] {
  const out: { url: string | null; path?: string; caption?: string; id?: string }[] = [];
  steps.forEach((s, i) => {
    if (s.kind !== "image") return;
    // The matching tool_end comes before the image event, in the same turn.
    const source = steps
      .slice(0, i)
      .reverse()
      .find((p) => p.kind === "tool_end" && (p.result as any)?.image_path === s.image_path);
    const r = (source?.result ?? {}) as Record<string, any>;
    out.push({ url: s.image_url ?? null, path: s.image_path, caption: r.caption, id: r.figure_id });
  });
  return out;
}
