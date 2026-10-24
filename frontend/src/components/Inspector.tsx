/* Right pane: how the selected answer was made — tool calls, sources, and memory.
   In:  one assistant message (stored or live) and the corpus's papers.
   Out: a turn-by-turn timeline of the loop's events with the run's final stats; the
        passages the answer cites; and what this answer leaves in the chat's short-term
        memory (its tool-call notes) and how much earlier history it was given. */

import { useEffect, useMemo, useRef, useState } from "react";
import type { AssistantMessage, Paper, Step } from "../types";
import { renderMarkdown } from "../markdown";
import { argLine, excerpts, humanise, outcome, plural, secs, who } from "../format";
import { Icon } from "../icons";

export type InspectorTab = "calls" | "sources" | "memory";

interface Props {
  open: boolean;
  message: AssistantMessage | null;
  isLatest: boolean;
  sample: boolean;
  papersById: Map<string, Paper>;
  tab: InspectorTab;
  setTab: (t: InspectorTab) => void;
  highlight: string | null;
  onClose: () => void;
}

export function Inspector(p: Props) {
  return (
    <aside className={`inspector${p.open ? " is-open" : ""}`} aria-label="The agent's working" aria-hidden={!p.open}>
      <div className="inspector-head">
        <div className="tabs" role="tablist">
          {(["calls", "sources", "memory"] as const).map((t) => (
            <button key={t} role="tab" aria-selected={p.tab === t} className={`tab${p.tab === t ? " is-on" : ""}`} onClick={() => p.setTab(t)}>
              {t === "calls" ? "Tool calls" : t === "sources" ? "Sources" : "Memory"}
            </button>
          ))}
        </div>
        <button className="icon-btn" onClick={p.onClose} aria-label="Hide the side panel"><Icon name="close" size={16} /></button>
      </div>
      {p.message && !p.isLatest && <div className="inspector-scope">Showing an earlier answer in this chat.</div>}
      <div className="inspector-body">
        {p.tab === "memory" ? (
          <Memory message={p.message} sample={p.sample} />
        ) : !p.message ? (
          <p className="quiet">Ask a question and each step the agent takes shows up here as it happens.</p>
        ) : p.tab === "calls" ? (
          <Timeline message={p.message} />
        ) : (
          <Sources message={p.message} papersById={p.papersById} highlight={p.highlight} />
        )}
      </div>
    </aside>
  );
}

// ------------------------------------------------------------------ timeline

type Row =
  | { type: "tool"; start: Step; end?: Step }
  | { type: "skipped" | "image" | "cap" | "answering"; step: Step };

function groupByTurn(steps: Step[]): { iteration: number; t_ms?: number; rows: Row[] }[] {
  const turns: { iteration: number; t_ms?: number; rows: Row[] }[] = [];
  const turn = (it: number, t?: number) => {
    let last = turns[turns.length - 1];
    if (!last || last.iteration !== it) {
      last = { iteration: it, t_ms: t, rows: [] };
      turns.push(last);
    }
    return last;
  };
  for (const s of steps) {
    const t = turn(s.iteration ?? 0, s.t_ms);
    if (s.kind === "thinking") continue;
    if (s.kind === "tool_start") t.rows.push({ type: "tool", start: s });
    else if (s.kind === "tool_end") {
      const open = [...t.rows].reverse().find((r) => r.type === "tool" && !r.end && r.start.tool_name === s.tool_name) as
        | Extract<Row, { type: "tool" }>
        | undefined;
      if (open) open.end = s;
      else t.rows.push({ type: "tool", start: { ...s, kind: "tool_start" }, end: s });
    } else if (s.kind === "skipped" || s.kind === "image" || s.kind === "cap" || s.kind === "answering") {
      t.rows.push({ type: s.kind, step: s });
    }
  }
  return turns;
}

function Timeline({ message }: { message: AssistantMessage }) {
  const turns = useMemo(() => groupByTurn(message.steps), [message.steps]);
  const running = message.status === "running";
  const last = message.steps[message.steps.length - 1];
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (running) bottom.current?.scrollIntoView({ block: "nearest" });
  }, [running, message.steps.length]);

  return (
    <div className="timeline">
      {turns.length === 0 && running && <p className="quiet">Starting…</p>}
      {turns.map((t) => (
        <section key={t.iteration} className="turn">
          <h3 className="turn-head">
            Turn {t.iteration + 1}
            {t.t_ms != null && <span className="turn-time">at {secs(t.t_ms)}</span>}
          </h3>
          {t.rows.length === 0 && !(running && last?.kind === "thinking" && last.iteration === t.iteration) && (
            <p className="quiet small">Thought, called no tools.</p>
          )}
          {t.rows.map((r, i) => <TimelineRow key={i} row={r} />)}
          {running && last?.kind === "thinking" && last.iteration === t.iteration && (
            <div className="row is-live"><span className="row-dot" /><span className="row-name">thinking</span></div>
          )}
        </section>
      ))}
      <div ref={bottom} />
      {!running && <RunStats message={message} />}
    </div>
  );
}

function TimelineRow({ row }: { row: Row }) {
  const [open, setOpen] = useState(false);

  if (row.type === "tool") {
    const { start, end } = row;
    const failed = !!(end?.result as any)?.error;
    return (
      <div className={`row tool-${start.tool_name}${end ? "" : " is-live"}${failed ? " is-failed" : ""}`}>
        <span className="row-dot" />
        <div className="row-main">
          <button className="row-toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
            <span className="row-name">{start.tool_name}</span>
            <span className="row-ms">{end ? secs(end.latency_ms ?? 0) : "working"}</span>
          </button>
          {start.args && Object.keys(start.args).length > 0 && <code className="row-args">{argLine(start.args, 70)}</code>}
          {end && <div className="row-out">{outcome(end.result)}</div>}
          {open && (
            <div className="row-detail">
              <div className="detail-label">Arguments</div>
              <pre>{JSON.stringify(start.args ?? {}, null, 2)}</pre>
              {end && (
                <>
                  <div className="detail-label">Result, as summarised for the trace</div>
                  <pre>{JSON.stringify(end.result ?? {}, null, 2)}</pre>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }
  const s = row.step;
  if (row.type === "skipped") {
    return (
      <div className="row is-skipped"><span className="row-dot" />
        <div className="row-main"><span className="row-name">{s.tool_name || "call"} skipped</span><div className="row-out">{s.reason}</div></div>
      </div>
    );
  }
  if (row.type === "image") {
    return (
      <div className="row tool-inspect_figure"><span className="row-dot" />
        <div className="row-main">
          <span className="row-name">looked at the figure</span>
          {s.image_url ? <img className="row-img" src={s.image_url} alt="" loading="lazy" /> : <div className="row-out">{s.image_path}</div>}
        </div>
      </div>
    );
  }
  if (row.type === "cap") {
    return (
      <div className="row is-cap"><span className="row-dot" />
        <div className="row-main"><span className="row-name">stopped calling tools</span>
          <div className="row-out">{s.reason || "reached its step limit, answering with what it has"}</div></div>
      </div>
    );
  }
  return (
    <div className="row is-answer"><span className="row-dot" />
      <div className="row-main"><span className="row-name">wrote the answer</span></div>
    </div>
  );
}

function RunStats({ message }: { message: AssistantMessage }) {
  const s = message.stats ?? {};
  const rows: [string, string][] = [];
  if (s.iterations != null) rows.push(["Turns", String(s.iterations)]);
  if (s.tool_calls != null) rows.push(["Tool calls", String(s.tool_calls)]);
  if (s.context_tokens != null) rows.push(["Context", `${s.context_tokens.toLocaleString()} tokens`]);
  if (message.elapsed_ms != null) rows.push(["Wall time", secs(message.elapsed_ms)]);
  if (s.stopped_because) rows.push(["Stopped because", humanise(s.stopped_because)]);
  if (s.history_turns != null) {
    const dropped = s.history_turns_dropped ? `, ${s.history_turns_dropped} older left out` : "";
    rows.push(["Earlier turns sent", `${s.history_turns}${dropped}`]);
  }
  if (message.error) rows.push(["Error", `${humanise(message.error.error)}${message.error.detail ? `: ${message.error.detail}` : ""}`]);
  if (s.run_id) rows.push(["Trace", `logs/traces, run ${s.run_id}`]);
  if (!rows.length) return null;
  return (
    <dl className="stats">
      {rows.map(([k, v]) => (
        <div key={k} className="stat"><dt>{k}</dt><dd>{v}</dd></div>
      ))}
    </dl>
  );
}

// ------------------------------------------------------------------ sources

function Sources({ message, papersById, highlight }: { message: AssistantMessage; papersById: Map<string, Paper>; highlight: string | null }) {
  const { refs } = useMemo(() => renderMarkdown(message.content), [message.content]);
  const found = useMemo(() => excerpts(message.steps), [message.steps]);
  const cited = new Set(refs.flatMap((r) => r.chunks));
  const uncited = [...found.keys()].filter((id) => !cited.has(id));
  const marked = useRef<HTMLLIElement>(null);

  useEffect(() => {
    marked.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [highlight]);

  if (message.status === "running") return <p className="quiet">Sources appear once the answer is written.</p>;
  if (!refs.length) return <p className="quiet">This answer cites no passages.</p>;

  return (
    <div className="sources">
      {refs.map((r) => {
        const paper = papersById.get(r.paper);
        return (
          <section key={r.paper} className="source">
            <div className="source-head">
              <span className="ref-n">{r.n}</span>
              <div>
                <div className="source-title">{paper?.title ?? r.paper}</div>
                {paper && <div className="source-who">{who(paper)}</div>}
                {paper?.pdf_url && <a className="ref-link" href={paper.pdf_url} target="_blank" rel="noopener noreferrer">arXiv</a>}
              </div>
            </div>
            <ul className="passages">
              {r.chunks.map((c) => (
                <li key={c} ref={c === highlight ? marked : undefined} className={`passage${c === highlight ? " is-marked" : ""}`}>
                  <code className="passage-id">{c}</code>
                  <p className="passage-text">
                    {found.get(c)?.text ? <mark>{found.get(c)!.text}…</mark> : <span className="quiet">No excerpt in this run's steps.</span>}
                  </p>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
      {uncited.length > 0 && (
        <p className="quiet small">
          {plural(uncited.length, "more passage")} retrieved but not cited. Excerpts are cut to the trace length set in config.yaml.
        </p>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ memory

/* Short-term memory, per chat (docs/DECISIONS.md D12). What a later question in this
   chat is sent from this answer: the question, the answer, and these notes. */
function Memory({ message, sample }: { message: AssistantMessage | null; sample: boolean }) {
  const notes = message?.notes ?? [];
  const s = message?.stats ?? {};
  return (
    <div className="memory">
      <h3>What later questions see</h3>
      <p>
        Each new question in this chat is sent with the earlier questions and answers, and a one-line note for each tool
        call. The passages themselves are not carried over, so the agent retrieves them again. Once the history passes its
        budget, the oldest turns are left out first.
      </p>
      {sample ? (
        <p className="quiet small">Samples are fixtures and are never sent to the agent.</p>
      ) : !message ? (
        <p className="quiet small">Nothing yet. Ask a question to start this chat.</p>
      ) : message.status === "running" ? (
        <p className="quiet small">The notes for this answer are saved when it finishes.</p>
      ) : (
        <>
          <div className="detail-label">Notes kept from this answer</div>
          {notes.length ? (
            <ul className="memory-notes">
              {notes.map((n, i) => <li key={i}><code>{n}</code></li>)}
            </ul>
          ) : (
            <p className="quiet small">None: this answer called no tools.</p>
          )}
          {s.history_turns != null && (
            <p className="quiet small">
              This answer was written with {plural(s.history_turns, "earlier turn")} in view
              {s.history_turns_dropped ? `, and ${s.history_turns_dropped} older left out` : ""}.
            </p>
          )}
        </>
      )}
      <p className="quiet small">Nothing is remembered from one chat to another.</p>
    </div>
  );
}
