/* Right pane: how the selected answer was made — tool calls, sources, and memory.
   In:  one assistant message (stored or live), the corpus's papers, and the long-term
        memory list from the API (re-read whenever memoryVersion changes).
   Out: a turn-by-turn timeline of the loop's events with the run's final stats; the
        passages the answer cites; long-term memory (what was used in and written from
        this answer, every active memory, the forgotten ones and why); and what this
        answer leaves in the chat's short-term memory. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "../api";
import type { AssistantMessage, MemoryRecord, MemoryReport, Paper, Step } from "../types";
import { renderMarkdown } from "../markdown";
import { argLine, excerpts, humanise, outcome, plural, relTime, secs, who } from "../format";
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
  memoryVersion: number;
  onShowSource: (chunk: string) => void;
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
          <Memory message={p.message} sample={p.sample} version={p.memoryVersion} onShowSource={p.onShowSource} />
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
  | { type: "skipped" | "image" | "cap" | "answering" | "memory"; step: Step };

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
    } else if (s.kind === "skipped" || s.kind === "image" || s.kind === "cap" || s.kind === "answering" || s.kind === "memory") {
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
  if (row.type === "memory") {
    const n = s.memories?.length ?? 0;
    const prefs = s.preferences?.length ?? 0;
    return (
      <div className="row is-memory"><span className="row-dot" />
        <div className="row-main">
          <span className="row-name">recalled from earlier chats</span>
          <div className="row-out">
            {[n ? plural(n, "finding") : "", prefs ? plural(prefs, "preference") : ""].filter(Boolean).join(", ")}
            {s.result && Array.isArray((s.result as any).chunks) && (s.result as any).chunks.length
              ? `, with ${plural((s.result as any).chunks.length, "passage")}` : ""}
          </div>
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
                  {found.get(c)?.tool === "memory" && <span className="mem-tag">from memory</span>}
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

const FORGET_REASON: Record<string, string> = {
  superseded: "Replaced by a newer finding that contradicts it",
  stale: "Its passage changed or left the index",
  decay: "Faded: not used for a long time",
  user: "Deleted by you",
};

const SKIPPED: Record<string, string> = {
  abstained: "Nothing written: no retrieval passed the relevance check, so the answer rests on no evidence.",
  no_new_retrieval: "Nothing written: this answer ran no new retrieval, so it found nothing new to remember.",
  run_failed: "Nothing written: the run failed.",
  empty_answer: "Nothing written: the answer is empty.",
};

/* Long-term memory across chats (D19, D20), then this chat's short-term notes (D12). */
function Memory({ message, sample, version, onShowSource }: {
  message: AssistantMessage | null; sample: boolean; version: number; onShowSource: (chunk: string) => void;
}) {
  const [all, setAll] = useState<MemoryRecord[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setAll(await api.listMemories(true));
      setListError(null);
    } catch (e) {
      setListError(e instanceof api.ApiError ? e.detail : "The memory list could not be loaded.");
    }
  }, []);

  useEffect(() => {
    if (!sample) load();
  }, [sample, version, load]);

  const remove = async (id: string) => {
    setConfirming(null);
    try {
      await api.deleteMemory(id);
    } catch (e) {
      setListError(e instanceof api.ApiError ? `Not deleted: ${e.detail}` : "Not deleted.");
    }
    load();
  };

  const injected = message?.steps.find((s) => s.kind === "memory");
  const active = (all ?? []).filter((m) => m.active);
  const inactive = (all ?? []).filter((m) => !m.active);

  return (
    <div className="memory">
      <h3>Across chats</h3>
      <p>
        Findings from earlier answers that their own passages support are remembered across chats. A later question
        on the same subject is given the relevant ones with those passages, to check and cite. Unused memories fade,
        and a newer finding that contradicts one replaces it.
      </p>
      {sample ? (
        <p className="quiet small">Samples are fixtures and are never sent to the agent.</p>
      ) : (
        <>
          {message && (
            <>
              <UsedInAnswer step={injected} report={message.memory} onShowSource={onShowSource} />
              <WrittenFromAnswer message={message} />
            </>
          )}

          <section className="mem-section">
            <div className="mem-head">
              <div className="detail-label">All memories{all ? ` (${active.length})` : ""}</div>
              <a className="mem-manage" href="#/memories">Manage memories</a>
            </div>
            {listError && <p className="small mem-error">{listError}</p>}
            {all === null && !listError && <p className="quiet small">Loading…</p>}
            {all !== null && active.length === 0 && <p className="quiet small">None yet. Grounded findings are kept after each answer in a chat.</p>}
            <ul className="mem-list">
              {active.map((m) => (
                <li key={m.id} className="mem-item">
                  {confirming === m.id ? (
                    <div className="confirm">
                      <span>Forget this?</span>
                      <button className="confirm-yes" onClick={() => remove(m.id)}>Forget</button>
                      <button className="confirm-no" onClick={() => setConfirming(null)}>Keep</button>
                    </div>
                  ) : (
                    <div className="mem-row">
                      <div className="mem-body">
                        <p className="mem-text">{m.text}</p>
                        <div className="mem-meta">
                          {m.kind === "preference" ? "Preference" : `Used ${m.use_count}×`}, last {relTime(m.last_used_at)}
                          {m.strength != null && m.kind === "finding" && <span title="use weight × recency; below the config minimum it is forgotten"> · strength {m.strength.toFixed(2)}</span>}
                        </div>
                      </div>
                      <button className="icon-btn mem-delete" onClick={() => setConfirming(m.id)} aria-label={`Forget: ${m.text}`}>
                        <Icon name="trash" size={15} />
                      </button>
                    </div>
                  )}
                </li>
              ))}
            </ul>
            {inactive.length > 0 && (
              <details className="mem-forgotten">
                <summary>Forgotten ({inactive.length})</summary>
                <ul className="mem-list">
                  {inactive.map((m) => (
                    <li key={m.id} className="mem-item is-gone">
                      <p className="mem-text">{m.text}</p>
                      <div className="mem-meta">
                        {FORGET_REASON[m.forget_reason ?? ""] ?? humanise(m.forget_reason)}, {relTime(m.forgotten_at ?? m.invalidated_at ?? m.last_used_at)}
                      </div>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </section>
        </>
      )}

      <h3 className="mem-short">In this chat</h3>
      <ShortTerm message={message} sample={sample} />
    </div>
  );
}

function ChunkLinks({ ids, onShowSource }: { ids: string[]; onShowSource: (chunk: string) => void }) {
  return (
    <span className="mem-chunks">
      {ids.map((c) => (
        <button key={c} className="link-btn mem-chunk" onClick={() => onShowSource(c)} title="Show in Sources">{c}</button>
      ))}
    </span>
  );
}

function UsedInAnswer({ step, report, onShowSource }: { step?: Step; report?: MemoryReport; onShowSource: (chunk: string) => void }) {
  const memories = step?.memories ?? [];
  const prefs = step?.preferences ?? [];
  const used = new Set((report?.used ?? []).map((u) => u.id));
  return (
    <section className="mem-section">
      <div className="detail-label">Used in this answer</div>
      {!memories.length && !prefs.length ? (
        <p className="quiet small">Nothing: no remembered finding was close enough to this question.</p>
      ) : (
        <ul className="mem-list">
          {memories.map((m) => (
            <li key={m.id} className="mem-item">
              <p className="mem-text">{m.text}</p>
              <div className="mem-meta">
                <code>{m.label}</code> · {used.has(m.id) ? "cited, so it counts as a use" : report ? "given but not cited, so not a use" : "given to the agent"}
              </div>
              <ChunkLinks ids={m.chunk_ids} onShowSource={onShowSource} />
            </li>
          ))}
          {prefs.map((p) => (
            <li key={p.id} className="mem-item">
              <p className="mem-text">{p.text}</p>
              <div className="mem-meta">Preference you stated earlier</div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function WrittenFromAnswer({ message }: { message: AssistantMessage }) {
  const r = message.memory;
  const rows: { tag: string; text: string; ids?: string[]; tone?: string }[] = [];
  for (const w of r?.written ?? []) rows.push({ tag: w.kind === "preference" ? "New preference" : "New", text: w.text, ids: w.chunk_ids });
  for (const f of r?.refreshed ?? []) rows.push({ tag: "Seen again", text: f.text, ids: f.chunk_ids_added });
  for (const i of r?.invalidated ?? []) rows.push({ tag: "Replaced", text: i.text, tone: "is-gone" });
  const dropped = (r?.dropped?.ungrounded ?? 0) + (r?.dropped?.uncited ?? 0);
  return (
    <section className="mem-section">
      <div className="detail-label">Written from this answer</div>
      {message.status === "running" ? (
        <p className="quiet small">Written once the answer is finished.</p>
      ) : !r ? (
        <p className="quiet small">Nothing recorded for this answer.</p>
      ) : (
        <>
          {r.error && <p className="small mem-error">Memory was not updated: {r.detail || humanise(r.error)}</p>}
          {r.skipped && <p className="quiet small">{SKIPPED[r.skipped] ?? humanise(r.skipped)}</p>}
          {!r.skipped && !r.error && rows.length === 0 && <p className="quiet small">Nothing new: no claim in this answer was both cited and supported by its passage.</p>}
          {rows.length > 0 && (
            <ul className="mem-list">
              {rows.map((row, i) => (
                <li key={i} className={`mem-item ${row.tone ?? ""}`}>
                  <p className="mem-text"><span className="mem-tag">{row.tag}</span>{row.text}</p>
                  {row.ids && row.ids.length > 0 && <div className="mem-meta"><code>{row.ids.join(" ")}</code></div>}
                </li>
              ))}
            </ul>
          )}
          {dropped > 0 && (
            <p className="quiet small">
              {plural(dropped, "claim")} left out: {r.dropped?.ungrounded ? `${r.dropped.ungrounded} not supported by the passage it cites` : ""}
              {r.dropped?.ungrounded && r.dropped?.uncited ? ", " : ""}
              {r.dropped?.uncited ? `${r.dropped.uncited} without a citation to a passage this answer had` : ""}.
            </p>
          )}
          {(r.forgotten?.length ?? 0) > 0 && <p className="quiet small">{plural(r.forgotten.length, "older memory", "older memories")} forgotten in the sweep after this answer.</p>}
        </>
      )}
    </section>
  );
}

/* Short-term memory, per chat (docs/DECISIONS.md D12). What a later question in this
   chat is sent from this answer: the question, the answer, and these notes. */
function ShortTerm({ message, sample }: { message: AssistantMessage | null; sample: boolean }) {
  const notes = message?.notes ?? [];
  const s = message?.stats ?? {};
  return (
    <>
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
    </>
  );
}
