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
import { renderMarkdown, renderMathText } from "../markdown";
import { argLine, excerpts, humanise, outcome, plural, relTime, secs, who } from "../format";
import { Icon } from "../icons";
import { QUIET, QUIET_SMALL } from "../ui";

// Inside the Memory tab every paragraph is 14px with 10px after it; memory texts are tighter.
const MEM_P = "mb-2.5 text-14 leading-1.6";
const MEM_QUIET = `${MEM_P} text-soft`;
const MEM_ERROR = `${MEM_P} text-danger`;
const MEM_TEXT = "text-13.5 leading-1.55 wrap-anywhere";
const MEM_META = "mt-[3px] text-12 text-faint";
const MEM_ITEM = "rounded-md border border-rule bg-raise px-2.5 py-2";
const DETAIL_PRE = "m-0 max-h-[280px] overflow-auto rounded-md border border-rule bg-raise px-2.5 py-2 text-11.5 leading-1.5 whitespace-pre-wrap wrap-anywhere";

/* A memory's wording, with any math in it typeset (escaped first: see markdown.ts). */
const mathText = (text: string) => ({ __html: renderMathText(text) });

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
    <aside
      className={`flex min-h-0 min-w-0 flex-col overflow-hidden border-l border-rule bg-panel
        max-xl:fixed max-xl:inset-y-0 max-xl:right-0 max-xl:z-20 max-xl:w-[min(var(--width-insp),92vw)] max-xl:shadow-md
        max-xl:transition-[translate,visibility] max-xl:duration-200 ${p.open ? "" : "invisible max-xl:translate-x-full"}`}
      aria-label="The agent's working" aria-hidden={!p.open}>
      <div className="flex h-[52px] flex-none items-center gap-1 border-b border-rule pr-2 pl-3">
        <div className="flex flex-1 gap-0.5" role="tablist">
          {(["calls", "sources", "memory"] as const).map((t) => (
            <button key={t} role="tab" aria-selected={p.tab === t} className={`min-h-8 rounded-md px-3 text-13.5 whitespace-nowrap transition-colors ${
                p.tab === t ? "bg-selected font-medium text-ink" : "text-soft hover:bg-hover hover:text-ink"}`} onClick={() => p.setTab(t)}>
              {t === "calls" ? "Tool calls" : t === "sources" ? "Sources" : "Memory"}
            </button>
          ))}
        </div>
        <button className="icon-btn" onClick={p.onClose} aria-label="Hide the side panel"><Icon name="close" size={16} /></button>
      </div>
      {p.message && !p.isLatest && <div className="border-b border-rule bg-hover px-4 py-1.5 text-12.5 text-soft">Showing an earlier answer in this chat.</div>}
      <div className="min-h-0 flex-1 overflow-y-auto px-4 pt-4 pb-8">
        {p.tab === "memory" ? (
          <Memory message={p.message} sample={p.sample} version={p.memoryVersion} onShowSource={p.onShowSource} />
        ) : !p.message ? (
          <p className={QUIET}>Ask a question and each step the agent takes shows up here as it happens.</p>
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
    <div>
      {turns.length === 0 && running && <p className={QUIET}>Starting…</p>}
      {turns.map((t) => (
        <section key={t.iteration} className="mb-4">
          <h3 className="mb-2 flex justify-between text-12.5 font-semibold text-soft">
            Turn {t.iteration + 1}
            {t.t_ms != null && <span className="font-normal text-faint tabular-nums">at {secs(t.t_ms)}</span>}
          </h3>
          {t.rows.length === 0 && !(running && last?.kind === "thinking" && last.iteration === t.iteration) && (
            <p className={QUIET_SMALL}>Thought, called no tools.</p>
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
        <div className="min-w-0 flex-1">
          <button
            className="-mx-1 -mt-0.5 box-content flex w-full justify-between gap-2 rounded-sm px-1 py-0.5 text-left hover:bg-hover focus-visible:-outline-offset-2"
            onClick={() => setOpen((o) => !o)} aria-expanded={open}>
            <span className="row-name">{start.tool_name}</span>
            <span className="flex-none text-12 text-faint tabular-nums">{end ? secs(end.latency_ms ?? 0) : "working"}</span>
          </button>
          {start.args && Object.keys(start.args).length > 0 && <code className="mt-0.5 block text-11.5 leading-1.5 wrap-anywhere text-soft">{argLine(start.args, 70)}</code>}
          {end && <div className="row-out">{outcome(end.result)}</div>}
          {open && (
            <div className="mt-2">
              <div className="detail-label">Arguments</div>
              <pre className={DETAIL_PRE}>{JSON.stringify(start.args ?? {}, null, 2)}</pre>
              {end && (
                <>
                  <div className="detail-label">Result, as summarised for the trace</div>
                  <pre className={DETAIL_PRE}>{JSON.stringify(end.result ?? {}, null, 2)}</pre>
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
      <div className="row"><span className="row-dot" />
        <div className="min-w-0 flex-1"><span className="row-name">{s.tool_name || "call"} skipped</span><div className="row-out">{s.reason}</div></div>
      </div>
    );
  }
  if (row.type === "image") {
    return (
      <div className="row tool-inspect_figure"><span className="row-dot" />
        <div className="min-w-0 flex-1">
          <span className="row-name">looked at the figure</span>
          {s.image_url ? <img className="mt-1.5 block max-h-[160px] max-w-full rounded-md border border-rule bg-white" src={s.image_url} alt="" loading="lazy" /> : <div className="row-out">{s.image_path}</div>}
        </div>
      </div>
    );
  }
  if (row.type === "memory") {
    const n = s.memories?.length ?? 0;
    const prefs = s.preferences?.length ?? 0;
    return (
      <div className="row"><span className="row-dot" />
        <div className="min-w-0 flex-1">
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
        <div className="min-w-0 flex-1"><span className="row-name">stopped calling tools</span>
          <div className="row-out">{s.reason || "reached its step limit, answering with what it has"}</div></div>
      </div>
    );
  }
  return (
    <div className="row is-answer"><span className="row-dot" />
      <div className="min-w-0 flex-1"><span className="row-name">wrote the answer</span></div>
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
    <dl className="mt-2 rounded-md border border-rule bg-raise px-4 py-3">
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-3 py-[3px] text-13">
          <dt className="text-soft">{k}</dt>
          <dd className="min-w-0 text-right tabular-nums wrap-anywhere">{v}</dd>
        </div>
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

  if (message.status === "running") return <p className={QUIET}>Sources appear once the answer is written.</p>;
  if (!refs.length) return <p className={QUIET}>This answer cites no passages.</p>;

  return (
    <div>
      {refs.map((r) => {
        const paper = papersById.get(r.paper);
        return (
          <section key={r.paper} className="mb-6">
            <div className="mb-2 flex gap-2.5">
              <span className="ref-n">{r.n}</span>
              <div>
                <div className="text-14 leading-1.4 font-medium">{paper?.title ?? r.paper}</div>
                {paper && <div className="text-12.5 text-soft">{who(paper)}</div>}
                {paper?.pdf_url && <a className="text-13" href={paper.pdf_url} target="_blank" rel="noopener noreferrer">arXiv</a>}
              </div>
            </div>
            <ul className="ml-[30px]">
              {r.chunks.map((c) => (
                <li key={c} ref={c === highlight ? marked : undefined} className={`mb-1.5 rounded-md border px-2.5 py-2 ${c === highlight ? "border-mark-edge bg-raise" : "border-transparent"}`}>
                  <code className="text-11 text-faint">{c}</code>
                  {found.get(c)?.tool === "memory" && <span className="mem-tag ml-1.5">from memory</span>}
                  <p className="mt-[3px] text-13.5 leading-1.6">
                    {found.get(c)?.text ? <mark>{found.get(c)!.text}…</mark> : <span className="text-soft">No excerpt in this run's steps.</span>}
                  </p>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
      {uncited.length > 0 && (
        <p className={QUIET_SMALL}>
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
    <div>
      <h3 className="mt-1 mb-2 text-14.5 font-semibold">Across chats</h3>
      <p className={MEM_P}>
        Findings from earlier answers that their own passages support are remembered across chats. A later question
        on the same subject is given the relevant ones with those passages, to check and cite. Unused memories fade,
        and a newer finding that contradicts one replaces it.
      </p>
      {sample ? (
        <p className={MEM_QUIET}>Samples are fixtures and are never sent to the agent.</p>
      ) : (
        <>
          {message && (
            <>
              <UsedInAnswer step={injected} report={message.memory} onShowSource={onShowSource} />
              <WrittenFromAnswer message={message} />
            </>
          )}

          <section className="mb-4">
            <div className="flex items-baseline justify-between gap-2">
              <div className="detail-label">All memories{all ? ` (${active.length})` : ""}</div>
              <a className="text-12.5" href="#/memories">Manage memories</a>
            </div>
            {listError && <p className={MEM_ERROR}>{listError}</p>}
            {all === null && !listError && <p className={MEM_QUIET}>Loading…</p>}
            {all !== null && active.length === 0 && <p className={MEM_QUIET}>None yet. Grounded findings are kept after each answer in a chat.</p>}
            <ul className="grid gap-2">
              {active.map((m) => (
                <li key={m.id} className={MEM_ITEM}>
                  {confirming === m.id ? (
                    <div className="confirm p-0">
                      <span>Forget this?</span>
                      <button className="confirm-yes" onClick={() => remove(m.id)}>Forget</button>
                      <button className="confirm-no" onClick={() => setConfirming(null)}>Keep</button>
                    </div>
                  ) : (
                    <div className="flex items-start gap-2">
                      <div className="min-w-0 flex-1">
                        <p className={MEM_TEXT} dangerouslySetInnerHTML={mathText(m.text)} />
                        <div className={MEM_META}>
                          {m.kind === "preference" ? "Preference" : `Used ${m.use_count}×`}, last {relTime(m.last_used_at)}
                          {m.strength != null && m.kind === "finding" && <span title="use weight × recency; below the config minimum it is forgotten"> · strength {m.strength.toFixed(2)}</span>}
                        </div>
                      </div>
                      <button className="icon-btn -mt-0.5 -mr-1 size-7 max-md:size-11" onClick={() => setConfirming(m.id)} aria-label={`Forget: ${m.text}`}>
                        <Icon name="trash" size={15} />
                      </button>
                    </div>
                  )}
                </li>
              ))}
            </ul>
            {inactive.length > 0 && (
              <details className="group mt-3">
                <summary className="cursor-pointer py-0.5 text-12.5 text-soft group-open:mb-2 hover:text-ink">Forgotten ({inactive.length})</summary>
                <ul className="grid gap-2">
                  {inactive.map((m) => (
                    <li key={m.id} className={MEM_ITEM}>
                      <p className={`${MEM_TEXT} text-soft`} dangerouslySetInnerHTML={mathText(m.text)} />
                      <div className={MEM_META}>
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

      <h3 className="mt-6 mb-2 border-t border-rule pt-4 text-14.5 font-semibold">In this chat</h3>
      <ShortTerm message={message} sample={sample} />
    </div>
  );
}

function ChunkLinks({ ids, onShowSource }: { ids: string[]; onShowSource: (chunk: string) => void }) {
  return (
    <span className="mt-[3px] flex flex-wrap gap-x-2 gap-y-0.5">
      {ids.map((c) => (
        <button key={c} className="link-btn font-mono text-11.5" onClick={() => onShowSource(c)} title="Show in Sources">{c}</button>
      ))}
    </span>
  );
}

function UsedInAnswer({ step, report, onShowSource }: { step?: Step; report?: MemoryReport; onShowSource: (chunk: string) => void }) {
  const memories = step?.memories ?? [];
  const prefs = step?.preferences ?? [];
  const used = new Set((report?.used ?? []).map((u) => u.id));
  return (
    <section className="mb-4">
      <div className="detail-label">Used in this answer</div>
      {!memories.length && !prefs.length ? (
        <p className={MEM_QUIET}>Nothing: no remembered finding was close enough to this question.</p>
      ) : (
        <ul className="grid gap-2">
          {memories.map((m) => (
            <li key={m.id} className={MEM_ITEM}>
              <p className={MEM_TEXT} dangerouslySetInnerHTML={mathText(m.text)} />
              <div className={MEM_META}>
                <code className="text-11 wrap-anywhere">{m.label}</code> · {used.has(m.id) ? "cited, so it counts as a use" : report ? "given but not cited, so not a use" : "given to the agent"}
              </div>
              <ChunkLinks ids={m.chunk_ids} onShowSource={onShowSource} />
            </li>
          ))}
          {prefs.map((p) => (
            <li key={p.id} className={MEM_ITEM}>
              <p className={MEM_TEXT} dangerouslySetInnerHTML={mathText(p.text)} />
              <div className={MEM_META}>Preference you stated earlier</div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function WrittenFromAnswer({ message }: { message: AssistantMessage }) {
  const r = message.memory;
  const rows: { tag: string; text: string; ids?: string[]; gone?: boolean }[] = [];
  for (const w of r?.written ?? []) rows.push({ tag: w.kind === "preference" ? "New preference" : "New", text: w.text, ids: w.chunk_ids });
  for (const f of r?.refreshed ?? []) rows.push({ tag: "Seen again", text: f.text, ids: f.chunk_ids_added });
  for (const i of r?.invalidated ?? []) rows.push({ tag: "Replaced", text: i.text, gone: true });
  const dropped = (r?.dropped?.ungrounded ?? 0) + (r?.dropped?.uncited ?? 0);
  return (
    <section className="mb-4">
      <div className="detail-label">Written from this answer</div>
      {message.status === "running" ? (
        <p className={MEM_QUIET}>Written once the answer is finished.</p>
      ) : !r ? (
        <p className={MEM_QUIET}>Nothing recorded for this answer.</p>
      ) : (
        <>
          {r.error && <p className={MEM_ERROR}>Memory was not updated: {r.detail || humanise(r.error)}</p>}
          {r.skipped && <p className={MEM_QUIET}>{SKIPPED[r.skipped] ?? humanise(r.skipped)}</p>}
          {!r.skipped && !r.error && rows.length === 0 && <p className={MEM_QUIET}>Nothing new: no claim in this answer was both cited and supported by its passage.</p>}
          {rows.length > 0 && (
            <ul className="grid gap-2">
              {rows.map((row, i) => (
                <li key={i} className={MEM_ITEM}>
                  <p className={`${MEM_TEXT}${row.gone ? " text-soft" : ""}`}>
                    <span className="mem-tag">{row.tag}</span><span dangerouslySetInnerHTML={mathText(row.text)} />
                  </p>
                  {row.ids && row.ids.length > 0 && <div className={MEM_META}><code className="text-11 wrap-anywhere">{row.ids.join(" ")}</code></div>}
                </li>
              ))}
            </ul>
          )}
          {dropped > 0 && (
            <p className={MEM_QUIET}>
              {plural(dropped, "claim")} left out: {r.dropped?.ungrounded ? `${r.dropped.ungrounded} not supported by the passage it cites` : ""}
              {r.dropped?.ungrounded && r.dropped?.uncited ? ", " : ""}
              {r.dropped?.uncited ? `${r.dropped.uncited} without a citation to a passage this answer had` : ""}.
            </p>
          )}
          {(r.forgotten?.length ?? 0) > 0 && <p className={MEM_QUIET}>{plural(r.forgotten.length, "older memory", "older memories")} forgotten in the sweep after this answer.</p>}
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
      <p className={MEM_P}>
        Each new question in this chat is sent with the earlier questions and answers, and a one-line note for each tool
        call. The passages themselves are not carried over, so the agent retrieves them again. Once the history passes its
        budget, the oldest turns are left out first.
      </p>
      {sample ? (
        <p className={MEM_QUIET}>Samples are fixtures and are never sent to the agent.</p>
      ) : !message ? (
        <p className={MEM_QUIET}>Nothing yet. Ask a question to start this chat.</p>
      ) : message.status === "running" ? (
        <p className={MEM_QUIET}>The notes for this answer are saved when it finishes.</p>
      ) : (
        <>
          <div className="detail-label">Notes kept from this answer</div>
          {notes.length ? (
            <ul className="mb-3 grid gap-1">
              {notes.map((n, i) => <li key={i}><code className="block text-11.5 leading-1.5 wrap-anywhere text-soft">{n}</code></li>)}
            </ul>
          ) : (
            <p className={MEM_QUIET}>None: this answer called no tools.</p>
          )}
          {s.history_turns != null && (
            <p className={MEM_QUIET}>
              This answer was written with {plural(s.history_turns, "earlier turn")} in view
              {s.history_turns_dropped ? `, and ${s.history_turns_dropped} older left out` : ""}.
            </p>
          )}
        </>
      )}
    </>
  );
}
