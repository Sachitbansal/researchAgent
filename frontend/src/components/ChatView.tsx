/* Centre pane: the conversation, the empty state, and the composer.
   In:  the displayed messages (the thread's stored turns plus the live run), the corpus,
        callbacks. Out: rendered turns. Answers are rendered markdown with numbered
        citations; a running answer shows what the agent is doing right now; an answer
        that found a gap in the corpus carries the "Fetch papers" card (D27). */

import { useEffect, useMemo, useRef, type KeyboardEvent, type MouseEvent } from "react";
import type { AssistantMessage, Corpus, Message, Paper } from "../types";
import { renderMarkdown } from "../markdown";
import { excerpts, figures, humanise, liveStatus, plural, secs, who } from "../format";
import type { CiteTarget } from "./CitationCard";
import { FetchCard, type FetchState } from "./FetchCard";
import { Icon } from "../icons";

// Mirrors config.yaml agent.fetch_query_max_chars. The server checks it; this only stops
// the input from accepting what the server would refuse.
const FETCH_QUERY_MAX = 300;

/* What the card under answer `i` can do. Only the latest answer of a real chat can be
   fetched from: the approval asks the agent to answer "my previous question". */
function fetchState(messages: Message[], i: number, sample: boolean, running: boolean): { state: FetchState; query?: string } | null {
  const m = messages[i] as AssistantMessage;
  // An approval turn whose retrieval still failed re-derives the same proposal; offering
  // the fetch that just ran again would be noise, so it gets no card.
  if (sample || !m.stats?.fetch_proposal || m.stats.approved_fetch) return null;
  const next = messages.slice(i + 1).find((x): x is AssistantMessage => x.role === "assistant");
  if (next?.stats?.approved_fetch) {
    const state = next.status === "running" ? "fetching" : next.status === "failed" ? "stopped" : "approved";
    return { state, query: next.stats.approved_fetch.query };
  }
  if (next) return { state: "superseded" };
  return { state: running ? "busy" : "actionable" };
}

interface Props {
  messages: Message[];
  sample: boolean;
  loading: boolean;
  papersById: Map<string, Paper>;
  corpus: Corpus | null;
  corpusError: string | null;
  running: boolean;
  draft: string;
  setDraft: (s: string) => void;
  onSend: (q: string) => void;
  // The "Fetch papers" click: the approved query, and the user bubble shown while it runs.
  onApproveFetch: (query: string, shown: string) => void;
  notice: string | null;
  onDismissNotice: () => void;
  selected: number | null;
  onSelect: (i: number) => void;
  onCite: (t: CiteTarget) => void;
}

export function ChatView(p: Props) {
  const scroller = useRef<HTMLDivElement>(null);
  const stuck = useRef(true);
  const lastSteps = (p.messages[p.messages.length - 1] as AssistantMessage | undefined)?.steps?.length ?? 0;

  // Follow the conversation as it grows, unless the reader has scrolled up to read.
  useEffect(() => {
    const el = scroller.current;
    if (el && stuck.current) el.scrollTop = el.scrollHeight;
  }, [p.messages.length, lastSteps]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        className="min-h-0 flex-1 overflow-y-auto"
        ref={scroller}
        onScroll={(e) => {
          const el = e.currentTarget;
          stuck.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
        }}
      >
        <div className="mx-auto max-w-col px-6 pt-8 pb-12 max-md:px-4 max-md:pt-6 max-md:pb-8">
          {p.sample && (
            <p className="mb-6 rounded-md border border-l-3 border-rule border-l-rule-strong bg-panel px-4 py-3 text-13.5 leading-1.55 text-soft">
              A sample chat, written by hand to show how the interface works. The answer did not come from the
              agent, but its citations point at real passages in the corpus. Ask your own question in a new chat.
            </p>
          )}
          {p.loading ? (
            <p className="my-[1em] py-8 text-center text-soft">Loading this chat…</p>
          ) : p.messages.length === 0 ? (
            <EmptyState corpus={p.corpus} error={p.corpusError} onPick={(q) => p.setDraft(q)} />
          ) : (
            p.messages.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="mb-6 flex justify-end">
                  <div className="max-w-[80%] rounded-lg border border-rule bg-hover px-3.5 py-2.5 leading-1.55 whitespace-pre-wrap wrap-anywhere">{m.content}</div>
                </div>
              ) : (
                <Answer
                  key={i}
                  index={i}
                  message={m}
                  papersById={p.papersById}
                  selected={p.selected === i}
                  onSelect={p.onSelect}
                  onCite={p.onCite}
                  fetch={fetchState(p.messages, i, p.sample, p.running)}
                  corpus={p.corpus}
                  onApproveFetch={p.onApproveFetch}
                />
              ),
            )
          )}
        </div>
      </div>

      <div className="mx-auto w-full max-w-col flex-none px-6 pb-3 max-md:px-3 max-md:pb-2">
        {p.notice && (
          <div className="mb-2 flex items-start gap-2 rounded-md border border-rule-strong bg-panel py-1.5 pr-1.5 pl-4 text-14" role="status">
            <span className="flex-1 pt-[7px]">{p.notice}</span>
            <button className="icon-btn" onClick={p.onDismissNotice} aria-label="Dismiss"><Icon name="close" size={16} /></button>
          </div>
        )}
        <Composer draft={p.draft} setDraft={p.setDraft} onSend={p.onSend} running={p.running} />
        <p className="mx-1 mt-2 text-center text-12.5 text-faint">
          {p.sample
            ? "Asking here starts a new chat. Samples are not sent to the agent."
            : "The agent sees this chat's earlier questions and answers, and checked findings from earlier chats with their passages."}
        </p>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ composer

function Composer({ draft, setDraft, onSend, running }: {
  draft: string; setDraft: (s: string) => void; onSend: (q: string) => void; running: boolean;
}) {
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [draft]);

  useEffect(() => {
    if (!running) box.current?.focus();
  }, [running]);

  const submit = () => {
    if (!running && draft.trim()) onSend(draft);
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <form
      className={`flex items-end gap-2 rounded-lg border border-rule-strong py-1.5 pr-1.5 pl-4 shadow-sm transition-[border-color,box-shadow]
        hover:not-focus-within:border-faint focus-within:border-accent focus-within:shadow-[0_0_0_3px_var(--color-accent-wash)] ${
        running ? "bg-panel" : "bg-raise"}`}
      onSubmit={(e) => { e.preventDefault(); submit(); }}
    >
      <textarea
        className="max-h-[240px] flex-1 resize-none bg-transparent py-[7px] font-sans text-15 leading-1.5 text-ink outline-none disabled:cursor-not-allowed"
        ref={box}
        rows={1}
        value={draft}
        disabled={running}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={onKey}
        placeholder={running ? "The agent is working on your question…" : "Ask about the papers"}
        aria-label="Your question"
      />
      <button
        className="grid size-9 flex-none place-items-center rounded-md bg-accent text-accent-ink transition-colors hover:bg-accent-hover disabled:bg-selected disabled:text-faint"
        type="submit" disabled={running || !draft.trim()} aria-label="Ask">
        <Icon name="send" size={18} />
      </button>
    </form>
  );
}

// ------------------------------------------------------------------ an answer

function Answer({ index, message, papersById, selected, onSelect, onCite, fetch, corpus, onApproveFetch }: {
  index: number;
  message: AssistantMessage;
  papersById: Map<string, Paper>;
  selected: boolean;
  onSelect: (i: number) => void;
  onCite: (t: CiteTarget) => void;
  fetch: { state: FetchState; query?: string } | null;
  corpus: Corpus | null;
  onApproveFetch: (query: string, shown: string) => void;
}) {
  const { html, refs } = useMemo(() => renderMarkdown(message.content), [message.content]);
  const found = useMemo(() => excerpts(message.steps), [message.steps]);
  const figs = useMemo(() => figures(message.steps), [message.steps]);
  const calls = message.steps.filter((s) => s.kind === "tool_end").length;

  const onClick = (e: MouseEvent<HTMLDivElement>) => {
    const chip = (e.target as HTMLElement).closest<HTMLButtonElement>(".cite");
    if (!chip) return;
    const chunk = chip.dataset.chunk!;
    onCite({
      chunk,
      paper: chip.dataset.paper!,
      rect: chip.getBoundingClientRect(),
      messageIndex: index,
      excerpt: found.get(chunk)?.text,
    });
  };

  if (message.status === "running") {
    return (
      <div className="mb-12" aria-live="polite">
        <div className="flex items-center gap-2 text-14.5 text-soft">
          <span className="size-2 flex-none animate-pulse rounded-full bg-accent" aria-hidden="true" />
          <span className="wrap-anywhere">{liveStatus(message.steps)}</span>
        </div>
        <button className="mt-2 ml-4 text-13.5 text-accent hover:text-accent-hover hover:underline" onClick={() => onSelect(index)}>
          {calls ? `${plural(calls, "tool call")} so far` : "Watch the tool calls"}
        </button>
        {figs.length > 0 && <Figures figs={figs} />}
      </div>
    );
  }

  const stats = message.stats ?? {};
  return (
    <div className="mb-12">
      {message.content ? (
        <div className="prose" onClick={onClick} dangerouslySetInnerHTML={{ __html: html }} />
      ) : null}

      {message.status === "failed" && (
        <div className="mt-1 mb-4 rounded-md border border-danger bg-danger-wash px-4 py-3 text-14 text-danger">
          <strong>{message.content ? "The run stopped before it finished." : "No answer."}</strong>{" "}
          {humanise(message.error?.error)}{message.error?.detail ? `: ${message.error.detail}` : ""}
        </div>
      )}

      {figs.length > 0 && <Figures figs={figs} />}

      {fetch && message.stats?.fetch_proposal && (
        <FetchCard
          proposal={message.stats.fetch_proposal}
          corpus={corpus}
          state={fetch.state}
          approvedQuery={fetch.query}
          maxChars={FETCH_QUERY_MAX}
          // The bubble while it runs; the stored turn's own wording replaces it after.
          onApprove={(q) => onApproveFetch(q, `Fetch papers on "${q}"`)}
        />
      )}

      {refs.length > 0 && (
        <ol className="mt-6 border-t border-rule pt-4 text-13.5">
          {refs.map((r) => {
            const paper = papersById.get(r.paper);
            return (
              <li key={r.paper} value={r.n} className="mb-2 flex gap-3 leading-1.5">
                <span className="ref-n">{r.n}</span>
                <span className="min-w-0">
                  <span className="font-medium">{paper?.title ?? r.paper}</span>
                  {paper && <span className="ml-1.5 text-soft">{who(paper)}</span>}
                  {paper?.pdf_url && (
                    <a className="ml-2 text-13" href={paper.pdf_url} target="_blank" rel="noopener noreferrer">arXiv</a>
                  )}
                </span>
              </li>
            );
          })}
        </ol>
      )}

      <div className={`mt-4 flex flex-wrap justify-between gap-x-4 gap-y-1 text-12.5 ${selected ? "text-soft" : "text-faint"}`}>
        <span>
          {stats.tool_calls != null && `${plural(stats.tool_calls, "tool call")} over ${plural(stats.iterations ?? 0, "turn")}`}
          {message.elapsed_ms != null && `, ${secs(message.elapsed_ms)}`}
          {stats.stopped_because && stats.stopped_because !== "answered" && `, stopped: ${humanise(stats.stopped_because)}`}
        </span>
        <button className="link-btn" onClick={() => onSelect(index)}>
          {selected ? "Shown in the side panel" : "Show the working"}
        </button>
      </div>
    </div>
  );
}

function Figures({ figs }: { figs: ReturnType<typeof figures> }) {
  return (
    <div className="mt-4 mb-1 grid grid-cols-[repeat(auto-fill,minmax(200px,1fr))] gap-3">
      {figs.map((f, i) => (
        <figure key={i} className="overflow-hidden rounded-md border border-rule bg-raise">
          {f.url ? (
            <a href={f.url} target="_blank" rel="noopener noreferrer"><img className="block max-h-[260px] w-full bg-white object-contain" src={f.url} alt={f.caption || f.id || "figure"} loading="lazy" /></a>
          ) : (
            <div className="px-3 py-4 text-13 text-soft">An image outside the corpus was shown to the model, not served here.</div>
          )}
          <figcaption className="px-3 py-2 text-12.5 leading-1.45 text-soft">
            {f.id && <span className="font-mono text-11.5 text-ink">{f.id}</span>} {f.caption}
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ empty state

/* Examples are built from the corpus's own topic labels at runtime: no topic name may be
   written into source (.claude/CLAUDE.md). With nothing indexed, it says so and invites a
   question anyway — search_literature can build a corpus from nothing. */
const LEDE = "mb-6 max-w-lede text-15 leading-1.6 text-soft";

function EmptyState({ corpus, error, onPick }: { corpus: Corpus | null; error: string | null; onPick: (q: string) => void }) {
  const topics = (corpus?.topics ?? []).filter((t) => t.count > 0);
  const n = corpus?.papers.length ?? 0;

  const examples: string[] = [];
  topics.slice(0, 2).forEach((t, i) => {
    examples.push(i === 0
      ? `What methods do the papers on ${t.label} propose, and how do they compare?`
      : `Where do the papers on ${t.label} disagree with each other?`);
  });
  if (n) examples.push("Which topics does the corpus cover, and how many papers are there for each?");
  if (topics[0]) examples.push(`What do the figures in the papers on ${topics[0].label} show?`);

  return (
    <div className="pt-[8vh]">
      <h2 className="mb-2 text-22 leading-1.3 font-semibold">Start with a question</h2>
      {error ? (
        <p className={LEDE}>{error}</p>
      ) : !corpus ? (
        <p className={LEDE}>Loading the corpus…</p>
      ) : n ? (
        <p className={LEDE}>
          {plural(n, "paper")} indexed, across {plural(topics.length, "topic")}. Every claim in an answer is cited,
          and each citation opens the passage behind it. If the papers don't cover your question, the agent says so and
          offers to fetch papers on it from arXiv, which you approve with a button.
        </p>
      ) : (
        <p className={LEDE}>
          Nothing is indexed yet. Ask anyway: the agent will say so and offer to fetch papers from arXiv, which takes a few
          minutes once you approve it.
        </p>
      )}
      {examples.length > 0 && <p className="mb-2 text-13 font-medium text-faint">Or try one of these, built from the indexed topics:</p>}
      {examples.length > 0 && (
        <ul className="grid gap-2">
          {examples.map((q) => (
            <li key={q}>
              <button
                className="block w-full rounded-md border border-rule bg-raise px-4 py-2.5 text-left text-14.5 leading-1.45 transition-colors hover:border-rule-strong hover:bg-hover focus-visible:-outline-offset-2 active:bg-selected"
                onClick={() => onPick(q)}
              >
                {q}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
