/* Centre pane: the conversation, the empty state, and the composer.
   In:  the displayed messages (the thread's stored turns plus the live run), the corpus,
        callbacks. Out: rendered turns. Answers are rendered markdown with numbered
        citations; a running answer shows what the agent is doing right now; an answer
        that found a gap in the corpus carries the "Fetch papers" card (D33). */

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
    <div className="chat">
      <div
        className="thread-scroll"
        ref={scroller}
        onScroll={(e) => {
          const el = e.currentTarget;
          stuck.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
        }}
      >
        <div className="thread">
          {p.sample && (
            <p className="sample-note">
              A sample chat, written by hand to show how the interface works. The answer did not come from the
              agent, but its citations point at real passages in the corpus. Ask your own question in a new chat.
            </p>
          )}
          {p.loading ? (
            <p className="quiet thread-loading">Loading this chat…</p>
          ) : p.messages.length === 0 ? (
            <EmptyState corpus={p.corpus} error={p.corpusError} onPick={(q) => p.setDraft(q)} />
          ) : (
            p.messages.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="turn-user"><div className="bubble">{m.content}</div></div>
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

      <div className="composer-wrap">
        {p.notice && (
          <div className="notice" role="status">
            <span>{p.notice}</span>
            <button className="icon-btn" onClick={p.onDismissNotice} aria-label="Dismiss"><Icon name="close" size={16} /></button>
          </div>
        )}
        <Composer draft={p.draft} setDraft={p.setDraft} onSend={p.onSend} running={p.running} />
        <p className="independence">
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
    <form className={`composer${running ? " is-running" : ""}`} onSubmit={(e) => { e.preventDefault(); submit(); }}>
      <textarea
        ref={box}
        rows={1}
        value={draft}
        disabled={running}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={onKey}
        placeholder={running ? "The agent is working on your question…" : "Ask about the papers"}
        aria-label="Your question"
      />
      <button className="send" type="submit" disabled={running || !draft.trim()} aria-label="Ask">
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
      <div className="turn-agent is-running" aria-live="polite">
        <div className="live">
          <span className="pulse" aria-hidden="true" />
          <span className="live-text">{liveStatus(message.steps)}</span>
        </div>
        <button className="live-calls" onClick={() => onSelect(index)}>
          {calls ? `${plural(calls, "tool call")} so far` : "Watch the tool calls"}
        </button>
        {figs.length > 0 && <Figures figs={figs} />}
      </div>
    );
  }

  const stats = message.stats ?? {};
  return (
    <div className={`turn-agent${selected ? " is-selected" : ""}`}>
      {message.content ? (
        <div className="prose" onClick={onClick} dangerouslySetInnerHTML={{ __html: html }} />
      ) : null}

      {message.status === "failed" && (
        <div className="failure">
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
        <ol className="refs">
          {refs.map((r) => {
            const paper = papersById.get(r.paper);
            return (
              <li key={r.paper} value={r.n}>
                <span className="ref-n">{r.n}</span>
                <span className="ref-body">
                  <span className="ref-title">{paper?.title ?? r.paper}</span>
                  {paper && <span className="ref-who">{who(paper)}</span>}
                  {paper?.pdf_url && (
                    <a className="ref-link" href={paper.pdf_url} target="_blank" rel="noopener noreferrer">arXiv</a>
                  )}
                </span>
              </li>
            );
          })}
        </ol>
      )}

      <div className="answer-foot">
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
    <div className="figures">
      {figs.map((f, i) => (
        <figure key={i} className="figure">
          {f.url ? (
            <a href={f.url} target="_blank" rel="noopener noreferrer"><img src={f.url} alt={f.caption || f.id || "figure"} loading="lazy" /></a>
          ) : (
            <div className="figure-missing">An image outside the corpus was shown to the model, not served here.</div>
          )}
          <figcaption>
            {f.id && <span className="figure-id">{f.id}</span>} {f.caption}
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
    <div className="empty">
      <h2 className="empty-title">Start with a question</h2>
      {error ? (
        <p className="empty-lede">{error}</p>
      ) : !corpus ? (
        <p className="empty-lede">Loading the corpus…</p>
      ) : n ? (
        <p className="empty-lede">
          {plural(n, "paper")} indexed, across {plural(topics.length, "topic")}. Every claim in an answer is cited,
          and each citation opens the passage behind it. If the papers don't cover your question, the agent says so and
          offers to fetch papers on it from arXiv, which you approve with a button.
        </p>
      ) : (
        <p className="empty-lede">
          Nothing is indexed yet. Ask anyway: the agent will say so and offer to fetch papers from arXiv, which takes a few
          minutes once you approve it.
        </p>
      )}
      {examples.length > 0 && <p className="examples-head">Or try one of these, built from the indexed topics:</p>}
      {examples.length > 0 && (
        <ul className="examples">
          {examples.map((q) => (
            <li key={q}><button onClick={() => onPick(q)}>{q}</button></li>
          ))}
        </ul>
      )}
    </div>
  );
}
