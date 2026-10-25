/* The "Fetch papers" card under an answer whose corpus lacked the subject (D14, D27).
   In:  the answer's fetch_proposal, the corpus (topic tags to labels), and its state:
        actionable, approved (fetching or fetched), or superseded by a later question.
   Out: the gap in a sentence, closest topics held, and a primary button that sends the
        approval with the (editable) query. The click is the user's approval; nothing else
        fetches. */

import { useId, useState } from "react";
import type { Corpus, FetchProposal } from "../types";

export type FetchState = "actionable" | "busy" | "fetching" | "approved" | "stopped" | "superseded";

interface Props {
  proposal: FetchProposal;
  corpus: Corpus | null;
  state: FetchState;
  approvedQuery?: string;
  maxChars: number;
  onApprove: (query: string) => void;
}

export function FetchCard({ proposal, corpus, state, approvedQuery, maxChars, onApprove }: Props) {
  const [query, setQuery] = useState(proposal.query);
  const inputId = useId();
  const noteId = useId();
  const labels = new Map((corpus?.topics ?? []).map((t) => [t.tag, t.label]));
  const closest = proposal.closest_topics.map((t) => labels.get(t) ?? t.replace(/_/g, " "));
  const subject = proposal.searched_for || proposal.query;
  const trimmed = query.trim();
  const canSend = state === "actionable" && trimmed.length > 0 && trimmed.length <= maxChars;

  return (
    <section
      className={`mt-4 mb-1 rounded-md border border-l-3 border-rule bg-panel px-4 py-3 text-14 leading-1.55 [&_em]:font-medium [&_em]:not-italic ${
        state === "approved" || state === "superseded" || state === "stopped" ? "border-l-rule-strong" : "border-l-accent"}`}
      aria-label="Fetch papers for this question">
      <p>
        The corpus has no papers on <em>{subject}</em>.
        {closest.length > 0 && <> Closest topics held: {closest.join(", ")}.</>}
      </p>

      {state === "fetching" || state === "approved" ? (
        <p className="mt-2 text-13.5 text-soft" role="status" aria-live="polite">
          {state === "fetching" ? <span className="live-dot" aria-hidden="true" /> : null}
          {state === "fetching"
            ? <>Fetching papers on <em>{approvedQuery ?? proposal.query}</em>. The steps below show the search.</>
            : <>Fetch approved for <em>{approvedQuery ?? proposal.query}</em>. The answer below uses what it found.</>}
        </p>
      ) : state === "stopped" ? (
        <p className="mt-2 text-12.5 text-faint">The fetch run stopped before it finished. Ask the question again to retry.</p>
      ) : state === "superseded" ? (
        <p className="mt-2 text-12.5 text-faint">A later question followed. Ask this one again to fetch papers on it.</p>
      ) : (
        <form
          className="mt-3"
          onSubmit={(e) => {
            e.preventDefault();
            if (canSend) onApprove(trimmed);
          }}
        >
          <label className="mb-1 block text-12.5 font-medium text-soft" htmlFor={inputId}>Search arXiv for</label>
          <div className="flex flex-wrap gap-2">
            <input
              id={inputId}
              className="field min-h-9 min-w-0 flex-[1_1_16rem] px-3 font-sans text-14 leading-1.4 disabled:cursor-not-allowed disabled:text-soft max-md:min-h-11"
              value={query}
              maxLength={maxChars}
              disabled={state !== "actionable"}
              onChange={(e) => setQuery(e.target.value)}
              aria-describedby={noteId}
            />
            <button
              className="min-h-9 flex-none rounded-md bg-accent px-4 text-14 font-medium text-accent-ink transition-colors enabled:hover:bg-accent-hover disabled:bg-selected disabled:text-faint max-md:min-h-11"
              type="submit" disabled={!canSend}>
              Fetch papers
            </button>
          </div>
          <p className="mt-2 text-12.5 text-faint" id={noteId}>
            {state === "busy"
              ? "The agent is busy. Fetching becomes available when it finishes."
              : "Takes a minute or more. Papers that pass the relevance check are added to the corpus permanently, then the agent answers your question again."}
          </p>
        </form>
      )}
    </section>
  );
}
