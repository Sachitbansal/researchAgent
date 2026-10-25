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
  // The rewritten query names what the user asked; the agent's own search wording can
  // drift (D28), so prefer the rewrite whenever there is one.
  const subject = proposal.query_source === "rewritten" ? proposal.query : proposal.searched_for || proposal.query;
  const trimmed = query.trim();
  const canSend = state === "actionable" && trimmed.length > 0 && trimmed.length <= maxChars;

  return (
    <section className={`fetch-card is-${state}`} aria-label="Fetch papers for this question">
      <p className="fetch-gap">
        The corpus has no papers on <em>{subject}</em>.
        {closest.length > 0 && <> Closest topics held: {closest.join(", ")}.</>}
      </p>

      {state === "fetching" || state === "approved" ? (
        <p className="fetch-status" role="status" aria-live="polite">
          {state === "fetching" ? <span className="live-dot" aria-hidden="true" /> : null}
          {state === "fetching"
            ? <>Fetching papers on <em>{approvedQuery ?? proposal.query}</em>. The steps below show the search.</>
            : <>Fetch approved for <em>{approvedQuery ?? proposal.query}</em>. The answer below uses what it found.</>}
        </p>
      ) : state === "stopped" ? (
        <p className="fetch-note">The fetch run stopped before it finished. Ask the question again to retry.</p>
      ) : state === "superseded" ? (
        <p className="fetch-note">A later question followed. Ask this one again to fetch papers on it.</p>
      ) : (
        <form
          className="fetch-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (canSend) onApprove(trimmed);
          }}
        >
          <label className="fetch-label" htmlFor={inputId}>
            Search arXiv for{proposal.query_source === "rewritten" && " (suggested from your question)"}
          </label>
          <div className="fetch-row">
            <input
              id={inputId}
              className="fetch-input"
              value={query}
              maxLength={maxChars}
              disabled={state !== "actionable"}
              onChange={(e) => setQuery(e.target.value)}
              aria-describedby={noteId}
            />
            <button className="fetch-btn" type="submit" disabled={!canSend}>
              Fetch papers
            </button>
          </div>
          <p className="fetch-note" id={noteId}>
            {state === "busy"
              ? "The agent is busy. Fetching becomes available when it finishes."
              : "Takes a minute or more. Papers that pass the relevance check are added to the corpus permanently, then the agent answers your question again."}
          </p>
        </form>
      )}
    </section>
  );
}
