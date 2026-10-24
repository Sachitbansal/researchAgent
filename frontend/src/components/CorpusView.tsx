/* The corpus: every indexed paper, grouped by the topic it was collected under.
   In:  /api/papers as loaded by App. Out: one section per topic with its planned arXiv
        query, and per paper its title, authors, year, chunk and figure counts and link.
   Papers whose tags match no current topic get a section of their own, not a silent drop. */

import type { Corpus, Paper, Topic } from "../types";
import { plural, who } from "../format";

export function CorpusView({ corpus, error }: { corpus: Corpus | null; error: string | null }) {
  if (error) return <div className="page"><p className="quiet">{error}</p></div>;
  if (!corpus) return <div className="page"><p className="quiet">Loading the corpus…</p></div>;

  const papers = corpus.papers;
  if (!papers.length) {
    return (
      <div className="page">
        <h2 className="page-title">Nothing indexed yet</h2>
        <p className="page-lede">
          {corpus.error ?? "Index a topic with python main.py index \"<topic>\", or just ask a question and the agent will search arXiv for papers first."}
        </p>
      </div>
    );
  }

  const topics = corpus.topics;
  const known = new Set(topics.map((t) => t.tag));
  const loose = papers.filter((p) => !(p.topic_tags || []).some((t) => known.has(t)));
  const sections: Topic[] = loose.length
    ? [...topics, { tag: null, label: "Collected under no current topic", arxiv_query: "", count: loose.length }]
    : topics;

  return (
    <div className="page">
      <h2 className="page-title">What the agent has read</h2>
      <p className="page-lede">
        {plural(papers.length, "paper")}, collected under {plural(topics.length, "topic")}, embedded with{" "}
        {corpus.embedding_model || "the configured model"}. The agent can add to this while answering, when the
        papers here don't cover a question.
      </p>
      {sections.map((t) => {
        const mine: Paper[] = t.tag === null ? loose : papers.filter((p) => (p.topic_tags || []).includes(t.tag!));
        return (
          <section key={t.tag ?? "_loose"} className="topic">
            <h3 className="topic-name">{t.label} <span className="topic-count">{mine.length}</span></h3>
            {t.arxiv_query && <p className="topic-query"><span>arXiv query</span><code>{t.arxiv_query}</code></p>}
            <ul className="papers">
              {mine.map((p) => (
                <li key={p.paper_id} className="paper">
                  <div className="paper-title">{p.title}</div>
                  <div className="paper-who">{who(p)}</div>
                  <div className="paper-meta">
                    <span>{p.pdf_name || p.paper_id}</span>
                    {p.n_chunks > 0 && <span>{plural(p.n_chunks, "chunk")}, {plural(p.n_figures, "figure")}</span>}
                    {p.parse_status && p.parse_status !== "ok" && <span className="paper-flag">{p.parse_status} parse</span>}
                    {p.pdf_url && <a href={p.pdf_url} target="_blank" rel="noopener noreferrer">arXiv</a>}
                  </div>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
