/* The corpus: every indexed paper, grouped by the topic it was collected under.
   In:  /api/papers as loaded by App. Out: one section per topic with its planned arXiv
        query, and per paper its title, authors, year, chunk and figure counts and link.
   Papers whose tags match no current topic get a section of their own, not a silent drop. */

import type { Corpus, Paper, Topic } from "../types";
import { plural, who } from "../format";
import { COUNT, PAGE, PAGE_LEDE, PAGE_TITLE, QUIET } from "../ui";

export function CorpusView({ corpus, error }: { corpus: Corpus | null; error: string | null }) {
  if (error) return <div className={PAGE}><p className={QUIET}>{error}</p></div>;
  if (!corpus) return <div className={PAGE}><p className={QUIET}>Loading the corpus…</p></div>;

  const papers = corpus.papers;
  if (!papers.length) {
    return (
      <div className={PAGE}>
        <h2 className={PAGE_TITLE}>Nothing indexed yet</h2>
        <p className={PAGE_LEDE}>
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
    <div className={PAGE}>
      <h2 className={PAGE_TITLE}>What the agent has read</h2>
      <p className={PAGE_LEDE}>
        {plural(papers.length, "paper")}, collected under {plural(topics.length, "topic")}, embedded with{" "}
        {corpus.embedding_model || "the configured model"}. When a question falls outside these papers, the agent
        offers to fetch more, and adds them only once you approve.
      </p>
      {sections.map((t) => {
        const mine: Paper[] = t.tag === null ? loose : papers.filter((p) => (p.topic_tags || []).includes(t.tag!));
        return (
          <section key={t.tag ?? "_loose"} className="mb-8">
            <h3 className="mb-1 flex items-baseline gap-2 text-16 font-semibold">{t.label} <span className={COUNT}>{mine.length}</span></h3>
            {t.arxiv_query && (
              <p className="mb-3 flex flex-wrap items-baseline gap-2 text-12.5 text-faint">
                <span>arXiv query</span><code className="text-soft wrap-anywhere">{t.arxiv_query}</code>
              </p>
            )}
            <ul className="border-t border-rule">
              {mine.map((p) => (
                <li key={p.paper_id} className="border-b border-rule py-3">
                  <div className="text-15 leading-1.4 font-medium">{p.title}</div>
                  <div className="mt-0.5 text-13.5 text-soft">{who(p)}</div>
                  <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-12.5 text-faint">
                    <span>{p.pdf_name || p.paper_id}</span>
                    {p.n_chunks > 0 && <span>{plural(p.n_chunks, "chunk")}, {plural(p.n_figures, "figure")}</span>}
                    {p.parse_status && p.parse_status !== "ok" && <span className="text-danger">{p.parse_status} parse</span>}
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
