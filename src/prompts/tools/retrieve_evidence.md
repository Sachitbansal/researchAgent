Retrieve passages, figures, and tables from the indexed papers that are relevant to a
query. This is your primary source of evidence — use it for any question about what the
papers say. Returns ranked chunks with paper, page, and section metadata so you can cite
them precisely. For a question with several parts, or one that spans several papers, pass
`queries` instead: one short search per part or paper (up to 3), in a single call. Each
chunk then names the `matched_query` it answers, and `subquery_sufficient` says which
parts found nothing: say so for those parts rather than concluding the papers are silent.
If `sufficient_evidence` comes back false, the index does not contain good evidence for
this query, and `coverage_gap.suggested_next` says what to do: rephrase and retry once, or
tell the user what the corpus lacks so they can approve a fetch. Do not call
`search_literature` unless the user has approved a fetch, and do not answer from your own
knowledge when this returns nothing.
