# Tool Specification

Five tools. The agent chooses which to call and in what order; nothing here is
sequenced by the system.

**Two rules that govern every tool:**

1. **Tools never call other tools.** All control flow goes through the agent. If
   `retrieve_evidence` returns `sufficient_evidence: false`, it does *not* trigger a
   fetch — it reports the gap, the agent describes it, and the user approves a fetch
   with the "Fetch papers" button (`docs/DECISIONS.md` D14, D24–D27). Automating that
   away would recreate a fixed pipeline, and fetch into a shared corpus unasked.
2. **Tools never raise.** Every failure path returns `{"error": "...", "detail": ...}`
   so a bad PDF or a dead API degrades the loop instead of killing it.

Tool descriptions below are the **exact text the model sees**. They are written to
disambiguate, because the most common failure in agentic systems is a model that
cannot tell two similar tools apart.

---

## 1. `search_literature`

Fetches new papers from arXiv and adds them to the index.

**Description shown to the model:**

> Search arXiv for papers on a topic, download them, and add them to the index. Call
> this only when the user has approved a fetch; their message says so explicitly. It
> adds papers to the corpus permanently, is slow (30–60s) and costs API calls. Do not
> use it to answer questions about papers that are already indexed; use
> `retrieve_evidence` for that. Do not call it twice for the same topic. Leave
> `categories` unset: the query planner already chooses them, and yours would narrow
> its choice a second time.

**Input**

```json
{
  "query":       "string, required — natural language topic",
  "max_results": "int, optional, default 8",
  "categories":  "string[], optional — arXiv categories e.g. ['cs.LG','cs.CL']"
}
```

**Behaviour**

1. LLM call translates `query` into arXiv query syntax (field prefixes, booleans,
   category filters)
2. arXiv API call, `sort_by=relevance`
3. **admission gate** — every result's `title + abstract` (arXiv search results carry
   the abstract) is scored against `query` by the retrieval cross-encoder
   (`retrieval.reranker_model`, the same instance the retriever holds). Papers below
   `collection.admission_threshold` are not downloaded, not indexed and not tagged; they
   are listed in `papers_rejected`. The gate sees papers already held too, so a dedup hit
   it rejects does not gain this topic's tag. If the cross-encoder cannot load or run,
   every result is admitted and `admission_gate` says `"skipped: <reason>"`
   (`src/corpus/admission.py`)
4. dedup against manifest on `arxiv_id`
5. download new PDFs only
6. extract text + figures/tables, chunk, describe figures, embed, append to index
7. update the manifest with `topic_tag` — and on a dedup hit, append that tag to the
   existing paper record *and* to every chunk record it already produced
   (`docs/DATA_SCHEMA.md`, **Topic naming**)

**Returns**

```json
{
  "papers_added":   [{"arxiv_id","paper_id","title","year","abstract_snippet",
                      "admission_score": "float | null — null when the gate was skipped"}],
  "papers_rejected": [{"arxiv_id","title","score"}],
  "admission_threshold": "float — collection.admission_threshold",
  "admission_gate": "'applied' | 'skipped: <reason>'",
  "papers_skipped": "int — already indexed",
  "chunks_added":   "int",
  "figures_added":  "int",
  "topic_tag":      "string",
  "total_indexed":  "int",
  "arxiv_query":    "string"
}
```

**When every result is rejected** the tool still returns a result, not an error:
`papers_added: []`, every paper in `papers_rejected`, `arxiv_coverage: "poor"`, and a
`note` saying arXiv does not seem to cover the topic well. Nothing is downloaded or
indexed, no topic entry is written to the manifest (a zero-paper topic would read as
coverage in `analyze_corpus`), and the retriever's cache is not invalidated.

**Failure modes:** arXiv unreachable, zero results, PDF download failure, PDF parse
failure. Partial success is reported, not thrown away — three good papers out of five
is a valid result. A skipped admission gate is reported in `admission_gate` and `note`,
not as an error.

**Rate limiting.** arXiv answers HTTP 429 per *IP*, for every query, not only the one
that tripped it, and the block lasts minutes. Searches and PDF downloads therefore
share one delay clock (`src/corpus/arxiv_fetch.py`); downloads do not go through
`arxiv.Result.download_pdf`, which bypasses it. On a 429 the tool returns

```json
{"error": "arxiv_rate_limited", "detail": "...when it lifts, and how to clear it",
 "partial": {"papers_added": [...], "chunks_added": 0, "topic_tag": "..."}}
```

and stops immediately rather than retrying — a retry is one more request into the
block. `partial` carries whatever was downloaded and indexed before the 429, because
dedup means papers not ingested now are never ingested. The expiry is recorded in
`paths.arxiv_cooldown` and checked before the next search sends anything, so varying
the wording and calling again costs zero requests; delete that file (or pass
`--clear-arxiv-cooldown` to `main.py index`) to clear it early.

---

## 2. `retrieve_evidence`

Core RAG. The tool the agent should reach for by default.

**Description shown to the model:**

> Retrieve passages, figures, and tables from the indexed papers that are relevant to a
> query. This is your primary source of evidence — use it for any question about what the
> papers say. Returns ranked chunks with paper, page, and section metadata so you can cite
> them precisely. For a question with several parts, or one that spans several papers,
> pass `queries` instead: one short search per part or paper (up to 3), in a single call.
> Each chunk then names the `matched_query` it answers, and `subquery_sufficient` says
> which parts found nothing: say so for those parts rather than concluding the papers are
> silent. If `sufficient_evidence` comes back false, the index does not contain good
> evidence for this query, and `coverage_gap.suggested_next` says what to do: rephrase and
> retry once, or tell the user what the corpus lacks so they can approve a fetch. Do not
> call `search_literature` unless the user has approved a fetch, and do not answer from
> your own knowledge when this returns nothing.

**Input**

```json
{
  "query":        "string — one search; this or `queries` is required",
  "queries":      "string[], optional — up to retrieval.max_subqueries (3) searches, one per
                   part of a multi-part or multi-paper question; `query`, if also given,
                   joins them first",
  "k":            "int, optional, default 5",
  "topic_filter": "string, optional — one tag; matches chunks whose topic_tags contain it",
  "chunk_types":  "string[], optional — subset of ['text','figure','table']"
}
```

Neither field alone is required by the schema. With both missing, the tool returns
`empty_query`. More than 3 distinct queries returns `too_many_queries`. Both are error
dicts, never exceptions.

**Behaviour**

1. per (sub-)query: bi-encoder top-`k_retrieve` (40) and, with `retrieval.hybrid`, BM25
   top-40 (D31). The metadata filter (`topic_tags` membership, `chunk_types`) applies to
   both lists alike, as a post-filter
2. Reciprocal Rank Fusion of every list into one pool, **capped at `k_retrieve`** however
   many lists feed it
3. cross-encoder rerank over the pool. With several queries, each candidate gets its max
   score over the sub-queries whose lists found it (`retrieval.multi_query_rerank:
   retrieving`; `all` scores every pair, D32)
4. **relevance gate** — scores vs. the empirically tuned threshold of the scorer that ran:
   the cross-encoder logit, or on the fallback path the dense cosine (never an RRF score)
5. neighbour expansion — attach `±1` adjacent chunks from the same paper
6. truncate each chunk to ~500 tokens
7. deduplicate against chunks already returned whose text is **still visible** in the
   conversation. Elision releases the chunks it hides, so they can be returned again
   (`docs/DECISIONS.md` D13)

**Returns**

```json
{
  "chunks": [{
    "chunk_id","paper_id","paper_title","page","section",
    "chunk_type","score","text",
    "figure_id": "present only when chunk_type is figure|table"
  }],
  "sufficient_evidence": "bool",
  "note": "string — why the gate failed, when it did; for several queries, which found
           nothing, and which cleared the gate but were outranked out of k",
  "n_candidates_considered": "int — the fused pool, at most k_retrieve",
  "subquery_sufficient": "{query: bool} — several queries only: did any candidate clear
                          the gate for that sub-query",
  "coverage_gap": "object — present only when the relevance gate failed, see below"
}
```

With several queries each chunk also carries `matched_query`, the sub-query that gave its
score. A single `query` returns exactly the shape it always did.

**`coverage_gap`** — a structured hint, returned only when a scorer ran and every
candidate fell below the relevance threshold. It is data, not a call: the tool still
fetches nothing; the agent describes the gap and the user decides whether to fetch.

```json
"coverage_gap": {
  "searched_for":   "string — the query as given; several queries joined with '; '",
  "best_score":     "float — the top candidate's score (over all sub-queries)",
  "threshold":      "float — the threshold actually used, for the scorer that ran",
  "scorer":         "'cross_encoder' | 'bi_encoder'",
  "closest_topics": ["topic_tag — up to 3, in the rank order of their best candidate"],
  "suggested_next": "'rephrase' | 'propose_fetch'",
  "proposed_query": "string — what to hand search_literature; today the query itself,
                     since search_literature plans its own arXiv syntax",
  "filtered":       "bool — the call set topic_filter or chunk_types (D24)"
}
```

- `scorer` and `threshold` follow whatever actually produced the scores, including a
  reranker that was requested but failed and fell back to bi-encoder cosines.
- `suggested_next` is `rephrase` when `best_score` is within the scorer's margin below
  the threshold (`retrieval.rephrase_margin` for the cross-encoder,
  `retrieval.bi_encoder_rephrase_margin` for cosines), else `propose_fetch`. The margins
  are measured; see `config.yaml`.
- Absent on every other `sufficient_evidence: false` path: a filter that matches no
  chunk, an index with no candidates, and "every passing chunk was already returned".
  None of those is a gap in the literature, and in the first two no scorer ran.

The agent loop reads the **last unfiltered** `retrieve_evidence` call of a run (no
`topic_filter`, no `chunk_types`; filtered calls neither propose nor clear, D24). If its
`coverage_gap` says `propose_fetch`, or it hit an `empty_index`, and `search_literature`
was not called after it, the run result carries `fetch_proposal: {query, agent_query,
query_source, searched_for, best_score, closest_topics}` (else `null`) for the user to
approve (`src/agent/fetch_proposal.py`). `query` is the user's question rewritten into a
keyword phrase by one `utility` call (`src/agent/fetch_query.py`, prompt
`src/prompts/fetch_query.md`; `query_source: "rewritten"`). On an LLM failure or a junk
reply it falls back to the agent's retrieve query (`query_source: "agent"`, with
`rewrite_error`). `agent_query` always keeps the agent's query (D28). It rides in the web
`answer` frame and the turn's stats. The approval is `POST /api/ask` with `approve_fetch: {query}` and a `thread_id`,
or `main.py ask --thread ID --approve-fetch "<query>"`. Either way the agent's message
comes from `src/prompts/fetch_approval.md` (D26).

**Note on figure chunks:** these come back with the *stored description plus
caption*, not the image. If detail is needed that the description does not carry,
that is what `inspect_figure` is for — and the agent makes that call, not this tool.

---

## 3. `inspect_figure`

Multimodal inspection of an actual image.

**Description shown to the model:**

> Look at the actual image of a figure or table from a paper, rather than its stored
> text description. Use this when `retrieve_evidence` returned a figure or table and
> you need detail the description may have lost — exact values, axis labels, the
> shape of a trend, or what a specific series does. Also use it to inspect an image
> the user has supplied. Requires a `figure_id` from a previous `retrieve_evidence`
> result, or a user-supplied image reference.

**Input**

```json
{
  "paper_id":   "string, optional — required for indexed figures",
  "figure_id":  "string, optional",
  "chunk_id":   "string, optional — alternative to paper_id+figure_id",
  "user_image": "string, optional — path/reference to a user-supplied image",
  "question":   "string, optional — what specifically to look for"
}
```

**Returns**

```json
{
  "image_path":  "string — the image the agent loop must attach",
  "mime_type":   "string",
  "caption":     "string — original caption from the paper",
  "stored_description": "string — the index-time description",
  "paper_id","page","figure_id"
}
```

For a user-supplied image, caption and stored description are absent.

**How the image actually reaches the model.** The tool returns the record above — a
structured dict, like every other tool — and the *agent loop* attaches the image to the
conversation immediately afterwards, via `llm_client.attach_images`, as an `image_url`
content part on a follow-up user turn.

This is not a stylistic choice. The provider surface is OpenRouter's
OpenAI-compatible `/chat/completions`, where a `tool` message's `content` is a string
and image parts are only valid on a `user` message. (Anthropic's *native* API does
accept image blocks inside `tool_result`; the OpenAI-compatible surface does not, and
the whole point of going through OpenRouter is that the provider stays a config value.)
Returning base64 inside the tool result would also have made this the one tool that
returns a raw string blob rather than a structured dict.

So the ordering the loop must preserve is:

```
assistant(tool_call: inspect_figure)
  -> tool(result: the dict above, no image bytes)
  -> user(image_url part + the question, attached by the loop)
  -> assistant(reads the image)
```

The tool still never calls another tool, and control flow still lives in the agent.

**Why both paths exist:** the indexed description makes figures *findable*; the raw
image makes them *readable*. Retrieval needs the first, precision needs the second.

---

## 4. `check_evidence_consistency`

NLI-based checking, two modes.

**Description shown to the model:**

> Check retrieved evidence for consistency using a natural language inference model.
> Two modes. `contradiction`: given several chunks, find pairs that disagree with each
> other — use this before answering when sources might conflict, so you can surface
> the disagreement rather than silently picking one side. `groundedness`: given your
> drafted answer text and the chunks you used, check whether each claim in the answer
> is actually supported by the evidence — use this before finalising an answer built
> from multiple sources. Claims flagged `neutral` are not supported by the retrieved
> evidence and should be removed, softened, or explicitly marked as uncertain.
> Treat the output as advisory: it is a heuristic signal, not a verdict.

**Input**

```json
{
  "mode":        "'contradiction' | 'groundedness', required",
  "chunk_ids":   "string[], required",
  "answer_text": "string — required for groundedness mode only"
}
```

**Behaviour**

- Chunks are split into sentences; the best-matching sentence is used as premise
  (NLI models degrade badly on paragraph-length inputs)
- Direction is fixed: **premise = evidence, hypothesis = claim**
- `groundedness` mode extracts atomic claims from `answer_text` **inside the tool**
  via its own LLM call — deliberately not left to the agent, which would otherwise be
  grading a claim list it selected itself

**Returns — `contradiction` mode**

```json
{
  "found": "bool",
  "conflicting_pairs": [{
    "chunk_a","chunk_b","snippet_a","snippet_b","confidence"
  }]
}
```

**Returns — `groundedness` mode**

```json
{
  "claims": [{
    "claim","label": "entailed|neutral|contradicted",
    "best_supporting_chunk","confidence"
  }],
  "grounded_ratio": "float",
  "unsupported_claims": ["string"]
}
```

`unsupported_claims` is returned explicitly so the agent knows exactly what to fix
rather than having to re-derive it.

**Stated limitations** (also in the architecture doc): general MNLI-trained models are
weak on scientific text; `neutral` conflates "unsupported" with "not understood";
confidence values are uncalibrated.

---

## 5. `analyze_corpus`

Computation over corpus **metadata**, not content. No retrieval, no LLM call.

**Description shown to the model:**

> Compute aggregate statistics over the indexed corpus: how many papers there are,
> which topics they cover, what years they span, and what sub-themes they cluster
> into. Use this for questions *about the corpus itself* — coverage, recency, size,
> thematic structure — and to orient yourself before deciding whether you need to
> fetch more literature. Do NOT use this for questions about what the papers say;
> that needs `retrieve_evidence`.

**Input**

```json
{
  "operation":    "'stats' | 'timeline' | 'cluster' | 'compare_topics'",
  "topic_filter": "string, optional",
  "params":       "object, optional — e.g. {'k': 4} for cluster"
}
```

**Operations**

| Operation | Computes |
|---|---|
| `stats` | papers per topic, chunk counts, figure/table counts, date range |
| `timeline` | per-topic year histogram, median year, recency |
| `cluster` | KMeans over embeddings, `k` by silhouette, TF-IDF cluster labels, **validated against `topic_tags` via adjusted Rand index + purity** |
| `compare_topics` | vocabulary overlap and centroid distance between topic groups |

**Returns**

```json
{
  "operation": "string",
  "result":    "object — numbers, labels, cluster memberships",
  "summary":   "string — one-paragraph natural language summary the agent can quote"
}
```

**Caching:** `cluster` results are cached against a manifest hash and recomputed only
when the index changes.

**`summary` is templated, not generated.** This tool makes no LLM call, so the
one-paragraph summary is produced by deterministic string templating over the numbers
already in `result` — one template per operation. That keeps the tool free of a model
dependency, keeps its output reproducible for eval, and means the summary can never
disagree with the `result` it is describing.

**Why this is the ML component and not decoration:** `k` is selected rather than
assumed, and clusters are scored against known labels. If the clusters fail to
recover the known topic split, that is a real diagnostic about embedding quality, and
it gets reported as such.

---

## Tool selection: expected behaviour

The system prompt does not prescribe an order. These are the patterns the design
anticipates — they are what the eval's tool-trace check looks for, not a script:

| Question shape | Likely trace |
|---|---|
| Factual, corpus has it | `retrieve_evidence` → answer |
| Factual, corpus lacks it | `retrieve_evidence` (gate fails, `propose_fetch`) → describe the gap, closest topics, and the button. On approval, next turn: `search_literature` → `retrieve_evidence` → cited answer |
| Near miss | `retrieve_evidence` (gate fails, `rephrase`) → one reworded `retrieve_evidence` → answer or abstain |
| Genuinely unanswerable | `retrieve_evidence` (gate fails) → possibly one retry → abstain |
| Multi-paper synthesis | `retrieve_evidence` → `check_evidence_consistency(contradiction)` → answer |
| Numeric detail in a plot | `retrieve_evidence` → `inspect_figure` → answer |
| "What's in this corpus?" | `analyze_corpus` → answer |

**Loop control:** each tool may be dispatched at most `agent.max_calls_per_tool` (10)
times per question, `search_literature` 5 (`agent.tool_call_limits`). There is no
iteration cap (`docs/DECISIONS.md` D4). Counting is per dispatch, not per turn: a call
past the limit gets `{"error": "tool_limit_reached", ...}` without running, and an
exhausted tool is dropped from the tools offered. Refused calls (repeated, bad JSON,
unknown tool, over the limit) cost nothing. Loop detection on repeated
`(tool_name, args)` pairs stays. The agent is forced to answer with what it has, and to
say the evidence is partial, when every tool is exhausted (`stopped_because:
"tools_exhausted"`) or after `agent.max_no_progress_turns` (3) consecutive turns in which
nothing was dispatched (`"no_progress"`).
