"""
`retrieve_evidence`: the core RAG tool — dense (+BM25, RRF), rerank, gate, expand, truncate.

In:  a query and/or up to retrieval.max_subqueries `queries`; optional k, topic_filter and
     chunk_types.
Out: {"chunks", "sufficient_evidence", "note", "n_candidates_considered"} (plus
     "subquery_sufficient" for several queries, "coverage_gap" when the gate fails), or an
     error dict. Never raises, never fetches literature: the hint is data for the agent.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from common.tokenization import TokenCounter
from config import CFG, Config
from corpus.chunk_store import ChunkStore
from corpus.manifest import Manifest, ManifestError
from retrieval.contextual import embed_format, embed_text
from retrieval.embedder import Embedder, EmbeddingError
from retrieval.lexical_index import LexicalIndex, reciprocal_rank_fusion
from retrieval.reranker import RerankError, Reranker
from retrieval.vector_index import IndexError_, VectorIndex

VALID_CHUNK_TYPES = ("text", "figure", "table")
MAX_CLOSEST_TOPICS = 3


def _error(code: str, detail: str, partial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"error": code, "detail": detail, "partial": partial}


def _trim_head(text: str, budget: int) -> str:
    """Last `budget` characters of `text`, started at a sentence boundary if one is near.

    Neighbour context that opens mid-word reads as corruption rather than as context.
    """
    if budget <= 0 or not text:
        return ""
    clipped = text[-budget:]
    for marker in (". ", "? ", "! "):
        position = clipped.find(marker)
        if 0 <= position < len(clipped) // 2:
            return clipped[position + len(marker):].lstrip()
    return clipped.lstrip()


def _trim_tail(text: str, budget: int) -> str:
    """First `budget` characters of `text`, ended at a sentence boundary if one is near."""
    if budget <= 0 or not text:
        return ""
    clipped = text[:budget]
    for marker in (". ", "? ", "! "):
        position = clipped.rfind(marker)
        if position > len(clipped) // 2:
            return clipped[: position + 1].rstrip()
    return clipped.rstrip()


class EvidenceRetriever:
    """Holds the models and the index so repeated calls in one conversation are cheap.

    Also holds the set of chunk_ids already returned, which is what stops a re-query
    from stacking the same passages into the context window a second time. The rule is
    "seen = still visible" (docs/DECISIONS.md D13): a chunk stays suppressed only while
    its text is in the conversation. The loop calls reset() at the start of a question
    and release() when elision takes a result's text out of view.
    """

    def __init__(
        self,
        config: Config = CFG,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.config = config
        self.counter = TokenCounter(config)
        self.embedder = embedder if embedder is not None else Embedder(config)
        self.reranker = reranker if reranker is not None else Reranker(config, self.counter)
        self._returned: Set[str] = set()
        self._index: Optional[VectorIndex] = None
        self._chunks: Optional[Dict[str, Dict[str, Any]]] = None
        self._by_position: Optional[Dict[str, Dict[int, str]]] = None
        self._lexical: Optional[LexicalIndex] = None
        self._corpus_signature = ""
        self.last_rerank_ms = 0
        self.last_rerank_pairs = 0
        self.last_pool: List[str] = []  # the fused candidate ids, for eval and debugging

    # ---- conversation state ----------------------------------------------------

    def reset(self, visible: Iterable[str] = ()) -> None:
        """Start a question: the seen set becomes exactly the chunks already visible.

        For a fresh conversation, and for a thread whose history carries notes but no
        chunk text, that is nothing, so the set starts empty.
        """
        self._returned = {str(chunk_id) for chunk_id in visible}

    def release(self, chunk_ids: Iterable[str]) -> None:
        """Make these chunks returnable again: their text has left the context window.

        Without this, an elided chunk stays suppressed for the rest of the question, so
        the model can neither read it nor get it back by asking again.
        """
        for chunk_id in chunk_ids:
            self._returned.discard(str(chunk_id))

    def invalidate(self) -> None:
        """Drop the cached index and chunk table so the next call re-reads from disk.

        Called after search_literature adds papers. Without it the retriever keeps
        searching the corpus as it stood when this conversation began, and newly fetched
        papers are invisible for the rest of the run — with no error, because a stale
        index still returns plausible results.
        """
        self._index = None
        self._chunks = None
        self._by_position = None
        self._lexical = None  # BM25 is built from the same chunk table, so it goes too

    @property
    def returned(self) -> Set[str]:
        return set(self._returned)

    # ---- corpus access ---------------------------------------------------------

    def _load(self) -> Optional[Dict[str, Any]]:
        """Load index and chunks once per instance. Returns an error dict on failure."""
        if self._index is not None and self._chunks is not None:
            return None
        try:
            manifest = Manifest.load(self.config, strict_model_check=True)
        except ManifestError as exc:
            return _error("index_stale", str(exc))
        # The stored vectors must have been embedded from the same first-stage text the
        # query is about to be compared with. An index embedded without headers, searched
        # as if it had them (or the reverse), still returns plausible results — silently
        # worse ones. A manifest from before the field existed was embedded plain.
        stored_format = str(manifest.data.get("embed_format") or "plain-v1")
        if manifest.data.get("faiss_id_map") and stored_format != embed_format(self.config):
            return _error(
                "index_stale",
                f"the index was embedded as {stored_format!r} but config asks for "
                f"{embed_format(self.config)!r} (retrieval.contextual_headers); re-embed it "
                "with `python main.py reindex`, which is local and never contacts arXiv",
            )
        try:
            self._index = VectorIndex.load(manifest, self.config)
        except IndexError_ as exc:
            return _error("index_unreadable", str(exc))
        self._corpus_signature = (
            f"{manifest.data.get('manifest_hash') or manifest.compute_hash()}:{stored_format}"
        )

        self._chunks = {chunk["chunk_id"]: chunk for chunk in ChunkStore(config=self.config)}
        by_position: Dict[str, Dict[int, str]] = {}
        for chunk in self._chunks.values():
            by_position.setdefault(chunk["paper_id"], {})[int(chunk["position"])] = chunk["chunk_id"]
        self._by_position = by_position
        return None

    def _allowed(
        self, topic_filter: Optional[str], chunk_types: Optional[Sequence[str]]
    ) -> Optional[Set[str]]:
        """chunk_ids passing the metadata filter, or None when no filter was asked for.

        topic_filter is a membership test against the chunk's topic_tags list, not an
        equality test — a chunk reached through two topics carries both tags.
        """
        if not topic_filter and not chunk_types:
            return None
        wanted_types = {str(t).lower() for t in (chunk_types or [])}
        allowed = set()
        for chunk_id, chunk in (self._chunks or {}).items():
            if topic_filter and topic_filter not in (chunk.get("topic_tags") or []):
                continue
            if wanted_types and str(chunk.get("chunk_type", "")).lower() not in wanted_types:
                continue
            allowed.add(chunk_id)
        return allowed

    # ---- assembly --------------------------------------------------------------

    def _expand(self, chunk: Dict[str, Any]) -> str:
        """Attach the adjacent chunks from the same paper, within the character budget.

        Recovers a claim severed at a chunk boundary without raising k. The citation
        still belongs to the middle chunk, so the neighbours are context, not sources.
        """
        span = int(self.config.retrieval.neighbour_expansion)
        budget = int(self.config.retrieval.max_chunk_chars)
        body = chunk["text"]
        if span <= 0:
            return body[:budget]

        positions = (self._by_position or {}).get(chunk["paper_id"], {})
        position = int(chunk["position"])
        before, after = [], []
        for offset in range(1, span + 1):
            previous = (self._chunks or {}).get(positions.get(position - offset, ""))
            following = (self._chunks or {}).get(positions.get(position + offset, ""))
            if previous:
                before.insert(0, previous["text"])
            if following:
                after.append(following["text"])

        remaining = budget - len(body)
        if remaining <= 0:
            return body[:budget]

        # Split what is left evenly, and give the *tail* of the preceding chunk rather
        # than its head — that is the part actually adjacent to this one.
        share = remaining // 2 if (before and after) else remaining
        head = _trim_head(" ".join(before), share) if before else ""
        tail = _trim_tail(" ".join(after), share) if after else ""
        return "\n".join(part for part in (head, body, tail) if part)[:budget]

    def _present(self, chunk: Dict[str, Any], score: float) -> Dict[str, Any]:
        record = {
            "chunk_id": chunk["chunk_id"],
            "paper_id": chunk["paper_id"],
            "paper_title": chunk["paper_title"],
            "page": chunk["page"],
            "section": chunk.get("section"),
            "chunk_type": chunk["chunk_type"],
            "score": round(float(score), 4),
            "text": self._expand(chunk),
        }
        if chunk.get("figure_id"):
            record["figure_id"] = chunk["figure_id"]
        return record

    def _coverage_gap(
        self, query: str, ranked: Sequence[Dict[str, Any]], best: float,
        threshold: float, cross_encoder_ran: bool, filtered: bool = False,
    ) -> Dict[str, Any]:
        """The structured hint returned when the relevance gate fails. Data, not a call.

        `suggested_next` is "rephrase" only when the best score sits within the scorer's
        rephrase margin below its threshold, else "propose_fetch". The margin is chosen
        by which scorer actually ran, for the same reason the threshold is: a logit
        margin applied to cosines would label every bi-encoder miss a near miss.
        `filtered` says the search was narrowed by topic_filter or chunk_types, so its
        failure says nothing about the corpus as a whole (D24): no proposal comes of it.
        For several queries, `query` is them joined and `best` is the best over all.
        """
        margin = float(
            self.config.retrieval.rephrase_margin if cross_encoder_ran
            else self.config.retrieval.bi_encoder_rephrase_margin
        )
        closest: List[str] = []
        for item in ranked:  # best first, so a topic's position is its best chunk's rank
            for tag in item.get("topic_tags") or []:
                if tag not in closest:
                    closest.append(tag)
            if len(closest) >= MAX_CLOSEST_TOPICS:
                break
        finite = math.isfinite(best)
        return {
            "searched_for": query,
            "best_score": round(float(best), 4) if finite else None,
            "threshold": threshold,
            "scorer": "cross_encoder" if cross_encoder_ran else "bi_encoder",
            "closest_topics": closest[:MAX_CLOSEST_TOPICS],
            "suggested_next": (
                "rephrase" if finite and best >= threshold - margin else "propose_fetch"
            ),
            # search_literature plans its own arXiv syntax from natural language, so the
            # query the agent already wrote is the right thing to hand it.
            "proposed_query": query,
            "filtered": bool(filtered),
        }

    # ---- stage one: the candidate pool -------------------------------------------

    def _lexical_index(self) -> LexicalIndex:
        """BM25 over the same header-prefixed text the bi-encoder embedded. Built in
        memory on first use, and rebuilt when the corpus it was built from moves."""
        if self._lexical is None or self._lexical.signature != self._corpus_signature:
            docs = [(chunk_id, embed_text(chunk, self.config, self.counter)[0])
                    for chunk_id, chunk in (self._chunks or {}).items()]
            self._lexical = LexicalIndex(
                docs,
                k1=float(self.config.retrieval.bm25_k1),
                b=float(self.config.retrieval.bm25_b),
                signature=self._corpus_signature,
            )
        return self._lexical

    def _pool(
        self, queries: Sequence[str], vectors: Sequence[Any], allowed: Optional[Set[str]]
    ) -> Dict[str, Any]:
        """Fuse every sub-query's dense (and, with hybrid on, BM25) list into one pool.

        Reciprocal Rank Fusion over ranks only, so cosines and BM25 scores never meet on
        one scale. The pool is capped at k_retrieve however many lists feed it: rerank
        cost is per candidate, and the cross-encoder must only ever see a shortlist.
        One query with hybrid off is a single list, so the pool is the dense top-k.
        """
        k_retrieve = int(self.config.retrieval.k_retrieve)
        hybrid = bool(self.config.retrieval.get("hybrid", False))
        rankings: List[List[str]] = []
        retrieved_by: Dict[str, Set[int]] = {}
        sources: Dict[str, Set[str]] = {}
        for position, (query, vector) in enumerate(zip(queries, vectors)):
            lists = [("dense", [cid for cid, _ in self._index.search(vector, k_retrieve, allowed=allowed)])]
            if hybrid:
                lexical = self._lexical_index().search(query, k_retrieve, allowed=allowed)
                lists.append(("bm25", [cid for cid, _ in lexical]))
            for source, ranking in lists:
                rankings.append(ranking)
                for chunk_id in ranking:
                    retrieved_by.setdefault(chunk_id, set()).add(position)
                    sources.setdefault(chunk_id, set()).add(source)
        fused = reciprocal_rank_fusion(
            rankings, int(self.config.retrieval.rrf_k), limit=k_retrieve
        )
        pool = [chunk_id for chunk_id, _ in fused if chunk_id in (self._chunks or {})]
        # Every candidate's dense cosine against every sub-query, from the stored vectors:
        # a BM25-only candidate has none from the search, and the fallback gate needs one.
        cosines = [self._index.cosines(vector, pool) for vector in vectors]
        return {"pool": pool, "fused": dict(fused), "retrieved_by": retrieved_by,
                "sources": sources, "cosines": cosines}

    # ---- stage two: rerank -------------------------------------------------------

    def _rerank(
        self, queries: Sequence[str], candidates: List[Dict[str, Any]],
        retrieved_by: Dict[str, Set[int]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        """Cross-encoder scores, best first, plus each sub-query's own scores.

        One query: the plain rerank. Several: each candidate's score is the max over the
        sub-queries it is scored against (D32). Which ones is
        retrieval.multi_query_rerank: "all" scores every candidate against every
        sub-query (pool x n pairs); "retrieving" only against the sub-queries whose dense
        or BM25 list returned it, which bounds the cost near one rerank. Raises
        RerankError, which the caller turns into the bi-encoder fallback.
        """
        if len(queries) == 1:
            ranked = self.reranker.rerank(queries[0], candidates)
            self.last_rerank_ms = int(getattr(self.reranker, "last_latency_ms", 0) or 0)
            self.last_rerank_pairs = len(candidates)
            return ranked, [{r["chunk_id"]: r["rerank_score"] for r in ranked}]

        mode = str(self.config.retrieval.get("multi_query_rerank", "all"))
        per_query: List[Dict[str, float]] = []
        elapsed, pairs = 0, 0
        for position, query in enumerate(queries):
            subset = candidates if mode == "all" else [
                c for c in candidates if position in retrieved_by.get(c["chunk_id"], set())
            ]
            scored = self.reranker.rerank(query, subset) if subset else []
            elapsed += int(getattr(self.reranker, "last_latency_ms", 0) or 0)
            pairs += len(subset)
            per_query.append({r["chunk_id"]: r["rerank_score"] for r in scored})
        self.last_rerank_ms, self.last_rerank_pairs = elapsed, pairs

        ranked = []
        for candidate in candidates:
            scores = [(scores_q[candidate["chunk_id"]], position)
                      for position, scores_q in enumerate(per_query)
                      if candidate["chunk_id"] in scores_q]
            if not scores:
                continue  # unreachable: every candidate was retrieved by some sub-query
            best, position = max(scores)
            ranked.append({**candidate, "rerank_score": best,
                           "matched_query": queries[position]})
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked, per_query

    # ---- the tool --------------------------------------------------------------

    def _queries(self, query: Any, queries: Any) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        """The sub-queries to run: `query` first if given, then `queries`, deduplicated."""
        if query is not None and not isinstance(query, str):
            return [], _error("bad_query", "query must be a string")
        if isinstance(queries, str):
            queries = [queries]
        if queries is not None and not isinstance(queries, (list, tuple)):
            return [], _error("bad_queries", "queries must be a list of strings")
        out: List[str] = []
        for item in ([query] if query is not None else []) + list(queries or []):
            if not isinstance(item, str):
                return [], _error("bad_queries", "queries must be a list of strings")
            flat = " ".join(item.split())
            if flat and flat.lower() not in {q.lower() for q in out}:
                out.append(flat)
        if not out:
            return [], _error("empty_query", "a non-empty query (or queries) is required")
        cap = int(self.config.retrieval.max_subqueries)
        if len(out) > cap:
            return [], _error(
                "too_many_queries",
                f"{len(out)} distinct queries given; at most {cap} per call "
                "(query counts as one). Split the rest into another call.",
            )
        return out, None

    def retrieve(
        self,
        query: Optional[str] = None,
        k: Optional[int] = None,
        topic_filter: Optional[str] = None,
        chunk_types: Optional[Sequence[str]] = None,
        queries: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve ranked evidence for `query` and/or `queries`. See docs/TOOLS.md section 2."""
        subqueries, failure = self._queries(query, queries)
        if failure is not None:
            return failure
        multi = len(subqueries) > 1
        label = "; ".join(subqueries)

        if chunk_types:
            bad = [t for t in chunk_types if str(t).lower() not in VALID_CHUNK_TYPES]
            if bad:
                return _error(
                    "bad_chunk_types",
                    f"unknown chunk_types {bad}; expected a subset of {list(VALID_CHUNK_TYPES)}",
                )

        failure = self._load()
        if failure is not None:
            return failure
        if self._index is None or len(self._index) == 0:
            return _error("empty_index", "no chunks are indexed; fetch literature first")

        k_final = int(k if k is not None else self.config.retrieval.k_final)
        if k_final <= 0:
            return _error("bad_k", f"k must be positive, got {k_final}")

        allowed = self._allowed(topic_filter, chunk_types)
        if allowed is not None and not allowed:
            return {
                "chunks": [],
                "sufficient_evidence": False,
                "note": f"no chunks match the filter (topic_filter={topic_filter!r}, "
                        f"chunk_types={list(chunk_types or [])})",
                "n_candidates_considered": 0,
            }

        try:
            vectors = [self.embedder.encode_query(q) for q in subqueries]
        except EmbeddingError as exc:
            return _error("embedding_failed", str(exc))

        stage_one = self._pool(subqueries, vectors, allowed)
        self.last_pool = list(stage_one["pool"])
        if not stage_one["pool"]:
            return {
                "chunks": [],
                "sufficient_evidence": False,
                "note": "the index returned no candidates for this query",
                "n_candidates_considered": 0,
            }

        # bi_score is the dense cosine — for several queries, the best over them. It is
        # what the fallback gate thresholds, so it is set for BM25-only candidates too.
        candidates = []
        for chunk_id in stage_one["pool"]:
            cosines = [c[chunk_id] for c in stage_one["cosines"] if chunk_id in c]
            candidates.append({**self._chunks[chunk_id],
                               "bi_score": max(cosines) if cosines else float("-inf")})

        rerank_enabled = bool(self.config.retrieval.rerank_enabled)
        rerank_note = ""
        per_query: List[Dict[str, float]] = []
        if rerank_enabled:
            try:
                ranked, per_query = self._rerank(subqueries, candidates, stage_one["retrieved_by"])
            except RerankError as exc:
                # Reranking is precision, not correctness. Losing it degrades ordering;
                # refusing to answer would be worse, so fall back to the bi-encoder.
                rerank_enabled = False
                rerank_note = f" (rerank unavailable: {exc})"
        if not rerank_enabled:
            # Fallback: keep the fused (RRF) order, but gate on the dense cosine. An RRF
            # score is a sum of reciprocal ranks — it says nothing about relevance, and
            # every candidate has one, so thresholding it would remove abstention.
            ranked = [{**c, "rerank_score": c["bi_score"]} for c in candidates]
            per_query = list(stage_one["cosines"])
            if multi:
                for item in ranked:
                    best = max(range(len(subqueries)),
                               key=lambda i: per_query[i].get(item["chunk_id"], float("-inf")))
                    item["matched_query"] = subqueries[best]
            self.last_rerank_ms, self.last_rerank_pairs = 0, 0

        # The gate is applied per chunk, not only to the best one. Gating on the top
        # score alone still hands back everything below it, which is the thing
        # docs/ARCHITECTURE.md section 7 says not to do: a query whose best hit scrapes
        # over the line would drag four weak chunks into the context window with it.
        # The threshold has to match whatever produced the score. Cross-encoder output is
        # an uncalibrated logit spanning roughly -11..+11; bi-encoder output is a cosine
        # in 0..1. Applying the cross-encoder's threshold to cosine scores lets every
        # candidate through, which does not fail loudly — it silently removes abstention
        # and answers absent-topic questions from whatever ranked first. That is the
        # failure mode here, so the threshold is chosen by which scorer actually ran,
        # including when reranking was requested but fell back above.
        threshold = float(
            self.config.retrieval.relevance_threshold if rerank_enabled
            else self.config.retrieval.bi_encoder_relevance_threshold
        )
        # Max over candidates, not ranked[0]: on the fallback path the list is in RRF
        # order, and its first entry need not hold the best cosine.
        top_score = max((item["rerank_score"] for item in ranked), default=float("-inf"))
        passing = [item for item in ranked if item["rerank_score"] >= threshold]

        extra: Dict[str, Any] = {}
        if multi:
            # Which sub-topic found nothing: the overall gate passes if any sub-query
            # does, so without this a missing half of the question would go unnoticed.
            extra["subquery_sufficient"] = {
                q: any(score >= threshold for score in scores.values())
                for q, scores in zip(subqueries, per_query)
            }

        if not passing:
            # Abstention as a mechanism: naive top-k always returns something, so without
            # this the model is handed weak chunks and invited to answer from them.
            return {
                "chunks": [],
                "sufficient_evidence": False,
                "note": f"best relevance score {top_score:.3f} is below the threshold "
                        f"{threshold:.3f}; the index has nothing good for this query"
                        + rerank_note,
                "n_candidates_considered": len(candidates),
                **extra,
                "coverage_gap": self._coverage_gap(
                    label, ranked, top_score, threshold, rerank_enabled,
                    filtered=bool(topic_filter or chunk_types),
                ),
            }

        selected: List[Dict[str, Any]] = []
        suppressed = 0
        for item in passing:
            if len(selected) >= k_final:
                break
            if item["chunk_id"] in self._returned:
                suppressed += 1
                continue
            selected.append(item)

        weak_dropped = len(ranked) - len(passing)

        for item in selected:
            self._returned.add(item["chunk_id"])

        notes: List[str] = []
        if suppressed:
            notes.append(f"{suppressed} chunk(s) were already returned earlier and were not repeated")
        if weak_dropped and len(selected) < k_final:
            notes.append(f"{weak_dropped} candidate(s) scored below the relevance threshold "
                         f"and were dropped, so fewer than k chunks are returned")
        if multi and not all(extra["subquery_sufficient"].values()):
            missing = [q for q, ok in extra["subquery_sufficient"].items() if not ok]
            notes.append("nothing cleared the relevance threshold for: "
                         + "; ".join(repr(q) for q in missing))
        if multi:
            # Max-score fusion lets one sub-query's strong chunks fill k, so a sub-query
            # can clear the gate and still have none of its chunks returned. Said
            # explicitly, so "sufficient" is not read as "its evidence is in front of you".
            shown = {item.get("matched_query") for item in selected}
            crowded = [q for q, ok in extra["subquery_sufficient"].items() if ok and q not in shown]
            if crowded and selected:
                notes.append("matched above the threshold but outranked by the other "
                             "sub-queries, so none of its chunks are shown (ask it on its "
                             "own to see them): " + "; ".join(repr(q) for q in crowded))
        if not selected:
            notes = ["every chunk above the relevance threshold was already returned earlier "
                     "in this conversation; rephrase the query or work with what you have"]
        note = "; ".join(notes)

        chunks = []
        for item in selected:
            record = self._present(item, item["rerank_score"])
            if multi and item.get("matched_query"):
                record["matched_query"] = item["matched_query"]
            chunks.append(record)

        return {
            "chunks": chunks,
            "sufficient_evidence": bool(selected),
            "note": note + rerank_note,
            "n_candidates_considered": len(candidates),
            "reranked": rerank_enabled,
            "rerank_ms": self.last_rerank_ms,
            **extra,
        }
