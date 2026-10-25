"""
Offline tests for the retrieval upgrade: headers, the embed hash, BM25 + RRF, multi-query.

In:  tmp_path corpora with a bag-of-words stub embedder (zlib.crc32, never hash()) and
     query-aware stub rerankers — no models, no network.
Out: assertions on the tokeniser, BM25 and RRF math, re-embedding on a header change, that
     headers never reach the cross-encoder or the agent, the fallback gate, multi-query.
"""

import zlib

import numpy as np
import pytest
import yaml

from agent.fetch_proposal import derive_fetch_proposal
from agent.tool_registry import build_default_registry
from common.records import chunk_record, content_hash, paper_record
from config import load_config
from corpus.chunk_store import ChunkStore
from corpus.manifest import Manifest
from retrieval import contextual
from retrieval.embedder import Embedder
from retrieval.indexer import index_chunks
from retrieval.lexical_index import LexicalIndex, reciprocal_rank_fusion, tokenize
from retrieval.reranker import RerankError
from tools.retrieve_evidence import EvidenceRetriever


# --------------------------------------------------------------------------- fixtures

@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    import config as config_module

    repo_root = config_module.REPO_ROOT
    local = tmp_path / "config.yaml"
    local.write_text((repo_root / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(config_module, "REPO_ROOT", tmp_path)
    built = load_config(local)
    built.paths.ensure()
    built.paths.prompts = repo_root / "src" / "prompts"
    return built


def with_retrieval(base, **overrides):
    """A copy of `base` with retrieval settings overridden, written through the YAML."""
    raw = yaml.safe_load(base.source_path.read_text(encoding="utf-8"))
    raw["retrieval"].update(overrides)
    name = "_".join(f"{k}-{v}" for k, v in sorted(overrides.items())).replace("/", "")
    scratch = base.paths.root / f"override_{zlib.crc32(name.encode())}.yaml"
    scratch.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    built = load_config(scratch)
    built.paths.prompts = base.paths.prompts
    return built


class BagEmbedder(Embedder):
    """Normalised bag of words, hashed with crc32: cosine is real word overlap.

    The query prefix is stripped so it does not add shared words to every query.
    """

    def __init__(self, config):
        super().__init__(config)
        self.encoded = []

    def _encode(self, texts):
        self.encoded.extend(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            if text.startswith(self.query_prefix):
                text = text[len(self.query_prefix):]
            for word in tokenize(text):
                out[row, zlib.crc32(word.encode("utf-8")) % self.dim] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class TableReranker:
    """Scores (query, chunk) by a table keyed on (query, chunk_id), else on chunk_id,
    else `default`. Records every (query, text) pair it was shown."""

    def __init__(self, scores=None, default=-10.0):
        self.scores = scores or {}
        self.default = default
        self.pairs = []
        self.batch_sizes = []
        self.last_latency_ms = 5

    def rerank(self, query, candidates):
        self.batch_sizes.append(len(candidates))
        out = []
        for c in candidates:
            self.pairs.append((query, c["text"], c["chunk_id"]))
            score = self.scores.get((query, c["chunk_id"]),
                                    self.scores.get(c["chunk_id"], self.default))
            out.append({**c, "rerank_score": score})
        out.sort(key=lambda item: item["rerank_score"], reverse=True)
        return out


class DeadReranker:
    last_latency_ms = 0

    def rerank(self, query, candidates):
        raise RerankError("model checkpoint missing")


def build_corpus(config, papers, embedder=None):
    """papers: {paper_id: (title, tags, [(text, section), ...])}"""
    manifest = Manifest.load(config, strict_model_check=False)
    store = ChunkStore(config=config)
    for paper_id, (title, tags, items) in papers.items():
        manifest.add_paper(paper_record(
            arxiv_id=paper_id, title=title, authors=[], abstract="", published="2024-01-01",
            year=2024, categories=[], pdf_url="", pdf_path="", topic_tags=tags))
        store.append([
            chunk_record(paper_id=paper_id, paper_title=title, topic_tags=tags,
                         chunk_type="text", text=text, page=1, position=position,
                         n_tokens=10, section=section)
            for position, (text, section) in enumerate(items)
        ])
    manifest.save()
    return index_chunks(config, embedder=embedder or BagEmbedder(config))


PAPERS = {
    "pA": ("SparseMixer Gradient Estimation", ["moe"], [
        ("expert routing gradients are estimated with a midpoint method", "Method"),
        ("downstream results improve over the dense baseline", "Experiments"),
        ("training converges twice as fast", "Experiments"),
    ]),
    "pB": ("Structured Pruning Without Retraining", ["pruning"], [
        ("calibration samples decide which heads to remove", "Method"),
        ("perplexity after pruning stays close to the original", "Results"),
    ]),
}


@pytest.fixture()
def corpus(cfg):
    build_corpus(cfg, PAPERS)
    return cfg


# --------------------------------------------------------------------------- tokeniser

def test_tokeniser_keeps_acronyms_names_numbers_and_greek():
    tokens = tokenize("TRSP and SparseMixer use lr 1e-2, then 0.01; λ=0.5 and α. Done.")
    for term in ("trsp", "sparsemixer", "1e-2", "0.01", "λ", "α", "lr", "0.5", "done"):
        assert term in tokens, term
    assert "1e" not in tokens and "2" not in tokens, "numeric compounds stay whole"
    assert "and" not in tokens and "use" in tokens


def test_tokeniser_splits_hyphenated_words_into_parts_too():
    tokens = tokenize("mixture-of-experts top-k routing")
    assert "mixture-of-experts" in tokens and "mixture" in tokens and "experts" in tokens
    assert "top-k" in tokens and "routing" in tokens


def test_tokeniser_keeps_negations():
    assert "no" in tokenize("no retraining") and "without" in tokenize("without retraining")


# ------------------------------------------------------------------------------ BM25

def test_bm25_ranks_the_rare_term_match_first_and_skips_non_matches():
    index = LexicalIndex([
        ("d1", "the model uses SparseMixer for routing"),
        ("d2", "the model uses dense routing"),
        ("d3", "baking bread at home"),
    ])
    hits = index.search("SparseMixer routing", k=5)
    assert [cid for cid, _ in hits] == ["d1", "d2"], "d3 shares no term and is no candidate"
    assert hits[0][1] > hits[1][1] > 0


def test_bm25_idf_and_length_normalisation():
    index = LexicalIndex([("short", "pruning"), ("long", "pruning " + "filler " * 40),
                          ("other", "routing")], k1=1.2, b=0.75)
    scores = dict(index.search("pruning", k=3))
    assert scores["short"] > scores["long"], "b=0.75 penalises the longer document"
    # Okapi with Lucene idf: idf = ln(1 + (N - df + 0.5) / (df + 0.5)), N=3, df=2.
    idf = np.log1p((3 - 2 + 0.5) / (2 + 0.5))
    lengths = np.array([1, 41, 1], dtype=float)
    norm = 1.2 * (1 - 0.75 + 0.75 * 1 / lengths.mean())
    assert scores["short"] == pytest.approx(idf * 1 * 2.2 / (1 + norm))


def test_bm25_respects_the_allowed_filter():
    index = LexicalIndex([("a", "routing experts"), ("b", "routing experts again")])
    assert [cid for cid, _ in index.search("routing", k=5, allowed={"b"})] == ["b"]


# ------------------------------------------------------------------------------- RRF

def test_rrf_sums_reciprocal_ranks():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], rrf_k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 62)
    order = [cid for cid, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], 60)]
    assert order[0] == "b", "found by both lists beats first in one"


def test_rrf_limit_caps_the_pool():
    lists = [[f"x{i}" for i in range(40)], [f"y{i}" for i in range(40)]]
    assert len(reciprocal_rank_fusion(lists, 60, limit=40)) == 40


# ------------------------------------------------------- headers and the embed hash

def test_header_is_title_and_section_and_skips_a_section_that_repeats_the_title():
    chunk = {"paper_title": "Paper T", "section": "Results", "text": "body"}
    assert contextual.header_for(chunk) == "Paper T — Results"
    assert contextual.header_for({**chunk, "section": "paper t"}) == "Paper T"
    assert contextual.header_for({**chunk, "section": None}) == "Paper T"


def test_embed_text_follows_the_flag(cfg):
    chunk = {"paper_title": "Paper T", "section": "Results", "text": "body text"}
    on = with_retrieval(cfg, contextual_headers=True)
    off = with_retrieval(cfg, contextual_headers=False)
    assert contextual.embed_text(chunk, on)[0] == "Paper T — Results\nbody text"
    assert contextual.embed_text(chunk, off)[0] == "body text"
    assert contextual.embed_hash("x", on) != contextual.embed_hash("x", off)


class WordCounter:
    """A token counter that counts words, so the budget arithmetic is visible."""

    def count(self, text):
        return len(text.split())

    def truncate(self, text, budget):
        return " ".join(text.split()[:budget])


def test_an_over_budget_header_is_truncated_never_the_chunk(cfg):
    on = with_retrieval(cfg, contextual_headers=True)
    budget = contextual.bi_encoder_budget(on)
    text = " ".join(["w"] * (budget - 3))
    chunk = {"paper_title": "A Very Long Paper Title Here", "section": "Results", "text": text}
    out, truncated = contextual.embed_text(chunk, on, WordCounter())
    assert truncated is True
    assert out.endswith("\n" + text), "the chunk text is intact"
    assert WordCounter().count(out) <= budget
    assert out.split("\n")[0] == "A Very Long"


def test_changing_the_header_format_re_embeds_while_content_hash_stays(cfg, monkeypatch):
    """The known bug: content_hash does not move when only the header does, so deciding
    on it would keep every stale vector. The embed hash must move, and force a rebuild."""
    on = with_retrieval(cfg, contextual_headers=True)
    build_corpus(on, PAPERS)
    before = ChunkStore(config=on).all()
    hashes_before = Manifest.load(on, strict_model_check=False).data["indexed_hashes"]
    assert index_chunks(on, embedder=BagEmbedder(on))["chunks_indexed"] == 0, "settled"

    monkeypatch.setattr(contextual, "HEADER_FORMAT_VERSION", 2)
    monkeypatch.setattr(contextual, "SEPARATOR", " | ")
    embedder = BagEmbedder(on)
    result = index_chunks(on, embedder=embedder)
    assert result["rebuilt"] is True and result["chunks_indexed"] == 5
    assert any(" | " in text for text in embedder.encoded), "the new header was embedded"

    after = ChunkStore(config=on).all()
    assert [c["content_hash"] for c in after] == [c["content_hash"] for c in before]
    assert [c["chunk_id"] for c in after] == [c["chunk_id"] for c in before]
    assert all(c["content_hash"] == content_hash(c["text"]) for c in after)
    manifest = Manifest.load(on, strict_model_check=False).data
    assert set(manifest["indexed_hashes"]).isdisjoint(hashes_before)
    assert manifest["embed_format"] == "header-v2"
    assert len(manifest["faiss_id_map"]) == len(manifest["indexed_hashes"]) == 5


def test_toggling_headers_re_embeds_and_the_retriever_refuses_a_mismatched_index(cfg):
    off = with_retrieval(cfg, contextual_headers=False)
    build_corpus(off, PAPERS)
    on = with_retrieval(cfg, contextual_headers=True)
    stale = EvidenceRetriever(on, embedder=BagEmbedder(on), reranker=TableReranker())
    assert stale.retrieve("routing")["error"] == "index_stale"

    result = index_chunks(on, embedder=BagEmbedder(on))
    assert result["rebuilt"] is True and result["chunks_indexed"] == 5
    fresh = EvidenceRetriever(on, embedder=BagEmbedder(on), reranker=TableReranker(default=1.0))
    assert fresh.retrieve("routing")["sufficient_evidence"] is True


def test_headers_make_the_title_findable_in_the_first_stage(cfg):
    """A query naming only the paper's title finds its chunks once headers are on, by
    dense cosine and by BM25 alike: neither chunk text contains the word."""
    on = with_retrieval(cfg, contextual_headers=True, hybrid=True,
                        bi_encoder_relevance_threshold=0.05)
    build_corpus(on, PAPERS)
    agent = EvidenceRetriever(on, embedder=BagEmbedder(on), reranker=DeadReranker())
    result = agent.retrieve("sparsemixer", k=5)
    assert {c["paper_id"] for c in result["chunks"]} == {"pA"}
    lexical = agent._lexical_index().search("sparsemixer", k=5)
    assert {cid for cid, _ in lexical} == {"pA__c0000", "pA__c0001", "pA__c0002"}

    off = with_retrieval(cfg, contextual_headers=False, hybrid=True,
                         bi_encoder_relevance_threshold=0.05)
    index_chunks(off, embedder=BagEmbedder(off))
    plain = EvidenceRetriever(off, embedder=BagEmbedder(off), reranker=DeadReranker())
    assert plain.retrieve("sparsemixer")["sufficient_evidence"] is False


def test_headers_never_reach_the_cross_encoder_or_the_agent(cfg):
    """Headers are first-stage only: the cross-encoder and the returned text see the
    plain chunk, exactly as stored."""
    on = with_retrieval(cfg, contextual_headers=True, hybrid=True)
    build_corpus(on, PAPERS)
    reranker = TableReranker(default=1.0)
    agent = EvidenceRetriever(on, embedder=BagEmbedder(on), reranker=reranker)
    result = agent.retrieve("sparsemixer gradient", k=5)
    assert result["sufficient_evidence"] is True and reranker.pairs
    stored = {c["chunk_id"]: c["text"] for c in ChunkStore(config=on)}
    for _, text, chunk_id in reranker.pairs:
        assert text == stored[chunk_id]
    for title, _, _ in PAPERS.values():
        assert all(title not in text for _, text, _ in reranker.pairs)
        assert all(title not in chunk["text"] for chunk in result["chunks"])


# ------------------------------------------------------------------- hybrid + gating

class BlindEmbedder(Embedder):
    """One-hot on the crc32 of the whole text: dense matches nothing but an exact copy."""

    def _encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row, zlib.crc32(text.encode("utf-8")) % self.dim] = 1.0
        return out


def test_bm25_brings_in_a_candidate_the_dense_list_missed(cfg):
    """With k_retrieve=2 and a dense model blind to the query, the dense list is two
    arbitrary chunks; BM25 contributes the one chunk that holds the exact term."""
    small = with_retrieval(cfg, hybrid=True, k_retrieve=2)
    build_corpus(small, PAPERS, embedder=BlindEmbedder(small))
    agent = EvidenceRetriever(small, embedder=BlindEmbedder(small),
                              reranker=TableReranker(default=1.0))
    agent.retrieve("perplexity", k=5)
    assert "pB__c0001" in agent.last_pool
    assert len(agent.last_pool) <= 2, "the fused pool is capped at k_retrieve"

    dense_only = with_retrieval(cfg, hybrid=False, k_retrieve=2)
    blind = EvidenceRetriever(dense_only, embedder=BlindEmbedder(dense_only),
                              reranker=TableReranker(default=1.0))
    blind.retrieve("perplexity", k=5)
    dense_ids = [cid for cid, _ in blind._index.search(
        BlindEmbedder(dense_only).encode_query("perplexity"), 2)]
    assert blind.last_pool == dense_ids, "hybrid off: the pool is the dense top-k, as before"


def test_filters_apply_to_bm25_exactly_as_to_dense(corpus):
    hybrid = with_retrieval(corpus, hybrid=True)
    agent = EvidenceRetriever(hybrid, embedder=BagEmbedder(hybrid), reranker=TableReranker(default=1.0))
    agent.retrieve("pruning calibration routing", topic_filter="moe")
    assert agent.last_pool and all(cid.startswith("pA") for cid in agent.last_pool)


def test_fallback_gate_thresholds_the_dense_cosine_not_the_rrf_score(corpus):
    """An RRF score is ~1/61 for the top candidate whatever its relevance. A query sharing
    no word with the corpus has cosine 0 everywhere: gating on RRF at 0.01 would pass it."""
    gated = with_retrieval(corpus, hybrid=True, bi_encoder_relevance_threshold=0.01)
    agent = EvidenceRetriever(gated, embedder=BagEmbedder(gated), reranker=DeadReranker())
    absent = agent.retrieve("sourdough fermentation")
    assert absent["sufficient_evidence"] is False
    gap = absent["coverage_gap"]
    assert gap["scorer"] == "bi_encoder" and gap["threshold"] == 0.01
    assert gap["best_score"] == pytest.approx(0.0, abs=1e-6), "the best cosine, not an RRF sum"

    agent.reset()
    present = agent.retrieve("downstream results baseline")
    assert present["sufficient_evidence"] is True and present["reranked"] is False
    top = present["chunks"][0]
    assert top["chunk_id"] == "pA__c0001"
    assert 0.5 < top["score"] <= 1.0, "the reported score is the cosine the gate used"


def test_fallback_with_rerank_disabled_also_gates_on_cosine(corpus):
    off = with_retrieval(corpus, hybrid=True, rerank_enabled=False,
                         bi_encoder_relevance_threshold=0.99)
    agent = EvidenceRetriever(off, embedder=BagEmbedder(off), reranker=TableReranker(default=99))
    assert agent.retrieve("downstream results")["sufficient_evidence"] is False


def test_cross_encoder_gate_still_abstains_on_an_absent_topic_with_hybrid(corpus):
    hybrid = with_retrieval(corpus, hybrid=True)
    agent = EvidenceRetriever(hybrid, embedder=BagEmbedder(hybrid),
                              reranker=TableReranker(default=-9.0))
    result = agent.retrieve("sourdough starter temperature")
    assert result["sufficient_evidence"] is False and result["chunks"] == []
    assert result["coverage_gap"]["scorer"] == "cross_encoder"
    assert result["coverage_gap"]["best_score"] == -9.0


def test_pool_is_capped_with_hybrid_and_three_subqueries(cfg):
    capped = with_retrieval(cfg, hybrid=True, k_retrieve=10, multi_query_rerank="all")
    build_corpus(capped, {"pA": ("Big Paper", ["t"], [
        (f"routing experts chunk {i} gradient pruning", "S") for i in range(60)])})
    reranker = TableReranker(default=1.0)
    agent = EvidenceRetriever(capped, embedder=BagEmbedder(capped), reranker=reranker)
    result = agent.retrieve(queries=["routing", "gradient", "pruning"], k=5)
    assert result["n_candidates_considered"] <= 10
    assert max(reranker.batch_sizes) <= 10, "rerank only ever sees the capped shortlist"


# ---------------------------------------------------------------------- multi-query

def test_multi_query_merges_both_papers_and_reports_which_query_matched(corpus):
    scores = {("downstream results", "pA__c0001"): 4.0,
              ("calibration samples", "pB__c0000"): 3.0}
    agent = EvidenceRetriever(corpus, embedder=BagEmbedder(corpus), reranker=TableReranker(scores))
    result = agent.retrieve(queries=["downstream results", "calibration samples"], k=5)
    ids = [c["chunk_id"] for c in result["chunks"]]
    assert ids == ["pA__c0001", "pB__c0000"]
    assert result["chunks"][0]["matched_query"] == "downstream results"
    assert result["chunks"][1]["matched_query"] == "calibration samples"
    assert result["subquery_sufficient"] == {"downstream results": True,
                                             "calibration samples": True}


def test_multi_query_rerank_takes_the_max_over_subqueries(corpus):
    all_mode = with_retrieval(corpus, multi_query_rerank="all")
    scores = {("q one", "pA__c0001"): -5.0, ("q two", "pA__c0001"): 2.5}
    agent = EvidenceRetriever(all_mode, embedder=BagEmbedder(all_mode),
                              reranker=TableReranker(scores))
    result = agent.retrieve(queries=["q one", "q two"], k=5)
    assert [c["chunk_id"] for c in result["chunks"]] == ["pA__c0001"]
    assert result["chunks"][0]["score"] == 2.5
    assert result["subquery_sufficient"] == {"q one": False, "q two": True}
    assert "'q one'" in result["note"], "the missing sub-topic is named"


def test_retrieving_mode_scores_a_candidate_only_against_queries_that_found_it(corpus):
    candidates = [dict(c) for c in ChunkStore(config=corpus) if c["chunk_id"] in
                  ("pA__c0001", "pB__c0000")]
    found_by = {"pA__c0001": {0}, "pB__c0000": {0, 1}}
    queries = ["downstream results", "calibration samples"]
    for mode, expected in (
        ("all", {(q, c) for q in queries for c in found_by}),
        ("retrieving", {("downstream results", "pA__c0001"),
                        ("downstream results", "pB__c0000"),
                        ("calibration samples", "pB__c0000")}),
    ):
        tuned = with_retrieval(corpus, multi_query_rerank=mode)
        reranker = TableReranker(default=1.0)
        agent = EvidenceRetriever(tuned, embedder=BagEmbedder(tuned), reranker=reranker)
        ranked, per_query = agent._rerank(queries, candidates, found_by)
        assert {(q, cid) for q, _, cid in reranker.pairs} == expected, mode
        assert agent.last_rerank_pairs == len(expected)
        assert {r["chunk_id"] for r in ranked} == set(found_by)


def test_multi_query_gap_joins_queries_and_uses_the_best_overall_score(corpus):
    scores = {("a topic", "pA__c0000"): -4.0, ("b topic", "pB__c0000"): -3.5}
    agent = EvidenceRetriever(corpus, embedder=BagEmbedder(corpus),
                              reranker=TableReranker(scores, default=-9.0))
    result = agent.retrieve(queries=["a topic", "b topic"])
    gap = result["coverage_gap"]
    assert result["sufficient_evidence"] is False
    assert gap["searched_for"] == "a topic; b topic"
    assert gap["best_score"] == -3.5
    assert result["subquery_sufficient"] == {"a topic": False, "b topic": False}


def test_query_alone_is_unchanged_and_query_joins_queries(corpus):
    agent = EvidenceRetriever(corpus, embedder=BagEmbedder(corpus), reranker=TableReranker(default=1.0))
    single = agent.retrieve("routing")
    assert "subquery_sufficient" not in single
    assert all("matched_query" not in c for c in single["chunks"])

    reranker = TableReranker(default=1.0)
    joined = EvidenceRetriever(corpus, embedder=BagEmbedder(corpus), reranker=reranker)
    result = joined.retrieve("routing", queries=["pruning", "Routing"])
    assert list(result["subquery_sufficient"]) == ["routing", "pruning"], "query first, deduped"


def test_bad_query_arguments_are_error_dicts(corpus):
    agent = EvidenceRetriever(corpus, embedder=BagEmbedder(corpus), reranker=TableReranker())
    assert agent.retrieve()["error"] == "empty_query"
    assert agent.retrieve(queries=["  ", ""])["error"] == "empty_query"
    assert agent.retrieve(queries=["a", "b", "c", "d"])["error"] == "too_many_queries"
    assert agent.retrieve("a", queries=["b", "c", "d"])["error"] == "too_many_queries"
    assert agent.retrieve(queries=[1, 2])["error"] == "bad_queries"


def test_registry_schema_offers_queries_capped_by_config(corpus):
    agent = EvidenceRetriever(corpus, embedder=BagEmbedder(corpus), reranker=TableReranker(default=1.0))
    registry = build_default_registry(agent, corpus)
    params = registry.schemas()[0]["function"]["parameters"]
    assert params["properties"]["queries"]["maxItems"] == corpus.retrieval.max_subqueries
    assert "required" not in params
    result = registry.dispatch("retrieve_evidence", {"queries": ["routing", "pruning"]})
    assert result["sufficient_evidence"] is True


def test_fetch_proposal_reads_queries_on_an_empty_index():
    records = [{"tool_name": "retrieve_evidence",
                "args": {"queries": ["diffusion distillation", "few-step sampling"]},
                "result_summary": {"error": "empty_index"}}]
    proposal = derive_fetch_proposal(records)
    assert proposal["query"] == "diffusion distillation; few-step sampling"


def test_fetch_proposal_from_a_failed_multi_query_gap(corpus):
    agent = EvidenceRetriever(corpus, embedder=BagEmbedder(corpus),
                              reranker=TableReranker(default=-9.0))
    result = agent.retrieve(queries=["sourdough", "baking"])
    records = [{"tool_name": "retrieve_evidence", "args": {"queries": ["sourdough", "baking"]},
                "result_summary": result}]
    proposal = derive_fetch_proposal(records)
    assert proposal and proposal["searched_for"] == "sourdough; baking"


def test_a_sufficient_subquery_crowded_out_of_k_is_named_in_the_note(corpus):
    scores = {("strong", "pA__c0000"): 5.0, ("strong", "pA__c0001"): 4.0,
              ("weak", "pB__c0000"): -1.0}
    tuned = with_retrieval(corpus, multi_query_rerank="all")
    agent = EvidenceRetriever(tuned, embedder=BagEmbedder(tuned), reranker=TableReranker(scores))
    result = agent.retrieve(queries=["strong", "weak"], k=2)
    assert [c["chunk_id"] for c in result["chunks"]] == ["pA__c0000", "pA__c0001"]
    assert result["subquery_sufficient"] == {"strong": True, "weak": True}
    assert "outranked" in result["note"] and "'weak'" in result["note"]


def test_fetch_query_rewrite_works_on_a_multi_query_proposal(corpus):
    from agent.fetch_query import rewrite_fetch_query
    from test_m5_agent import ScriptedClient, text_response

    agent = EvidenceRetriever(corpus, embedder=BagEmbedder(corpus),
                              reranker=TableReranker(default=-9.0))
    result = agent.retrieve(queries=["sourdough", "baking"])
    records = [{"tool_name": "retrieve_evidence", "args": {"queries": ["sourdough", "baking"]},
                "result_summary": result}]
    proposal = derive_fetch_proposal(records)
    good = rewrite_fetch_query(dict(proposal), "How do I bake sourdough?",
                               client=ScriptedClient(text_response("sourdough baking")),
                               config=corpus)
    assert good["query"] == "sourdough baking" and good["agent_query"] == "sourdough; baking"
    junk = rewrite_fetch_query(dict(proposal), "How do I bake sourdough?",
                               client=ScriptedClient(text_response("")), config=corpus)
    assert junk["query"] == "sourdough; baking" and junk["query_source"] == "agent"
