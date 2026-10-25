"""
Phase-A retrieval diagnosis: where gold chunks are lost — the pool, the rerank, or the query.

In:  eval/questions.json, the on-disk index, and optionally a logs/traces directory (read
     only) to replay each question's first agent retrieve_evidence call. Local, no LLM.
Out: eval/results/retrieval_diagnosis.json (per gold chunk: bi-encoder rank and post-rerank
     rank, pool and rerank recall, raw question vs agent query) plus a printed summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.metrics import mean  # noqa: E402
from eval.retrieval_eval import config_with, load_questions  # noqa: E402

POOL_KS = (5, 10, 20, 40)
RERANK_KS = (1, 3, 5)

# The live-run failure the user reported: the agent said SparseMixer reports no downstream
# results; 2310_00811v1__c0023 (and its neighbour c0022) say it does. The live query is not
# in any kept trace, so these are reconstructions of how such a question gets phrased.
CASE_CHUNKS = ("2310_00811v1__c0023", "2310_00811v1__c0022")
CASE_QUERIES = (
    "Does SparseMixer report any downstream results?",
    "SparseMixer downstream performance results",
    "SparseMixer downstream task evaluation fine-tuning results",
    "Switch Transformer with SparseMixer downstream performance compared to dense model",
    "sparse backpropagation MoE training downstream results",
    "does sparse mixer improve downstream tasks",
    "SparseMixer only reports training speedup no downstream evaluation",
    "SparseMixer results",
    "SparseMixer experiments evaluation",
    "What does SparseMixer achieve",
    "SparseMixer GLUE fine-tuning results",
    "SparseMixer limitations",
    "SparseMixer machine translation BLEU",
    "Sparse Backpropagation for MoE Training experimental results",
    "SparseMixer gradient estimator results benchmarks",
    "SparseMixer pre-training convergence speed",
)


def first_agent_calls(traces_dir: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """question_id -> the args of that run's first retrieve_evidence call. Read only."""
    calls: Dict[str, Dict[str, Any]] = {}
    if traces_dir is None or not traces_dir.is_dir():
        return calls
    for path in sorted(traces_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = record.get("question_id")
            if qid and record.get("tool_name") == "retrieve_evidence" and qid not in calls:
                calls[qid] = {**(record.get("args") or {}), "_trace": path.name}
                break
    return calls


class Probe:
    """The current pipeline's two stages, taken apart so each can be measured alone."""

    def __init__(self, config: Any) -> None:
        from corpus.chunk_store import ChunkStore
        from corpus.manifest import Manifest
        from retrieval.embedder import Embedder
        from retrieval.reranker import Reranker
        from retrieval.vector_index import VectorIndex

        self.config = config
        self.embedder = Embedder(config)
        self.reranker = Reranker(config)
        self.index = VectorIndex.load(Manifest.load(config, strict_model_check=False), config)
        self.chunks = {c["chunk_id"]: c for c in ChunkStore(config=config)}
        self.k_retrieve = int(config.retrieval.k_retrieve)
        self.threshold = float(config.retrieval.relevance_threshold)

    def allowed(self, topic_filter: Optional[str], chunk_types: Optional[Sequence[str]]):
        if not topic_filter and not chunk_types:
            return None
        wanted = {str(t).lower() for t in (chunk_types or [])}
        return {cid for cid, c in self.chunks.items()
                if (not topic_filter or topic_filter in (c.get("topic_tags") or []))
                and (not wanted or str(c.get("chunk_type", "")).lower() in wanted)}

    def run(self, query: str, gold: Sequence[str], topic_filter: Optional[str] = None,
            chunk_types: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Full bi-encoder ranking, the k_retrieve pool, and the rerank of that pool."""
        allowed = self.allowed(topic_filter, chunk_types)
        vector = self.embedder.encode_query(query)
        full = self.index.search(vector, len(self.index), allowed=allowed)
        bi_rank = {cid: rank for rank, (cid, _) in enumerate(full, start=1)}
        bi_score = dict(full)
        pool = [cid for cid, _ in full[: self.k_retrieve]]
        ranked = self.reranker.rerank(query, [self.chunks[c] for c in pool if c in self.chunks])
        rr_rank = {r["chunk_id"]: rank for rank, r in enumerate(ranked, start=1)}
        rr_score = {r["chunk_id"]: r["rerank_score"] for r in ranked}
        # Diagnosis only: score the (<=6) gold chunks that missed the pool with the
        # cross-encoder, to ask "had they been shortlisted, would rerank have kept them?"
        # This is never done in the pipeline, and never over the index.
        missing = [g for g in gold if g not in rr_rank and g in self.chunks]
        rescue = {}
        if missing:
            rescued = self.reranker.rerank(query, [self.chunks[g] for g in missing])
            rescue = {r["chunk_id"]: r["rerank_score"] for r in rescued}
        fifth = ranked[4]["rerank_score"] if len(ranked) >= 5 else float("-inf")
        order = [r["chunk_id"] for r in ranked]
        gold_set = set(gold)
        passing = [r["chunk_id"] for r in ranked if r["rerank_score"] >= self.threshold]
        per_gold = []
        for g in gold:
            entry = {
                "chunk_id": g,
                "section": (self.chunks.get(g) or {}).get("section"),
                "bi_rank": bi_rank.get(g, "filtered out"),
                "bi_score": round(bi_score[g], 4) if g in bi_score else None,
                "rerank_rank": rr_rank.get(g, "not in pool"),
                "rerank_score": round(rr_score[g], 3) if g in rr_score else None,
            }
            if g in rescue:
                entry["ce_score_if_shortlisted"] = round(rescue[g], 3)
                entry["would_make_top5"] = rescue[g] > fifth
            per_gold.append(entry)
        n = len(gold_set) or 1
        return {
            "query": query,
            "topic_filter": topic_filter,
            "chunk_types": list(chunk_types) if chunk_types else None,
            "pool_recall": {str(k): round(len(set(pool[:k]) & gold_set) / n, 4) for k in POOL_KS},
            "rerank_recall": {str(k): round(len(set(order[:k]) & gold_set) / n, 4) for k in RERANK_KS},
            "gated_recall@5": round(len(set(passing[:5]) & gold_set) / n, 4),
            "top_rerank_score": round(ranked[0]["rerank_score"], 3) if ranked else None,
            "gold": per_gold,
            "top5": [{"chunk_id": r["chunk_id"], "rerank_score": round(r["rerank_score"], 3)}
                     for r in ranked[:5]],
        }


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean pool and rerank recall over rows, plus where each gold chunk ended up."""
    if not rows:
        return {}
    out = {
        "n_questions": len(rows),
        "pool_recall": {str(k): mean([r["pool_recall"][str(k)] for r in rows]) for k in POOL_KS},
        "rerank_recall": {str(k): mean([r["rerank_recall"][str(k)] for r in rows]) for k in RERANK_KS},
        "gated_recall@5": mean([r["gated_recall@5"] for r in rows]),
    }
    golds = [g for r in rows for g in r["gold"]]
    in_pool = [g for g in golds if isinstance(g["rerank_rank"], int)]
    out["gold_total"] = len(golds)
    out["gold_in_pool"] = len(in_pool)
    out["gold_in_pool_top5_after_rerank"] = sum(1 for g in in_pool if g["rerank_rank"] <= 5)
    out["gold_in_bi_top5"] = sum(1 for g in golds if isinstance(g["bi_rank"], int) and g["bi_rank"] <= 5)
    missing = [g for g in golds if not isinstance(g["rerank_rank"], int)]
    out["gold_missing_pool_but_ce_would_top5"] = sum(1 for g in missing if g.get("would_make_top5"))
    ranks = [g["bi_rank"] for g in golds if isinstance(g["bi_rank"], int)]
    out["median_bi_rank_of_gold"] = statistics.median(ranks) if ranks else None
    return out


def diagnose_split(probe: Probe, held_out: bool, agent_calls: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    questions = [q for q in load_questions(held_out=held_out) if not q["expect_abstention"]]
    raw_rows, agent_rows = [], []
    for q in questions:
        row = probe.run(q["question"], q["gold_chunk_ids"])
        row["question_id"] = q["question_id"]
        raw_rows.append(row)
        call = agent_calls.get(q["question_id"])
        if call and call.get("query"):
            arow = probe.run(call["query"], q["gold_chunk_ids"],
                             call.get("topic_filter"), call.get("chunk_types"))
            arow["question_id"] = q["question_id"]
            arow["k"] = call.get("k")
            arow["trace"] = call.get("_trace")
            agent_rows.append(arow)
    return {
        "raw_question": {"summary": summarise(raw_rows), "rows": raw_rows},
        "agent_first_query": {"summary": summarise(agent_rows), "rows": agent_rows,
                              "missing_traces": [q["question_id"] for q in questions
                                                 if q["question_id"] not in agent_calls]},
    }


def case_study(probe: Probe) -> Dict[str, Any]:
    """Where the SparseMixer downstream-results chunks rank, query by query."""
    out = []
    for query in CASE_QUERIES:
        row = probe.run(query, list(CASE_CHUNKS))
        out.append({"query": query, "gold": row["gold"], "top5": row["top5"],
                    "top_rerank_score": row["top_rerank_score"]})
    return {"chunks": list(CASE_CHUNKS), "queries": out}


# Multi-query calls the agent could make for a "does it report X?" question: the broad
# wording that loses c0023 at k=5, paired with a sub-query naming the specific part.
CASE_MULTI = (
    ["SparseMixer results", "SparseMixer downstream performance"],
    ["SparseMixer limitations", "SparseMixer downstream results"],
    ["SparseMixer gradient estimator results benchmarks", "SparseMixer downstream tasks"],
)


def case_study_pipeline(config: Any) -> Dict[str, Any]:
    """The same case through the live retrieve_evidence (whatever config is on: hybrid,
    headers, multi-query): is c0023 in the fused pool, where does it rank, is it in the
    k=5 the agent sees."""
    from tools.retrieve_evidence import EvidenceRetriever

    retriever = EvidenceRetriever(config)

    def probe(call: Dict[str, Any]) -> Dict[str, Any]:
        retriever.reset()
        wide = retriever.retrieve(k=int(config.retrieval.k_retrieve), **call)
        pool = list(retriever.last_pool)
        retriever.reset()
        top5 = retriever.retrieve(k=5, **call)
        ranked = [c["chunk_id"] for c in wide.get("chunks", [])]
        return {
            **call,
            "pool_size": len(pool),
            "chunks": {cid: {"in_pool": cid in pool,
                             "rank_after_gate": ranked.index(cid) + 1 if cid in ranked else None,
                             "in_top5": cid in [c["chunk_id"] for c in top5.get("chunks", [])]}
                       for cid in CASE_CHUNKS},
            "subquery_sufficient": top5.get("subquery_sufficient"),
        }

    return {
        "config": {"contextual_headers": bool(config.retrieval.get("contextual_headers", False)),
                   "hybrid": bool(config.retrieval.get("hybrid", False)),
                   "multi_query_rerank": config.retrieval.get("multi_query_rerank")},
        "single": [probe({"query": q}) for q in CASE_QUERIES],
        "multi": [probe({"queries": list(qs)}) for qs in CASE_MULTI],
    }


def _print_summary(label: str, block: Dict[str, Any]) -> None:
    for variant in ("raw_question", "agent_first_query"):
        s = block[variant]["summary"]
        if not s:
            print(f"  {label} / {variant}: no rows")
            continue
        pool = " ".join(f"@{k}={s['pool_recall'][str(k)]:.3f}" for k in POOL_KS)
        rr = " ".join(f"@{k}={s['rerank_recall'][str(k)]:.3f}" for k in RERANK_KS)
        print(f"  {label:<9} {variant:<18} n={s['n_questions']:<2} pool {pool} | rerank {rr} | "
              f"gold in pool {s['gold_in_pool']}/{s['gold_total']}, top5 after rerank "
              f"{s['gold_in_pool_top5_after_rerank']}, in bi top5 {s['gold_in_bi_top5']}, "
              f"missed pool but CE would top5 {s['gold_missing_pool_but_ce_would_top5']}, "
              f"median bi rank {s['median_bi_rank_of_gold']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--traces", type=Path, default=ROOT / "logs" / "traces",
                        help="a logs/traces directory to replay first agent queries from (read only)")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "eval" / "results" / "retrieval_diagnosis.json")
    parser.add_argument("--case-only", action="store_true",
                        help="only the SparseMixer case, through the live pipeline as well")
    args = parser.parse_args()

    config = config_with()
    if args.case_only:
        result = {"sparsemixer_case": case_study(Probe(config)),
                  "sparsemixer_case_pipeline": case_study_pipeline(config)}
        for row in result["sparsemixer_case_pipeline"]["single"] + \
                result["sparsemixer_case_pipeline"]["multi"]:
            asked = row.get("query") or row.get("queries")
            c = row["chunks"][CASE_CHUNKS[0]]
            print(f"  {str(asked)[:80]!r}: c0023 pool={c['in_pool']} "
                  f"rank={c['rank_after_gate']} top5={c['in_top5']}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"\nwritten: {args.out}")
        return 0
    probe = Probe(config)
    calls = first_agent_calls(args.traces)
    result = {
        "config": {"k_retrieve": probe.k_retrieve, "relevance_threshold": probe.threshold,
                   "contextual_headers": bool(config.retrieval.get("contextual_headers", False)),
                   "hybrid": bool(config.retrieval.get("hybrid", False)),
                   "traces": str(args.traces),
                   "probe": "dense-only pool (bi-encoder top-40) + cross-encoder rerank, "
                            "whatever retrieval.hybrid says; sparsemixer_case_pipeline "
                            "(--case-only) is the live pipeline"},
        "tuning": diagnose_split(probe, False, calls),
        "held_out": diagnose_split(probe, True, calls),
        "sparsemixer_case": case_study(probe),
    }
    print("retrieval diagnosis (tuning split is the one to read; held-out is reported only)")
    _print_summary("tuning", result["tuning"])
    _print_summary("held_out", result["held_out"])
    print("\nSparseMixer downstream case:")
    for row in result["sparsemixer_case"]["queries"]:
        marks = ", ".join(f"{g['chunk_id'][-5:]} bi#{g['bi_rank']} rr#{g['rerank_rank']}"
                          f" ce={g['rerank_score'] if g['rerank_score'] is not None else g.get('ce_score_if_shortlisted')}"
                          for g in row["gold"])
        print(f"  {row['query'][:70]!r}: {marks}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
