"""
Retrieval-only evaluation: recall, MRR and latency for one retrieval configuration.

In:  the question set, plus overrides for retrieval config (threshold, rerank on/off).
Out: per-question and aggregate metrics. Calls retrieve_evidence directly, with no agent
     and no LLM, so a threshold sweep and the rerank ablation are cheap and repeatable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # so `eval.` resolves when run as a script

from eval.metrics import (  # noqa: E402
    any_gold_paper_hit,
    mean,
    recall_at_k,
    reciprocal_rank,
)


def load_questions(held_out: Optional[bool] = None) -> List[Dict[str, Any]]:
    """The question set, optionally filtered to the held-out split or its complement.

    held_out=False is the tuning split. The held-out third is run once, after tuning
    stops; looking at it earlier would turn it into training data.
    """
    payload = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    questions = payload["questions"]
    if held_out is None:
        return questions
    return [q for q in questions if bool(q["held_out"]) is held_out]


def config_with(**retrieval_overrides: Any):
    """A Config identical to the project's but with retrieval settings overridden.

    Written through the YAML rather than poked into the loaded object: config sections
    are read-only on purpose, and a sweep that mutates them in place would be testing a
    state the real system can never be in.
    """
    import config as config_module

    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["retrieval"].update(retrieval_overrides)
    scratch = ROOT / "eval" / "results" / "_sweep_config.yaml"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_module.load_config(scratch)


def evaluate_retrieval(
    questions: List[Dict[str, Any]],
    config: Any,
    k: int = 5,
    recall_ks: Optional[List[int]] = None,
    embedder: Any = None,
    reranker: Any = None,
) -> Dict[str, Any]:
    """Run retrieve_evidence over every question and score it against the gold labels."""
    from tools.retrieve_evidence import EvidenceRetriever

    recall_ks = recall_ks or [1, 3, 5]
    retriever = EvidenceRetriever(config, embedder=embedder, reranker=reranker)

    rows: List[Dict[str, Any]] = []
    for question in questions:
        retriever.reset()
        started = time.perf_counter()
        result = retriever.retrieve(question["question"], k=k)
        latency_ms = int((time.perf_counter() - started) * 1000)

        chunks = result.get("chunks", []) if "error" not in result else []
        retrieved = [c["chunk_id"] for c in chunks]
        papers = [c["paper_id"] for c in chunks]
        gold = question["gold_chunk_ids"]

        rows.append({
            "question_id": question["question_id"],
            "category": question["category"],
            "expect_abstention": question["expect_abstention"],
            "sufficient_evidence": bool(result.get("sufficient_evidence", False)),
            "n_retrieved": len(retrieved),
            "recall": {str(n): recall_at_k(retrieved, gold, n) for n in recall_ks},
            "mrr": reciprocal_rank(retrieved, gold),
            "gold_paper_hit": any_gold_paper_hit(papers, question["gold_paper_ids"]),
            "latency_ms": latency_ms,
            "retrieved": retrieved,
        })

    answerable = [r for r in rows if not r["expect_abstention"]]
    abstaining = [r for r in rows if r["expect_abstention"]]

    return {
        "n_questions": len(rows),
        "recall": {str(n): mean([r["recall"][str(n)] for r in answerable]) for n in recall_ks},
        "mrr": mean([r["mrr"] for r in answerable]),
        "gold_paper_hit_rate": mean([float(r["gold_paper_hit"]) for r in answerable]),
        "gate_correct_on_answerable": mean(
            [float(r["sufficient_evidence"]) for r in answerable]
        ),
        "gate_correct_on_absent": mean(
            [float(not r["sufficient_evidence"]) for r in abstaining]
        ),
        "median_latency_ms": int(sorted(r["latency_ms"] for r in rows)[len(rows) // 2]) if rows else 0,
        "rows": rows,
    }
