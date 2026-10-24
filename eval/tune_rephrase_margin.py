"""
Measures how far below the relevance gate failed retrievals score, to set rephrase_margin.

In:  the tuning split of eval/questions.json, run through retrieve_evidence offline with
     each scorer; optionally a logs/traces directory of past agent runs (read only).
Out: eval/results/rephrase_margin.json — per failure, the distance below the threshold
     and what it was (answerable vs absent; in traces, what the agent did next).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.retrieval_eval import config_with, load_questions  # noqa: E402

_NOTE = re.compile(r"best relevance score (-?[\d.]+) is below the threshold (-?[\d.]+)")


def probe(scorer: str) -> List[Dict[str, Any]]:
    """Top score and gate outcome for each tuning question, with one scorer."""
    from retrieval.embedder import Embedder
    from retrieval.reranker import Reranker
    from tools.retrieve_evidence import EvidenceRetriever

    cross = scorer == "cross_encoder"
    config = config_with(rerank_enabled=cross)
    retriever = EvidenceRetriever(config, embedder=Embedder(config),
                                  reranker=Reranker(config) if cross else None)
    threshold = float(config.retrieval.relevance_threshold if cross
                      else config.retrieval.bi_encoder_relevance_threshold)
    rows = []
    for question in load_questions(held_out=False):
        retriever.reset()
        result = retriever.retrieve(question["question"], k=5)
        gap = result.get("coverage_gap")
        best = gap["best_score"] if gap else (result["chunks"][0]["score"] if result.get("chunks") else None)
        rows.append({
            "question_id": question["question_id"],
            "category": question["category"],
            "answerable": not question["expect_abstention"],
            "best_score": best,
            "passed_gate": bool(result.get("sufficient_evidence")),
            "distance_below": round(threshold - best, 4) if gap and best is not None else None,
        })
    return rows


def from_traces(traces_dir: Path, exclude_question_ids: List[str]) -> List[Dict[str, Any]]:
    """Every gated-out retrieve_evidence call in past traces, with what happened next.

    Read only. Runs tagged with a held-out question id are skipped so the held-out third
    stays unseen. The label (was the gap real?) is a human judgement made from `query`
    and `then`, recorded in the config comment rather than guessed here.
    """
    failures = []
    for path in sorted(traces_dir.glob("*.jsonl")):
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if any(r.get("question_id") in exclude_question_ids for r in records):
            continue
        for position, record in enumerate(records):
            if record.get("tool_name") != "retrieve_evidence":
                continue
            match = _NOTE.search(str((record.get("result_summary") or {}).get("note", "")))
            if not match:
                continue
            best, threshold = float(match.group(1)), float(match.group(2))
            later = records[position + 1:]
            then: Optional[str] = None
            for nxt in later:
                if nxt.get("tool_name") == "search_literature":
                    then = "search_literature"
                    break
                if nxt.get("tool_name") == "retrieve_evidence":
                    summary = nxt.get("result_summary") or {}
                    then = f"retrieve {'passed' if summary.get('chunks') else 'failed'}: " \
                           f"{(nxt.get('args') or {}).get('query', '')[:80]}"
                    break
            failures.append({
                "run_id": record.get("run_id"),
                "question_id": record.get("question_id"),
                "query": (record.get("args") or {}).get("query"),
                "best_score": best,
                "threshold": threshold,
                "distance_below": round(threshold - best, 4),
                "then": then,
            })
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--traces", type=Path, default=None,
                        help="a logs/traces directory to read (never written)")
    args = parser.parse_args()

    held_out = [q["question_id"] for q in load_questions(held_out=True)]
    result: Dict[str, Any] = {
        "cross_encoder": probe("cross_encoder"),
        "bi_encoder": probe("bi_encoder"),
    }
    if args.traces is not None:
        result["trace_failures"] = from_traces(args.traces, held_out)

    for scorer in ("cross_encoder", "bi_encoder"):
        print(f"\n{scorer}: tuning-split gate failures")
        for row in result[scorer]:
            if not row["passed_gate"]:
                print(f"  {row['question_id']:<5} {row['category']:<17} best={row['best_score']:>8} "
                      f"below by {row['distance_below']}")
    for row in result.get("trace_failures", []):
        print(f"  trace {row['run_id']} {str(row['question_id']):<5} below by "
              f"{row['distance_below']:>6.3f}  {row['query'][:60]!r} -> {row['then']}")

    out = ROOT / "eval" / "results" / "rephrase_margin.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
