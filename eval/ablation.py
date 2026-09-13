"""
Rerank ablation: the same eval set with the cross-encoder on and off.

In:  the tuning split and the tuned threshold from config.
Out: quality and latency for both configurations, reported as measured. A marginal gain
     at this corpus size is a finding, not something to hide.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.retrieval_eval import config_with, evaluate_retrieval, load_questions  # noqa: E402


def run(held_out: bool = False) -> Dict[str, Any]:
    """Evaluate retrieval with rerank enabled and disabled, sharing one model instance."""
    from retrieval.embedder import Embedder
    from retrieval.reranker import Reranker

    questions = load_questions(held_out=held_out)
    base = config_with()
    embedder = Embedder(base)
    reranker = Reranker(base)

    out: Dict[str, Any] = {"n_questions": len(questions), "arms": {}}
    for label, enabled in (("rerank_on", True), ("rerank_off", False)):
        # With rerank off the bi-encoder's cosine is what the gate sees, and cosine and
        # cross-encoder logits are not on the same scale, so the tuned threshold does not
        # transfer. The arm is run ungated to keep the comparison about ranking quality.
        config = config_with(
            rerank_enabled=enabled,
            relevance_threshold=base.retrieval.relevance_threshold if enabled else -1e9,
        )
        result = evaluate_retrieval(questions, config, embedder=embedder, reranker=reranker)
        out["arms"][label] = {
            "recall": result["recall"],
            "mrr": result["mrr"],
            "gold_paper_hit_rate": result["gold_paper_hit_rate"],
            "median_latency_ms": result["median_latency_ms"],
        }
    return out


def main() -> int:
    result = run()
    on, off = result["arms"]["rerank_on"], result["arms"]["rerank_off"]
    print(f"rerank ablation over {result['n_questions']} tuning questions\n", flush=True)
    header = f"{'arm':<12}  {'recall@1':>8}  {'recall@3':>8}  {'recall@5':>8}  {'MRR':>6}  {'paper':>6}  {'latency':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for label, arm in (("rerank on", on), ("rerank off", off)):
        print(f"{label:<12}  {arm['recall']['1']:>8.3f}  {arm['recall']['3']:>8.3f}  "
              f"{arm['recall']['5']:>8.3f}  {arm['mrr']:>6.3f}  "
              f"{arm['gold_paper_hit_rate']:>6.3f}  {arm['median_latency_ms']:>7}ms", flush=True)

    delta_mrr = on["mrr"] - off["mrr"]
    delta_latency = on["median_latency_ms"] - off["median_latency_ms"]
    print(f"\nrerank changes MRR by {delta_mrr:+.3f} and median latency by "
          f"{delta_latency:+d}ms", flush=True)

    out = ROOT / "eval" / "results" / "ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"written: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
