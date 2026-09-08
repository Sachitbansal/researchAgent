"""
The M4 verification gate from docs/BUILD_PLAN.md: retrieve_evidence, end to end.

In:  an index built by M3.
Out: pass/fail per check across three queries — clearly answerable, clearly absent, and
     borderline — plus rerank latency and proof the cross-encoder saw only the shortlist.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import CFG  # noqa: E402
from corpus.chunk_store import ChunkStore  # noqa: E402
from retrieval.reranker import Reranker  # noqa: E402
from tools.retrieve_evidence import EvidenceRetriever  # noqa: E402

ANSWERABLE = "how does the gating network route tokens to experts"
ABSENT = "what is the optimal fermentation temperature for sourdough starter culture"
BORDERLINE = "how do convolutional layers affect inference latency"

failures: List[str] = []
_started = time.monotonic()


def _elapsed() -> str:
    return f"{time.monotonic() - _started:6.1f}s"


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{_elapsed()}] [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""),
          flush=True)
    if not ok:
        failures.append(label)


def show(title: str, result: Dict[str, Any]) -> None:
    print(f"\n[{_elapsed()}] ---- {title} ----", flush=True)
    print(f"  sufficient_evidence={result.get('sufficient_evidence')} "
          f"candidates={result.get('n_candidates_considered')} "
          f"rerank_ms={result.get('rerank_ms')}", flush=True)
    if result.get("note"):
        print(f"  note: {result['note']}", flush=True)
    for chunk in result.get("chunks", []):
        print(f"    {chunk['score']:8.3f}  {chunk['chunk_id']}  p{chunk['page']}  "
              f"[{chunk['section']}]", flush=True)
        print(f"            {' '.join(chunk['text'].split())[:110]}...", flush=True)


def main() -> int:
    total_chunks = ChunkStore().count()
    if total_chunks == 0:
        print("No chunks indexed. Run scripts/m3_gate.py first.")
        return 1
    print(f"corpus: {total_chunks} chunks, threshold={CFG.retrieval.relevance_threshold}\n", flush=True)

    # Count rerank pairs so the shortlist claim is measured, not asserted.
    seen_pairs: List[int] = []
    reranker = Reranker(CFG)
    original_rerank = reranker.rerank

    def counting_rerank(query, candidates):
        seen_pairs.append(len(candidates))
        return original_rerank(query, candidates)

    reranker.rerank = counting_rerank  # type: ignore[method-assign]
    retriever = EvidenceRetriever(CFG, reranker=reranker)

    # 1. Clearly answerable.
    answerable = retriever.retrieve(ANSWERABLE, k=5)
    show("answerable query", answerable)
    check("answerable query returns evidence", answerable.get("sufficient_evidence") is True,
          f"{len(answerable.get('chunks', []))} chunks")
    check("results are ranked descending",
          all(a["score"] >= b["score"] for a, b in
              zip(answerable.get("chunks", []), answerable.get("chunks", [])[1:])))
    check("every chunk carries citable metadata",
          all(all(chunk.get(field) is not None for field in
                  ("chunk_id", "paper_id", "paper_title", "page", "chunk_type"))
              for chunk in answerable.get("chunks", [])))
    check("chunk text respects the character cap",
          all(len(c["text"]) <= CFG.retrieval.max_chunk_chars for c in answerable.get("chunks", [])))
    check("rerank latency was recorded", answerable.get("rerank_ms", 0) > 0,
          f"{answerable.get('rerank_ms')}ms for {seen_pairs[-1] if seen_pairs else 0} pairs")

    # 2. The cross-encoder must see the shortlist, never the corpus.
    check("cross-encoder ran on the shortlist, not the index",
          bool(seen_pairs) and max(seen_pairs) <= CFG.retrieval.k_retrieve < total_chunks,
          f"max pairs scored={max(seen_pairs) if seen_pairs else 0}, "
          f"k_retrieve={CFG.retrieval.k_retrieve}, corpus={total_chunks}")

    # 3. Clearly absent: the gate must fire rather than hand back weak chunks.
    retriever.reset()
    absent = retriever.retrieve(ABSENT, k=5)
    show("absent query", absent)
    check("absent query returns sufficient_evidence: false",
          absent.get("sufficient_evidence") is False)
    check("absent query returns no chunks", not absent.get("chunks"))
    check("absent query explains why", bool(absent.get("note")))

    # 4. Borderline: whatever it decides, it must decide coherently.
    retriever.reset()
    borderline = retriever.retrieve(BORDERLINE, k=5)
    show("borderline query", borderline)
    check("borderline query is internally consistent",
          bool(borderline.get("chunks")) == borderline.get("sufficient_evidence"),
          f"sufficient={borderline.get('sufficient_evidence')} "
          f"chunks={len(borderline.get('chunks', []))}")

    # 5. Cross-call dedup within one conversation.
    retriever.reset()
    first = retriever.retrieve(ANSWERABLE, k=3)
    second = retriever.retrieve(ANSWERABLE, k=3)
    first_ids = {c["chunk_id"] for c in first.get("chunks", [])}
    second_ids = {c["chunk_id"] for c in second.get("chunks", [])}
    check("a repeated query does not re-return the same chunks", not (first_ids & second_ids),
          f"first={len(first_ids)} second={len(second_ids)} overlap={len(first_ids & second_ids)}")
    retriever.reset()
    third = retriever.retrieve(ANSWERABLE, k=3)
    check("reset() clears the conversation's dedup state",
          {c["chunk_id"] for c in third.get("chunks", [])} == first_ids)

    # 6. Filters and bad input.
    retriever.reset()
    figures_only = retriever.retrieve(ANSWERABLE, k=5, chunk_types=["figure", "table"])
    check("chunk_types filter is honoured",
          all(c["chunk_type"] in ("figure", "table") for c in figures_only.get("chunks", [])),
          f"{len(figures_only.get('chunks', []))} figure/table chunks")
    retriever.reset()
    unknown_topic = retriever.retrieve(ANSWERABLE, topic_filter="no_such_topic")
    check("an unmatched topic_filter fails the gate rather than ignoring the filter",
          unknown_topic.get("sufficient_evidence") is False and not unknown_topic.get("chunks"))
    check("empty query returns an error dict",
          retriever.retrieve("").get("error") == "empty_query")
    check("bad chunk_types returns an error dict",
          retriever.retrieve("x", chunk_types=["nonsense"]).get("error") == "bad_chunk_types")

    print(f"\n{'M4 GATE PASSED' if not failures else 'M4 GATE FAILED: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
