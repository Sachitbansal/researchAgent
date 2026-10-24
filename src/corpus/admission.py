"""
Admission gate for arXiv search results: score each paper's title + abstract against the
search topic with the cross-encoder, before any PDF is downloaded.

In:  the natural-language topic, the arXiv results (title, summary, short id), a reranker
     and the admission threshold.
Out: {"admitted", "rejected", "scores", "status", "threshold"}. Never raises: a reranker
     that cannot load or run admits everything and says so in `status`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

GATE_APPLIED = "applied"
SKIPPED_PREFIX = "skipped: "

# Carried through Reranker.rerank, which returns its input re-sorted by score. Without a
# key the scores could not be mapped back to the arXiv result they belong to.
_INDEX_KEY = "_admission_index"


def admission_text(title: str, abstract: str) -> str:
    """The passage the gate scores. One definition, shared by runtime and tuning.

    eval/tune_admission.py calls this too, so the threshold is measured on exactly the
    text the gate sees at runtime rather than on a near copy that could drift.
    """
    return "\n\n".join(part.strip() for part in (title or "", abstract or "") if part.strip())


def score_texts(query: str, texts: Sequence[str], reranker: Any) -> List[float]:
    """Cross-encoder scores for (query, text) pairs, in input order. May raise.

    Goes through Reranker.rerank rather than the model directly, so pairs are truncated
    by the same `_pair` budget as retrieval and the tokenizer limits hold.
    """
    candidates = [{"text": text, _INDEX_KEY: index} for index, text in enumerate(texts)]
    ranked = reranker.rerank(query, candidates)
    scores: List[Optional[float]] = [None] * len(texts)
    for item in ranked:
        scores[int(item[_INDEX_KEY])] = float(item["rerank_score"])
    if any(score is None or math.isnan(score) for score in scores):
        raise ValueError("the cross-encoder did not return a usable score for every paper")
    return [float(score) for score in scores]  # type: ignore[arg-type]


def _describe(result: Any) -> Tuple[str, str, str]:
    """(arxiv_id, title, abstract) from an arxiv.Result-like object."""
    arxiv_id = str(result.get_short_id()) if hasattr(result, "get_short_id") else ""
    return arxiv_id, str(getattr(result, "title", "") or ""), str(getattr(result, "summary", "") or "")


def gate_results(
    query: str,
    results: Sequence[Any],
    reranker: Any,
    threshold: float,
) -> Dict[str, Any]:
    """Split `results` into admitted and rejected by cross-encoder score.

    A paper below `threshold` is not downloaded, not indexed and not tagged. When the
    reranker is missing or fails, every paper is admitted and `status` says why: the
    gate is a filter on a search the agent already chose to run, and losing it should
    degrade precision, not turn the search into a failure.
    """
    results = list(results)
    described = [_describe(result) for result in results]

    if reranker is None:
        return _skipped(results, threshold, "no cross-encoder is available")
    try:
        scores = score_texts(query, [admission_text(t, a) for _, t, a in described], reranker)
    except Exception as exc:  # RerankError, a CUDA fault, a bad checkpoint: all the same here
        return _skipped(results, threshold, f"{type(exc).__name__}: {exc}")

    admitted: List[Any] = []
    rejected: List[Dict[str, Any]] = []
    by_id: Dict[str, float] = {}
    for result, (arxiv_id, title, _), score in zip(results, described, scores):
        by_id[arxiv_id] = round(score, 4)
        if score >= threshold:
            admitted.append(result)
        else:
            rejected.append({"arxiv_id": arxiv_id, "title": title, "score": round(score, 4)})
    return {
        "admitted": admitted,
        "rejected": rejected,
        "scores": by_id,
        "status": GATE_APPLIED,
        "threshold": float(threshold),
    }


def _skipped(results: List[Any], threshold: float, reason: str) -> Dict[str, Any]:
    return {
        "admitted": list(results),
        "rejected": [],
        "scores": {},
        "status": SKIPPED_PREFIX + reason,
        "threshold": float(threshold),
    }
