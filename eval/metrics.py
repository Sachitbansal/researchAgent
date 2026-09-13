"""
Scoring functions for the eval: retrieval recall, MRR, fact coverage, abstention.

In:  what the system retrieved or answered, and the gold labels for that question.
Out: plain numbers. No model calls except the bi-encoder used for fact coverage, so a
     run is reproducible and a tuning sweep is not at the mercy of sampling.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold chunks appearing in the first k retrieved.

    Undefined with no gold chunks, which is the abstention case; those questions are
    scored by abstention_correct instead, never by recall.
    """
    gold_set = set(gold)
    if not gold_set:
        return float("nan")
    return len(set(retrieved[:k]) & gold_set) / len(gold_set)


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """1/rank of the first gold chunk, or 0.0 if none was retrieved."""
    gold_set = set(gold)
    if not gold_set:
        return float("nan")
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold_set:
            return 1.0 / position
    return 0.0


def any_gold_paper_hit(retrieved_papers: Sequence[str], gold_papers: Sequence[str]) -> bool:
    """Whether the right paper was reached, even if not the exact chunk.

    A softer signal than chunk recall: the gold chunk set is hand-labelled and cannot be
    exhaustive, so a neighbouring chunk from the right paper is often a real hit that
    chunk-level recall scores as a miss.
    """
    return bool(set(retrieved_papers) & set(gold_papers))


def abstention_correct(answer: str, sufficient_evidence: bool, expect_abstention: bool) -> bool:
    """Did the system abstain exactly when it should have?

    Abstention is read from the retrieval gate and from the answer text together: the
    gate firing is the mechanism, but the failure being tested is the model answering
    anyway from parametric knowledge.
    """
    stated = looks_like_abstention(answer)
    if expect_abstention:
        return stated or not sufficient_evidence
    return not stated


_ABSTAIN_MARKERS = (
    "cannot answer", "can't answer", "no evidence", "not in the corpus",
    "does not contain", "do not contain", "doesn't contain", "no information",
    "not covered", "unable to answer", "insufficient evidence", "nothing in the",
    "not present in", "no relevant", "does not appear",
)


def looks_like_abstention(answer: str) -> bool:
    """Whether an answer says it cannot answer, rather than answering."""
    lowered = " ".join((answer or "").lower().split())
    return any(marker in lowered for marker in _ABSTAIN_MARKERS)


def fact_coverage(
    answer: str,
    expected_facts: Sequence[str],
    embedder: Any,
    threshold: float = 0.62,
) -> Dict[str, Any]:
    """How many expected facts the answer actually states.

    Scored by embedding similarity between each expected fact and the best-matching
    sentence of the answer, not by substring matching: the facts are written as
    paraphrases, so a literal match would score correct answers as wrong. Deterministic,
    which is what makes a threshold sweep comparable across runs.
    """
    facts = [f for f in expected_facts if f.strip()]
    if not facts:
        return {"covered": 0, "total": 0, "ratio": float("nan"), "per_fact": []}

    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer or "") if len(s.split()) >= 3]
    if not sentences:
        return {"covered": 0, "total": len(facts), "ratio": 0.0,
                "per_fact": [{"fact": f, "similarity": 0.0, "covered": False} for f in facts]}

    fact_vectors = np.asarray(embedder.encode_passages(list(facts)))
    sentence_vectors = np.asarray(embedder.encode_passages(sentences))
    similarity = fact_vectors @ sentence_vectors.T

    per_fact = []
    for index, fact in enumerate(facts):
        best = float(similarity[index].max())
        per_fact.append({"fact": fact, "similarity": round(best, 4),
                         "covered": best >= threshold})
    covered = sum(1 for entry in per_fact if entry["covered"])
    return {"covered": covered, "total": len(facts),
            "ratio": round(covered / len(facts), 4), "per_fact": per_fact}


def tool_trace_match(used: Sequence[str], expected: Sequence[str]) -> Dict[str, Any]:
    """Compare the tools actually called against the ones the design anticipated.

    A soft check by design. The agent is allowed to reach the right answer another way;
    this records where it diverged so unexpected traces get looked at rather than
    silently passing.
    """
    used_set, expected_set = set(used), set(expected)
    return {
        "expected": sorted(expected_set),
        "used": list(used),
        "missing": sorted(expected_set - used_set),
        "unexpected": sorted(used_set - expected_set),
        "matched": expected_set.issubset(used_set),
    }


def mean(values: Sequence[float]) -> float:
    """Mean over the values that are defined, ignoring NaN."""
    usable = [v for v in values if v == v]
    return round(float(np.mean(usable)), 4) if usable else float("nan")
