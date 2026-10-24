"""
Derives a run's fetch proposal from its tool-call trace: the CRAG "propose, don't fetch".

In:  the run's trace records, in call order (agent.trace.TraceWriter.records).
Out: {"query", "searched_for", "best_score", "closest_topics"} when the run ended on an
     uncovered gap the corpus cannot fill, else None. The user approves the fetch; the
     loop never runs search_literature on its own (docs/DECISIONS.md).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

RETRIEVE_TOOL = "retrieve_evidence"
SEARCH_TOOL = "search_literature"


def _coverage_gap(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    summary = record.get("result_summary")
    gap = summary.get("coverage_gap") if isinstance(summary, dict) else None
    return gap if isinstance(gap, dict) else None


def derive_fetch_proposal(records: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The proposal for this run, or None.

    Read from the LAST retrieve_evidence call only, so a later retrieval that passed the
    gate clears an earlier gap. It is a proposal only if that call's hint said
    "propose_fetch" and search_literature was not called after it: a fetch already made
    has nothing left to propose.
    """
    last_retrieve = None
    for position, record in enumerate(records):
        if record.get("tool_name") == RETRIEVE_TOOL:
            last_retrieve = position
    if last_retrieve is None:
        return None

    gap = _coverage_gap(records[last_retrieve])
    if gap is None or gap.get("suggested_next") != "propose_fetch":
        return None
    if any(r.get("tool_name") == SEARCH_TOOL for r in records[last_retrieve + 1:]):
        return None
    return {
        "query": gap.get("proposed_query") or gap.get("searched_for"),
        "searched_for": gap.get("searched_for"),
        "best_score": gap.get("best_score"),
        "closest_topics": list(gap.get("closest_topics") or []),
    }
