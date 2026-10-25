"""
The CRAG fetch round-trip: derive a run's fetch proposal, and build the user's approval.

In:  the run's trace records, in call order (agent.trace.TraceWriter.records); or, for an
     approval, the query the user approved plus the config (prompt files, length cap).
Out: {"query", "searched_for", "best_score", "closest_topics"} or None; and the approval
     as {"query", "question", "display"} or an error dict. The loop never runs
     search_literature on its own: the user approves (docs/DECISIONS.md D14, D30-D33).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from config import CFG, Config

RETRIEVE_TOOL = "retrieve_evidence"
SEARCH_TOOL = "search_literature"
# Filled into the approval prompt files. str.replace, not str.format: a query may carry
# braces, and a prompt file may one day carry an example in braces.
QUERY_PLACEHOLDER = "{query}"


def _coverage_gap(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    summary = record.get("result_summary")
    gap = summary.get("coverage_gap") if isinstance(summary, dict) else None
    return gap if isinstance(gap, dict) else None


def is_filtered(record: Dict[str, Any]) -> bool:
    """Whether this retrieve_evidence call was narrowed by topic_filter or chunk_types.

    Read from the call's arguments, not its gap: a passing call has no gap, and it has to
    be recognised too. The same truthiness as the retriever's own filter, so an empty
    string or an empty list is no filter.
    """
    args = record.get("args")
    if not isinstance(args, dict):
        return False
    return bool(args.get("topic_filter") or args.get("chunk_types"))


def _empty_index(record: Dict[str, Any]) -> bool:
    summary = record.get("result_summary")
    return isinstance(summary, dict) and summary.get("error") == "empty_index"


def derive_fetch_proposal(records: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The proposal for this run, or None.

    Read from the LAST UNFILTERED retrieve_evidence call only (D30). A filtered call is
    skipped entirely, failing or passing: a figure-only search that finds no figure says
    nothing about whether the corpus covers the subject. So a later unfiltered retrieval
    that passed clears an earlier gap, and a later filtered one changes nothing.

    It is a proposal only if that call's hint said "propose_fetch", or the index was
    empty (a cold start has no scorer and so no hint, but plainly lacks the subject), and
    search_literature was not called after it: a fetch already made has nothing left to
    propose.
    """
    last = None
    for position, record in enumerate(records):
        if record.get("tool_name") == RETRIEVE_TOOL and not is_filtered(record):
            last = position
    if last is None:
        return None
    if any(r.get("tool_name") == SEARCH_TOOL for r in records[last + 1:]):
        return None

    record = records[last]
    if _empty_index(record):
        query = str((record.get("args") or {}).get("query") or "").strip()
        if not query:
            return None
        return {"query": query, "searched_for": query, "best_score": None,
                "closest_topics": []}

    gap = _coverage_gap(record)
    if gap is None or gap.get("suggested_next") != "propose_fetch" or gap.get("filtered"):
        return None
    return {
        "query": gap.get("proposed_query") or gap.get("searched_for"),
        "searched_for": gap.get("searched_for"),
        "best_score": gap.get("best_score"),
        "closest_topics": list(gap.get("closest_topics") or []),
    }


def approval_request(query: Any, config: Config = CFG) -> Dict[str, Any]:
    """The run that carries out an approved fetch: {"query", "question", "display"}.

    `question` is what the agent is sent, from src/prompts/fetch_approval.md, so the
    wording lives in one prompt file and not in the browser or the CLI (D32). `display`
    is the short form stored as the turn's question (fetch_approval_title.md); it is
    replayed to the model as history in later turns, so it is a prompt file too.
    Never raises: a missing, empty or over-long query is an error dict.
    """
    if not isinstance(query, str):
        return {"error": "bad_fetch_query", "detail": "the approved query must be a string"}
    flat = " ".join(query.split())
    if not flat:
        return {"error": "empty_fetch_query", "detail": "the approved query is empty"}
    limit = int(config.agent.fetch_query_max_chars)
    if len(flat) > limit:
        return {"error": "fetch_query_too_long",
                "detail": f"the approved query is {len(flat)} characters; the limit is {limit}"}
    try:
        question = config.prompt("fetch_approval").strip().replace(QUERY_PLACEHOLDER, flat)
        display = config.prompt("fetch_approval_title").strip().replace(QUERY_PLACEHOLDER, flat)
    except Exception as exc:  # a missing prompt file must come back as data, not raise
        return {"error": "fetch_prompt_missing", "detail": str(exc)}
    return {"query": flat, "question": question, "display": display}
