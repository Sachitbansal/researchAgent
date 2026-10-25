"""
Rewrites a fetch proposal's query from the user's own question, in one LLM call (D28).

In:  a proposal from agent.fetch_proposal.derive_fetch_proposal, the turn's question, the
     previous turn's question in a thread (or None), an LLMClient (role `utility`), config.
Out: a copy of the proposal with `query` (the keyword phrase, or the agent's query when the
     rewrite fails or is junk), `agent_query`, `query_source` ("rewritten" | "agent"), and
     `rewrite_error` on a fallback. Never raises.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

from config import CFG, Config

log = logging.getLogger("sciagent.fetch_query")

PROMPT = "fetch_query"
# Filled in one pass, so a question that happens to contain the other placeholder cannot
# have it substituted a second time.
_PLACEHOLDERS = re.compile(r"\{(question|previous_question)\}")
# Wrappers a model puts around a bare phrase despite being told not to. Stripped, not
# rejected: the phrase inside is what was asked for.
_WRAPPERS = "\"'`“”‘’ "
# A reply that opens like a question is a question even without its question mark.
_QUESTION_START = re.compile(
    r"^(how|what|why|which|when|where|who|whom|whose|does|do|did|is|are|was|were|can|"
    r"could|should|would|will)\b", re.IGNORECASE)

SOURCE_REWRITTEN = "rewritten"
SOURCE_AGENT = "agent"


def build_prompt(question: str, previous_question: Optional[str], config: Config = CFG) -> str:
    """The rewrite prompt: src/prompts/fetch_query.md with both questions filled in.

    Raises ConfigError if the file is missing; rewrite_fetch_query turns that into a
    fallback. The earlier question is blank when there is none, as the prompt says.
    """
    values = {"question": " ".join((question or "").split()),
              "previous_question": " ".join((previous_question or "").split())}
    template = config.prompt(PROMPT)
    return _PLACEHOLDERS.sub(lambda m: values[m.group(1)], template)


def clean_phrase(text: Any, config: Config = CFG) -> Tuple[Optional[str], str]:
    """The model's reply as a usable phrase, or (None, why it is junk).

    Junk: empty; more than one line; shaped like a sentence or a question (a "?",
    closing sentence punctuation, an opening question word, or more than
    agent.fetch_query_max_words words); longer than agent.fetch_query_max_chars.
    """
    if not isinstance(text, str):
        return None, "reply was not text"
    stripped = text.strip().strip(_WRAPPERS).strip()
    if not stripped:
        return None, "empty reply"
    if "\n" in stripped:
        return None, "reply ran over more than one line"
    phrase = " ".join(stripped.split())
    limit = int(config.agent.fetch_query_max_chars)
    if len(phrase) > limit:
        return None, f"reply is {len(phrase)} characters; the limit is {limit}"
    if "?" in phrase or phrase[-1] in ".!" or _QUESTION_START.match(phrase):
        return None, "reply is a sentence or a question, not a keyword phrase"
    max_words = int(config.agent.fetch_query_max_words)
    if len(phrase.split()) > max_words:
        return None, f"reply has {len(phrase.split())} words; the limit is {max_words}"
    return phrase, ""


def rewrite_fetch_query(proposal: Dict[str, Any], question: str,
                        previous_question: Optional[str] = None,
                        client: Optional[Any] = None,
                        config: Config = CFG) -> Dict[str, Any]:
    """The proposal with its query rewritten from `question`, or kept with the reason.

    The agent's last unfiltered retrieve query drifts from what was asked, and the
    admission gate scores against whatever query is fetched, so a drifted query admits
    papers on the broader subject. One `utility` call, no retry: a failure costs nothing
    but the old query. `client` is anything with LLMClient.complete, so tests pass a stub.
    """
    result = dict(proposal)
    agent_query = proposal.get("query")
    result["agent_query"] = agent_query

    def fallback(reason: str) -> Dict[str, Any]:
        log.warning("fetch query rewrite fell back to the agent's query: %s", reason)
        result.update(query=agent_query, query_source=SOURCE_AGENT, rewrite_error=reason)
        return result

    try:
        prompt = build_prompt(question, previous_question, config)
        if client is None:
            from llm_client import LLMClient
            client = LLMClient(config=config)
        log.info("rewriting the fetch query from the question (agent query %r)", agent_query)
        response = client.complete([{"role": "user", "content": prompt}], role="utility",
                                   timeout=float(config.llm.planning_timeout_s),
                                   max_retries=1)
        phrase, why = clean_phrase(getattr(response, "text", None), config)
    except Exception as exc:  # LLMError, a missing prompt file, a broken stub: all fall back
        return fallback(f"{type(exc).__name__}: {getattr(exc, 'detail', None) or exc}")
    if phrase is None:
        return fallback(why)
    result.update(query=phrase, query_source=SOURCE_REWRITTEN)
    return result
