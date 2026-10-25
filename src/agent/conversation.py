"""
The message array the loop re-sends on every call — the agent's working memory.

In:  a system prompt, optional prior-turn messages (a thread's history), an optional
     memory block with the chunk ids it shows (long-term memory, D19), the user question,
     assistant turns, and tool results.
Out: `messages()`, the full conversation in provider wire format. Enforces the pairing
     rules the API requires, and elides old tool content (and the memory block) when the
     array outgrows budget, reporting which chunk ids that took out of view.

The provider is stateless. /chat/completions remembers nothing between requests, so the
model "recalls" its own earlier tool call only because that call and its result are
sitting in the array we re-send. This class owns that array and nothing else does.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Sequence

from common.tokenization import TokenCounter
from config import CFG, Config
from llm_client import LLMResponse, Message


# What the memory block becomes once elided. Not a prompt to iterate on: a stand-in, like
# the tool-result placeholder below.
MEMORY_ELIDED = ("[earlier-chat memory elided: its {n} passage(s) are out of view and can be "
                 "retrieved again]")


class ConversationError(RuntimeError):
    """The message array is in a state the API would reject."""


def _chunk_ids(content: Any) -> List[str]:
    """chunk_ids in a serialised tool result, or [] when it is not one that carries chunks."""
    if not isinstance(content, str):
        return []
    try:
        payload = json.loads(content)
    except ValueError:
        return []
    chunks = payload.get("chunks") if isinstance(payload, dict) else None
    if not isinstance(chunks, list):
        return []
    return [str(c["chunk_id"]) for c in chunks if isinstance(c, dict) and c.get("chunk_id")]


def _serialise(result: Any) -> str:
    """Tool results are dicts; the wire format wants a string."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"error": "unserialisable_result", "detail": repr(result)[:500]})


class Conversation:
    """Builds and holds the messages array, guarding the rules that break silently.

    Three of them are structural and the API rejects the request if they are violated:
    every assistant tool_call needs exactly one tool message, the tool_call_id must match
    exactly, and the assistant turn must be appended before its results. The fourth —
    keeping the array inside a token budget — is ours to manage.
    """

    def __init__(self, system_prompt: str, question: str, config: Config = CFG,
                 history: Optional[Sequence[Message]] = None,
                 memory_block: str = "", memory_chunk_ids: Sequence[str] = ()) -> None:
        self.config = config
        self.counter = TokenCounter(config)
        # Earlier turns of a thread go between the system prompt and the question, as
        # plain user/assistant text (memory.history). Never tool messages: a replayed
        # tool_call would need its result replayed too, or the request is invalid.
        prior = [copy.deepcopy(dict(m)) for m in (history or [])]
        if any(m.get("role") not in ("user", "assistant") or m.get("tool_calls")
               for m in prior):
            raise ConversationError("history may hold only plain user/assistant messages")
        self._messages: List[Message] = [{"role": "system", "content": system_prompt}, *prior]
        # Long-term memory (D19): one user message after the history and before the
        # question, carrying earlier chats' findings with their chunks' text. It is
        # evidence in view like a tool result, so its chunk ids count as visible and it
        # is elided like one; but it is not a tool message, because no tool was called.
        self._evidence: Dict[int, List[str]] = {}
        if memory_block:
            self._evidence[len(self._messages)] = [str(c) for c in memory_chunk_ids]
            self._messages.append({"role": "user", "content": memory_block})
        self._messages.append({"role": "user", "content": question})
        # Where this question starts. final_text() looks only after it, or a run that
        # fails before answering would report the previous turn's answer as its own.
        self._question_index = len(self._messages) - 1
        self._pending: List[str] = []  # tool_call_ids awaiting a result
        self.elided = 0
        self.last_released: List[str] = []

    # ---- reading ---------------------------------------------------------------

    def messages(self) -> List[Message]:
        """The whole conversation, ready to send. Every call re-sends all of it."""
        if self._pending:
            raise ConversationError(
                f"tool_call_id(s) {self._pending} have no result yet; the API rejects an "
                "assistant tool_call that is not answered by a tool message"
            )
        return [copy.deepcopy(message) for message in self._messages]

    def __len__(self) -> int:
        return len(self._messages)

    def token_estimate(self) -> int:
        """Approximate size of the array, for the elision decision."""
        total = 0
        for message in self._messages:
            content = message.get("content")
            if isinstance(content, str):
                total += self.counter.count(content)
            for call in message.get("tool_calls") or []:
                total += self.counter.count(str((call.get("function") or {}).get("arguments", "")))
        return total

    def final_text(self) -> str:
        """The last assistant message's text for THIS question — the answer, once the loop
        stops. Prior-turn answers in the history are never returned."""
        for message in reversed(self._messages[self._question_index + 1:]):
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                return message["content"]
        return ""

    def visible_chunk_ids(self) -> List[str]:
        """chunk_ids whose text is in the array right now: un-elided tool results only.

        The retriever's "already returned" set must be a subset of this (docs/DECISIONS.md
        D13). Suppressing a chunk the model can no longer read leaves it with neither the
        text nor a way to get it back.
        """
        seen: List[str] = []
        for index, message in enumerate(self._messages):
            if index in self._evidence:
                ids = self._evidence[index]
            elif message.get("role") == "tool":
                ids = _chunk_ids(message.get("content"))
            else:
                continue
            for chunk_id in ids:
                if chunk_id not in seen:
                    seen.append(chunk_id)
        return seen

    # ---- writing ---------------------------------------------------------------

    def add_assistant(self, response: LLMResponse) -> Message:
        """Append the assistant turn exactly as the provider returned it.

        Verbatim, not reconstructed. The tool_call ids in this message are the only link
        to the results that follow, so re-deriving the message risks changing an id and
        silently breaking the pairing.
        """
        message = response.assistant_message()
        self._messages.append(message)
        self._pending = [call.id for call in response.tool_calls]
        return message

    def add_tool_result(self, tool_call_id: str, result: Any) -> Message:
        """Append one tool result, answering the tool_call with this exact id."""
        if tool_call_id not in self._pending:
            raise ConversationError(
                f"no unanswered tool_call with id {tool_call_id!r}; "
                f"awaiting {self._pending or 'nothing'}"
            )
        message: Message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": _serialise(result),
        }
        self._messages.append(message)
        self._pending.remove(tool_call_id)
        return message

    def add_user(self, content: Any) -> Message:
        """Append a user turn. Images ride here, never in a tool result."""
        message: Message = {"role": "user", "content": content}
        self._messages.append(message)
        return message

    # ---- context budget --------------------------------------------------------

    def elide_if_needed(self) -> int:
        """Shrink the array to fit the budget. Returns how many results were elided.

        Elides the *content* of the oldest tool results and leaves the messages in place.
        Deleting them instead would orphan the assistant tool_calls they answer, and the
        API rejects a request whose tool_calls are unanswered — so a deletion strategy
        fails outright rather than degrading.

        The placeholder still tells the agent that search already happened, so it does
        not simply repeat it.

        The chunk ids an elided result carried are left in `last_released` (this call's
        only): their text is gone from view, so the loop releases them from the
        retriever's dedup set and a later query may return them again (D13).
        """
        budget = int(self.config.agent.max_context_tokens)
        elided = 0
        released: List[str] = []
        for index, message in enumerate(self._messages):
            if self.token_estimate() <= budget:
                break
            if index in self._evidence:
                # The memory block, oldest of all. Elided whole; its chunks are released.
                if self._evidence[index]:
                    released.extend(c for c in self._evidence[index] if c not in released)
                    message["content"] = MEMORY_ELIDED.format(n=len(self._evidence[index]))
                    self._evidence[index] = []
                    elided += 1
                continue
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str) or content.startswith("[earlier result elided"):
                continue
            # Read the ids before the content is replaced: afterwards they are gone.
            released.extend(c for c in _chunk_ids(content) if c not in released)
            message["content"] = self._placeholder(content)
            elided += 1
        self.elided += elided
        # Only ids no other un-elided result still shows are out of view.
        still_visible = set(self.visible_chunk_ids()) if released else set()
        self.last_released = [c for c in released if c not in still_visible]
        return elided

    def _placeholder(self, content: str) -> str:
        """A short stand-in naming what was dropped, so the turn still carries meaning."""
        limit = int(self.config.agent.elision_placeholder_chars)
        try:
            payload = json.loads(content)
        except ValueError:
            return f"[earlier result elided: {len(content)} characters]"[:limit]

        chunks = payload.get("chunks") if isinstance(payload, dict) else None
        if isinstance(chunks, list) and chunks:
            papers = sorted({str(chunk.get("paper_id", "?")) for chunk in chunks})
            return (
                f"[earlier result elided: {len(chunks)} chunk(s) from {', '.join(papers)}. "
                "You already ran this search.]"
            )[:limit]
        return "[earlier result elided. You already ran this search.]"[:limit]
