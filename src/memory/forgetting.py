"""
Automatic forgetting for long-term memory (docs/DECISIONS.md D20): pure functions, no I/O.

In:  memory rows (use_count, last_used_at, kind, content_hashes), the current time, the
     current chunk_id -> content_hash map, and the `memory` config numbers.
Out: each memory's strength, the ids decay or the cap evicts, and the ids whose evidence
     changed under them (stale). The store applies the verdicts; nothing here writes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

DAY_S = 86_400.0


def use_weight(use_count: int) -> float:
    """1 + ln(uses): a memory found once weighs 1, and each further use counts for less.

    Logarithmic so a finding used fifty times outlives one used once by months, not by
    years. Being found counts as the first use, so use_count is never below 1.
    """
    return 1.0 + math.log(max(1, int(use_count)))


def recency(last_used_at: float, now: float, half_life_days: float) -> float:
    """0.5 ** (days since last use / half-life): 1.0 when just used, 0.5 one half-life on."""
    days = max(0.0, float(now) - float(last_used_at)) / DAY_S
    return 0.5 ** (days / float(half_life_days))


def strength(memory: Mapping[str, Any], now: float, half_life_days: float) -> float:
    """use_weight x recency. What decay and the cap rank memories by."""
    last = memory.get("last_used_at")
    return use_weight(int(memory.get("use_count") or 1)) * recency(
        float(now if last is None else last), now, half_life_days)


def decay_victims(memories: Sequence[Mapping[str, Any]], now: float, half_life_days: float,
                  min_strength: float, max_active: int) -> List[Dict[str, Any]]:
    """Findings to forget with reason 'decay': every one below min_strength, then the
    weakest of the rest while more than max_active remain. Weakest first.

    Preferences are exempt. They are never cited, so the use rule would decay every one
    of them however much it still applies; a preference goes only when a newer one
    contradicts or restates it, or by hand.
    """
    findings = [m for m in memories if m.get("kind", "finding") == "finding"]
    scored = sorted(((strength(m, now, half_life_days), m) for m in findings),
                    key=lambda pair: (pair[0], float(pair[1].get("last_used_at") or 0.0)))
    victims = [{"id": m["id"], "strength": s, "why": "below_min_strength"}
               for s, m in scored if s < min_strength]
    survivors = [(s, m) for s, m in scored if s >= min_strength]
    excess = len(survivors) - int(max_active)
    victims += [{"id": m["id"], "strength": s, "why": "over_cap"}
                for s, m in survivors[:max(0, excess)]]
    return victims


def is_stale(memory: Mapping[str, Any], current: Mapping[str, str]) -> bool:
    """True when any supporting chunk is gone, or its text is no longer what was checked.

    Compared by content_hash, never by chunk_id alone: a chunk_id is positional and
    survives a re-ingest that changes the text under it (.claude/CLAUDE.md). A memory with
    no stored hash for one of its chunks cannot be re-checked, so it counts as stale too.
    """
    hashes = memory.get("content_hashes") or {}
    for chunk_id in memory.get("chunk_ids") or []:
        stored = hashes.get(chunk_id)
        if stored is None or current.get(chunk_id) != stored:
            return True
    return False


def stale_ids(memories: Sequence[Mapping[str, Any]], current: Mapping[str, str]) -> List[str]:
    """Ids of findings whose evidence changed. Preferences carry no chunks: never stale."""
    return [m["id"] for m in memories
            if m.get("kind", "finding") == "finding" and is_stale(m, current)]


def min_strength_for(days: float, half_life_days: float, use_count: int = 1) -> float:
    """The min_strength at which a memory with `use_count` uses goes after `days` unused.

    How config's min_strength is chosen: used once and untouched for ~90 days.
    """
    return use_weight(use_count) * 0.5 ** (float(days) / float(half_life_days))
