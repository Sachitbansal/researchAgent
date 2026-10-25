"""
The first-stage text of a chunk: "{paper_title} — {section}\\n{text}", and its hash.

In:  a chunk record, the config (retrieval.contextual_headers) and a TokenCounter.
Out: the exact string the bi-encoder embeds and BM25 indexes, plus `embed_hash`, the sha256
     of that string plus its format. The chunk record's text and content_hash never change.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Tuple

from common.tokenization import TokenCounter
from config import CFG, Config

# Bump when the header layout below changes. It is part of every embed_hash, so a bump
# re-embeds the whole index on the next index_chunks run instead of keeping old vectors.
HEADER_FORMAT_VERSION = 1
SEPARATOR = " — "
# [CLS] and [SEP]: what the bi-encoder adds around a passage (docs/ARCHITECTURE.md §5).
BI_ENCODER_SPECIALS = 2


def headers_enabled(config: Config = CFG) -> bool:
    return bool(config.retrieval.get("contextual_headers", False))


def embed_format(config: Config = CFG) -> str:
    """Names what the stored vectors were built from. Recorded in the manifest."""
    return f"header-v{HEADER_FORMAT_VERSION}" if headers_enabled(config) else "plain-v1"


def bi_encoder_budget(config: Config = CFG) -> int:
    """Tokens a passage may use in the bi-encoder: its window minus the two specials."""
    return int(config.embedding.max_seq_tokens) - BI_ENCODER_SPECIALS


def header_for(chunk: Dict[str, Any]) -> str:
    """'{paper_title} — {section}', or the title alone when the section is missing or
    only repeats the title (a paper's first chunk is often "sectioned" by its title)."""
    title = " ".join(str(chunk.get("paper_title") or "").split())
    section = " ".join(str(chunk.get("section") or "").split())
    if section and section.lower() != title.lower():
        return f"{title}{SEPARATOR}{section}" if title else section
    return title


def embed_text(
    chunk: Dict[str, Any], config: Config = CFG, counter: Optional[TokenCounter] = None
) -> Tuple[str, bool]:
    """(the text the first stage sees, whether the header had to be shortened).

    Only the HEADER is ever cut, never the chunk: the chunk's tokens are what make it
    findable, and the bi-encoder silently drops whatever runs past its window. The
    budget counts the passage prefix too, which the embedder prepends at encode time.
    """
    text = str(chunk.get("text") or "")
    if not headers_enabled(config):
        return text, False
    header = header_for(chunk)
    if not header:
        return text, False

    counter = counter if counter is not None else TokenCounter(config)
    prefix = str(config.embedding.passage_prefix or "")
    budget = bi_encoder_budget(config)
    full = f"{header}\n{text}"
    if counter.count(f"{prefix}{full}") <= budget:
        return full, False

    room = budget - counter.count(f"{prefix}{text}")
    while room > 0:
        short = counter.truncate(header, room)
        candidate = f"{short}\n{text}"
        if short and counter.count(f"{prefix}{candidate}") <= budget:
            return candidate, True
        room -= 1
    return text, True


def embed_hash(text: str, config: Config = CFG) -> str:
    """sha256 of the exact embedder input (passage prefix + `text`), its format and model.

    The embedding cache key, and what the manifest's `indexed_hashes` records. A header
    change moves it while the chunk's content_hash stays put — which is the point. The
    model is in it so a cached vector can never be served to a different model.
    """
    prefix = str(config.embedding.passage_prefix or "")
    payload = f"{config.embedding.model}\n{embed_format(config)}\n{prefix}{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
