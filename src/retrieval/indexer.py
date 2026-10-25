"""
Embeds any chunk not yet in the index and appends it: the M3 incremental path.

In:  chunks.jsonl and the existing index; nothing else is required.
Out: {"chunks_indexed", "chunks_skipped", "cache_hits", "total_indexed", ...}. A changed
     model, or a changed embedder input (text, header or format: the embed hash), rebuilds.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from corpus.chunk_store import ChunkStore
from config import CFG, Config
from common.tokenization import TokenCounter
from retrieval.contextual import embed_hash, embed_text
from retrieval.embedder import Embedder, EmbeddingCache, EmbeddingError
from corpus.manifest import Manifest, ManifestError
from retrieval.vector_index import IndexError_, VectorIndex


def model_changed(manifest: Manifest, config: Config = CFG) -> bool:
    """True when the manifest was built with a different embedding model or dimension.

    Appending vectors from a different model to an existing index silently corrupts
    retrieval — the vectors are not comparable — so this forces a rebuild rather than
    warning about it.
    """
    stored_model = manifest.data.get("embedding_model")
    stored_dim = manifest.data.get("embedding_dim")
    if not manifest.data.get("faiss_id_map"):
        return False
    return stored_model != config.embedding.model or int(stored_dim or 0) != int(config.embedding.dim)


def index_chunks(
    config: Config = CFG,
    embedder: Optional[Embedder] = None,
    rebuild: bool = False,
) -> Dict[str, Any]:
    """Embed and index every chunk that is not already in the index.

    Returns an error dict rather than raising, so a missing model or a corrupt index
    degrades the caller instead of killing it.
    """
    try:
        manifest = Manifest.load(config, strict_model_check=False)
    except ManifestError as exc:
        return {"error": "manifest_unreadable", "detail": str(exc), "partial": None}

    store = ChunkStore(config=config)
    chunks = store.all()
    if not chunks:
        return {"error": "no_chunks", "detail": "chunks.jsonl is empty; run M2 ingest first",
                "partial": None}

    rebuild_reason = ""
    if model_changed(manifest, config):
        rebuild_reason = (
            f"embedding model changed from {manifest.data.get('embedding_model')} "
            f"to {config.embedding.model}"
        )
    force_rebuild = rebuild or bool(rebuild_reason)

    try:
        existing = VectorIndex.load(manifest, config)
    except IndexError_ as exc:
        return {"error": "index_unreadable", "detail": str(exc), "partial": None}

    # What the bi-encoder actually embeds: the chunk text, plus a "title — section" header
    # when retrieval.contextual_headers is on. The chunk record's text and content_hash
    # never change, so content_hash cannot say whether a vector is current — the embed
    # hash (sha256 of the exact embedder input plus its format) can.
    counter = TokenCounter(config)
    inputs: Dict[str, str] = {}
    hashes: Dict[str, str] = {}
    headers_truncated = 0
    for chunk in chunks:
        text, truncated = embed_text(chunk, config, counter)
        inputs[chunk["chunk_id"]] = text
        hashes[chunk["chunk_id"]] = embed_hash(text, config)
        headers_truncated += int(truncated)

    # A chunk_id is positional within its paper, so it survives a re-ingest even when the
    # text beneath it changes — figure chunks gaining a description is exactly that.
    # Skipping on id alone therefore leaves a stale vector in place, and retrieval quietly
    # keeps matching the old text. Staleness is checked on the embed hash instead, which
    # also moves when only the header or its format does.
    stale = [c for c in chunks if existing.is_stale(c["chunk_id"], hashes[c["chunk_id"]])]
    if stale and not force_rebuild:
        force_rebuild = True
        rebuild_reason = (
            f"{len(stale)} chunk(s) changed embedder input under an unchanged chunk_id "
            f"(e.g. {stale[0]['chunk_id']}); a flat index cannot replace a vector in "
            "place without renumbering every position after it, so it is rebuilt"
        )

    index = VectorIndex(int(config.embedding.dim), config) if force_rebuild else existing

    already = set() if force_rebuild else index.indexed
    pending = [chunk for chunk in chunks if chunk["chunk_id"] not in already]
    if not pending:
        return {
            "chunks_indexed": 0,
            "chunks_skipped": len(chunks),
            "cache_hits": 0,
            "stale_detected": 0,
            "total_indexed": len(index),
            "rebuilt": False,
            "headers_truncated": headers_truncated,
        }

    embedder = embedder if embedder is not None else Embedder(config)
    cache = EmbeddingCache(config)
    keys = [hashes[chunk["chunk_id"]] for chunk in pending]
    hits_before = sum(1 for key in keys if cache.get(key) is not None)

    try:
        vectors = embedder.encode_passages(
            [inputs[chunk["chunk_id"]] for chunk in pending], cache=cache, keys=keys
        )
    except EmbeddingError as exc:
        return {"error": "embedding_failed", "detail": str(exc), "partial": None}

    try:
        index.add([chunk["chunk_id"] for chunk in pending], vectors, keys)
        index.save(manifest)
    except IndexError_ as exc:
        return {"error": "index_write_failed", "detail": str(exc), "partial": None}

    cache.save()
    return {
        "chunks_indexed": len(pending),
        "chunks_skipped": len(chunks) - len(pending),
        "cache_hits": hits_before,
        "stale_detected": len(stale),
        "total_indexed": len(index),
        "rebuilt": force_rebuild,
        "rebuild_reason": rebuild_reason,
        "headers_truncated": headers_truncated,
    }


def search(
    query: str,
    k: int = 10,
    config: Config = CFG,
    embedder: Optional[Embedder] = None,
) -> List[Dict[str, Any]]:
    """Raw bi-encoder similarity search, returning resolved chunk records.

    This is stage one only. The cross-encoder rerank and the relevance gate are M4's
    job; nothing here decides whether the evidence is good enough.
    """
    manifest = Manifest.load(config, strict_model_check=False)
    index = VectorIndex.load(manifest, config)
    if len(index) == 0:
        return []

    embedder = embedder if embedder is not None else Embedder(config)
    hits = index.search(embedder.encode_query(query), k)

    by_id = {chunk["chunk_id"]: chunk for chunk in ChunkStore(config=config)}
    out: List[Dict[str, Any]] = []
    for chunk_id, score in hits:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            continue  # the map points at a chunk that no longer exists
        out.append({**chunk, "score": score})
    return out
