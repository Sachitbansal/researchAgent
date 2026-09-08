"""
Offline tests for M3: embedding cache, BGE prefixes, FAISS lockstep, incremental append.

In:  a tmp_path config and a stub embedder — no model download, no network.
Out: assertions that vectors and chunk_ids stay aligned, that the map survives a
     save/load cycle, and that a changed embedding model forces a rebuild.
"""

import numpy as np
import pytest

from config import load_config
from embedder import Embedder, EmbeddingCache
from indexer import index_chunks, model_changed
from manifest import Manifest
from records import chunk_record, content_hash, paper_record
from vector_index import IndexError_, VectorIndex


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    import config as config_module

    repo_root = config_module.REPO_ROOT
    local = tmp_path / "config.yaml"
    local.write_text((repo_root / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(config_module, "REPO_ROOT", tmp_path)
    built = load_config(local)
    built.paths.ensure()
    built.paths.prompts = repo_root / "src" / "prompts"
    return built


class StubEmbedder(Embedder):
    """Deterministic unit vectors, so tests never download a model."""

    def __init__(self, config):
        super().__init__(config)
        self.encoded = []

    def _encode(self, texts):
        self.encoded.extend(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row, hash(text) % self.dim] = 1.0
        return out


def seed_chunks(cfg, paper_id="p1", n=4, tag="alpha"):
    from chunk_store import ChunkStore

    manifest = Manifest.load(cfg)
    if not manifest.has_paper(paper_id):
        manifest.add_paper(paper_record(
            arxiv_id=paper_id, title="T", authors=[], abstract="", published="2024-01-01",
            year=2024, categories=[], pdf_url="", pdf_path="", topic_tags=[tag]))
        manifest.save()
    records = [
        chunk_record(paper_id=paper_id, paper_title="T", topic_tags=[tag], chunk_type="text",
                     text=f"{paper_id} chunk number {i} about routing", page=1, position=i,
                     n_tokens=8)
        for i in range(n)
    ]
    ChunkStore(config=cfg).append(records)
    return records


# ------------------------------------------------------------------- embedding cache

def test_cache_round_trips_and_rejects_wrong_dimensions(cfg):
    cache = EmbeddingCache(cfg)
    vector = np.ones(cfg.embedding.dim, dtype=np.float32)
    cache.put("abc", vector)
    cache.save()

    reloaded = EmbeddingCache(cfg)
    assert reloaded.get("abc") is not None
    assert reloaded.get("abc").shape == (cfg.embedding.dim,)
    assert reloaded.get("missing") is None

    import json
    payload = json.loads(cfg.paths.embedding_cache.read_text())
    payload["wrong"] = [1.0, 2.0]
    cfg.paths.embedding_cache.write_text(json.dumps(payload))
    assert EmbeddingCache(cfg).get("wrong") is None, "a wrong-width vector is a cache miss"


def test_passages_reuse_the_cache_across_papers(cfg):
    """Identical text in two papers is embedded once; both keep their own chunk record."""
    embedder = StubEmbedder(cfg)
    cache = EmbeddingCache(cfg)
    shared = "an identical boilerplate sentence"

    embedder.encode_passages([shared], cache=cache)
    calls_after_first = len(embedder.encoded)
    embedder.encode_passages([shared], cache=cache)

    assert len(embedder.encoded) == calls_after_first, "the second paper must hit the cache"
    assert cache.get(content_hash(shared)) is not None


# ------------------------------------------------------------------------ BGE prefixes

def test_query_gets_the_instruction_prefix_and_passages_do_not(cfg):
    """BGE is trained asymmetrically. Prefixing both sides, or neither, degrades
    retrieval, so the instruction belongs to the query-encode path alone."""
    embedder = StubEmbedder(cfg)
    embedder.encode_passages(["a passage about routing"])
    embedder.encode_query("a question about routing")

    passage_text, query_text = embedder.encoded[0], embedder.encoded[1]
    assert not passage_text.startswith(cfg.embedding.query_prefix)
    assert query_text.startswith(cfg.embedding.query_prefix)


# ------------------------------------------------------------------------ vector index

def test_add_refuses_a_mismatched_batch(cfg):
    index = VectorIndex(4, cfg)
    with pytest.raises(IndexError_, match="lockstep"):
        index.add(["a", "b"], np.zeros((1, 4), dtype=np.float32))


def test_search_ranks_descending_and_respects_a_filter(cfg):
    index = VectorIndex(4, cfg)
    index.add(["a", "b", "c"], np.eye(3, 4, dtype=np.float32))
    query = np.array([1, 0, 0, 0], dtype=np.float32)

    hits = index.search(query, 3)
    assert hits[0][0] == "a"
    assert all(x[1] >= y[1] for x, y in zip(hits, hits[1:]))
    assert all(chunk_id in {"b", "c"} for chunk_id, _ in index.search(query, 3, allowed={"b", "c"}))


def test_map_survives_a_save_and_load_cycle(cfg):
    """faiss_id_map exists nowhere else and is unrecoverable if lost."""
    manifest = Manifest.load(cfg)
    index = VectorIndex(4, cfg)
    index.add(["x0", "x1", "x2"], np.eye(3, 4, dtype=np.float32))
    index.save(manifest)

    reloaded = VectorIndex.load(Manifest.load(cfg), cfg)
    assert reloaded.chunk_ids == ["x0", "x1", "x2"]
    assert reloaded.index.ntotal == 3
    assert reloaded.vectors.shape == (3, 4)
    assert reloaded.search(np.array([0, 1, 0, 0], dtype=np.float32), 1)[0][0] == "x1"


def test_load_refuses_a_map_that_disagrees_with_the_index(cfg):
    """A shorter or longer map means every lookup past the divergence resolves to the
    wrong chunk — silently. Refusing to load is the only safe response."""
    manifest = Manifest.load(cfg)
    index = VectorIndex(4, cfg)
    index.add(["x0", "x1", "x2"], np.eye(3, 4, dtype=np.float32))
    index.save(manifest)

    broken = Manifest.load(cfg)
    broken.data["faiss_id_map"] = ["x0", "x1"]
    broken.save()

    with pytest.raises(IndexError_, match="faiss_id_map"):
        VectorIndex.load(Manifest.load(cfg), cfg)


def test_load_refuses_when_the_index_file_is_missing(cfg):
    manifest = Manifest.load(cfg)
    manifest.data["faiss_id_map"] = ["ghost"]
    manifest.save()
    with pytest.raises(IndexError_, match="must be rebuilt"):
        VectorIndex.load(Manifest.load(cfg), cfg)


# ---------------------------------------------------------------------------- indexer

def test_indexing_is_incremental(cfg):
    seed_chunks(cfg, "p1", n=3)
    first = index_chunks(cfg, embedder=StubEmbedder(cfg))
    assert first["chunks_indexed"] == 3 and first["chunks_skipped"] == 0

    seed_chunks(cfg, "p2", n=2)
    second = index_chunks(cfg, embedder=StubEmbedder(cfg))
    assert second["chunks_indexed"] == 2, "only the new paper is embedded"
    assert second["chunks_skipped"] == 3
    assert second["total_indexed"] == 5

    third = index_chunks(cfg, embedder=StubEmbedder(cfg))
    assert third["chunks_indexed"] == 0


def test_appending_preserves_existing_map_positions(cfg):
    seed_chunks(cfg, "p1", n=3)
    index_chunks(cfg, embedder=StubEmbedder(cfg))
    before = list(Manifest.load(cfg).data["faiss_id_map"])

    seed_chunks(cfg, "p2", n=2)
    index_chunks(cfg, embedder=StubEmbedder(cfg))
    after = list(Manifest.load(cfg).data["faiss_id_map"])

    assert after[: len(before)] == before, "an append must never reorder existing entries"


def test_changed_embedding_model_forces_a_rebuild(cfg):
    seed_chunks(cfg, "p1", n=3)
    index_chunks(cfg, embedder=StubEmbedder(cfg))

    manifest = Manifest.load(cfg)
    assert not model_changed(manifest, cfg)
    manifest.data["embedding_model"] = "some/other-model"
    manifest.save()

    # A strict load is what refuses a mismatched manifest outright; index_chunks loads
    # non-strict precisely so it can rebuild instead of dying.
    from manifest import ManifestError

    with pytest.raises(ManifestError, match="must be rebuilt"):
        Manifest.load(cfg)
    assert model_changed(Manifest.load(cfg, strict_model_check=False), cfg)

    rebuilt = index_chunks(cfg, embedder=StubEmbedder(cfg))
    assert rebuilt["rebuilt"] is True
    assert rebuilt["chunks_indexed"] == 3, "everything is re-embedded, not appended"
    assert "some/other-model" in rebuilt["rebuild_reason"]


def test_indexing_with_no_chunks_returns_an_error_dict(cfg):
    result = index_chunks(cfg, embedder=StubEmbedder(cfg))
    assert result["error"] == "no_chunks"
