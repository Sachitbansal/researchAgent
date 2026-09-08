"""
Smoke tests for the config loader: shapes, path resolution, and loud failure on typos.

In:  the repo's real config.yaml, plus small temp files for the failure paths.
Out: assertions only. No network, no API key required.
"""

from pathlib import Path

import pytest

from config import CFG, Config, ConfigError, REPO_ROOT, load_config


def test_required_sections_are_present():
    for section in ("llm", "collection", "chunking", "embedding", "retrieval", "nli", "agent"):
        assert section in CFG.as_dict()


def test_values_have_the_documented_types():
    assert isinstance(CFG.retrieval.k_retrieve, int)
    assert isinstance(CFG.retrieval.rerank_enabled, bool)
    assert isinstance(CFG.analysis.cluster_k_range, list)
    assert CFG.retrieval.k_final < CFG.retrieval.k_retrieve


def test_chunk_ceiling_fits_both_encoder_windows():
    """The chunk ceiling is derived, not guessed: see docs/ARCHITECTURE.md section 5."""
    cross_encoder_window = 512
    specials = 3  # [CLS] query [SEP] chunk [SEP]
    budget = cross_encoder_window - CFG.retrieval.max_query_tokens - specials

    assert CFG.chunking.max_tokens <= budget, "a chunk + query would overflow the reranker"
    assert CFG.chunking.max_tokens <= CFG.embedding.max_seq_tokens, "chunks truncate at embed time"
    assert CFG.chunking.target_tokens < CFG.chunking.max_tokens
    assert CFG.chunking.overlap_tokens < CFG.chunking.target_tokens


def test_bge_prefixes_are_asymmetric():
    """BGE puts the instruction on the query only; prefixing passages degrades retrieval."""
    assert CFG.embedding.model.startswith("BAAI/bge-")
    assert CFG.embedding.query_prefix.strip()
    assert CFG.embedding.passage_prefix == ""
    assert CFG.embedding.normalize is True


def test_arxiv_client_settings_are_present():
    assert CFG.collection.request_delay_s > 0
    assert CFG.collection.sort_by == "relevance"
    assert "api_url" not in CFG.collection, "the arxiv package owns its own endpoint"


def test_unknown_key_raises_instead_of_returning_none():
    with pytest.raises(ConfigError, match="no config key 'retrieval.k_retreive'"):
        _ = CFG.retrieval.k_retreive


def test_sections_are_read_only():
    with pytest.raises(ConfigError):
        CFG.retrieval.k_final = 99


def test_paths_are_absolute_and_match_the_data_schema():
    paths = CFG.paths
    assert paths.manifest == REPO_ROOT / "data/index/manifest.json"
    assert paths.chunks == REPO_ROOT / "data/index/chunks.jsonl"
    assert paths.embeddings == REPO_ROOT / "data/index/embeddings.npy"
    assert paths.faiss_index == REPO_ROOT / "data/index/faiss.index"
    assert paths.embedding_cache == REPO_ROOT / "data/cache/embeddings.json"
    assert paths.descriptions == REPO_ROOT / "data/cache/descriptions.json"
    assert paths.arxiv_queries == REPO_ROOT / "data/cache/arxiv_queries.json"
    assert paths.clusters == REPO_ROOT / "data/cache/clusters.json"
    assert all(path.is_absolute() for path in paths.directories())


def test_embedding_cache_is_separate_from_the_index_vectors():
    """data/cache/embeddings.json is a content-hash cache; embeddings.npy is the index."""
    assert CFG.paths.embedding_cache != CFG.paths.embeddings
    assert CFG.paths.embedding_cache.parent == CFG.paths.cache
    assert CFG.paths.embeddings.parent == CFG.paths.index


def test_derived_paths():
    assert CFG.paths.paper_pdf("2103_14030v2").name == "2103_14030v2.pdf"
    assert CFG.paths.figure_image("p1", "p1__f03").parent.name == "p1"
    assert CFG.paths.trace_file("run-1").parent == CFG.paths.traces


def test_ensure_is_idempotent():
    CFG.paths.ensure()
    CFG.paths.ensure()
    assert all(path.is_dir() for path in CFG.paths.directories())


def test_model_roles_are_configured():
    for role in ("agent", "vision", "utility"):
        assert CFG.model_for(role)
    with pytest.raises(ConfigError, match="unknown llm role"):
        CFG.model_for("nonsense")


def test_system_prompt_loads_from_a_file():
    assert "retrieve_evidence" in CFG.prompt("system")
    with pytest.raises(ConfigError, match="prompt not found"):
        CFG.prompt("does_not_exist")


def test_missing_section_is_rejected(tmp_path: Path):
    bad = tmp_path / "config.yaml"
    bad.write_text("llm:\n  provider: openrouter\n")
    with pytest.raises(ConfigError, match="missing sections"):
        load_config(bad)


def test_missing_file_is_rejected(tmp_path: Path):
    with pytest.raises(ConfigError, match="config file not found"):
        Config(tmp_path / "nope.yaml")
