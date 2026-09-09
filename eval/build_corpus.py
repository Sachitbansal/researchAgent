"""
Builds the eval corpus: two adjacent-but-distinct topics into one shared index.

In:  eval/topics.yaml.
Out: data/ populated and indexed, and a printed manifest of what landed, so the eval
     questions can be written from the abstracts before the system is ever run on them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import CFG  # noqa: E402
from corpus.manifest import Manifest  # noqa: E402
from retrieval.indexer import index_chunks  # noqa: E402
from tools.search_literature import LiteratureSearcher  # noqa: E402


def main() -> int:
    spec = yaml.safe_load((ROOT / "eval" / "topics.yaml").read_text(encoding="utf-8"))
    searcher = LiteratureSearcher(CFG)

    for tag, settings in spec["topics"].items():
        print(f"\n=== {tag}: {settings['query']!r}", flush=True)
        result = searcher.search(
            settings["query"],
            max_results=settings.get("max_results"),
            categories=settings.get("categories"),
        )
        if "error" in result:
            print(f"    FAILED {result['error']}: {result['detail']}", flush=True)
            continue
        # search_literature tags by a slug of the query; retag to the eval's own name so
        # questions can reference a stable tag.
        manifest = Manifest.load(CFG, strict_model_check=False)
        print(f"    added {len(result['papers_added'])}, skipped {result['papers_skipped']}, "
              f"chunks {result['chunks_added']}, figures {result['figures_added']}", flush=True)
        print(f"    tagged {result['topic_tag']!r}", flush=True)
        if result.get("failures"):
            print(f"    failures: {result['failures']}", flush=True)

    print("\n=== indexing", flush=True)
    print("   ", index_chunks(CFG), flush=True)

    manifest = Manifest.load(CFG, strict_model_check=False)
    print(f"\n=== corpus: {len(manifest.papers)} papers", flush=True)
    for paper_id, paper in sorted(manifest.papers.items()):
        print(f"\n  {paper_id}  ({paper['year']})  tags={paper['topic_tags']}", flush=True)
        print(f"    {paper['title']}", flush=True)
        print(f"    chunks={paper['n_chunks']} figures={paper['n_figures']} "
              f"status={paper['parse_status']}", flush=True)
        print(f"    ABSTRACT: {paper['abstract'][:700]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
