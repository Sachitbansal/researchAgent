"""
The corpus as the browser shows it: indexed papers grouped by the topic they came from.

In:  a Config (for the manifest path). Out: a JSON-ready dict of papers, topics, the
     embedding model and the manifest's update time — or an empty corpus with an error
     line when nothing is indexed. Read-only; it never touches the index itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def read_corpus(config: Any) -> Dict[str, Any]:
    """The indexed papers and the topics they were collected under, for the corpus view.

    Read fresh on every request rather than cached: indexing from the CLI while the
    server is up should show up on the next reload, not on the next restart.
    """
    path = Path(config.paths.manifest)
    if not path.exists():
        return {"papers": [], "topics": [], "error": "no index yet — run: python main.py index \"<topic>\""}

    manifest = json.loads(path.read_text(encoding="utf-8"))
    raw_topics = manifest.get("topics") or {}
    papers: List[Dict[str, Any]] = []
    for record in manifest.get("papers", {}).values():
        pdf_path = record.get("pdf_path") or ""
        papers.append({
            "paper_id": record.get("paper_id"),
            "arxiv_id": record.get("arxiv_id"),
            "title": record.get("title") or record.get("paper_id"),
            "authors": record.get("authors") or [],
            "year": record.get("year"),
            "categories": record.get("categories") or [],
            "topic_tags": record.get("topic_tags") or [],
            "n_chunks": record.get("n_chunks", 0),
            "n_figures": record.get("n_figures", 0),
            "pdf_name": Path(pdf_path).name if pdf_path else "",
            "pdf_url": record.get("pdf_url") or "",
            "parse_status": record.get("parse_status"),
        })
    papers.sort(key=lambda p: (str(p["title"]).lower()))

    topics = [
        {
            "tag": tag,
            # The topic as it was typed, which reads better than the slug it became.
            "label": (body or {}).get("query") or tag.replace("_", " "),
            "arxiv_query": (body or {}).get("arxiv_query", ""),
            "count": sum(1 for p in papers if tag in p["topic_tags"]),
        }
        for tag, body in raw_topics.items()
    ]
    topics.sort(key=lambda t: -t["count"])
    return {
        "papers": papers,
        "topics": topics,
        "embedding_model": manifest.get("embedding_model", ""),
        "updated_at": manifest.get("updated_at", ""),
    }
