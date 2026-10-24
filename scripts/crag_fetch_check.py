"""
Manual CRAG check: one real search_literature call against a THROWAWAY copy of the data.

In:  config.yaml's data dir (copied whole to a temp dir), an absent-but-fetchable query
     from eval/topics.yaml, and --max-results (default 3).
Out: a JSON report on stdout — admission gate outcome, index growth in the copy, and
     proof the real data/index files are byte-identical before and after. The copy is
     deleted. Refuses to run against the real data dir, or while an arXiv cooldown is live.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ("data_dir", "papers_dir", "index_dir", "cache_dir", "logs_dir")


# ---- isolation: pure path checks, no src import -------------------------------------

def configured_paths(config_file: Path, root: Optional[Path] = None) -> Dict[str, Path]:
    """Every data path a config file names, resolved the way config.Paths resolves them."""
    root = ROOT if root is None else root  # read at call time, not bound at import
    raw = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    return {key: (root / raw["paths"][key]).resolve() for key in PATH_KEYS}


def refusal_reason(candidates: Iterable[Path], real_data: Path) -> Optional[str]:
    """Why these paths are unsafe to write, or None when all are outside the real data.

    A path equal to, inside, or containing the real data dir is refused: any of the
    three lets the check write into the corpus the eval numbers describe.
    """
    real = Path(real_data).resolve()
    for candidate in candidates:
        path = Path(candidate).resolve()
        if path == real or real in path.parents or path in real.parents:
            return f"{path} overlaps the real data dir {real}; refusing to run"
    return None


# ---- copy, config, hashing ----------------------------------------------------------

def hash_files(directory: Path) -> Dict[str, str]:
    """sha256 of every file under `directory`, keyed by relative path."""
    out: Dict[str, str] = {}
    for path in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
        out[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def write_scratch_config(scratch: Path) -> Path:
    """config.yaml with every path pointed into `scratch`, written as absolute paths.

    Absolute, because config.Paths joins REPO_ROOT with the value and an absolute value
    wins that join — so the copy is used whatever directory this runs from.
    """
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    data = scratch / "data"
    raw["paths"] = {
        "data_dir": str(data),
        "papers_dir": str(data / "papers"),
        "index_dir": str(data / "index"),
        "cache_dir": str(data / "cache"),
        "logs_dir": str(scratch / "logs"),
        "prompts_dir": str(ROOT / "src" / "prompts"),
    }
    path = scratch / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _counts(manifest_path: Path) -> Dict[str, int]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"papers": 0, "vectors": 0}
    return {"papers": len(manifest.get("papers", {})),
            "vectors": len(manifest.get("faiss_id_map", []))}


# ---- the check ----------------------------------------------------------------------

def run_check(query: str, max_results: int) -> Dict[str, Any]:
    real = configured_paths(ROOT / "config.yaml")
    real_data, real_index = real["data_dir"], real["index_dir"]
    real_cooldown = real["cache_dir"] / "arxiv_cooldown.json"

    sys.path.insert(0, str(ROOT / "src"))
    from corpus.arxiv_fetch import read_cooldown  # no config import behind this one

    remaining = read_cooldown(real_cooldown)
    if remaining > 0:
        return {"status": "refused", "reason": f"arXiv cooldown active for {remaining:.0f}s "
                                               f"({real_cooldown}); not touching arXiv"}

    before = hash_files(real_index)
    scratch = Path(tempfile.mkdtemp(prefix="crag_fetch_check_"))
    report: Dict[str, Any] = {"query": query, "max_results": max_results, "scratch": str(scratch)}
    try:
        shutil.copytree(real_data, scratch / "data", symlinks=False)
        config_file = write_scratch_config(scratch)
        reason = refusal_reason(configured_paths(config_file).values(), real_data)
        if reason:
            return {"status": "refused", "reason": reason}

        # Must be set before anything under src/ is imported: config.CFG is built at
        # import time from $SCIAGENT_CONFIG, and every module defaults to it.
        os.environ["SCIAGENT_CONFIG"] = str(config_file)
        from config import CFG
        from tools.search_literature import LiteratureSearcher

        # The safety net if anything imported config before the variable was set: CFG
        # would then be the real config, and its paths would fail this check.
        reason = refusal_reason([CFG.paths.data, CFG.paths.index, CFG.paths.cache,
                                 CFG.paths.papers, CFG.paths.logs], real_data)
        if not reason and Path(CFG.source_path).resolve() != config_file.resolve():
            reason = f"config was loaded from {CFG.source_path}, not the scratch config"
        if reason:
            return {"status": "refused", "reason": reason}

        report["copy_before"] = _counts(CFG.paths.manifest)
        started = time.perf_counter()
        result = LiteratureSearcher(CFG).search(query, max_results=max_results)
        report["elapsed_s"] = round(time.perf_counter() - started, 1)
        report["copy_after"] = _counts(CFG.paths.manifest)
        report["index_growth"] = {
            key: report["copy_after"][key] - report["copy_before"][key]
            for key in report["copy_before"]
        }
        payload = result.get("partial") if "error" in result else result
        payload = payload or {}
        report["search"] = {
            "error": result.get("error"),
            "detail": result.get("detail"),
            "admission_gate": payload.get("admission_gate"),
            "admission_threshold": payload.get("admission_threshold"),
            "papers_added": [{"arxiv_id": p.get("arxiv_id"), "title": p.get("title"),
                              "admission_score": p.get("admission_score")}
                             for p in payload.get("papers_added", [])],
            "papers_rejected": payload.get("papers_rejected", []),
            "chunks_added": payload.get("chunks_added"),
            "figures_added": payload.get("figures_added"),
            "arxiv_query": payload.get("arxiv_query"),
            "note": payload.get("note"),
        }

        # A 429 is per IP, so the cooldown recorded inside the copy must outlive it —
        # otherwise deleting the copy deletes the only record and the next real run
        # hammers arXiv straight into the block. Only the cooldown memo is carried back.
        copy_cooldown = CFG.paths.arxiv_cooldown
        if read_cooldown(copy_cooldown) > read_cooldown(real_cooldown):
            real_cooldown.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(copy_cooldown, real_cooldown)
            report["cooldown_propagated_to"] = str(real_cooldown)
        report["status"] = "rate_limited" if result.get("error") == "arxiv_rate_limited" else "ran"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        report["scratch_deleted"] = not scratch.exists()
        after = hash_files(real_index)
        report["real_index_identical"] = before == after
        report["real_index_files"] = sorted(before)
    return report


def default_query() -> str:
    raw = yaml.safe_load((ROOT / "eval" / "topics.yaml").read_text(encoding="utf-8"))
    return str(raw["crag_fetch_check_query"])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CRAG fetch check on a throwaway data copy.")
    parser.add_argument("--query", default=None, help="defaults to eval/topics.yaml's query")
    parser.add_argument("--max-results", type=int, default=3)
    args = parser.parse_args(argv)

    report = run_check(args.query or default_query(), args.max_results)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report.get("status") == "refused":
        return 2
    return 0 if report.get("real_index_identical") else 1


if __name__ == "__main__":
    sys.exit(main())
