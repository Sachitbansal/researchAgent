"""
Sweeps search_literature's admission threshold offline, against the indexed abstracts.

In:  the manifest's paper records (title + abstract), the eval topics and the tuning
     split of eval/questions.json, plus off-domain queries from eval/topics.yaml.
Out: eval/results/admission_sweep.json — per-group score distributions, a sweep, and a
     recommendation. No arXiv calls and no LLM calls: only the local cross-encoder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.retrieval_eval import load_questions  # noqa: E402

CANDIDATES = [-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]

# Each pair is (query, paper_id, group, shape). Groups are what the gate must do with
# the pair; shape is whether the query is a short topic phrase (what search_literature
# is usually called with) or a full question (what a coverage_gap's proposed_query is).
Pair = Tuple[str, str, str, str]

POSITIVE = "positive"
ABSENT = "absent_topic"
CROSS = "cross_topic"


def build_pairs(papers: Dict[str, Dict[str, Any]], topics: Dict[str, Dict[str, Any]],
                questions: Sequence[Dict[str, Any]], off_domain: Sequence[str]) -> List[Pair]:
    """Every labelled (query, abstract) pair, from data already on disk."""
    pairs: List[Pair] = []
    by_topic = {tag: set(entry.get("paper_ids", [])) for tag, entry in topics.items()}

    # Positives, topic-shaped: each topic's own query against the papers indexed under it.
    # Cross-topic negatives, topic-shaped: the same query against the other topic's papers.
    # Adjacent but distinct is exactly the hard case.
    for tag, entry in topics.items():
        for paper_id in papers:
            group = POSITIVE if paper_id in by_topic[tag] else CROSS
            pairs.append((entry["query"], paper_id, group, "topic"))

    for question in questions:
        gold = set(question.get("gold_paper_ids") or [])
        if question.get("expect_abstention"):
            # Absent subject: no indexed abstract should admit it. This includes the
            # absent_fetchable questions — arXiv has papers for them, but not these.
            for paper_id in papers:
                pairs.append((question["question"], paper_id, ABSENT, "question"))
            continue
        # Positives, question-shaped: a question against its own gold papers. Its
        # cross-topic negatives are the papers of topics none of its gold papers is in;
        # questions spanning both topics (mp03) therefore contribute no negatives.
        gold_topics = {tag for tag, ids in by_topic.items() if gold & ids}
        for paper_id in papers:
            paper_topics = {tag for tag, ids in by_topic.items() if paper_id in ids}
            if paper_id in gold:
                pairs.append((question["question"], paper_id, POSITIVE, "question"))
            elif not (paper_topics & gold_topics):
                pairs.append((question["question"], paper_id, CROSS, "question"))

    for query in off_domain:
        for paper_id in papers:
            pairs.append((query, paper_id, ABSENT, "topic"))
    return pairs


def score_pairs(pairs: Sequence[Pair], papers: Dict[str, Dict[str, Any]], reranker) -> List[float]:
    """Cross-encoder scores, through the exact text and scoring path the gate uses."""
    from corpus.admission import admission_text, score_texts

    scores: List[float] = [0.0] * len(pairs)
    by_query: Dict[str, List[int]] = {}
    for index, (query, _, _, _) in enumerate(pairs):
        by_query.setdefault(query, []).append(index)
    for query, indices in by_query.items():
        texts = [admission_text(papers[pairs[i][1]]["title"], papers[pairs[i][1]]["abstract"])
                 for i in indices]
        for i, score in zip(indices, score_texts(query, texts, reranker)):
            scores[i] = score
    return scores


def _rate(values: Sequence[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else float("nan")


def _hmean(values: Sequence[float]) -> float:
    usable = [v for v in values if v == v]
    if not usable or any(v == 0 for v in usable):
        return 0.0
    return round(len(usable) / sum(1.0 / v for v in usable), 4)


def _distribution(scores: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(scores)
    if not ordered:
        return {}
    return {"n": len(ordered), "min": round(ordered[0], 3),
            "median": round(ordered[len(ordered) // 2], 3), "max": round(ordered[-1], 3)}


def sweep(pairs: Sequence[Pair], scores: Sequence[float],
          thresholds: Sequence[float] = CANDIDATES) -> Dict[str, Any]:
    """Gate behaviour per threshold, overall and per query shape."""
    rows = []
    for threshold in thresholds:
        row: Dict[str, Any] = {"threshold": threshold}
        for shape in ("all", "topic", "question"):
            def picked(group: str) -> List[float]:
                return [s for (q, p, g, sh), s in zip(pairs, scores)
                        if g == group and (shape == "all" or sh == shape)]
            admit = _rate([s >= threshold for s in picked(POSITIVE)])
            absent = _rate([s < threshold for s in picked(ABSENT)])
            cross = _rate([s < threshold for s in picked(CROSS)])
            # Harmonic mean of the three, so a threshold cannot win by abandoning one:
            # admitting everything scores 0 on the rejections, rejecting everything 0 on
            # the admissions.
            row[shape] = {"admits_positive": admit, "rejects_absent": absent,
                          "rejects_cross_topic": cross,
                          "balanced": _hmean([admit, absent, cross])}
        rows.append(row)
    # Chosen on TOPIC-shaped pairs first: search_literature is called with a topic
    # phrase, and the question-shaped positives are noisy labels (a multi-paper
    # question's gold set includes papers its wording barely touches). Question-shaped
    # balance breaks ties, then pooled, then the more permissive threshold, as
    # relevance_threshold's tie did: a wrong rejection costs the user a paper they asked for.
    best = max(rows, key=lambda r: (r["topic"]["balanced"], r["question"]["balanced"],
                                    r["all"]["balanced"], -r["threshold"]))
    return {"rows": rows, "recommended": best}


def main() -> int:
    from config import CFG
    from corpus.manifest import Manifest
    from retrieval.reranker import Reranker

    manifest = Manifest.load(CFG, strict_model_check=False)
    papers = {pid: p for pid, p in manifest.papers.items() if p.get("abstract") or p.get("title")}
    topics = manifest.topics
    off_domain = yaml.safe_load((ROOT / "eval" / "topics.yaml").read_text(encoding="utf-8"))[
        "admission_off_domain_queries"]
    questions = load_questions(held_out=False)  # tuning split only, never the held-out third

    pairs = build_pairs(papers, topics, questions, off_domain)
    scores = score_pairs(pairs, papers, Reranker(CFG))
    result = sweep(pairs, scores)

    groups = {}
    for group in (POSITIVE, ABSENT, CROSS):
        for shape in ("topic", "question"):
            groups[f"{group}/{shape}"] = _distribution(
                [s for (q, p, g, sh), s in zip(pairs, scores) if g == group and sh == shape])
    result.update({
        "n_pairs": len(pairs),
        "n_papers": len(papers),
        "question_ids": [q["question_id"] for q in questions],
        "distributions": groups,
        "pairs": [{"query": q, "paper_id": p, "group": g, "shape": sh, "score": round(s, 4)}
                  for (q, p, g, sh), s in zip(pairs, scores)],
    })

    print(f"{len(pairs)} pairs over {len(papers)} abstracts; tuning questions only\n")
    for name, dist in groups.items():
        print(f"  {name:<24} {dist}")
    print(f"\n{'thresh':>7}  {'admit+':>7}  {'rej absent':>10}  {'rej cross':>9}  {'balanced':>8}"
          f"   | topic-shaped bal  question-shaped bal")
    for row in result["rows"]:
        a = row["all"]
        print(f"{row['threshold']:>7.1f}  {a['admits_positive']:>7.3f}  {a['rejects_absent']:>10.3f}  "
              f"{a['rejects_cross_topic']:>9.3f}  {a['balanced']:>8.3f}   | "
              f"{row['topic']['balanced']:>16.3f}  {row['question']['balanced']:>19.3f}")
    print(f"\nrecommended admission_threshold: {result['recommended']['threshold']}")

    out = ROOT / "eval" / "results" / "admission_sweep.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
