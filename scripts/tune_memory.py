"""
Measure long-term memory's thresholds on real answers (docs/DECISIONS.md D19).

In:  the non-abstaining, cited answers in eval/results/{tuning,held_out}_split.json, their
     questions, and the shipped index; the utility LLM (claim extraction with the real
     memory_extract prompt, then one paraphrase and one contradiction per claim), the
     bi-encoder and the NLI model. LLM replies are cached in eval/results/memory_claims.json.
Out: eval/results/memory_thresholds.json and a printed summary: cosine distributions for
     paraphrases / swaps / distinct claims / question-to-claim, the NLI verdicts that decide
     contradiction rules, and how many claims the min_pair_similarity floor drops.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "eval" / "results"
CACHE = RESULTS / "memory_claims.json"
OUT = RESULTS / "memory_thresholds.json"

VARIANTS_PROMPT = """For each numbered claim, write two new sentences.
paraphrase: the same claim in different words, same meaning, same paper or method named.
contradiction: about the same paper or method, but so it cannot be true at the same time
as the claim: change a number, swap the named method, or negate the result. Keep the
wording as close to the original as you can.
Return only a JSON array, one object per claim, in order:
[{"paraphrase": "...", "contradiction": "..."}]"""


def pct(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "min": round(float(a.min()), 4),
            **{f"p{q}": round(float(np.percentile(a, q)), 4) for q in (5, 10, 25, 50, 75, 90, 95)},
            "max": round(float(a.max()), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--refresh", action="store_true", help="ignore the LLM reply cache")
    args = parser.parse_args()

    from analysis.nli import NLIModel
    from config import CFG
    from llm_client import LLMClient
    from memory.longterm import LongTermMemory, cited_chunk_ids
    from retrieval.embedder import Embedder

    questions = {q["question_id"]: q["question"] for q in
                 json.loads((ROOT / "eval" / "questions.json").read_text())["questions"]}
    rows = []
    for name in ("tuning_split.json", "held_out_split.json"):
        for row in json.loads((RESULTS / name).read_text())["rows"]:
            if row.get("answer") and not row.get("abstained") and cited_chunk_ids(row["answer"]):
                rows.append(row)

    embedder, nli, client = Embedder(CFG), NLIModel(CFG), LLMClient(config=CFG)
    memory = LongTermMemory(store=None, config=CFG, embedder=embedder, nli=nli, client=client)
    table = memory.chunks()
    cache: Dict[str, Any] = {} if args.refresh or not CACHE.exists() else json.loads(CACHE.read_text())

    # ---- 1. extraction + grounding, per answer --------------------------------------
    per_answer: Dict[str, List[Dict[str, Any]]] = {}
    grounding = {"extracted": 0, "uncited": 0, "entailed": 0, "entailed_without_floor": 0,
                 "best_premise_similarity_of_entailed_without_floor": []}
    for row in rows:
        qid = row["question_id"]
        key = f"extract:{qid}"
        if key not in cache:
            got = memory.extract(questions[qid], row["answer"])
            if "error" in got:
                print(f"  {qid}: {got}", file=sys.stderr)
                continue
            cache[key] = got
            CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
        allowed = set(cited_chunk_ids(row["answer"])) & set(table)
        findings = []
        for f in cache[key]["findings"]:
            grounding["extracted"] += 1
            ids = [c for c in f["chunk_ids"] if c in allowed]
            if ids:
                findings.append({"text": f["text"], "check": f.get("check") or f["text"],
                                 "chunk_ids": ids})
            else:
                grounding["uncited"] += 1
        with_floor = memory.ground(findings, table)
        without = memory.ground(findings, table, min_similarity=-1.0)
        grounding["entailed"] += sum(f["label"] == "entailed" for f in with_floor)
        grounding["entailed_without_floor"] += sum(f["label"] == "entailed" for f in without)
        grounding["best_premise_similarity_of_entailed_without_floor"] += [
            f["best_premise_similarity"] for f in without if f["label"] == "entailed"]
        per_answer[qid] = [f for f in with_floor if f["label"] == "entailed"]
        print(f"  {qid}: {len(findings)} cited claims, "
              f"{sum(f['label'] == 'entailed' for f in with_floor)} entailed "
              f"({sum(f['label'] == 'entailed' for f in without)} without the floor)", flush=True)
    floor_sims = grounding.pop("best_premise_similarity_of_entailed_without_floor")
    grounding["best_premise_similarity_of_entailed_without_floor"] = pct(floor_sims)
    grounding["preferences_extracted"] = sum(len(cache.get(f"extract:{r['question_id']}", {})
                                                 .get("preferences") or []) for r in rows)

    claims = [(qid, f["text"]) for qid, fs in per_answer.items() for f in fs]
    texts = [t for _, t in claims]
    print(f"{len(claims)} grounded claims from {len(per_answer)} answers", flush=True)

    # ---- 2. paraphrase + contradiction variants -------------------------------------
    variants: List[Dict[str, str]] = []
    for start in range(0, len(texts), 12):
        batch = texts[start:start + 12]
        key = "variants:" + "|".join(batch)
        if key not in cache:
            numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(batch))
            reply = client.complete([{"role": "system", "content": VARIANTS_PROMPT},
                                     {"role": "user", "content": numbered}], role="utility",
                                    timeout=90.0, max_retries=3)
            match = re.search(r"\[.*\]", reply.text or "", re.DOTALL)
            parsed = json.loads(match.group(0)) if match else []
            if len(parsed) != len(batch):
                print(f"  variants batch {start}: got {len(parsed)} for {len(batch)}", file=sys.stderr)
                parsed = (parsed + [{}] * len(batch))[: len(batch)]
            cache[key] = parsed
            CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
        variants += cache[key]

    base = embedder.encode_passages(texts)
    pairs = [(i, v) for i, v in enumerate(variants)
             if isinstance(v, dict) and v.get("paraphrase") and v.get("contradiction")]
    para = embedder.encode_passages([v["paraphrase"] for _, v in pairs])
    contra = embedder.encode_passages([v["contradiction"] for _, v in pairs])
    para_cos = [float(base[i] @ para[k]) for k, (i, _) in enumerate(pairs)]
    contra_cos = [float(base[i] @ contra[k]) for k, (i, _) in enumerate(pairs)]

    threshold = float(CFG.memory.contradiction_threshold)

    def verdicts(old_new: List[tuple]) -> List[Dict[str, Any]]:
        scored = nli.predict([p for a, b in old_new for p in ((a, b), (b, a))])
        out = []
        for k in range(len(old_new)):
            fwd, bwd = scored[2 * k]["scores"], scored[2 * k + 1]["scores"]
            hit = [s["contradiction"] >= threshold and s["contradiction"] > s["entailment"]
                   for s in (fwd, bwd)]
            out.append({"both": all(hit), "either": any(hit),
                        "fwd_entail": fwd["entailment"], "bwd_entail": bwd["entailment"]})
        return out

    contra_v = verdicts([(texts[i], v["contradiction"]) for i, v in pairs])
    para_v = verdicts([(texts[i], v["paraphrase"]) for i, v in pairs])

    # ---- 3. distinct claims: same answer (related) and across answers ---------------
    same, cross, same_pairs = [], [], []
    for a, b in itertools.combinations(range(len(claims)), 2):
        cos = float(base[a] @ base[b])
        if claims[a][0] == claims[b][0]:
            same.append(cos)
            same_pairs.append((a, b, cos))
        else:
            cross.append(cos)
    all_pairs = [(a, b, float(base[a] @ base[b]))
                 for a, b in itertools.combinations(range(len(claims)), 2)]

    def false_contradictions(floor: float) -> Dict[str, Any]:
        near = [(a, b) for a, b, c in all_pairs if c >= floor]
        if not near:
            return {"pairs": 0, "both": 0, "either": 0}
        v = verdicts([(texts[a], texts[b]) for a, b in near])
        return {"pairs": len(near), "both": sum(x["both"] for x in v),
                "either": sum(x["either"] for x in v),
                "examples_both": [[texts[a], texts[b]] for (a, b), x in zip(near, v) if x["both"]][:5]}

    # ---- 4. question (query prefix) vs claims ---------------------------------------
    qids = sorted(per_answer)
    qvec = embedder.encode_queries([questions[q] for q in qids])
    own, other = [], []
    for qi, qid in enumerate(qids):
        for ci, (cq, _) in enumerate(claims):
            (own if cq == qid else other).append(float(qvec[qi] @ base[ci]))
    best_own = []
    for qi, qid in enumerate(qids):
        scores = [float(qvec[qi] @ base[ci]) for ci, (cq, _) in enumerate(claims) if cq == qid]
        if scores:
            best_own.append(max(scores))

    report = {
        "answers": len(rows), "grounded_claims": len(claims), "variant_pairs": len(pairs),
        "grounding": grounding,
        "cosine": {
            "paraphrase": pct(para_cos),
            "contradiction_swap": pct(contra_cos),
            "distinct_same_answer": pct(same),
            "distinct_cross_answer": pct(cross),
            "question_vs_own_claims": pct(own),
            "question_vs_best_own_claim": pct(best_own),
            "question_vs_other_claims": pct(other),
        },
        "nli": {
            "threshold": threshold,
            "swap_detected_both": sum(v["both"] for v in contra_v),
            "swap_detected_either": sum(v["either"] for v in contra_v),
            "paraphrase_flagged_both": sum(v["both"] for v in para_v),
            "paraphrase_flagged_either": sum(v["either"] for v in para_v),
            "of": len(pairs),
            "false_contradictions_distinct": {str(f): false_contradictions(f)
                                              for f in (0.70, 0.75, 0.80, 0.85)},
        },
        "swap_above": {str(t): sum(c >= t for c in contra_cos) for t in (0.80, 0.85, 0.90, 0.92, 0.95)},
        "paraphrase_above": {str(t): sum(c >= t for c in para_cos) for t in (0.85, 0.88, 0.90, 0.92, 0.95)},
        "distinct_above": {str(t): sum(c >= t for c in same + cross) for t in (0.85, 0.88, 0.90, 0.92, 0.95)},
        "question_best_own_above": {str(t): sum(c >= t for c in best_own)
                                    for t in (0.55, 0.60, 0.65, 0.70)},
        "question_own_above": {str(t): sum(c >= t for c in own) for t in (0.55, 0.60, 0.65, 0.70)},
        "question_other_above": {str(t): sum(c >= t for c in other) for t in (0.55, 0.60, 0.65, 0.70)},
        "examples": {
            "lowest_paraphrase": sorted(
                ([round(c, 3), texts[i], v["paraphrase"]] for c, (i, v) in zip(para_cos, pairs)))[:3],
            "highest_distinct": [[round(c, 3), texts[a], texts[b]] for a, b, c in
                                 sorted(all_pairs, key=lambda x: -x[2])[:5]],
        },
    }
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("grounding", "cosine", "nli", "swap_above",
                                             "paraphrase_above", "distinct_above",
                                             "question_best_own_above", "question_own_above",
                                             "question_other_above")}, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
