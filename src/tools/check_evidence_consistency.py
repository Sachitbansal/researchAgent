"""
`check_evidence_consistency`: NLI over retrieved evidence, in two modes.

In:  mode ('contradiction' | 'groundedness'), chunk_ids, and answer_text for groundedness.
Out: contradiction -> {"found", "conflicting_pairs"}; groundedness -> {"claims",
     "grounded_ratio", "unsupported_claims"}. Advisory, never a verdict; the model is a
     general MNLI checkpoint and is weak on scientific text.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analysis.nli import NLIError, NLIModel
from common.tokenization import split_sentences
from config import CFG, Config
from corpus.chunk_store import ChunkStore
from llm_client import LLMClient, LLMError

MODES = ("contradiction", "groundedness")
_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)
_CITATION = re.compile(r"\[[A-Za-z0-9_]+__c\d+\]")


def _error(code: str, detail: str, partial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"error": code, "detail": detail, "partial": partial}


def _sentences(text: str, cap: int) -> List[str]:
    """Sentences worth scoring: long enough to assert something, capped for cost."""
    usable = [s for s in split_sentences(text) if len(s.split()) >= 4]
    return usable[:cap]


class ConsistencyChecker:
    """Contradiction and groundedness checks over retrieved chunks.

    Claim extraction happens inside this tool, never in the agent. If the agent supplied
    its own claim list it would be grading a list it selected, and could quietly omit the
    claims it knows are unsupported.
    """

    def __init__(
        self,
        config: Config = CFG,
        client: Optional[LLMClient] = None,
        nli: Optional[NLIModel] = None,
    ) -> None:
        self.config = config
        self.client = client
        self.nli = nli if nli is not None else NLIModel(config)

    # ---- shared ----------------------------------------------------------------

    def _resolve(self, chunk_ids: Sequence[str]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        wanted = list(dict.fromkeys(str(cid) for cid in chunk_ids))
        found = {c["chunk_id"]: c for c in ChunkStore(config=self.config)
                 if c["chunk_id"] in set(wanted)}
        missing = [cid for cid in wanted if cid not in found]
        return found, missing

    # ---- mode: contradiction ---------------------------------------------------

    def contradiction(self, chunk_ids: Sequence[str]) -> Dict[str, Any]:
        """Find chunk pairs whose sentences disagree."""
        chunks, missing = self._resolve(chunk_ids)
        if len(chunks) < 2:
            return _error(
                "too_few_chunks",
                f"contradiction checking needs at least 2 known chunks, resolved "
                f"{len(chunks)}" + (f"; unknown ids: {missing}" if missing else ""),
            )

        cap = int(self.config.nli.max_sentences_per_chunk)
        threshold = float(self.config.nli.contradiction_threshold)
        max_pairs = int(self.config.nli.max_pairs)

        by_id = {cid: _sentences(chunk["text"], cap) for cid, chunk in chunks.items()}
        pairs: List[Tuple[str, str]] = []
        provenance: List[Tuple[str, str, str, str]] = []

        for left, right in combinations(sorted(chunks), 2):
            if len(provenance) >= max_pairs * cap:
                break
            for premise in by_id[left]:
                for hypothesis in by_id[right]:
                    pairs.append((premise, hypothesis))
                    provenance.append((left, right, premise, hypothesis))

        if not pairs:
            return {"found": False, "conflicting_pairs": [], "n_pairs_scored": 0,
                    "note": "the chunks contain no sentences long enough to compare"}

        try:
            scored = self.nli.predict(pairs)
        except NLIError as exc:
            return _error("nli_unavailable", str(exc))

        best_per_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for (left, right, premise, hypothesis), result in zip(provenance, scored):
            confidence = result["scores"].get("contradiction", 0.0)
            if confidence < threshold:
                continue
            key = (left, right)
            if key not in best_per_pair or confidence > best_per_pair[key]["confidence"]:
                best_per_pair[key] = {
                    "chunk_a": left, "chunk_b": right,
                    "snippet_a": premise, "snippet_b": hypothesis,
                    "confidence": round(float(confidence), 4),
                }

        conflicts = sorted(best_per_pair.values(), key=lambda c: c["confidence"], reverse=True)
        return {
            "found": bool(conflicts),
            "conflicting_pairs": conflicts,
            "n_pairs_scored": len(pairs),
            "unknown_chunk_ids": missing,
            "note": (
                "Advisory only: this is a general MNLI model and is weak on scientific "
                "text. Verify a flagged pair by reading both snippets before reporting a "
                "disagreement."
            ),
        }

    # ---- mode: groundedness ----------------------------------------------------

    def extract_claims(self, answer_text: str) -> Tuple[List[str], Optional[str]]:
        """Split a drafted answer into atomic claims, with one LLM call.

        Inside the tool by design. An agent that supplied its own claim list would be
        choosing what to be graded on.
        """
        cleaned = _CITATION.sub("", answer_text).strip()
        if not cleaned:
            return [], "answer_text is empty once citations are stripped"

        client = self.client if self.client is not None else LLMClient(config=self.config)
        messages = [
            {"role": "system", "content": self.config.prompt("claim_extraction")},
            {"role": "user", "content": cleaned},
        ]
        try:
            response = client.complete(
                messages, role="utility",
                timeout=float(self.config.llm.planning_timeout_s),
                max_retries=int(self.config.llm.planning_retries),
            )
        except LLMError as exc:
            return [], f"{exc.code}: {exc.detail}"

        match = _JSON_ARRAY.search(response.text or "")
        if not match:
            return [], "the claim-extraction reply contained no JSON array"
        try:
            payload = json.loads(match.group(0))
        except ValueError as exc:
            return [], f"the claim-extraction reply was not valid JSON: {exc}"
        if not isinstance(payload, list):
            return [], "the claim-extraction reply was not a list"

        claims = [str(item).strip() for item in payload if str(item).strip()]
        return claims[: int(self.config.nli.max_claims)], None

    def groundedness(self, chunk_ids: Sequence[str], answer_text: str) -> Dict[str, Any]:
        """Check each claim in a drafted answer against the chunks it was built from."""
        if not (answer_text or "").strip():
            return _error("missing_answer_text",
                          "groundedness mode requires the drafted answer_text")

        chunks, missing = self._resolve(chunk_ids)
        if not chunks:
            return _error(
                "no_chunks",
                f"none of the given chunk_ids are in the corpus: {missing}",
            )

        claims, failure = self.extract_claims(answer_text)
        if failure:
            return _error("claim_extraction_failed", failure)
        if not claims:
            return {"claims": [], "grounded_ratio": 1.0, "unsupported_claims": [],
                    "note": "the answer makes no checkable factual claims"}

        cap = int(self.config.nli.max_sentences_per_chunk)
        premises: List[Tuple[str, str]] = []
        for chunk_id, chunk in sorted(chunks.items()):
            for sentence in _sentences(chunk["text"], cap):
                premises.append((chunk_id, sentence))
        if not premises:
            return _error("no_premises",
                          "the given chunks contain no sentences long enough to check against")

        pairs = [(sentence, claim) for claim in claims for _, sentence in premises]
        try:
            scored = self.nli.predict(pairs)
        except NLIError as exc:
            return _error("nli_unavailable", str(exc))

        entail_threshold = float(self.config.nli.entailment_threshold)
        contradiction_threshold = float(self.config.nli.contradiction_threshold)

        results: List[Dict[str, Any]] = []
        stride = len(premises)
        for index, claim in enumerate(claims):
            window = scored[index * stride:(index + 1) * stride]
            best_entail = max(
                range(len(window)), key=lambda i: window[i]["scores"].get("entailment", 0.0)
            )
            best_contra = max(
                range(len(window)), key=lambda i: window[i]["scores"].get("contradiction", 0.0)
            )
            entail_score = window[best_entail]["scores"].get("entailment", 0.0)
            contra_score = window[best_contra]["scores"].get("contradiction", 0.0)

            if entail_score >= entail_threshold:
                label, confidence, chunk_id = "entailed", entail_score, premises[best_entail][0]
            elif contra_score >= contradiction_threshold:
                label, confidence, chunk_id = ("contradicted", contra_score,
                                               premises[best_contra][0])
            else:
                label, confidence, chunk_id = "neutral", entail_score, premises[best_entail][0]

            results.append({
                "claim": claim,
                "label": label,
                "best_supporting_chunk": chunk_id,
                "confidence": round(float(confidence), 4),
            })

        entailed = sum(1 for r in results if r["label"] == "entailed")
        return {
            "claims": results,
            "grounded_ratio": round(entailed / len(results), 4),
            "unsupported_claims": [r["claim"] for r in results if r["label"] != "entailed"],
            "n_claims": len(results),
            "n_premises": len(premises),
            "unknown_chunk_ids": missing,
            "note": (
                "Advisory only. A 'neutral' label means the model did not find support — "
                "which can equally mean the claim is unsupported or that the model did not "
                "understand the scientific text. Confidences are uncalibrated."
            ),
        }

    # ---- entry point -----------------------------------------------------------

    def check(
        self,
        mode: str,
        chunk_ids: Sequence[str],
        answer_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        mode = (mode or "").strip().lower()
        if mode not in MODES:
            return _error("unknown_mode", f"mode must be one of {list(MODES)}, got {mode!r}")
        if not chunk_ids:
            return _error("missing_chunk_ids", "at least one chunk_id is required")
        if mode == "contradiction":
            return self.contradiction(chunk_ids)
        return self.groundedness(chunk_ids, answer_text or "")


CHECK_CONSISTENCY_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": list(MODES),
            "description": (
                "contradiction: find chunks that disagree with each other. "
                "groundedness: check your drafted answer against the chunks you used."
            ),
        },
        "chunk_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "chunk_ids from earlier retrieve_evidence results.",
        },
        "answer_text": {
            "type": "string",
            "description": "Your drafted answer. Required for groundedness mode only.",
        },
    },
    "required": ["mode", "chunk_ids"],
    "additionalProperties": False,
}
