"""
Hand-written Okapi BM25 over the first-stage chunk text: the keyword half of hybrid retrieval.

In:  (chunk_id, text) pairs — the same header-prefixed text the bi-encoder embeds — and
     retrieval.bm25_k1 / bm25_b; then a query string and an optional allowed-id filter.
Out: ranked (chunk_id, bm25_score) pairs, best first, only for chunks sharing a query term.
     Built in memory with numpy/scipy; nothing is persisted.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import sparse

# Letters and digits in any script (so λ, α, ε survive), joined across internal hyphens
# and dots so "1e-2", "0.01", "top-1" and "state-of-the-art" stay one token. Trailing
# punctuation is never part of a token, so "model." and "model" match.
_TOKEN = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*")
_HAS_LETTER = re.compile(r"[^\W\d_]")

# Light on purpose: only words that carry no topic. Negations and quantity words are
# kept, because "no warm-up" and "without warm-up" are what a query is about.
STOPWORDS = frozenset("""
a an and are as at be been being by for from has have had in into is it its of on or
that the their them then there these they this those to was were which while who will
with what when where why how do does did can could should would we our you your
""".split())


def tokenize(text: str) -> List[str]:
    """Lowercased terms of `text`, with hyphenated words also split into their parts.

    "state-of-the-art" yields the compound and "state", "art", so a query
    written either way matches. Numeric compounds ("1e-2", "0.01") are kept whole only,
    since their pieces ("1e", "2") would match every number in a results table.
    """
    out: List[str] = []
    for match in _TOKEN.finditer((text or "").lower()):
        term = match.group(0)
        if term not in STOPWORDS:
            out.append(term)
        if "-" in term:
            parts = [p for p in re.split(r"[-.]", term) if p]
            if all(_HAS_LETTER.search(p) for p in parts):
                out.extend(p for p in parts if p not in STOPWORDS)
    return out


class LexicalIndex:
    """BM25 over a fixed set of documents, as one sparse term-weight matrix.

    Every document's per-term BM25 weight is precomputed once, so a query is a column
    sum over its terms. At this corpus size (~1k chunks) building takes well under a
    second, so it is rebuilt from the chunk table rather than persisted.
    """

    def __init__(
        self,
        docs: Sequence[Tuple[str, str]],
        k1: float = 1.2,
        b: float = 0.75,
        signature: str = "",
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.signature = signature
        self.chunk_ids: List[str] = [chunk_id for chunk_id, _ in docs]
        self.vocabulary: Dict[str, int] = {}

        rows, cols, counts = [], [], []
        lengths = np.zeros(len(docs), dtype=np.float64)
        for row, (_, text) in enumerate(docs):
            terms = tokenize(text)
            lengths[row] = len(terms)
            tf: Dict[int, int] = {}
            for term in terms:
                col = self.vocabulary.setdefault(term, len(self.vocabulary))
                tf[col] = tf.get(col, 0) + 1
            for col, count in tf.items():
                rows.append(row)
                cols.append(col)
                counts.append(count)

        n_docs, n_terms = len(docs), len(self.vocabulary)
        self.n_docs = n_docs
        if n_docs == 0 or n_terms == 0:
            self.weights = sparse.csc_matrix((n_docs, n_terms), dtype=np.float64)
            return

        tf = np.asarray(counts, dtype=np.float64)
        rows_arr = np.asarray(rows)
        cols_arr = np.asarray(cols)
        df = np.bincount(cols_arr, minlength=n_terms).astype(np.float64)
        # The Lucene form of idf: log(1 + (N - df + 0.5) / (df + 0.5)). Never negative,
        # so a term in most documents still adds a little rather than subtracting.
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        avgdl = float(lengths.mean()) or 1.0
        norm = self.k1 * (1.0 - self.b + self.b * lengths[rows_arr] / avgdl)
        values = idf[cols_arr] * tf * (self.k1 + 1.0) / (tf + norm)
        self.weights = sparse.csc_matrix(
            (values, (rows_arr, cols_arr)), shape=(n_docs, n_terms), dtype=np.float64
        )

    def __len__(self) -> int:
        return self.n_docs

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for `query`; each distinct query term counts once."""
        cols = sorted({self.vocabulary[t] for t in tokenize(query) if t in self.vocabulary})
        if not cols or self.n_docs == 0:
            return np.zeros(self.n_docs, dtype=np.float64)
        return np.asarray(self.weights[:, cols].sum(axis=1)).ravel()

    def search(
        self, query: str, k: int, allowed: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """Top-k (chunk_id, score), best first. Only documents sharing a term qualify:
        a zero score is no evidence, and ranking those would hand RRF arbitrary order."""
        if k <= 0 or self.n_docs == 0:
            return []
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")
        out: List[Tuple[str, float]] = []
        for slot in order:
            score = float(scores[slot])
            if score <= 0.0:
                break
            chunk_id = self.chunk_ids[slot]
            if allowed is not None and chunk_id not in allowed:
                continue
            out.append((chunk_id, score))
            if len(out) >= k:
                break
        return out


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], rrf_k: int = 60, limit: Optional[int] = None
) -> List[Tuple[str, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of 1 / (rrf_k + rank_d), rank from 1.

    Uses ranks only, so a cosine list and a BM25 list combine without either's scale
    mattering. Ties keep first-seen order, which makes the fusion deterministic.
    """
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(fused.items(), key=lambda item: -item[1])
    return ordered[:limit] if limit is not None else ordered
