"""BM25 keyword shortlist over the ingested passages.

This is the cheap, imprecise first pass from the rerank cookbook: it narrows
the corpus down before TypeSafe judges each candidate individually. BM25
alone is not used to produce the final ranking.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .ingest import Passage

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, passages: list[Passage]) -> None:
        self._passages = passages
        # BM25Okapi chokes on an empty corpus (e.g. no documents ingested
        # yet); leave it unset and short-circuit shortlist() instead.
        self._bm25 = BM25Okapi([_tokenize(p.text) for p in passages]) if passages else None

    def __len__(self) -> int:
        return len(self._passages)

    def shortlist(self, query: str, k: int) -> list[Passage]:
        if self._bm25 is None:
            return []
        # Top-k by score, zero-score candidates included: a BM25 miss just
        # means no lexical overlap, not that TypeSafe can't judge it. Dropping
        # them here would starve the reranker of evidence it never got to see.
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(self._passages)), key=lambda i: scores[i], reverse=True)
        return [self._passages[i] for i in ranked[:k]]
