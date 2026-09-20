"""Orchestrates the extractive RAG pipeline:

    BM25 shortlist -> TypeSafe rerank + gate -> existence check -> excerpts

There is no generation step anywhere. The "answer" this returns is always
the corpus's own text, selected and ranked -- never synthesized.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from typesafe_sdk import TypeSafeClient

from . import config
from .ingest import discover_documents, load_corpus
from .judge import JudgedPassage, check_answerable, judge_candidates
from .retrieval import BM25Index


def _group_key(judged: JudgedPassage) -> tuple[str, str]:
    """Passages are grouped by (source document, section) so a job/section
    heading that happens to repeat verbatim across two different documents
    (e.g. two resumes both having a "Summary" section) isn't treated as one
    group."""
    return (judged.passage.source, judged.passage.context or judged.passage.id)


def _diversify(
    gated: list[JudgedPassage], max_excerpts: int, max_per_context: int, strong_min: float
) -> list[JudgedPassage]:
    """Take the best-scoring excerpts, breadth-first across document/section
    groups before depth within one. Without this, one section with many
    strong bullets (e.g. six at one employer) crowds out every other group
    for a "which companies..." style query, even though each individual
    bullet is correctly judged -- the problem is entirely in how the top-N
    is chosen."""
    kept: list[JudgedPassage] = []
    per_group: dict[tuple[str, str], int] = {}
    included_ids: set[str] = set()

    def take(judged: JudgedPassage) -> bool:
        kept.append(judged)
        included_ids.add(judged.passage.id)
        key = _group_key(judged)
        per_group[key] = per_group.get(key, 0) + 1
        return len(kept) >= max_excerpts

    # Pass 1: one excerpt per distinct group, best-scoring first, so every
    # strongly-relevant group gets a seat before anyone gets a second one.
    # Restricted to strong_min: a passage that barely cleared the gate
    # shouldn't out-compete a clearly on-topic passage elsewhere for a slot.
    seen_groups: set[tuple[str, str]] = set()
    for judged in gated:
        if judged.combined < strong_min:
            continue
        key = _group_key(judged)
        if key in seen_groups:
            continue
        seen_groups.add(key)
        if take(judged):
            return kept

    # Pass 2: fill remaining slots with the next-best excerpts, still capped
    # per group so this pass adds depth without one group retaking everything.
    for judged in gated:
        if judged.passage.id in included_ids:
            continue
        if per_group.get(_group_key(judged), 0) >= max_per_context:
            continue
        if take(judged):
            break

    return kept


@dataclass(frozen=True)
class AnswerResult:
    verdict: str  # "answered" | "partial" | "not_found"
    confidence: float
    excerpts: list[JudgedPassage]


class ExtractiveRag:
    def __init__(self, corpus_dir: Path = config.CORPUS_DIR) -> None:
        self._corpus_dir = corpus_dir
        self._lock = threading.Lock()
        self._index, self._documents, self._empty_documents = self._build_index()
        self._client = TypeSafeClient()

    def _build_index(self) -> tuple[BM25Index, list[str], list[str]]:
        paths = discover_documents(self._corpus_dir)
        passages = load_corpus(paths)
        # A document that contributes no passages (unreadable, blank, OCR
        # found nothing) would otherwise be silently unqueryable.
        with_text = {p.source for p in passages}
        empty = [p.name for p in paths if p.name not in with_text]
        return BM25Index(passages), [p.name for p in paths], empty

    def refresh(self) -> int:
        """Re-scan the corpus directory and rebuild the search index --
        call after adding/removing files. Returns the new passage count.
        Safe alongside in-flight requests: ask() reads its own reference to
        self._index once, and this only ever replaces that reference, never
        mutates the object in place."""
        with self._lock:
            self._index, self._documents, self._empty_documents = self._build_index()
            return len(self._index)

    @property
    def corpus_dir(self) -> Path:
        return self._corpus_dir

    @property
    def documents(self) -> list[str]:
        return list(self._documents)

    @property
    def empty_documents(self) -> list[str]:
        """Documents that were found but yielded no searchable text."""
        return list(self._empty_documents)

    def __len__(self) -> int:
        return len(self._index)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ExtractiveRag":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def ask(self, query: str) -> AnswerResult:
        index = self._index  # snapshot: unaffected by a concurrent refresh()
        corpus_size = len(index)
        k = corpus_size if corpus_size <= config.FULL_SCAN_MAX_PASSAGES else config.SHORTLIST_SIZE
        shortlist = index.shortlist(query, k)
        if not shortlist:
            return AnswerResult(verdict="not_found", confidence=1.0, excerpts=[])

        judged = judge_candidates(self._client, query, shortlist, config.MAX_WORKERS)
        gated = [
            j
            for j in judged
            if j.relevant >= config.THRESHOLDS["relevant_min"]
            and j.usable >= config.THRESHOLDS["usable_min"]
        ]
        gated.sort(key=lambda j: j.combined, reverse=True)
        kept = _diversify(gated, config.MAX_EXCERPTS, config.MAX_PER_CONTEXT, config.STRONG_MIN)

        confidence = check_answerable(self._client, query, [j.passage for j in kept])
        if confidence >= config.ANSWERED_MIN:
            verdict = "answered"
        elif confidence >= config.PARTIAL_MIN:
            verdict = "partial"
        else:
            verdict = "not_found"

        return AnswerResult(verdict=verdict, confidence=confidence, excerpts=kept)
