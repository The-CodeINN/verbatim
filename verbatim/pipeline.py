"""Orchestrates the grounded RAG pipeline:

    BM25 shortlist -> TypeSafe rerank + gate -> existence check -> excerpts

The engine is stateless: every call is given the passages to search, so it
holds no documents, index, or files between calls. That is what lets it run on
serverless hosts (where memory and disk don't persist and instances don't
share state) with the browser -- or, for the CLI, a local folder -- as the
library.

ask() stops at the excerpts: the answer is the source text itself, selected
and ranked, never synthesized. answer() goes one step further -- Gemini writes
cited statements from those excerpts, and every statement is verified against
its source (see verify.py) before it is returned.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from asyncer import asyncify
from typesafe_sdk import AsyncTypeSafeClient

from . import config
from .ingest import Passage
from .judge import JudgedPassage, check_answerable, judge_candidates
from .retrieval import BM25Index
from .verify import VerifiedClaim, WithheldClaim, verify_claims

if TYPE_CHECKING:
    from .generate import ClaimWriter


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


@dataclass(frozen=True)
class GroundedAnswer:
    """A written answer in which every statement passed verification.

    `claims` holds only verified statements; `withheld` records what Gemini
    proposed but failed a check. `evidence` is the numbered excerpt list the
    claims cite, always shown so the answer can be checked by eye."""

    verdict: str  # "answered" | "partial" | "not_found"
    confidence: float
    evidence: list[JudgedPassage]
    claims: list[VerifiedClaim]
    withheld: list[WithheldClaim]

    @property
    def text(self) -> str:
        return " ".join(f"{c.statement} [{c.passage_number}]" for c in self.claims)


@dataclass(frozen=True)
class Status:
    """Progress note for the UI ("Judging 90 passages…")."""

    message: str


@dataclass(frozen=True)
class EvidenceFound:
    """The numbered excerpts any claims will cite, plus the answerability verdict."""

    verdict: str
    confidence: float
    evidence: list[JudgedPassage]


@dataclass(frozen=True)
class ClaimVerified:
    claim: VerifiedClaim


@dataclass(frozen=True)
class Finished:
    answer: GroundedAnswer


AnswerEvent = Status | EvidenceFound | ClaimVerified | Finished


class Engine:
    """Answers questions over whatever passages it is handed."""

    def __init__(self) -> None:
        self._client = AsyncTypeSafeClient()
        self._writer: ClaimWriter | None = None  # built on first use; see _get_writer

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._writer is not None:
            await self._writer.aclose()

    async def __aenter__(self) -> "Engine":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _get_writer(self) -> "ClaimWriter":
        """The Gemini/ADK writer is created on first use, so extractive-only
        use never pays for (or needs credentials for) Google ADK. There is no
        await between the check and the assignment, so this can't race."""
        if self._writer is None:
            from .generate import ClaimWriter

            self._writer = ClaimWriter()
        return self._writer

    @staticmethod
    def _shortlist(query: str, passages: Sequence[Passage]) -> list[Passage]:
        """Small corpora are judged in full; BM25 only narrows large ones."""
        if len(passages) <= config.FULL_SCAN_MAX_PASSAGES:
            return list(passages)
        return BM25Index(list(passages)).shortlist(query, config.SHORTLIST_SIZE)

    async def _rank(self, query: str, shortlist: list[Passage]) -> list[JudgedPassage]:
        judged = await judge_candidates(self._client, query, shortlist, config.MAX_WORKERS)
        gated = [
            j
            for j in judged
            if j.relevant >= config.THRESHOLDS["relevant_min"]
            and j.usable >= config.THRESHOLDS["usable_min"]
        ]
        gated.sort(key=lambda j: j.combined, reverse=True)
        return _diversify(gated, config.MAX_EXCERPTS, config.MAX_PER_CONTEXT, config.STRONG_MIN)

    async def _assess(self, query: str, kept: list[JudgedPassage]) -> AnswerResult:
        confidence = await check_answerable(self._client, query, [j.passage for j in kept])
        if confidence >= config.ANSWERED_MIN:
            verdict = "answered"
        elif confidence >= config.PARTIAL_MIN:
            verdict = "partial"
        else:
            verdict = "not_found"
        return AnswerResult(verdict=verdict, confidence=confidence, excerpts=kept)

    async def ask(self, query: str, passages: Sequence[Passage]) -> AnswerResult:
        shortlist = await asyncify(self._shortlist)(query, passages)
        if not shortlist:
            return AnswerResult(verdict="not_found", confidence=1.0, excerpts=[])
        return await self._assess(query, await self._rank(query, shortlist))

    async def answer_stream(
        self, query: str, passages: Sequence[Passage]
    ) -> AsyncIterator[AnswerEvent]:
        """Retrieve and gate evidence exactly as ask() does, then have Gemini
        write from that evidence alone and verify every statement -- yielding
        progress as it goes. Only verified claims are ever yielded as claims;
        Gemini's raw output is never streamed, because it is unverified until
        verify.py has checked it. When the evidence doesn't answer the
        question, Gemini is never called."""
        yield Status("Searching your documents…")
        shortlist = await asyncify(self._shortlist)(query, passages)
        if not shortlist:
            result = AnswerResult(verdict="not_found", confidence=1.0, excerpts=[])
        else:
            yield Status(f"Judging {len(shortlist)} passages with TypeSafe…")
            kept = await self._rank(query, shortlist)
            yield Status("Checking whether the evidence answers the question…")
            result = await self._assess(query, kept)

        yield EvidenceFound(result.verdict, result.confidence, result.excerpts)
        if result.verdict == "not_found" or not result.excerpts:
            yield Finished(GroundedAnswer(result.verdict, result.confidence, result.excerpts, [], []))
            return

        evidence = [j.passage for j in result.excerpts]
        writer = self._get_writer()
        yield Status("Gemini is drafting cited statements…")
        drafts = await writer.draft(query, evidence)
        yield Status(f"Verifying {len(drafts)} statement(s) against your documents…")
        claims, withheld = await verify_claims(self._client, drafts, evidence)

        for claim in claims:
            yield ClaimVerified(claim)
        yield Finished(GroundedAnswer(result.verdict, result.confidence, result.excerpts, claims, withheld))

    async def answer(self, query: str, passages: Sequence[Passage]) -> GroundedAnswer:
        """Non-streaming form of answer_stream(): just the final result."""
        final: GroundedAnswer | None = None
        async for event in self.answer_stream(query, passages):
            if isinstance(event, Finished):
                final = event.answer
        assert final is not None  # answer_stream always ends with Finished
        return final
