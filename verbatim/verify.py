"""Verify every drafted claim against the source before it can be shown.

A claim survives only if BOTH checks pass, following the citation_check
cookbook:

1. String check (code, exact): its quote must literally appear in the passage
   it cites. A quote that isn't there means the model invented or altered
   source text.
2. Semantic check (TypeSafe `Choice`): the cited passage must "support" the
   statement -- not contradict it, and not merely say nothing about it -- with
   confidence above config.VERIFY_MIN. This is what catches claims that quote
   real text but assert something the text doesn't say (added facts, outside
   knowledge, overstated conclusions).

Anything else is withheld, with the reason recorded. The result is that no
statement is displayed unless it is tied to real source text and judged
supported; it does not make the judge infallible, which is why the excerpts
themselves are always shown alongside.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import anyio
from asyncer import create_task_group
from typesafe_sdk import AsyncTypeSafeClient, Choice

from . import config
from .claims import ClaimDraft
from .ingest import Passage

_SUPPORT = Choice(
    instructions=(
        "A statement was written from a source passage. How does the passage "
        "relate to the statement? Judge strictly: every fact in the statement "
        "(names, numbers, dates, causes, descriptions, and the overall claim) "
        "must be present in the passage."
    ),
    criteria={
        "supports": (
            "Every fact in the statement is explicitly stated in the passage or "
            "follows from it directly, and the statement adds nothing."
        ),
        "contradicts": "The passage says something that conflicts with the statement.",
        "says_nothing": (
            "The statement includes at least one fact, detail, or conclusion "
            "that the passage does not state, or the passage is unrelated."
        ),
    },
)

# Typographic variants that PDFs, OCR, and models render inconsistently.
_TYPOGRAPHY = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}
)


def _squash(text: str) -> str:
    """Compare on characters, not spacing: PDF extraction and OCR drop or add
    spaces ("Theenergyworld"), so whitespace and case are ignored -- but every
    other character must match."""
    return re.sub(r"\s+", "", text.translate(_TYPOGRAPHY)).casefold()


def quote_in_passage(quote: str, passage: Passage) -> bool:
    needle = _squash(quote)
    return len(needle) >= config.MIN_QUOTE_CHARS and needle in _squash(passage.display_text)


@dataclass(frozen=True)
class VerifiedClaim:
    statement: str
    passage_number: int  # 1-based position in the evidence list
    passage: Passage
    quote: str
    confidence: float


@dataclass(frozen=True)
class WithheldClaim:
    statement: str
    reason: str


async def _verify_one(
    client: AsyncTypeSafeClient,
    draft: ClaimDraft,
    evidence: list[Passage],
    limit: anyio.Semaphore,
) -> VerifiedClaim | WithheldClaim:
    if not 1 <= draft.passage <= len(evidence):
        return WithheldClaim(draft.statement, "cited a passage that doesn't exist")
    passage = evidence[draft.passage - 1]

    if not quote_in_passage(draft.quote, passage):
        return WithheldClaim(draft.statement, "its supporting quote isn't in the cited source text")

    async with limit:
        response = await client.system_one(
            state={
                "statement": draft.statement,
                "source_passage": {
                    "source": passage.source,
                    "section": passage.context,
                    "text": passage.text,
                },
            },
            questions={"support": _SUPPORT},
        )
    answer = response.answers["support"]
    if answer.choice != "supports":
        return WithheldClaim(draft.statement, f"the cited source {answer.choice.replace('_', ' ')}")
    if answer.confidence < config.VERIFY_MIN:
        return WithheldClaim(
            draft.statement, f"support was uncertain ({answer.confidence:.2f} < {config.VERIFY_MIN:.2f})"
        )
    return VerifiedClaim(
        statement=draft.statement,
        passage_number=draft.passage,
        passage=passage,
        quote=draft.quote,
        confidence=answer.confidence,
    )


async def verify_claims(
    client: AsyncTypeSafeClient, drafts: list[ClaimDraft], evidence: list[Passage]
) -> tuple[list[VerifiedClaim], list[WithheldClaim]]:
    """Verify all drafted claims concurrently; results keep draft order."""
    limit = anyio.Semaphore(config.MAX_WORKERS)
    async with create_task_group() as tg:
        outcomes = [tg.soonify(_verify_one)(client, d, evidence, limit) for d in drafts]
    results = [o.value for o in outcomes]
    verified = [r for r in results if isinstance(r, VerifiedClaim)]
    withheld = [r for r in results if isinstance(r, WithheldClaim)]
    return verified, withheld
