"""TypeSafe judgments over BM25 candidates: pairwise rerank, a quality gate,
and an existence check -- the pattern shared by the rerank,
classifying_rag_passages, and semantic_find cookbooks.

No text is generated here. Every call returns a probability (a `Noul`) that
code then thresholds and sorts; TypeSafe never sees the whole corpus at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import anyio
from asyncer import create_task_group
from typesafe_sdk import AsyncTypeSafeClient, Noul

from .ingest import Passage

_RELEVANT = Noul(
    instructions=(
        "The candidate is one passage from a document corpus. Does it address "
        "what the query is actually asking about? If the query names a "
        "specific entity, project, section, or document, the passage must be "
        "about that same one, not just similar content found elsewhere in "
        "the corpus. If the query asks several things at once, a passage "
        "that addresses any one of them counts."
    ),
    criteria={
        "true": (
            "The passage matches the topic, entity, or fact the query asks "
            "about -- including any specific entity/section/document the "
            "query names."
        ),
        "false": (
            "The passage is from the corpus but about something unrelated to "
            "the query, or about a different entity/section/document than "
            "the one the query names."
        ),
    },
)

_USABLE = Noul(
    instructions=(
        "Could this passage, on its own, help answer the query -- does it "
        "contain a concrete fact (a name, date, number, or specific detail) "
        "rather than just a bare heading or label?"
    ),
    criteria={
        "true": "The passage contains a specific fact usable in an answer.",
        "false": "The passage is a bare heading/label, or too vague to use.",
    },
)

_ANSWERED = Noul(
    instructions=(
        "Given only these evidence passages -- not the rest of the corpus -- "
        "do they answer the query?"
    ),
    criteria={
        "true": "The evidence passages directly answer the query.",
        "false": "The evidence passages do not answer the query.",
    },
)


@dataclass(frozen=True)
class JudgedPassage:
    passage: Passage
    relevant: float
    usable: float

    @property
    def combined(self) -> float:
        return (self.relevant + self.usable) / 2


async def judge_candidates(
    client: AsyncTypeSafeClient,
    query: str,
    candidates: list[Passage],
    max_concurrency: int,
) -> list[JudgedPassage]:
    """Score every candidate against the query concurrently, one TypeSafe
    request per pair -- each question is about exactly one passage. A semaphore
    bounds how many requests are in flight at once."""
    limit = anyio.Semaphore(max_concurrency)

    async def judge_one(passage: Passage) -> JudgedPassage:
        async with limit:
            response = await client.system_one(
                state={
                    "query": query,
                    "passage": {
                        "source": passage.source,
                        "section_context": passage.context,
                        "text": passage.text,
                    },
                },
                questions={"relevant": _RELEVANT, "usable": _USABLE},
            )
        return JudgedPassage(
            passage=passage,
            relevant=response.answers["relevant"].noul,
            usable=response.answers["usable"].noul,
        )

    async with create_task_group() as tg:
        results = [tg.soonify(judge_one)(p) for p in candidates]
    return [r.value for r in results]


async def check_answerable(client: AsyncTypeSafeClient, query: str, evidence: list[Passage]) -> float:
    """Probability that the surviving evidence -- and only that evidence --
    answers the query. This is what lets the app say "not found" instead of
    confidently returning irrelevant excerpts."""
    if not evidence:
        return 0.0
    response = await client.system_one(
        state={
            "query": query,
            "evidence": [{"source": p.source, "text": p.display_text} for p in evidence],
        },
        questions={"answered": _ANSWERED},
    )
    return response.answers["answered"].noul
