"""The contract between the Gemini writer (generate.py) and the verifier
(verify.py). Kept free of heavy imports so the verifier, API and CLI can use
it without loading Google ADK."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class GenerationUnavailable(RuntimeError):
    """Raised when no Gemini credentials are configured."""


def is_gemini_configured() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


class ClaimDraft(BaseModel):
    statement: str = Field(description="One self-contained sentence supported by a single passage.")
    passage: int = Field(description="The number of the evidence passage this statement comes from.")
    quote: str = Field(
        description="An exact, contiguous quote copied verbatim from that passage that supports the statement."
    )


class Draft(BaseModel):
    claims: list[ClaimDraft]
