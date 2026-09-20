"""FastAPI app: a stateless API plus the static page.

Nothing is stored server-side. Each browser keeps its own library (IndexedDB)
and sends the passages with every question; `/api/ingest` turns an uploaded
file into passages and forgets it. That is what makes this safe to run on a
serverless host, where memory and disk don't persist and instances don't share
state: there is no index to keep consistent and no corpus to lose.

Every route except /api/config requires the access token (see auth.py), because
each question spends TypeSafe and Gemini quota.
"""

import tempfile
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field, model_validator

from . import config, ingest
from .auth import AccessDep, auth_required
from .claims import GenerationUnavailable, is_gemini_configured
from .ingest import Passage
from .pipeline import ClaimVerified, Engine, EvidenceFound, Finished, Status

load_dotenv()

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    app.state.engine = Engine()
    yield
    await app.state.engine.aclose()


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


EngineDep = Annotated[Engine, Depends(get_engine)]

# Only /api/config is public (the page needs it to know whether to ask for a
# token). Everything else, including anything that spends quota, is guarded at
# the router level.
public = APIRouter(prefix="/api", tags=["config"])
router = APIRouter(prefix="/api", tags=["qa"], dependencies=[AccessDep])


class PassageModel(BaseModel):
    id: str = Field(min_length=1, max_length=300)
    source: str = Field(max_length=300)
    page: int | None = None
    text: str = Field(max_length=config.MAX_PASSAGE_CHARS)
    is_bullet: bool = False
    context: str = Field(default="", max_length=config.MAX_PASSAGE_CHARS)

    def to_passage(self) -> Passage:
        return Passage(
            id=self.id,
            source=self.source,
            page=self.page,
            text=self.text,
            is_bullet=self.is_bullet,
            context=self.context,
        )

    @classmethod
    def from_passage(cls, passage: Passage) -> "PassageModel":
        return cls(
            id=passage.id,
            source=passage.source,
            page=passage.page,
            text=passage.text,
            is_bullet=passage.is_bullet,
            context=passage.context,
        )


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    passages: list[PassageModel] = Field(max_length=config.MAX_REQUEST_PASSAGES)

    @model_validator(mode="after")
    def _unique_ids(self) -> "AskRequest":
        ids = [p.id for p in self.passages]
        if len(ids) != len(set(ids)):
            raise ValueError("passage ids must be unique")
        return self

    def to_passages(self) -> list[Passage]:
        return [p.to_passage() for p in self.passages]


class ConfigResponse(BaseModel):
    auth_required: bool
    gemini_configured: bool
    max_upload_bytes: int
    max_request_passages: int


class CheckResponse(BaseModel):
    ok: bool


class IngestResponse(BaseModel):
    document: str
    passages: list[PassageModel]  # empty means no searchable text could be read


class ExcerptOut(BaseModel):
    text: str
    source: str
    relevant: float
    usable: float


class AskResponse(BaseModel):
    query: str
    verdict: Literal["answered", "partial", "not_found"]
    confidence: float
    excerpts: list[ExcerptOut]


class SourceOut(BaseModel):
    number: int
    text: str
    source: str
    citation: str  # e.g. "brochure.pdf p.2"


class ClaimOut(BaseModel):
    statement: str
    source_number: int  # which entry in `sources` supports it
    quote: str


class WithheldOut(BaseModel):
    statement: str
    reason: str


class AnswerResponse(BaseModel):
    query: str
    verdict: Literal["answered", "partial", "not_found"]
    confidence: float
    answer: str  # only verified statements, each followed by its [source number]
    claims: list[ClaimOut]
    sources: list[SourceOut]
    withheld: list[WithheldOut]


class StatusEvent(BaseModel):
    message: str


class EvidenceEvent(BaseModel):
    verdict: Literal["answered", "partial", "not_found"]
    confidence: float
    sources: list[SourceOut]


class DoneEvent(BaseModel):
    answer: str
    claim_count: int
    withheld_reasons: list[str]  # reasons only: unverified text is never sent to the browser


class ErrorEvent(BaseModel):
    message: str


@public.get("/config")
def get_config() -> ConfigResponse:
    return ConfigResponse(
        auth_required=auth_required(),
        gemini_configured=is_gemini_configured(),
        max_upload_bytes=config.MAX_UPLOAD_BYTES,
        max_request_passages=config.MAX_REQUEST_PASSAGES,
    )


@router.get("/check")
def check() -> CheckResponse:
    """Lets the page validate a token without spending anything."""
    return CheckResponse(ok=True)


@router.post("/ingest")
def ingest_document(
    file: Annotated[UploadFile, File(description="A PDF, .txt, or .md file to turn into passages")],
) -> IngestResponse:
    """Parse (and, for scans, OCR) one file into passages and return them. The
    file is held only for the duration of the request; the browser keeps the
    result. Blocking work, so this is a plain `def` (run in a worker thread)."""
    name = Path(file.filename or "").name  # strip any path components
    if not name or Path(name).suffix.lower() not in ingest.SUPPORTED_SUFFIXES:
        accepted = ", ".join(sorted(ingest.SUPPORTED_SUFFIXES))
        raise HTTPException(
            status_code=400, detail=f"Unsupported file: {file.filename!r}. Accepted types: {accepted}."
        )
    contents = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(contents) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{name} is larger than the {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(contents)
        passages = ingest.load_corpus([path])
    return IngestResponse(document=name, passages=[PassageModel.from_passage(p) for p in passages])


@router.post("/ask")
async def ask(request: AskRequest, engine: EngineDep) -> AskResponse:
    result = await engine.ask(request.query, request.to_passages())
    return AskResponse(
        query=request.query,
        verdict=result.verdict,
        confidence=result.confidence,
        excerpts=[
            ExcerptOut(
                text=j.passage.display_text,
                source=j.passage.source,
                relevant=j.relevant,
                usable=j.usable,
            )
            for j in result.excerpts
        ],
    )


@router.post("/answer")
async def answer(request: AskRequest, engine: EngineDep) -> AnswerResponse:
    try:
        result = await engine.answer(request.query, request.to_passages())
    except GenerationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # TypeSafe/Gemini/network failures: report, don't leak a traceback
        raise HTTPException(status_code=502, detail=f"Answer failed: {exc}") from exc

    return AnswerResponse(
        query=request.query,
        verdict=result.verdict,
        confidence=result.confidence,
        answer=result.text,
        claims=[
            ClaimOut(statement=c.statement, source_number=c.passage_number, quote=c.quote)
            for c in result.claims
        ],
        sources=[
            SourceOut(
                number=n,
                text=j.passage.display_text,
                source=j.passage.source,
                citation=j.passage.citation,
            )
            for n, j in enumerate(result.evidence, start=1)
        ],
        withheld=[WithheldOut(statement=w.statement, reason=w.reason) for w in result.withheld],
    )


@router.post("/answer/stream", response_class=EventSourceResponse)
async def stream_answer(request: AskRequest, engine: EngineDep) -> AsyncIterable[ServerSentEvent]:
    """Server-Sent Events: progress, then the evidence, then each verified
    statement, then a final summary. Gemini's raw output is never streamed;
    it isn't shown until it has been verified."""
    try:
        async for event in engine.answer_stream(request.query, request.to_passages()):
            match event:
                case Status(message=message):
                    yield ServerSentEvent(event="status", data=StatusEvent(message=message))
                case EvidenceFound(verdict=verdict, confidence=confidence, evidence=evidence):
                    sources = [
                        SourceOut(
                            number=n,
                            text=j.passage.display_text,
                            source=j.passage.source,
                            citation=j.passage.citation,
                        )
                        for n, j in enumerate(evidence, start=1)
                    ]
                    yield ServerSentEvent(
                        event="evidence",
                        data=EvidenceEvent(verdict=verdict, confidence=confidence, sources=sources),
                    )
                case ClaimVerified(claim=claim):
                    yield ServerSentEvent(
                        event="claim",
                        data=ClaimOut(
                            statement=claim.statement,
                            source_number=claim.passage_number,
                            quote=claim.quote,
                        ),
                    )
                case Finished(answer=final):
                    yield ServerSentEvent(
                        event="done",
                        data=DoneEvent(
                            answer=final.text,
                            claim_count=len(final.claims),
                            withheld_reasons=[w.reason for w in final.withheld],
                        ),
                    )
    except GenerationUnavailable as exc:
        yield ServerSentEvent(event="error", data=ErrorEvent(message=str(exc)))
    except Exception as exc:  # the response has started; report it in-band
        yield ServerSentEvent(event="error", data=ErrorEvent(message=f"Answer failed: {exc}"))


app = FastAPI(title="verbatim", lifespan=lifespan)
app.include_router(public)
app.include_router(router)
app.frontend("/", directory=_WEB_DIR)
