"""FastAPI app: a small UI in front of the extractive RAG pipeline.

The pipeline (corpus ingest + BM25 index + TypeSafe client) is built once at
startup via the lifespan and reused across requests; `ask()` is blocking
(sync TypeSafe client, thread pool for the fan-out), so its route stays a
plain `def` and FastAPI runs it in its own worker thread. Document upload
and deletion write to the corpus directory and rebuild the index in place.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from . import ingest
from .pipeline import ExtractiveRag

load_dotenv()

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # generous for a single document


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    app.state.rag = ExtractiveRag()
    yield
    app.state.rag.close()


def get_rag(request: Request) -> ExtractiveRag:
    return request.app.state.rag


RagDep = Annotated[ExtractiveRag, Depends(get_rag)]

router = APIRouter(prefix="/api", tags=["qa"])


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


class DocumentsResponse(BaseModel):
    documents: list[str]
    passage_count: int
    empty_documents: list[str]  # found, but no searchable text could be read


def _documents_response(rag: ExtractiveRag, passage_count: int) -> DocumentsResponse:
    return DocumentsResponse(
        documents=rag.documents,
        passage_count=passage_count,
        empty_documents=rag.empty_documents,
    )


@router.get("/ask")
def ask(q: Annotated[str, Query(min_length=1, max_length=300)], rag: RagDep) -> AskResponse:
    result = rag.ask(q)
    return AskResponse(
        query=q,
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


@router.get("/documents")
def list_documents(rag: RagDep) -> DocumentsResponse:
    return _documents_response(rag, len(rag))


@router.post("/documents")
def upload_documents(
    files: Annotated[list[UploadFile], File(description="PDF, .txt, or .md files to ingest")],
    rag: RagDep,
) -> DocumentsResponse:
    accepted = ", ".join(sorted(ingest.SUPPORTED_SUFFIXES))
    for upload in files:
        name = Path(upload.filename or "").name  # strip any path components
        if not name or Path(name).suffix.lower() not in ingest.SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file: {upload.filename!r}. Accepted types: {accepted}.",
            )
        contents = upload.file.read()
        if len(contents) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"{name} exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
            )
        rag.corpus_dir.mkdir(parents=True, exist_ok=True)
        (rag.corpus_dir / name).write_bytes(contents)

    count = rag.refresh()
    return _documents_response(rag, count)


@router.delete("/documents/{filename}")
def delete_document(filename: str, rag: RagDep) -> DocumentsResponse:
    safe_name = Path(filename).name
    target = rag.corpus_dir / safe_name
    if safe_name != filename or not target.is_file():
        raise HTTPException(status_code=404, detail=f"No such document: {filename!r}")
    target.unlink()

    count = rag.refresh()
    return _documents_response(rag, count)


app = FastAPI(title="verbatim", lifespan=lifespan)
app.include_router(router)
app.frontend("/", directory=_WEB_DIR)
