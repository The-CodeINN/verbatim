"""Turn a folder of documents into stable, addressable passages.

Supports PDF (via pypdf, with an OCR fallback for pages that have no text
layer -- see ocr.py) and plain text/Markdown files. Both are reduced to the
same shape before chunking: a sequence of (page, line) pairs, where page is
None for non-paginated formats. From there, one shared heuristic handles
both:

- pypdf preserves each visual line as its own text line but doesn't mark
  paragraph or bullet boundaries, so a line that wraps mid-sentence comes
  back as its own line starting with a lowercase letter. Plain text/Markdown
  files can be hard-wrapped the same way. Either way, a bullet or an
  uppercase-led line starts a new passage; anything else is a continuation
  of the previous line.
- A bullet on its own doesn't say what section/heading it belongs to, so
  each bullet is tagged with the short non-bullet lines that most recently
  preceded it (job title + company in a resume, a heading in a document).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from . import config
from .ocr import ocr_pages

logger = logging.getLogger(__name__)

_PDF_SUFFIX = ".pdf"
_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = _TEXT_SUFFIXES | {_PDF_SUFFIX}

# U+FFFD is accepted defensively: some PDFs' bullet glyphs decode to it.
# Bullet glyphs (including the middle dot OCR often reads them as) may sit flush
# against the text ("·Reservoir engineer"); "-", "*" and "1." need a space so
# "-5 degrees" or "2024" aren't mistaken for list items.
_BULLET_RE = re.compile(r"^(?:[•·●◦�]\s*|[*-]\s+|\d+[.)]\s+)")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+")
_FIRST_ALPHA_RE = re.compile(r"[A-Za-z]")
_MIN_PASSAGE_CHARS = 8
_LABEL_MAX_CHARS = 40

# A bullet/paragraph is tagged with the short lines (heading, job title,
# company) that most recently preceded it -- a rolling window, not a full
# reset, so it self-corrects a line or two into the next section without
# needing to detect section boundaries explicitly.
_HEADER_MAX_CHARS = 90
_HEADER_WINDOW = 2


@dataclass(frozen=True)
class Passage:
    id: str
    source: str  # the file this passage came from, e.g. "resume.pdf"
    page: int | None  # None for non-paginated formats (.txt/.md)
    text: str
    is_bullet: bool
    context: str = ""  # the heading/section this passage belongs to, if any

    @property
    def display_text(self) -> str:
        return f"{self.context} — {self.text}" if self.context else self.text

    @property
    def citation(self) -> str:
        return f"{self.source} p.{self.page + 1}" if self.page is not None else self.source


def discover_documents(corpus_dir: Path) -> list[Path]:
    """All supported files under corpus_dir, recursively."""
    if not corpus_dir.is_dir():
        return []
    return sorted(
        p for p in corpus_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _starts_new_passage(line: str) -> bool:
    """A bullet or an uppercase-led line starts a new passage; anything
    else is a soft-wrapped continuation of the previous line."""
    if _BULLET_RE.match(line):
        return True
    match = _FIRST_ALPHA_RE.search(line)
    # No letters at all (e.g. a stray "35%." fragment) is virtually always
    # the tail of a wrapped sentence, never a new heading -- default to
    # continuation rather than starting a new passage.
    return match is not None and match.group().isupper()


def _clean(line: str) -> tuple[str, bool]:
    is_bullet = bool(_BULLET_RE.match(line))
    text = _BULLET_RE.sub("", line) if is_bullet else line
    return text.strip(), is_bullet


def _iter_pdf_lines(path: Path) -> Iterator[tuple[int | None, str]]:
    reader = PdfReader(str(path))
    page_texts = [page.extract_text() or "" for page in reader.pages]
    # Pages with no text layer (scans, "Print to PDF" output) are OCR'd; pages
    # that do have text keep the fast, exact path.
    ocr_lines = ocr_pages(path, [i for i, text in enumerate(page_texts) if not text.strip()])
    for page_number, text in enumerate(page_texts):
        for raw_line in ocr_lines.get(page_number, text.splitlines()):
            yield page_number, raw_line


def _iter_text_lines(path: Path) -> Iterator[tuple[int | None, str]]:
    content = path.read_text(encoding="utf-8", errors="replace")
    strip_heading = path.suffix.lower() in (".md", ".markdown")
    for raw_line in content.splitlines():
        yield None, (_MD_HEADING_RE.sub("", raw_line) if strip_heading else raw_line)


def _merge_lines(lines: Iterable[tuple[int | None, str]]) -> list[tuple[int | None, str, bool]]:
    chunks: list[tuple[int | None, str, bool]] = []
    for page, raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        cleaned, is_bullet = _clean(line)
        if not cleaned:
            continue

        on_same_chunk_run = chunks and chunks[-1][0] == page
        # A short line ending in ":" is a label ("Language:", "Duration:") for
        # whatever follows it, so the next line joins it rather than becoming
        # a separate, meaningless passage. Bullets always start a new passage.
        follows_label = (
            on_same_chunk_run
            and chunks[-1][1].endswith(":")
            and len(chunks[-1][1]) <= _LABEL_MAX_CHARS
            and not _BULLET_RE.match(line)
        )
        if on_same_chunk_run and (follows_label or not _starts_new_passage(line)):
            prev_page, prev_text, prev_bullet = chunks[-1]
            chunks[-1] = (prev_page, f"{prev_text} {cleaned}", prev_bullet)
        else:
            chunks.append((page, cleaned, is_bullet))
    return chunks


def _with_header_context(
    chunks: list[tuple[int | None, str, bool]],
) -> list[tuple[int | None, str, bool, str]]:
    window: list[str] = []
    enriched: list[tuple[int | None, str, bool, str]] = []
    for page, text, is_bullet in chunks:
        if is_bullet:
            enriched.append((page, text, True, " — ".join(window)))
        else:
            enriched.append((page, text, False, ""))
            if len(text) <= _HEADER_MAX_CHARS:
                window.append(text)
                del window[:-_HEADER_WINDOW]
    return enriched


def _document_chunks(path: Path) -> list[tuple[int | None, str, bool]]:
    lines = _iter_pdf_lines(path) if path.suffix.lower() == _PDF_SUFFIX else _iter_text_lines(path)
    return _merge_lines(lines)


def _split_long(text: str, limit: int) -> list[str]:
    """Break text longer than `limit` into pieces at word boundaries. A file
    with no paragraph or bullet structure would otherwise become one giant
    passage, which is too big to send to a judge (or to accept over the API)."""
    pieces: list[str] = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:  # no sensible boundary: hard cut
            cut = limit
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces


def load_corpus(paths: Iterable[Path]) -> list[Passage]:
    passages: list[Passage] = []
    for path in paths:
        try:
            chunks = _with_header_context(_document_chunks(path))
        except Exception:
            # One unreadable/corrupt file shouldn't take the whole index down;
            # it just contributes no passages (and is reported as empty).
            logger.exception("Could not ingest %s", path)
            continue
        for i, (page, text, is_bullet, context) in enumerate(chunks):
            if len(text) < _MIN_PASSAGE_CHARS:
                continue
            locator = f"p{page}-{i}" if page is not None else str(i)
            for part, piece in enumerate(_split_long(text, config.MAX_PASSAGE_CHARS)):
                passages.append(
                    Passage(
                        id=f"{path.name}:{locator}" + (f"~{part}" if part else ""),
                        source=path.name,
                        page=page,
                        text=piece,
                        is_bullet=is_bullet,
                        context=context,
                    )
                )
    return passages
