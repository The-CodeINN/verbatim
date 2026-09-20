"""OCR fallback for PDF pages that have no extractable text.

Scans, and PDFs produced by "Print to PDF" (which turns text into vector
outlines), have no text layer for pypdf to read. For those pages we render
the whole page to an image with pdfium and OCR it with RapidOCR (both pip
wheels -- no Poppler or Tesseract install needed). The heavy imports are
deferred so text-only corpora never pay for them.

OCR is slow (roughly 15-20s per page on CPU), and the index is rebuilt on
every upload/delete, so results are cached on disk keyed by the file's
content hash: each page is OCR'd once per distinct file.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from . import config

_RENDER_DPI = 200
# Boxes whose left edges are within this fraction of the page width belong to
# the same text column; a bigger jump starts a new column.
_COLUMN_GAP_FRACTION = 0.06

_engine: Any = None
_engine_lock = threading.Lock()


def _get_engine() -> Any:
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR()
        return _engine


def _reading_order(result: list[Any], page_width: float) -> list[str]:
    """OCR returns boxes in detection order. On a multi-column page, sorting
    by y alone interleaves the columns, so group boxes into columns by left
    edge, then read each column top to bottom, left column first. Approximate
    for irregular layouts (sidebars, callouts), good for ordinary columns."""
    items = [
        (min(p[0] for p in box), min(p[1] for p in box), text.strip())
        for box, text, _confidence in result
        if text.strip()
    ]
    items.sort()
    max_gap = page_width * _COLUMN_GAP_FRACTION

    columns: list[list[tuple[float, float, str]]] = []
    for item in items:
        if columns and item[0] - columns[-1][-1][0] <= max_gap:
            columns[-1].append(item)
        else:
            columns.append([item])

    return [text for column in columns for _x, _y, text in sorted(column, key=lambda i: i[1])]


def _ocr_page(pdf: Any, page_number: int) -> list[str]:
    import numpy as np

    image = pdf[page_number].render(scale=_RENDER_DPI / 72).to_pil().convert("RGB")
    result, _elapsed = _get_engine()(np.array(image))
    return _reading_order(result or [], image.width)


def ocr_pages(path: Path, page_numbers: list[int]) -> dict[int, list[str]]:
    """OCR the given (0-indexed) pages of a PDF, returning their text lines in
    reading order. Cached per file content, so repeat calls are instant."""
    if not page_numbers:
        return {}

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache_file = config.OCR_CACHE_DIR / f"{digest}.json"
    try:
        cached: dict[str, list[str]] = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cached = {}

    missing = [n for n in page_numbers if str(n) not in cached]
    if missing:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            for n in missing:
                cached[str(n)] = _ocr_page(pdf, n)
        finally:
            pdf.close()
        # The cache only saves time; on a read-only or ephemeral filesystem a
        # failed write must not throw away the OCR we just paid for.
        try:
            config.OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    return {n: cached[str(n)] for n in page_numbers}
