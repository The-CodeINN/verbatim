"""Tunable policy for the extractive RAG pipeline.

Keeping thresholds here means the gate can be retuned without touching the
questions or the retrieval code (see the classifying_rag_passages cookbook).
"""

import os
import tempfile
from pathlib import Path

# --- local corpus (CLI only): every PDF/.txt/.md file under this directory is
# ingested. Override with VERBATIM_CORPUS_DIR. The web app doesn't use it: there,
# each browser keeps its own library and sends the passages with each question.
CORPUS_DIR = Path(
    os.environ.get("VERBATIM_CORPUS_DIR")
    or Path(__file__).resolve().parent.parent / "corpus"
)

# --- OCR results for scanned PDFs are cached here (keyed by file content hash)
# so the slow OCR pass runs once per distinct file. On Vercel the deployment is
# read-only and only /tmp is writable (and not shared between instances), so
# the cache is best-effort there; the browser keeps the durable copy.
_ON_VERCEL = bool(os.environ.get("VERCEL"))
OCR_CACHE_DIR = Path(
    os.environ.get("VERBATIM_OCR_CACHE_DIR")
    or (
        Path(tempfile.gettempdir()) / "verbatim-ocr"
        if _ON_VERCEL
        else Path(__file__).resolve().parent.parent / ".cache" / "ocr"
    )
)

# --- web API limits. Every question sends its passages in the request body, and
# each passage costs TypeSafe calls, so these bound both payload and spend.
MAX_REQUEST_PASSAGES = 500
MAX_PASSAGE_CHARS = 4000
# Vercel Functions reject request bodies over 4.5 MB, so stay safely under it.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024

# --- retrieval ---
SHORTLIST_SIZE = 20  # BM25 candidates handed to TypeSafe for judging
# Below this corpus size, BM25 pre-filtering isn't worth its recall cost --
# judge every passage instead. BM25 shortlisting is a scale optimization
# (the rerank cookbook uses it over thousands of documents); a small corpus
# is cheap enough to score in full, and full-scan sidesteps BM25's blind
# spot for queries with no lexical overlap with the answer (e.g. an
# employer's name never appearing in the words "what companies worked at").
FULL_SCAN_MAX_PASSAGES = 100
MAX_EXCERPTS = 5  # excerpts shown in the final answer
MAX_PER_CONTEXT = 2  # cap per section, so one section's bullets can't crowd out the rest
STRONG_MIN = 0.70  # combined score to guarantee a context its own excerpt slot;
# below this, a passage only just cleared the gate and shouldn't out-compete
# a genuinely on-topic passage from another section for a scarce slot.
MAX_WORKERS = 8  # concurrent TypeSafe calls

# --- gate: drop candidates that don't clear both bars ---
THRESHOLDS = {
    "relevant_min": 0.45,  # below this, the passage doesn't address the query
    "usable_min": 0.40,  # below this, it's on-topic but not a usable fact
}

# --- generative answers (Gemini via Google ADK), verified before display ---
# Deliberately the cheapest tier: Gemini only turns a few already-vetted excerpts
# into short cited sentences, and verify.py rejects anything unsupported, so a
# weaker model costs recall (more withheld statements), never accuracy.
GEMINI_MODEL = os.environ.get("VERBATIM_GEMINI_MODEL") or "gemini-3.5-flash-lite"
MAX_CLAIMS = 8  # cap on statements Gemini may draft per answer
MIN_QUOTE_CHARS = 12  # a supporting quote shorter than this proves nothing
# A claim is shown only if TypeSafe says its cited passage "supports" it with at
# least this confidence (the citation_check cookbook auto-accepts at 0.8).
VERIFY_MIN = 0.80

# --- existence check over the surviving evidence (see semantic_find cookbook) ---
ANSWERED_MIN = 0.70  # evidence clearly answers the query
PARTIAL_MIN = 0.35  # evidence partially answers it; below this is "not found"
