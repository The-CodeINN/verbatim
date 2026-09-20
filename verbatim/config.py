"""Tunable policy for the extractive RAG pipeline.

Keeping thresholds here means the gate can be retuned without touching the
questions or the retrieval code (see the classifying_rag_passages cookbook).
"""

import os
from pathlib import Path

# --- corpus: every PDF/.txt/.md file under this directory is ingested.
# Override with the VERBATIM_CORPUS_DIR env var to point at a different
# folder without touching code.
CORPUS_DIR = Path(
    os.environ.get("VERBATIM_CORPUS_DIR")
    or Path(__file__).resolve().parent.parent / "corpus"
)

# --- OCR results for scanned PDFs are cached here (keyed by file content hash)
# so the slow OCR pass runs once per distinct file, not on every re-index.
OCR_CACHE_DIR = Path(
    os.environ.get("VERBATIM_OCR_CACHE_DIR")
    or Path(__file__).resolve().parent.parent / ".cache" / "ocr"
)

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

# --- existence check over the surviving evidence (see semantic_find cookbook) ---
ANSWERED_MIN = 0.70  # evidence clearly answers the query
PARTIAL_MIN = 0.35  # evidence partially answers it; below this is "not found"
