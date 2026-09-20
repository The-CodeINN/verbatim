# verbatim

Ask questions about your own documents and get back the exact passages that
answer them — verbatim, never paraphrased. Built with
[TypeSafe](https://docs.typesafe.ai) typed judgments instead of a generative
LLM, so there is nothing to hallucinate.

Works on PDFs (including scanned ones, via OCR), `.txt`, and Markdown, from a
CLI or a small web UI.

## Why this isn't a normal RAG app

TypeSafe's System One models (Jev) don't generate text — they only return typed
`Choice`/`Score`/`Noul` judgments over state you give them. So there's no
"stuff retrieved chunks into a prompt and let the model write an answer" step.
Instead:

1. **Ingest** ([verbatim/ingest.py](verbatim/ingest.py)) — every PDF,
   `.txt`, and `.md` file in `corpus/` is split into stable, addressable
   passages, tagged with the section/heading they belong to and which
   document they came from.
2. **BM25 shortlist** ([verbatim/retrieval.py](verbatim/retrieval.py)) —
   cheap keyword search narrows the corpus to ~20 candidates per query.
3. **TypeSafe rerank + gate** ([verbatim/judge.py](verbatim/judge.py)) —
   each candidate is scored independently with `Noul` questions (relevant?
   usable?) and thresholded, mirroring the
   [rerank](https://docs.typesafe.ai/cookbooks/rerank_typesafe.md) and
   [classifying_rag_passages](https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md)
   cookbooks.
4. **Existence check** — a final `Noul` asks whether the surviving evidence
   actually answers the query, so the app can say "not found" instead of
   confidently returning irrelevant excerpts (see
   [semantic_find](https://docs.typesafe.ai/cookbooks/semantic_find.md)).
5. **Answer** ([verbatim/pipeline.py](verbatim/pipeline.py)) — the
   ranked excerpts *are* the answer. Nothing is synthesized, so nothing can
   be hallucinated.

## Setup

```sh
uv sync
```

Set `TYPESAFE_API_KEY` in `.env` (get one at the
[TypeSafe console](https://console.typesafe.ai/)).

Drop any PDF, `.txt`, or `.md` file into `corpus/` (or add it from the web
UI). Point at a different folder with the `VERBATIM_CORPUS_DIR` env var
instead of editing code.

Scanned PDFs (and "Print to PDF" output, which has no text layer) are OCR'd
automatically with [RapidOCR](https://github.com/RapidAI/RapidOCR) — no
Tesseract or Poppler install needed. It's slow (roughly 15–20 s per page on
CPU), so results are cached in `.cache/ocr/` by file content and each page is
only OCR'd once. OCR'd text can be missing spaces and reading order on
complex layouts is approximate; a document that yields no text at all is
flagged in the UI.

## Usage

**CLI:**

```sh
uv run main.py "How long does the program last?"
uv run main.py                      # interactive mode
```

**Web UI** ([verbatim/app.py](verbatim/app.py), [web/index.html](web/index.html)):

```sh
uv run fastapi dev verbatim/app.py
```

Open `http://127.0.0.1:8000` to ask questions and add/remove documents from
the browser — uploads land in `corpus/` and the index rebuilds immediately,
no restart needed. The same rules apply: no generation, ever — see
`GET /api/ask`, `GET/POST /api/documents`, and `DELETE /api/documents/{filename}`
in [verbatim/app.py](verbatim/app.py) for the full API.

## Tuning

Retrieval size, gate thresholds, and the answered/partial/not-found cutoffs
all live in [verbatim/config.py](verbatim/config.py) — policy changes
don't require touching the questions or the retrieval code.
