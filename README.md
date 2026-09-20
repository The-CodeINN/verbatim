# verbatim

Ask questions about your own documents and get back the exact passages that
answer them — verbatim. Built with [TypeSafe](https://docs.typesafe.ai) typed
judgments, so the retrieval side has nothing to hallucinate; an optional
written answer (Gemini, via Google ADK) is shown only after every statement in
it has been checked against your documents.

Works on PDFs (including scanned ones, via OCR), `.txt`, and Markdown, from a
CLI or a small web UI.

## How it works

TypeSafe's System One models (Jev) don't generate text — they only return typed
`Choice`/`Score`/`Noul` judgments over state you give them. So retrieval is
selection, not generation:

1. **Ingest** ([verbatim/ingest.py](verbatim/ingest.py)) — each document is split
   into stable, addressable passages, tagged with the section/heading they belong
   to and which document they came from.
2. **BM25 shortlist** ([verbatim/retrieval.py](verbatim/retrieval.py)) —
   cheap keyword search narrows large libraries to ~20 candidates per question
   (libraries under 100 passages are judged in full).
3. **TypeSafe rerank + gate** ([verbatim/judge.py](verbatim/judge.py)) —
   each candidate is scored independently with `Noul` questions (relevant?
   usable?) and thresholded, mirroring the
   [rerank](https://docs.typesafe.ai/cookbooks/rerank_typesafe.md) and
   [classifying_rag_passages](https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md)
   cookbooks.
4. **Existence check** — a final `Noul` asks whether the surviving evidence
   actually answers the question, so the app says "not found" instead of
   confidently returning irrelevant excerpts (see
   [semantic_find](https://docs.typesafe.ai/cookbooks/semantic_find.md)).
5. **Excerpts** ([verbatim/pipeline.py](verbatim/pipeline.py)) — the ranked
   excerpts are the answer. Nothing is synthesized.

### Stateless by design

The engine holds no documents, index, or files between questions: every call is
handed the passages to search. In the web app your library therefore lives in
**your browser** (IndexedDB), and each question sends the passages along with
it. The server keeps nothing, which is what lets it run on serverless hosts
(memory and disk don't persist there, and instances don't share state), and
gives every user a private library for free. The CLI does the same with a local
folder as its library.

Consequences worth knowing:

- The library is per-browser: clearing site data deletes it and it doesn't
  follow you between devices. Use **Export library** / **Import library**.
- Each question sends your passages to the server, TypeSafe, and Gemini.
- A single file is capped at 4 MB (Vercel rejects larger request bodies) and a
  single question can search at most 500 passages.

### Written answers, without trusting the LLM

Optionally, Gemini turns those vetted excerpts into short cited sentences. It
is treated as untrusted: [verify.py](verbatim/verify.py) shows a statement only
if **both** checks pass, following the
[citation_check](https://docs.typesafe.ai/cookbooks/citation_check.md) cookbook:

- **String check** — the exact quote Gemini cites must appear in the passage it
  cites.
- **Semantic check** — a TypeSafe `Choice` (`supports` / `contradicts` /
  `says_nothing`) must say the passage *supports* the statement, at ≥ 0.80
  confidence.

Anything else is withheld. If the evidence doesn't answer the question, Gemini
is never called; if nothing survives verification you get the excerpts alone.

This makes unsupported statements very unlikely to reach you, not impossible:
the semantic check is a calibrated judgment, not a proof, so the numbered source
excerpts are always shown next to the answer. It is deliberately strict, which
costs recall — a fragment like `Language: English` can't support "the program is
taught in English" because it never says what it's about — and that is why the
default model can be the cheapest one: a weak Gemini yields more withheld
statements, not wrong ones.

The Gemini side follows Google ADK's documented patterns for an async server
(one long-lived agent and runner, `run_async`, `output_schema` + `output_key`,
a stateless agent, one session per question deleted afterwards) —
see [verbatim/generate.py](verbatim/generate.py). Only *verified* statements are
streamed to the browser; Gemini's raw tokens never are, since they would appear
before they had been checked.

## Setup

```sh
uv sync
```

Put your settings in `.env`:

| Variable | Needed for |
| --- | --- |
| `TYPESAFE_API_KEY` | everything ([TypeSafe console](https://console.typesafe.ai/)) |
| `GOOGLE_API_KEY` | written answers only ([Google AI Studio](https://aistudio.google.com/app/apikey)) |
| `VERBATIM_ACCESS_TOKEN` | the web API: callers must send it as a bearer token (see below) |

Optional: `VERBATIM_CORPUS_DIR` (the CLI's documents folder, default `corpus/`),
`VERBATIM_GEMINI_MODEL` (default `gemini-3.5-flash-lite`; try a larger
model if too many statements are withheld, or if Google retires this one),
`VERBATIM_OCR_CACHE_DIR`.

Scanned PDFs (and "Print to PDF" output, which has no text layer) are OCR'd
automatically with [RapidOCR](https://github.com/RapidAI/RapidOCR) — no
Tesseract or Poppler install needed. It's slow (on the order of 15–30 s per page
on CPU), so results are cached in `.cache/ocr/` by file content and each page is
only OCR'd once. OCR'd text can be missing spaces and reading order on
complex layouts is approximate; a document that yields no text at all is
flagged.

## Usage

**CLI** — searches the documents in `corpus/`:

```sh
uv run main.py "How long does the program last?"            # ranked excerpts
uv run main.py --answer "How long does the program last?"   # verified written answer
uv run main.py                                              # interactive mode
```

**Web UI** ([verbatim/app.py](verbatim/app.py), [web/index.html](web/index.html)):

```sh
uv run fastapi dev verbatim/app.py
```

Open `http://127.0.0.1:8000`, add documents, and ask. With "Write an answer with
Gemini" on, progress and each verified statement stream in as they're ready;
hover a `[n]` citation to see the exact quote behind it.

API (all but `/api/config` require the access token, if one is set):

| Route | |
| --- | --- |
| `GET /api/config` | public: whether a token is needed, whether Gemini is configured, limits |
| `GET /api/check` | validates a token |
| `POST /api/ingest` | one file in → its passages out (nothing stored) |
| `POST /api/ask` | `{query, passages}` → ranked excerpts |
| `POST /api/answer` | `{query, passages}` → verified written answer (JSON) |
| `POST /api/answer/stream` | the same, as Server-Sent Events |

### Access token

Every question spends your TypeSafe and Gemini quota, so set
`VERBATIM_ACCESS_TOKEN` to a long random value (for example
`python -c "import secrets; print(secrets.token_urlsafe(32))"`). The page asks
for it once and remembers it in that browser. Locally, with no token set, the API
is open. **On Vercel it fails closed**: with no token configured, every API route
returns 503 instead of serving.

### Deploying to Vercel

`[tool.vercel] entrypoint` in `pyproject.toml` points Vercel at the app and
`web/` is served from the CDN. In the Vercel project settings add
`TYPESAFE_API_KEY`, `GOOGLE_API_KEY`, and `VERBATIM_ACCESS_TOKEN`. The server is
stateless, so there is no database or blob store to provision.

Limits to plan around: request bodies over 4.5 MB are rejected (hence the 4 MB
file cap); a function may run 300 s on Hobby, and a scanned PDF spends most of
that on OCR, so only short scans fit in one request (a handful of pages; do
larger scans locally, then use **Import library**); and only `/tmp` is writable,
so the OCR cache there is best-effort.

### Inspecting the Gemini agent (ADK dev UI)

`claim_writer/` exposes the same agent the app uses as `root_agent`, so ADK's dev
UI can find it. From the project root:

```sh
uv run adk web --port 8000
```

Pick `claim_writer` (the `verbatim` entry in the list is the app package, not an
agent). The agent takes JSON rather than free text — paste, for example:

```json
{"question": "How long does the program last?",
 "evidence": [{"number": 1, "source": "brochure.pdf", "section": "", "text": "Duration: 16 months"}]}
```

Its output there is unverified; in the app every statement is checked before it
is shown. ADK Web is for development only.

## Tuning

Retrieval size, gate thresholds, the answered/partial/not-found cutoffs, the
verification threshold, request limits, and the Gemini model all live in
[verbatim/config.py](verbatim/config.py) — policy changes don't require
touching the questions or the retrieval code.
