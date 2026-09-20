"""Ask questions about the documents in corpus/, grounded entirely in their
own text.

By default every answer is a ranked excerpt straight from the source
document(s), selected by TypeSafe's typed judgments over a BM25 shortlist.
Drop any PDF/.txt/.md file into corpus/ (or point VERBATIM_CORPUS_DIR at
another folder) and it's queryable. See verbatim/pipeline.py for the full
flow, or run verbatim/app.py for a web UI that can add/remove documents live.

Usage:
    uv run main.py "How long does the program last?"
    uv run main.py --answer "How long does the program last?"   # written answer
    uv run main.py                      # interactive mode
    uv run main.py --answer             # interactive, written answers

--answer has Gemini (via Google ADK) write from the retrieved excerpts, then
shows only the statements that pass verification; needs GOOGLE_API_KEY.
"""

import asyncio
import sys

from dotenv import load_dotenv

from verbatim import config
from verbatim.claims import GenerationUnavailable
from verbatim.ingest import Passage, discover_documents, load_corpus
from verbatim.pipeline import AnswerResult, Engine, GroundedAnswer


def _print_result(query: str, result: AnswerResult) -> None:
    print(f"\nQ: {query}")
    print(f"Verdict: {result.verdict} (confidence {result.confidence:.2f})")
    if not result.excerpts:
        print("  No relevant passages were found.")
        return
    for rank, judged in enumerate(result.excerpts, start=1):
        print(
            f"  [{rank}] (relevant={judged.relevant:.2f} usable={judged.usable:.2f}) "
            f"{judged.passage.display_text}"
        )


def _print_answer(query: str, result: GroundedAnswer) -> None:
    print(f"\nQ: {query}")
    print(f"Verdict: {result.verdict} (confidence {result.confidence:.2f})")
    if result.verdict == "not_found" or not result.evidence:
        print("  Not found in your documents.")
        return
    if result.claims:
        print(f"\n  {result.text}\n")
    else:
        print("  No statement could be verified against the sources; excerpts only.")
    for n, judged in enumerate(result.evidence, start=1):
        print(f"  [{n}] {judged.passage.citation}: {judged.passage.display_text}")
    if result.withheld:
        print(f"  ({len(result.withheld)} unverified statement(s) withheld)")


async def _run(engine: Engine, passages: list[Passage], query: str, grounded: bool) -> None:
    if not grounded:
        _print_result(query, await engine.ask(query, passages))
        return
    try:
        _print_answer(query, await engine.answer(query, passages))
    except GenerationUnavailable as exc:
        print(f"\n{exc}")


async def main() -> None:
    load_dotenv()
    args = sys.argv[1:]
    grounded = "--answer" in args
    queries = [a for a in args if a != "--answer"]

    # The engine is stateless; the CLI's "library" is just the corpus folder,
    # ingested once up front and handed to every question.
    paths = discover_documents(config.CORPUS_DIR)
    passages = load_corpus(paths)
    with_text = {p.source for p in passages}
    for path in paths:
        if path.name not in with_text:
            print(f"Warning: no searchable text could be read from {path.name}")

    # One event loop for the whole session: the async TypeSafe client and the
    # ADK runner are long-lived and must not be shared across loops.
    async with Engine() as engine:
        if queries:
            for query in queries:
                await _run(engine, passages, query, grounded)
            return

        names = ", ".join(p.name for p in paths) or "(no documents in corpus/)"
        print(f"Ask questions about {names} (empty line to quit).")
        while True:
            try:
                query = (await asyncio.to_thread(input, "> ")).strip()
            except EOFError:
                break
            if not query:
                break
            await _run(engine, passages, query, grounded)


if __name__ == "__main__":
    asyncio.run(main())
