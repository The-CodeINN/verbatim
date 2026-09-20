"""Ask questions about the documents in corpus/, grounded entirely in their
own text.

No generative model is involved: every answer is a ranked excerpt straight
from the source document(s), selected by TypeSafe's typed judgments over a
BM25 shortlist. Drop any PDF/.txt/.md file into corpus/ (or point
VERBATIM_CORPUS_DIR at another folder) and it's queryable. See
verbatim/pipeline.py for the full flow, or run verbatim/app.py for
a web UI that can add/remove documents live.

Usage:
    uv run main.py "How long does the program last?"
    uv run main.py                      # interactive mode
"""

import sys

from dotenv import load_dotenv

from verbatim.pipeline import AnswerResult, ExtractiveRag


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


def main() -> None:
    load_dotenv()
    with ExtractiveRag() as rag:
        queries = sys.argv[1:]
        if queries:
            for query in queries:
                _print_result(query, rag.ask(query))
            return

        print(f"Ask questions about {', '.join(rag.documents) or '(no documents loaded)'} (empty line to quit).")
        while True:
            try:
                query = input("> ").strip()
            except EOFError:
                break
            if not query:
                break
            _print_result(query, rag.ask(query))


if __name__ == "__main__":
    main()
