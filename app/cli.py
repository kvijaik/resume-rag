"""Simple command-line interface for the Resume RAG app.

Usage:
    python -m app.cli "Have requirement for Java Full stack developer, provide matching resume"
"""

import sys  # standard library: read command-line arguments (sys.argv), used for the sys.path tweak below
from pathlib import Path  # standard library: filesystem paths

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add the project root to sys.path so `from app import ...` resolves when this file is run directly
from app.rag_chain import answer_query  # noqa: E402  # the single entry point that runs guardrails -> retrieval -> generation


def main():
    if len(sys.argv) > 1:  # a query was passed as command-line arguments -> run once and exit (non-interactive mode)
        query = " ".join(sys.argv[1:])  # rejoin all args into a single query string (handles unquoted multi-word input too)
        result = answer_query(query)  # run the full RAG pipeline for this one query
        print(f"\nQuery: {query}")  # echo the query back for clarity
        print(f"Status: {result['status']}")  # "unclear" | "no_match" | "matched"
        print(f"Sources: {result['sources']}")  # resume filenames the answer was grounded in (empty unless matched)
        print(f"\nAnswer:\n{result['answer']}\n")  # the generated (or template) answer text
        return  # done; skip the interactive loop below

    print("Resume RAG CLI. Type a hiring requirement, or 'quit' to exit.\n")  # no CLI args given -> fall into interactive REPL mode
    while True:  # loop until the user asks to quit
        query = input("> ").strip()  # prompt for a query and trim surrounding whitespace
        if query.lower() in {"quit", "exit"}:  # allow either word (case-insensitive) to end the session
            break
        result = answer_query(query)  # run the full RAG pipeline for this query
        print(f"\n[{result['status']}] {result['answer']}\n")  # compact one-block status + answer output for interactive use


if __name__ == "__main__":  # only run when executed directly (`python -m app.cli ...`), not on import
    main()
