"""Show retrieval, the model input, and citation checks for the LA visit."""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from support.answer_model import generate_answer_with_openai, scripted_answer_generator
from support.la_notes import DOCUMENTS, TRIP_QUESTION, keyword_search
from rag_pipeline import (
    answer_question,
    evaluate_retrieval,
    hybrid_search,
    prepare_chunks,
    vector_search,
)


def print_retrieval(question: str, k: int) -> None:
    """Print what each search returns for one question."""

    chunks = prepare_chunks(DOCUMENTS)
    print(f"Question: {question}")
    print(f"  keyword (k={k}): {keyword_search(question, chunks, k)}")
    print(f"  vector  (k={k}): {vector_search(question, chunks, k)}")
    print(f"  hybrid  (k={k}): {hybrid_search(question, chunks, k)}")


def print_evaluation() -> None:
    """Print mean recall@k over the fixed evaluation questions."""

    chunks = prepare_chunks(DOCUMENTS)
    print("Mean recall@k over EVAL_QUESTIONS")
    print("  search    k=1    k=2    k=3")
    for name, search in [
        ("keyword", keyword_search),
        ("vector", vector_search),
        ("hybrid", hybrid_search),
    ]:
        scores = [evaluate_retrieval(search, chunks, k) for k in (1, 2, 3)]
        print(f"  {name:<8}" + "".join(f"{score:7.2f}" for score in scores))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="Ask a live model once instead of using fixed responses.",
    )
    parser.add_argument(
        "--question",
        default=TRIP_QUESTION,
        help="The visitor question to answer.",
    )
    args = parser.parse_args()

    if args.live:
        lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL")
        if lm_studio_base_url and not os.getenv("OPENAI_MODEL"):
            parser.error(
                "Set OPENAI_MODEL to the model identifier loaded in LM Studio."
            )
        if not lm_studio_base_url and not os.getenv("OPENAI_API_KEY"):
            parser.error("Set OPENAI_API_KEY before using --live.")
        runs = [("live", generate_answer_with_openai)]
    else:
        if args.question != TRIP_QUESTION:
            parser.error("Fixed responses only answer the trip question.")
        runs = [
            (mode, scripted_answer_generator(mode))
            for mode in ("supported", "citation_errors", "injected")
        ]

    print_retrieval(args.question, k=2)
    print_retrieval("Where can I leave my car?", k=2)
    print()
    print_evaluation()

    for label, generator in runs:
        result = await answer_question(args.question, generator)
        print(f"\n--- Answer: {label} ---")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
