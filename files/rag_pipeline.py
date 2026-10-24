"""HW 3: answer an LA visit question from retrieved, cited evidence.

This is the only code file you edit. Each TODO below shows its input, the
expected output, and the supplied helper to call, so you do not need to open
``support/``. The supplied ``answer_question`` function at the bottom connects
your TODOs:

    prepare chunks -> hybrid search -> model input -> model -> verify

The model is the only step that may produce untrusted text. Every other step
is ordinary Python that the tests can check exactly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypedDict

from support.la_notes import (
    DOCUMENTS,
    EVAL_QUESTIONS,
    Chunk,
    SourceDocument,
    embed,
    keyword_search,
    vector_norm,
)
from support.answer_model import AnswerGenerator, GroundedAnswer


class ClaimCheck(TypedDict):
    text: str
    supported: bool
    useful_citations: list[str]
    unused_citations: list[str]


class CitationReport(TypedDict):
    accepted: bool
    claims: list[ClaimCheck]
    citation_recall: float
    citation_precision: float


Search = Callable[[str, list[Chunk], int], list[str]]


# TODO 1: Prepare the chunk collection.
#
# Input: a list of documents shaped like this (``body`` is Markdown):
#
#   {"document": "LA Visitor Notes", "snapshot": "B", "status": "verified",
#    "source": "Griffith Observatory, ...",
#    "body": "## C1: Griffith Observatory Saturday building hours\n"
#            "Saturday building hours: 10 a.m. to 10 p.m.\n"
#            "\n"
#            "## C2: Griffith Observatory paid parking\n"
#            "Paid parking is available in the Observatory lot and along West\n"
#            "Observatory Road. ...\n"}
#
# Output: one ``Chunk`` dict per ``## `` section, in order, for documents whose
# ``status`` is ``"verified"``. Skip the ``"altered"`` snapshot entirely so its
# wrong 8 p.m. closing time never reaches the model. For the C1 section above:
#
#   {"chunk_id": "C1",                       # text before the colon
#    "section": "Griffith Observatory Saturday building hours",  # after it
#    "document": "LA Visitor Notes", "snapshot": "B", "source": "...",
#    "text": "Griffith Observatory Saturday building hours\n"
#            "Saturday building hours: 10 a.m. to 10 p.m."}
#
# ``text`` is the section title, a newline, then the section's non-empty lines
# stripped and joined with single spaces. The body alone never says "Griffith
# Observatory", so keeping the heading lets a chunk stand on its own.
def prepare_chunks(documents: list[SourceDocument]) -> list[Chunk]:
    raise NotImplementedError("TODO 1: follow the instructions above")


# TODO 2: Rank chunks by cosine similarity.
#
# ``cosine_similarity(a, b)`` returns (a · b) / (‖a‖ × ‖b‖), or 0.0 when either
# vector has length zero. ``vector_norm(v)`` is supplied.
#
#   cosine_similarity([1, 0], [0.8, 0.6])  ->  0.8
#   cosine_similarity([3, 4], [6, 8])      ->  1.0
#
# ``vector_search(query, chunks, k)`` turns text into a vector with the
# supplied ``embed(text) -> list[float]``. Embed the query and each chunk's
# ``"text"``, score every chunk, and return the ``chunk_id`` values of the top
# ``k``: highest score first, ties in collection order, zero scores left out.
#
#   vector_search("Where can I leave my car?", chunks, 2)  ->  ["C7", "C2"]
def cosine_similarity(a: list[float], b: list[float]) -> float:
    raise NotImplementedError("TODO 2: follow the instructions above")


def vector_search(query: str, chunks: list[Chunk], k: int) -> list[str]:
    raise NotImplementedError("TODO 2: follow the instructions above")


# TODO 3: Combine keyword and vector candidates.
#
# Return the keyword IDs followed by the vector IDs, keeping only the first
# occurrence of each ID.
#
#   hybrid_merge(["C1", "C3"], ["C2", "C1"])  ->  ["C1", "C3", "C2"]
#
# The order records where each candidate first appeared. It is not a shared
# relevance score.
def hybrid_merge(keyword_ids: list[str], vector_ids: list[str]) -> list[str]:
    raise NotImplementedError("TODO 3: follow the instructions above")


def hybrid_search(query: str, chunks: list[Chunk], k: int) -> list[str]:
    """Run both searches with the same ``k`` and merge their candidates."""

    return hybrid_merge(
        keyword_search(query, chunks, k),
        vector_search(query, chunks, k),
    )


# TODO 4: Build the model input.
#
# ``cards`` is a list of chunks from TODO 1. Return
# ``json.dumps(payload, indent=2)`` where ``payload`` is:
#
#   {"question": question,
#    "evidence_cards": [
#        {"chunk_id": ..., "document": ..., "section": ...,
#         "snapshot": ..., "source": ..., "text": ...},   # one per card,
#        ...                                               # in given order
#    ]}
#
# Keep the original chunk text, not a summary. Retrieved text goes only inside
# "evidence_cards"; the model instructions are supplied separately.
def build_model_input(question: str, cards: list[Chunk]) -> str:
    raise NotImplementedError("TODO 4: follow the instructions above")


def normalize(text: str) -> str:
    """Lowercase ``text`` and collapse whitespace for quote matching."""

    return " ".join(text.lower().split())


# TODO 5: Check every claim against the cards that reached the model.
#
# ``answer`` comes from the model and looks like this:
#
#   {"claims": [{"text": "Pay stations take credit cards only.",
#                "citations": ["C2"],
#                "quote": "Pay stations take credit cards only."}, ...],
#    "missing_info": [...]}
#
# A citation is *useful* when the claim's quote is not blank, the citation
# names one of ``cards``, and that card's text contains the quote. Compare
# ``normalize(quote) in normalize(card["text"])`` (``normalize`` is just
# above). Every other citation is *unused*. A claim is *supported* when it has
# at least one useful citation.
#
# Return:
#
#   {"accepted": True only if there is at least one claim and all are supported,
#    "claims": [{"text": ..., "supported": ...,
#                "useful_citations": [...], "unused_citations": [...]}, ...],
#    "citation_recall": supported claims / all claims,         # 0.0 if none
#    "citation_precision": useful citations / all citations}   # 0.0 if none
#
# Slide 48's draft has 4 claims and 5 citations: recall 3/4, precision 3/5.
def verify_answer(answer: GroundedAnswer, cards: list[Chunk]) -> CitationReport:
    raise NotImplementedError("TODO 5: follow the instructions above")


# TODO 6: Measure retrieval with recall@k.
#
# ``recall_at_k(ranked_ids, required_ids, k)`` returns the fraction of
# ``required_ids`` found in the first ``k`` entries of ``ranked_ids``.
#
#   recall_at_k(["C1", "C3", "C2"], ["C1", "C2"], 2)  ->  0.5
#   recall_at_k(["C1", "C3", "C2"], ["C1", "C2"], 3)  ->  1.0
#
# ``evaluate_retrieval(search, chunks, k)`` loops over the supplied
# ``EVAL_QUESTIONS``, a list like:
#
#   [{"question": "Where can I leave my car?", "required_ids": ["C2"]}, ...]
#
# For each one, call ``ranked = search(question, chunks, k)`` and score it with
# ``recall_at_k(ranked, required_ids, len(ranked))`` so hybrid search gets
# credit for every candidate it returns. Return the mean score.
def recall_at_k(ranked_ids: list[str], required_ids: list[str], k: int) -> float:
    raise NotImplementedError("TODO 6: follow the instructions above")


def evaluate_retrieval(search: Search, chunks: list[Chunk], k: int) -> float:
    raise NotImplementedError("TODO 6: follow the instructions above")


# ---------------------------------------------------------------------------
# Supplied workflow. Do not edit below this line.
# ---------------------------------------------------------------------------


async def answer_question(
    question: str,
    answer_generator: AnswerGenerator,
    *,
    k: int = 2,
) -> dict[str, object]:
    """Retrieve evidence, ask the model, and check its citations."""

    chunks = prepare_chunks(DOCUMENTS)
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    retrieved_ids = hybrid_search(question, chunks, k)
    cards = [by_id[chunk_id] for chunk_id in retrieved_ids]

    model_input = build_model_input(question, cards)
    answer = await answer_generator(model_input)
    report = verify_answer(answer, cards)

    return {
        "status": "answered" if report["accepted"] else "rejected",
        "question": question,
        "retrieved_ids": retrieved_ids,
        "answer": answer,
        "report": report,
    }
