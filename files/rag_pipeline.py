"""HW 3: answer an LA visit question from retrieved, cited evidence.

Read ``la_notes.py`` and ``answer_model.py`` before editing this file. The
supplied ``answer_question`` function at the bottom connects your TODOs:

    prepare chunks -> hybrid search -> model input -> model -> verify

The model is the only step that may produce untrusted text. Every other step
is ordinary Python that the tests can check exactly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypedDict

from la_notes import (
    DOCUMENTS,
    EVAL_QUESTIONS,
    Chunk,
    SourceDocument,
    embed,
    keyword_search,
    vector_norm,
)
from answer_model import AnswerGenerator, GroundedAnswer


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
# Return one ``Chunk`` for each ``## `` section of every document whose
# ``status`` is ``"verified"``. Skip the altered snapshot entirely so its
# 8 p.m. closing time can never reach the model.
#
# A heading line looks like ``## C2: Griffith Observatory paid parking``.
# Use the text before the colon as ``chunk_id`` and the text after it as
# ``section``. Copy ``document``, ``snapshot``, and ``source`` from the
# document.
#
# Build ``text`` as the section title, a newline, and the section's non-empty
# body lines stripped and joined with single spaces. For example:
#
#   "Griffith Observatory Saturday building hours\n"
#   "Saturday building hours: 10 a.m. to 10 p.m."
#
# The body alone never says "Griffith Observatory". Keeping the heading lets
# a retrieved chunk be understood without the rest of the document.
def prepare_chunks(documents: list[SourceDocument]) -> list[Chunk]:
    raise NotImplementedError("TODO 1: follow the instructions above")


# TODO 2: Rank chunks by cosine similarity.
#
# ``cosine_similarity`` returns (a · b) / (‖a‖ × ‖b‖). Use the supplied
# ``vector_norm``. Return 0.0 when either vector has length zero.
#
# ``vector_search`` embeds the query and every chunk's ``text`` with the
# supplied ``embed`` function. Return the IDs of the top ``k`` chunks, highest
# score first. Break ties by collection order, and leave out chunks whose
# score is 0.0. This matches the supplied ``keyword_search``.
def cosine_similarity(a: list[float], b: list[float]) -> float:
    raise NotImplementedError("TODO 2: follow the instructions above")


def vector_search(query: str, chunks: list[Chunk], k: int) -> list[str]:
    raise NotImplementedError("TODO 2: follow the instructions above")


# TODO 3: Combine keyword and vector candidates.
#
# Return the keyword IDs followed by the vector IDs, keeping only the first
# occurrence of each ID. Session 07's example: keyword [C1, C3] and vector
# [C2, C1] give [C1, C3, C2]. The order records where each candidate first
# appeared. It is not a shared relevance score.
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
# Return a JSON string (use ``json.dumps(..., indent=2)``) with two keys:
#
#   "question":       the visitor's question
#   "evidence_cards": one object per card, in the given order, with the keys
#                     chunk_id, document, section, snapshot, source, and text
#
# The cards carry the original chunk text, not a summary. Retrieved text goes
# only inside "evidence_cards". It never joins the model instructions, which
# live in ``answer_model.MODEL_INSTRUCTIONS``.
def build_model_input(question: str, cards: list[Chunk]) -> str:
    raise NotImplementedError("TODO 4: follow the instructions above")


def normalize(text: str) -> str:
    """Lowercase ``text`` and collapse whitespace for quote matching."""

    return " ".join(text.lower().split())


# TODO 5: Check every claim against the cards that reached the model.
#
# A citation is *useful* when the claim's normalized quote is non-empty, the
# citation names one of ``cards``, and that card's normalized text contains
# the quote. Compare with the supplied ``normalize`` on both sides. Every
# other citation is *unused*, including IDs that were never retrieved. A claim
# is *supported* when its quote is not blank and at least one citation is
# useful.
#
#   citation_recall    = supported claims / all claims
#   citation_precision = useful citations / all citations
#
# Use 0.0 for either rate when its denominator is zero. Accept the answer only
# when it has at least one claim and every claim is supported. Unused
# citations lower precision but do not block acceptance.
#
# Slide 48's draft has 4 claims and 5 citations: recall 3/4, precision 3/5.
def verify_answer(answer: GroundedAnswer, cards: list[Chunk]) -> CitationReport:
    raise NotImplementedError("TODO 5: follow the instructions above")


# TODO 6: Measure retrieval with recall@k.
#
# ``recall_at_k`` returns the fraction of ``required_ids`` that appear in the
# first ``k`` entries of ``ranked_ids``. Session 07's example: ranking
# [C1, C3, C2] with required {C1, C2} gives 0.5 at k=2 and 1.0 at k=3.
#
# ``evaluate_retrieval`` runs ``search(question, chunks, k)`` for every entry
# in ``EVAL_QUESTIONS`` and returns the mean recall@k. Score the full list
# that ``search`` returns (passing ``len(ranked)`` as the cutoff to
# ``recall_at_k``), so hybrid search is credited for every candidate it passes
# on.
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
