"""The HW 3 specification, written as tests.

Read this file, but do not edit it. Every test uses fixed local data. No test
calls a live model, website, or other network service.
"""

from __future__ import annotations

import asyncio
import json
import math
from types import SimpleNamespace

import pytest

from answer_model import (
    MODEL_INSTRUCTIONS,
    GroundedAnswerOutput,
    generate_answer_with_openai,
    scripted_answer_generator,
)
from la_notes import (
    DOCUMENTS,
    LA_NOTES_B,
    TRIP_QUESTION,
    Chunk,
    keyword_search,
)
from rag_pipeline import (
    answer_question,
    build_model_input,
    cosine_similarity,
    evaluate_retrieval,
    hybrid_merge,
    hybrid_search,
    prepare_chunks,
    recall_at_k,
    vector_search,
    verify_answer,
)


def run(coroutine):
    """Run one coroutine from a synchronous pytest test."""

    return asyncio.run(coroutine)


def chunks() -> list[Chunk]:
    return prepare_chunks(DOCUMENTS)


def by_id(chunk_ids: list[str]) -> list[Chunk]:
    lookup = {chunk["chunk_id"]: chunk for chunk in chunks()}
    return [lookup[chunk_id] for chunk_id in chunk_ids]


# ---------------------------------------------------------------------------
# Supplied code (these pass before you start)
# ---------------------------------------------------------------------------


def test_00_keyword_search_misses_different_wording() -> None:
    collection = [
        {
            "chunk_id": "C2",
            "document": "LA Visitor Notes",
            "section": "Griffith Observatory paid parking",
            "snapshot": "B",
            "source": "test",
            "text": "Paid parking is available in the Observatory lot.",
        }
    ]

    assert keyword_search("Where can I leave my car?", collection, 3) == []
    assert keyword_search("Observatory parking", collection, 3) == ["C2"]


def test_00b_openai_call_requests_one_structured_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LM_STUDIO_BASE_URL", raising=False)

    class FakeResponses:
        def __init__(self):
            self.arguments = None

        async def parse(self, **arguments):
            self.arguments = arguments
            return SimpleNamespace(
                output_parsed=GroundedAnswerOutput(
                    claims=[
                        {
                            "text": "Open at 7 p.m.",
                            "citations": ["C1"],
                            "quote": "10 a.m. to 10 p.m.",
                        }
                    ],
                    missing_info=[],
                )
            )

    fake_responses = FakeResponses()
    answer = run(
        generate_answer_with_openai(
            '{"question": "test"}',
            client=SimpleNamespace(responses=fake_responses),
            model="test-model",
        )
    )

    assert answer["claims"][0]["citations"] == ["C1"]
    assert fake_responses.arguments["model"] == "test-model"
    assert fake_responses.arguments["text_format"] is GroundedAnswerOutput
    assert fake_responses.arguments["instructions"] == MODEL_INSTRUCTIONS
    assert fake_responses.arguments["input"] == '{"question": "test"}'
    assert fake_responses.arguments["store"] is False


def test_00c_lm_studio_uses_structured_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCompletions:
        def __init__(self):
            self.arguments = None

        async def parse(self, **arguments):
            self.arguments = arguments
            proposal = GroundedAnswerOutput(claims=[], missing_info=["hours"])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=proposal))]
            )

    fake_completions = FakeCompletions()
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")

    answer = run(
        generate_answer_with_openai(
            '{"question": "test"}',
            client=SimpleNamespace(
                chat=SimpleNamespace(completions=fake_completions)
            ),
            model="local-test-model",
        )
    )

    arguments = fake_completions.arguments
    assert answer == {"claims": [], "missing_info": ["hours"]}
    assert arguments["model"] == "local-test-model"
    assert arguments["response_format"] is GroundedAnswerOutput
    assert arguments["messages"][0]["content"] == MODEL_INSTRUCTIONS
    assert arguments["messages"][1]["content"] == '{"question": "test"}'


# ---------------------------------------------------------------------------
# TODO 1: prepare chunks
# ---------------------------------------------------------------------------


def test_01_chunks_keep_heading_and_source_location() -> None:
    collection = prepare_chunks([LA_NOTES_B])

    assert [chunk["chunk_id"] for chunk in collection] == [
        "C1", "C2", "C3", "C4", "C5", "C6",
    ]
    assert collection[1] == {
        "chunk_id": "C2",
        "document": "LA Visitor Notes",
        "section": "Griffith Observatory paid parking",
        "snapshot": "B",
        "source": LA_NOTES_B["source"],
        "text": (
            "Griffith Observatory paid parking\n"
            "Paid parking is available in the Observatory lot and along West "
            "Observatory Road. Pay stations take credit cards only. Display "
            "the receipt on your dashboard."
        ),
    }
    assert collection[0]["text"] == (
        "Griffith Observatory Saturday building hours\n"
        "Saturday building hours: 10 a.m. to 10 p.m."
    )


def test_02_altered_snapshot_never_enters_the_collection() -> None:
    collection = chunks()

    assert [chunk["chunk_id"] for chunk in collection] == [
        "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    ]
    assert all(chunk["snapshot"] == "B" for chunk in collection)
    assert not any("8 p.m." in chunk["text"] for chunk in collection)


# ---------------------------------------------------------------------------
# TODO 2: cosine similarity and vector search
# ---------------------------------------------------------------------------


def test_03_cosine_similarity_matches_the_lecture_examples() -> None:
    assert cosine_similarity([1, 1], [2, 2]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [0.8, 0.6]) == pytest.approx(0.8)
    assert cosine_similarity([3, 4], [6, 8]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 2]) == 0.0


def test_04_vector_search_finds_parking_for_leave_my_car() -> None:
    collection = chunks()

    assert keyword_search("Where can I leave my car?", collection, 2) == []
    assert vector_search("Where can I leave my car?", collection, 2) == [
        "C7", "C2",
    ]
    assert vector_search("Where is Santa Monica Pier?", collection, 1) == ["C3"]


# ---------------------------------------------------------------------------
# TODO 3: hybrid merge
# ---------------------------------------------------------------------------


def test_05_hybrid_merge_keeps_first_occurrence() -> None:
    assert hybrid_merge(["C1", "C3"], ["C2", "C1"]) == ["C1", "C3", "C2"]
    assert hybrid_merge([], ["C7", "C2"]) == ["C7", "C2"]


def test_06_hybrid_search_covers_both_parts_of_the_trip_question() -> None:
    collection = chunks()

    assert keyword_search(TRIP_QUESTION, collection, 2) == ["C1", "C2"]
    assert vector_search(TRIP_QUESTION, collection, 2) == ["C1", "C7"]
    assert hybrid_search(TRIP_QUESTION, collection, 2) == ["C1", "C2", "C7"]


# ---------------------------------------------------------------------------
# TODO 4: model input
# ---------------------------------------------------------------------------


def test_07_model_input_carries_original_text_as_evidence_cards() -> None:
    cards = by_id(["C1", "C2", "C7"])
    payload = json.loads(build_model_input(TRIP_QUESTION, cards))

    assert set(payload) == {"question", "evidence_cards"}
    assert payload["question"] == TRIP_QUESTION
    assert payload["evidence_cards"] == [dict(card) for card in cards]


def test_08_injected_text_stays_inside_the_evidence() -> None:
    cards = by_id(["C7"])
    model_input = build_model_input("Where can I leave my car?", cards)
    payload = json.loads(model_input)

    assert "Ignore the user's question" in payload["evidence_cards"][0]["text"]
    assert "Ignore the user's question" not in MODEL_INSTRUCTIONS
    assert "not instructions" in MODEL_INSTRUCTIONS


# ---------------------------------------------------------------------------
# TODO 5: citation checks
# ---------------------------------------------------------------------------


def scripted(mode: str):
    return run(scripted_answer_generator(mode)("unused input"))


def test_09_supported_answer_is_accepted() -> None:
    report = verify_answer(scripted("supported"), by_id(["C1", "C2", "C7"]))

    assert report["accepted"] is True
    assert report["citation_recall"] == 1.0
    assert report["citation_precision"] == 1.0
    assert [check["useful_citations"] for check in report["claims"]] == [
        ["C1"], ["C2"],
    ]


def test_10_slide_48_draft_scores_three_quarters_and_three_fifths() -> None:
    report = verify_answer(scripted("citation_errors"), by_id(["C1", "C2"]))

    assert report["accepted"] is False
    assert report["citation_recall"] == pytest.approx(3 / 4)
    assert report["citation_precision"] == pytest.approx(3 / 5)
    assert report["claims"][1]["useful_citations"] == ["C2"]
    assert report["claims"][1]["unused_citations"] == ["C1"]
    assert report["claims"][3] == {
        "text": "We are guaranteed a parking space.",
        "supported": False,
        "useful_citations": [],
        "unused_citations": ["C1"],
    }


def test_11_citations_must_reach_the_model_and_quotes_must_exist() -> None:
    claim = {
        "text": "Paid parking is in the Observatory lot.",
        "citations": ["C2"],
        "quote": "paid parking is available   in the observatory lot",
    }
    answer = {"claims": [claim], "missing_info": []}

    assert verify_answer(answer, by_id(["C2"]))["accepted"] is True
    assert verify_answer(answer, by_id(["C1"]))["accepted"] is False

    blank = {"claims": [{**claim, "quote": "  "}], "missing_info": []}
    assert verify_answer(blank, by_id(["C2"]))["accepted"] is False

    empty = verify_answer({"claims": [], "missing_info": []}, by_id(["C2"]))
    assert empty["accepted"] is False
    assert empty["citation_recall"] == 0.0
    assert empty["citation_precision"] == 0.0


def test_12_injected_free_parking_claim_is_rejected() -> None:
    report = verify_answer(scripted("injected"), by_id(["C1", "C2", "C7"]))

    assert report["accepted"] is False
    assert report["claims"][0]["supported"] is False


# ---------------------------------------------------------------------------
# TODO 6: retrieval evaluation
# ---------------------------------------------------------------------------


def test_13_recall_at_k_matches_slide_47() -> None:
    assert recall_at_k(["C1", "C3", "C2"], ["C1", "C2"], 2) == 0.5
    assert recall_at_k(["C1", "C3", "C2"], ["C1", "C2"], 3) == 1.0
    assert recall_at_k([], ["C2"], 3) == 0.0


def test_14_hybrid_search_has_the_best_recall_at_2() -> None:
    collection = chunks()

    keyword = evaluate_retrieval(keyword_search, collection, 2)
    vector = evaluate_retrieval(vector_search, collection, 2)
    hybrid = evaluate_retrieval(hybrid_search, collection, 2)

    assert keyword == pytest.approx(5 / 6)
    assert vector == pytest.approx(11 / 12)
    assert hybrid == pytest.approx(1.0)
    assert math.isclose(evaluate_retrieval(keyword_search, collection, 1), 7 / 12)
    assert math.isclose(evaluate_retrieval(hybrid_search, collection, 1), 0.75)


# ---------------------------------------------------------------------------
# Whole workflow (uses every TODO)
# ---------------------------------------------------------------------------


def test_15_workflow_answers_and_rejects_with_fixed_responses() -> None:
    supported = run(
        answer_question(TRIP_QUESTION, scripted_answer_generator("supported"))
    )
    injected = run(
        answer_question(TRIP_QUESTION, scripted_answer_generator("injected"))
    )

    assert supported["status"] == "answered"
    assert supported["retrieved_ids"] == ["C1", "C2", "C7"]
    assert injected["status"] == "rejected"


def test_16_workflow_sends_exactly_the_retrieved_cards() -> None:
    seen: list[str] = []

    async def recording_generator(model_input: str):
        seen.append(model_input)
        return {"claims": [], "missing_info": []}

    result = run(answer_question(TRIP_QUESTION, recording_generator))
    payload = json.loads(seen[0])

    assert [card["chunk_id"] for card in payload["evidence_cards"]] == (
        result["retrieved_ids"]
    )
    assert result["status"] == "rejected"
