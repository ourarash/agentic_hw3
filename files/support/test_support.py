"""Tests for the supplied code in ``support/``. These pass before you start."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from support.answer_model import (
    MODEL_INSTRUCTIONS,
    GroundedAnswerOutput,
    generate_answer_with_openai,
)
from support.la_notes import keyword_search


def run(coroutine):
    """Run one coroutine from a synchronous pytest test."""

    return asyncio.run(coroutine)


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
