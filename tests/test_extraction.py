"""Deterministic tests for the structured-output layer. No API key required.

That last part is the point. The interesting behaviour in `extraction.py` is
what happens when the model gets it *wrong* -- and you cannot reliably provoke
a real model into returning malformed JSON on demand. So the model is faked and
the failures are scripted, which makes the retry path testable, fast, and free.
This is the same seam phase 3 will use to test the agent loop.
"""

import json

import pytest
from pydantic import ValidationError

from reposage.extraction import (
    EXTRACTION_TEMPERATURE,
    ExtractionError,
    IssueSummary,
    extract,
    extract_json_block,
    schema_block,
)

VALID_PAYLOAD = {
    "issue_number": 4212,
    "title": "Dependency override not applied to WebSocket routes",
    "category": "bug",
    "one_line_problem": "app.dependency_overrides was ignored for WebSocket endpoints during tests.",
    "resolution": "Fixed by resolving overrides in the WebSocket dependency solver.",
    "files_mentioned": ["fastapi/routing.py", "fastapi/dependencies/utils.py"],
    "eval_question": "Do dependency overrides apply to WebSocket routes?",
    "eval_answer": "Yes, they are resolved through the same solver as HTTP routes.",
    "answerable_from_repo": True,
    "confidence": 0.9,
}


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeClient:
    """A scripted model: hands back canned replies and records how it was called."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if not self.replies:
            raise AssertionError("FakeClient was called more times than it had replies")
        return FakeResponse(self.replies.pop(0))


def test_extract_json_block_survives_fences_and_prose():
    """Real replies arrive wrapped in chatter; the parser has to cope."""
    messy = (
        "Sure! Here is the record you asked for:\n\n"
        "```json\n"
        '{"a": 1, "nested": {"b": 2}, "tricky": "a closing brace } inside a string"}\n'
        "```\n"
        "Let me know if you need anything else."
    )
    parsed = json.loads(extract_json_block(messy))
    assert parsed["nested"]["b"] == 2
    assert parsed["tricky"].endswith("inside a string")


def test_extract_json_block_flags_truncation():
    """An unterminated object means max_tokens, not a bad prompt -- say so."""
    with pytest.raises(ExtractionError, match="max_tokens"):
        extract_json_block('{"issue_number": 1, "title": "half a repl')


def test_schema_block_contains_every_field():
    """The prompt's schema is generated, so it cannot drift from the model."""
    schema = schema_block(IssueSummary)
    for field_name in IssueSummary.model_fields:
        assert field_name in schema
    # The enum's allowed values must reach the prompt too, or the model is
    # guessing at the category vocabulary.
    assert "feature_request" in schema


def test_valid_payload_validates_and_extra_fields_are_rejected():
    summary = IssueSummary.model_validate(VALID_PAYLOAD)
    assert summary.issue_number == 4212
    assert summary.category.value == "bug"

    # A hallucinated field must fail loudly rather than be silently dropped.
    with pytest.raises(ValidationError):
        IssueSummary.model_validate({**VALID_PAYLOAD, "severity": "high"})


def test_retry_feeds_the_validation_error_back_to_the_model():
    """The core behaviour of phase 1: correct, don't just re-ask."""
    broken = dict(VALID_PAYLOAD, confidence=4.2)  # out of the 0-1 range
    client = FakeClient(json.dumps(broken), json.dumps(VALID_PAYLOAD))

    summary = extract(client, IssueSummary, "irrelevant thread text")

    assert summary.confidence == 0.9
    assert len(client.calls) == 2

    # The second request must carry three things: the original prompt, the
    # model's own bad reply, and the validator's error. Drop any one of them
    # and the retry is just a second coin flip.
    second_turn = client.calls[1]["messages"]
    assert [m["role"] for m in second_turn] == ["user", "assistant", "user"]
    assert "4.2" in second_turn[1]["content"]
    assert "confidence" in second_turn[2]["content"]
    assert "less than or equal to 1" in second_turn[2]["content"]


def test_empty_reply_is_retried_without_being_replayed():
    """An empty assistant turn is a 400 on OpenAI-style APIs; never send one."""
    client = FakeClient("", json.dumps(VALID_PAYLOAD))

    summary = extract(client, IssueSummary, "irrelevant thread text")

    assert summary.issue_number == 4212
    assert len(client.calls) == 2
    assert [m["role"] for m in client.calls[1]["messages"]] == ["user"]


def test_gives_up_after_max_attempts():
    """Bounded, and the failure carries the evidence needed to debug it."""
    client = FakeClient("not json at all", "still not json", "nope")

    with pytest.raises(ExtractionError) as excinfo:
        extract(client, IssueSummary, "irrelevant thread text", max_attempts=3)

    assert len(client.calls) == 3
    assert len(excinfo.value.attempts) == 3


def test_parsed_output_always_uses_temperature_zero():
    """A design decision worth a test: sampling variety is a defect here."""
    client = FakeClient(json.dumps(VALID_PAYLOAD))
    extract(client, IssueSummary, "irrelevant thread text")
    assert client.calls[0]["temperature"] == EXTRACTION_TEMPERATURE == 0.0
