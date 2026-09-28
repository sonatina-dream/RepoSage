"""Structured output: turn a model's text reply into a validated Pydantic object.

The prompt includes the JSON Schema; the reply's JSON is pulled out, validated, and on
failure the model gets its own output plus the error and is asked to fix it.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

T = TypeVar("T", bound=BaseModel)

# Always 0 for parsed output: the same input should give the same record.
EXTRACTION_TEMPERATURE = 0.0

MAX_ATTEMPTS = 3


class ExtractionError(RuntimeError):
    """Raised when no valid object came back; `attempts` holds every raw reply."""

    def __init__(self, message: str, attempts: list[str] | None = None) -> None:
        """Store the message and the raw replies from each attempt."""
        super().__init__(message)
        self.attempts = attempts or []


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
# A fixed list, so free text like "Bug" or "defect" can't become new categories.
# Pydantic sends class docstrings to the model inside the JSON Schema: keep them short.
class IssueCategory(str, Enum):
    """The kind of issue."""

    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    DOCUMENTATION = "documentation"
    QUESTION = "question"
    PERFORMANCE = "performance"
    OTHER = "other"


class IssueSummary(BaseModel):
    """Summary of one closed GitHub issue."""

    # Reject unknown fields, so a hallucinated field fails validation instead of vanishing.
    model_config = ConfigDict(extra="forbid")

    issue_number: int = Field(description="The GitHub issue number.")
    title: str = Field(min_length=1, description="The issue title, verbatim.")
    category: IssueCategory = Field(description="Best-fitting category.")
    one_line_problem: str = Field(
        max_length=240,
        description="The user's actual problem in one sentence, no restating of the title.",
    )
    resolution: str = Field(
        description="How it was resolved, per the thread. 'unclear' if the thread does not say.",
    )
    files_mentioned: list[str] = Field(
        default_factory=list,
        description="Repository file paths named in the thread. Empty list if none.",
    )

    # Closed issues are real answered questions: free ground truth for the eval set.
    eval_question: str = Field(
        description=(
            "A question about the repository that this issue's resolution answers, "
            "phrased as a developer unfamiliar with the issue would ask it."
        )
    )
    eval_answer: str = Field(description="The correct answer, grounded in the thread.")
    answerable_from_repo: bool = Field(
        description=(
            "True only if the answer could be derived from the repository's own code "
            "and docs, without the issue thread."
        )
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="0-1 confidence that this summary is faithful."
    )


# --------------------------------------------------------------------------
# Prompting and parsing
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You extract structured records from GitHub issue threads. You return one "
    "JSON object and nothing else: no prose, no explanation, no markdown fence. "
    "You never invent facts that are not present in the thread; where the thread "
    "does not say, you say so in the field rather than guessing."
)


def schema_block(model_cls: Type[BaseModel]) -> str:
    """Return the model's JSON Schema as text, to put in the prompt."""
    return json.dumps(model_cls.model_json_schema(), indent=2)


def build_prompt(model_cls: Type[BaseModel], source_text: str) -> str:
    """Build the extraction prompt: instructions, the schema, then the issue text."""
    return (
        "Extract one record from the issue thread below.\n\n"
        "It must validate against this JSON Schema:\n\n"
        f"{schema_block(model_cls)}\n\n"
        "Return only the JSON object.\n\n"
        "--- ISSUE THREAD ---\n"
        f"{source_text}\n"
        "--- END ---"
    )


def extract_json_block(text: str) -> str:
    """Return the first complete {...} JSON object in the text, ignoring text around it.

    Counts braces (skipping ones inside strings) so nested objects and stray prose both work.
    """
    start = text.find("{")
    if start == -1:
        raise ExtractionError(f"No JSON object found in model output: {text[:200]!r}")

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    # Unbalanced braces almost always mean the reply hit max_tokens; say so.
    raise ExtractionError(
        "JSON object is unterminated -- the reply was probably truncated at "
        "max_tokens. Raise max_tokens for this call."
    )


def extract(
    client: Any,
    model_cls: Type[T],
    source_text: str,
    *,
    model: str | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    max_tokens: int = 1500,
) -> T:
    """Ask the model for a `model_cls` object, retrying with the error message until it validates.

    `client` needs only a .complete() method; with model=None the client's quality model is used.
    """
    if model is None:
        model = getattr(client, "quality_model", None)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": build_prompt(model_cls, source_text)}
    ]
    attempts: list[str] = []
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        response = client.complete(
            messages=messages,
            system=SYSTEM_PROMPT,
            model=model,
            max_tokens=max_tokens,
            temperature=EXTRACTION_TEMPERATURE,
        )
        raw = response.text
        attempts.append(raw)

        try:
            return model_cls.model_validate_json(extract_json_block(raw))
        except (ValidationError, ExtractionError) as exc:
            last_error = str(exc)
            if attempt == max_attempts:
                break

            # Nothing to correct, and an empty assistant turn is a 400 on OpenAI-style APIs.
            if not raw.strip():
                continue

            # Replay the model's own reply plus the exact error, so it knows what to fix.
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That output failed validation with the following error:\n\n"
                        f"{last_error}\n\n"
                        "Return the corrected JSON object only. Fix exactly what the "
                        "error describes; do not change fields that were already valid."
                    ),
                }
            )

    raise ExtractionError(
        f"Failed to extract a valid {model_cls.__name__} after {max_attempts} "
        f"attempt(s). Last error: {last_error}",
        attempts=attempts,
    )
