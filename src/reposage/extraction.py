"""Structured output: getting JSON out of a text model, reliably.

The problem. A language model emits text. Everything downstream of it -- a
database write, a tool call, an eval harness -- wants a typed object. The gap
between those two is where most LLM applications actually break, and it breaks
in a specific, boring way: it works for the first fifty inputs and then one
issue body contains a code fence, and the model wraps its JSON in another code
fence, and `json.loads` throws at 3am.

The naive fix is "ask nicely and hope": put "respond with JSON" in the prompt
and call `json.loads` on whatever comes back. That fails in at least four ways
that all look different in the logs:

  - the model prefixes "Here's the JSON you asked for:" before the object
  - it wraps the object in a ```json fence
  - it produces syntactically valid JSON with the wrong shape -- a string where
    you wanted a list, an invented field, a missing one
  - it runs past `max_tokens` and hands you a truncated object

Only the first two are parsing problems. The third is a *schema* problem, and
it is the one that silently poisons data: `json.loads` succeeds, your code
reads `summary["resolution"]`, and gets a KeyError three functions away from
the cause.

The design here answers all four:

  1. Send the JSON Schema itself in the prompt, not an English description of
     it. The schema is generated from the Pydantic model, so the instruction
     and the validation can never drift apart.
  2. Extract the first balanced JSON object from the reply, so prose and fences
     are survivable rather than fatal.
  3. Validate with Pydantic. A shape error becomes an exception here, at the
     boundary, not a KeyError somewhere downstream.
  4. On failure, retry -- but not by asking again identically. Send the model
     back its own broken output *and the validator's error message*, and ask it
     to fix that. A model that cannot write valid JSON unprompted is usually
     perfectly capable of correcting a specific, named mistake.

One thing worth knowing now, because it makes phase 2 make sense: there is a
better mechanism for this than prompting. If you define a *tool* whose input
schema is your Pydantic model and force the model to call it, the API itself
constrains the output shape and the parsing problem largely disappears. We are
doing it the hard way here deliberately -- the retry loop below is the same
shape as the error-feedback loop the agent will need in phase 3, and it is
easier to understand when it is the only thing going on.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import SONNET

T = TypeVar("T", bound=BaseModel)

# Structured extraction is always temperature 0. Sampling variety is a feature
# when you want prose and a defect when you want a parseable object: the same
# input should produce the same record every time, or your eval set is not a
# fixed target. (Temperature 0 is near-deterministic, not guaranteed identical --
# see phase 0, demo 3.)
EXTRACTION_TEMPERATURE = 0.0

MAX_ATTEMPTS = 3


class ExtractionError(RuntimeError):
    """Raised when the model could not produce a valid object in N attempts.

    Carries the attempt history so a failure is debuggable without a rerun --
    you can see what the model actually said each time.
    """

    def __init__(self, message: str, attempts: list[str] | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class IssueCategory(str, Enum):
    """A closed enum, so "kind of a bug?" cannot become a new category.

    Free-text categories are the classic structured-output trap: you get
    "bug", "Bug", "bug report" and "defect" in the same column, and only
    notice when you try to group by it in phase 5.
    """

    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    DOCUMENTATION = "documentation"
    QUESTION = "question"
    PERFORMANCE = "performance"
    OTHER = "other"


class IssueSummary(BaseModel):
    """One closed GitHub issue, reduced to something an eval set can use.

    `extra="forbid"` is load-bearing. Pydantic's default is to ignore fields it
    did not expect, which means a model that hallucinates a plausible-looking
    `severity` field produces a valid object and you never find out. Forbidding
    extras turns that hallucination into a validation error -- which the retry
    loop then shows to the model, and it usually drops the field and moves on.
    """

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

    # The reason this project extracts issues at all. Every closed issue is a
    # question a human already asked and got answered, which makes it free
    # ground truth for the phase 5 eval set.
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
    """The Pydantic model rendered as JSON Schema, for the prompt.

    Generated rather than hand-written on purpose. A hand-written description
    of the shape is a second source of truth, and it goes stale the first time
    someone adds a field to the model and forgets the prompt.
    """
    return json.dumps(model_cls.model_json_schema(), indent=2)


def build_prompt(model_cls: Type[BaseModel], source_text: str) -> str:
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
    """Pull the first balanced JSON object out of a model reply.

    Why brace-matching rather than a regex: a regex for "the JSON bit" either
    stops at the first `}` (breaking on any nested object -- and our schema has
    nested objects) or is greedy (swallowing trailing prose). Counting braces
    while skipping string literals handles both, and handles a `}` that appears
    inside a string value, which is common in issue text.
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

    # Unbalanced braces almost always means the reply was cut off at
    # max_tokens. Say that, rather than "invalid JSON", because the fix is
    # different: a bigger max_tokens, not a better prompt.
    raise ExtractionError(
        "JSON object is unterminated -- the reply was probably truncated at "
        "max_tokens. Raise max_tokens for this call."
    )


def extract(
    client: Any,
    model_cls: Type[T],
    source_text: str,
    *,
    model: str = SONNET,
    max_attempts: int = MAX_ATTEMPTS,
    max_tokens: int = 1500,
) -> T:
    """Extract a validated `model_cls` instance, retrying with the error text.

    `client` is anything with a `.complete(...)` returning an object with
    `.text` -- our LLMClient in production, a fake in the tests. Keeping the
    dependency this loose is what makes the retry behaviour testable without an
    API key, which matters because retry logic is exactly the code you cannot
    afford to leave untested.
    """
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
        except (ValidationError, ExtractionError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt == max_attempts:
                break

            # The two turns below are the whole trick. The assistant turn is
            # the model's own broken reply -- without it the model has no idea
            # what it is being asked to correct, because the API kept no memory
            # of the last call. The user turn is the validator's message,
            # verbatim: Pydantic's errors name the field and the expected type,
            # which is far more actionable than "that was wrong, try again".
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
