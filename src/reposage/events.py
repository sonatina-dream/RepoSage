"""The typed events an agent run emits, in order; the phase 8 API streams these to the browser.

A run is: StepStarted, [TextDelta], then per tool call a ToolCallEvent and a ToolResultEvent,
repeated for each step, ending in exactly one Final.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

# Why a run ended. Every run records exactly one.
FINAL_ANSWER = "final_answer"
ITERATION_CAP = "iteration_cap"
BUDGET_EXCEEDED = "budget_exceeded"
OUTPUT_TRUNCATED = "output_truncated"
TRANSPORT_ERROR = "transport_error"


@dataclass(frozen=True)
class Event:
    """Base class: every event has a `type` name and serialises to a plain dict."""

    type: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        """The event as JSON-ready data, with its type name included."""
        return {"type": self.type, **asdict(self)}


@dataclass(frozen=True)
class StepStarted(Event):
    """A new model call is about to be made."""

    type: ClassVar[str] = "step_started"
    step: int


@dataclass(frozen=True)
class TextDelta(Event):
    """Text the model wrote this step. Whole-step for now: tool-using calls are not streamed."""

    type: ClassVar[str] = "text_delta"
    step: int
    text: str


@dataclass(frozen=True)
class ToolCallEvent(Event):
    """The model asked to run a tool. `needs_approval` is True for risky tools."""

    type: ClassVar[str] = "tool_call"
    step: int
    id: str
    name: str
    arguments: dict[str, Any]
    needs_approval: bool = False


@dataclass(frozen=True)
class ToolResultEvent(Event):
    """What came back for a tool call. `note` says if it was denied or blocked as a repeat."""

    type: ClassVar[str] = "tool_result"
    step: int
    id: str
    name: str
    content: str
    is_error: bool
    duration_s: float
    note: str = ""


@dataclass(frozen=True)
class Final(Event):
    """The run is over. `answer` is the last text the model wrote (possibly partial)."""

    type: ClassVar[str] = "final"
    answer: str
    exit_reason: str
    steps: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    detail: str = ""
