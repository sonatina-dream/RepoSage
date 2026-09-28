"""The shared types and interface every provider implements.

Vendors differ in how they place the system prompt, shape replies, tool calls and
tool results, and name stop reasons; each provider translates to and from these types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, Sequence

# --------------------------------------------------------------------------
# Neutral message history
# --------------------------------------------------------------------------
# RepoSage's own history format; each provider's `_to_wire` translates it. Every entry is one of:
#
#   {"role": "user",      "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
#   {"role": "tool",      "tool_call_id": str, "content": str, "is_error": bool}
Message = dict[str, Any]


@dataclass
class ToolCall:
    """A model's request to run one tool, with its arguments already parsed into a dict."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The result of running one tool; `is_error` tells the model the tool failed."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class TokenUsage:
    """Token counts for one request; cached tokens are part of the input, not extra."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


@dataclass
class ProviderReply:
    """One model reply in RepoSage's own terms: text, usage, stop reason and tool calls."""

    text: str
    usage: TokenUsage
    stop_reason: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = field(default=None, repr=False)


# The only stop reasons this project acts on.
STOP_END_TURN = "end_turn"
STOP_MAX_TOKENS = "max_tokens"
STOP_TOOL_USE = "tool_use"


class ProviderContractError(RuntimeError):
    """Raised when a vendor returns something its own API rules say it should not."""


def looks_like_unparsed_tool_call(text: str, tool_names: Sequence[str]) -> bool:
    """Return True if the whole text is a JSON object naming one of our tools.

    Catches DeepSeek sending a tool call as plain text; strict, so prose about a tool never matches.
    """
    stripped = text.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return False
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(parsed, dict):
        return False

    for key in ("name", "function", "tool", "tool_name"):
        value = parsed.get(key)
        if isinstance(value, dict):
            value = value.get("name")
        if isinstance(value, str) and value in tool_names:
            return True
    return False


class Provider(Protocol):
    """What LLMClient needs from a vendor; retries, budget and accounting stay in LLMClient."""

    name: str
    retryable_errors: tuple[type[Exception], ...]

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        system: str | None,
        max_tokens: int,
        temperature: float,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> ProviderReply:
        """Send one request, offering `tools` (RepoSage tool specs) if given."""
        ...

    def stream(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        """Yield text chunks, then return the TokenUsage when the stream ends."""
        ...
