"""The shape every provider must present to the rest of RepoSage.

The point of this layer is *not* "support many vendors" as an end in itself.
It is that the differences between vendors are small, specific, and exactly
the kind of thing that quietly corrupts an agent loop if you let them leak
upward. Naming them in one place is what keeps the rest of the codebase honest.

What actually differs, at the point this project touches it:

  system prompt   Anthropic takes it as a top-level `system` parameter.
                  OpenAI-style takes it as a message with role "system" at the
                  head of the list. Sending it the wrong way is the single most
                  common porting bug between the two.

  reply shape     Anthropic returns `content` as a *list of blocks*, each with
                  a type -- text and tool_use sit side by side in one list.
                  OpenAI-style returns text in `message.content` and tool calls
                  in a *sibling field*, `message.tool_calls`.

  tool arguments  Anthropic delivers them already parsed, as an object.
                  OpenAI-style delivers a JSON *string* you must parse -- and
                  which can be malformed, so parsing it is a fallible step.

  tool results    Anthropic wants them as `tool_result` blocks inside a **user**
                  message, several to a message. OpenAI-style wants one message
                  per result with role "tool". Anthropic also has a real
                  `is_error` flag; OpenAI-style has nowhere to put one.

  stop reason     "end_turn"/"max_tokens"/"tool_use" versus "stop"/"length"/
                  "tool_calls". Same meanings, different spellings.

  usage           Anthropic reports input/output tokens. DeepSeek additionally
                  splits the prompt into cache hits and misses, billed at very
                  different rates.

  errors          Each SDK raises its own exception classes for the same
                  transport failures, so each provider declares its own.

Because tool results have to travel in the message history, and the two vendors
disagree about how that history is even shaped, RepoSage keeps its own neutral
history and lets each provider serialise it on the way out. That neutral format
is documented under `Message` below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, Sequence

# --------------------------------------------------------------------------
# Neutral message history
# --------------------------------------------------------------------------
# RepoSage's own history format. Every entry is one of:
#
#   {"role": "user",      "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
#   {"role": "tool",      "tool_call_id": str, "content": str, "is_error": bool}
#
# Providers translate this into their wire format in `_to_wire`. Keeping our
# own format is what lets the agent loop in phase 3 be written once. The name
# "tool" for the result role is borrowed from the OpenAI convention, which is a
# small bias worth admitting: a neutral format still has to pick spellings, and
# picking one vendor's makes that vendor's translator look deceptively simple.
Message = dict[str, Any]


@dataclass
class ToolCall:
    """A model's request to run one tool. It has not been run yet.

    `arguments` is always a parsed dict by the time it reaches this class, even
    for vendors that transmit a JSON string, so nothing downstream has to care
    which did what.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The outcome of running one tool, on its way back to the model.

    `is_error` is the important field. A tool that failed is not an exception
    to propagate -- it is information the model can act on ("that path doesn't
    exist, try another"). Crashing instead means the agent can never recover
    from its own bad guess.
    """

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class TokenUsage:
    """Normalised token counts for one request.

    `input_tokens` is the whole prompt. `cached_input_tokens` is the part of it
    the provider served from its prompt cache and bills more cheaply -- a
    subset, not an addition.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


@dataclass
class ProviderReply:
    """One model reply, in the vocabulary RepoSage uses rather than a vendor's."""

    text: str
    usage: TokenUsage
    stop_reason: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = field(default=None, repr=False)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# Normalised stop reasons. Deliberately few: these are the only distinctions
# this project acts on, and inventing more would mean inventing equivalences
# that are not real.
STOP_END_TURN = "end_turn"
STOP_MAX_TOKENS = "max_tokens"
STOP_TOOL_USE = "tool_use"


class ProviderContractError(RuntimeError):
    """The provider returned something its own protocol says it should not.

    Distinct from a transport error (retry it) and from a model mistake (send
    it back). This one means the vendor broke its side of the contract, and the
    only useful response is to say so loudly, naming the vendor -- because the
    alternative is spending an evening debugging your own correct code.
    """


def looks_like_unparsed_tool_call(text: str, tool_names: Sequence[str]) -> bool:
    """Detect a tool call that arrived as text instead of as structured data.

    This exists for a specific, documented defect: DeepSeek intermittently
    serialises a tool call into the content field and reports the finish reason
    as "stop", so a caller that trusts either signal alone silently drops the
    call. (github.com/deepseek-ai/DeepSeek-V3/issues/1244)

    The check is deliberately strict, because the obvious loose version -- "does
    the text mention a tool name?" -- is catastrophically wrong for this
    project. RepoSage answers questions *about source code*. A model explaining
    what `get_file` does is not calling it, and that ambiguity is precisely the
    reason structured tool calls exist in the first place.

    So we require the entire message to be a JSON object naming one of the
    tools we actually offered. Prose that merely discusses a tool has other
    text around it and will not match.
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
    """What `LLMClient` needs from a vendor, and nothing more.

    Note what is *absent*: retries, budget enforcement, and token accounting.
    Those are cross-cutting concerns that must behave identically no matter who
    serves the request, so they live in LLMClient and providers stay dumb
    translators. In particular the spend ceiling must never be per-provider --
    it is the one guard rail that has to hold everywhere.
    """

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
        stop_sequences: Sequence[str] | None = None,
    ) -> ProviderReply:
        """Send one request.

        `tools` is a list of RepoSage tool specifications -- `{"name",
        "description", "input_schema"}` -- which each provider reshapes into
        its own vendor format. See `ToolRegistry.specifications()`.
        """
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
        """Yield text chunks; *return* a TokenUsage when exhausted.

        A generator's `return` value is delivered to whoever consumes it with
        `yield from`, which is how LLMClient gets the usage totals without a
        callback or a mutable box. Usage is only knowable at the end of a
        stream, and this keeps that fact visible rather than hidden in a side
        effect.
        """
        ...
