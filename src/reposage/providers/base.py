"""The shape every provider must present to the rest of RepoSage.

The point of this layer is *not* "support many vendors" as an end in itself.
It is that the differences between vendors are small, specific, and exactly
the kind of thing that quietly corrupts an agent loop if you let them leak
upward. Naming them in one place is what keeps the rest of the codebase honest.

What actually differs between the Anthropic and DeepSeek (OpenAI-compatible)
APIs, at the point this project touches them:

  system prompt   Anthropic takes it as a top-level `system` parameter.
                  OpenAI-style takes it as a message with role "system" at the
                  head of the list. Sending it the wrong way is the single most
                  common porting bug between the two.

  reply shape     Anthropic returns `content` as a *list of blocks*, each with
                  a type. OpenAI-style returns `choices[0].message.content` as
                  a plain string. This difference stops being cosmetic in
                  phase 2, when one of those Anthropic blocks becomes a
                  tool_use block and the OpenAI-style equivalent appears in a
                  sibling field instead.

  stop reason     "end_turn" / "max_tokens" versus "stop" / "length". Same
                  meanings, different spellings. Normalised here, because code
                  that checks whether a reply was truncated should not have to
                  know who produced it.

  usage           Anthropic reports input/output tokens. DeepSeek additionally
                  splits the prompt into cache hits and misses, billed at very
                  different rates. Our normalised usage carries the split, and
                  providers that do not report one send zero.

  errors          Each SDK raises its own exception classes for the same
                  transport failures, so each provider declares which of its
                  own are worth retrying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, Sequence

Message = dict[str, Any]


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
    raw: Any = field(default=None, repr=False)


# Normalised stop reasons. Kept deliberately small: these are the only two
# distinctions this project acts on, and inventing more would mean inventing
# mappings that are not really equivalent across vendors.
STOP_END_TURN = "end_turn"
STOP_MAX_TOKENS = "max_tokens"
STOP_TOOL_USE = "tool_use"  # unused until phase 2, named here so it is one word


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
        stop_sequences: Sequence[str] | None = None,
    ) -> ProviderReply:
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
        stream, and this keeps that fact visible in the type rather than
        hidden in a side effect.
        """
        ...
