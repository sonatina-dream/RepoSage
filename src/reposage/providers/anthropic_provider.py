"""Anthropic Messages API, translated into RepoSage's vocabulary."""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import anthropic

from .base import (
    STOP_END_TURN,
    STOP_MAX_TOKENS,
    STOP_TOOL_USE,
    Message,
    ProviderReply,
    TokenUsage,
)

# Anthropic's spellings on the left, ours on the right. Anything unmapped is
# passed through unchanged rather than silently coerced -- an unknown stop
# reason is something you want to see in a log, not have normalised away.
_STOP_REASONS = {
    "end_turn": STOP_END_TURN,
    "max_tokens": STOP_MAX_TOKENS,
    "tool_use": STOP_TOOL_USE,
    "stop_sequence": STOP_END_TURN,
}


class AnthropicProvider:
    name = "anthropic"

    # "The request never got a fair hearing" -- worth resending unchanged.
    # A 400 for a malformed request or a 401 for a bad key will fail
    # identically however often you retry, so they are absent here.
    retryable_errors = (
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
    )

    def __init__(self, api_key: str, timeout_s: float) -> None:
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_s,
            # LLMClient owns the retry policy; see its module docstring.
            max_retries=0,
        )

    def _kwargs(
        self,
        model: str,
        messages: Sequence[Message],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": list(messages),
        }
        # Top-level parameter, not a message with role="system". This is the
        # difference that makes the DeepSeek provider's version of this method
        # look different.
        if system:
            kwargs["system"] = system
        return kwargs

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
        kwargs = self._kwargs(model, messages, system, max_tokens, temperature)
        if stop_sequences:
            kwargs["stop_sequences"] = list(stop_sequences)

        response = self._client.messages.create(**kwargs)

        # `content` is a list of typed blocks, not a string. Today they are all
        # text; in phase 2 tool_use blocks join them, which is why this filters
        # by type instead of reaching for content[0].
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

        return ProviderReply(
            text=text,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                # Anthropic supports prompt caching, but RepoSage does not
                # enable it yet, so there is no split to report.
                cached_input_tokens=0,
            ),
            stop_reason=_STOP_REASONS.get(response.stop_reason, response.stop_reason or ""),
            raw=response,
        )

    def stream(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        kwargs = self._kwargs(model, messages, system, max_tokens, temperature)

        with self._client.messages.stream(**kwargs) as stream:
            for chunk in stream.text_stream:
                yield chunk
            final = stream.get_final_message()

        return TokenUsage(
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )
