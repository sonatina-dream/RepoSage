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
    ToolCall,
    TokenUsage,
)

_STOP_REASONS = {
    "end_turn": STOP_END_TURN,
    "max_tokens": STOP_MAX_TOKENS,
    "tool_use": STOP_TOOL_USE,
    "stop_sequence": STOP_END_TURN,
}


class AnthropicProvider:
    """Talks to Claude models through Anthropic's SDK."""

    name = "anthropic"

    # Only failures worth resending unchanged; a 400 or 401 would fail the same way again.
    retryable_errors = (
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
    )

    def __init__(self, api_key: str, timeout_s: float) -> None:
        """Create the SDK client with its own retries turned off."""
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_s,
            max_retries=0,  # LLMClient owns the retry policy.
        )

    # -- translation -------------------------------------------------------

    @staticmethod
    def _tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert RepoSage tool specs to Anthropic's format (almost identical)."""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"],
            }
            for tool in tools
        ]

    @staticmethod
    def _to_wire(messages: Sequence[Message]) -> list[Message]:
        """Convert RepoSage's message history into Anthropic's message list.

        Tool results become tool_result blocks in a user message; empty assistant text is left out.
        """
        wire: list[Message] = []
        previous_role = None

        for message in messages:
            role = message["role"]

            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": message["content"],
                    "is_error": bool(message.get("is_error")),
                }
                # Consecutive results share one user turn; separate turns would be a 400.
                if previous_role == "tool":
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
            elif role == "assistant":
                content: list[dict[str, Any]] = []
                if message.get("content"):
                    content.append({"type": "text", "text": message["content"]})
                for call in message.get("tool_calls", []):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                wire.append({"role": "assistant", "content": content})
            else:
                wire.append({"role": "user", "content": message["content"]})

            previous_role = role

        return wire

    @staticmethod
    def _usage(usage: Any) -> TokenUsage:
        """Convert Anthropic's usage object into TokenUsage."""
        # Prompt caching is not enabled yet, so there is no cached split to report.
        return TokenUsage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)

    def _kwargs(
        self,
        model: str,
        messages: Sequence[Message],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Build the request fields shared by complete() and stream()."""
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": self._to_wire(messages),
            # Newer SDKs dropped `temperature` from their signatures; the API still
            # accepts it, so send it in the request body instead.
            "extra_body": {"temperature": temperature},
        }
        # Anthropic takes the system prompt as a top-level parameter, not a message.
        if system:
            kwargs["system"] = system
        return kwargs

    # -- requests ----------------------------------------------------------

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
        """Send one request and return the reply's text, tool calls and usage."""
        kwargs = self._kwargs(model, messages, system, max_tokens, temperature)
        if tools:
            kwargs["tools"] = self._tools(tools)

        response = self._client.messages.create(**kwargs)

        # Text and tool calls arrive mixed together in one list of blocks.
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(block.text)
            elif kind == "tool_use":
                # Arguments arrive already parsed.
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return ProviderReply(
            text="".join(text_parts),
            usage=self._usage(response.usage),
            stop_reason=_STOP_REASONS.get(response.stop_reason, response.stop_reason or ""),
            tool_calls=tool_calls,
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
        """Yield text chunks as they arrive, then return the usage."""
        kwargs = self._kwargs(model, messages, system, max_tokens, temperature)

        with self._client.messages.stream(**kwargs) as stream:
            for chunk in stream.text_stream:
                yield chunk
            final = stream.get_final_message()

        return self._usage(final.usage)
