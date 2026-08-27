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
    name = "anthropic"

    # "The request never got a fair hearing" -- worth resending unchanged. A
    # 400 for a malformed request or a 401 for a bad key will fail identically
    # however often you retry, so they are absent here.
    retryable_errors = (
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
    )

    def __init__(self, api_key: str, timeout_s: float) -> None:
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_s,
            max_retries=0,  # LLMClient owns the retry policy.
        )

    # -- translation -------------------------------------------------------

    @staticmethod
    def _tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Our tool spec is already Anthropic's shape, near enough."""
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
        """Serialise neutral history into Anthropic's message list.

        Two rules here are not obvious and both produce API errors when broken:

        1. Tool results are `tool_result` blocks inside a **user** message.
           Several results can share one message -- and they *must*, because
           the API requires alternating roles. Emitting one user message per
           result gives you consecutive user turns and a 400. Hence the
           batching loop below.

        2. An assistant turn's text and its tool_use blocks live in the same
           `content` list. An empty text block is rejected, so it is omitted
           rather than sent blank -- which happens routinely, since a model
           calling a tool often says nothing at all first.
        """
        wire: list[Message] = []
        index = 0

        while index < len(messages):
            message = messages[index]
            role = message["role"]

            if role == "tool":
                # Collect this result and every consecutive one into a single
                # user turn.
                blocks: list[dict[str, Any]] = []
                while index < len(messages) and messages[index]["role"] == "tool":
                    result = messages[index]
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": result["tool_call_id"],
                            "content": result["content"],
                            "is_error": bool(result.get("is_error")),
                        }
                    )
                    index += 1
                wire.append({"role": "user", "content": blocks})
                continue

            if role == "assistant":
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

            index += 1

        return wire

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
            "messages": self._to_wire(messages),
        }
        # Top-level parameter, not a message with role="system". This is the
        # difference that makes the DeepSeek provider's equivalent look
        # different.
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
        stop_sequences: Sequence[str] | None = None,
    ) -> ProviderReply:
        kwargs = self._kwargs(model, messages, system, max_tokens, temperature)
        if tools:
            kwargs["tools"] = self._tools(tools)
        if stop_sequences:
            kwargs["stop_sequences"] = list(stop_sequences)

        response = self._client.messages.create(**kwargs)

        # `content` is one list holding both kinds of block. This is the shape
        # difference that stops being cosmetic the moment tools exist: text and
        # tool calls arrive interleaved in a single sequence, rather than in
        # two separate fields.
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(block.text)
            elif kind == "tool_use":
                # Arguments arrive already parsed. Nothing to fail here -- the
                # DeepSeek provider is not so lucky.
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return ProviderReply(
            text="".join(text_parts),
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                # Anthropic supports prompt caching; RepoSage does not enable
                # it yet, so there is no split to report.
                cached_input_tokens=0,
            ),
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
        kwargs = self._kwargs(model, messages, system, max_tokens, temperature)

        with self._client.messages.stream(**kwargs) as stream:
            for chunk in stream.text_stream:
                yield chunk
            final = stream.get_final_message()

        return TokenUsage(
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )
