"""DeepSeek, reached through its OpenAI-compatible API with the `openai` client.

Differences handled here: system prompt as the first message, tool arguments as a JSON
string, cache-hit token counts, usage only on the last stream chunk, thinking turned off,
and a known defect where a tool call arrives as plain text.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Sequence

import openai

from .base import (
    STOP_END_TURN,
    STOP_MAX_TOKENS,
    STOP_TOOL_USE,
    Message,
    ProviderContractError,
    ProviderReply,
    ToolCall,
    TokenUsage,
    looks_like_unparsed_tool_call,
)

BASE_URL = "https://api.deepseek.com"

# V4 thinks by default and bills it against max_tokens; with thinking on and tools
# offered, earlier reasoning_content would also have to be sent back.
THINKING_DISABLED = {"thinking": {"type": "disabled"}}

_FINISH_REASONS = {
    "stop": STOP_END_TURN,
    "length": STOP_MAX_TOKENS,
    "tool_calls": STOP_TOOL_USE,
}


class DeepSeekProvider:
    """Talks to DeepSeek models through the OpenAI-compatible API."""

    name = "deepseek"

    # ProviderContractError is intermittent, so a retry usually succeeds (and each retry prints why).
    retryable_errors = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError,
        ProviderContractError,
    )

    def __init__(self, api_key: str, timeout_s: float) -> None:
        """Create an OpenAI client pointed at DeepSeek, with its own retries turned off."""
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=timeout_s,
            max_retries=0,  # LLMClient owns the retry policy.
        )

    # -- translation -------------------------------------------------------

    @staticmethod
    def _tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Wrap each RepoSage tool spec in OpenAI's "function" format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _to_wire(messages: Sequence[Message], system: str | None) -> list[Message]:
        """Convert RepoSage's message history into an OpenAI-style message list.

        The system prompt goes first; each tool result is its own message, with errors marked "ERROR:".
        """
        wire: list[Message] = []
        if system:
            wire.append({"role": "system", "content": system})

        for message in messages:
            role = message["role"]

            if role == "tool":
                content = message["content"]
                # This format has no is_error field, so the failure goes in the text.
                if message.get("is_error"):
                    content = f"ERROR: {content}"
                wire.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_call_id"],
                        "content": content,
                    }
                )
                continue

            if role == "assistant":
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get("content") or None,
                }
                calls = message.get("tool_calls", [])
                if calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in calls
                    ]
                wire.append(entry)
                continue

            wire.append({"role": "user", "content": message["content"]})

        return wire

    @staticmethod
    def _usage(usage: Any) -> TokenUsage:
        """Convert DeepSeek's usage object into TokenUsage, including cache hits."""
        if usage is None:
            return TokenUsage()
        return TokenUsage(
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
            # prompt_tokens == hit + miss, so the hit count is a subset.
            cached_input_tokens=getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
        )

    @staticmethod
    def _parse_tool_calls(message: Any) -> list[ToolCall]:
        """Parse the reply's tool calls; malformed arguments raise ProviderContractError."""
        calls: list[ToolCall] = []
        for raw in getattr(message, "tool_calls", None) or []:
            try:
                arguments = json.loads(raw.function.arguments or "{}")
            except (json.JSONDecodeError, ValueError) as exc:
                raise ProviderContractError(
                    f"deepseek returned unparseable arguments for tool "
                    f"{raw.function.name!r}: {raw.function.arguments!r}"
                ) from exc
            if not isinstance(arguments, dict):
                raise ProviderContractError(
                    f"deepseek returned non-object arguments for tool "
                    f"{raw.function.name!r}: {arguments!r}"
                )
            calls.append(ToolCall(id=raw.id, name=raw.function.name, arguments=arguments))
        return calls

    @staticmethod
    def _check_contract(
        text: str,
        tool_calls: Sequence[ToolCall],
        finish_reason: str,
        tools: Sequence[dict[str, Any]] | None,
    ) -> None:
        """Raise ProviderContractError if a tool call came back as text, or went missing."""
        tool_names = [tool["name"] for tool in tools or []]

        if tools and not tool_calls and looks_like_unparsed_tool_call(text, tool_names):
            raise ProviderContractError(
                "deepseek serialised a tool call into the message content "
                f"instead of tool_calls (finish_reason={finish_reason!r}). This "
                "is a known, unresolved provider defect -- "
                "github.com/deepseek-ai/DeepSeek-V3/issues/1244. The request is "
                "usually fine on retry. Content was: " + text.strip()[:300]
            )

        if finish_reason == "tool_calls" and not tool_calls:
            raise ProviderContractError(
                "deepseek reported finish_reason='tool_calls' but returned no "
                "tool_calls to run."
            )

    def _kwargs(
        self,
        model: str,
        messages: Sequence[Message],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Build the request fields shared by complete() and stream()."""
        return {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self._to_wire(messages, system),
            "extra_body": THINKING_DISABLED,
        }

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

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        finish_reason = choice.finish_reason or ""

        # A plain string, and null when the model only called a tool.
        text = choice.message.content or ""
        tool_calls = self._parse_tool_calls(choice.message)
        self._check_contract(text, tool_calls, finish_reason, tools)

        return ProviderReply(
            text=text,
            usage=self._usage(response.usage),
            stop_reason=_FINISH_REASONS.get(finish_reason, finish_reason),
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
        stream = self._client.chat.completions.create(
            **self._kwargs(model, messages, system, max_tokens, temperature),
            stream=True,
            # Without this the usage totals may never arrive and the run looks free.
            stream_options={"include_usage": True},
        )

        usage = TokenUsage()
        for chunk in stream:
            if chunk.usage is not None:
                usage = self._usage(chunk.usage)
            # The final usage chunk has no choices, so don't assume choices[0] exists.
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

        return usage
