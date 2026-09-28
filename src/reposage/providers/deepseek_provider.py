"""DeepSeek, reached through its OpenAI-compatible endpoint.

DeepSeek publishes no SDK of its own; it serves an OpenAI-shaped API, so the
`openai` client is pointed at a different base URL. That is a wire-format
detail and it stops here -- nothing above this module knows or cares.

Four things are genuinely different rather than merely spelled differently, and
each costs money or correctness if ignored:

  The system prompt is the first message, not a top-level parameter. Send an
  Anthropic-shaped call and it is silently dropped: the model answers anyway,
  slightly worse, with no error to tell you why.

  Tool arguments arrive as a JSON *string*, not an object, so parsing them is a
  fallible step that needs handling. Anthropic hands you a dict.

  Prompt caching is automatic and the usage object reports the split. Cache
  hits bill at roughly a thirtieth of a miss. From phase 3 the agent resends a
  growing history every turn, so most of the prompt becomes a cache hit -- an
  accounting layer that ignores the split overstates cost by an order of
  magnitude.

  Streaming usage arrives only on the final chunk, and reliably only when
  `stream_options={"include_usage": True}` is requested. Omit it and totals can
  simply be absent, which surfaces later as a run that appears to have cost
  nothing.

Thinking is on by default for the V4 models, and the hidden reasoning is
billed against `max_tokens` before any answer text appears. A 250-token budget
can be spent entirely on reasoning, leaving an empty reply with finish reason
"length". It is switched off here so that `max_tokens` means answer tokens, as
it does on Anthropic by default. Turning it back on is a deliberate choice,
not a default: with tools offered, DeepSeek then requires every earlier turn's
`reasoning_content` to be sent back, which this module does not yet do.

And one outright defect, which is why `_check_contract` exists: the model
intermittently serialises a tool call into `content` as plain text and reports
the finish reason as "stop", so a caller trusting either signal alone silently
drops the call. (github.com/deepseek-ai/DeepSeek-V3/issues/1244, unresolved.)
We cannot fix it, but we can refuse to fail quietly.
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

# Sent on every request. See the module docstring.
THINKING_DISABLED = {"thinking": {"type": "disabled"}}

_FINISH_REASONS = {
    "stop": STOP_END_TURN,
    "length": STOP_MAX_TOKENS,
    "tool_calls": STOP_TOOL_USE,
}


class DeepSeekProvider:
    name = "deepseek"

    # ProviderContractError is in here, which is unusual -- it is normally a
    # "stop and tell someone" error, not a transport hiccup. It earns its place
    # because the defect it flags is intermittent (roughly one reply in ten),
    # so a retry usually succeeds. Retrying is not the same as hiding: every
    # attempt prints the reason, and if all of them fail the final error still
    # names the defect. Recover automatically, never silently.
    retryable_errors = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError,
        ProviderContractError,
    )

    def __init__(self, api_key: str, timeout_s: float) -> None:
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=timeout_s,
            max_retries=0,  # LLMClient owns the retry policy.
        )

    # -- translation -------------------------------------------------------

    @staticmethod
    def _tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Wrap each tool in the OpenAI `function` envelope.

        Note `parameters` where Anthropic says `input_schema`. The JSON Schema
        inside is identical, which is the reason one registry can serve both.
        """
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
        """Serialise neutral history into an OpenAI-style message list.

        Three translations worth naming:

        - The system prompt becomes the first message. See the module
          docstring; this is the porting bug that fails silently.
        - Tool calls hang off the assistant message in a sibling field, and
          their arguments must be re-serialised to a JSON *string*.
        - Each tool result is its own message with role "tool". There is no
          `is_error` field anywhere in this format, so failure has to be
          carried in the text itself -- which is why errors are prefixed. The
          model reads the prefix; there is nothing else for it to read.
        """
        wire: list[Message] = []
        if system:
            wire.append({"role": "system", "content": system})

        for message in messages:
            role = message["role"]

            if role == "tool":
                content = message["content"]
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
        calls: list[ToolCall] = []
        for raw in getattr(message, "tool_calls", None) or []:
            try:
                arguments = json.loads(raw.function.arguments or "{}")
            except (json.JSONDecodeError, ValueError) as exc:
                # Malformed JSON in the arguments string is the model's
                # mistake, not the transport's -- but it arrives here, before
                # the registry can validate anything. Surfacing it as a
                # contract error keeps the failure legible instead of
                # producing a tool call with silently empty arguments.
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
        """Refuse to let the known tool-call-as-text defect pass silently.

        Two checks, and the first is the one that matters:

        If tools were offered, the reply produced no structured tool calls, and
        the entire content is a JSON object naming one of those tools, then the
        call was serialised into the wrong field. Raising here costs a failed
        request. Not raising costs an evening of debugging a correct agent loop
        that appears to ignore its own tools.

        The second check is the reverse, and is pure paranoia: a finish reason
        of "tool_calls" with nothing in the field.
        """
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
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self._to_wire(messages, system),
            "extra_body": THINKING_DISABLED,
        }
        if tools:
            kwargs["tools"] = self._tools(tools)
        if stop_sequences:
            kwargs["stop"] = list(stop_sequences)

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # A plain string here, not a list of blocks, and legitimately null when
        # the model produced only a tool call.
        text = choice.message.content or ""
        tool_calls = self._parse_tool_calls(choice.message)
        self._check_contract(text, tool_calls, choice.finish_reason or "", tools)

        return ProviderReply(
            text=text,
            usage=self._usage(response.usage),
            stop_reason=_FINISH_REASONS.get(
                choice.finish_reason, choice.finish_reason or ""
            ),
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
        stream = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=self._to_wire(messages, system),
            stream=True,
            extra_body=THINKING_DISABLED,
            # Without this the usage totals may never arrive and the run looks
            # free. See the module docstring.
            stream_options={"include_usage": True},
        )

        usage = TokenUsage()
        for chunk in stream:
            if chunk.usage is not None:
                usage = self._usage(chunk.usage)
            # The final usage-bearing chunk carries no choices at all, so this
            # cannot assume choices[0] exists.
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

        return usage
