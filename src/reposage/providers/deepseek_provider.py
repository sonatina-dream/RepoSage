"""DeepSeek, reached through its OpenAI-compatible endpoint.

DeepSeek does not publish its own SDK; it serves an OpenAI-shaped API, so the
`openai` client is pointed at a different base URL. That is a wire-format
detail and it stops here -- nothing above this module knows or cares.

Two things are genuinely different rather than merely spelled differently, and
both cost money or correctness if ignored:

  Prompt caching is automatic and the usage object reports the split
  (`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`). Cache hits bill at
  roughly 1/30th of a miss. From phase 3 onward the agent resends a growing
  history every turn, so by the second iteration most of the prompt is a cache
  hit -- an accounting layer that ignores the split will overstate cost by an
  order of magnitude.

  Streaming usage arrives only on the final chunk, and only reliably when
  `stream_options={"include_usage": True}` is requested. Omit it and the totals
  can simply be absent, which shows up much later as a run that appears to have
  cost nothing.

One known defect worth carrying in your head rather than in a comment you skip:
the model intermittently emits a tool call as plain text in `content` instead
of populating `tool_calls`, with the finish reason reported as "stop". That is
a phase 2 concern -- there is nothing to guard here until tools exist -- but it
is the reason phase 2 will validate that a reply's shape matches its finish
reason rather than trusting either alone.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import openai

from .base import (
    STOP_END_TURN,
    STOP_MAX_TOKENS,
    STOP_TOOL_USE,
    Message,
    ProviderReply,
    TokenUsage,
)

BASE_URL = "https://api.deepseek.com"

_FINISH_REASONS = {
    "stop": STOP_END_TURN,
    "length": STOP_MAX_TOKENS,
    "tool_calls": STOP_TOOL_USE,
}


class DeepSeekProvider:
    name = "deepseek"

    retryable_errors = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )

    def __init__(self, api_key: str, timeout_s: float) -> None:
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=timeout_s,
            max_retries=0,  # LLMClient owns the retry policy.
        )

    @staticmethod
    def _messages(messages: Sequence[Message], system: str | None) -> list[Message]:
        """Fold the system prompt into the message list.

        This is the port that bites people. Anthropic keeps `system` outside
        the conversation; OpenAI-style APIs make it the first message. Passing
        an Anthropic-shaped call straight through would drop the system prompt
        entirely -- and the model would answer anyway, slightly worse, with no
        error to tell you why.
        """
        payload = list(messages)
        if system:
            payload.insert(0, {"role": "system", "content": system})
        return payload

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
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self._messages(messages, system),
        }
        if stop_sequences:
            kwargs["stop"] = list(stop_sequences)

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        return ProviderReply(
            # A plain string here, not a list of blocks. `or ""` because the
            # field is legitimately null when the model produced no text.
            text=choice.message.content or "",
            usage=self._usage(response.usage),
            stop_reason=_FINISH_REASONS.get(
                choice.finish_reason, choice.finish_reason or ""
            ),
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
            messages=self._messages(messages, system),
            stream=True,
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
