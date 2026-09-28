"""The client every model call goes through: cost tracking, a spend ceiling and retries.

Vendor wire formats live in providers/; nothing here knows which vendor is serving.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .config import Settings, estimate_cost, format_usd
from .providers import Message, Provider, TokenUsage, ToolCall, build_provider


class BudgetExceeded(RuntimeError):
    """Raised before sending a request that could push total spend past the ceiling."""


@dataclass
class Usage:
    """Running totals of calls, tokens and cost for one client."""

    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def summary(self) -> str:
        """One line with calls, tokens and total cost."""
        cached = (
            f" ({self.cached_input_tokens} cached)" if self.cached_input_tokens else ""
        )
        return (
            f"{self.calls} call(s), {self.input_tokens} in{cached} / "
            f"{self.output_tokens} out tokens, {format_usd(self.cost_usd)}"
        )


@dataclass
class LLMResponse:
    """One model reply: its text, tool calls, token counts, cost and stop reason."""

    text: str
    model: str
    provider: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float
    stop_reason: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = field(default=None, repr=False)

    @property
    def truncated(self) -> bool:
        """True if the reply was cut off by the max_tokens limit."""
        return self.stop_reason == "max_tokens"

    @property
    def wants_tools(self) -> bool:
        """True if the model asked for tools and is waiting for their results."""
        return bool(self.tool_calls)


class LLMClient:
    """Sends requests to the configured provider and tracks what they cost."""

    def __init__(
        self,
        settings: Settings | None = None,
        provider: Provider | None = None,
    ) -> None:
        """Use the given settings and provider, or build them from the environment."""
        self.settings = settings or Settings.from_env()
        self.usage = Usage()

        # Tests inject a fake provider here.
        self._provider = provider or build_provider(
            self.settings.provider,
            self.settings.require_api_key(),
            self.settings.request_timeout_s,
        )

    @property
    def quality_model(self) -> str:
        """The model to use where reasoning quality matters most."""
        return self.settings.quality_model

    # -- budget ------------------------------------------------------------

    @property
    def remaining_budget_usd(self) -> float:
        """Dollars left before the spend ceiling."""
        return max(0.0, self.settings.spend_ceiling_usd - self.usage.cost_usd)

    def _estimate_input_tokens(
        self,
        messages: Sequence[Message],
        system: str | None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> int:
        """Roughly estimate prompt tokens (about 4 characters each), tool schemas included."""
        chars = len(system or "")
        for message in messages:
            chars += len(str(message.get("content") or ""))
            for call in message.get("tool_calls", []):
                chars += len(call.name) + len(json.dumps(call.arguments))
        for tool in tools or []:
            chars += len(json.dumps(tool))
        return chars // 4

    def _guard_budget(
        self,
        model: str,
        messages: Sequence[Message],
        system: str | None,
        max_tokens: int,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Raise BudgetExceeded if the worst-case cost of this request would pass the ceiling."""
        projected = estimate_cost(
            model,
            self._estimate_input_tokens(messages, system, tools),
            # Worst case: every allowed output token is used, and nothing is cached.
            max_tokens,
        )
        if self.usage.cost_usd + projected > self.settings.spend_ceiling_usd:
            raise BudgetExceeded(
                f"Refusing to send: spent {format_usd(self.usage.cost_usd)} of "
                f"{format_usd(self.settings.spend_ceiling_usd)}, and this call could "
                f"cost up to {format_usd(projected)}. Raise "
                f"Settings.spend_ceiling_usd if this is expected."
            )

    def _record(self, model: str, usage: TokenUsage) -> float:
        """Add one call's tokens and cost to the running totals; return its cost."""
        cost = estimate_cost(
            model,
            usage.input_tokens,
            usage.output_tokens,
            cached_input_tokens=usage.cached_input_tokens,
        )
        self.usage.calls += 1
        self.usage.input_tokens += usage.input_tokens
        self.usage.cached_input_tokens += usage.cached_input_tokens
        self.usage.output_tokens += usage.output_tokens
        self.usage.cost_usd += cost
        return cost

    # -- transport ---------------------------------------------------------

    def _with_retries(self, operation, description: str):
        """Run `operation`, retrying retryable errors with growing, randomised waits."""
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                return operation()
            except self._provider.retryable_errors as exc:
                last_error = exc
                if attempt == self.settings.max_retries:
                    break
                # Random jitter stops clients that failed together from retrying together.
                delay = (2 ** attempt) + random.uniform(0, 0.5)
                # Print the reason: some retryable errors are provider defects worth seeing.
                print(
                    f"  [retry] {description} failed ({type(exc).__name__}: "
                    f"{str(exc)[:160]}); retrying in {delay:.1f}s "
                    f"({attempt + 1}/{self.settings.max_retries})"
                )
                time.sleep(delay)
        raise RuntimeError(
            f"{description} failed after {self.settings.max_retries} retries "
            f"({type(last_error).__name__}: {last_error})"
        ) from last_error

    # -- public API --------------------------------------------------------

    @staticmethod
    def _as_messages(prompt: str | None, messages: Sequence[Message] | None) -> list[Message]:
        """Turn a plain prompt or a message list into a message list (exactly one is allowed)."""
        if (prompt is None) == (messages is None):
            raise ValueError("Pass exactly one of `prompt` or `messages`.")
        if prompt is not None:
            return [{"role": "user", "content": prompt}]
        return list(messages or [])

    def complete(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[Message] | None = None,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send one request (the whole conversation each time) and return the reply with its cost."""
        model = model or self.settings.model
        max_tokens = max_tokens or self.settings.max_tokens
        temperature = self.settings.temperature if temperature is None else temperature
        payload = self._as_messages(prompt, messages)

        self._guard_budget(model, payload, system, max_tokens, tools)

        reply = self._with_retries(
            lambda: self._provider.complete(
                model=model,
                messages=payload,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
            ),
            f"{self._provider.name}.complete",
        )

        cost = self._record(model, reply.usage)

        return LLMResponse(
            text=reply.text,
            model=model,
            provider=self._provider.name,
            input_tokens=reply.usage.input_tokens,
            cached_input_tokens=reply.usage.cached_input_tokens,
            output_tokens=reply.usage.output_tokens,
            cost_usd=cost,
            stop_reason=reply.stop_reason,
            tool_calls=reply.tool_calls,
            raw=reply.raw,
        )

    def stream(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[Message] | None = None,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Yield the reply's text piece by piece; usage is recorded when the stream ends.

        Never retried: a half-delivered stream can't be resent without duplicating text.
        """
        model = model or self.settings.model
        max_tokens = max_tokens or self.settings.max_tokens
        temperature = self.settings.temperature if temperature is None else temperature
        payload = self._as_messages(prompt, messages)

        self._guard_budget(model, payload, system, max_tokens)

        # `yield from` passes chunks through and returns the provider's final usage.
        usage = yield from self._provider.stream(
            model=model,
            messages=payload,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        self._record(model, usage or TokenUsage())


def describe_target(settings: Settings) -> str:
    """One line naming the provider, model and spend ceiling, printed at the start of a run."""
    return (
        f"provider={settings.provider} model={settings.model} "
        f"ceiling={format_usd(settings.spend_ceiling_usd)}"
    )
