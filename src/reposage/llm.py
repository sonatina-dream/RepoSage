"""The one door every model call in RepoSage passes through.

Three things this adds over calling a vendor SDK directly, each because of
something that bites later:

1. **Token and cost accounting.** Every response carries usage. If you do not
   add it up at the moment of the call you will never answer "what did that
   agent run cost?", because by phase 3 a single user question becomes eight
   API calls and nothing else sees all of them.

2. **A spend ceiling.** Enforced here rather than at the call sites, and
   deliberately not per-provider: it is the one guard rail that has to hold
   everywhere. An agent loop with a bug is a `while True:` around a paid API;
   the ceiling turns that from an invoice into an exception.

3. **Explicit retries.** Both SDKs would retry some failures for us. We turn
   that off and do it by hand, because from phase 3 the loop must distinguish
   two failures that look similar and need opposite handling: "the transport
   failed, resend the same request" and "the model returned something we
   cannot use, send it back with the error". Only the first belongs here.

What is *not* here is any knowledge of a vendor's wire format. That lives in
providers/, so that everything above this line -- extraction, the agent loop,
the eval harness -- is written once and runs against either.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .config import Settings, estimate_cost, format_usd, provider_for
from .providers import Message, Provider, TokenUsage, ToolCall, build_provider


class BudgetExceeded(RuntimeError):
    """Raised when a request would push cumulative spend past the ceiling.

    Raised *before* the request is sent. A budget check that runs afterwards
    only tells you about money you have already spent.
    """


@dataclass
class Usage:
    """Running totals for one client's lifetime."""

    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def summary(self) -> str:
        cached = (
            f" ({self.cached_input_tokens} cached)" if self.cached_input_tokens else ""
        )
        return (
            f"{self.calls} call(s), {self.input_tokens} in{cached} / "
            f"{self.output_tokens} out tokens, {format_usd(self.cost_usd)}"
        )


@dataclass
class LLMResponse:
    """One model reply, with its accounting attached.

    `stop_reason` matters more than it looks. "end_turn" means the model
    finished its thought; "max_tokens" means we cut it off mid-sentence, which
    silently corrupts anything downstream that parses the text. Phase 1 turns
    that into a validation error; phase 3 has to handle it in the loop.
    """

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
        return self.stop_reason == "max_tokens"

    @property
    def wants_tools(self) -> bool:
        """The model asked for tools and is waiting for their results.

        Phase 2 acts on this once, by hand. Phase 3 turns it into the loop
        condition.
        """
        return bool(self.tool_calls)


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        provider: Provider | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.usage = Usage()

        # Injection seam for tests and for phase 5, where the judge and the
        # system under test must not share a budget.
        self._provider = provider or build_provider(
            self.settings.provider,
            self.settings.require_api_key(),
            self.settings.request_timeout_s,
        )

    @property
    def quality_model(self) -> str:
        """The model to use where reasoning quality is the deliverable."""
        return self.settings.quality_model

    # -- budget ------------------------------------------------------------

    @property
    def remaining_budget_usd(self) -> float:
        return max(0.0, self.settings.spend_ceiling_usd - self.usage.cost_usd)

    def _estimate_input_tokens(
        self,
        messages: Sequence[Message],
        system: str | None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> int:
        """Cheap pre-flight estimate: roughly four characters per token.

        Exact token-counting endpoints exist, but they cost a round trip per
        call. For a guard whose job is "stop a runaway loop", a rough
        over-estimate is the right trade: it errs toward stopping early, and
        the exact figure arrives with the response anyway.

        Tool schemas are counted, and they are not a rounding error. They are
        injected into the model's context on *every* call, so three tools with
        thorough descriptions can be a larger fixed cost than the user's
        question. A budget guard that ignores them under-estimates every
        tool-enabled request.
        """
        chars = len(system or "")
        for message in messages:
            content = message.get("content") or ""
            chars += len(content) if isinstance(content, str) else len(str(content))
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
        projected = estimate_cost(
            model,
            self._estimate_input_tokens(messages, system, tools),
            # Worst case: assume the model uses every output token we allowed,
            # and assume none of the prompt hits a cache. We cannot know either
            # until we have already paid for the call.
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
        """Exponential backoff with jitter.

        The jitter is not decoration. Without it, every worker that hit the
        same rate limit retries at the same instant and re-creates the burst
        that caused it -- exactly the traffic shape phase 6 produces when
        several requests stream at once.
        """
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                return operation()
            except self._provider.retryable_errors as exc:
                last_error = exc
                if attempt == self.settings.max_retries:
                    break
                delay = (2 ** attempt) + random.uniform(0, 0.5)
                # The reason is printed, not just the type. Some retryable
                # failures are provider defects rather than transport hiccups,
                # and a silent retry of those would hide exactly what you need
                # to see.
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
        stop_sequences: Sequence[str] | None = None,
    ) -> LLMResponse:
        """Send one request and return the text plus its accounting.

        `messages` is the full conversation, every time. Both APIs are
        stateless: neither remembers the previous call, so "the conversation"
        is a list we resend and grow ourselves. That is not a limitation to
        work around -- it is what makes the agent loop in phase 3 inspectable,
        and why context budget becomes a real problem there.
        """
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
                stop_sequences=stop_sequences,
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
        """Yield text as it is generated.

        Streaming changes nothing about what the model produces or what it
        costs -- same tokens, same price. It changes when the user sees the
        first one, which is the difference between a six-second wait and a
        response that starts immediately.

        Note the `yield from`: it forwards every chunk to our caller *and*
        hands us the provider's return value when the stream ends. Usage is
        only knowable at the end, so this is where accounting happens.

        Retries are deliberately absent. A stream that fails halfway has
        already delivered text to the caller; resending it would duplicate
        output rather than repair it. Phase 6 handles mid-stream failure at
        the transport layer, where the client can be told to discard.
        """
        model = model or self.settings.model
        max_tokens = max_tokens or self.settings.max_tokens
        temperature = self.settings.temperature if temperature is None else temperature
        payload = self._as_messages(prompt, messages)

        self._guard_budget(model, payload, system, max_tokens)

        usage = yield from self._provider.stream(
            model=model,
            messages=payload,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        self._record(model, usage or TokenUsage())


def describe_target(settings: Settings) -> str:
    """One line naming who is about to be billed, for the top of a script run."""
    return (
        f"provider={settings.provider} model={settings.model} "
        f"ceiling={format_usd(settings.spend_ceiling_usd)}"
    )
