"""A thin wrapper around the Anthropic Messages API.

Three things this adds over calling the SDK directly, each of which exists
because of something that will bite us later:

1. **Token and cost accounting.** Every response carries a `usage` block. If
   you do not add it up at the moment of the call you will never be able to
   answer "what did that agent run cost?", because by phase 3 a single user
   question becomes eight API calls and there is no other place that sees all
   of them.

2. **A spend ceiling.** Enforced here rather than at the call sites, because
   the client is the only chokepoint every request must pass through. An agent
   loop with a bug is a `while True:` around a paid API; the ceiling is what
   turns that from an invoice into an exception.

3. **Explicit retries.** The SDK retries some failures for us. We turn that off
   and do it by hand, because from phase 3 onward the loop has to distinguish
   two failures that look similar and need opposite handling: "the transport
   failed, try the same request again" and "the model returned something we
   cannot use, send it back with the error". Only the first belongs here.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import anthropic

from .config import Settings, estimate_cost, format_usd

Message = dict[str, Any]


class BudgetExceeded(RuntimeError):
    """Raised when a request would push cumulative spend past the ceiling.

    Raised *before* the request is sent. A budget check that runs after the
    call has already told you about money you have already spent.
    """


@dataclass
class Usage:
    """Running totals for one client's lifetime."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.calls} call(s), {self.input_tokens} in / "
            f"{self.output_tokens} out tokens, {format_usd(self.cost_usd)}"
        )


@dataclass
class LLMResponse:
    """One model reply, with the accounting attached to it.

    `stop_reason` matters more than it looks. "end_turn" means the model
    finished its thought; "max_tokens" means we cut it off mid-sentence, which
    silently corrupts anything downstream that tries to parse the text. Phase 1
    turns that into a validation error; phase 3 has to handle it in the loop.
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    stop_reason: str | None
    raw: Any = field(default=None, repr=False)

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


# Errors that mean "the request never got a fair hearing" -- worth retrying
# with the exact same payload. Anything else (a 400 for a malformed request,
# a 401 for a bad key) will fail identically no matter how often we retry, and
# retrying it just wastes wall-clock time.
_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)


class LLMClient:
    """The single door through which every model call in RepoSage passes."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.usage = Usage()

        if client is not None:
            # Injection seam for tests and for phase 5, where the judge and the
            # system under test must not share a budget.
            self._client = client
        else:
            self._client = anthropic.Anthropic(
                api_key=self.settings.require_api_key(),
                timeout=self.settings.request_timeout_s,
                # We own the retry policy; see the module docstring.
                max_retries=0,
            )

    # -- budget ------------------------------------------------------------

    @property
    def remaining_budget_usd(self) -> float:
        return max(0.0, self.settings.spend_ceiling_usd - self.usage.cost_usd)

    def _estimate_input_tokens(self, messages: Sequence[Message], system: str | None) -> int:
        """Cheap pre-flight estimate: roughly four characters per token.

        The API offers an exact `count_tokens` endpoint, and it is free, but it
        costs a network round trip per call. For a guard whose job is "stop a
        runaway loop", a rough over-estimate is the right trade: it errs toward
        stopping early, and the exact figure arrives with the response anyway.
        """
        chars = len(system or "")
        for message in messages:
            content = message.get("content", "")
            chars += len(content) if isinstance(content, str) else len(str(content))
        return chars // 4

    def _guard_budget(self, model: str, messages: Sequence[Message], system: str | None,
                      max_tokens: int) -> None:
        projected = estimate_cost(
            model,
            self._estimate_input_tokens(messages, system),
            # Worst case: assume the model uses every output token we allowed.
            # We cannot know the real length until we have paid for it.
            max_tokens,
        )
        if self.usage.cost_usd + projected > self.settings.spend_ceiling_usd:
            raise BudgetExceeded(
                f"Refusing to send: spent {format_usd(self.usage.cost_usd)} of "
                f"{format_usd(self.settings.spend_ceiling_usd)}, and this call could "
                f"cost up to {format_usd(projected)}. Raise "
                f"Settings.spend_ceiling_usd if this is expected."
            )

    def _record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = estimate_cost(model, input_tokens, output_tokens)
        self.usage.calls += 1
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.cost_usd += cost
        return cost

    # -- transport ---------------------------------------------------------

    def _with_retries(self, operation, description: str):
        """Exponential backoff with jitter.

        The jitter is not decoration. Without it, every worker that hit the same
        rate limit retries at the same instant and re-creates the burst that
        caused it -- which is exactly the shape of traffic phase 6 will produce
        when several requests stream at once.
        """
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                return operation()
            except _RETRYABLE as exc:
                last_error = exc
                if attempt == self.settings.max_retries:
                    break
                delay = (2 ** attempt) + random.uniform(0, 0.5)
                print(
                    f"  [retry] {description} failed ({type(exc).__name__}); "
                    f"retrying in {delay:.1f}s "
                    f"({attempt + 1}/{self.settings.max_retries})"
                )
                time.sleep(delay)
        raise RuntimeError(
            f"{description} failed after {self.settings.max_retries} retries"
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
        stop_sequences: Sequence[str] | None = None,
    ) -> LLMResponse:
        """Send one request and return the text plus its accounting.

        `messages` is the full conversation, every time. The API is stateless:
        it has no memory of the previous call, so "the conversation" is just a
        list we resend and grow ourselves. That is not a limitation to work
        around -- it is what makes the agent loop in phase 3 inspectable, and
        it is why context budget becomes a real problem there.
        """
        model = model or self.settings.model
        max_tokens = max_tokens or self.settings.max_tokens
        temperature = self.settings.temperature if temperature is None else temperature
        payload = self._as_messages(prompt, messages)

        self._guard_budget(model, payload, system, max_tokens)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": payload,
        }
        # The system prompt is a top-level parameter, not a message with
        # role="system". Sending it as a message is the most common porting
        # bug from other providers' APIs.
        if system:
            kwargs["system"] = system
        if stop_sequences:
            kwargs["stop_sequences"] = list(stop_sequences)

        response = self._with_retries(
            lambda: self._client.messages.create(**kwargs), "messages.create"
        )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        cost = self._record(model, response.usage.input_tokens, response.usage.output_tokens)

        return LLMResponse(
            text=text,
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost,
            stop_reason=response.stop_reason,
            raw=response,
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
        costs -- the same tokens, the same price. It changes when the user sees
        the first one, which is the difference between a six-second wait and a
        response that starts immediately. Note that usage only becomes known at
        the *end* of the stream, so the accounting happens after the last chunk.
        """
        model = model or self.settings.model
        max_tokens = max_tokens or self.settings.max_tokens
        temperature = self.settings.temperature if temperature is None else temperature
        payload = self._as_messages(prompt, messages)

        self._guard_budget(model, payload, system, max_tokens)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": payload,
        }
        if system:
            kwargs["system"] = system

        with self._client.messages.stream(**kwargs) as stream:
            for chunk in stream.text_stream:
                yield chunk
            final = stream.get_final_message()

        self._record(model, final.usage.input_tokens, final.usage.output_tokens)
