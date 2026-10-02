"""The agent loop: call the model, run the tools it asks for, feed results back, repeat.

No framework. The whole idea is a `while` loop with explicit ways out:

    final answer      the model replied without asking for tools
    iteration cap     it kept asking for tools past `max_iterations`
    budget exceeded   this run hit its cost limit, or the client's spend ceiling
    output truncated  a reply hit max_tokens, so it (or its tool arguments) is cut off
    transport error   the provider kept failing after the client's retries

Failures split two ways. Transport failures (network, rate limits) are retried inside
LLMClient and, if they persist, end the run. Tool and contract failures (unknown tool,
bad arguments, a denied approval, a repeated call) go back to the model as error
results, because the model can often fix them on its next step.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .events import (
    BUDGET_EXCEEDED,
    FINAL_ANSWER,
    ITERATION_CAP,
    OUTPUT_TRUNCATED,
    TRANSPORT_ERROR,
    Event,
    Final,
    StepStarted,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
)
from .llm import BudgetExceeded, LLMClient
from .providers import Message, ToolCall, ToolResult
from .tools import ToolRegistry, results_to_messages
from .tracing import TraceWriter

SYSTEM_PROMPT = (
    "You answer questions about a source code repository. You have tools that read "
    "the repository; use them rather than answering from memory, because your memory "
    "of this repository may be out of date or wrong.\n\n"
    "Rules:\n"
    "1. Every factual claim about the code must cite the file and line you read it "
    "from, written exactly as `path/to/file.py:123` (path, colon, line number, with no "
    "space or words between). Never write \"at line 123\" or \"in file.py (line 123)\". "
    "Cite lines you actually saw in a tool result.\n"
    "2. If the repository does not contain the answer, say exactly: \"not found in the "
    "repository\", and describe what you searched. Never guess or fill gaps from memory.\n"
    "3. Search first (search_code) when you do not know the file, then read it "
    "(get_file) with a line range. Do not repeat a tool call you already made.\n"
    "4. When you have enough evidence, stop calling tools and write the answer."
)

# Returns True to allow a risky tool call, False to deny it.
ApprovalCallback = Callable[[ToolCall], bool]


@dataclass
class AgentResult:
    """Everything a finished run produced."""

    answer: str
    exit_reason: str
    steps: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    events: list[Event] = field(default_factory=list)
    tool_calls: int = 0
    trace_path: str | None = None

    @property
    def ok(self) -> bool:
        """True if the run ended with a real final answer."""
        return self.exit_reason == FINAL_ANSWER


def _signature(call: ToolCall) -> str:
    """A stable key for a tool call: its name plus its arguments in sorted-key JSON."""
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"


class Agent:
    """Answers one question per `stream()` / `run()` call, using the registry's tools."""

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int = 8,
        max_cost_usd: float = 0.25,
        approve: ApprovalCallback | None = None,
        tracer: TraceWriter | None = None,
        temperature: float | None = None,
    ) -> None:
        """Set the limits. With no `approve` callback, risky tools are always denied."""
        self.client = client
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_cost_usd = max_cost_usd
        self.approve = approve
        self.tracer = tracer
        self.temperature = temperature

    # -- running -----------------------------------------------------------

    def run(self, question: str) -> AgentResult:
        """Run to completion and return the result, with every event it emitted."""
        events = list(self.stream(question))
        final = events[-1]
        assert isinstance(final, Final)
        return AgentResult(
            answer=final.answer,
            exit_reason=final.exit_reason,
            steps=final.steps,
            cost_usd=final.cost_usd,
            input_tokens=final.input_tokens,
            output_tokens=final.output_tokens,
            events=events,
            tool_calls=sum(isinstance(e, ToolCallEvent) for e in events),
            trace_path=str(self.tracer.path) if self.tracer else None,
        )

    def stream(self, question: str) -> Iterator[Event]:
        """Run the loop, yielding events as they happen; the last one is always a Final."""
        history: list[Message] = [{"role": "user", "content": question}]
        seen_calls: set[str] = set()
        usage_before = (
            self.client.usage.input_tokens,
            self.client.usage.output_tokens,
            self.client.usage.cost_usd,
        )
        started = time.monotonic()
        answer = ""
        steps = 0

        def finish(reason: str, detail: str = "") -> Final:
            """Build the Final event and write the closing trace record."""
            usage = self.client.usage
            final = Final(
                answer=answer,
                exit_reason=reason,
                steps=steps,
                input_tokens=usage.input_tokens - usage_before[0],
                output_tokens=usage.output_tokens - usage_before[1],
                cost_usd=usage.cost_usd - usage_before[2],
                detail=detail,
            )
            self._trace(
                {**final.to_dict(), "type": "run_end", "elapsed_s": time.monotonic() - started}
            )
            return final

        self._trace(
            {
                "type": "run_start",
                "question": question,
                "provider": self.client.settings.provider,
                "model": self.client.settings.model,
                "system_prompt": self.system_prompt,
                "tools": self.registry.names,
                "max_iterations": self.max_iterations,
                "max_cost_usd": self.max_cost_usd,
            }
        )

        while True:
            if steps >= self.max_iterations:
                yield finish(
                    ITERATION_CAP, f"stopped after {self.max_iterations} model calls"
                )
                return
            run_cost = self.client.usage.cost_usd - usage_before[2]
            if run_cost >= self.max_cost_usd:
                yield finish(
                    BUDGET_EXCEEDED,
                    f"run cost ${run_cost:.4f} reached the ${self.max_cost_usd:.2f} limit",
                )
                return

            steps += 1
            yield StepStarted(step=steps)

            # -- one model call ------------------------------------------
            call_started = time.monotonic()
            before = self.client.usage.cost_usd
            try:
                reply = self.client.complete(
                    messages=history,
                    system=self.system_prompt,
                    tools=self.registry.specifications(),
                    temperature=self.temperature,
                )
            except BudgetExceeded as exc:
                yield finish(BUDGET_EXCEEDED, str(exc))
                return
            except RuntimeError as exc:
                # LLMClient already retried; what is left is a persistent transport failure.
                yield finish(TRANSPORT_ERROR, str(exc))
                return
            latency = time.monotonic() - call_started

            answer = reply.text.strip() or answer
            if reply.text.strip():
                yield TextDelta(step=steps, text=reply.text)

            step_record = {
                "type": "step",
                "step": steps,
                "model": reply.model,
                "latency_s": round(latency, 3),
                "input_tokens": reply.input_tokens,
                "cached_input_tokens": reply.cached_input_tokens,
                "output_tokens": reply.output_tokens,
                "cost_usd": self.client.usage.cost_usd - before,
                "stop_reason": reply.stop_reason,
                "history_messages": len(history),
                "text": reply.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in reply.tool_calls
                ],
                "tool_results": [],
            }

            # A cut-off reply can't be trusted: its tool arguments may be half-written JSON.
            if reply.truncated:
                self._trace(step_record)
                yield finish(
                    OUTPUT_TRUNCATED,
                    f"reply hit the {self.client.settings.max_tokens}-token output limit",
                )
                return

            if not reply.wants_tools:
                self._trace(step_record)
                yield finish(FINAL_ANSWER)
                return

            # -- run the requested tools ---------------------------------
            history.append(
                {"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls}
            )
            results: list[ToolResult] = []
            for call in reply.tool_calls:
                tool = self.registry.get(call.name)
                risky = tool is not None and tool.risk == "risky"
                yield ToolCallEvent(
                    step=steps,
                    id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                    needs_approval=risky,
                )

                note = ""
                call_began = time.monotonic()
                signature = _signature(call)
                if signature in seen_calls:
                    note = "repeat"
                    result = ToolResult(
                        call.id,
                        "You already made this exact call (same tool, same arguments) "
                        "earlier in this run, and its result is above in the conversation. "
                        "Use that result, or try a different query or line range. If you "
                        "have enough evidence, write your final answer.",
                        is_error=True,
                    )
                elif risky and not (self.approve and self.approve(call)):
                    note = "denied"
                    result = ToolResult(
                        call.id,
                        f"The user denied permission to run {call.name!r}. Do not retry it; "
                        "answer with what you can learn from the read-only tools.",
                        is_error=True,
                    )
                else:
                    result = self.registry.dispatch(call)
                seen_calls.add(signature)
                results.append(result)

                duration = time.monotonic() - call_began
                step_record["tool_results"].append(
                    {
                        "id": call.id,
                        "name": call.name,
                        "is_error": result.is_error,
                        "note": note,
                        "duration_s": round(duration, 4),
                        "content": result.content,
                    }
                )
                yield ToolResultEvent(
                    step=steps,
                    id=call.id,
                    name=call.name,
                    content=result.content,
                    is_error=result.is_error,
                    duration_s=duration,
                    note=note,
                )

            history.extend(results_to_messages(results))
            self._trace(step_record)

    def _trace(self, record: dict) -> None:
        """Write a record if tracing is on."""
        if self.tracer:
            self.tracer.write(record)
