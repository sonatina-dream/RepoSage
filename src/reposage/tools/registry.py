"""The tool registry: one place where a tool's schema and its handler stay married.

A tool has two halves. There is the schema the model sees -- a name, a
description, and a JSON Schema for the arguments -- and there is the Python
function that actually runs. Keeping those in two places guarantees they drift:
you rename a parameter in the function, the schema still advertises the old
name, and the failure surfaces at runtime, inside the agent loop, on some
iteration you cannot easily reproduce.

So the schema is *generated* from a Pydantic model that the handler also
receives. There is one source of truth. This is the same argument as phase 1's
`schema_block`, applied to a different problem.

The registry is also the only sensible home for the things every tool needs and
no tool should implement twice:

  Argument validation. Arguments come from a language model. They are untrusted
  input in the ordinary security sense, and they are also frequently *nearly*
  right -- a string where a number belongs, a missing optional. Validating
  centrally means every tool gets the same treatment.

  Errors as results, not exceptions. This is the design decision that makes an
  agent an agent. When a tool fails, the failure is returned to the model as a
  tool result flagged `is_error`, and the model gets to try something else. If
  a bad path raised instead, the run would die on the model's first typo.

  An output budget. A tool result is not paid for once. It joins the message
  history, and the history is resent on every subsequent turn -- so one
  unbounded `search_code` result is billed again on every turn that follows it.
  Truncation is visible to the model on purpose: told it is seeing 20 of 143
  matches, it narrows the search; left to assume it saw everything, it answers
  confidently from a fifth of the evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Type

from pydantic import BaseModel, ValidationError

from ..providers import ToolCall, ToolResult

# Roughly 2000 tokens. Generous enough for a good chunk of a source file,
# small enough that a runaway result cannot poison the rest of the run.
DEFAULT_MAX_RESULT_CHARS = 8_000


@dataclass
class Tool:
    """One callable, plus everything the model needs to decide to call it."""

    name: str
    description: str
    params: Type[BaseModel]
    handler: Callable[[Any], str]

    def specification(self) -> dict[str, Any]:
        """The vendor-neutral tool spec. Providers reshape this.

        `description` is not documentation -- it is the only thing the model
        reads when deciding which tool to use, so it belongs in the same
        register as a prompt. Tool-selection accuracy is a phase 5 metric, and
        a vague description is the usual reason it is bad.
        """
        schema = self.params.model_json_schema()
        # The model gets no value from a schema title echoing the class name,
        # and every token in here is spent on every call.
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }


@dataclass
class ToolInvocation:
    """One executed call, kept for tracing.

    Phase 3 builds a real trace on top of this. It is here already because the
    moment there is more than one tool call in flight, "what did it actually
    run, in what order, and how long did each take" stops being answerable from
    the printed output.
    """

    name: str
    arguments: dict[str, Any]
    result: ToolResult
    duration_s: float


class ToolRegistry:
    def __init__(self, max_result_chars: int = DEFAULT_MAX_RESULT_CHARS) -> None:
        self._tools: dict[str, Tool] = {}
        self.max_result_chars = max_result_chars
        self.invocations: list[ToolInvocation] = []

    # -- registration ------------------------------------------------------

    def add(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered.")
        self._tools[tool.name] = tool
        return tool

    def register(
        self, name: str, description: str, params: Type[BaseModel]
    ) -> Callable[[Callable[[Any], str]], Callable[[Any], str]]:
        """Decorator form: the handler and its schema declared together."""

        def decorator(handler: Callable[[Any], str]) -> Callable[[Any], str]:
            self.add(Tool(name=name, description=description, params=params, handler=handler))
            return handler

        return decorator

    # -- inspection --------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def specifications(self) -> list[dict[str, Any]]:
        """What gets sent to the model on every single call."""
        return [tool.specification() for tool in self._tools.values()]

    # -- execution ---------------------------------------------------------

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_result_chars:
            return text
        kept = text[: self.max_result_chars]
        dropped = len(text) - self.max_result_chars
        # Said in the result itself, because the model has no other way to know.
        return (
            f"{kept}\n\n[truncated: {dropped} more characters were cut. Narrow "
            f"the request — a line range, a tighter pattern — to see the rest.]"
        )

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Run one requested tool and return a result the model can read.

        Every failure path below produces `is_error=True` rather than raising.
        That is deliberate, and it is the whole reason the agent can recover:
        an unknown tool name, a badly typed argument and a handler that threw
        are all things the model can respond to sensibly if you tell it what
        happened.

        The one thing not caught here is a bug in this method itself. If the
        registry is broken, that is ours, and it should crash.
        """
        started = time.monotonic()

        def finish(content: str, is_error: bool = False) -> ToolResult:
            result = ToolResult(
                tool_call_id=call.id, content=self._truncate(content), is_error=is_error
            )
            self.invocations.append(
                ToolInvocation(
                    name=call.name,
                    arguments=call.arguments,
                    result=result,
                    duration_s=time.monotonic() - started,
                )
            )
            return result

        tool = self._tools.get(call.name)
        if tool is None:
            # Naming the alternatives matters. "Unknown tool" tells the model
            # nothing; listing what exists lets it pick the right one next turn.
            return finish(
                f"No tool named {call.name!r}. Available tools: "
                f"{', '.join(self.names)}.",
                is_error=True,
            )

        try:
            arguments = tool.params.model_validate(call.arguments)
        except ValidationError as exc:
            # Pydantic's message names the field and the expected type, which
            # is far more actionable than "invalid arguments". Same trick as
            # the phase 1 extraction retry: hand back the validator's own words.
            return finish(
                f"Invalid arguments for {call.name!r}:\n{exc}", is_error=True
            )

        try:
            output = tool.handler(arguments)
        except Exception as exc:  # noqa: BLE001 - deliberate; see docstring
            return finish(f"{type(exc).__name__}: {exc}", is_error=True)

        return finish(output if output else "(the tool returned no output)")

    def dispatch_all(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Run every call in one assistant turn.

        A model can request several tools at once, and *every* one needs a
        result before the conversation can continue -- both APIs reject a turn
        with an unanswered call. Running them sequentially is fine at this
        scale; phase 3 can parallelise once there is a loop worth optimising.
        """
        return [self.dispatch(call) for call in calls]


def results_to_messages(results: list[ToolResult]) -> list[dict[str, Any]]:
    """Turn results into neutral history entries to append before the next call."""
    return [
        {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": result.content,
            "is_error": result.is_error,
        }
        for result in results
    ]
