"""The tool registry: keeps each tool's schema and handler together, and runs tool calls.

Arguments are validated, failures come back to the model as results (never exceptions),
and long outputs are cut to a budget with a visible note.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Type

from pydantic import BaseModel, ValidationError

from ..providers import ToolCall, ToolResult

# About 2,000 tokens: room for a good chunk of a file, but capped.
DEFAULT_MAX_RESULT_CHARS = 8_000


# "safe" tools only read; "risky" tools change something and need approval before the agent runs them.
RiskLevel = Literal["safe", "risky"]


@dataclass
class Tool:
    """A tool: its name, the description the model reads, its argument model and its function."""

    name: str
    description: str
    params: Type[BaseModel]
    handler: Callable[[Any], str]
    risk: RiskLevel = "safe"

    def specification(self) -> dict[str, Any]:
        """Return the tool's spec (name, description, argument schema) for providers to send."""
        schema = self.params.model_json_schema()
        # The title just repeats the class name; drop it to save tokens on every call.
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }


@dataclass
class ToolInvocation:
    """A record of one tool call that ran: name, arguments, result and duration."""

    name: str
    arguments: dict[str, Any]
    result: ToolResult
    duration_s: float


class ToolRegistry:
    """Holds the available tools and runs the calls the model asks for."""

    def __init__(self, max_result_chars: int = DEFAULT_MAX_RESULT_CHARS) -> None:
        """Start empty, with a character limit for each tool result."""
        self._tools: dict[str, Tool] = {}
        self.max_result_chars = max_result_chars
        self.invocations: list[ToolInvocation] = []

    # -- registration ------------------------------------------------------

    def add(self, tool: Tool) -> Tool:
        """Register a tool; raises if the name is already taken."""
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered.")
        self._tools[tool.name] = tool
        return tool

    # -- inspection --------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Names of the registered tools."""
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        """Return the tool with this name, or None."""
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        """True if a tool with this name is registered."""
        return name in self._tools

    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def specifications(self) -> list[dict[str, Any]]:
        """Specs of all tools; these are sent to the model on every call."""
        return [tool.specification() for tool in self._tools.values()]

    # -- execution ---------------------------------------------------------

    def _truncate(self, text: str) -> str:
        """Cut text to the result limit, adding a note that tells the model what was cut."""
        if len(text) <= self.max_result_chars:
            return text
        kept = text[: self.max_result_chars]
        dropped = len(text) - self.max_result_chars
        return (
            f"{kept}\n\n[truncated: {dropped} more characters were cut. Narrow "
            f"the request — a line range, a tighter pattern — to see the rest.]"
        )

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Run one tool call and return its result; failures become error results, not exceptions."""
        started = time.monotonic()

        def finish(content: str, is_error: bool = False) -> ToolResult:
            """Build the (truncated) result and log the call."""
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
            # List the real tools so the model can pick the right one next turn.
            return finish(
                f"No tool named {call.name!r}. Available tools: "
                f"{', '.join(self.names)}.",
                is_error=True,
            )

        try:
            arguments = tool.params.model_validate(call.arguments)
        except ValidationError as exc:
            # Pydantic's message names the bad field and expected type; pass it on.
            return finish(
                f"Invalid arguments for {call.name!r}:\n{exc}", is_error=True
            )

        try:
            output = tool.handler(arguments)
        except Exception as exc:  # noqa: BLE001 - errors go back to the model
            return finish(f"{type(exc).__name__}: {exc}", is_error=True)

        return finish(output if output else "(the tool returned no output)")

    def dispatch_all(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Run every tool call from one model turn, in order (each call needs an answer)."""
        return [self.dispatch(call) for call in calls]


def results_to_messages(results: list[ToolResult]) -> list[dict[str, Any]]:
    """Turn tool results into history messages to add before the next model call."""
    return [
        {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": result.content,
            "is_error": result.is_error,
        }
        for result in results
    ]
