"""Measuring what the model will see, before we send it.

The provider reports exact input tokens only after a call. To decide whether to shrink
the history before a call, we need an estimate now. English and code average roughly
four characters per token, which is close enough to compare against a budget; the exact
count the provider returns afterwards is logged next to it so the estimate can be checked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from .providers import Message

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the tokens in `text`, rounding up so an empty string is the only zero."""
    return -(-len(text) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: Message) -> int:
    """Estimate one history message, counting an assistant message's tool calls too."""
    total = estimate_tokens(message.get("content") or "")
    for call in message.get("tool_calls") or []:
        total += estimate_tokens(call.name) + estimate_tokens(
            json.dumps(call.arguments, default=str)
        )
    return total


def estimate_context_tokens(
    messages: Sequence[Message],
    system: str = "",
    tools: Sequence[dict[str, Any]] | None = None,
) -> int:
    """Estimate a whole request: system prompt, tool specs and every history message."""
    total = estimate_tokens(system)
    if tools:
        total += estimate_tokens(json.dumps(list(tools), default=str))
    return total + sum(estimate_message_tokens(m) for m in messages)


@dataclass(frozen=True)
class ContextBudget:
    """How big a request may get, and the point at which compaction should start.

    `compaction_threshold` is a fraction of `max_tokens`: starting to shrink at 0.8
    leaves headroom for the next reply and tool result instead of hitting the wall.
    """

    max_tokens: int = 30_000
    compaction_threshold: float = 0.8

    def __post_init__(self) -> None:
        """Reject values that would make the threshold meaningless."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 < self.compaction_threshold <= 1:
            raise ValueError("compaction_threshold must be in (0, 1]")

    @property
    def threshold_tokens(self) -> int:
        """The estimated size at which compaction should kick in."""
        return int(self.max_tokens * self.compaction_threshold)

    def should_compact(self, estimated_tokens: int) -> bool:
        """True once the estimate reaches the compaction threshold."""
        return estimated_tokens >= self.threshold_tokens
