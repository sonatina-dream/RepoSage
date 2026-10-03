"""Context measurement and the budget, offline."""

import pytest

from reposage.context import (
    ContextBudget,
    estimate_context_tokens,
    estimate_message_tokens,
    estimate_tokens,
)
from reposage.providers import ToolCall
from reposage.tracing import TraceWriter, read_trace

from test_agent import call, make_agent, reply, registry, ran  # noqa: F401


def test_estimate_tokens_rounds_up():
    """Four characters per token, rounded up; only the empty string is zero."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_assistant_message_counts_its_tool_calls():
    """A tool call costs tokens even when the assistant wrote no text."""
    plain = {"role": "assistant", "content": "", "tool_calls": []}
    with_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [ToolCall("c1", "get_file", {"path": "src/a.py"})],
    }
    assert estimate_message_tokens(plain) == 0
    assert estimate_message_tokens(with_call) > 0


def test_context_includes_system_and_tool_specs():
    """The request is more than the history: system prompt and tool specs count too."""
    messages = [{"role": "user", "content": "x" * 40}]
    history_only = estimate_context_tokens(messages)
    assert history_only == 10
    assert estimate_context_tokens(messages, "s" * 40) == 20
    assert estimate_context_tokens(messages, "", [{"name": "t"}]) > history_only


def test_budget_threshold_and_validation():
    """The threshold is a fraction of the max, and nonsense values are rejected."""
    budget = ContextBudget(max_tokens=1000, compaction_threshold=0.8)
    assert budget.threshold_tokens == 800
    assert not budget.should_compact(799)
    assert budget.should_compact(800)
    with pytest.raises(ValueError):
        ContextBudget(max_tokens=0)
    with pytest.raises(ValueError):
        ContextBudget(compaction_threshold=1.5)


def test_trace_records_estimate_and_threshold_flag(registry, tmp_path):
    """Each step logs the estimated context size and whether it crossed the threshold."""
    tracer = TraceWriter(tmp_path / "run.jsonl")
    agent, _ = make_agent(
        registry,
        reply(calls=[call("c1", "a.py")]),
        reply(text="done"),
        tracer=tracer,
        context_budget=ContextBudget(max_tokens=2000, compaction_threshold=0.5),
    )
    agent.run("where?")

    records = read_trace(tracer.path)
    start = records[0]
    steps = [r for r in records if r["type"] == "step"]
    assert start["context_max_tokens"] == 2000 and start["context_threshold_tokens"] == 1000
    assert all(s["estimated_context_tokens"] > 0 for s in steps)
    # History only grows, so the second estimate is larger than the first.
    assert steps[1]["estimated_context_tokens"] > steps[0]["estimated_context_tokens"]
    assert all(isinstance(s["over_compaction_threshold"], bool) for s in steps)
