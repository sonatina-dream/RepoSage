"""The whole round trip, against a scripted provider. No API key, no network.

This is the test that would catch a broken phase 2 end to end: the model asks
for a tool, we run it, the result goes back matched by id, and the second
request carries the full history. A live model cannot be relied on to request a
tool on demand, so the provider is scripted instead — the same seam the phase 3
agent loop will be tested through.
"""

import pytest
from pydantic import BaseModel, Field

from reposage.config import Settings
from reposage.llm import BudgetExceeded, LLMClient
from reposage.providers import ProviderReply, TokenUsage, ToolCall
from reposage.tools import ToolRegistry, results_to_messages
from reposage.tools.registry import Tool


class LookupParams(BaseModel):
    path: str = Field(description="Path to read.")


class ScriptedProvider:
    """Replays canned replies and records exactly what it was sent."""

    name = "scripted"
    retryable_errors = ()

    def __init__(self, *replies: ProviderReply) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []

    def complete(self, **kwargs) -> ProviderReply:
        self.requests.append(kwargs)
        return self.replies.pop(0)


def reply(text="", tool_calls=None, stop_reason="end_turn") -> ProviderReply:
    return ProviderReply(
        text=text,
        usage=TokenUsage(input_tokens=100, output_tokens=20),
        stop_reason=stop_reason,
        tool_calls=tool_calls or [],
    )


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.add(
        Tool(
            name="get_file",
            description="Read a file.",
            params=LookupParams,
            handler=lambda p: f"1: contents of {p.path}",
        )
    )
    return reg


@pytest.fixture
def settings():
    return Settings(provider="deepseek", api_key="unused", spend_ceiling_usd=10.0)


def test_full_round_trip(registry, settings):
    call = ToolCall(id="call_1", name="get_file", arguments={"path": "app.py"})
    provider = ScriptedProvider(
        reply(tool_calls=[call], stop_reason="tool_use"),
        reply(text="It does X. app.py:1"),
    )
    client = LLMClient(settings, provider=provider)
    specifications = registry.specifications()

    history = [{"role": "user", "content": "What does app.py do?"}]
    first = client.complete(messages=history, tools=specifications)

    assert first.wants_tools
    assert first.stop_reason == "tool_use"

    results = registry.dispatch_all(first.tool_calls)
    assert results[0].tool_call_id == call.id  # matched by id, never by position
    assert "contents of app.py" in results[0].content

    history.append(
        {"role": "assistant", "content": first.text, "tool_calls": first.tool_calls}
    )
    history.extend(results_to_messages(results))

    second = client.complete(messages=history, tools=specifications)
    assert "app.py:1" in second.text

    # The second request must carry the whole conversation: the question, the
    # assistant turn *including its tool call*, and the result. Drop the middle
    # one and the result is an orphan that both real APIs reject.
    sent = provider.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool"]
    assert sent[1]["tool_calls"][0].id == "call_1"
    assert sent[2]["tool_call_id"] == "call_1"

    assert client.usage.calls == 2


def test_a_model_may_decline_to_use_tools(registry, settings):
    """Not a failure. Forcing a call would defeat the point of offering a choice."""
    provider = ScriptedProvider(reply(text="I can answer that directly."))
    client = LLMClient(settings, provider=provider)

    response = client.complete(prompt="What is 2+2?", tools=registry.specifications())
    assert not response.wants_tools
    assert registry.invocations == []


def test_a_failing_tool_still_produces_a_result_the_model_can_read(registry, settings):
    """The model's bad guess must be recoverable, not fatal."""
    bad = ToolCall(id="call_9", name="get_file", arguments={"wrong_field": "x"})
    provider = ScriptedProvider(
        reply(tool_calls=[bad], stop_reason="tool_use"),
        reply(text="Let me try a different path."),
    )
    client = LLMClient(settings, provider=provider)

    first = client.complete(prompt="Read something", tools=registry.specifications())
    results = registry.dispatch_all(first.tool_calls)

    assert results[0].is_error
    assert "path" in results[0].content  # names the field it was missing
    # And it is still a well-formed result, so the conversation can continue.
    assert results[0].tool_call_id == "call_9"


def test_tool_schemas_count_against_the_budget(registry):
    """They are sent on every call, so a guard that ignores them under-counts."""
    tight = Settings(provider="deepseek", api_key="unused", spend_ceiling_usd=1e-9)
    client = LLMClient(tight, provider=ScriptedProvider(reply()))

    with pytest.raises(BudgetExceeded):
        client.complete(prompt="hi", tools=registry.specifications())
