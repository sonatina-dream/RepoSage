"""End-to-end tool-calling round trip against a scripted provider: no API key, no network."""

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
    """A fake provider that replays canned replies and records every request."""

    name = "scripted"
    retryable_errors = ()

    def __init__(self, *replies: ProviderReply) -> None:
        """Store the replies to hand back, in order."""
        self.replies = list(replies)
        self.requests: list[dict] = []

    def complete(self, **kwargs) -> ProviderReply:
        """Record the request and return the next canned reply."""
        self.requests.append(kwargs)
        return self.replies.pop(0)


def reply(text="", tool_calls=None, stop_reason="end_turn") -> ProviderReply:
    """Build a canned ProviderReply with fixed token counts."""
    return ProviderReply(
        text=text,
        usage=TokenUsage(input_tokens=100, output_tokens=20),
        stop_reason=stop_reason,
        tool_calls=tool_calls or [],
    )


@pytest.fixture
def registry():
    """A registry with one fake get_file tool."""
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
    """Settings for a fake DeepSeek client with a $10 ceiling."""
    return Settings(provider="deepseek", api_key="unused", spend_ceiling_usd=10.0)


def test_full_round_trip(registry, settings):
    """Checks the whole loop: tool request, execution, and results sent back matched by id."""
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
    """Checks that answering without tools is treated as a normal reply."""
    provider = ScriptedProvider(reply(text="I can answer that directly."))
    client = LLMClient(settings, provider=provider)

    response = client.complete(prompt="What is 2+2?", tools=registry.specifications())
    assert not response.wants_tools
    assert registry.invocations == []


def test_a_failing_tool_still_produces_a_result_the_model_can_read(registry, settings):
    """Checks that a failing tool becomes an error result, not a crash."""
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
    """Checks that the budget guard counts tool schemas, which are sent on every call."""
    # Haiku is flat-rate, so this holds at any hour: the bare prompt projects
    # $0.000005 (one output token); the schemas add ~50 input tokens (~$0.00005).
    tight = Settings(provider="anthropic", api_key="unused", spend_ceiling_usd=2e-5)
    client = LLMClient(tight, provider=ScriptedProvider(reply()))

    with pytest.raises(BudgetExceeded):
        client.complete(prompt="hi", max_tokens=1, tools=registry.specifications())
    client.complete(prompt="hi", max_tokens=1)  # the same request without schemas fits
