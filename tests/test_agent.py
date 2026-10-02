"""The agent loop against a scripted provider: no API key, no network."""

import pytest
from pydantic import BaseModel, Field

from reposage.agent import Agent
from reposage.config import Settings
from reposage.events import (
    BUDGET_EXCEEDED,
    FINAL_ANSWER,
    ITERATION_CAP,
    OUTPUT_TRUNCATED,
    TRANSPORT_ERROR,
    Final,
    ToolCallEvent,
    ToolResultEvent,
)
from reposage.llm import LLMClient
from reposage.providers import ProviderReply, TokenUsage, ToolCall
from reposage.tools import ToolRegistry
from reposage.tools.registry import Tool
from reposage.tracing import TraceWriter, read_trace


class PathParams(BaseModel):
    path: str = Field(description="Path to read.")


class ScriptedProvider:
    """Replays canned replies (or raises canned errors) and records every request."""

    name = "scripted"
    retryable_errors = (ConnectionError,)

    def __init__(self, *replies) -> None:
        """Store the replies to hand back, in order."""
        self.replies = list(replies)
        self.requests: list[dict] = []

    def complete(self, **kwargs) -> ProviderReply:
        """Record the request; return or raise the next canned item."""
        self.requests.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def reply(text="", calls=None, stop_reason=None) -> ProviderReply:
    """Build a canned reply; it stops for tools if calls are given, else ends the turn."""
    calls = calls or []
    return ProviderReply(
        text=text,
        usage=TokenUsage(input_tokens=100, output_tokens=20),
        stop_reason=stop_reason or ("tool_use" if calls else "end_turn"),
        tool_calls=calls,
    )


def call(id_, path="a.py", name="get_file") -> ToolCall:
    """Build a tool call with one path argument."""
    return ToolCall(id=id_, name=name, arguments={"path": path})


@pytest.fixture
def ran():
    """Collects the paths the fake tools actually executed."""
    return []


@pytest.fixture
def registry(ran):
    """A registry with a safe get_file and a risky delete_file."""
    reg = ToolRegistry()

    def get_file(p):
        ran.append(("get_file", p.path))
        if p.path == "missing.py":
            raise FileNotFoundError("no such file")
        return f"1: contents of {p.path}"

    def delete_file(p):
        ran.append(("delete_file", p.path))
        return "deleted"

    reg.add(Tool("get_file", "Read.", PathParams, get_file))
    reg.add(Tool("delete_file", "Delete.", PathParams, delete_file, risk="risky"))
    return reg


def make_agent(registry, *replies, ceiling=10.0, max_retries=0, **kwargs):
    """An Agent over a scripted provider; returns (agent, provider)."""
    provider = ScriptedProvider(*replies)
    settings = Settings(
        provider="deepseek", api_key="x", spend_ceiling_usd=ceiling, max_retries=max_retries
    )
    return Agent(LLMClient(settings, provider=provider), registry, **kwargs), provider


def test_multi_step_run_ends_with_final_answer(registry, ran):
    """Two tool steps, then an answer: the loop feeds each result back and stops on its own."""
    agent, provider = make_agent(
        registry,
        reply(calls=[call("c1", "a.py")]),
        reply(calls=[call("c2", "b.py")]),
        reply(text="It lives in b.py:1."),
    )
    result = agent.run("where?")

    assert result.exit_reason == FINAL_ANSWER and result.ok
    assert result.answer == "It lives in b.py:1."
    assert result.steps == 3 and result.tool_calls == 2
    assert ran == [("get_file", "a.py"), ("get_file", "b.py")]
    # The third request carries the whole history: question, 2 x (assistant + tool).
    assert len(provider.requests[2]["messages"]) == 5
    assert isinstance(result.events[-1], Final)
    assert result.cost_usd > 0


def test_event_order(registry):
    """Events arrive as step_started, tool_call, tool_result, ..., final."""
    agent, _ = make_agent(registry, reply(calls=[call("c1")]), reply(text="done"))
    types = [e.type for e in agent.run("q").events]
    assert types == [
        "step_started", "tool_call", "tool_result", "step_started", "text_delta", "final",
    ]


def test_iteration_cap(registry):
    """A model that never stops asking for tools is cut off, with the reason recorded."""
    replies = [reply(calls=[call(f"c{i}", f"f{i}.py")]) for i in range(5)]
    agent, provider = make_agent(registry, *replies, max_iterations=3)
    result = agent.run("q")

    assert result.exit_reason == ITERATION_CAP
    assert result.steps == 3 and len(provider.requests) == 3


def test_output_truncated(registry):
    """A reply cut off by max_tokens ends the run instead of acting on partial output."""
    agent, _ = make_agent(registry, reply(text="The answer is par", stop_reason="max_tokens"))
    result = agent.run("q")

    assert result.exit_reason == OUTPUT_TRUNCATED
    assert result.answer == "The answer is par"


def test_run_cost_limit(registry):
    """The per-run cost limit stops the loop before the next model call."""
    agent, provider = make_agent(
        registry, reply(calls=[call("c1")]), reply(text="never"), max_cost_usd=0.0
    )
    result = agent.run("q")

    assert result.exit_reason == BUDGET_EXCEEDED
    assert provider.requests == []


def test_client_spend_ceiling_ends_run(registry):
    """Hitting the client's ceiling mid-run is reported as budget_exceeded, not raised."""
    agent, _ = make_agent(registry, reply(text="x"), ceiling=0.0)
    assert agent.run("q").exit_reason == BUDGET_EXCEEDED


def test_tool_error_goes_back_to_model_and_run_recovers(registry):
    """A tool that raises becomes an error result; the model sees it and tries again."""
    agent, provider = make_agent(
        registry,
        reply(calls=[call("c1", "missing.py")]),
        reply(calls=[call("c2", "real.py")]),
        reply(text="found real.py:1"),
    )
    result = agent.run("q")

    assert result.exit_reason == FINAL_ANSWER
    first_tool_msg = provider.requests[1]["messages"][-1]
    assert first_tool_msg["is_error"] is True
    assert "FileNotFoundError" in first_tool_msg["content"]


def test_unknown_tool_is_an_error_result_not_a_crash(registry):
    """A hallucinated tool name comes back as an error naming the real tools."""
    agent, provider = make_agent(
        registry, reply(calls=[call("c1", name="grep")]), reply(text="ok")
    )
    result = agent.run("q")

    assert result.ok
    assert "Available tools" in provider.requests[1]["messages"][-1]["content"]


def test_loop_guard_blocks_identical_repeat(registry, ran):
    """The same call with the same arguments is not run twice; the model is told why."""
    agent, provider = make_agent(
        registry,
        reply(calls=[call("c1", "a.py")]),
        reply(calls=[call("c2", "a.py")]),
        reply(text="done"),
    )
    result = agent.run("q")

    assert ran == [("get_file", "a.py")]
    results = [e for e in result.events if isinstance(e, ToolResultEvent)]
    assert [r.note for r in results] == ["", "repeat"]
    assert "already made this exact call" in provider.requests[2]["messages"][-1]["content"]


def test_loop_guard_ignores_argument_order_but_not_values(registry, ran):
    """Different arguments are a different call and run normally."""
    agent, _ = make_agent(
        registry,
        reply(calls=[call("c1", "a.py")]),
        reply(calls=[call("c2", "b.py")]),
        reply(text="done"),
    )
    agent.run("q")
    assert ran == [("get_file", "a.py"), ("get_file", "b.py")]


def test_risky_tool_denied_without_callback(registry, ran):
    """With no approval callback a risky tool never runs, and the model is told it was denied."""
    agent, provider = make_agent(
        registry, reply(calls=[call("c1", name="delete_file")]), reply(text="ok")
    )
    result = agent.run("q")

    assert ran == []
    events = [e for e in result.events if isinstance(e, ToolCallEvent)]
    assert events[0].needs_approval is True
    tool_msg = provider.requests[1]["messages"][-1]
    assert tool_msg["is_error"] and "denied" in tool_msg["content"]


def test_risky_tool_approved_runs(registry, ran):
    """An approval callback that says yes lets the risky tool run."""
    seen = []
    agent, _ = make_agent(
        registry,
        reply(calls=[call("c1", name="delete_file")]),
        reply(text="ok"),
        approve=lambda c: seen.append(c.name) or True,
    )
    agent.run("q")

    assert ran == [("delete_file", "a.py")] and seen == ["delete_file"]


def test_safe_tools_never_ask_for_approval(registry):
    """The callback is only consulted for risky tools."""
    asked = []
    agent, _ = make_agent(
        registry,
        reply(calls=[call("c1")]),
        reply(text="ok"),
        approve=lambda c: asked.append(c) or False,
    )
    agent.run("q")
    assert asked == []


def test_persistent_transport_failure_ends_run_after_retries(registry, monkeypatch):
    """Transport errors are retried by the client; if they persist the run ends, not crashes."""
    monkeypatch.setattr("reposage.llm.time.sleep", lambda s: None)
    agent, provider = make_agent(
        registry, *[ConnectionError("down")] * 3, max_retries=2
    )
    result = agent.run("q")

    assert result.exit_reason == TRANSPORT_ERROR
    assert len(provider.requests) == 3  # 1 try + 2 retries


def test_transient_transport_failure_is_retried_invisibly(registry, monkeypatch):
    """One dropped connection followed by success never reaches the model or the events."""
    monkeypatch.setattr("reposage.llm.time.sleep", lambda s: None)
    agent, _ = make_agent(registry, ConnectionError("blip"), reply(text="fine"), max_retries=2)
    result = agent.run("q")

    assert result.ok and result.steps == 1


def test_trace_has_one_record_per_step(registry, tmp_path):
    """The JSONL trace holds run_start, one record per step with tokens and tools, and run_end."""
    tracer = TraceWriter(tmp_path / "t.jsonl")
    agent, _ = make_agent(
        registry, reply(calls=[call("c1")]), reply(text="done"), tracer=tracer
    )
    agent.run("q")
    records = read_trace(tracer.path)

    assert [r["type"] for r in records] == ["run_start", "step", "step", "run_end"]
    step1 = records[1]
    assert step1["input_tokens"] == 100 and step1["cost_usd"] > 0
    assert step1["tool_calls"][0]["name"] == "get_file"
    assert step1["tool_results"][0]["content"].startswith("1: contents")
    assert records[-1]["exit_reason"] == FINAL_ANSWER


def test_event_to_dict_is_json_ready():
    """Events serialise with their type name, for the phase 8 API."""
    event = ToolCallEvent(step=1, id="c1", name="get_file", arguments={"path": "a"})
    assert event.to_dict()["type"] == "tool_call"
