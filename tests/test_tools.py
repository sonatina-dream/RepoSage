"""The registry and the three tools. No API key, no network, no clone.

Everything here runs against a throwaway repository built in a temp directory,
which is what makes it fast and deterministic. The behaviour worth testing is
almost entirely the *failure* behaviour: a registry whose happy path works is
easy, and a registry that turns every failure into something the model can
recover from is the actual design.
"""

import json

import pytest
from pydantic import BaseModel, Field

from reposage.providers import ToolCall
from reposage.tools import ToolRegistry, build_repo_tools, results_to_messages
from reposage.tools.registry import Tool
from reposage.tools.repo_tools import PathEscapeError, _resolve_within


class EchoParams(BaseModel):
    text: str = Field(description="What to echo back.")
    times: int = Field(default=1, ge=1, le=5, description="How many times.")


def echo(params: EchoParams) -> str:
    return " ".join([params.text] * params.times)


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.add(Tool(name="echo", description="Echo text back.", params=EchoParams, handler=echo))
    return reg


@pytest.fixture
def repo(tmp_path):
    """A tiny fake repository: two source files and a directory to skip."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "routing.py").write_text(
        "\n".join(f"line {n} def solve_dependencies(): pass" if n == 3 else f"line {n}"
                  for n in range(1, 31)),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Demo\nsolve_dependencies is documented here.\n",
                                        encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("solve_dependencies\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def repo_registry(repo) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in build_repo_tools(repo, line_cap=10):
        reg.add(tool)
    return reg


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id=f"call_{name}", name=name, arguments=arguments)


# -- schema generation -----------------------------------------------------

def test_schema_is_generated_from_the_handler_s_own_model(registry):
    """One source of truth: rename a field and the advertised schema follows."""
    spec = registry.specifications()[0]
    assert spec["name"] == "echo"
    schema = spec["input_schema"]
    assert set(schema["properties"]) == {"text", "times"}
    assert schema["required"] == ["text"]
    # Per-field descriptions must survive into the schema — they are what the
    # model reads to decide what to put in each argument.
    assert "What to echo back." in json.dumps(schema)


# -- dispatch: every failure is a result, not an exception -----------------

def test_unknown_tool_returns_an_error_result_naming_the_alternatives(registry):
    result = registry.dispatch(call("get_file", path="x"))
    assert result.is_error
    assert "echo" in result.content  # tells the model what it could have called


def test_invalid_arguments_return_the_validator_s_own_message(registry):
    result = registry.dispatch(call("echo", text="hi", times=99))
    assert result.is_error
    assert "times" in result.content
    assert "less than or equal to 5" in result.content


def test_a_handler_that_raises_becomes_an_error_result(registry):
    def explode(params: EchoParams) -> str:
        raise KeyError("boom")

    registry.add(Tool(name="explode", description="Fails.", params=EchoParams, handler=explode))
    result = registry.dispatch(call("explode", text="x"))
    assert result.is_error
    assert "KeyError" in result.content


def test_results_are_truncated_visibly(registry):
    registry.max_result_chars = 50
    result = registry.dispatch(call("echo", text="x" * 200))
    assert "[truncated:" in result.content
    assert "Narrow the request" in result.content
    assert not result.is_error  # truncation is not a failure


def test_dispatch_all_answers_every_call_in_the_turn(registry):
    """Both APIs reject a turn with an unanswered tool call."""
    calls = [call("echo", text="a"), call("nope"), call("echo", text="b")]
    results = registry.dispatch_all(calls)
    assert len(results) == 3
    assert [r.tool_call_id for r in results] == [c.id for c in calls]
    assert [r.is_error for r in results] == [False, True, False]


def test_results_convert_to_neutral_history_entries(registry):
    messages = results_to_messages(registry.dispatch_all([call("echo", text="a")]))
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "call_echo"


# -- get_file --------------------------------------------------------------

def test_get_file_numbers_lines_and_states_the_range(repo_registry):
    result = repo_registry.dispatch(call("get_file", path="pkg/routing.py",
                                         start_line=2, end_line=4))
    assert not result.is_error
    assert "lines 2-4 of 30" in result.content
    assert "2: line 2" in result.content
    assert "1: line 1" not in result.content


def test_get_file_caps_an_unbounded_read_and_says_so(repo_registry):
    """A whole large file is paid for again on every later turn."""
    result = repo_registry.dispatch(call("get_file", path="pkg/routing.py"))
    assert "showing lines 1-10 of 30" in result.content
    assert "start_line" in result.content


def test_get_file_refuses_to_escape_the_repository(repo, repo_registry):
    with pytest.raises(PathEscapeError):
        _resolve_within(repo, "../../../etc/passwd")

    # And through the registry it is an error result, not a crash.
    result = repo_registry.dispatch(call("get_file", path="../../../etc/passwd"))
    assert result.is_error
    assert "outside the repository" in result.content


def test_get_file_missing_path_suggests_a_next_move(repo_registry):
    result = repo_registry.dispatch(call("get_file", path="pkg/mian.py"))
    assert "search_code" in result.content


# -- search_code -----------------------------------------------------------

def test_search_code_finds_matches_and_skips_dot_git(repo_registry):
    result = repo_registry.dispatch(call("search_code", pattern="solve_dependencies"))
    assert "pkg/routing.py:3" in result.content
    assert "README.md:2" in result.content
    assert ".git" not in result.content


def test_search_code_reports_that_it_capped_the_results(repo_registry):
    result = repo_registry.dispatch(call("search_code", pattern="line", max_results=5))
    assert "showing 5" in result.content


def test_search_code_glob_narrows_the_search(repo_registry):
    result = repo_registry.dispatch(
        call("search_code", pattern="solve_dependencies", glob="*.md")
    )
    assert "README.md" in result.content
    assert "routing.py" not in result.content


def test_search_code_rejects_a_bad_regex_without_crashing(repo_registry):
    result = repo_registry.dispatch(call("search_code", pattern="([unclosed"))
    assert "Invalid regular expression" in result.content


def test_search_code_says_nothing_found_rather_than_returning_empty(repo_registry):
    result = repo_registry.dispatch(call("search_code", pattern="zzz_not_present"))
    assert "No matches" in result.content
