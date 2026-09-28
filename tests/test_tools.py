"""Tests for the tool registry and the get_file / search_code tools, on a tiny fake repo."""

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
    """Return the text repeated `times` times."""
    return " ".join([params.text] * params.times)


@pytest.fixture
def registry() -> ToolRegistry:
    """A registry with one echo tool."""
    reg = ToolRegistry()
    reg.add(Tool(name="echo", description="Echo text back.", params=EchoParams, handler=echo))
    return reg


@pytest.fixture
def repo(tmp_path):
    """A tiny fake repository: two source files and a .git folder to skip."""
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
    """A registry with the repo tools on the fake repo, capped at 10 lines."""
    reg = ToolRegistry()
    for tool in build_repo_tools(repo, line_cap=10):
        reg.add(tool)
    return reg


def call(name: str, **arguments) -> ToolCall:
    """Build a ToolCall whose id is based on the tool name."""
    return ToolCall(id=f"call_{name}", name=name, arguments=arguments)


# -- schema generation -----------------------------------------------------

def test_schema_is_generated_from_the_handler_s_own_model(registry):
    """Checks that the tool schema comes from the handler's Pydantic model."""
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
    """Checks that an unknown tool gives an error result listing the real tools."""
    result = registry.dispatch(call("get_file", path="x"))
    assert result.is_error
    assert "echo" in result.content  # tells the model what it could have called


def test_invalid_arguments_return_the_validator_s_own_message(registry):
    """Checks that bad arguments return Pydantic's error message as an error result."""
    result = registry.dispatch(call("echo", text="hi", times=99))
    assert result.is_error
    assert "times" in result.content
    assert "less than or equal to 5" in result.content


def test_a_handler_that_raises_becomes_an_error_result(registry):
    """Checks that an exception inside a tool becomes an error result."""
    def explode(params: EchoParams) -> str:
        """Always raise, to simulate a broken tool."""
        raise KeyError("boom")

    registry.add(Tool(name="explode", description="Fails.", params=EchoParams, handler=explode))
    result = registry.dispatch(call("explode", text="x"))
    assert result.is_error
    assert "KeyError" in result.content


def test_results_are_truncated_visibly(registry):
    """Checks that long results are cut with a note, and not marked as errors."""
    registry.max_result_chars = 50
    result = registry.dispatch(call("echo", text="x" * 200))
    assert "[truncated:" in result.content
    assert "Narrow the request" in result.content
    assert not result.is_error  # truncation is not a failure


def test_dispatch_all_answers_every_call_in_the_turn(registry):
    """Checks that every call in a turn gets its own result, in order."""
    calls = [call("echo", text="a"), call("nope"), call("echo", text="b")]
    results = registry.dispatch_all(calls)
    assert len(results) == 3
    assert [r.tool_call_id for r in results] == [c.id for c in calls]
    assert [r.is_error for r in results] == [False, True, False]
    assert (results[0].content, results[2].content) == ("a", "b")  # each call answered, in order


def test_results_convert_to_neutral_history_entries(registry):
    """Checks that results become 'tool' history messages."""
    messages = results_to_messages(registry.dispatch_all([call("echo", text="a")]))
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "call_echo"


# -- get_file --------------------------------------------------------------

def test_get_file_numbers_lines_and_states_the_range(repo_registry):
    """Checks that get_file numbers lines and states the range shown."""
    result = repo_registry.dispatch(call("get_file", path="pkg/routing.py",
                                         start_line=2, end_line=4))
    assert not result.is_error
    assert "lines 2-4 of 30" in result.content
    assert "2: line 2" in result.content
    assert "1: line 1" not in result.content


def test_get_file_caps_an_unbounded_read_and_says_so(repo_registry):
    """Checks that reading a large file without a range is capped, with a note."""
    result = repo_registry.dispatch(call("get_file", path="pkg/routing.py"))
    assert "showing lines 1-10 of 30" in result.content
    assert "start_line" in result.content


def test_get_file_refuses_to_escape_the_repository(repo, repo_registry):
    """Checks that paths leading outside the repository are refused."""
    with pytest.raises(PathEscapeError):
        _resolve_within(repo, "../../../etc/passwd")

    # And through the registry it is an error result, not a crash.
    result = repo_registry.dispatch(call("get_file", path="../../../etc/passwd"))
    assert result.is_error
    assert "outside the repository" in result.content


def test_get_file_missing_path_suggests_a_next_move(repo_registry):
    """Checks that a missing file gets a hint to use search_code."""
    result = repo_registry.dispatch(call("get_file", path="pkg/mian.py"))
    assert "search_code" in result.content


# -- search_code -----------------------------------------------------------

def test_search_code_finds_matches_and_skips_dot_git(repo_registry):
    """Checks that search_code finds matches and skips the .git folder."""
    result = repo_registry.dispatch(call("search_code", pattern="solve_dependencies"))
    assert "pkg/routing.py:3" in result.content
    assert "README.md:2" in result.content
    assert ".git" not in result.content


def test_search_code_reports_that_it_capped_the_results(repo_registry):
    """Checks that search_code says when it shows only some matches."""
    result = repo_registry.dispatch(call("search_code", pattern="line", max_results=5))
    assert "showing 5" in result.content


def test_search_code_glob_narrows_the_search(repo_registry):
    """Checks that a glob limits which files are searched."""
    result = repo_registry.dispatch(
        call("search_code", pattern="solve_dependencies", glob="*.md")
    )
    assert "README.md" in result.content
    assert "routing.py" not in result.content


def test_search_code_rejects_a_bad_regex_without_crashing(repo_registry):
    """Checks that an invalid regex returns a message instead of crashing."""
    result = repo_registry.dispatch(call("search_code", pattern="([unclosed"))
    assert "Invalid regular expression" in result.content


def test_search_code_says_nothing_found_rather_than_returning_empty(repo_registry):
    """Checks that no matches gives a clear 'No matches' message."""
    result = repo_registry.dispatch(call("search_code", pattern="zzz_not_present"))
    assert "No matches" in result.content
