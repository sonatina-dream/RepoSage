"""Translating one neutral history into two vendors' wire formats.

These are the tests that would have caught every porting bug in this project.
Each one pins a difference that produces either an API error or — worse —
silently wrong behaviour: a dropped system prompt, consecutive user turns, a
tool call that never gets executed.

No network and no API key: every method under test is static, so the providers
are exercised without being constructed.
"""

import json

import pytest

from reposage.providers import ToolCall
from reposage.providers.anthropic_provider import AnthropicProvider
from reposage.providers.base import ProviderContractError, looks_like_unparsed_tool_call
from reposage.providers.deepseek_provider import DeepSeekProvider

CALL = ToolCall(id="call_1", name="get_file", arguments={"path": "a.py"})

# One user question, one assistant turn that called two tools, two results.
HISTORY = [
    {"role": "user", "content": "How does routing work?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            CALL,
            ToolCall(id="call_2", name="search_code", arguments={"pattern": "route"}),
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "1: import x", "is_error": False},
    {"role": "tool", "tool_call_id": "call_2", "content": "no such file", "is_error": True},
]

TOOL_SPEC = [
    {
        "name": "get_file",
        "description": "Read a file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
    }
]


# -- Anthropic -------------------------------------------------------------

def test_anthropic_batches_tool_results_into_one_user_turn():
    """Two results, one message — because the API requires alternating roles.

    Emitting one user message per result produces consecutive user turns and a
    400 from the API. This is the least obvious rule in the whole translation.
    """
    wire = AnthropicProvider._to_wire(HISTORY)
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]

    blocks = wire[2]["content"]
    assert [b["type"] for b in blocks] == ["tool_result", "tool_result"]
    assert [b["tool_use_id"] for b in blocks] == ["call_1", "call_2"]
    # Anthropic has a real error flag; it must be carried, not flattened.
    assert blocks[0]["is_error"] is False
    assert blocks[1]["is_error"] is True


def test_anthropic_omits_an_empty_text_block():
    """A model calling a tool often says nothing first, and blank text is rejected."""
    blocks = AnthropicProvider._to_wire(HISTORY)[1]["content"]
    assert [b["type"] for b in blocks] == ["tool_use", "tool_use"]
    assert blocks[0]["input"] == {"path": "a.py"}  # already an object, not a string


def test_anthropic_keeps_text_and_tool_calls_in_one_content_list():
    history = [{"role": "assistant", "content": "Let me look.", "tool_calls": [CALL]}]
    blocks = AnthropicProvider._to_wire(history)[0]["content"]
    assert [b["type"] for b in blocks] == ["text", "tool_use"]


def test_anthropic_tool_spec_uses_input_schema():
    spec = AnthropicProvider._tools(TOOL_SPEC)[0]
    assert set(spec) == {"name", "description", "input_schema"}


# -- DeepSeek --------------------------------------------------------------

def test_deepseek_puts_the_system_prompt_in_the_message_list():
    """The porting bug that fails silently: no error, just a worse answer."""
    wire = DeepSeekProvider._to_wire(HISTORY, "be terse")
    assert wire[0] == {"role": "system", "content": "be terse"}


def test_deepseek_serialises_tool_arguments_to_a_json_string():
    """Anthropic sends an object here; OpenAI-style sends a string."""
    wire = DeepSeekProvider._to_wire(HISTORY, None)
    assistant = next(m for m in wire if m["role"] == "assistant")
    arguments = assistant["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"path": "a.py"}


def test_deepseek_gives_each_result_its_own_message_and_marks_errors_in_text():
    """There is no is_error field in this format, so the text has to carry it."""
    wire = DeepSeekProvider._to_wire(HISTORY, None)
    results = [m for m in wire if m["role"] == "tool"]
    assert len(results) == 2
    assert results[0]["content"] == "1: import x"
    assert results[1]["content"].startswith("ERROR: ")


def test_deepseek_tool_spec_wraps_the_same_schema_in_a_function_envelope():
    spec = DeepSeekProvider._tools(TOOL_SPEC)[0]
    assert spec["type"] == "function"
    # Different key, identical JSON Schema — which is why one registry serves both.
    assert spec["function"]["parameters"] == TOOL_SPEC[0]["input_schema"]


# -- the known DeepSeek defect --------------------------------------------

def test_a_bare_json_object_naming_a_tool_is_flagged():
    text = '{"name": "get_file", "arguments": {"path": "a.py"}}'
    assert looks_like_unparsed_tool_call(text, ["get_file"])


def test_prose_that_merely_discusses_a_tool_is_not_flagged():
    """The false positive that would matter most.

    RepoSage answers questions *about source code*. A model explaining what
    get_file does is not calling it — and a loose check would break exactly the
    questions this project exists to answer.
    """
    for text in [
        "You can use get_file to read a file, like get_file(path='a.py').",
        'Call it with {"name": "get_file"} — that is the JSON shape it expects.',
        "The registry exposes get_file, search_code and list_issues.",
    ]:
        assert not looks_like_unparsed_tool_call(text, ["get_file"])


def test_unknown_tool_names_in_json_are_not_flagged():
    assert not looks_like_unparsed_tool_call('{"name": "some_other_thing"}', ["get_file"])


def test_contract_check_raises_when_a_tool_call_arrives_as_text():
    with pytest.raises(ProviderContractError, match="serialised a tool call"):
        DeepSeekProvider._check_contract(
            text='{"name": "get_file", "arguments": {"path": "a.py"}}',
            tool_calls=[],
            finish_reason="stop",
            tools=TOOL_SPEC,
        )


def test_contract_check_raises_on_tool_calls_finish_with_no_calls():
    with pytest.raises(ProviderContractError, match="no tool_calls"):
        DeepSeekProvider._check_contract("", [], "tool_calls", TOOL_SPEC)


def test_contract_check_passes_a_normal_reply():
    DeepSeekProvider._check_contract("Routing works like this...", [], "stop", TOOL_SPEC)
    DeepSeekProvider._check_contract("", [CALL], "tool_calls", TOOL_SPEC)


# -- thinking mode --------------------------------------------------------

class _Sent(Exception):
    """Raised by the fake transport once the request has been captured."""


class _CapturingCompletions:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        raise _Sent


def _provider_with_capture() -> tuple[DeepSeekProvider, _CapturingCompletions]:
    provider = DeepSeekProvider(api_key="unused", timeout_s=1.0)
    completions = _CapturingCompletions()
    provider._client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})})()
    return provider, completions


@pytest.mark.parametrize("call", ["complete", "stream"])
def test_deepseek_requests_disable_thinking(call):
    """V4 thinks by default and bills the reasoning against max_tokens.

    Left on, a small max_tokens is spent entirely on hidden reasoning and the
    reply comes back empty with finish reason "length".
    """
    provider, completions = _provider_with_capture()
    request = dict(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        max_tokens=10,
        temperature=0.0,
    )
    with pytest.raises(_Sent):
        if call == "complete":
            provider.complete(**request)
        else:
            next(provider.stream(**request))

    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
