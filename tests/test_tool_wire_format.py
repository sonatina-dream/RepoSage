"""Tests that one neutral history is translated correctly into each vendor's wire format."""

import json
from types import SimpleNamespace

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
    """Checks that consecutive tool results share one user message on Anthropic."""
    wire = AnthropicProvider._to_wire(HISTORY)
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]

    blocks = wire[2]["content"]
    assert [b["type"] for b in blocks] == ["tool_result", "tool_result"]
    assert [b["tool_use_id"] for b in blocks] == ["call_1", "call_2"]
    # Anthropic has a real error flag; it must be carried, not flattened.
    assert blocks[0]["is_error"] is False
    assert blocks[1]["is_error"] is True


def test_anthropic_omits_an_empty_text_block():
    """Checks that an empty assistant text is left out, since Anthropic rejects it."""
    blocks = AnthropicProvider._to_wire(HISTORY)[1]["content"]
    assert [b["type"] for b in blocks] == ["tool_use", "tool_use"]
    assert blocks[0]["input"] == {"path": "a.py"}  # already an object, not a string


def test_anthropic_keeps_text_and_tool_calls_in_one_content_list():
    """Checks that assistant text and tool calls sit in one content list."""
    history = [{"role": "assistant", "content": "Let me look.", "tool_calls": [CALL]}]
    blocks = AnthropicProvider._to_wire(history)[0]["content"]
    assert [b["type"] for b in blocks] == ["text", "tool_use"]


def test_anthropic_tool_spec_uses_input_schema():
    """Checks that Anthropic tool specs carry the schema as input_schema."""
    spec = AnthropicProvider._tools(TOOL_SPEC)[0]
    assert set(spec) == {"name", "description", "input_schema"}


# -- DeepSeek --------------------------------------------------------------

def test_deepseek_puts_the_system_prompt_in_the_message_list():
    """Checks that DeepSeek gets the system prompt as the first message."""
    wire = DeepSeekProvider._to_wire(HISTORY, "be terse")
    assert wire[0] == {"role": "system", "content": "be terse"}


def test_deepseek_serialises_tool_arguments_to_a_json_string():
    """Checks that DeepSeek tool-call arguments are sent as a JSON string."""
    wire = DeepSeekProvider._to_wire(HISTORY, None)
    assistant = next(m for m in wire if m["role"] == "assistant")
    arguments = assistant["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"path": "a.py"}


def test_deepseek_gives_each_result_its_own_message_and_marks_errors_in_text():
    """Checks that each DeepSeek tool result is its own message, with errors marked in the text."""
    wire = DeepSeekProvider._to_wire(HISTORY, None)
    results = [m for m in wire if m["role"] == "tool"]
    assert len(results) == 2
    assert results[0]["content"] == "1: import x"
    assert results[1]["content"].startswith("ERROR: ")


def test_deepseek_tool_spec_wraps_the_same_schema_in_a_function_envelope():
    """Checks that DeepSeek tool specs wrap the same schema in a function envelope."""
    spec = DeepSeekProvider._tools(TOOL_SPEC)[0]
    assert spec["type"] == "function"
    # Different key, identical JSON Schema — which is why one registry serves both.
    assert spec["function"]["parameters"] == TOOL_SPEC[0]["input_schema"]


# -- the known DeepSeek defect --------------------------------------------

def test_a_bare_json_object_naming_a_tool_is_flagged():
    """Checks that a reply that is only a JSON tool call is flagged."""
    text = '{"name": "get_file", "arguments": {"path": "a.py"}}'
    assert looks_like_unparsed_tool_call(text, ["get_file"])


def test_prose_that_merely_discusses_a_tool_is_not_flagged():
    """Checks that text merely talking about a tool is not flagged."""
    for text in [
        "You can use get_file to read a file, like get_file(path='a.py').",
        'Call it with {"name": "get_file"} — that is the JSON shape it expects.',
        "The registry exposes get_file, search_code and list_issues.",
    ]:
        assert not looks_like_unparsed_tool_call(text, ["get_file"])


def test_unknown_tool_names_in_json_are_not_flagged():
    """Checks that JSON naming a tool we didn't offer is not flagged."""
    assert not looks_like_unparsed_tool_call('{"name": "some_other_thing"}', ["get_file"])


def test_contract_check_raises_when_a_tool_call_arrives_as_text():
    """Checks that a tool call sent as text raises ProviderContractError."""
    with pytest.raises(ProviderContractError, match="serialised a tool call"):
        DeepSeekProvider._check_contract(
            text='{"name": "get_file", "arguments": {"path": "a.py"}}',
            tool_calls=[],
            finish_reason="stop",
            tools=TOOL_SPEC,
        )


def test_contract_check_raises_on_tool_calls_finish_with_no_calls():
    """Checks that finish_reason 'tool_calls' with no calls raises."""
    with pytest.raises(ProviderContractError, match="no tool_calls"):
        DeepSeekProvider._check_contract("", [], "tool_calls", TOOL_SPEC)


def test_contract_check_passes_a_normal_reply():
    """Checks that normal text and normal tool calls pass the contract check."""
    DeepSeekProvider._check_contract("Routing works like this...", [], "stop", TOOL_SPEC)
    DeepSeekProvider._check_contract("", [CALL], "tool_calls", TOOL_SPEC)


# -- thinking mode --------------------------------------------------------

class _Sent(Exception):
    """Raised by the fake transport once it has captured the request."""


class _CapturingCompletions:
    """A fake chat.completions object that records the request, then stops."""
    def __init__(self) -> None:
        """Start with no captured request."""
        self.kwargs: dict = {}

    def create(self, **kwargs):
        """Save the request arguments, then raise _Sent."""
        self.kwargs = kwargs
        raise _Sent


def _provider_with_capture() -> tuple[DeepSeekProvider, _CapturingCompletions]:
    """Build a DeepSeekProvider whose API client is the capturing fake."""
    provider = DeepSeekProvider(api_key="unused", timeout_s=1.0)
    completions = _CapturingCompletions()
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider, completions


@pytest.mark.parametrize("call", ["complete", "stream"])
def test_deepseek_requests_disable_thinking(call):
    """Checks that DeepSeek requests turn thinking off (it would use up max_tokens)."""
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


def test_anthropic_request_kwargs_match_the_sdk_signature():
    """Checks that every request field is accepted by the installed SDK's create() and stream()."""
    import inspect

    from anthropic.resources.messages import Messages

    provider = AnthropicProvider.__new__(AnthropicProvider)
    kwargs = provider._kwargs("m", [], "sys", 10, 0.0)
    assert kwargs["extra_body"] == {"temperature": 0.0}
    for method in (Messages.create, Messages.stream):
        inspect.signature(method).bind(None, **kwargs)
