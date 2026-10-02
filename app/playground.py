"""RepoSage playground -- a browser UI for trying what phases 0 to 2 built.

    streamlit run app/playground.py

Three tabs: a streaming chat (phase 0), structured extraction with its retry
attempts (phase 1), and a tool-calling round trip, step by step (phase 2). The
sidebar shows spend against the ceiling. This is a learning aid, not the
production UI planned for phase 8.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reposage.config import Settings, format_usd  # noqa: E402
from reposage.extraction import ExtractionError, IssueSummary, extract  # noqa: E402
from reposage.llm import BudgetExceeded, LLMClient  # noqa: E402
from reposage.repo import DEFAULT_REPO  # noqa: E402
from reposage.tools import build_default_registry, results_to_messages  # noqa: E402

TOOL_SYSTEM_PROMPT = (
    "You answer questions about a source code repository. Use the tools to read it "
    "rather than answering from memory. Cite every claim as `path/to/file.py:123`. "
    "If the tools do not give you enough, say what is missing instead of guessing."
)

st.set_page_config(page_title="RepoSage playground", page_icon="🔎", layout="wide")


def get_client(provider: str) -> LLMClient:
    """Return this session's client for a provider, so spend accumulates across actions."""
    clients = st.session_state.setdefault("clients", {})
    if provider not in clients:
        clients[provider] = LLMClient(Settings.from_env(provider=provider))
    return clients[provider]


def show_cost(response) -> None:
    """Show a one-line token and cost caption for a reply."""
    cached = f" ({response.cached_input_tokens} cached)" if response.cached_input_tokens else ""
    st.caption(
        f"{response.model} · {response.input_tokens} in{cached} / "
        f"{response.output_tokens} out · {format_usd(response.cost_usd)} · "
        f"stop: {response.stop_reason}"
    )


# -- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.title("🔎 RepoSage")
    provider = st.radio("Provider", ["deepseek", "anthropic"], horizontal=True)
    client = get_client(provider)
    st.caption(f"Model: `{client.settings.model}`")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.0, 0.1)
    st.divider()
    st.metric("Spent this session", format_usd(client.usage.cost_usd))
    st.caption(
        f"{client.usage.summary()}\n\nBudget left: {format_usd(client.remaining_budget_usd)}"
    )

chat_tab, extract_tab, tools_tab = st.tabs(
    ["💬 Chat (phase 0)", "🧾 Extraction (phase 1)", "🛠️ Tool calling (phase 2)"]
)

# -- phase 0: chat ---------------------------------------------------------
with chat_tab:
    st.write(
        "The API is stateless: this page re-sends the whole conversation on every "
        "turn, which is why input tokens grow."
    )
    system = st.text_input("System prompt", "You explain engineering concepts concisely.")
    history = st.session_state.setdefault("chat", [])
    if st.button("Clear conversation"):
        history.clear()
        st.rerun()
    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Say something", key="chat_input"):
        history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(
                    client.stream(messages=history, system=system, temperature=temperature)
                )
                history.append({"role": "assistant", "content": reply})
            except BudgetExceeded as exc:
                history.pop()
                st.error(str(exc))
        st.rerun()

# -- phase 1: extraction ---------------------------------------------------
with extract_tab:
    st.write("Paste a GitHub issue thread. The model must return a valid `IssueSummary`.")
    text = st.text_area("Issue text", height=220, placeholder="Issue #123: ...")
    if st.button("Extract", disabled=not text.strip()):
        try:
            with st.spinner("Extracting..."):
                before = client.usage.calls
                summary = extract(client, IssueSummary, text)
            st.success(f"Valid record after {client.usage.calls - before} attempt(s)")
            st.json(summary.model_dump(mode="json"))
        except ExtractionError as exc:
            st.error(str(exc))
            for number, raw in enumerate(exc.attempts, 1):
                with st.expander(f"Attempt {number} raw output"):
                    st.code(raw)
        except BudgetExceeded as exc:
            st.error(str(exc))
    with st.expander("The schema the model is given"):
        st.json(IssueSummary.model_json_schema())

# -- phase 2: tool calling -------------------------------------------------
with tools_tab:
    repo = st.text_input("Repository", DEFAULT_REPO)
    question = st.text_area(
        "Question",
        "What does the function that resolves a request's dependencies do, and "
        "where is it defined? Cite the file and line.",
    )
    if st.button("Run one round trip"):
        try:
            with st.spinner("Cloning (first run only) and asking..."):
                registry, clone = build_default_registry(repo)
            specs = registry.specifications()

            history = [{"role": "user", "content": question}]
            with st.status("Turn 1: question + tool schemas", expanded=True) as status:
                first = client.complete(
                    messages=history, system=TOOL_SYSTEM_PROMPT, tools=specs,
                    max_tokens=1024, temperature=temperature,
                )
                show_cost(first)
                if first.text.strip():
                    st.write(first.text)
                for call in first.tool_calls:
                    st.code(f"{call.name}({json.dumps(call.arguments)})", language="python")
                status.update(label=f"Turn 1: model asked for {len(first.tool_calls)} tool(s)")

            if not first.wants_tools:
                st.subheader("Answer (no tools needed)")
                st.write(first.text)
            else:
                results = registry.dispatch_all(first.tool_calls)
                with st.status("We run the tools -- the model runs nothing", expanded=True):
                    for inv in registry.invocations:
                        icon = "❌" if inv.result.is_error else "✅"
                        st.markdown(f"{icon} **{inv.name}** · {inv.duration_s * 1000:.0f} ms")
                        st.code(inv.result.content[:1500])

                history.append(
                    {"role": "assistant", "content": first.text, "tool_calls": first.tool_calls}
                )
                history.extend(results_to_messages(results))
                second = client.complete(
                    messages=history, system=TOOL_SYSTEM_PROMPT, tools=specs,
                    max_tokens=1024, temperature=temperature,
                )
                st.subheader("Answer")
                if second.wants_tools:
                    st.info(
                        "The model wants more tools: "
                        + ", ".join(c.name for c in second.tool_calls)
                        + ". One round trip only until the phase 3 loop."
                    )
                st.write(second.text)
                show_cost(second)
                st.caption(
                    f"Turn 2 sent {second.input_tokens - first.input_tokens} more input "
                    "tokens than turn 1."
                )
        except BudgetExceeded as exc:
            st.error(str(exc))
        except Exception as exc:  # surface clone/network errors in the page, not a traceback
            st.error(f"{type(exc).__name__}: {exc}")
    if st.checkbox("Show the tool schemas the model receives"):
        try:
            st.json(build_default_registry(repo)[0].specifications())
        except Exception as exc:
            st.warning(str(exc))
