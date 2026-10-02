"""RepoSage playground -- a browser UI for trying what phases 0 to 3 built.

    streamlit run app/playground.py

Four tabs: a streaming chat (phase 0), structured extraction with its retry
attempts (phase 1), a tool-calling round trip, step by step (phase 2), and the
full agent loop with its live event stream and trace (phase 3). The sidebar
shows spend against the ceiling. This is a learning aid, not the
production UI planned for phase 8.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reposage.agent import Agent  # noqa: E402
from reposage.config import Settings, format_usd  # noqa: E402
from reposage.events import (  # noqa: E402
    Final, StepStarted, TextDelta, ToolCallEvent, ToolResultEvent,
)
from reposage.extraction import ExtractionError, IssueSummary, extract  # noqa: E402
from reposage.llm import BudgetExceeded, LLMClient  # noqa: E402
from reposage.repo import DEFAULT_REPO  # noqa: E402
from reposage.tools import build_default_registry, results_to_messages  # noqa: E402
from reposage.tracing import TraceWriter, new_trace_path, read_trace  # noqa: E402

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

chat_tab, extract_tab, tools_tab, agent_tab = st.tabs(
    [
        "💬 Chat (phase 0)",
        "🧾 Extraction (phase 1)",
        "🛠️ Tool calling (phase 2)",
        "🤖 Agent (phase 3)",
    ]
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

# -- phase 3: the agent loop -----------------------------------------------
EXIT_LABELS = {
    "final_answer": ("success", "Final answer"),
    "iteration_cap": ("warning", "Stopped: iteration cap"),
    "budget_exceeded": ("warning", "Stopped: budget exceeded"),
    "output_truncated": ("warning", "Stopped: output truncated"),
    "transport_error": ("error", "Stopped: provider kept failing"),
}

with agent_tab:
    st.write(
        "The same tools as phase 2, but now in a loop: the model keeps asking for "
        "tools until it can answer. Every step is traced to `data/traces/`."
    )
    agent_repo = st.text_input("Repository", DEFAULT_REPO, key="agent_repo")
    agent_question = st.text_area(
        "Question",
        "How does FastAPI turn a request validation error into a 422 response?",
        key="agent_question",
    )
    col_a, col_b = st.columns(2)
    max_iterations = col_a.number_input("Max iterations (model calls)", 1, 20, 8)
    max_cost = col_b.number_input("Max cost for this run (USD)", 0.001, 1.0, 0.25, 0.01, format="%.3f")

    if st.button("Run the agent", disabled=not agent_question.strip()):
        try:
            with st.spinner("Cloning (first run only)..."):
                agent_registry, _ = build_default_registry(agent_repo)
            tracer = TraceWriter(new_trace_path())
            agent = Agent(
                client, agent_registry, max_iterations=int(max_iterations),
                max_cost_usd=float(max_cost), tracer=tracer, temperature=temperature,
            )
            step_box = None
            final = None
            for event in agent.stream(agent_question):
                if isinstance(event, StepStarted):
                    step_box = st.status(f"Step {event.step}: model is thinking...", expanded=True)
                elif isinstance(event, TextDelta):
                    step_box.write(event.text)
                elif isinstance(event, ToolCallEvent):
                    step_box.code(f"{event.name}({json.dumps(event.arguments)})", language="python")
                elif isinstance(event, ToolResultEvent):
                    icon = "🔁" if event.note == "repeat" else "⛔" if event.note == "denied" else (
                        "❌" if event.is_error else "✅"
                    )
                    step_box.caption(
                        f"{icon} {event.name} · {event.duration_s * 1000:.0f} ms"
                        + (f" · {event.note}" if event.note else "")
                    )
                    step_box.code(event.content[:800])
                    step_box.update(label=f"Step {event.step}: ran {event.name}")
                elif isinstance(event, Final):
                    final = event
            kind, label = EXIT_LABELS[final.exit_reason]
            getattr(st, kind)(label + (f" -- {final.detail}" if final.detail else ""))
            st.subheader("Answer")
            st.write(final.answer or "(no answer)")
            st.caption(
                f"{final.steps} step(s) · {final.input_tokens} in / {final.output_tokens} out · "
                f"{format_usd(final.cost_usd)}"
            )
            with st.expander(f"Trace ({tracer.path.name}) -- one JSON record per step"):
                for record in read_trace(tracer.path):
                    st.json(record, expanded=False)
        except Exception as exc:  # surface clone/network errors in the page, not a traceback
            st.error(f"{type(exc).__name__}: {exc}")
