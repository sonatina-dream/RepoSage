"""Command line: `python -m reposage ask "How does dependency overriding work?"`."""

from __future__ import annotations

import argparse
import json
import sys

from .agent import Agent, ApprovalCallback
from .config import Settings, format_usd
from .events import Final, StepStarted, TextDelta, ToolCallEvent, ToolResultEvent
from .llm import LLMClient, describe_target
from .repo import DEFAULT_REPO
from .tools import build_default_registry
from .tracing import TraceWriter, new_trace_path


def ask_on_terminal(call) -> bool:
    """Approval callback for the CLI: show the call and ask y/N."""
    answer = input(f"  ? allow {call.name}({json.dumps(call.arguments)})? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the agent, print each event as it happens; return the exit code."""
    parser = argparse.ArgumentParser(prog="reposage", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="Ask the agent a question about a repository.")
    ask.add_argument("question")
    ask.add_argument("--repo", default=DEFAULT_REPO)
    ask.add_argument("--provider", help="deepseek or anthropic; defaults to .env")
    ask.add_argument("--model")
    ask.add_argument("--max-iterations", type=int, default=8)
    ask.add_argument("--max-cost", type=float, default=0.25, help="USD limit for this run.")
    ask.add_argument("--no-trace", action="store_true", help="Do not write a JSONL trace.")
    args = parser.parse_args(argv)

    settings = Settings.from_env(provider=args.provider, model=args.model)
    client = LLMClient(settings)
    print(describe_target(settings))
    registry, _ = build_default_registry(args.repo)

    tracer = None if args.no_trace else TraceWriter(new_trace_path())
    approve: ApprovalCallback | None = ask_on_terminal if sys.stdin.isatty() else None
    agent = Agent(
        client,
        registry,
        max_iterations=args.max_iterations,
        max_cost_usd=args.max_cost,
        approve=approve,
        tracer=tracer,
    )

    final: Final | None = None
    for event in agent.stream(args.question):
        if isinstance(event, StepStarted):
            print(f"\n[step {event.step}]")
        elif isinstance(event, TextDelta):
            print(f"  model: {event.text.strip()[:300]}")
        elif isinstance(event, ToolCallEvent):
            flag = " (needs approval)" if event.needs_approval else ""
            print(f"  -> {event.name}({json.dumps(event.arguments)}){flag}")
        elif isinstance(event, ToolResultEvent):
            status = event.note or ("ERROR" if event.is_error else "ok")
            print(f"  <- [{status}] {event.content.replace(chr(10), ' ')[:140]}")
        elif isinstance(event, Final):
            final = event

    assert final is not None
    print(f"\n{'=' * 76}\n{final.answer or '(no answer)'}\n{'=' * 76}")
    print(
        f"exit: {final.exit_reason}"
        + (f" ({final.detail})" if final.detail else "")
        + f" | {final.steps} step(s) | {final.input_tokens} in / {final.output_tokens} out"
        + f" | {format_usd(final.cost_usd)}"
    )
    if tracer:
        print(f"trace: {tracer.path}")
    return 0 if final.exit_reason == "final_answer" else 1


if __name__ == "__main__":
    sys.exit(main())
