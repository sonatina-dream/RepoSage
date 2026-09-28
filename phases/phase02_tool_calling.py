"""Phase 2 -- one tool-calling round trip, executed by hand.

    python phases/phase02_tool_calling.py
    python phases/phase02_tool_calling.py --question "Where is APIRouter defined?"
    python phases/phase02_tool_calling.py --show-schemas

There is no loop here. That is the point. The model gets one chance to ask for
tools, we run them, we send the results back, and it answers. Every message is
printed as it is built, so the round trip is visible rather than inferred:

    1. user question + tool schemas   ->  model
    2. model replies with tool_use blocks, stop_reason "tool_use"
    3. WE execute the tools. The model runs nothing.
    4. results go back as tool results, matched by id
    5. model answers from what it read

Phase 3 turns step 5's "or asks for more tools" into a loop. Everything that
makes that loop safe -- bounded results, errors as data, id matching -- is
already here, because it is much easier to see when only one turn is happening.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reposage.config import Settings, format_usd  # noqa: E402
from reposage.llm import LLMClient, describe_target  # noqa: E402
from reposage.repo import DEFAULT_REPO  # noqa: E402
from reposage.tools import build_default_registry, results_to_messages  # noqa: E402

RULE = "=" * 76

SYSTEM_PROMPT = (
    "You answer questions about a source code repository. You have tools that "
    "read the repository; use them rather than answering from memory, because "
    "your memory of this repository may be out of date or wrong.\n\n"
    "Every factual claim about the code must cite the file and line you read it "
    "from, in the form `path/to/file.py:123`. If the tools do not give you "
    "enough to answer, say exactly what is missing instead of guessing."
)

DEFAULT_QUESTION = (
    "What does the function that resolves a request's dependencies do, and "
    "where is it defined? Cite the file and line."
)


def banner(title: str) -> None:
    """Print a title between two ruler lines."""
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> None:
    """Ask one question, run the tools the model requests, and send the results back once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--provider", help="deepseek or anthropic; defaults to .env")
    parser.add_argument("--model", help="Defaults to the provider's fast model.")
    parser.add_argument(
        "--show-schemas",
        action="store_true",
        help="Print the tool schemas exactly as the model receives them.",
    )
    args = parser.parse_args()

    settings = Settings.from_env(provider=args.provider, model=args.model)
    client = LLMClient(settings)
    print(describe_target(settings))

    registry, clone = build_default_registry(args.repo)
    print(f"repo={args.repo} clone={clone}")
    print(f"tools={', '.join(registry.names)}")

    specifications = registry.specifications()
    if args.show_schemas:
        banner("What the model actually sees")
        print(json.dumps(specifications, indent=2))
        # These schemas are sent on every call, so they cost input tokens every time.
        print(f"\n(~{len(json.dumps(specifications)) // 4} tokens, on every request)")

    # -- turn 1: ask, offering tools ---------------------------------------
    banner("Turn 1 — the question, with tools offered")
    print(f"Q: {args.question}\n")

    history = [{"role": "user", "content": args.question}]
    first = client.complete(
        messages=history,
        system=SYSTEM_PROMPT,
        tools=specifications,
        max_tokens=1024,
    )

    print(f"stop_reason : {first.stop_reason}")
    print(f"text        : {first.text.strip()[:400] or '(none — it went straight for a tool)'}")
    print(f"tool_calls  : {len(first.tool_calls)}")

    if not first.wants_tools:
        # Not a failure: a model that can answer without a tool should.
        banner("It answered without tools")
        print(first.text.strip())
        print(f"\nUsage: {client.usage.summary()}")
        return

    for tool_call in first.tool_calls:
        print(f"  -> {tool_call.name}({json.dumps(tool_call.arguments)})  id={tool_call.id}")

    # -- step 3: we execute; the model never does ---------------------------
    banner("Executing the tools ourselves")
    results = registry.dispatch_all(first.tool_calls)

    for invocation in registry.invocations:
        status = "ERROR" if invocation.result.is_error else "ok"
        preview = invocation.result.content.replace("\n", " ")[:160]
        print(f"[{status:>5}] {invocation.name} ({invocation.duration_s * 1000:.0f} ms)")
        print(f"         {preview}")

    # -- turn 2: hand the results back --------------------------------------
    banner("Turn 2 — the same conversation, with results appended")

    # Replay the assistant turn with its tool calls; results without them are rejected.
    history.append(
        {"role": "assistant", "content": first.text, "tool_calls": first.tool_calls}
    )
    history.extend(results_to_messages(results))

    for message in history:
        role = message["role"]
        if role == "tool":
            flag = " (is_error)" if message["is_error"] else ""
            body = message["content"].replace("\n", " ")[:80]
            print(f"  {role:<9}{flag} id={message['tool_call_id']} :: {body}")
        else:
            calls = message.get("tool_calls", [])
            suffix = f" + {len(calls)} tool_call(s)" if calls else ""
            print(f"  {role:<9} :: {(message['content'] or '(no text)')[:80]}{suffix}")

    second = client.complete(
        messages=history,
        system=SYSTEM_PROMPT,
        tools=specifications,
        max_tokens=1024,
    )

    banner("The answer")
    if second.wants_tools:
        # Normal: one lookup often leads to the next. Looping is phase 3's job.
        print(
            f"The model wants {len(second.tool_calls)} more tool call(s): "
            f"{', '.join(c.name for c in second.tool_calls)}.\n"
            "Phase 2 does a single round trip, so we stop here. Turning this "
            "into 'keep going until it stops asking' is phase 3."
        )
    print(second.text.strip())

    banner("Cost")
    print(f"turn 1: {format_usd(first.cost_usd)}   turn 2: {format_usd(second.cost_usd)}")
    print(f"total : {client.usage.summary()}")
    # The growth between turns is the tool results plus the replayed history.
    print(
        f"\nTurn 2 sent {second.input_tokens - first.input_tokens} more input "
        f"tokens than turn 1. That growth, every turn, is what phase 3 has to "
        f"manage."
    )


if __name__ == "__main__":
    main()
