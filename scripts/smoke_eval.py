"""Smoke eval: 10 hand-written questions, each naming the file(s) a correct answer must cite.

    python scripts/smoke_eval.py                        # default provider from .env
    python scripts/smoke_eval.py --provider anthropic
    python scripts/smoke_eval.py --only q01 q02

A question passes when the run ended with a final answer AND every required file
appears in the answer as a `path:line` citation. This is a smoke test, not the phase 5
eval: it catches a broken loop or a model that stopped citing, nothing subtler.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reposage.agent import Agent  # noqa: E402
from reposage.config import Settings, format_usd  # noqa: E402
from reposage.llm import LLMClient, describe_target  # noqa: E402
from reposage.repo import DEFAULT_REPO  # noqa: E402
from reposage.tools import build_default_registry  # noqa: E402
from reposage.tracing import TraceWriter, new_trace_path  # noqa: E402

QUESTIONS = ROOT / "data" / "smoke" / "questions.json"


def cited_files(answer: str, required: list[str]) -> list[str]:
    """Return the required files that appear in the answer followed by `:<line>`."""
    return [
        path
        for path in required
        if re.search(re.escape(path) + r":\d+", answer)
    ]


def main() -> int:
    """Run every question, print pass/fail per question and the totals; exit 1 if any fail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--only", nargs="*", help="Run just these question ids.")
    parser.add_argument("--max-iterations", type=int, default=8)
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text())
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]

    settings = Settings.from_env(provider=args.provider, model=args.model)
    print(describe_target(settings))
    registry, _ = build_default_registry(args.repo)

    passed = 0
    total_cost = 0.0
    for item in questions:
        # A fresh client per question keeps each cost reading independent.
        client = LLMClient(Settings.from_env(provider=args.provider, model=args.model))
        agent = Agent(
            client,
            registry,
            max_iterations=args.max_iterations,
            tracer=TraceWriter(new_trace_path()),
        )
        result = agent.run(item["question"])
        found = cited_files(result.answer, item["must_cite"])
        ok = result.ok and len(found) == len(item["must_cite"])
        passed += ok
        total_cost += result.cost_usd
        missing = [p for p in item["must_cite"] if p not in found]
        print(
            f"{'PASS' if ok else 'FAIL'} {item['id']} | {result.exit_reason} | "
            f"{result.steps} steps | {format_usd(result.cost_usd)}"
            + (f" | missing citation: {', '.join(missing)}" if missing else "")
        )

    count = len(questions)
    print(f"\nscore: {passed}/{count}  total cost: {format_usd(total_cost)}  "
          f"per question: {format_usd(total_cost / max(count, 1))}")
    return 0 if passed == count else 1


if __name__ == "__main__":
    sys.exit(main())
