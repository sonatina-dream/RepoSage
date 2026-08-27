"""Verify a provider is reachable, correctly wired, and priced — for ~$0.00001.

    python scripts/check_provider.py
    python scripts/check_provider.py --provider anthropic

Run this whenever something changes underneath you: a new key, a new machine, a
new provider, a vendor deprecating a model. It makes one tiny call and one tiny
stream, then prints what they cost.

It exists because the failures it catches are all *silent* ones. A dropped
system prompt, a usage object that never arrives, a model ID the vendor
retired — none of those raise. They just make your output slightly worse or
your cost report quietly wrong, and you find out weeks later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reposage.config import API_KEY_ENV, Settings, format_usd, spec_for  # noqa: E402
from reposage.llm import LLMClient, describe_target  # noqa: E402

OK = "  ok  "
BAD = " FAIL "


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", help="deepseek or anthropic; defaults to .env")
    args = parser.parse_args()

    try:
        settings = Settings.from_env(provider=args.provider)
    except ValueError as exc:
        print(f"[{BAD}] {exc}")
        return 1

    settings.max_tokens = 40
    print(f"[{OK}] settings: {describe_target(settings)}")

    if not settings.api_key:
        variable = API_KEY_ENV[settings.provider]
        print(f"[{BAD}] {variable} is empty. Add it to .env.")
        return 1
    print(f"[{OK}] {API_KEY_ENV[settings.provider]} present ({len(settings.api_key)} chars)")

    spec = spec_for(settings.model)
    price = spec.price_at()
    band = "peak" if spec.is_peak(__import__("datetime").datetime.now(
        __import__("datetime").timezone.utc)) else "off-peak"
    print(
        f"[{OK}] pricing: {settings.model} at {band} rates — "
        f"${price.input_per_mtok}/MTok in, ${price.output_per_mtok}/MTok out"
    )

    client = LLMClient(settings)

    # One non-streaming call. The system prompt is deliberately something the
    # reply must obey, so a silently-dropped system prompt shows up as a wrong
    # answer rather than as nothing at all.
    try:
        response = client.complete(
            prompt="Say the word: online",
            system="Reply with exactly one lowercase word and no punctuation.",
        )
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic tool
        print(f"[{BAD}] complete() failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"[{OK}] complete(): {response.text.strip()!r}")
    print(
        f"        stop_reason={response.stop_reason} "
        f"in={response.input_tokens} (cached={response.cached_input_tokens}) "
        f"out={response.output_tokens} cost={format_usd(response.cost_usd)}"
    )
    if response.text.strip().lower() != "online":
        print(
            "        note: the reply ignored the system prompt. On an "
            "OpenAI-style endpoint that usually means it was never sent."
        )

    # And one stream. Usage arrives only at the end, so a non-zero token count
    # here proves the end-of-stream accounting actually fired.
    before = client.usage.calls
    try:
        chunks = sum(1 for _ in client.stream(prompt="Count to three.", max_tokens=30))
    except Exception as exc:  # noqa: BLE001
        print(f"[{BAD}] stream() failed: {type(exc).__name__}: {exc}")
        return 1

    if client.usage.calls == before:
        print(f"[{BAD}] stream() yielded {chunks} chunks but recorded no usage.")
        return 1
    print(f"[{OK}] stream(): {chunks} chunks, usage recorded")

    print(f"\nTotal for this check: {client.usage.summary()}")
    print(f"Budget remaining: {format_usd(client.remaining_budget_usd)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
