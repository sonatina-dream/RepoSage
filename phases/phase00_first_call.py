"""Phase 0 -- the five things about the API that everything else rests on.

Run one demo at a time:

    python phases/phase00_first_call.py --demo 3
    python phases/phase00_first_call.py --all

Total cost of --all on either provider's fast model is a fraction of a cent. Every demo prints what it
spent, because the habit of looking at that number is half of cost discipline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# The package is not installed, so make `src/` importable. `pip install -e .`
# would also work; this keeps the scripts runnable straight from a clone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reposage.config import Settings, format_usd  # noqa: E402
from reposage.llm import LLMClient, describe_target  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# --------------------------------------------------------------------------
def demo_1_first_call(client: LLMClient) -> None:
    """Messages: the request is a list of turns, the reply is a list of blocks.

    Two things to notice in the output. The system prompt is a separate
    parameter, not a message -- it is instruction, not conversation. And the
    reply's `content` is a *list of blocks*, not a string: today they are all
    text blocks, but in phase 2 the same list starts carrying `tool_use` blocks
    alongside the text, and code that assumed `content[0].text` breaks.
    """
    banner("Demo 1 - the shape of a call")

    response = client.complete(
        prompt="In two sentences, what is retrieval-augmented generation?",
        system="You explain engineering concepts to senior developers. No preamble.",
        max_tokens=200,
    )

    print(response.text)
    print(f"\nstop_reason: {response.stop_reason}")
    print(f"content blocks: {[block.type for block in response.raw.content]}")


# --------------------------------------------------------------------------
def demo_2_tokens_and_cost(client: LLMClient) -> None:
    """Tokens and cost: where the money actually goes.

    The asymmetry is the lesson. Output tokens cost roughly five times what
    input tokens cost, so a short question with a long answer can cost more
    than a long question with a short answer. Practical consequence for later:
    stuffing retrieved code into the prompt (phase 4) is cheaper than it feels,
    while an agent that thinks out loud at length (phase 3) is expensive.
    """
    banner("Demo 2 - tokens and cost")

    long_in_short_out = client.complete(
        prompt=(
            "Here is a paragraph of context:\n\n"
            + ("The quick brown fox jumps over the lazy dog. " * 60)
            + "\n\nReply with exactly one word: OK"
        ),
        max_tokens=10,
    )
    short_in_long_out = client.complete(
        prompt="Write roughly 150 words about why type hints help large Python codebases.",
        max_tokens=400,
    )

    for label, response in [
        ("long input, short output", long_in_short_out),
        ("short input, long output", short_in_long_out),
    ]:
        print(
            f"{label:<26} in={response.input_tokens:>4} out={response.output_tokens:>4} "
            f"cost={format_usd(response.cost_usd)}"
        )

    print(
        "\nNearly the same number of tokens moved in each case; the second one "
        "costs several times more, because it moved them in the expensive "
        "direction."
    )


# --------------------------------------------------------------------------
def demo_3_temperature(client: LLMClient) -> None:
    """Temperature: how much the sampler is allowed to wander.

    At 0 the model takes the highest-probability token at each step, so the
    same prompt gives you (very nearly) the same answer every time. Note
    "nearly": temperature 0 is not a determinism guarantee -- floating-point
    non-associativity in batched inference means identical inputs can still
    diverge occasionally. Treat it as "as repeatable as you can get", not as a
    seed.

    The rule this project follows: temperature 0 for anything that will be
    parsed, validated, evaluated or cited. Higher only where variety is the
    product.
    """
    banner("Demo 3 - temperature")

    prompt = "Give a six-word tagline for a tool that answers questions about a codebase."

    for temperature in (0.0, 1.0):
        print(f"\ntemperature={temperature}")
        for _ in range(3):
            response = client.complete(prompt=prompt, temperature=temperature, max_tokens=40)
            print(f"  {response.text.strip()}")


# --------------------------------------------------------------------------
def demo_4_streaming(client: LLMClient) -> None:
    """Streaming: same tokens, same price, different wall-clock experience.

    Streaming does not make generation faster. It makes the *first* token
    visible immediately instead of after the whole reply is generated, which is
    the difference between a UI that feels broken and one that does not. The
    catch, and it matters for phase 6: usage totals only arrive at the end of
    the stream, so per-request cost accounting has to happen after the last
    chunk, not alongside it.
    """
    banner("Demo 4 - streaming")

    started = time.monotonic()
    first_chunk_at: float | None = None

    for chunk in client.stream(
        prompt="Explain in about 80 words why an agent needs a maximum iteration count.",
        max_tokens=250,
    ):
        if first_chunk_at is None:
            first_chunk_at = time.monotonic() - started
        print(chunk, end="", flush=True)

    total = time.monotonic() - started
    print(
        f"\n\nfirst token after {first_chunk_at:.2f}s, "
        f"full reply after {total:.2f}s"
    )


# --------------------------------------------------------------------------
def demo_5_statelessness(client: LLMClient) -> None:
    """Statelessness: there is no conversation, only a list you resend.

    The API remembers nothing between calls. What feels like a conversation is
    the client resending the whole history every time. Two consequences that
    shape the rest of this project:

      - Cost grows quadratically with turns. Turn ten pays to re-read turns one
        through nine. This is why phase 3 needs a context budget rather than an
        ever-growing list.
      - Nothing is hidden. The entire state the model sees is a list you own
        and can print, which is what makes an agent loop debuggable at all.
    """
    banner("Demo 5 - statelessness")

    history = [{"role": "user", "content": "My favourite number is 17. Just acknowledge it."}]
    first = client.complete(messages=history)
    print(f"turn 1 -> {first.text.strip()}")

    without = client.complete(prompt="What is my favourite number?")
    print(f"\nasked fresh          -> {without.text.strip()}")

    history.append({"role": "assistant", "content": first.text})
    history.append({"role": "user", "content": "What is my favourite number?"})
    with_history = client.complete(messages=history)
    print(f"asked with history   -> {with_history.text.strip()}")

    print(
        f"\ninput tokens: {without.input_tokens} without history vs "
        f"{with_history.input_tokens} with it. That gap is what you pay on "
        f"every single turn, forever."
    )


DEMOS = {
    1: demo_1_first_call,
    2: demo_2_tokens_and_cost,
    3: demo_3_temperature,
    4: demo_4_streaming,
    5: demo_5_statelessness,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=int, choices=sorted(DEMOS), help="Run a single demo.")
    parser.add_argument("--all", action="store_true", help="Run all five.")
    parser.add_argument("--provider", help="anthropic or deepseek; defaults to REPOSAGE_PROVIDER.")
    parser.add_argument("--model", help="Defaults to the provider's fast model.")
    args = parser.parse_args()

    if not args.demo and not args.all:
        parser.error("Pass --demo N or --all.")

    settings = Settings.from_env(provider=args.provider, model=args.model)
    client = LLMClient(settings)
    print(describe_target(settings))

    for number in sorted(DEMOS) if args.all else [args.demo]:
        DEMOS[number](client)

    banner(f"Session usage: {client.usage.summary()}")


if __name__ == "__main__":
    main()
