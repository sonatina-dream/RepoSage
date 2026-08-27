"""Model IDs, pricing, runtime settings, and cost maths.

Why this file exists at all:

An LLM application has two kinds of constants that a normal service does not.
The first is the model identifier, which is not a stable abstraction the way a
database driver version is -- swapping `haiku` for `sonnet` changes the quality,
latency and price of every answer the system gives. The second is a price list
that lives on someone else's website and changes without your deployment
knowing. Scattering either of those through the codebase means you can never
answer "what did that run cost?" or "which model produced this eval number?".

So both live here, in one file, and everything else imports from it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # Optional: keeps the test suite importable without the dependency.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - convenience only
    pass


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# Two models, two jobs:
#
#   HAIKU  - the default while iterating. Roughly 5x cheaper than Sonnet on
#            output tokens. Anything that is a loop, a smoke test, or a demo
#            you will run twenty times runs here.
#   SONNET - used where the quality of the reasoning is the point: extraction
#            from messy real text, the agent's tool-selection decisions, and
#            the LLM-as-judge in phase 5.
#
# A note on aliases. `claude-sonnet-5` is a *floating alias*: it points at
# whatever the current Sonnet 5 snapshot is, so a model upgrade can silently
# change your outputs. That is fine while iterating and unacceptable for
# evaluation. Before phase 5 records any measured number, pin these to dated
# snapshots (e.g. `claude-haiku-4-5-20251001`), because "we scored 0.82" only
# means something if you can say which weights scored it.
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"

DEFAULT_MODEL = HAIKU


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------
# USD per million tokens. This table is a liability: it is a copy of a number
# that lives on Anthropic's pricing page, and copies go stale. Hence the
# verification date -- if you are reading this long after it, re-check before
# quoting any cost figure in the README.
#
# Source: https://platform.claude.com/docs/en/about-claude/pricing
PRICING_LAST_VERIFIED = "2026-08-27"


@dataclass(frozen=True)
class ModelPrice:
    """Price per *million* tokens, kept in the units the pricing page uses.

    Storing dollars-per-million rather than dollars-per-token keeps the
    literals human-checkable against the published table. The division to
    per-token happens once, in `estimate_cost`.
    """

    input_per_mtok: float
    output_per_mtok: float


PRICING: dict[str, ModelPrice] = {
    "claude-haiku-4-5-20251001": ModelPrice(1.00, 5.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
    "claude-sonnet-5": ModelPrice(2.00, 10.00),
    "claude-sonnet-4-5-20250929": ModelPrice(3.00, 15.00),
    "claude-opus-5": ModelPrice(5.00, 25.00),
}


class UnknownModelError(KeyError):
    """Raised when we are asked to price a model we have no entry for.

    Deliberately loud. The tempting alternative -- fall back to 0.0 for an
    unknown model -- produces a cost report that reads "$0.00" for the most
    expensive run you ever made. A missing price is a bug, not a zero.
    """


def price_for(model: str) -> ModelPrice:
    """Look up a model's price, tolerating dated snapshots of a known family."""
    if model in PRICING:
        return PRICING[model]

    # `claude-sonnet-5-20260401` should price like `claude-sonnet-5`. Match the
    # longest known prefix so `claude-haiku-4-5-...` cannot accidentally match
    # a shorter, cheaper family name.
    candidates = [name for name in PRICING if model.startswith(name)]
    if candidates:
        return PRICING[max(candidates, key=len)]

    raise UnknownModelError(
        f"No price entry for model {model!r}. Add it to PRICING in config.py "
        f"(known: {', '.join(sorted(PRICING))})."
    )


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD for one request's token counts.

    Note the asymmetry: output tokens cost about 5x what input tokens cost.
    That single fact drives most cost decisions in this project. It is why a
    long retrieved context is cheaper than it feels, why an agent that narrates
    its reasoning at length is expensive, and why `max_tokens` is a budget
    lever and not just a safety limit.
    """
    price = price_for(model)
    return (
        input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok
    ) / 1_000_000


def format_usd(amount: float) -> str:
    """Render small dollar amounts without rounding them into invisibility."""
    if amount and abs(amount) < 0.01:
        return f"${amount:.6f}"
    return f"${amount:.4f}"


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, default))


@dataclass
class Settings:
    """Runtime knobs, resolved once from the environment."""

    api_key: str | None = None
    model: str = DEFAULT_MODEL

    # Caps the *output* of a single call. Also the number the budget guard has
    # to assume the model will actually produce, since output length is not
    # knowable in advance.
    max_tokens: int = 1024

    # 0.0 for anything whose output we parse or evaluate; higher only when we
    # deliberately want variety. Phase 0 demo 3 shows why.
    temperature: float = 0.0

    # Hard ceiling on cumulative spend for the lifetime of one LLMClient.
    # Small on purpose: this is the number that stops a misbehaving agent loop
    # in phase 3 from running up a bill overnight.
    spend_ceiling_usd: float = 1.00

    # Transport-level retries we perform ourselves (see llm.py).
    max_retries: int = 3
    request_timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=os.getenv("REPOSAGE_MODEL", DEFAULT_MODEL),
            max_tokens=_env_int("REPOSAGE_MAX_TOKENS", 1024),
            temperature=_env_float("REPOSAGE_TEMPERATURE", 0.0),
            spend_ceiling_usd=_env_float("REPOSAGE_SPEND_CEILING_USD", 1.00),
            max_retries=_env_int("REPOSAGE_MAX_RETRIES", 3),
            request_timeout_s=_env_float("REPOSAGE_TIMEOUT_S", 60.0),
        )

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
                "fill it in, or export the variable in your shell."
            )
        return self.api_key
