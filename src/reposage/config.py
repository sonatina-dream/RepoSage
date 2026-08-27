"""Providers, model IDs, pricing, runtime settings, and cost maths.

Why this file exists at all:

An LLM application has two kinds of constants that a normal service does not.
The first is the model identifier, which is not a stable abstraction the way a
database driver version is -- swapping one model for another changes the
quality, latency and price of every answer the system gives. The second is a
price list that lives on someone else's website and changes without your
deployment knowing. Scattering either of those through the codebase means you
can never answer "what did that run cost?" or "which model produced this eval
number?".

So both live here, in one file, and everything else imports from it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

try:  # Optional: keeps the test suite importable without the dependency.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - convenience only
    pass


# --------------------------------------------------------------------------
# Providers and models
# --------------------------------------------------------------------------
# RepoSage speaks to more than one vendor. Not for its own sake: the phase 5
# eval suite is far more interesting when the same agent can be pointed at two
# different models and the difference measured. It also buys a debugging tool
# that matters from phase 3 onward -- when the agent misbehaves, flipping one
# environment variable tells you whether the fault is in your loop or in the
# provider.
ANTHROPIC = "anthropic"
DEEPSEEK = "deepseek"

# Anthropic. Aliases like `claude-sonnet-5` float: they point at whatever the
# current snapshot is, so a model upgrade can silently change your outputs.
# Fine while iterating, unacceptable for evaluation -- pin dated snapshots
# before phase 5 records any measured number.
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"

# DeepSeek. Note these are served through an OpenAI-compatible endpoint, which
# is a wire-format detail the provider layer absorbs; see providers/.
DEEPSEEK_FLASH = "deepseek-v4-flash"
DEEPSEEK_PRO = "deepseek-v4-pro"

# Every provider gets two roles rather than one default. `fast` is what runs
# while iterating -- loops, smoke tests, anything you will run twenty times.
# `quality` is for the places where the reasoning itself is the deliverable:
# extraction from messy real text, the agent's tool-selection decisions, and
# the LLM-as-judge in phase 5.
PROVIDER_MODELS = {
    ANTHROPIC: {"fast": HAIKU, "quality": SONNET},
    DEEPSEEK: {"fast": DEEPSEEK_FLASH, "quality": DEEPSEEK_PRO},
}

DEFAULT_PROVIDER = os.getenv("REPOSAGE_PROVIDER", DEEPSEEK)


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------
# USD per million tokens. This table is a liability: it is a copy of numbers
# that live on two vendors' pricing pages, and copies go stale. Hence the
# verification date -- if you are reading this long after it, re-check before
# quoting any cost figure in the README.
#
# Sources:
#   https://platform.claude.com/docs/en/about-claude/pricing
#   https://api-docs.deepseek.com/quick_start/pricing
PRICING_LAST_VERIFIED = "2026-08-27"

# DeepSeek charges a higher rate during defined peak windows and half that
# outside them. Weekends are entirely off-peak. Hours are UTC; from Rome
# (UTC+2 in summer) this puts 08:00-12:00 local squarely in the expensive
# band, which is most of a working morning.
DEEPSEEK_PEAK_WINDOWS_UTC = ((1, 4), (6, 10))
DEEPSEEK_PEAK_WEEKDAYS = (0, 1, 2, 3, 4)  # Monday-Friday


@dataclass(frozen=True)
class ModelPrice:
    """Price per *million* tokens, in the units the pricing pages use.

    Storing dollars-per-million rather than dollars-per-token keeps the
    literals human-checkable against the published tables. The division to
    per-token happens once, in `estimate_cost`.

    `cached_input_per_mtok` is set only for providers that bill prompt-cache
    hits at a separate rate and report the split in their usage object.
    DeepSeek does both; it matters enormously from phase 3, where an agent
    loop resends a growing history on every turn and most of that history is
    a cache hit by the second iteration.
    """

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None


@dataclass(frozen=True)
class ModelSpec:
    """A model, its provider, and how it is billed.

    `off_peak` is None for flat-rate models, which is most of them. Where it
    is set, `peak_windows_utc` says when the standard rate applies; everything
    outside those windows bills at the discounted rate.
    """

    id: str
    provider: str
    standard: ModelPrice
    off_peak: ModelPrice | None = None
    peak_windows_utc: tuple[tuple[int, int], ...] = ()
    peak_weekdays: tuple[int, ...] = ()

    def is_peak(self, when: datetime) -> bool:
        if self.off_peak is None:
            return True
        moment = when.astimezone(timezone.utc)
        if moment.weekday() not in self.peak_weekdays:
            return False
        return any(start <= moment.hour < end for start, end in self.peak_windows_utc)

    def price_at(self, when: datetime | None = None) -> ModelPrice:
        """Resolve the applicable price.

        `when` is a parameter rather than an implicit call to `now()` so that
        cost maths stays testable. A function that reads the wall clock cannot
        be asserted against, and this one decides what you are billed.
        """
        moment = when or datetime.now(timezone.utc)
        return self.standard if self.is_peak(moment) else (self.off_peak or self.standard)


def _anthropic(model_id: str, input_price: float, output_price: float) -> ModelSpec:
    return ModelSpec(model_id, ANTHROPIC, ModelPrice(input_price, output_price))


def _deepseek(model_id: str, peak: ModelPrice, off_peak: ModelPrice) -> ModelSpec:
    return ModelSpec(
        model_id,
        DEEPSEEK,
        standard=peak,
        off_peak=off_peak,
        peak_windows_utc=DEEPSEEK_PEAK_WINDOWS_UTC,
        peak_weekdays=DEEPSEEK_PEAK_WEEKDAYS,
    )


MODELS: dict[str, ModelSpec] = {
    # Anthropic -- flat rate.
    HAIKU: _anthropic(HAIKU, 1.00, 5.00),
    "claude-haiku-4-5": _anthropic("claude-haiku-4-5", 1.00, 5.00),
    SONNET: _anthropic(SONNET, 2.00, 10.00),
    "claude-sonnet-4-5-20250929": _anthropic("claude-sonnet-4-5-20250929", 3.00, 15.00),
    "claude-opus-5": _anthropic("claude-opus-5", 5.00, 25.00),
    # DeepSeek -- peak / off-peak, with a separate cache-hit input rate.
    DEEPSEEK_FLASH: _deepseek(
        DEEPSEEK_FLASH,
        peak=ModelPrice(0.44, 1.32, cached_input_per_mtok=0.014),
        off_peak=ModelPrice(0.22, 0.66, cached_input_per_mtok=0.007),
    ),
    DEEPSEEK_PRO: _deepseek(
        DEEPSEEK_PRO,
        peak=ModelPrice(1.32, 3.96, cached_input_per_mtok=0.044),
        off_peak=ModelPrice(0.66, 1.98, cached_input_per_mtok=0.022),
    ),
}


class UnknownModelError(KeyError):
    """Raised when we are asked to price a model we have no entry for.

    Deliberately loud. The tempting alternative -- fall back to 0.0 for an
    unknown model -- produces a cost report that reads "$0.00" for the most
    expensive run you ever made. A missing price is a bug, not a zero.
    """


def spec_for(model: str) -> ModelSpec:
    """Look up a model, tolerating dated snapshots of a known family."""
    if model in MODELS:
        return MODELS[model]

    # `claude-sonnet-5-20260401` should price like `claude-sonnet-5`. Match the
    # longest known prefix so a shorter, cheaper family name cannot win.
    candidates = [name for name in MODELS if model.startswith(name)]
    if candidates:
        return MODELS[max(candidates, key=len)]

    raise UnknownModelError(
        f"No entry for model {model!r}. Add it to MODELS in config.py "
        f"(known: {', '.join(sorted(MODELS))})."
    )


def provider_for(model: str) -> str:
    return spec_for(model).provider


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_input_tokens: int = 0,
    at: datetime | None = None,
) -> float:
    """Cost in USD for one request's token counts.

    `input_tokens` is the *total* prompt size; `cached_input_tokens` is the
    subset of it that hit the provider's prompt cache and is billed at the
    cheaper rate. Providers that do not report a cache split simply pass 0.

    Note the asymmetry between input and output: output costs roughly three to
    five times what input costs, depending on the model. That single fact
    drives most cost decisions in this project. It is why stuffing retrieved
    code into a prompt (phase 4) is cheaper than it feels, why an agent that
    narrates its reasoning at length (phase 3) is expensive, and why
    `max_tokens` is a budget lever and not just a safety limit.
    """
    price = spec_for(model).price_at(at)

    cached = max(0, min(cached_input_tokens, input_tokens))
    uncached = input_tokens - cached
    cached_rate = (
        price.cached_input_per_mtok
        if price.cached_input_per_mtok is not None
        else price.input_per_mtok
    )

    return (
        uncached * price.input_per_mtok
        + cached * cached_rate
        + output_tokens * price.output_per_mtok
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


# Which environment variable holds the key, per provider.
API_KEY_ENV = {ANTHROPIC: "ANTHROPIC_API_KEY", DEEPSEEK: "DEEPSEEK_API_KEY"}


@dataclass
class Settings:
    """Runtime knobs, resolved once from the environment."""

    provider: str = DEFAULT_PROVIDER
    api_key: str | None = None
    model: str | None = None  # None -> the provider's `fast` model

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

    def __post_init__(self) -> None:
        if self.provider not in PROVIDER_MODELS:
            raise ValueError(
                f"Unknown provider {self.provider!r}. "
                f"Choose one of: {', '.join(PROVIDER_MODELS)}."
            )
        if self.model is None:
            self.model = PROVIDER_MODELS[self.provider]["fast"]

    @property
    def fast_model(self) -> str:
        return PROVIDER_MODELS[self.provider]["fast"]

    @property
    def quality_model(self) -> str:
        return PROVIDER_MODELS[self.provider]["quality"]

    @classmethod
    def from_env(cls, provider: str | None = None, model: str | None = None) -> "Settings":
        """Resolve settings, letting a caller override the provider.

        The override matters because the API key is provider-specific: asking
        for DeepSeek must read DEEPSEEK_API_KEY, not whatever the default
        provider's variable happens to hold. Selecting the provider and
        selecting its key are one decision, so they happen in one place.
        """
        provider = provider or os.getenv("REPOSAGE_PROVIDER", DEFAULT_PROVIDER)
        if provider not in API_KEY_ENV:
            raise ValueError(
                f"Unknown provider {provider!r}. Choose one of: {', '.join(API_KEY_ENV)}."
            )
        return cls(
            provider=provider,
            api_key=os.getenv(API_KEY_ENV[provider]),
            model=model or os.getenv("REPOSAGE_MODEL") or None,
            max_tokens=_env_int("REPOSAGE_MAX_TOKENS", 1024),
            temperature=_env_float("REPOSAGE_TEMPERATURE", 0.0),
            spend_ceiling_usd=_env_float("REPOSAGE_SPEND_CEILING_USD", 1.00),
            max_retries=_env_int("REPOSAGE_MAX_RETRIES", 3),
            request_timeout_s=_env_float("REPOSAGE_TIMEOUT_S", 60.0),
        )

    def require_api_key(self) -> str:
        if not self.api_key:
            variable = API_KEY_ENV[self.provider]
            raise RuntimeError(
                f"{variable} is not set, but provider is {self.provider!r}. "
                f"Copy .env.example to .env and fill it in, or export the "
                f"variable in your shell."
            )
        return self.api_key
