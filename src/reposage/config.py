"""Providers, model IDs, prices, runtime settings and cost maths, all in one place."""

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
ANTHROPIC = "anthropic"
DEEPSEEK = "deepseek"

# Aliases like `claude-sonnet-5` float to the latest snapshot; pin dated IDs before evals.
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"

# Served through an OpenAI-compatible endpoint; see providers/.
DEEPSEEK_FLASH = "deepseek-v4-flash"
DEEPSEEK_PRO = "deepseek-v4-pro"

# `fast` for iterating; `quality` where the reasoning itself is the deliverable.
PROVIDER_MODELS = {
    ANTHROPIC: {"fast": HAIKU, "quality": SONNET},
    DEEPSEEK: {"fast": DEEPSEEK_FLASH, "quality": DEEPSEEK_PRO},
}

DEFAULT_PROVIDER = os.getenv("REPOSAGE_PROVIDER", DEEPSEEK)


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------
# USD per million tokens, copied from the vendors' pricing pages. Re-check if stale:
#   https://platform.claude.com/docs/en/about-claude/pricing
#   https://api-docs.deepseek.com/quick_start/pricing
PRICING_LAST_VERIFIED = "2026-08-27"

# DeepSeek bills the full rate inside these UTC windows on weekdays, half outside them.
DEEPSEEK_PEAK_WINDOWS_UTC = ((1, 4), (6, 10))
DEEPSEEK_PEAK_WEEKDAYS = (0, 1, 2, 3, 4)  # Monday-Friday


@dataclass(frozen=True)
class ModelPrice:
    """Prices in USD per million tokens; the cached rate is set only where a vendor has one."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None


@dataclass(frozen=True)
class ModelSpec:
    """A model, its provider, and how it is billed (off_peak is None for flat-rate models)."""

    id: str
    provider: str
    standard: ModelPrice
    off_peak: ModelPrice | None = None
    peak_windows_utc: tuple[tuple[int, int], ...] = ()
    peak_weekdays: tuple[int, ...] = ()

    def is_peak(self, when: datetime) -> bool:
        """Return True if the full (peak) rate applies at this moment."""
        if self.off_peak is None:
            return True
        moment = when.astimezone(timezone.utc)
        if moment.weekday() not in self.peak_weekdays:
            return False
        return any(start <= moment.hour < end for start, end in self.peak_windows_utc)

    def price_at(self, when: datetime | None = None) -> ModelPrice:
        """Return the price that applies at `when` (default: now)."""
        moment = when or datetime.now(timezone.utc)
        return self.standard if self.is_peak(moment) else (self.off_peak or self.standard)


def _anthropic(model_id: str, input_price: float, output_price: float) -> ModelSpec:
    """Build a flat-rate Anthropic model entry."""
    return ModelSpec(model_id, ANTHROPIC, ModelPrice(input_price, output_price))


def _deepseek(model_id: str, peak: ModelPrice, off_peak: ModelPrice) -> ModelSpec:
    """Build a DeepSeek model entry with peak and off-peak prices."""
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
    """Raised for a model with no price entry, instead of silently pricing it at $0."""


def spec_for(model: str) -> ModelSpec:
    """Look up a model's price entry; dated snapshots match their family name."""
    if model in MODELS:
        return MODELS[model]

    # Longest matching prefix wins, so a shorter, cheaper family cannot.
    candidates = [name for name in MODELS if model.startswith(name)]
    if candidates:
        return MODELS[max(candidates, key=len)]

    raise UnknownModelError(
        f"No entry for model {model!r}. Add it to MODELS in config.py "
        f"(known: {', '.join(sorted(MODELS))})."
    )


def provider_for(model: str) -> str:
    """Return the provider name that serves a model."""
    return spec_for(model).provider


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_input_tokens: int = 0,
    at: datetime | None = None,
) -> float:
    """Return the USD cost of one request; cached input tokens bill at the cheaper rate."""
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
    """Format dollars, with 6 decimals for amounts under a cent so they don't show as $0."""
    if amount and abs(amount) < 0.01:
        return f"${amount:.6f}"
    return f"${amount:.4f}"


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    """Read a number from an environment variable, or return the default if unset."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    """Read a whole number from an environment variable, or return the default."""
    return int(_env_float(name, default))


# Which environment variable holds the key, per provider.
API_KEY_ENV = {ANTHROPIC: "ANTHROPIC_API_KEY", DEEPSEEK: "DEEPSEEK_API_KEY"}


@dataclass
class Settings:
    """Runtime settings: provider, key, model, limits and retries."""

    provider: str = DEFAULT_PROVIDER
    api_key: str | None = None
    model: str | None = None  # None -> the provider's `fast` model

    # Max output per call; the budget guard assumes the model uses all of it.
    max_tokens: int = 1024

    # 0.0 for anything we parse; higher only when we want variety.
    temperature: float = 0.0

    # Hard cap on total spend for one LLMClient.
    spend_ceiling_usd: float = 1.00

    # Transport-level retries we perform ourselves (see llm.py).
    max_retries: int = 3
    request_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        """Reject unknown providers and default the model to the fast one."""
        if self.provider not in PROVIDER_MODELS:
            raise ValueError(
                f"Unknown provider {self.provider!r}. "
                f"Choose one of: {', '.join(PROVIDER_MODELS)}."
            )
        if self.model is None:
            self.model = self.fast_model

    @property
    def fast_model(self) -> str:
        """The provider's cheap model, for iterating."""
        return PROVIDER_MODELS[self.provider]["fast"]

    @property
    def quality_model(self) -> str:
        """The provider's stronger model, for reasoning-heavy work."""
        return PROVIDER_MODELS[self.provider]["quality"]

    @classmethod
    def from_env(cls, provider: str | None = None, model: str | None = None) -> "Settings":
        """Build settings from environment variables; the API key follows the chosen provider."""
        provider = provider or os.getenv("REPOSAGE_PROVIDER", DEFAULT_PROVIDER)
        if provider not in API_KEY_ENV:
            raise ValueError(
                f"Unknown provider {provider!r}. Choose one of: {', '.join(API_KEY_ENV)}."
            )
        return cls(
            provider=provider,
            api_key=os.getenv(API_KEY_ENV[provider]),
            model=model or os.getenv("REPOSAGE_MODEL") or None,
            max_tokens=_env_int("REPOSAGE_MAX_TOKENS", cls.max_tokens),
            temperature=_env_float("REPOSAGE_TEMPERATURE", cls.temperature),
            spend_ceiling_usd=_env_float("REPOSAGE_SPEND_CEILING_USD", cls.spend_ceiling_usd),
            max_retries=_env_int("REPOSAGE_MAX_RETRIES", cls.max_retries),
            request_timeout_s=_env_float("REPOSAGE_TIMEOUT_S", cls.request_timeout_s),
        )

    def require_api_key(self) -> str:
        """Return the API key, or raise a clear error saying which variable to set."""
        if not self.api_key:
            variable = API_KEY_ENV[self.provider]
            raise RuntimeError(
                f"{variable} is not set, but provider is {self.provider!r}. "
                f"Copy .env.example to .env and fill it in, or export the "
                f"variable in your shell."
            )
        return self.api_key
