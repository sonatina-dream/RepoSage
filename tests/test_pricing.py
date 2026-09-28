"""Tests for cost maths and settings: peak/off-peak pricing, cache rates and API keys."""

from datetime import datetime, timezone

import pytest

from reposage.config import (
    ANTHROPIC,
    DEEPSEEK,
    DEEPSEEK_FLASH,
    HAIKU,
    Settings,
    UnknownModelError,
    estimate_cost,
    provider_for,
    spec_for,
)

# Tuesday. 09:00 UTC falls inside DeepSeek's 06:00-10:00 peak window; 14:00
# does not. Both are fixed rather than derived from "now", because a test that
# reads the wall clock passes or fails depending on when CI happens to run.
PEAK = datetime(2026, 8, 25, 9, 0, tzinfo=timezone.utc)
OFF_PEAK = datetime(2026, 8, 25, 14, 0, tzinfo=timezone.utc)
SATURDAY = datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc)


def test_flat_rate_model_ignores_the_clock():
    """Checks that a flat-rate model costs the same at any time of day."""
    peak = estimate_cost(HAIKU, 1_000_000, 1_000_000, at=PEAK)
    off_peak = estimate_cost(HAIKU, 1_000_000, 1_000_000, at=OFF_PEAK)
    assert peak == off_peak == pytest.approx(6.00)  # $1 in + $5 out


def test_deepseek_off_peak_is_half_of_peak():
    """Checks that DeepSeek's off-peak price is half the peak price."""
    peak = estimate_cost(DEEPSEEK_FLASH, 1_000_000, 1_000_000, at=PEAK)
    off_peak = estimate_cost(DEEPSEEK_FLASH, 1_000_000, 1_000_000, at=OFF_PEAK)
    assert peak == pytest.approx(0.44 + 1.32)
    assert off_peak == pytest.approx(peak / 2)


def test_weekends_are_never_peak():
    """Checks that weekend hours always bill at the off-peak rate."""
    assert not spec_for(DEEPSEEK_FLASH).is_peak(SATURDAY)
    assert estimate_cost(DEEPSEEK_FLASH, 1_000_000, 0, at=SATURDAY) == pytest.approx(0.22)


def test_cached_prompt_tokens_bill_at_the_cache_rate():
    """Checks that cached prompt tokens are a cheaper part of the prompt, not extra tokens."""
    all_fresh = estimate_cost(DEEPSEEK_FLASH, 1_000_000, 0, at=PEAK)
    all_cached = estimate_cost(
        DEEPSEEK_FLASH, 1_000_000, 0, cached_input_tokens=1_000_000, at=PEAK
    )
    assert all_fresh == pytest.approx(0.44)
    assert all_cached == pytest.approx(0.014)
    assert all_cached < all_fresh / 10


def test_cached_tokens_cannot_exceed_the_prompt():
    """Checks that a bad cache count can never produce a negative cost."""
    cost = estimate_cost(DEEPSEEK_FLASH, 1000, 0, cached_input_tokens=99_999, at=PEAK)
    assert cost > 0


def test_unknown_model_raises_rather_than_pricing_at_zero():
    """Checks that pricing an unknown model raises instead of returning $0."""
    with pytest.raises(UnknownModelError):
        estimate_cost("gpt-9-imaginary", 100, 100)


def test_dated_snapshots_price_like_their_family():
    """Checks that a dated model ID gets its family's price."""
    assert provider_for("claude-sonnet-5-20260401") == ANTHROPIC
    assert estimate_cost("claude-sonnet-5-20260401", 1_000_000, 0) == pytest.approx(2.00)


def test_settings_pick_the_right_key_variable_per_provider(monkeypatch):
    """Checks that each provider reads its own API key variable."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")

    assert Settings.from_env(provider=ANTHROPIC).api_key == "sk-ant-test"
    assert Settings.from_env(provider=DEEPSEEK).api_key == "sk-ds-test"


def test_each_provider_has_a_fast_and_a_quality_model():
    """Checks that every provider defines both a fast and a quality model."""
    for provider in (ANTHROPIC, DEEPSEEK):
        settings = Settings(provider=provider, api_key="x")
        assert settings.model == settings.fast_model
        assert settings.quality_model != settings.fast_model
        # Both must be priceable, or cost reporting silently breaks later.
        assert provider_for(settings.fast_model) == provider
        assert provider_for(settings.quality_model) == provider
