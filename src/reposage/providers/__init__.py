"""Builds the provider for a vendor; SDKs are imported only when that vendor is chosen."""

from __future__ import annotations

from ..config import ANTHROPIC, DEEPSEEK
from .base import (
    Message,
    Provider,
    ProviderContractError,
    ProviderReply,
    TokenUsage,
    ToolCall,
    ToolResult,
)

__all__ = [
    "Message",
    "Provider",
    "ProviderContractError",
    "ProviderReply",
    "TokenUsage",
    "ToolCall",
    "ToolResult",
    "build_provider",
]


def build_provider(name: str, api_key: str, timeout_s: float) -> Provider:
    """Create the provider for `name`, importing its SDK only now."""
    if name == ANTHROPIC:
        try:
            from .anthropic_provider import AnthropicProvider
        except ImportError as exc:  # pragma: no cover - install-time only
            raise RuntimeError("Provider 'anthropic' needs: pip install anthropic") from exc
        return AnthropicProvider(api_key=api_key, timeout_s=timeout_s)

    if name == DEEPSEEK:
        try:
            from .deepseek_provider import DeepSeekProvider
        except ImportError as exc:  # pragma: no cover - install-time only
            raise RuntimeError("Provider 'deepseek' needs: pip install openai") from exc
        return DeepSeekProvider(api_key=api_key, timeout_s=timeout_s)

    raise ValueError(f"Unknown provider {name!r}.")
