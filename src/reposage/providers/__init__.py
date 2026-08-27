"""Provider construction. Imports are lazy on purpose.

Each vendor's SDK is a real dependency with real install weight. Importing
both at module load would mean a DeepSeek-only user cannot start the project
without also installing Anthropic's SDK, and vice versa. So the import happens
inside the branch that needs it, and the ImportError is translated into a
message that says what to install.
"""

from __future__ import annotations

from ..config import ANTHROPIC, DEEPSEEK
from .base import Message, Provider, ProviderReply, TokenUsage

__all__ = ["Message", "Provider", "ProviderReply", "TokenUsage", "build_provider"]


def build_provider(name: str, api_key: str, timeout_s: float) -> Provider:
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
