"""The tools the agent can call, and a helper that builds the default set."""

from __future__ import annotations

from pathlib import Path

from ..repo import DEFAULT_REPO, ensure_clone
from .github_tools import build_github_tools
from .registry import Tool, ToolInvocation, ToolRegistry, results_to_messages
from .repo_tools import PathEscapeError, build_repo_tools

__all__ = [
    "Tool",
    "ToolInvocation",
    "ToolRegistry",
    "PathEscapeError",
    "build_repo_tools",
    "build_github_tools",
    "build_default_registry",
    "results_to_messages",
]


def build_default_registry(repo: str = DEFAULT_REPO) -> tuple[ToolRegistry, Path]:
    """Clone `repo` if needed and return a registry with all three tools, plus the clone path."""
    # Clone up front: a slow clone mid-turn would look like a hung model.
    clone = ensure_clone(repo)
    registry = ToolRegistry()
    for tool in build_repo_tools(clone) + build_github_tools(repo):
        registry.add(tool)
    return registry, clone
