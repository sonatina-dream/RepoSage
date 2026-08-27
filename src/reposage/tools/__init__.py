"""Tools the agent can call, and the registry that keeps them honest."""

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


def build_default_registry(
    repo: str = DEFAULT_REPO, root: Path | None = None
) -> tuple[ToolRegistry, Path]:
    """The three tools RepoSage ships, wired to a local clone of `repo`.

    Returns the registry and the clone path. Cloning happens here rather than
    lazily inside a tool because a multi-second `git clone` in the middle of an
    agent turn looks exactly like a hung model.
    """
    clone = root or ensure_clone(repo)
    registry = ToolRegistry()
    for tool in build_repo_tools(clone) + build_github_tools(repo):
        registry.add(tool)
    return registry, clone
