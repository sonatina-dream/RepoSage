"""Makes and finds the local shallow clone that the code tools read from."""

from __future__ import annotations

import subprocess
from pathlib import Path

DEFAULT_REPO = "fastapi/fastapi"
CLONE_ROOT = Path(__file__).resolve().parents[2] / "data" / "repos"


class CloneError(RuntimeError):
    """Raised when `git clone` fails."""


def clone_path(repo: str, root: Path | None = None) -> Path:
    """Return the folder where `owner/name` is (or will be) cloned."""
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(f"Expected 'owner/name', got {repo!r}.")
    return (root or CLONE_ROOT) / repo.replace("/", "__")


def ensure_clone(repo: str = DEFAULT_REPO, root: Path | None = None) -> Path:
    """Shallow-clone the repository unless it is already on disk; return its folder."""
    destination = clone_path(repo, root)
    if (destination / ".git").exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    print(f"Cloning {url} (shallow) → {destination} ...")

    completed = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", url, str(destination)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CloneError(
            f"git clone failed for {repo}:\n{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return destination
