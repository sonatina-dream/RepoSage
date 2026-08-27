"""The local checkout the code tools read from.

Why a clone rather than the GitHub API. An agent makes five to ten tool calls
to answer one question. Over the API that is five to ten network round trips
against a rate limit, and the latency dominates the loop -- you feel it as an
agent that takes half a minute to say something simple. A shallow clone is a
one-off cost, after which every read is a local file read: free, instant, and
available offline. Phase 4 needs a local copy to index anyway.

Shallow (`--depth 1`) because nothing here needs history yet. When phase 4 or 5
wants "what changed and when", this is the function that grows a `depth`
argument -- and that is a deliberate decision to make then, with a reason,
rather than paying for the full history of a large repository now.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

DEFAULT_REPO = "fastapi/fastapi"
CLONE_ROOT = Path(__file__).resolve().parents[2] / "data" / "repos"


class CloneError(RuntimeError):
    pass


def clone_path(repo: str, root: Path | None = None) -> Path:
    """Where `owner/name` lives on disk."""
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(f"Expected 'owner/name', got {repo!r}.")
    return (root or CLONE_ROOT) / repo.replace("/", "__")


def ensure_clone(repo: str = DEFAULT_REPO, root: Path | None = None) -> Path:
    """Clone the repository if it is not already on disk. Returns its path."""
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
