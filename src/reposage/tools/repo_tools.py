"""The get_file and search_code tools, which read a local clone of the repository.

The clone root is fixed when the tools are built, so the model can never choose it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .registry import Tool

# Folders never worth searching, skipped by name at any depth.
SKIP_DIRECTORIES = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".tox", ".idea", ".vscode", "site-packages",
}

MAX_SEARCHABLE_BYTES = 1_000_000  # skip anything bigger; it is data, not code
DEFAULT_FILE_LINE_CAP = 200


class PathEscapeError(ValueError):
    """Raised when a requested path points outside the repository."""


def _resolve_within(root: Path, candidate: str) -> Path:
    """Return the full path of `candidate` inside `root`; raise PathEscapeError if it escapes."""
    # Resolve first: it collapses ".." and follows symlinks, so both escapes get caught.
    root = root.resolve()
    target = (root / candidate).resolve()
    if target != root and root not in target.parents:
        raise PathEscapeError(
            f"{candidate!r} resolves outside the repository. Paths must be "
            f"relative to the repository root."
        )
    return target


def _number_lines(lines: list[str], first_line_number: int) -> str:
    """Prefix each line with its line number, so the model can cite "file:line"."""
    width = len(str(first_line_number + len(lines) - 1))
    return "\n".join(
        f"{first_line_number + offset:>{width}}: {line}"
        for offset, line in enumerate(lines)
    )


def _files_to_search(root: Path) -> list[Path]:
    """List every file under `root` in sorted order, never entering skipped folders."""
    found: list[Path] = []
    for folder, dirnames, filenames in os.walk(root):
        # Assigning to dirnames[:] stops os.walk from entering skipped folders.
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRECTORIES]
        # Files are checked by name too (a submodule's `.git` is a file).
        found.extend(Path(folder, name) for name in filenames if name not in SKIP_DIRECTORIES)
    return sorted(found)


class GetFileParams(BaseModel):
    path: str = Field(
        description="File path relative to the repository root, e.g. 'fastapi/routing.py'."
    )
    start_line: Optional[int] = Field(
        default=None, ge=1, description="First line to return, 1-based. Omit for the start."
    )
    end_line: Optional[int] = Field(
        default=None, ge=1, description="Last line to return, inclusive. Omit for the end."
    )


class SearchCodeParams(BaseModel):
    pattern: str = Field(
        description="A Python regular expression to search for in file contents."
    )
    glob: Optional[str] = Field(
        default=None,
        description="Restrict to paths matching this glob, e.g. '*.py' or 'docs/*.md'.",
    )
    max_results: int = Field(
        default=20, ge=1, le=100, description="Maximum matching lines to return."
    )
    ignore_case: bool = Field(default=False, description="Case-insensitive matching.")


def build_repo_tools(root: Path, line_cap: int = DEFAULT_FILE_LINE_CAP) -> list[Tool]:
    """Return the get_file and search_code tools for the clone at `root`."""
    root = Path(root).resolve()

    def get_file(params: GetFileParams) -> str:
        """Return a file's lines with numbers; without a range, only the first `line_cap` lines."""
        target = _resolve_within(root, params.path)
        if not target.is_file():
            # A hint, not just a refusal: the model's next move should be a better path.
            return (
                f"No file at {params.path!r}. Use search_code to locate it, or "
                f"check the path is relative to the repository root."
            )

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)

        start = params.start_line or 1
        if start > total:
            return f"{params.path} has {total} lines; start_line {start} is past the end."
        end = min(params.end_line or total, total)
        if end < start:
            return f"end_line ({end}) is before start_line ({start})."

        truncated = ""
        # A whole large file is expensive, and it is paid for again on every later turn.
        if params.start_line is None and params.end_line is None and total > line_cap:
            end = line_cap
            truncated = (
                f"\n\n[showing lines 1-{line_cap} of {total}. Request a range "
                f"with start_line/end_line to see more.]"
            )

        header = f"{params.path} (lines {start}-{end} of {total})\n"
        return header + _number_lines(lines[start - 1 : end], start) + truncated

    def search_code(params: SearchCodeParams) -> str:
        """Return lines matching a regex, as "path:line: text", with a count of all matches."""
        try:
            flags = re.IGNORECASE if params.ignore_case else 0
            expression = re.compile(params.pattern, flags)
        except re.error as exc:
            return f"Invalid regular expression {params.pattern!r}: {exc}"

        matches: list[str] = []
        total_found = 0

        for path in _files_to_search(root):
            # Also skips FIFOs and broken symlinks, which os.walk lists as files.
            if not path.is_file():
                continue
            if params.glob and not path.match(params.glob):
                continue
            try:
                if path.stat().st_size > MAX_SEARCHABLE_BYTES:
                    continue
                raw = path.read_bytes()
            except OSError:
                continue
            # A NUL byte near the start means a binary file; skip it.
            if b"\x00" in raw[:1024]:
                continue

            for number, line in enumerate(
                raw.decode("utf-8", errors="replace").splitlines(), start=1
            ):
                if expression.search(line):
                    total_found += 1
                    if len(matches) < params.max_results:
                        relative = path.relative_to(root).as_posix()
                        matches.append(f"{relative}:{number}: {line.strip()[:200]}")

        if not matches:
            return (
                f"No matches for {params.pattern!r}"
                + (f" in {params.glob}" if params.glob else "")
                + ". Try a broader pattern or drop the glob."
            )

        # Say when this is a subset, so the model narrows the search instead of guessing.
        header = (
            f"{total_found} match(es); showing {len(matches)}.\n"
            if total_found > len(matches)
            else f"{total_found} match(es).\n"
        )
        return header + "\n".join(matches)

    return [
        Tool(
            name="get_file",
            description=(
                "Read the contents of one file from the repository, with line "
                "numbers. Use this once you know which file you need — to read "
                "an implementation, or to get the exact lines to cite. Prefer a "
                "start_line/end_line range for large files; without one, only "
                "the first 200 lines are returned. Use search_code first if you "
                "do not already know the path."
            ),
            params=GetFileParams,
            handler=get_file,
        ),
        Tool(
            name="search_code",
            description=(
                "Search the repository's file contents with a regular "
                "expression, returning matching lines as 'path:line: text'. "
                "Use this to find where something is defined or used when you "
                "do not know the file — a function name, a class, a setting, an "
                "error message. Returns matching lines only, not full context: "
                "follow up with get_file to read around a match."
            ),
            params=SearchCodeParams,
            handler=search_code,
        ),
    ]
