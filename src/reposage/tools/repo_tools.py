"""get_file and search_code, reading a local checkout.

Both tools are built by a factory that closes over the clone root, so the root
is fixed at construction and cannot be supplied by the model. That is the first
half of the security story; `_resolve_within` is the second.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .registry import Tool

# Directories that are never worth searching, and would dominate the results if
# they were. Checked by name at every level rather than by path prefix.
SKIP_DIRECTORIES = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".tox", ".idea", ".vscode", "site-packages",
}

MAX_SEARCHABLE_BYTES = 1_000_000  # skip anything bigger; it is data, not code
DEFAULT_FILE_LINE_CAP = 200


class PathEscapeError(ValueError):
    """The requested path resolves outside the repository."""


def _resolve_within(root: Path, candidate: str) -> Path:
    """Resolve `candidate` under `root`, refusing anything that escapes.

    This is not paranoia. The path comes from a language model, and from phase
    4 that model will have been reading repository content -- issue text,
    README files, code comments -- any of which can contain instructions aimed
    at it. `get_file("../../../.ssh/id_rsa")` is one sentence away at all times.

    `resolve()` before the check, not after: it collapses `..` segments *and*
    follows symlinks, so a symlink pointing out of the tree is caught too. A
    check performed on the unresolved string would pass and then read the wrong
    file.
    """
    root = root.resolve()
    target = (root / candidate).resolve()
    if target != root and root not in target.parents:
        raise PathEscapeError(
            f"{candidate!r} resolves outside the repository. Paths must be "
            f"relative to the repository root."
        )
    return target


def _number_lines(lines: list[str], first_line_number: int) -> str:
    """Prefix each line with its number.

    Citations are the entire promise of this project, and a citation needs a
    line number. If the model only ever sees bare source text, the best it can
    do is quote -- and a quote is not a reference. Numbering here is what makes
    "routing.py:412" possible at all.
    """
    width = len(str(first_line_number + len(lines) - 1))
    return "\n".join(
        f"{first_line_number + offset:>{width}}: {line}"
        for offset, line in enumerate(lines)
    )


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
    root = Path(root).resolve()

    def get_file(params: GetFileParams) -> str:
        target = _resolve_within(root, params.path)
        if not target.is_file():
            # Phrased as a hint, not just a refusal: this string goes back to
            # the model, and its next move should be a better path.
            return (
                f"No file at {params.path!r}. Use search_code to locate it, or "
                f"check the path is relative to the repository root."
            )

        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)

        start = params.start_line or 1
        end = params.end_line or total
        if start > total:
            return f"{params.path} has {total} lines; start_line {start} is past the end."
        end = min(end, total)
        if end < start:
            return f"end_line ({end}) is before start_line ({start})."

        window = lines[start - 1 : end]
        truncated = ""
        # A whole large file is rarely what is wanted and always expensive --
        # it is paid for again on every later turn, since history is resent.
        if params.start_line is None and params.end_line is None and total > line_cap:
            window = window[:line_cap]
            end = line_cap
            truncated = (
                f"\n\n[showing lines 1-{line_cap} of {total}. Request a range "
                f"with start_line/end_line to see more.]"
            )

        header = f"{params.path} (lines {start}-{end} of {total})\n"
        return header + _number_lines(window, start) + truncated

    def search_code(params: SearchCodeParams) -> str:
        try:
            flags = re.IGNORECASE if params.ignore_case else 0
            expression = re.compile(params.pattern, flags)
        except re.error as exc:
            return f"Invalid regular expression {params.pattern!r}: {exc}"

        matches: list[str] = []
        total_found = 0

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRECTORIES for part in path.relative_to(root).parts):
                continue
            if params.glob and not path.match(params.glob):
                continue
            try:
                if path.stat().st_size > MAX_SEARCHABLE_BYTES:
                    continue
                raw = path.read_bytes()
            except OSError:
                continue
            # A NUL byte in the first block is the standard cheap test for
            # "binary", and avoids decoding megabytes of images.
            if b"\x00" in raw[:1024]:
                continue

            relative = path.relative_to(root).as_posix()
            for number, line in enumerate(
                raw.decode("utf-8", errors="replace").splitlines(), start=1
            ):
                if expression.search(line):
                    total_found += 1
                    if len(matches) < params.max_results:
                        matches.append(f"{relative}:{number}: {line.strip()[:200]}")

        if not matches:
            return (
                f"No matches for {params.pattern!r}"
                + (f" in {params.glob}" if params.glob else "")
                + ". Try a broader pattern or drop the glob."
            )

        # Telling the model it is seeing a subset is the difference between it
        # narrowing the search and it answering confidently from a fraction of
        # the evidence.
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
