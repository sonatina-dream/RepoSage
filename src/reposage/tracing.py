"""Run traces: one JSON object per line, one line per agent step, so a run can be replayed by reading it."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACE_ROOT = Path(__file__).resolve().parents[2] / "data" / "traces"


def new_trace_path(root: Path | None = None) -> Path:
    """A fresh, timestamped trace file path under `root` (default data/traces/)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (root or TRACE_ROOT) / f"run-{stamp}.jsonl"


class TraceWriter:
    """Appends records to a JSONL file, flushing each so a crashed run still leaves its trace."""

    def __init__(self, path: Path) -> None:
        """Create the parent folder; the file appears on the first write."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        """Append one record as a line, stamped with the UTC time."""
        stamped = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stamped, default=str) + "\n")


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Load every record of a trace file."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
