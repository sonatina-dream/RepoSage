"""The list_issues tool: reads issues from the GitHub API (issues are not in the clone)."""

from __future__ import annotations

import os
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, Field

from .registry import Tool

GITHUB_API = "https://api.github.com"


class ListIssuesParams(BaseModel):
    state: Literal["open", "closed", "all"] = Field(
        default="closed",
        description="Issue state. Closed issues usually contain the resolution.",
    )
    query: Optional[str] = Field(
        default=None,
        description="Case-insensitive substring to filter issue titles by.",
    )
    limit: int = Field(default=5, ge=1, le=20, description="Maximum issues to return.")


def github_headers() -> dict[str, str]:
    """GitHub API headers, with GITHUB_TOKEN if set (raises the limit from 60 to 5000 requests/hour)."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def build_github_tools(repo: str) -> list[Tool]:
    """Return the list_issues tool for one repository."""

    def list_issues(params: ListIssuesParams) -> str:
        """List issues, most-commented first, optionally filtered by title."""
        with httpx.Client(timeout=20.0, headers=github_headers()) as http:
            response = http.get(
                f"{GITHUB_API}/repos/{repo}/issues",
                params={
                    "state": params.state,
                    "per_page": 100,
                    "sort": "comments",
                    "direction": "desc",
                },
            )
            if response.status_code == 403:
                # Tell the model it hit a rate limit, so it doesn't conclude there are no issues.
                return (
                    "GitHub rate limit reached. Set GITHUB_TOKEN to raise the "
                    "limit from 60 to 5000 requests/hour, or try again later."
                )
            response.raise_for_status()
            # The issues endpoint also returns pull requests; drop them.
            issues = [item for item in response.json() if "pull_request" not in item]

        if params.query:
            needle = params.query.lower()
            issues = [issue for issue in issues if needle in issue["title"].lower()]

        issues = issues[: params.limit]
        if not issues:
            return f"No {params.state} issues found" + (
                f" matching {params.query!r}." if params.query else "."
            )

        lines = []
        for issue in issues:
            labels = ", ".join(label["name"] for label in issue.get("labels", []))
            body = (issue.get("body") or "").strip().replace("\n", " ")
            lines.append(
                f"#{issue['number']} [{issue['state']}] {issue['title']}\n"
                f"    comments={issue.get('comments', 0)} labels=[{labels}]\n"
                f"    {body[:300]}\n"
                f"    {issue['html_url']}"
            )
        return "\n".join(lines)

    return [
        Tool(
            name="list_issues",
            description=(
                f"List issues from the {repo} GitHub repository, most-discussed "
                "first. Use this for questions about reported problems, known "
                "bugs, or how maintainers responded to something — not for "
                "questions about how the code works, which get_file and "
                "search_code answer better and faster."
            ),
            params=ListIssuesParams,
            handler=list_issues,
        )
    ]
