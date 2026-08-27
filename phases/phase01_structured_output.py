"""Phase 1 -- turn closed GitHub issues into validated records.

    python phases/phase01_structured_output.py --repo fastapi/fastapi --limit 5

Why closed issues, and why now. Phase 5 needs an evaluation set: questions
about the repository with known-correct answers. Writing one by hand is slow
and biased toward what you already know the system can do. A closed issue is
better raw material -- a real developer asked a real question and a maintainer
answered it, and the thread contains both. Mining those into typed records is
exactly the structured-output problem, so phase 1 builds the eval seed corpus
as a side effect of learning the technique.

The records that land in data/issues/ are *candidates*, not ground truth. A
model summarised them, and phase 5 will not trust a model's summary as the
yardstick for that same model. They get reviewed by hand before they count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reposage.config import Settings, format_usd  # noqa: E402
from reposage.extraction import ExtractionError, IssueSummary, extract  # noqa: E402
from reposage.llm import BudgetExceeded, LLMClient, describe_target  # noqa: E402

GITHUB_API = "https://api.github.com"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "issues"


def github_headers() -> dict[str, str]:
    """Unauthenticated is 60 requests/hour per IP; a bare token gives 5000."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_closed_issues(repo: str, limit: int) -> list[dict]:
    """Fetch closed issues, filtering out pull requests.

    The GitHub issues endpoint returns PRs as well -- they are issues with a
    `pull_request` key. A PR thread is a code review, not a question with an
    answer, so it is the wrong shape for the eval set. Over-fetch and filter
    rather than assuming a fixed ratio.
    """
    with httpx.Client(timeout=30.0, headers=github_headers()) as http:
        response = http.get(
            f"{GITHUB_API}/repos/{repo}/issues",
            params={
                "state": "closed",
                "per_page": min(100, limit * 3),
                "sort": "comments",  # busy threads carry the actual answer
                "direction": "desc",
            },
        )
        response.raise_for_status()
        issues = [item for item in response.json() if "pull_request" not in item][:limit]

        for issue in issues:
            comments = http.get(issue["comments_url"], params={"per_page": 10})
            comments.raise_for_status()
            issue["_comments"] = comments.json()

    return issues


def render_thread(issue: dict, max_chars: int = 6000) -> str:
    """Flatten an issue and its comments into the text the model reads.

    Truncated deliberately. Some threads run to tens of thousands of
    characters, and paying to send all of it buys very little: the problem
    statement is at the top and the resolution is usually in the first few
    replies. Truncating to a known budget also keeps the per-issue cost
    predictable, which is what makes a 200-issue run a decision you can price
    beforehand rather than discover afterwards.
    """
    parts = [
        f"Issue #{issue['number']}: {issue['title']}",
        f"State: {issue['state']}  Labels: {[l['name'] for l in issue.get('labels', [])]}",
        "",
        (issue.get("body") or "(no description)").strip(),
    ]
    for comment in issue.get("_comments", []):
        author = comment.get("user", {}).get("login", "unknown")
        parts.append(f"\n--- comment by {author} ---\n{(comment.get('body') or '').strip()}")

    thread = "\n".join(parts)
    if len(thread) > max_chars:
        thread = thread[:max_chars] + "\n[... thread truncated ...]"
    return thread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="fastapi/fastapi")
    parser.add_argument("--provider", help="anthropic or deepseek; defaults to REPOSAGE_PROVIDER.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--model",
        default=None,
        help="Defaults to the provider's quality model — extraction quality is "
             "the point here, so this is not the usual fast tier.",
    )
    args = parser.parse_args()

    settings = Settings.from_env(provider=args.provider)
    client = LLMClient(settings)
    print(describe_target(settings))

    print(f"Fetching up to {args.limit} closed issues from {args.repo} ...")
    issues = fetch_closed_issues(args.repo, args.limit)
    print(f"Got {len(issues)}.\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0

    for issue in issues:
        number = issue["number"]
        print(f"#{number} {issue['title'][:60]}")

        try:
            summary: IssueSummary = extract(
                client,
                IssueSummary,
                render_thread(issue),
                model=args.model or client.quality_model,
            )
        except ExtractionError as exc:
            # A failure here is data, not a crash. Some threads genuinely do
            # not contain an answer, and the run should continue and tell you
            # how many it dropped.
            print(f"   skipped: {exc}\n")
            continue
        except BudgetExceeded as exc:
            print(f"\nStopped by the spend ceiling: {exc}")
            break

        destination = OUTPUT_DIR / f"{args.repo.replace('/', '_')}_{number}.json"
        destination.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        saved += 1

        print(f"   category={summary.category.value} confidence={summary.confidence}")
        print(f"   Q: {summary.eval_question}")
        print(f"   answerable from repo alone: {summary.answerable_from_repo}\n")

    print(f"Saved {saved}/{len(issues)} records to {OUTPUT_DIR}")
    print(f"Usage: {client.usage.summary()}")
    if saved:
        print(f"Cost per usable record: {format_usd(client.usage.cost_usd / saved)}")


if __name__ == "__main__":
    main()
