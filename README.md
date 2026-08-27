# RepoSage

A question-answering agent for a real open-source repository, built from
scratch to understand how these systems actually work.

Ask it "how does dependency overriding work in tests?" and it answers from the
repository's own code, citing the file and line behind every claim. If it
cannot ground a claim in the repo, it says so rather than inventing one.

Default target: [`fastapi/fastapi`](https://github.com/fastapi/fastapi).

## Why it is built this way

Phases 2 to 4 use no agent framework. The tool-calling round trip, the agent
loop, the context budget and the retrieval layer are all written by hand,
because the goal is to understand the machinery rather than to configure it. A
LangGraph port lands on a separate branch afterwards, so the same system exists
both ways and the trade-off is visible instead of asserted.

## Roadmap

| Phase | What it covers | Status |
|------:|----------------|--------|
| 0 | API fundamentals: messages, tokens, cost, temperature, streaming, statelessness | Done |
| 1 | Structured output: JSON schema, Pydantic validation, retry on validation error | Next |
| 2 | Tool calling: schemas, the `tool_use` → execute → `tool_result` round trip, a registry | Planned |
| 3 | Agent harness from scratch: the loop, iteration cap, context budget, tool errors, tracing | Planned |
| 4 | RAG: structure-aware chunking, embeddings, hybrid retrieval, reranking, citations | Planned |
| 5 | Evaluation: ground-truth set from closed issues, retrieval and groundedness metrics, CI gate | Planned |
| 6 | Production: FastAPI + SSE, Docker, tracing, per-request cost, injection guardrail, UI | Planned |
| 7 | Write-up: architecture, measured results, decisions and trade-offs | Planned |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY

python phases/phase00_first_call.py --all
```

## Layout

```
src/reposage/config.py   model IDs, pricing table, settings, cost maths
src/reposage/llm.py      API wrapper: complete(), stream(), accounting, spend ceiling
phases/phase00_*.py      five runnable demos of the API fundamentals
```

## Cost discipline

Every model call goes through one client, and that client enforces a
cumulative spend ceiling — checked *before* each request, assuming worst-case
output length. An agent loop with a bug should raise an exception, not an
invoice. Haiku is the default while iterating; Sonnet is used only where the
quality of the reasoning is the deliverable.

The pricing table in `config.py` carries the date it was last checked against
Anthropic's published prices, because a hardcoded price list is a liability
that should at least be honest about its age.
