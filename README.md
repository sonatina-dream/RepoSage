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
| 1 | Structured output: JSON schema, Pydantic validation, retry on validation error | Done |
| 2 | Tool calling: schemas, the `tool_use` → execute → `tool_result` round trip, a registry | Next |
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
python phases/phase01_structured_output.py --repo fastapi/fastapi --limit 5

pytest        # 7 tests, no API key needed
```

## Layout

```
src/reposage/config.py       model IDs, pricing table, settings, cost maths
src/reposage/llm.py          API wrapper: complete(), stream(), accounting, spend ceiling
src/reposage/extraction.py   Pydantic schemas + validated extraction with retry
phases/phase00_*.py          five runnable demos of the API fundamentals
phases/phase01_*.py          mines closed GitHub issues into validated records
tests/test_extraction.py     deterministic tests against a scripted fake model
data/issues/                 extracted eval candidates (regenerable, gitignored)
```

## Decisions and trade-offs so far

**The spend ceiling lives in the client, not at the call sites.** It is the one
chokepoint every request passes through, and it is checked before sending,
assuming worst-case output length — a budget check that runs afterwards only
tells you about money already spent.

**Temperature 0 for anything that gets parsed.** Sampling variety is a feature
for prose and a defect for a record you are going to validate, evaluate or
diff. Worth saying explicitly: temperature 0 is near-deterministic, not a seed.

**Validation failures re-prompt with the error text.** The retry sends the
model back its own broken output *plus* the validator's message, rather than
asking again identically. A model that cannot produce valid JSON unprompted is
usually able to fix a specific, named mistake — and because the API is
stateless, without replaying its own reply it has no idea what it is correcting.

**Pydantic models forbid extra fields.** The default — silently ignoring
unexpected keys — turns a hallucinated field into a valid object you never find
out about.

**The SDK's built-in retries are turned off.** From phase 3, the loop has to
tell "the transport failed, resend it" apart from "the model returned something
unusable, send the error back". Those need opposite handling, so the retry
policy is explicit and ours.

**Retrieval will be a tool, not a pipeline.** In phase 4 the agent decides when
to search, rather than every question being forced through a fixed retrieve →
stuff → answer chain. Questions like "what changed in this file recently?" do
not need a vector search, and a pipeline cannot decline to run.

**Closed issues seed the eval set.** Each one is a question a real developer
asked and a maintainer answered, which is free ground truth. The extracted
records are candidates only — phase 5 will not use a model's own summary as the
yardstick for that model without a human pass over it first.

## Cost discipline

Every model call goes through one client, and that client enforces a
cumulative spend ceiling — checked *before* each request, assuming worst-case
output length. An agent loop with a bug should raise an exception, not an
invoice. Haiku is the default while iterating; Sonnet is used only where the
quality of the reasoning is the deliverable.

The pricing table in `config.py` carries the date it was last checked against
Anthropic's published prices, because a hardcoded price list is a liability
that should at least be honest about its age.
