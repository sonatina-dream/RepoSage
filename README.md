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

## Providers

RepoSage runs against **DeepSeek** (default) or **Anthropic**, selected with
one environment variable:

```bash
REPOSAGE_PROVIDER=deepseek    # or: anthropic
```

This is not vendor-neutrality for its own sake. It buys two concrete things:

- **A debugging control.** From phase 3, when the agent misbehaves, flipping
  the provider tells you whether the fault is in your loop or in the model.
  Without that, every strange trace is ambiguous.
- **A phase 5 result instead of a phase 5 assumption.** Running the same eval
  suite against both turns "which model should this use?" into a measured
  answer with a cost-per-correct-answer attached.

Everything above `providers/` — extraction, the agent loop, the eval harness —
is written once. What the provider layer absorbs:

| | Anthropic | DeepSeek (OpenAI-compatible) |
|---|---|---|
| System prompt | top-level `system` parameter | first message, `role: "system"` |
| Reply body | `content`, a list of typed blocks | `choices[0].message.content`, a string |
| Stop reason | `end_turn` / `max_tokens` | `stop` / `length` |
| Prompt cache | not enabled here | automatic; usage reports hit/miss split |
| Pricing | flat rate | peak / off-peak, and a separate cache-hit rate |

The system-prompt row is the one that bites: send an Anthropic-shaped call to
an OpenAI-style endpoint and the system prompt is silently dropped. The model
answers anyway, slightly worse, with no error to tell you why.

## Roadmap

| Phase | What it covers | Status |
|------:|----------------|--------|
| 0 | API fundamentals: messages, tokens, cost, temperature, streaming, statelessness | Done |
| 1 | Structured output: JSON schema, Pydantic validation, retry on validation error | Done |
| 2 | Tool calling: schemas, the tool-call → execute → result round trip, a registry | Next |
| 3 | Agent harness from scratch: the loop, iteration cap, context budget, tool errors, tracing | Planned |
| 4 | RAG: structure-aware chunking, embeddings, hybrid retrieval, reranking, citations | Planned |
| 5 | Evaluation: ground-truth set from closed issues, retrieval and groundedness metrics, CI gate | Planned |
| 6 | Production: FastAPI + SSE, Docker, tracing, per-request cost, injection guardrail, UI | Planned |
| 7 | Write-up: architecture, measured results, decisions and trade-offs | Planned |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add DEEPSEEK_API_KEY (or ANTHROPIC_API_KEY)

python phases/phase00_first_call.py --all
python phases/phase01_structured_output.py --repo fastapi/fastapi --limit 5

pytest                      # 16 tests, no API key needed
```

Both phase scripts take `--provider` if you want to run one against the other
vendor without editing `.env`.

## Layout

```
src/reposage/config.py                providers, model IDs, pricing, settings, cost maths
src/reposage/llm.py                   accounting, spend ceiling, retries — vendor-agnostic
src/reposage/providers/base.py        the contract, and what actually differs between vendors
src/reposage/providers/*_provider.py  one wire-format translator each
src/reposage/extraction.py            Pydantic schemas + validated extraction with retry
phases/phase00_*.py                   five runnable demos of the API fundamentals
phases/phase01_*.py                   mines closed GitHub issues into validated records
tests/                                16 deterministic tests, scripted fakes, no API key
data/issues/                          extracted eval candidates (regenerable, gitignored)
```

## Cost discipline

Every model call goes through one client, and that client enforces a
cumulative spend ceiling — checked *before* each request, assuming worst-case
output length. An agent loop with a bug should raise an exception, not an
invoice. The ceiling is deliberately not per-provider: it is the one guard rail
that has to hold everywhere.

Each provider exposes a **fast** and a **quality** model rather than a single
default. Fast runs while iterating; quality is reserved for the places where
the reasoning itself is the deliverable — extraction from messy text, the
agent's tool-selection decisions, and the phase 5 judge.

Cost accounting handles two things a flat rate does not: DeepSeek's peak /
off-peak schedule, and prompt-cache hits, which bill at roughly a thirtieth of
a miss. The second matters enormously from phase 3, where the agent resends a
growing history every turn — ignore the cache split and you overstate cost by
an order of magnitude.

The pricing table in `config.py` carries the date it was last checked against
both vendors' published prices, because a hardcoded price list is a liability
that should at least be honest about its age.

## Decisions and trade-offs so far

**The spend ceiling lives in the client, not at the call sites.** It is the one
chokepoint every request passes through, and it is checked before sending,
assuming worst-case output length — a budget check that runs afterwards only
tells you about money already spent.

**Providers are dumb translators.** Retries, budget and accounting are
cross-cutting concerns that must behave identically regardless of who serves
the request, so they stay in `LLMClient`. A provider knows one wire format and
nothing else.

**Provider SDKs are imported lazily.** Selecting DeepSeek should not require
installing Anthropic's SDK. The import happens inside the branch that needs it,
and a missing package produces "pip install X", not a traceback.

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

**The SDKs' built-in retries are turned off.** From phase 3, the loop has to
tell "the transport failed, resend it" apart from "the model returned something
unusable, send the error back". Those need opposite handling, so the retry
policy is explicit and ours. Streaming is deliberately not retried: a stream
that fails halfway has already delivered text, so resending duplicates rather
than repairs.

**Cost maths takes a timestamp rather than reading the clock.** A function that
calls `now()` internally cannot be asserted against, and this one decides what
you are billed.

**Retrieval will be a tool, not a pipeline.** In phase 4 the agent decides when
to search, rather than every question being forced through a fixed retrieve →
stuff → answer chain. Questions like "what changed in this file recently?" do
not need a vector search, and a pipeline cannot decline to run.

**Closed issues seed the eval set.** Each one is a question a real developer
asked and a maintainer answered, which is free ground truth. The extracted
records are candidates only — phase 5 will not use a model's own summary as the
yardstick for that model without a human pass over it first.
