# RepoSage

A question-answering agent for a real open-source repository, built from
scratch to understand how these systems actually work.

Ask it "how does dependency overriding work in tests?" and it answers from the
repository's own code, citing the file and line behind every claim. If it
cannot ground a claim in the repo, it says so rather than inventing one.

Default target: [`fastapi/fastapi`](https://github.com/fastapi/fastapi).

## Why it is built this way

Phases 2 to 5 use no agent framework. The tool-calling round trip, the agent
loop, context management and the retrieval layer are all written by hand,
because the goal is to understand the machinery rather than to configure it.
Phase 6 then rebuilds the agent in LangGraph and runs both versions against the
same eval suite, so the trade-off is measured instead of asserted.

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
| 2 | Tool calling: schemas, the tool-call → execute → result round trip, a registry | Done |
| 3 | Agent harness, the core loop: stop conditions, iteration cap, tool errors, tracing, streaming events, human approval of risky actions, and a 10-question smoke eval | Next |
| 3b | Context engineering and memory: token budget, compacting long histories, short- vs long-term memory, subagents | Planned |
| 4 | RAG: structure-aware chunking, embeddings, hybrid retrieval, reranking, citations | Planned |
| 5 | Evaluation: ground-truth set from closed issues, retrieval and groundedness metrics, LLM-as-judge, CI gate | Planned |
| 6 | LangGraph: state graphs, tool nodes, checkpointers, interrupts, multi-agent patterns; hand-written vs framework on the same evals | Planned |
| 7 | MCP: expose the repo tools as a Model Context Protocol server, use them from Claude Code / Desktop | Planned |
| 8 | Production: FastAPI + SSE, Docker, tracing, per-request cost, injection guardrail, UI | Planned |
| 9 | Write-up: architecture, measured results, decisions and trade-offs | Planned |

## Quick start

Needs Python 3.11+. On macOS the interpreter is `python3`, not `python` —
`python`, `pip` and `pytest` only appear on your PATH once the venv is active.

```bash
python3 --version                  # 3.11 or newer
python3 -m venv .venv
source .venv/bin/activate          # from here, `python` means the venv's

pip install -r requirements.txt
cp .env.example .env               # add DEEPSEEK_API_KEY (or ANTHROPIC_API_KEY)

python scripts/check_provider.py   # ~$0.00001 — run this first
pytest -q                          # 53 tests, no API key needed

python phases/phase00_first_call.py --all
python phases/phase01_structured_output.py --repo fastapi/fastapi --limit 5
python phases/phase02_tool_calling.py --show-schemas
```

`phase02` shallow-clones the target repository into `data/repos/` on first run.

If `python3` is missing or older than 3.11, install it with
[Homebrew](https://brew.sh): `brew install python@3.12`.

Both phase scripts take `--provider` if you want to run one against the other
vendor without editing `.env`.

## Layout

```
src/reposage/config.py                providers, model IDs, pricing, settings, cost maths
src/reposage/llm.py                   accounting, spend ceiling, retries — vendor-agnostic
src/reposage/providers/base.py        the contract, and what actually differs between vendors
src/reposage/providers/*_provider.py  one wire-format translator each
src/reposage/extraction.py            Pydantic schemas + validated extraction with retry
src/reposage/tools/registry.py        schema-from-handler, dispatch, errors-as-results
src/reposage/tools/repo_tools.py      get_file, search_code — path-confined, line-numbered
src/reposage/tools/github_tools.py    list_issues
src/reposage/repo.py                  the shallow clone the code tools read
phases/phase00_*.py                   five runnable demos of the API fundamentals
phases/phase01_*.py                   mines closed GitHub issues into validated records
phases/phase02_*.py                   one tool-calling round trip, every message printed
tests/                                53 deterministic tests, scripted fakes, no API key
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

**Tool errors are results, not exceptions.** A tool that fails returns its
failure to the model flagged `is_error`, and the model tries something else. If
a bad path raised instead, the run would die on the model's first typo. This is
the same shape as phase 1's validation-error retry, and it is what makes an
agent an agent rather than a script.

**The tool schema is generated from the handler's own Pydantic model.** Two
sources of truth would drift, and the drift surfaces at runtime inside the agent
loop, on an iteration you cannot reproduce.

**Tool descriptions are prompt engineering, not documentation.** They are the
only thing the model reads when choosing a tool, and they are re-sent on every
single call — about 720 tokens for these three. Tool-selection accuracy is a
phase 5 metric; a vague description is the usual reason it is bad.

**Tool results have a hard output budget, and truncation is visible.** A result
is not paid for once: it joins the history and is resent every subsequent turn.
Told it is seeing 20 of 143 matches, the model narrows its search; left to
assume it saw everything, it answers confidently from a fifth of the evidence.

**Paths are confined to the clone, inside the tool.** The path comes from a
model that will, from phase 4, have been reading repository text — issues, code
comments — any of which can carry instructions aimed at it. Resolution happens
before the containment check so that `..` and symlinks are both caught.

**`search_code` is regex, not semantic — deliberately.** Phase 4 adds semantic
retrieval as a *second* tool rather than a replacement, so the agent chooses and
phase 5 can measure whether embeddings actually beat grep. Plenty of RAG systems
would have been better off as ripgrep and nobody checked.

**A provider that breaks its own protocol raises a distinct error.** DeepSeek
intermittently serialises a tool call into the message content and reports the
finish reason as `stop`. The check that catches it is deliberately strict — it
requires the *entire* message to be a JSON object naming an offered tool —
because a loose "does the text mention a tool name?" test would misfire
constantly on a system whose whole job is discussing source code. It is retried
automatically, and every attempt prints why: recover automatically, never
silently.

**Retrieval will be a tool, not a pipeline.** In phase 4 the agent decides when
to search, rather than every question being forced through a fixed retrieve →
stuff → answer chain. Questions like "what changed in this file recently?" do
not need a vector search, and a pipeline cannot decline to run.

**Closed issues seed the eval set.** Each one is a question a real developer
asked and a maintainer answered, which is free ground truth. The extracted
records are candidates only — phase 5 will not use a model's own summary as the
yardstick for that model without a human pass over it first.
