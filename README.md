# Coding Agent

[![CI](https://github.com/kevalya2003/coding-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/kevalya2003/coding-agent/actions/workflows/ci.yml)

A model-agnostic agent that investigates a software issue, edits a local repository,
runs its tests, and submits a patch only after verification.

This project is intentionally more than an LLM wrapper. It includes:

- a guarded tool layer that cannot read or write outside the target repository;
- an owner-defined, allow-listed verification command without a shell;
- mandatory test verification before submission;
- step and token budgets, plus an optional provider-priced dollar budget;
- repeated-action loop detection;
- bounded recovery from malformed tool calls the provider rejects outright;
- JSONL trajectories for replay and failure analysis; and
- a benchmark runner for fixed, reproducible task sets.

## Architecture

```text
issue + repository
        |
        v
  agent control loop <------ OpenAI-compatible model
        |
        +----> search / read / edit / diff
        +----> fixed, allow-listed verification command
        |
        v
 budget + loop guards ---> JSONL trajectory ---> benchmark report
```

The model never receives an unrestricted terminal. Its only process tool runs the fixed
verification command chosen by the repository owner. Arguments are validated against
configured prefixes and executed with `shell=False`. A command that does not match the
allow list is rejected before the first model call, so a mistyped `--test-command` costs
nothing.

## Quick start

Python 3.8+ is supported.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Add an API key to `.env`. Any provider exposing an OpenAI-compatible chat-completions
endpoint can be used.

Run the agent against a local repository:

```powershell
coding-agent run `
  --repo C:\path\to\repository `
  --issue "The parser crashes when the input ends with a trailing comma." `
  --test-command "python -m pytest -q"
```

Run the included unit tests:

```powershell
python -m pytest -q
```

## Reproducible evaluation

Copy `benchmarks/tasks.example.json` to `benchmarks/tasks.json` and replace the example
with pinned public repositories and commits. Each task declares:

- repository URL;
- immutable base commit;
- issue text;
- verification command; and
- optional setup command.

Task files are executable configuration: setup and test commands run on the host. Author
them yourself and use Docker or a disposable VM for repositories you do not trust.

Then run:

```powershell
coding-agent evaluate `
  --tasks benchmarks/tasks.json `
  --output runs/baseline
```

The evaluator creates a clean checkout for every task and writes:

- `trajectories/*.jsonl` — every model response and tool result;
- `predictions.jsonl` — generated patches and task outcomes; and
- `summary.json` — success rate, token use, cost, and failure categories.

Keep the task file fixed across experiments. Change one feature at a time to produce an
honest ablation study. The local report calls this a **submission rate**: passing the
configured command is not proof that hidden benchmark tests pass.

### Offline seeded suite

Real repositories usually pass their own tests at any given commit, so they cannot be
used as tasks without separate fail-to-pass tests. For cheap iteration there is a
generated suite of eight small repositories, each holding one deliberate defect and a
test that catches it:

```powershell
python benchmarks/build_local_tasks.py
coding-agent evaluate --tasks benchmarks/tasks.local.json --output runs/seeded-v1
```

The generator refuses to emit a task whose suite already passes, so the set cannot
silently become meaningless. These are seeded defects, not real-world issues. Report
them as a seeded-bug suite and never as SWE-bench.

## SWE-bench

This repository produces standard unified diffs, so its patches can be evaluated with
the official SWE-bench harness. Do not claim a SWE-bench score until the official hidden
tests have been run. A small fixed subset is recommended while developing to control
cost.

## Safety and cost controls

Defaults in `configs/baseline.yaml` cap each task at 20 model steps, 200,000 input
tokens, and 20,000 output tokens. The US$1.00 ceiling becomes active only after you enter
your provider's current input/output prices in the configuration; the included zero
values mean “price unknown,” not “free.” Always set a provider-side account limit too.

The agent operates on a disposable checkout during evaluation, but that is not an OS
sandbox: repository test code still runs on the host. Use Docker or a disposable VM for
untrusted code. When using `run` directly, point it at a repository where uncommitted
edits are acceptable.

## Reporting results

Report measurements, never placeholders. A complete report contains:

1. baseline and improved submission rates on the same pinned task set;
2. an ablation for each control-loop change;
3. a failure taxonomy derived from reviewed trajectories;
4. task success against inference cost; and
5. two readable trajectories: one success and one instructive failure.

## Development

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

The suite covers the workspace boundary, patch generation against a real `git apply`,
the budget accounting, configuration loading, and the control loop driven by a scripted
model, so none of it needs an API key or network access.

