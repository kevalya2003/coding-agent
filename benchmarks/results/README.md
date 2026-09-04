# Results

## seeded-v1

First measured run of the agent, on the eight seeded-bug tasks produced by
`benchmarks/build_local_tasks.py`.

| | |
|---|---|
| Date | 2026-09-04 |
| Model | `openai/gpt-oss-120b` (Groq, OpenAI-compatible endpoint) |
| Config | `configs/baseline.yaml` — 20 steps, 200k input tokens, 20k output tokens, temperature 0 |
| Verification | `python -m pytest -q` |
| Submission rate | **6 / 8 (75%)** |
| Tokens | 148,197 in, 11,380 out |
| Cost | unpriced; the config carries 0.0 per million, which means price unknown |

| Task | Status | Steps | Tokens |
|---|---|---|---|
| slugify-trailing-separators | submitted | 9 | 9,343 |
| roman-subtractive-notation | budget_exhausted | 20 | 60,491 |
| chunked-drops-final-partial | submitted | 10 | 11,436 |
| binary-search-misses-last | submitted | 9 | 8,588 |
| duration-ignores-components | budget_exhausted | 20 | 41,175 |
| csv-quoted-comma | submitted | 8 | 8,369 |
| lru-evicts-wrong-entry | submitted | 8 | 7,579 |
| wrap-ignores-separator-width | submitted | 12 | 12,596 |

Submission rate means the agent produced a patch after its verification command exited
zero. These are seeded defects in generated repositories, not real-world issues, and this
is not a SWE-bench score.

### Failure taxonomy

Both failures share one cause, and it is not model capability: **neither run ever called
`run_tests`.**

| | roman-subtractive-notation | duration-ignores-components |
|---|---|---|
| `search_code` | 1 | 1 |
| `read_file` | 11 | 12 |
| `replace_in_file` | 7 | 5 |
| `run_tests` | **0** | **0** |
| protocol errors recovered | 1 | 2 |

Each run spent its entire step budget alternating between reading a file and editing it,
never once checking whether the edit worked, until `max_steps` ended the task. Both are
also the two tasks whose fix requires several coordinated edits rather than one line,
which is what pulled the model into an edit-reread cycle it never broke out of.

Two existing guards failed to catch this:

- the repeated-action detector compares tool-call signatures, and every edit carried
  different `old_text`, so the churn never looked repetitive; and
- `require_tests_before_submit` only refuses a `submit` that is not yet verified. It has
  nothing to say about a run that never tries to submit at all.

`SYSTEM_PROMPT` does state that `run_tests` must pass after the final edit, so the
instruction is present and simply not followed under pressure. The candidate fix is a
loop-level nudge rather than more prompt text: after some number of consecutive edits
with no intervening verification, inject a message telling the model to run the tests.
That is a single-variable change and the submission rate above is the baseline to
measure it against.

### Artefacts

- `seeded-v1/summary.json` — aggregate counts
- `seeded-v1/predictions.jsonl` — final patch and outcome per task
- `seeded-v1/*.jsonl` — full trajectory for every task, including both failures
