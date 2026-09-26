# E4 — Confirmation on untouched official LongMemEval-S (cleaned)

Status: **CONFIRMATORY** for retrieval-side coverage. Locked before the dataset is downloaded.
Harness: `scripts/offline_budgeted_evidence.py --kind longmemeval`. Zero provider calls.

## Workload

`longmemeval_s_cleaned.json` from HF `xiaowu0162/longmemeval-cleaned` (expected 277,383,467
bytes, MIT). SHA-256 recorded on download. 500 questions; the 30 `_abs` questions and any question
without a `has_answer` turn are counted and excluded from coverage (official retrieval convention),
expected 470 scored. Each question has its own haystack (~50 sessions, ~115k tokens).

## Disclosure before running

The harness was smoke-tested on the local *oracle* file (evidence sessions only, invalid as a
retrieval benchmark) at B=2,048: turn 0.830, nbr1 0.602, prop0.7 0.806. λ=0.7 had already been
frozen on LoCoMo (E3). No parameter is changed in response to the smoke test.

## Arms (all raw BM25S keys; LongMemEval has no speaker names or captions)

`turn` (baseline), `prop0.7` (primary), `nbr1`, `session`; budgets 2,048 / 4,096 / 8,192 / 16,384
rendered proxy tokens (render = session date + role + text).

## Hypotheses and predictions

- **H4a (transfer of the neighbourhood mechanism):** the E1 gain came from local, cheap
  neighbours. LongMemEval evidence is mostly in user turns and neighbours (assistant replies) are
  long. Prediction: fixed `nbr1` does **not** beat `turn` (Δ ≤ 0 at every budget).
- **H4b (propagation is safe):** `prop0.7` is non-inferior to `turn`: paired complete-coverage
  Δ ≥ −0.01 at B=4,096 (lower 95% CI bound ≥ −0.03), and ≥ `nbr1` at every budget.
- **H4c (units):** whole sessions win only at the largest budgets.

## Statistics

Questions have separate haystacks, so paired question-level bootstrap (4,000 resamples) is the
primary interval; question_type slices reported descriptively. All arms and budgets reported.

## Reproduction target in the same run

Official LongMemEval retrieval metric (user-turn index, session-level recall_any@k / recall_all@k,
k ∈ {5, 10}) for BM25 via the official repository code, compared with the paper's reported BM25
numbers, as a faithful-reproduction check of the workload and metric.
