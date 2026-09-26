# E5 — Does budgeted evidence coverage become correct answers? (LongMemEval-S cleaned)

Status: **CONFIRMATORY** (answer level; untouched workload). Locked before any answer exists.
Harness: `scripts/budgeted_answer_eval.py` (official LongMemEval reader + judge prompts, verbatim).
Operator approval (2026-09-25): free reader + GLM-4.6 judge, hard cap $4 across E5+E6.

## Setup

- Dataset: `longmemeval_s_cleaned.json` (SHA-256 d6f21ea9…c3a442), all 500 questions incl. 30 `_abs`.
- Reader: Z.AI `glm-4.7-flash` (free tier), thinking disabled, official CoT prompt, T=0, 800 tokens.
- Judge: Z.AI `glm-4.6`, thinking disabled, official per-type prompts incl. abstention, T=0,
  10 tokens, label = "yes" substring. **Deviation:** official judge is gpt-4o-2024-08-06. We report
  a blinded audit of ≥100 judge decisions (author re-grades without seeing the label) and the
  disagreement rate.
- Arms: `none`, `oracle` (answer sessions), `raw|turn|4096`, `raw|prop0.7|4096` on all 500;
  `full` (whole ~110k-token haystack) on the deterministic 101-question stratified subset.
- Plans: reader `research/plans/live-e5-reader-flash-20260925.json` ($0 rates),
  judge `research/plans/live-e5-judge-glm46-20260925.json` (priced, cap $2.00). Spend is
  cumulative per plan. A 3-question prop@4k + 1-question full-context smoke run goes first.

## Hypotheses

- **H5a (coverage predicts correctness):** over the two retrieval arms, answer correctness is
  better separated by complete has_answer coverage than by any-hit (AUC and accuracy gap
  P(correct | complete) − P(correct | not complete) > P(correct | any) − P(correct | none)).
- **H5b (propagation):** prop@4k accuracy is non-inferior to turn@4k (paired Δ ≥ −0.02, question
  bootstrap), gaining on single-session types and losing on multi-session, mirroring E4.
- **H5c (budget vs full context):** with this small reader, 4k-token retrieval (≈3.7% of the
  haystack) is at least as accurate as full context on the 101-question subset; oracle ≫ both.
- **H5d (failure decomposition):** report, for each retrieval arm, the share of wrong answers
  with complete evidence (reader failure) vs incomplete evidence (retrieval failure), and the
  share of right answers without complete evidence.

## Statistics and failures

Paired question-level bootstrap (10,000) and exact McNemar for paired arms; per-type slices
descriptive. Provider failures stop a pass and are recorded; resumed passes skip `ok` rows.
Questions never answered remain in denominators as failures in the final table.

## Smoke log (appended after locking; not a result)

2026-09-25 18:14 UTC — reader smoke (prop@4k, 3 q, workers=1): q `2bf43736` answered in 3.8 s
(4,170 prompt / 202 completion tokens, reasoning_tokens 0, answer matches gold); q `e6041065`
failed with HTTP 429 after 3 retries → pass stopped (`provider_failure`), receipt written.
The GLM-4.7-Flash free tier rate-limits almost immediately: the harness needs request pacing
(e.g. ≥10–20 s spacing / patient 429 backoff) before E5 can run. Spend so far: ≈$0.00
(1 GLM-4.6 preflight call, 16 in / 4 out tokens; reader calls free).
