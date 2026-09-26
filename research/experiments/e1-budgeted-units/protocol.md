# E1 — Retrieval unit and representation at equal rendered-token budget (LoCoMo)

Status: **EXPLORATORY** (LoCoMo's ten conversations have been inspected by prior work).
Locked before running. Harness: `scripts/offline_budgeted_evidence.py`. Zero provider calls.

## Failure being targeted

The valid raw BM25S baseline reaches 0.5447 Recall@10 (reproduced exactly: 0.544696, n=1,977).
Prior selector work showed candidate generation is the larger gap. Structural probe
(`tmp/research/locality_probe2.py`): category-4 single-hop multi-evidence is local (83% within ±2
turns of one session) while category-1 "multi-hop" is cross-session aggregation (3% same
session; 38% span ≥3 sessions). These are different failures and may need different mechanisms.

## Hypotheses and predictions

- **H1a neighbour/reply expansion** (small-to-big / sentence-window, established): gold evidence
  is often the reply to a lexically matching turn. Prediction: `next1` or `nbr1` beats `turn` in
  complete-evidence coverage at equal budget B∈{1024, 2048} overall, driven by single-hop and
  temporal; ~0 on multi-hop.
- **H1b session-diverse cap** (MMR-style diversity, established): multi-hop needs one fact per
  session. Prediction: `cap2` beats `turn` on multi-hop complete coverage at B∈{1024, 2048}.
- **H1c session units** (LongMemEval session keys): whole sessions lose at small B and may win
  only at B=4096.
- **H1d representation**: speaker prefix (+caption) raises coverage over raw at every B.

## Comparisons and controls

- Baseline: `raw|turn` at each budget. The budget ladder is the "just retrieve more text"
  control: a strategy is only a better mechanism if it beats `turn` at the *same* budget, and is
  compared against `turn` at 2× budget as the cost-matched upper reference.
- Everything is charged for the full rendered context line (date + speaker + text + caption), with
  a declared regex token proxy `\w+|[^\w\s]`.
- Adversarial category is reported separately and excluded from "answerable" summaries.

## Metrics and decision rule

Primary: complete-evidence coverage (all gold turn IDs packed). Secondary: fractional recall,
any-hit, catastrophic miss, mean tokens used. Paired deltas vs baseline via conversation-cluster
bootstrap (10 clusters, 4,000 resamples) — wide by construction; treat as descriptive.

A strategy advances to confirmation (untouched LongMemEval-S) only if at B=1024 and B=2048 its
paired complete-coverage delta is ≥ +0.02 with cluster-CI lower bound > 0, and no answerable
category shows a point delta below −0.02. All attempted variants are reported.

## Budget

Local CPU only, ~1–3 min wall time. No models, no network.
