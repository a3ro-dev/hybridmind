# E7 — Production native-4096 dense channel at equal budget (LoCoMo)

Status: **EXPLORATORY** (LoCoMo inspected). Locked before any vector exists. Operator approved
a $3 cap (2026-09-25). Plan: `research/plans/live-embed-locomo-4096-20260925.json`
(projected $1.33, ceiling $2.77); embedding script `scripts/embed_locomo_4096.py`.

## Failure being targeted

Multi-hop (cross-session aggregation) misses are dominated by third-person/first-person
vocabulary gaps (E1/E2); 80.5% of missed gold turns rank below 25 under BM25S.

## Arms (all packed by the E1 harness at equal rendered-token budgets 1k/2k/4k)

- `dense_bare`: cosine between bare question and `spk_cap` turn text (what production sends).
- `dense_inst`: question wrapped in Qwen3-Embedding's documented query instruction
  (`Instruct: …\nQuery:…`); documents unchanged.
- `hybrid`: RRF (k=60, equal weights) of BM25S `spk_cap` and `dense_inst` rankings.
- each with `turn` and `prop0.7` packing (λ frozen from E3).

## Hypotheses

- **H7a** query instruction: `dense_inst` ≥ `dense_bare` + 0.02 complete coverage at 2k.
- **H7b** complementarity: `hybrid|turn` beats `spk_cap|turn` (BM25S) at 2k overall, and by
  ≥ +0.05 on multi-hop complete coverage.
- **H7c** stacking: `hybrid|prop0.7` ≥ `hybrid|turn` (propagation still helps with a dense channel).

## Statistics

Conversation-cluster bootstrap (10 clusters), descriptive; all arms reported. The live run is
recorded by a receipt (calls, wall seconds, cache SHA-256); failure → receipt, no retries
beyond the engine's transient retry policy, no fallback model.
