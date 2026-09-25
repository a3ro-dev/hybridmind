# E4 analysis — CONFIRMATORY (official LongMemEval-S cleaned; locked in af85e84)

Dataset: `longmemeval_s_cleaned.json`, 277,383,467 bytes, SHA-256 d6f21ea9…c3a442 (= HF LFS oid).
Artifact: `experiments/results/offline-lme-s-cleaned-budgeted-evidence-e4-20260925.json`.
470 of 500 scored (30 `_abs` excluded); 68 s CPU; 0 calls. Paired question-level bootstrap.

## Complete has_answer coverage (raw BM25S keys; render = date + role + text)

| budget | turn | prop0.7 | nbr1 | session |
|---|---|---|---|---|
| 2,048 | 0.734 | 0.717 (−0.017 [−0.040, +0.004]) | 0.526 (−0.209) | 0.115 |
| 4,096 | 0.791 | 0.798 (+0.006 [−0.015, +0.026]) | 0.709 (−0.083) | 0.287 |
| 8,192 | 0.845 | 0.849 (+0.004) | 0.830 (−0.015) | 0.711 |
| 16,384 | 0.889 | 0.898 (+0.009 [−0.011, +0.028]) | 0.891 (+0.002) | 0.866 (−0.023) |

## Verdicts

- H4a (fixed window does not transfer): **supported** — nbr1 never beats turn (tie at 16k,
  +0.002 with CI across 0; large losses at 2–4k). Neighbours are long assistant replies.
- H4b (propagation is safe): **supported** — Δ +0.006 at 4k, CI lower bound −0.015 ≥ −0.03;
  prop ≥ nbr1 at every budget.
- H4c (sessions win at large budgets): **refuted** — sessions never win.

## Cross-dataset regularity (4k, by question type)

prop helps local types (single-session-assistant 0.946→1.000, single-session-user
0.969→1.000, preference 0.500→0.600) and slightly costs dispersed types (multi-session
0.628→0.612, temporal 0.780→0.764, knowledge-update 0.931→0.917) — the same depth-vs-breadth
split as LoCoMo single-hop vs multi-hop.

## Official retrieval reproduction (same run day; official code vendored verbatim)

`scripts/reproduce_longmemeval_retrieval.py` → flat-BM25, user-turn index. Official averaging
scores **419** questions (30 abstention + **51 with no user-side has_answer turn** skipped), not
the 470 quoted by third parties. Session level: recall_any@5 0.888, recall_all@5 0.742,
recall_any@10 0.926, recall_all@10 0.823, recall_any@50 **1.000** (≈50 sessions per haystack).
Turn level: recall_any@10 0.804, recall_all@10 0.592. → Session recall_any is saturated by
plain BM25; "100%" session hit rates at 50–60 retrieved items are trivial by construction.
