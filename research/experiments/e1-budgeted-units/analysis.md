# E1 analysis — EXPLORATORY (LoCoMo, inspected corpus)

Artifact: `experiments/results/offline-locomo-budgeted-evidence-e1-20260925.json` (rows kept).
Scored 1,977 / 1,986 questions (9 excluded: malformed or unresolvable evidence). 42 s CPU, 0 calls.
Harness reproduces the published raw BM25S Recall@10 exactly (0.544696) when asked for top-10;
BM25S tie order changes with k (47/1,977 top-10 sets differ), so the harness breaks ties by
chronology.

## Result

Complete-evidence coverage (all gold IDs packed), raw index text, cluster-bootstrap CI:

| budget | turn | nbr1 (±1) | next1 | cap2 | session |
|---|---|---|---|---|---|
| 512 | 0.515 | 0.597 (+0.082 [0.052,0.107]) | 0.593 | 0.458 (−0.057) | 0.008 |
| 1024 | 0.583 | 0.675 (+0.093 [0.063,0.119]) | 0.670 | 0.485 (−0.098) | 0.373 |
| 2048 | 0.641 | 0.749 (+0.108 [0.084,0.132]) | 0.724 | 0.522 (−0.119) | 0.638 |
| 4096 | 0.711 | 0.809 (+0.099 [0.069,0.128]) | 0.778 | 0.690 (−0.020) | 0.765 (+0.054) |

nbr1 at 1,024 tokens beats turn at 2,048 → a mechanism win, not "more text".

By category at 2,048 (raw): single-hop 0.736 → 0.893 (+0.157); multi-hop 0.190 → 0.208
(+0.018); temporal 0.709 → 0.741; open-domain 0.272 → 0.315; adversarial 0.771 → 0.913.

## Verdicts vs protocol

- H1a (neighbour/reply expansion): **supported** (passes the locked rule at 1,024 and 2,048;
  no answerable category below −0.02 at 2,048; multi-hop −0.007 at 1,024 is within −0.02).
- H1b (session-diverse cap): **refuted** in this form — reordering pushes high-scoring
  same-session turns behind weak cross-session turns; worse on every category incl. multi-hop.
- H1c (session units): as predicted, useless below ~1k (sessions ≈ 800–1,900 tokens), +0.054
  at 4,096 overall but −0.068 on multi-hop (budget concentrated in few sessions).
- H1d (representation): speaker prefix +0.02–0.03; adding image captions +0.03–0.04 at
  1–4k over raw for `turn`; within nbr1 the representation gain shrinks to ~+0.01.

## What it means

Local evidence (single-hop, adversarial) is recoverable cheaply by expanding hits to their
conversational neighbourhood. Cross-session aggregation (LoCoMo cat. 1) is the unsolved stage:
complete coverage 0.12 / 0.19 / 0.29 at 1k / 2k / 4k tokens with every tested unit.
Next: complementary candidate generation for aggregation questions (dense, query expansion,
derived fact/observation units as a labelled diagnostic), measured by multi-hop complete coverage
at equal budget; then answer-level reading to test whether complete coverage predicts accuracy.
