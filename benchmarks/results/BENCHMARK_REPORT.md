# HybridMind Evaluation Evidence Report

## Status

This report lists only results reproducible from artifacts currently present in
the repository workspace. Historical headline accuracy values are excluded when
their per-question answer ledger or source run is unavailable.

## Verified LoCoMo Retrieval Ledger

Artifact: `ledger_locomo_7d21dbb5fc67.jsonl`

- 1,985 retrieval-only rows; no answer-stage accuracy is present.
- `eval_stats.py` operates on 1,973 unique question hashes because duplicate
  question text collides in the historical ledger schema.
- Hit@1: 5.37% (95% CI 4.41%-6.39%)
- Hit@3: 12.67% (95% CI 11.25%-14.14%)
- Hit@5: 16.57% (95% CI 14.95%-18.25%)
- Hit@10: 21.69% (95% CI 19.87%-23.52%)
- Hit@25: 31.88% (95% CI 29.85%-33.96%)

These are answer-text-overlap retrieval proxies, not exact evidence recall and
not downstream QA accuracy.

## Verified KV-Reduction Frontier

Artifact: `kv_reduction_locomo_checkpoint.json`

The source MemoryBench checkpoint has 755 completed searches. At `k=10`, mean
memory-context token reduction is 98.93%, answer-overlap proxy hit is 11.97%,
and exact annotated-source recall is 15.71%. At `k=100`, exact source recall
reaches only 55.78% while context reduction falls to 88.41%.

The initial hypothesis that HybridMind can retain at least 80% answer-bearing
context at `k=10` while reducing proportional KV working set by at least 90%
is rejected.

## Verified Query-Local Lexical Reranking

Artifact: `kv_reduction_locomo_lexical_rrf.json`

On the same 755 completed searches, a bounded 500-candidate 50/50 fusion of the
existing rank and query-local lexical rank improves exact annotated-source
Recall@10 from 15.71% to 33.77%. The paired improvement is 18.05 percentage
points with a bootstrap 95% CI of 15.27-20.85 points. The answer-overlap proxy
rises from 11.97% to 21.01%.

The offline rerank costs 59.31 ms mean and 73.63 ms p95. These measurements
exclude live candidate generation and cross-encoder inference.

This passes the predeclared 5-point effect threshold on the reused checkpoint,
but it is not an independent confirmation: the same checkpoint informed the
hypothesis, only 755 of 1,986 searches are complete, and no checkpoint result
contains a non-null cross-encoder score.

## Unverified Claims

- No complete answer-stage LoCoMo ledger is available, so LoCoMo accuracy is not
  currently verified.
- No LongMemEval or MuSiQue per-question ledger is present, so their historical
  accuracy, Hit@k, and MRR values are not currently verified.
- The lexical reranking artifact provides a paired retrospective estimate, but
  no independent held-out A/B answer ledger is available.

## Reproduction

```powershell
.\.venv\Scripts\python.exe eval_stats.py ci `
  benchmarks\results\ledger_locomo_7d21dbb5fc67.jsonl

.\.venv\Scripts\python.exe benchmarks\kv_reduction_eval.py `
  --checkpoint memorybench\data\runs\hybridmind-locomo-fixed-20260726\checkpoint.json `
  --k-values 1,3,5,10,25,50,100 `
  --output benchmarks\results\kv_reduction_locomo_checkpoint.json

.\.venv\Scripts\python.exe benchmarks\kv_reduction_eval.py `
  --checkpoint memorybench\data\runs\hybridmind-locomo-fixed-20260726\checkpoint.json `
  --checkpoint-ranking local-lexical-rrf `
  --local-lexical-pool-size 500 `
  --local-lexical-weight 0.5 `
  --k-values 1,3,5,10,25,50,100 `
  --output benchmarks\results\kv_reduction_locomo_lexical_rrf.json
```
