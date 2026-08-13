# HybridMind Evaluation Evidence Report

## Status

This report lists only results reproducible from artifacts currently present in
the repository workspace. Historical headline accuracy values are excluded when
their per-question answer ledger or source run is unavailable.

**Integrity notice (2026-08-13):** the checked-in `ledger_locomo_7d21dbb5fc67.jsonl`,
`kv_reduction_locomo_checkpoint.json`, and `kv_reduction_locomo_lexical_rrf.json`
are legacy audit artifacts, not valid current benchmark results. They predate
conversation-scoped stable evidence IDs, immutable run manifests, exact
evidence-ID gates, and live resource/spend budgets. The JSON files are retained
unchanged for historical auditability and must not be relabelled as reruns.
`benchmarks/kv_reduction_eval.py` now rejects those legacy sources rather than
reproducing their old pass/fail statements.

## Current Valid Offline Sparse Baseline

Artifact: `offline_locomo_bm25s.json`

This is a real, zero-provider LoCoMo retrieval baseline, not an end-to-end
HybridMind result. It creates a separate BM25S index for each of the 10
conversations and scores raw turn text against question text using exact,
conversation-qualified evidence IDs.

- Dataset SHA-256, host/runtime, raw metric confidence intervals, exclusions,
  and latency are recorded in the artifact.
- 1,977 evidence-bearing questions were eligible. Two questions with malformed
  annotation fragments and three whose cited turn is absent from the supplied
  conversation were excluded and individually recorded rather than repaired or
  silently scored wrong.
- At k=10, exact evidence recall is **54.47%** (bootstrap 95% CI
  52.42–56.62%), any-hit is **59.48%**, all-hit is **50.38%**, and the declared
  regex-token prompt-source reduction proxy is **98.32%**.
- At k=100, recall rises to **77.38%** while source reduction falls to
  **82.69%**.
- Local sparse query latency is p50 **0.36 ms**, p95 **0.68 ms**, and p99
  **1.03 ms** on the recorded host. This excludes API, embedding, graph,
  reranking, reader inference, and concurrency.

The preregistered exact-evidence gate is at least 95%. This baseline fails it by
a wide margin. It therefore supports neither the 70–80% context-substitution
target nor a KV-cache replacement claim.

Reproduce without network calls:

```powershell
.\.venv\Scripts\python.exe scripts\offline_locomo_sparse_baseline.py
```

## Deprecated Historical LoCoMo Retrieval Ledger

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

## Deprecated Historical Context-Reduction Frontier

Artifact: `kv_reduction_locomo_checkpoint.json`

The source MemoryBench checkpoint has 755 completed searches. At `k=10`, mean
memory-context token reduction is 98.93%, answer-overlap proxy hit is 11.97%,
and exact annotated-source recall is 15.71%. At `k=100`, exact source recall
reaches only 55.78% while context reduction falls to 88.41%.

The old hypothesis is invalid as a KV-cache claim. It used an answer-overlap
proxy for the quality gate and inferred proportional KV reduction from token
counts without measuring model cache allocation, eviction, memory, or cost.

## Deprecated Historical Query-Local Lexical Reranking

Artifact: `kv_reduction_locomo_lexical_rrf.json`

On the same 755 completed searches, a bounded 500-candidate 50/50 fusion of the
existing rank and query-local lexical rank improves exact annotated-source
Recall@10 from 15.71% to 33.77%. The paired improvement is 18.05 percentage
points with a bootstrap 95% CI of 15.27-20.85 points. The answer-overlap proxy
rises from 11.97% to 21.01%.

The offline rerank costs 59.31 ms mean and 73.63 ms p95. These measurements
exclude live candidate generation and cross-encoder inference.

The paired exact-source text-containment delta remains a useful retrospective
observation, but it is not a current ablation result or independent
confirmation: the same partial checkpoint informed the hypothesis, only 755 of
1,986 searches are complete, candidates lack stable evidence-ID metadata, and
no checkpoint result contains a non-null cross-encoder score.

## Current Bounded Resource Measurement

Artifact: `offline_resource_frontier.json`

The refreshed zero-network run built 256 deterministic 4096-dimensional
synthetic vectors and issued 32 sequential vector + sparse + graph component
queries. Component-sequence latency was p95 **0.962 ms** and p99 **1.088 ms**;
serialized components occupied **8,781,050 bytes** and observed process peak RSS
was **117,891,072 bytes**. Embedding inference, HTTP, reranking, reader inference,
real-corpus quality, and concurrency are excluded.

Its capacity rows are arithmetic projections. At 256 source tokens per vector,
the current duplicated float32 + FAISS vector-component lower bounds are 1.29 GB
for 10M source tokens, 5.16 GB for 40M, and 12.9 GB for 100M, before text, SQLite,
BM25, NetworkX, mappings, allocation, or build scratch. They do not establish
large-corpus feasibility.

## Current Live TEI Canary

Artifacts: `live_tei_canary_plan.json`, `live_tei_canary_result.json`

After the full offline suite passed, a checksum-bound plan admitted exactly one
TEI embedding request, eight accounted input tokens, 60 seconds runtime/wall,
zero LLM/reranker calls, and a conservative modeled maximum spend of **$0.17**.
The request ended in a `ReadTimeout`; no retry or fallback occurred and no
evaluation was admitted. Therefore live 4096-dimensional response correctness
and latency remain unverified.

## Unverified Claims

- No complete answer-stage LoCoMo ledger is available, so LoCoMo accuracy is not
  currently verified.
- No LongMemEval or MuSiQue per-question ledger is present, so their historical
  accuracy, Hit@k, and MRR values are not currently verified.
- The lexical reranking artifact provides a paired retrospective estimate, but
  no independent held-out A/B answer ledger is available.

## Reproduction

The commands below document how the legacy artifacts were produced. The
current evaluator intentionally fails closed on them. A new run must first
re-ingest LoCoMo with stable evidence IDs and produce a v2 immutable ledger.

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
