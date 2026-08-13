# HybridMind resource characterization

The only current validated performance artifact is
`benchmarks/results/offline_resource_frontier.json`. It is a bounded synthetic
component measurement, not an API benchmark, provider benchmark, retrieval
quality result, or 10M–100M feasibility demonstration.

The latest default offline run used 256 deterministic 4096-dimensional vectors,
32 queries, and made zero network/provider calls. Its sequential local
vector+BM25S+graph component latency was approximately p50 0.53 ms, p95 0.68 ms,
and p99 0.80 ms on the recorded host. Serialized derived components occupied
about 8.78 MB and observed process peak RSS was about 117 MB. Raw samples,
versions, host details, build times, and replay hashes are in the JSON artifact.

These numbers exclude HTTP, remote query embedding, cross-encoder reranking,
answer generation, concurrency, and a real corpus. Synthetic self-hit@1 is only
an index integrity check.

At 256 source tokens per chunk, the current duplicated float32+HNSW vector
representation has analytic lower bounds of roughly 1.29 GB at 10M source
tokens, 5.16 GB at 40M, and 12.9 GB at 100M, before SQLite, text, BM25,
NetworkX, mappings, allocator overhead, or build scratch. Therefore the current
all-in-memory layout is not a validated 100M-token deployment architecture.

Regenerate the bounded report with:

```powershell
.\.venv\Scripts\python.exe scripts\offline_resource_frontier.py `
  --output benchmarks\results\offline_resource_frontier.json
```

See `docs/RESOURCE_SPEED_TOKENOMICS.md` for formulas, validator guarantees,
capacity alternatives, and the default-deny live evaluation gate.
