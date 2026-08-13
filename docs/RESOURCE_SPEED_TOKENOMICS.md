# Resource, speed, and token-economics protocol

## Claim boundary

HybridMind has an offline, bounded measurement path. It measures only the local synthetic
components that actually execute: generation of deterministic 4096-dimensional vectors, FAISS
HNSW insertion/search, BM25S build/search, NetworkX construction/traversal, serialization, and
sampled process RSS. It makes zero external calls and performs no embedding, reranking, or reader
inference.

The generated report separates three evidence classes:

- `measured_offline`: observations made during the current local process;
- `analytic_projection`: byte arithmetic derived from declared dimensions, chunking, and index
  formulas;
- `scenario_projection`: prompt-token and dollar calculations from caller-supplied usage and
  prices.

The validator rejects reports that label projections as measurements, omit raw latency/RSS/file
size evidence, produce inconsistent percentiles or totals, report provider calls, or claim model
KV-cache reduction. Synthetic self-hit rates are integrity checks, not retrieval-quality evidence.

Run the bounded default workload:

```powershell
.\.venv\Scripts\python.exe scripts\offline_resource_frontier.py `
  --output benchmarks/results/offline_resource_frontier.json
```

The benchmark defaults to 256 vectors and refuses to exceed 4,096 vectors or 512 MiB of estimated
vector working memory unless those safety bounds are changed explicitly. Even then it refuses a
working set larger than half of currently available RAM. Its p50/p95/p99 timings describe a
sequential vector + sparse + graph component sequence, not HTTP end-to-end latency.

## Capacity arithmetic

At mean chunk size `C` and source-token count `T`, the number of stored dense vectors is
`ceil(T/C)`. One 4096-dimensional float32 vector is 16,384 bytes. FAISS documents the HNSW flat
estimate as `4*d + M*2*4` bytes/vector; for `d=4096, M=32`, that is 16,640 bytes/vector. The current
`VectorIndex` additionally retains one normalized float32 copy per vector for rebuilding, so its
vector-component lower bound is 33,024 bytes/vector before Python mappings and allocators.

With the illustrative `C=256`:

| Source tokens | Vectors | Raw float32, one copy | FAISS HNSW32 estimate | Current HybridMind vector lower bound |
|---:|---:|---:|---:|---:|
| 10M | 39,063 | 0.640 GB | 0.650 GB | 1.290 GB |
| 40M | 156,250 | 2.560 GB | 2.600 GB | 5.160 GB |
| 100M | 390,625 | 6.400 GB | 6.500 GB | 12.900 GB |

These decimal byte figures exclude SQLite, source text, BM25, NetworkX, IDs, allocator overhead,
fragmentation, and build scratch space. They are lower bounds, not measured feasibility. The
current duplicated all-in-memory representation should therefore not be advertised as a validated
100M-token architecture.

The report also shows encoding-only alternatives: float16 (`2*d` bytes/vector), scalar 8-bit
(`d` bytes/vector), and a configurable product-quantization code size. It additionally reports
each encoding plus the 256-byte HNSW32 link estimate. Those values do not include codebooks,
coarse quantizers, IDs, non-HNSW graph links, or quality loss, and HybridMind does not currently
implement compressed vectors. DiskANN demonstrates that SSD-resident graph search can move the RAM frontier, but
its billion-scale SIFT result does not transfer directly to 4096-dimensional Qwen embeddings.
SPFresh establishes that global rebuilds can cause update resource and latency spikes; it motivates
future update-amplification measurements rather than establishing HybridMind's update performance.
Each generated report compares the current vector-component lower bound with host RAM and applies
a conservative half-RAM gate. Passing that gate still reports feasibility as unestablished because
the excluded components can dominate; failing it is sufficient evidence to reject that in-memory
configuration on the measured host.

## Token accounting

Prompt-source reduction means

```text
(baseline prompt source tokens - retrieved unique source tokens)
/ baseline prompt source tokens
```

The baseline and uniqueness rule must be declared. This is not transformer KV-cache reduction.
The cost scenario records embedding calls/tokens, reranker calls/pairs/tokens, reader calls and
input/output tokens separately. When any priced usage is nonzero, a live plan must specify every
rate explicitly; `unpriced` scenarios return `projected_cost_usd: null` instead of pretending that
zero means free. Self-hosted/serverless time is recorded separately as provider runtime seconds
times a caller-supplied USD/second rate, because token-only pricing cannot represent GPU billing.

## Live evaluation gate

Preflight calls can themselves consume resources. `scripts/preflight.py` therefore requires a
live-plan JSON file even for warming. The plan binds to a validated offline report by SHA-256,
requires the same host and a maximum report age, selects only the providers to check, and fixes:

- planned and maximum queries, embedding calls/tokens, reranker calls/pairs/tokens, and reader
  calls/input/output tokens;
- complete pricing and a maximum estimated USD spend;
- maximum wall time;
- maximum local measured p95/p99, build time, serialized bytes, and peak RSS;
- minimum currently available memory and free disk;
- an acknowledgement that preflight usage is included.

Validate without any provider call:

```powershell
.\.venv\Scripts\python.exe scripts\preflight.py `
  --plan docs\LIVE_EVAL_PLAN.example.json `
  --validate-only
```

After replacing the example checksum, rates, usage, and machine-specific limits, omit
`--validate-only` to check only the selected providers. A missing, malformed, stale, unpriced,
over-budget, resource-violating, or checksum-mismatched plan exits before any provider check. The
plan is a preflight admission control; the evaluation client must also enforce its counters during
the run because a preflight file cannot prevent another process from overspending.

## Primary sources

- Malkov and Yashunin, *Efficient and robust approximate nearest neighbor search using
  Hierarchical Navigable Small World graphs* (2016): https://arxiv.org/abs/1603.09320
- FAISS, *Guidelines to choose an index* (implementation memory formula):
  https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
- Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single
  Node* (NeurIPS 2019):
  https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html
- Xu et al., *SPFresh: Incremental In-Place Update for Billion-Scale Vector Search* (SOSP 2023):
  https://www.microsoft.com/en-us/research/publication/spfresh-incremental-in-place-update-for-billion-scale-vector-search/
