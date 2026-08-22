# HybridMind pre-change baseline

Date: 2026-08-14 (Asia/Calcutta)  
Commit: `3422f226d5adc650802e5aeb87a0302b0765635f`  
Initial branch/worktree: `main`, clean, tracking `origin/main`  
Execution class: offline; zero embedding, reranker, reader, judge, or other external provider calls

## What this baseline establishes

The checked-out system is a local SQLite-authoritative retrieval service with
FAISS HNSW, BM25S, and a NetworkX `MultiDiGraph`. The fresh baseline can
measure the sparse retriever and bounded local index mechanics without a live
embedding endpoint. It cannot establish dense, graph, hybrid, cross-encoder,
answer-generation, long-context, or state-of-the-art claims.

The aggregate machine record is
`experiments/baselines/20260814-3422f226.json`. The two raw measured artifacts
are:

- `experiments/baselines/20260814-3422f226-locomo-bm25s.json` (SHA-256
  `ad0bd46990658e5cb143132049febd91627fd3bebf178ec297fe14065e46b7e0`)
- `experiments/baselines/20260814-3422f226-resource-frontier.json` (SHA-256
  `9f6a9ac2151f68d69659be6dadfd4df3d95148c6fb7c8065abff6fcb7bfa8002`)

## Verification baseline

| Check | Result | Evidence |
|---|---:|---|
| Primary Python suite | PASS | 279 passed, 3 skipped in 16.56 s; first fresh process wall time 23.71 s |
| Legacy `verify/` suite | FAIL | 12 passed, 4 failed in 1.71 s |
| Python compilation | PASS | forced `compileall` over application, evaluation, scripts, and tests |
| Python dependency consistency | PASS | `.venv` `pip check` reports no broken requirements |
| Ruff | NOT RUN | not installed in the required `.venv` |
| MemoryBench test | PASS | 1 passed in 1.520 s |
| MemoryBench format check | FAIL | one HybridMind provider file is unformatted; configured `ui` globs match no files |
| MemoryBench TypeScript check | FAIL | `bun-types` is requested by `tsconfig` but unavailable |

The four legacy verification failures are not interchangeable with production
regressions. The verification fixture disables auto-edges and fact extraction,
and supplies no policy-allowed LLM summary provider, while the stale tests
expect auto-edge creation, fact extraction, and consolidation to execute. They
must be repaired to state their dependencies explicitly rather than weakening
the fail-closed production defaults.

`scripts/multi_domain_eval.py --help` is itself unsafe as a diagnostic: the
script has no argument parser and immediately attempted a localhost HTTP
connection. The connection was refused and no provider call was made.

## Fresh LoCoMo sparse baseline

Dataset: `memorybench/data/benchmarks/locomo/locomo10.json`, SHA-256
`79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
The run indexes raw turns independently per conversation and ranks only with
BM25S against the question text. Relevance is exact canonical evidence ID.

| Metric | Result |
|---|---:|
| Samples / turns | 10 / 5,882 |
| Questions total / evidence-bearing measured | 1,986 / 1,977 |
| Invalid annotation questions / unresolved evidence questions | 2 / 3 |
| MRR, first exact evidence | 0.3855 |
| Exact-evidence Recall@1 | 0.2566 |
| Exact-evidence Recall@5 | 0.4639 |
| Exact-evidence Recall@10 | 0.5447 |
| Exact-evidence Recall@25 | 0.6399 |
| Exact-evidence Recall@50 | 0.7082 |
| Exact-evidence Recall@100 | 0.7738 |
| Any exact evidence hit@10 | 0.5948 |
| All exact evidence hit@10 | 0.5038 |
| Prompt-source reduction proxy@10 | 0.9832 |
| Query latency mean / p50 / p95 / p99 | 0.337 / 0.256 / 0.338 / 0.448 ms |

These are real dataset retrieval measurements but not end-to-end HybridMind
metrics. They exclude dense embeddings, graph traversal, reranking,
decomposition, salience, answer generation, and judging. The token reduction
uses a regex proxy and says nothing about realized transformer KV allocation.

## Fresh bounded resource baseline

The synthetic run uses 512 deterministic 4096-dimensional vectors, 512 BM25S
documents, a 512-node graph, 64 queries, and `top_k=10`.

| Metric | Result |
|---|---:|
| Total component build time | 0.133 s |
| Serialized vector + sparse + graph bytes | 17,557,781 |
| Peak RSS / RSS increase | 125,964,288 / 23,740,416 bytes |
| Serial component p50 / p95 / p99 | 0.732 / 0.954 / 1.147 ms |
| Deterministic replay | identical ordered result hash |

The analytic current-vector-component lower bounds at 256 source tokens per
chunk are 1.29 GB at 10M tokens, 5.16 GB at 40M tokens, and 12.90 GB at 100M
tokens. These exclude text, SQLite, BM25, graph objects, allocators, and build
scratch space. On this 16.87 GB host, the 100M lower bound alone exceeds the
half-RAM gate. Feasibility is therefore not established.

## Live and end-to-end benchmark status

No fresh live plan was created and no live preflight was performed. Dense,
vector-sparse, graph, hybrid, cross-encoder, decomposition, and reader runs are
therefore intentionally absent. The LongMemEval dataset is present but its
evaluator requires the live API. The MuSiQue dataset is absent. Historical
MemoryBench checkpoint artifacts are not reused as current evidence because
they predate the repository's current evidence-ID, manifest, server-attestation,
and budget contracts.

## Environment

- Windows 11 10.0.26100; Acer Nitro ANV15-51
- Intel Core i5-13420H, 12 logical processors, 16,869,351,424 bytes RAM
- NVIDIA RTX 4050 Laptop GPU, 6,141 MiB, driver 595.97
- Python 3.13.5 from `.venv`
- Torch 2.11.0 CPU build; CUDA unavailable to this Python environment
- NumPy 2.4.3, FAISS CPU 1.13.2, BM25S 0.3.9, NetworkX 3.6.1
- Node 24.5.0 and pnpm 11.1.3

## Baseline verdict

The production core is substantially hardened, but the only fresh quality
number currently available is a sparse-only LoCoMo baseline. It has useful
lexical recall and very low local latency, yet Recall@10 of 0.5447 is far below
the preregistered 0.95 exact-evidence gate. The present evidence does not
support calling HybridMind SOTA, near-SOTA, or a validated long-context/KV
replacement.
