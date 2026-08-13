# HybridMind adversarial audit and remediation record

Date: 2026-08-13
Scope: repository code, tests, offline LoCoMo corpus, bounded local resource run,
and one budget-gated RunPod TEI canary
Evidence convention: **FACT** is directly established; **INFERENCE** follows
from code/runtime evidence; **UNPROVEN** means the repository cannot establish
the claim.

This document distinguishes the attacked baseline from the repaired working
tree. A fixed defect remains part of the audit history; it is not silently
rewritten as if it never existed.

## 1. Executive verdict

**Baseline verdict:** a conventional local RAG/database prototype with real
dense, sparse, and graph components, surrounded by partially functional memory
research ideas and documentation that materially exceeded the evidence. Several
benchmark controls were invalid, the cross-encoder was never invoked by the
advertised evaluators, persistence could fabricate success, and an unauthenticated
network deployment was possible. Classification: **over-engineered prototype +
partially implemented research ideas + benchmark theater**.

**Current verdict:** a substantially hardened engineering prototype. The core
storage and retrieval paths are real; strict vector validation, exact controlled
retrieval modes, typed graph semantics, failure receipts, safe portable
snapshots, provider policy, API security, and process mutation coordination now
have behavioral tests. It is still not a research-grade memory system: graph
gain is unmeasured, historical graph query time is not wired end to end,
`memory_kind` and node confidence do not affect ranking, consolidation is lossy
derived summarization rather than Observer/Reflector, and no valid hybrid held-out
gain exists. The single live 4096-d TEI canary timed out.

## 2. Repository architecture

| Subsystem | Executable path | Current reality |
|---|---|---|
| REST/API | `main.py`, `api/*.py` | FastAPI service with auth/host/CORS/rate controls and coordinated mutations. |
| Authoritative store | `storage/sqlite_store.py` | SQLite nodes, edges, chunks, bitemporal versions, exact 4096-d blobs. |
| Dense retrieval | `engine/embedding.py` → `storage/vector_index.py` → `engine/vector_search.py` | Remote 4096-d embedding plus FAISS HNSW; no permitted lower-dimensional fallback. |
| Sparse retrieval | `storage/bm25_index.py` | BM25S default; independently reachable raw-BM25 controlled mode. |
| Graph | `storage/graph_index.py`, `engine/graph_search.py` | NetworkX `MultiDiGraph`, typed/directed/confidence/validity-aware primitives. |
| Fusion/ranking | `engine/hybrid_ranker.py`, `engine/fusion.py` | RRF/linear/MLP options; hybrid graph expansion usually starts from dense/sparse candidates. |
| Reranking | `engine/reranker.py` | Optional mxbai cross-encoder with a validated exact pool cap. |
| Fact ingestion | `main.py:1211`, `engine/fact_extractor.py` | Policy-gated extraction, deterministic IDs, strict dates, conservative slot-conflict heuristic. |
| Temporal state | `storage/sqlite_store.py:187`, `engine/temporal.py`, `storage/graph_index.py:463` | Valid/assertion history and graph-time primitives; historical query wiring is incomplete. |
| Salience | `engine/salience.py:19` | Bounded recency/access/degree multiplier; real but conventional, not ACT-R. |
| Routing | `engine/query_router.py:26` | Regex query classification and routed omitted weights. |
| Decomposition | `engine/query_decomposition.py:46` | Optional LLM subquestions with entity/temporal guards; benefit unmeasured. |
| Consolidation | `engine/consolidation.py:113` | Lossy derived summary retained alongside exact sources and provenance edges. |
| Persistence | `storage/mindfile.py:172` | SQLite-derived, checksum-verified, pickle-free portable archive; derived indexes rebuild. |
| Evaluation | `eval_*`, `eval_ledger.py`, `scripts/ablation_matrix.py` | Exact evidence IDs and immutable artifacts; some benchmark protocols remain incomplete. |
| Offline evidence | `scripts/offline_locomo_sparse_baseline.py`, `scripts/offline_resource_frontier.py` | Real zero-provider sparse baseline and bounded synthetic resource measurement. |

If the README were deleted, the code would reveal a local SQLite + FAISS + BM25S
+ NetworkX retrieval service with optional LLM extraction/decomposition/summary
calls. It would not reveal a transformer-integrated memory architecture or a
validated replacement for model KV cache.

## 3. Biggest LARP findings

1. **FACT, baseline, CRITICAL — cross-encoder theater.** Evaluators requested
   `top_k=max(top_k, rerank_pool, 25)`, while the ranker reranked only when the
   pool exceeded `top_k`; the cross-encoder never ran. Current evaluators and
   `engine/hybrid_ranker.py:124-127` enforce `pool == 0` or `pool >= top_k`, use
   the exact cap, and require execution proof. **Fixed.**
2. **FACT, baseline, CRITICAL — invalid ablations.** Lexical and temporal/lifecycle
   stages contaminated vector-only, sparse-only, and graph-only. Current
   `engine/hybrid_ranker.py:494-535` activates lifecycle policy only for hybrid;
   sparse-only candidate order comes only from raw BM25 at `:390-403`. **Fixed.**
3. **FACT, baseline, CRITICAL — persistence success could be fabricated.** Snapshot
   exceptions were swallowed, the consistent SQLite backup was discarded, file
   naming prevented rotation/restore discovery, and integrity checked the wrong
   vector artifact. `storage/mindfile.py:172-263,652-705` and
   `api/dependencies.py:247-260` now propagate failure and use a verified SQLite-
   derived portable archive. **Fixed.**
4. **FACT, baseline, CRITICAL — unsafe exposure.** The service could bind publicly
   without authentication, used wildcard CORS, and exposed destructive/admin
   endpoints. `main.py:56-70,306-337` now refuses unsafe binds and configures
   explicit host/CORS/auth/rate controls. **Fixed for a single-secret deployment;
   not multi-tenant authorization.**
5. **FACT, current, HIGH — graph independence is unproven.** Hybrid graph anchors
   default to the top three prior candidates (`engine/hybrid_ranker.py:424-443`).
   Graph-only requires explicit anchors (`:132-133`), but no gold-independent
   per-question anchor artifact exists. **Not theater in code; unproven gain.**
6. **FACT, current, HIGH — “temporal reasoning” still exceeds the complete path.**
   Graph primitives accept `as_of` (`storage/graph_index.py:410-531`), but
   `engine/graph_search.py:33-67` exposes no `as_of`, and hybrid traversal calls
   omit it. Historical graph retrieval therefore defaults to wall time. **Partial.**
7. **FACT, current, HIGH — Observer/Reflector was naming around summarization.**
   Current docs no longer make the claim. `engine/consolidation.py:47-110` is a
   hierarchical LLM summary; sources are retained and linked, but there are no
   distinct observer and reflector algorithms. **Terminology corrected.**
8. **FACT, current, HIGH — node `memory_kind` and confidence are structurally
   stored, not functionally ranked.** They survive SQL/graph persistence, but no
   default retrieval score consumes node confidence or changes policy by memory
   kind. Edge confidence is functional. **Still open.**
9. **FACT, baseline, HIGH — contradiction was semantic similarity, not
   contradiction.** Current `engine/consolidation.py:443-479` explicitly avoids
   cosine inference and uses a conservative slot/value heuristic. The heuristic
   can still misclassify multi-valued preferences such as liking tea and coffee.
   **Safer, not natural-language inference.**
10. **FACT, current, HIGH — the 10M–100M/KV-cache story is unproved.** The resource
    report marks model KV reduction false (`engine/resource_accounting.py:285`).
    Current 4096-d in-memory vector-component lower bounds are 1.29 GB/5.16
    GB/12.9 GB at 10M/40M/100M source tokens before text, BM25, graph, Python, and
    scratch memory. **Research target, not a capability.**

## 4. Claim-versus-code audit

| Claim | Code reality | Evidence | Verdict |
|---|---|---|---|
| Strict native 4096 everywhere | Enforced at provider responses, SQLite blobs, vector adds/batches/rebuild/query, and resource arithmetic. | `storage/sqlite_store.py:20,263-290`; `storage/vector_index.py:39-52,149-150,445-447`; `engine/resource_accounting.py:30,95` | MOSTLY TRUE on primary paths |
| Independent tri-signal retrieval | Dense and sparse are independent; graph-only is explicit-anchor traversal; hybrid graph is normally candidate-anchored. | `engine/hybrid_ranker.py:135-137,320-357,424-443` | PARTIALLY TRUE |
| RRF + mxbai reranker | RRF is real; CE execution/cap is now validated. No held-out gain measurement exists. | `engine/fusion.py`; `engine/hybrid_ranker.py:124-127`; eval execution flags | TRUE implementation, UNPROVEN benefit |
| Causal/typed graph | Type, direction, confidence, validity change path legality/strength. | `storage/graph_index.py:463-531` | TRUE, conventional |
| Temporal reasoning | Bitemporal storage and graph primitives exist; historical graph query wiring and historical text candidate generation do not. | `storage/sqlite_store.py:187-223`; `engine/graph_search.py:33-67` | PARTIALLY IMPLEMENTED |
| Structured memory | Entities/time/confidence/kind persist; time and edge confidence function, node kind/confidence largely do not. | SQLite schema + ranker audit | PARTIALLY IMPLEMENTED |
| ACT-R-like salience | Deterministic recency/access/degree weighted score. | `engine/salience.py:19-52` | MISLEADING historically; docs corrected |
| Observer/Reflector | Hierarchical lossy derived summary, exact sources retained. | `engine/consolidation.py:47-128` | FALSE historically; corrected |
| Deterministic evaluation | Client artifacts are reproducible/immutable; provider output, timestamps, server state, CUDA/FAISS and multi-process scheduling are not bit-identical. | `eval_ledger.py:186-216,282-350`; ablation attestation below | MISLEADING historically; bounded claim now |
| Production provider policy | URL/key binding and explicit research opt-in are centralized for primary LLM/embedding paths. | `engine/provider_policy.py:15-52`; `engine/embedding.py`; `engine/llm_client.py` | MOSTLY TRUE |
| Atomic `.mind` persistence | Archive publication/restore is staged and verified; manifest hashes are integrity, not authenticity. | `storage/mindfile.py:172-263,652-705` | MOSTLY TRUE |
| Benchmark-proven gains | No valid held-out hybrid gain exists; sparse-only LoCoMo baseline is real and below gate. | offline artifact and live receipt | FALSE/UNPROVEN |

## 5. Embedding integrity audit

**FACT:** Primary ingestion/query/storage paths now reject any non-finite vector
whose shape is not exactly `(4096,)`. Batch/rebuild code validates row count,
unique IDs, dimensionality, and finite values before swapping live state.
Provider URL and key selection are bound to provider policy.

**Residual trust boundary:** legacy direct `VectorIndex.load`, `GraphIndex.load`,
and BM25 local tooling can deserialize trusted local pickle/FAISS artifacts.
The primary API constructs derived indexes with `index_path=None` and portable
archives contain no pickle (`storage/mindfile.py:7,185`). Therefore “strict
primary runtime invariant” is defensible; “every arbitrary legacy tool and
untrusted local file” is not.

**Live evidence:** the budgeted TEI call returned no vector. Dimension correctness
of the configured live endpoint remains **UNPROVEN**.

## 6. Retrieval audit

- Dense, raw BM25, and graph traversal have distinct candidate paths.
- Explicit caller weights now win; routing supplies only omitted values
  (`engine/hybrid_ranker.py:139-180`).
- RRF validates weights, rank constant, and duplicate IDs.
- `min_score` is applied after meaningful output normalization rather than to raw
  ~`1/(60+r)` values.
- Controlled modes bypass lifecycle, salience, temporal decay, visual merge, and
  all post-rankers; this is covered by behavior tests.
- Supported named modes remain `hybrid`, `vector_only`, `sparse_only`,
  `vector_sparse`, and `graph_only`; there are no independently implemented
  `vector_graph` or `sparse_graph` modes.

**Answer:** current HybridMind is genuinely hybrid in implementation, but graph
usually expands/reweights candidates anchored by other signals. It is not yet
evidence-backed as a superior hybrid retriever.

## 7. Graph audit

Typed parallel edges are stored by edge ID; updates/deletes no longer erase every
relation between two nodes. Typed direction, confidence, half-open validity, and
temporal decay influence traversal/path strength (`storage/graph_index.py:134-136,
252-304,463-531`). Rebuilds validate off to the side before swapping live state.

Remaining limitations:

- public graph traversal does not accept historical `as_of`;
- `get_shortest_path` remains a simple path utility rather than the full
  validity/confidence/type policy;
- hybrid graph anchors are normally derived from other retrieval signals;
- no held-out reachability or answer gain attributable to graph edges exists;
- auto-edge quality and graph growth at large corpus scale are unmeasured.

The graph is **functionally real, but its retrieval value is unproven**.

## 8. Temporal and validity audit

Current storage separates event time, validity time, assertion time, and access
time. Every nonempty event/valid boundary must be a complete ISO-8601 value and
is canonicalized to UTC (`storage/sqlite_store.py:310-353`). Node versions
preserve assertion/validity history; clear/forget/hard compaction erase history
according to explicit policy. LoCoMo timestamps are parsed from the dataset's
declared format, preserve the raw value, and record the dataset's missing-timezone
UTC ordering assumption.

Open gaps:

- historical node text versions are not candidate-generatable from vector/BM25;
- public graph/hybrid traversal does not propagate query `as_of`;
- conflict detection is a narrow slot heuristic, not temporal NLI;
- natural-language query-time parsing is intentionally permissive and must never
  be reused for persisted data.

## 9. Structured memory audit

Fact extraction is fail-closed, bounded, schema-validated, strict about dates,
and retry-idempotent for identical normalized fact text. Deterministic fact and
edge IDs prevent duplicate retries. Container/session scoping is null-safe,
including legacy empty tags. Causal `led_to`, `supersedes`, `contradicts`, and
provenance `derived_from` edges have observable graph semantics.

Fields only count as functional when consumed. Event/valid times and edge
confidence count. Node confidence and `memory_kind` currently do not. Entities
help inference/metadata but do not establish entity reasoning. Raw turns remain
important retrieval documents.

## 10. Salience and access audit

`compute_salience` is a bounded weighted combination of recency, access frequency,
and degree centrality (`engine/salience.py:19-52`). Access-tracked search bypasses
the response cache, preventing cached hits from suppressing access updates.
Maximum graph degree is precomputed per search/prune rather than rescanned for
each candidate.

It is still a score multiplier with hand-set weights, not a learned memory model.
It can produce popularity feedback, and no ablation demonstrates downstream
quality. It is off by default and is no longer called ACT-R.

## 11. Query routing and decomposition audit

Routing is regex classification, not a model (`engine/query_router.py:1-48`). It
changes omitted weights for temporal/multihop/entity/default types and preserves
explicit caller values. `metadata_filter` is explicitly reserved and always
`None`, avoiding the former implicit extracted-fact filter.

Decomposition is an optional LLM call with output bounds, novel-entity rejection,
single-question rejection, deduplication, and temporal-constraint preservation
(`engine/query_decomposition.py:41-119`). These guards make it safer; no controlled
benchmark proves it improves recall enough to justify cost.

## 12. Observer/Reflector audit

There is no Observer/Reflector architecture. Consolidation hierarchically
summarizes all supplied source facts, refuses truncation-as-success, keeps exact
sources, creates deterministic summary IDs, and records `derived_from` provenance.
It cannot archive/replace exact facts (`engine/consolidation.py:113-130`). This is
real derived summarization, not reflection or reversible compression. A summary
can omit a future-relevant detail, but the source remains searchable.

## 13. Benchmark and ablation audit

Baseline defects included cross-conversation LoCoMo leakage, answer-string
relevance, dead CE, invalid graph payloads, same-set tuning, skipped failures,
and relabelled plan artifacts. Current LoCoMo retrieval uses conversation scope
and exact qualified evidence IDs; ledgers have unique run IDs, immutable
manifests, failure receipts, and completion row-count checks
(`eval_ledger.py:186-216,282-350`). Same-set LoCoMo sweep is quarantined before
data/API access (`eval_locomo_retrieval.py:662-677`).

The ablation matrix is honestly labelled
`client_request_attested_not_server_runtime_attested`
(`scripts/ablation_matrix.py:52,248`). It pins RRF and `top_k`, but cannot attest
the external server commit, environment, index, or corpus. Graph-only remains
plan-only without gold-independent anchors. Pre-rerank ordering is not exposed,
so CE lift cannot be measured. LongMemEval and MuSiQue lack protocol-complete
per-instance ingestion/isolation. Their results cannot be called official
benchmark results.

The deterministic normalized answer-overlap score is an internal heuristic, not
an LLM judge or official benchmark accuracy.

## 14. Determinism audit

Now deterministic within a declared scope: stable normalized identities,
question IDs, evidence IDs, seeded offline sampling, explicit tie-break ordering,
immutable manifests, and completion receipts. Not deterministic: wall timestamps,
remote model responses, provider retries, GPU kernels, approximate-index behavior
across versions/hosts, and multi-process scheduling. “Bit-identical system” is
not defensible; “reproducible client artifact with pinned inputs/config” is.

## 15. Provider policy audit

`engine/provider_policy.py:15-52` validates allowed HTTPS hosts and RunPod IDs.
Embedding and LLM selection bind URLs to the matching key; custom URLs require
explicit opt-in. Hack Club is research-only and cannot silently replace Z.AI or
RunPod. Legacy RunPod admin scripts now use bearer-auth REST rather than keys in
URLs. No research-proxy or LLM provider call occurred in this remediation run.

## 16. Security audit

Fixed: unsafe non-loopback startup, wildcard credentialed CORS, missing trusted
hosts, unauthenticated destructive endpoints, key-in-URL admin calls, raw provider
exception/body reporting, and executable pickle in portable archives.

Residuals:

- one shared API secret is authentication, not tenant-level authorization;
- TLS, identity, roles, audit logs, and a secret manager belong at deployment;
- rate limiting and mutation coordination are process-local;
- manifest SHA-256 detects corruption but does not authenticate provenance;
- trusted local legacy pickle loaders remain a tooling trust boundary.

## 17. Test quality audit

Fresh full suite: **279 passed, 3 skipped in 20.21 s**. Compilation and
`git diff --check` passed after integration. High-value tests now fail on CE pool
contract violations, controlled-mode contamination, non-4096/non-finite vectors,
parallel-edge deletion, temporal boundary errors, container leakage, erasure
history retention, snapshot corruption/partial publish, provider-policy bypass,
cache deadlock/access suppression, mutation races, preflight overrun/retry, and
destructive-path rollback.

Limitations: most retrieval tests are in-process fixtures/mocks; no real CE,
end-to-end API corpus, crash-injection process, multi-worker, or held-out hybrid
quality suite ran. Green tests prove contracts, not research benefit.

## 18. Failure modes

| Failure | Current behavior | Verdict |
|---|---|---|
| Wrong/non-finite embedding | Rejected before storage/index swap | fail loudly |
| Malformed LLM facts | Extraction error; no empty-success fallback | fail loudly |
| Provider timeout | Bounded/retried only within explicit policy; eval records failure | fail closed |
| Corrupt snapshot | Exact-file/checksum/semantic validation; corrupt newest can fall back | safe recovery |
| Partial snapshot publish | Temp + fsync + atomic replace; error propagates | safe within OS limits |
| Derived-index mutation failure | SQL rollback plus full rebuild | fail closed |
| Concurrent in-process mutation | Shared async/sync coordinator | serialized |
| Multi-process mutation | No distributed lock | open risk |
| Bulk process crash | Item workflow can leave committed prefix | not crash-atomic |
| Clear/forget | Current rows, chunks, versions, incident edges, derived summaries erased | explicit erasure |
| Live TEI timeout | One call, no retry/fallback, evaluation denied | correct negative result |

## 19. Performance and scalability concerns

The bounded local run measured only 256 synthetic vectors and 32 queries:
component-sequence p95 0.962 ms, p99 1.088 ms, 8,781,050 serialized bytes, and
117,891,072 peak RSS. It excludes embedding, HTTP, reranking, reader, and real
corpus quality. These are component measurements, not production latency.

The current vector representation retains FAISS HNSW data plus a normalized
float32 rebuild cache. At 4096 dimensions its vector-only lower bound is 33,024
bytes/vector. A 100M-token/256-token-chunk projection needs 390,625 vectors and
12.9 GB before excluded components, exceeding half the 16.87 GB host RAM. BM25,
NetworkX, text, mappings, allocation, and build scratch can dominate. Compressed
or disk-resident vectors are not implemented. Large-scale feasibility remains
**UNPROVEN**.

## 20. Documentation accuracy

The README, architecture, algorithm, performance, phase status, and agent
instructions were rewritten to remove production/SOTA/Observer/Reflector/KV-cache
claims not supported by code. Documentation now distinguishes implementation,
measurement, projection, and research target. Remaining risk is ordinary drift;
the evaluation artifacts, not prose, must remain authoritative.

## 21. What is actually novel?

No component is established as research-novel. The useful engineering combination
is: local authoritative SQLite; exact 4096-d dense storage; sparse and graph
indexes; typed temporal multi-edges; bitemporal versions; conservative derived
summaries; strict provider and artifact policy; and adversarial evaluation gates.
That integration can be valuable without being novel.

## 22. What is conventional?

FAISS HNSW, BM25S, NetworkX traversal, RRF, cross-encoder reranking, regex routing,
LLM fact extraction, LLM summarization, recency/frequency/degree salience, FastAPI,
SQLite WAL, and checksum archives are conventional techniques. “Real, but
conventional” is the correct description.

## 23. What is currently unproven?

- hybrid > raw BM25 or dense retrieval on a held-out, correctly isolated corpus;
- graph-only retrieval with gold-independent query anchors;
- graph, temporal, salience, decomposition, reranker, or summary marginal gain;
- live RunPod TEI 4096-d correctness/latency (the canary timed out);
- live cross-encoder execution and latency;
- official LoCoMo/LongMemEval/MuSiQue answer scores;
- 10M/40M/100M-token real-corpus feasibility;
- prompt-source reduction preserving answer quality;
- any transformer KV-cache reduction;
- multi-process safety, adversarial archive authenticity, or multi-tenant security;
- research novelty.

## 24. Top ten next fixes

1. Pass an explicit `as_of` through API → graph engine → traversal/proximity and
   make historical text versions candidate-generatable.
2. Define node-confidence and memory-kind semantics, then ablate them; otherwise
   remove them from capability claims.
3. Produce gold-independent per-question graph anchors and graph-only evaluation.
4. Expose pre-CE ordering/model/pool receipts to measure actual reranker lift.
5. Build protocol-complete, snapshot-attested LoCoMo/LongMemEval/MuSiQue ingestion.
6. Run held-out dense/sparse/hybrid/graph comparisons only after TEI is healthy.
7. Replace compensating bulk commits with a request journal or crash-atomic design.
8. Add a distributed lock/queue and external rate limiter for multi-worker use.
9. Sign/HMAC snapshot manifests when adversarial artifact provenance matters.
10. Implement measured compressed/disk-resident vector alternatives before making
    100M-token feasibility claims.

## 25. Final scores

| Dimension | Attacked baseline | Current repaired tree |
|---|---:|---:|
| Engineering quality | 5.0 | 7.5 |
| Architectural coherence | 4.0 | 6.5 |
| Retrieval sophistication | 4.0 | 5.5 |
| Graph usefulness | 2.0 | 4.0 |
| Temporal correctness | 2.0 | 5.0 |
| Memory correctness | 3.0 | 6.5 |
| Benchmark integrity | 2.0 | 5.5 |
| Test quality | 6.0 | 8.0 |
| Reproducibility | 2.0 | 6.0 |
| Production readiness | 2.0 | 4.5 |
| Documentation accuracy | 3.0 | 8.0 |
| Research novelty | 2.0 | 2.5 |
| LARP / overclaiming | 8.0 | 3.0 |

## 26. Final classification

**Baseline:** over-engineered prototype, partially implemented research ideas,
benchmark theater, and architecture theater in specific subsystems.

**Current:** strong engineering prototype with real conventional hybrid-retrieval
machinery and materially improved correctness/security. It is not a genuine
research-grade system, not production-ready for multi-tenant/multi-worker use,
and not benchmark-proven.

## Evidence artifacts

- `benchmarks/results/offline_locomo_bm25s.json` — 10 conversations, 5,882 turns,
  1,977 eligible exact-evidence questions; BM25S exact evidence recall@10
  0.544696 (95% bootstrap CI 0.524168–0.566162), prompt-source token-proxy
  reduction@10 0.983186, MRR 0.385478.
- `benchmarks/results/offline_resource_frontier.json` — bounded local component
  measurement and analytic capacity projections; zero external calls.
- `benchmarks/results/live_tei_canary_plan.json` — checksum-bound one-call plan,
  modeled maximum $0.17.
- `benchmarks/results/live_tei_canary_result.json` — one 60-second `ReadTimeout`,
  no retry/fallback, no evaluation admitted.

Research interpretation was checked against the original RRF paper, LoCoMo,
LongMemEval/LongMemEval-V2, RULER, RetrievalAttention, Quest, PyramidKV,
Memorizing Transformers, RAPTOR, HippoRAG, HNSW/FAISS, DiskANN, and SPFresh.
Those papers motivate experiments; none transfers its result to this repository.

## The one-sentence verdict

After deleting the marketing, HybridMind is a much better-hardened local RAG
database with promising graph and temporal plumbing, but it still has no valid
evidence that this complexity beats a simple BM25 baseline, and its only bounded
live 4096-dimensional embedding canary timed out.
