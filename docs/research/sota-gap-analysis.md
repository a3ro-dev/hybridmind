# HybridMind state-of-the-art gap analysis

**Audit date:** 2026-08-14  
**Audited checkout:** `3422f226d5adc650802e5aeb87a0302b0765635f` (`main`)  
**Repository:** the checked-out `D:\hybridmind` tree, whose Git remote resolves to `a3ro-dev/hybridmind`  
**Evidence policy:** repository measurements, direct code inspection, and literature claims are labelled separately. Cross-system scores are not treated as directly comparable unless the dataset, question subset, reader, judge, retrieval depth, and information boundary match.

## 2026-08-22 implementation update

The original audit below is retained because it records the state that caused
the engineering work. Its five highest-priority implementation gaps are now
addressed in the current dirty worktree:

- search responses carry a corpus generation, resolved-control hash, actual
  channel counts, HNSW controls, graph anchors, and cross-encoder execution
  evidence; cache keys include the authoritative generation;
- timezone-aware `as_of` now reaches dense, sparse, graph, final filtering, and
  graph traversal, with half-open validity tests and cache separation;
- exact-text dedup now retains distinct evidence provenance, scope, memory kind,
  and validity intervals;
- explicitly enabled sparse, graph, ColBERT, GNN, visual, lexical, and reranker
  stages fail closed instead of silently becoming another condition; vector
  compaction refuses to drop an active row whose raw vector is unavailable;
- bit-identical raw/active 4096-d vectors use a lossless SQLite `NULL` override
  representation. A 512-node offline experiment saved exactly 16,384 bytes per
  node and reduced its database from 17,297,408 to 8,908,800 bytes (48.50%)
  while preserving bit-exact logical reads, with zero provider calls.

Final verification on the completed research state passed 387 Python tests
with 3 skipped, all 16 legacy verification tests, all 4 MemoryBench TypeScript
tests, formatting, Python compilation, and `.venv` dependency checks. The
historical pre-change results in section 2.1 remain verbatim for auditability.

Two research controls were also made explicit without changing production
defaults: `hnsw_ef_search`/`hnsw_ef_construction` are configuration and trace
fields, and sparse retrieval may use a source-preserving `metadata.sparse_text`
key while responses continue to return authoritative text and evidence IDs.
The sparse-key mechanism is positive only on LoCoMo so far; it remains
exploratory. The architecture-neutral program and current evidence are in
`docs/research/design-space-experiment-program.md`.

## Executive finding

HybridMind is a serious local retrieval service, not a transformer KV-cache replacement and not a validated 10M–100M-token memory system. Its strongest verified properties are its SQLite-authoritative write path, exact 4096-dimensional runtime embedding contract, portable snapshot validation, explicit controlled retrieval modes, and unusually careful evaluation ledgers. Its largest weaknesses are not a missing fashionable component. They are execution-integrity and semantic-contract gaps:

1. The live search response does not attest the corpus generation, resolved configuration, actual candidate channels, reranker execution, or graph anchors. A harness manifest can therefore prove what it requested but not the production path that ran.
2. Temporal validity exists in SQLite and graph primitives but is not wired through the public search request. Point-in-time questions cannot reliably constrain dense, sparse, and graph evidence to what was valid then.
3. Exact-text deduplication can collapse distinct memories with different evidence IDs, provenance, temporal intervals, or scopes. This can make a readable result list look cleaner while making exact evidence recall wrong.
4. Several explicitly enabled backends or experimental stages catch broad exceptions and silently become no-ops or downgrade to another backend. That violates the repository's fail-closed contract and invalidates causal ablations.
5. The default write path stores the same 4096-d vector in both `embedding` and `raw_embedding`, approximately doubling SQLite vector payload when graph-conditioned embeddings are disabled.
6. HybridMind has raw episodes, structured facts, temporal fields, graph relations, salience, and lossy summaries, but it does not yet expose a coherent heterogeneous-memory retrieval contract comparable to newer profile/event/source-record systems.
7. The only fresh end-to-end-quality baseline achievable without waking a provider is sparse exact-evidence retrieval. It is useful and real, but it does not establish answer accuracy, dense retrieval quality, graph gain, or large-scale behavior.

The evidence does **not** support calling HybridMind state of the art, near-state-of-the-art, or a proven large-context substitute. It currently qualifies as a well-hardened experimental retrieval service with some production-grade persistence contracts and several unproven research modules.

## 1. Evidence classes used in this report

- **Measured repository behavior** means a command was executed on this checkout and the artifact or test output is retained.
- **Verified implementation fact** means the behavior follows from the executed code path and is covered by direct inspection, usually with an existing test.
- **Literature claim** means the cited paper or official repository reports the result. It is not a HybridMind result and is not independently reproduced here.
- **Hypothesis** means a falsifiable proposed mechanism and is not stated as an implemented gain.
- **Speculation** means a plausible direction without enough evidence to prioritize for implementation.

## 2. Reproducible baseline

### 2.1 Environment and verification

Measured on an Acer Nitro ANV15-51 with an Intel i5-13420H (12 logical CPUs), 16.87 GB RAM, and an RTX 4050 Laptop GPU with 6,141 MiB VRAM. The repository `.venv` uses Python 3.13.5. PyTorch 2.11.0 is CPU-only in this environment. Relevant versions include NumPy 2.4.3, `faiss-cpu` 1.13.2, `bm25s` 0.3.9, NetworkX 3.6.1, FastAPI 0.135.2, Pydantic 2.12.5, and pytest 9.0.2.

Measured verification before production changes:

- Primary offline suite: **279 passed, 3 skipped**, 16.56 s on the recorded rerun.
- Python compilation: passed.
- Python dependency check: passed.
- `memorybench` frontend/provider test: 1 passed.
- Legacy `verify/` suite: 4 failed, 12 passed. All four failures expect features that their own fixtures explicitly disable or do not provide: automatic edges while `AUTO_EDGES=false`, consolidation/community summaries without an LLM provider, and fact extraction while `FACT_EXTRACTION=false`. These are stale test contracts, not evidence that fail-closed production defaults should be weakened.
- Frontend format check: failed because one provider file is unformatted and configured `ui/**/*.ts(x)` globs match no files.
- TypeScript check: failed because the configured `bun-types` package is absent.
- Ruff was not present in `.venv`, so no lint result is claimed.
- No provider call was made.

The human-readable baseline is in `experiments/reports/baseline.md`; immutable source artifacts are under `experiments/baselines/`.

### 2.2 Fresh LoCoMo sparse retrieval baseline

The offline baseline uses one BM25S index per LoCoMo conversation, raw dialogue-turn text as the document, question text as the query, and canonical dialogue IDs as relevance labels. It excludes malformed and unresolved annotations rather than guessing them. It makes zero network, embedding, reranking, or reader calls.

| Metric | Fresh result |
|---|---:|
| Questions in dataset | 1,986 |
| Exact-evidence questions scored | 1,977 |
| Turns indexed | 5,882 |
| MRR, first exact evidence | 0.385478 |
| Exact evidence recall@1 | 0.256600 |
| Exact evidence recall@5 | 0.463909 |
| Exact evidence recall@10 | 0.544696 |
| Exact evidence recall@25 | 0.639894 |
| Exact evidence recall@50 | 0.708226 |
| Exact evidence recall@100 | 0.773835 |
| Any exact evidence hit@10 | 0.594841 |
| All exact evidence hit@10 | 0.503794 |
| Query latency mean / p95 / p99 | 0.337 / 0.338 / 0.448 ms |

Artifact: `experiments/baselines/20260814-3422f226-locomo-bm25s.json`, SHA-256 `ad0bd46990658e5cb143132049febd91627fd3bebf178ec297fe14065e46b7e0`. Dataset SHA-256: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.

This is a genuine retrieval baseline but **not** a HybridMind end-to-end score. It says nothing about remote dense embeddings, graph retrieval, cross-encoder reranking, question answering, hallucination rate, or provider cost.

### 2.3 Fresh resource frontier

The offline resource probe used 512 deterministic finite 4096-d vectors and 64 queries at top-10. It measured 17,557,781 serialized bytes, 125,964,288 bytes peak process RSS, 23,740,416 bytes RSS increase, 0.13267 s index build, and deterministic replay equality. Component search latency was 0.732 ms p50, 0.954 ms p95, and 1.147 ms p99.

The current vector-component analytic lower bounds are approximately 1.29 GB at the repository's 10M source-token scenario, 5.16 GB at 40M, and 12.90 GB at 100M. These exclude SQLite/text payload, BM25, graph, process overhead, remote embedding/reader cost, and any model KV cache. They therefore cannot substantiate a 10M–100M system claim.

Artifact: `experiments/baselines/20260814-3422f226-resource-frontier.json`, SHA-256 `9f6a9ac2151f68d69659be6dadfd4df3d95148c6fb7c8065abff6fcb7bfa8002`.

## 3. Actual executable architecture

### 3.1 Ingestion-to-answer trace

1. FastAPI routes receive raw nodes, bulk nodes, edges, or opt-in structured session facts.
2. Node text is embedded by a configured remote service. The runtime selector accepts only native, finite, exactly 4096-dimensional output. It does not project, pad, truncate, or mix widths.
3. `storage/sqlite_store.py` writes the authoritative node, metadata, temporal fields, version state, and optional entity mappings in a SQLite/WAL transaction.
4. FAISS HNSW, BM25/BM25S, and a NetworkX `MultiDiGraph` are derived indexes. API mutations update them in the same process and rebuild them from validated SQLite when compensation is required.
5. `engine/hybrid_ranker.py` creates independently controlled dense, sparse, and graph candidates, applies temporal and heuristic scores in some modes, fuses ranks, and optionally applies configured rerankers.
6. `/search/hybrid` returns retrieved nodes. The repository's evaluation scripts can optionally call a separately configured answer model, but the core search service is retrieval, not a complete answer-generation product.

### 3.2 What is strongly implemented

- SQLite/WAL is authoritative; derived indexes are rebuilt from it.
- Node and edge mutations use transactions/savepoints and compensate on derived-index failures.
- Remote embedding provider identities and credentials are endpoint-bound.
- Runtime embeddings are validated as finite, native 4096-d arrays.
- `search_mode` explicitly controls vector-only, sparse-only, vector+sparse, graph-only, and hybrid paths.
- Weighted reciprocal-rank fusion defaults to `k=60`; explicit request weights are not intentionally replaced by routing.
- Reranker pool `0` means disabled; positive pools are bounded and must contain the requested final `top_k`.
- Exact evidence IDs and conversation scoping are implemented in the current evaluators.
- Evaluation runs create immutable manifests, per-question ledgers, checksummed completion receipts, and failed receipts.
- Portable v2 `.mind.zip` restore is path-checked, checksummed, and semantically validated against SQLite; the portable format avoids untrusted pickle/NetworkX deserialization.

### 3.3 Documented or implied behavior that is incomplete

| Claimed or implied capability | Measured implementation reality |
|---|---|
| Temporal point-in-time search | SQLite and graph primitives have validity fields and `as_of` support, but public search requests and `GraphSearchEngine` do not carry `as_of` through the complete candidate path. |
| Provenance-aware retrieval | Nodes retain metadata, but exact-text dedup can remove distinct evidence IDs before final ranking. |
| Graph benefit | Graph candidates exist, but no fresh request/server/corpus-attested paired result establishes positive held-out graph gain. Hybrid graph expansion may seed from top fused candidates, so independence must be measured carefully. |
| Entity memory | Normalized node entities exist and optional entity edges can be inferred, but robust alias resolution, cross-session identity persistence, and pronoun resolution are not established. |
| Conflict resolution | Structured facts can create contradiction/supersession relations and validity intervals, but the general raw-node write path does not provide comprehensive contradiction resolution. |
| Consolidation | Optional provenance-linked summaries exist. They are lossy derived summaries, not a demonstrated Observer/Reflector architecture, and their retrieval gain is unproven. |
| Learned fusion/GNN/ColBERT | Configuration and code paths exist, but untrained or absent checkpoints are not implemented results. Several paths can silently skip execution on exceptions. |
| Large-scale memory | Resource projections and planning scripts exist. No executed 10M–100M evidence-ID evaluation is retained. |
| Local embedding fallback | A legacy local `EmbeddingEngine` class and dependency comments remain, but the runtime selector correctly refuses it for production. The dead code and dependency description conflict with the repository contract. |
| Safe derived persistence | Portable v2 snapshots are safe, but legacy vector/BM25/graph constructors still support pickle-backed files when passed paths. |
| Complete snapshot semantics | Portable graph export omits edge validity, supersession, and confidence fields present in SQLite; SQLite still protects authoritative restore semantics, but the derived graph artifact is incomplete. |

## 4. Correctness and production risks ranked by severity

### P0: benchmark execution cannot be independently attested

`eval_ledger.py` seals evaluator configuration, dataset hash, source hashes, Git state, per-question rows, and budget use. The server response, however, does not expose a corpus-generation identifier, resolved settings hash, candidate-channel counts, graph anchors, temporal scope, model identities, or a complete stage trace. A receipt can say `search_mode=hybrid` without proving that dense, sparse, graph, and reranker stages all executed against the intended corpus generation.

**Consequence:** component gains are not causally attributable, and stale/misconfigured servers can produce formally sealed but scientifically weak results.

### P0: temporal validity is not an end-to-end query contract

Edges and nodes have temporal fields, and graph primitives can accept an `as_of` time. The public request model does not. Graph traversal in `engine/graph_search.py` can use inactive edges, and dense/sparse candidates are not globally filtered to a caller's point in time.

**Consequence:** “what was true then?” and “what is true now?” can both retrieve temporally invalid evidence. This is a semantic correctness defect, not merely a ranking opportunity.

### P0: exact-text dedup can erase provenance

The hybrid ranker deduplicates by exact text before graph expansion and again on output. Two separately observed facts, two scopes, or two temporal versions can have identical text but different node/evidence IDs.

**Consequence:** exact evidence recall, provenance, confidence aggregation, contradiction visibility, and temporal history can all be corrupted by a presentation-oriented dedup rule.

### P0: enabled features may fail open

Examples found by direct inspection:

- Sparse backend factory can silently downgrade SPLADE to BM25S, then BM25S to pure BM25.
- FAISS absence silently selects NumPy search despite the configured architecture.
- Automatic edge inference catches broad failures and can return zero inferred edges.
- Entity extraction may catch all errors and act as if no entities exist.
- Optional GNN, ColBERT, and lexical/reranking blocks can catch broad exceptions and disappear from a run.
- Vector compaction logs and skips a missing raw vector cache entry instead of aborting the rebuild.

**Consequence:** configuration labels cease to identify executed systems. A failed ablation can masquerade as a valid “feature on” condition.

### P1: duplicated default embedding payload

When no explicit raw embedding is provided, SQLite defaults `raw_embedding_blob` to the same bytes as `embedding_blob`. With graph-conditioned embeddings off by default, every node stores two identical 4096-float vectors.

**Consequence:** roughly 16 KiB avoidable SQLite payload per node before page/index overhead, plus corresponding cache/rebuild pressure. The fix must preserve graph-conditioning provenance and migration compatibility.

### P1: process-local derived state and caches

The Docker command launches one Uvicorn worker, so the bundled deployment is consistent. The repository nonetheless depends on Gunicorn and does not actively reject multi-worker launch. Database managers, derived indexes, and query invalidation are process-local.

**Consequence:** a custom multi-worker deployment can commit SQLite in one worker while another serves stale derived indexes and cached results. Either multi-worker use must be explicitly unsupported/fail-fast, or a corpus-generation synchronization design is required.

### P1: legacy unsafe persistence remains callable

Vector, graph, and BM25 classes retain pickle-backed `save`/`load` behavior and some constructors auto-load when a path is supplied. Current `DatabaseManager` builds derived indexes from SQLite with `index_path=None`, and portable v2 restore is safe.

**Consequence:** alternative callers can reintroduce unsafe deserialization or stale derived state. Legacy APIs should be quarantined, removed, or guarded as trusted-local migration only.

### P2: misleading response semantics and validation gaps

- Controlled hybrid endpoint modes still report `search_type="hybrid"`.
- Graph `direction` is not strongly validated at the request schema.
- `multi_domain_eval.py --help` attempts a live localhost request instead of providing parser help.
- Legacy test expectations and frontend tool configuration have drifted.

These do not dominate retrieval quality, but they increase operator error and weaken observability.

## 5. Independent research findings

### 5.1 Benchmark evolution

**LongMemEval (ICLR 2025).** The benchmark evaluates information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. The paper decomposes memory design into indexing, retrieval, and reading. Reported experiments find value decomposition, fact-augmented keys, and time-aware query expansion useful; on its temporal subset, a strong-LLM time-range extractor improved recall while a weaker 8B extractor could hallucinate or miss temporal cues. This supports explicit temporal scoping, but it also warns against silently trusting LLM-extracted dates. [Paper](https://arxiv.org/abs/2410.10813) · [official repository](https://github.com/xiaowu0162/LongMemEval)

**LongMemEval-V2 (May 2026, work in progress).** LME-V2 shifts from user-history recall to learning environment experience. It has 451 curated questions over histories up to 500 trajectories and 115M tokens, covering static state, dynamic state, workflows, environment gotchas, and premise awareness. Its AgentRunbook-R stores raw states, events, and strategy notes in separate pools; AgentRunbook-C stores trajectories as files and lets a coding agent inspect them. The reported coding-agent result is stronger but substantially slower, making accuracy-latency frontier reporting central. This benchmark is directly relevant to the repository's 10M–100M aspirations, but HybridMind does not yet ingest its multimodal trajectories or implement its evidence interface. [Paper](https://arxiv.org/abs/2605.12493) · [official repository](https://github.com/xiaowu0162/LongMemEval-V2)

**LoCoMo (ACL 2024).** LoCoMo contains ten long multi-session conversations with dialogue IDs and QA evidence annotations, plus summarization and multimodal dialogue tasks. It is useful for exact-evidence retrieval and temporal/multi-hop cases but is small at the conversation level, and published answer scores often use different subsets, readers, judges, and retrieval depths. HybridMind's fresh baseline therefore reports exact evidence IDs rather than comparing its retrieval number to vendor answer accuracy. [Paper](https://arxiv.org/abs/2402.17753) · [official repository](https://github.com/snap-research/locomo)

**BEAM (ICLR 2026).** BEAM contains 100 coherent conversations and 2,000 validated questions across 128K, 500K, 1M, and 10M-token scales. Its LIGHT method combines episodic memory, working memory, and a salient-fact scratchpad, with reported ablations showing complementary contributions. BEAM is the closest available scale gate below LME-V2's 115M-token histories, but the dataset and ingestion/evaluation path are not currently integrated in HybridMind. [Paper](https://arxiv.org/abs/2510.27246) · [official repository](https://github.com/mohammadtavakoli78/BEAM)

### 5.2 Repeated architectural principles in strong recent systems

**Heterogeneous representation with source fidelity.** LeanMem separates stable profile memory, temporally evolving event memory, and source-grounded record memory. Records retain pointers to original dialogue rather than trusting lossy summaries. It selectively evolves events and allocates retrieval budgets by query evidence demand. The August 2026 v1 paper reports gains and paired tests on LoCoMo and LongMemEval-S, but it is a new arXiv preprint and has not been independently reproduced here. Its principle maps well to HybridMind's existing raw nodes, structured facts, temporal validity, and provenance-linked summaries. [LeanMem paper](https://arxiv.org/abs/2608.03463)

**Different memory tiers and deferred consolidation.** LightMem reports sensory filtering, topic-aware short-term consolidation, and sleep-time long-term updates, with efficiency as a first-class metric. SimpleMem reports structured multi-view compression, asynchronous consolidation, and query-aware retrieval scope. MemoryOS uses short-, mid-, and long-term personal memory tiers. These systems differ materially, but they converge on avoiding uniform processing of every interaction and on moving expensive maintenance off the critical query path. [LightMem paper](https://arxiv.org/abs/2510.18866) · [SimpleMem paper](https://arxiv.org/abs/2601.02553) · [MemoryOS paper](https://arxiv.org/abs/2506.06326)

**Bi-temporal, source-linked facts.** Zep/Graphiti distinguishes valid time from transaction time, links semantic facts back to source episodes, resolves entities, and invalidates overlapping contradictions rather than deleting history. Its own paper explicitly notes that the small DMR benchmark fits in modern context windows and is inadequate as a strong memory test. HybridMind already stores many analogous fields but does not complete the point-in-time retrieval contract. [Zep paper](https://arxiv.org/abs/2501.13956) · [Graphiti repository](https://github.com/getzep/graphiti)

**Extraction plus explicit update operations.** Mem0 extracts candidate memories, compares them with retrieved prior memories, and selects add/update/delete/no-op behavior; its graph variant reports a modest aggregate improvement over the base system in its paper. The headline latency and token reductions are relative to full-context processing and should not be transferred to HybridMind. [Mem0 paper](https://arxiv.org/abs/2504.19413) · [official repository](https://github.com/mem0ai/mem0)

**Graph propagation is useful only when it retains factual retrieval.** HippoRAG 2 combines passage integration with Personalized PageRank and online LLM use, reporting gains on associative tasks without the factual-memory regression observed in some graph RAG systems. Its datasets are knowledge QA rather than conversational memory, so the mechanism is relevant but the score is not directly portable. [HippoRAG 2 paper](https://arxiv.org/abs/2502.14802) · [official repository](https://github.com/OSU-NLP-Group/HippoRAG)

**Agent-managed hierarchical context.** MemGPT introduced virtual context management with explicit memory tiers and model-controlled movement between them; Letta is the current framework descended from that work. This is a broader agent architecture than HybridMind's retrieval service and should not be conflated with a retrieval-index improvement. [MemGPT paper](https://arxiv.org/abs/2310.08560) · [Letta repository](https://github.com/letta-ai/letta)

**Adaptive retrieval can be valuable, but every controller has a cost.** LongMemEval uses time-range expansion, LeanMem uses an LLM retrieval planner, and LME-V2 compares RAG controllers with coding-agent search. These results support query-dependent streams and budgets. They do not show that an unvalidated heuristic router or an unpriced provider call is automatically beneficial.

### 5.3 What does not follow from the literature

- A graph is not intrinsically better than dense+sparse retrieval. It must add gold evidence beyond its anchors and candidate channels.
- Compression is not intrinsically memory. Lossy summaries can improve cost while deleting future evidence.
- LLM-based entity or temporal extraction is not ground truth. It requires schema validation, source spans, confidence, and failure receipts.
- A larger retrieval `k` is not a free quality gain. It changes context tokens, reader difficulty, and latency.
- An answer judge score cannot substitute for exact retrieval recall, and exact retrieval recall cannot substitute for grounded answer quality.
- Published LoCoMo percentages are frequently incomparable because papers select different categories, readers, judges, prompt formats, and retrieval depths.
- No cited system proves that this repository can serve 100M tokens within the measured hardware or cost envelope.

## 6. Competitive architecture matrix

The matrix reports architecture, not a leaderboard.

| Dimension | HybridMind (measured checkout) | LeanMem | Zep / Graphiti | Mem0 | HippoRAG 2 | MemGPT / Letta | LME-V2 AgentRunbook |
|---|---|---|---|---|---|---|---|
| Raw source retention | Raw nodes retained; summaries provenance-linked | Record memory points to source | Episodes retained and linked | Conversation plus extracted memories | Passages integrated with graph | Archival memory / sources | Raw trajectory slices or files retained |
| Memory types | Raw nodes, opt-in facts, derived summaries, graph | Profile, event, record | Episodes, entities, semantic facts | Extracted facts; graph variant | Passages, phrase/entity graph | Core, archival, recall/context tiers | Raw states, events, strategy notes; or files |
| Write policy | Raw writes default; fact extraction opt-in | Ignore/profile/event/record scheduler | LLM entity/fact extraction and resolution | Extract plus add/update/delete/no-op | Offline graph/index construction | Agent-controlled memory tools | Trajectory processing into dedicated pools |
| Temporal model | Node/edge validity exists; query wiring incomplete | Event anchors and evolution | Bi-temporal validity and transaction time | Time metadata; less explicit in base paper | Not a conversational bi-temporal system | Agent-managed records | Dynamic-state/event pool |
| Provenance | Metadata and source edges; text dedup risk | Source pointer for records | Bidirectional episode/fact linkage | Memory history/update operations | Passage linkage | Inspectable memory blocks/sources | Raw trajectory evidence |
| Entity resolution | Optional heuristic/entity path; unproven aliases | NER for record keys | Embedding + full-text + LLM resolution | Graph entities/relations | Phrase/entity graph | Model-managed | Query/controller dependent |
| Retrieval | Dense, sparse, graph, RRF, optional rerank | Type-selective adaptive retrieval | Hybrid vector/full-text/graph | Vector; graph variant | PPR plus passage retrieval | Agent/tool search | Multi-pool dense controller or coding-agent inspection |
| Point-in-time search | Incomplete end-to-end | Temporal event selection | First-class design | Not primary paper contribution | No | Memory content dependent | Dynamic history questions, not a general TKG API |
| Consolidation | Optional lossy derived summaries | Event-only localized deferred evolution | Fact invalidation/update | Update/consolidation operations | Graph integration | Reflection/self-editing | Notes/events or filesystem artifacts |
| Forgetting | Archive/access state; no validated selective pruning | Ignore and selective maintenance | Invalidate, preserve history | Delete operation | Not primary focus | Agent-managed | Not primary focus |
| Adaptive `k` / streams | Fixed top-k plus heuristic routing; no attested planner | Query-specific types and budgets | Search API dependent | Search parameters | PPR/retrieval controls | Agent decides actions | Controller may skip streams; coding agent adaptive |
| Evaluation status here | Fresh sparse exact-ID baseline only | Literature claim only | Literature claim only | Literature claim only | Literature claim only | Literature claim only | Literature claim only |

## 7. Ranked falsifiable hypotheses

The ranking weights correctness, scientific identifiability, expected leverage, implementation cost, and zero-provider reproducibility. “Success” means a predeclared result on a held-out or adversarial set; passing unit tests alone is not success.

### H1 — server-attested retrieval makes component evaluations valid

- **Hypothesis:** Adding a deterministic corpus generation and execution trace to every search response will reveal stale/missing stages in tests and make paired ablation receipts causally interpretable without materially changing latency.
- **Subsystem:** API models, database manager, ranker, mutation lifecycle, evaluators.
- **Difficulty:** medium.
- **Expected quality impact:** none directly; high impact on validity.
- **Latency/cost:** sub-millisecond metadata assembly, zero provider cost.
- **Risk:** leaking sensitive configuration or provider credentials; trace must expose hashes/identities, never secrets.
- **Experiment:** run controlled sparse-only, vector+sparse, graph-only, reranker-off, and reranker-on test doubles; assert response traces match executed stages and change corpus generation after every committed mutation.
- **Success:** all stage-on conditions prove nonzero execution evidence; stage-off conditions prove absence; cached responses carry the generation that created them; no secrets; focused latency increase below 1 ms p95 on local deterministic tests.
- **Failure:** trace is derived from requested flags rather than actual execution, or can describe a stale corpus as current.

### H2 — provenance-safe dedup fixes exact evidence loss without increasing semantic redundancy excessively

- **Hypothesis:** Deduplicating only when text, scope, temporal validity, and provenance identity are equivalent will preserve exact evidence IDs and contradiction/history visibility, while a presentation-level grouping field can still help readers collapse true duplicates.
- **Subsystem:** hybrid ranker and search response.
- **Difficulty:** low to medium.
- **Expected quality impact:** positive on exact evidence recall in duplicate/adversarial cases; neutral on ordinary cases.
- **Latency/cost:** negligible; result count may increase.
- **Risk:** more near-identical context tokens.
- **Experiment:** adversarial corpus with identical text across sessions, evidence IDs, validity intervals, and scopes; compare old text-only dedup with provenance-aware dedup under identical top-k.
- **Success:** 100% preservation of distinct gold IDs and temporal versions; unchanged ordering on a no-duplicate regression corpus; bounded context increase reported.
- **Failure:** any exact evidence is removed solely because another node has the same text.

### H3 — an explicit `as_of` contract prevents temporally invalid retrieval

- **Hypothesis:** Propagating a validated point-in-time timestamp through dense, sparse, graph, cache, and final filtering will make current-vs-historical questions deterministic and prevent future leakage.
- **Subsystem:** request models, SQLite filters, graph search, hybrid ranker, cache key, response trace.
- **Difficulty:** medium.
- **Expected quality impact:** large on adversarial temporal reversals; neutral elsewhere.
- **Latency/cost:** modest metadata/filter overhead, zero provider cost.
- **Risk:** incorrect boundary semantics and timestamps without timezones.
- **Experiment:** histories with overlapping/superseded facts, exact boundary instants, invalid future edges, and identical text across versions.
- **Success:** all candidates satisfy `valid_from <= as_of < valid_until` with explicit null-bound semantics; no cache cross-contamination; graph path excludes inactive edges.
- **Failure:** any future or expired node/edge can affect score or traversal at the requested instant.

### H4 — enabled optional retrieval features must fail closed

- **Hypothesis:** Replacing silent backend/stage downgrades with typed execution errors will turn false ablations into failed receipts without breaking default opt-out operation.
- **Subsystem:** sparse/vector factories, edge inference, optional rerankers, persistence compaction.
- **Difficulty:** medium.
- **Expected quality impact:** no direct gain; prevents invalid results and partial mutations.
- **Latency/cost:** none in success path.
- **Risk:** deployments that unknowingly relied on fallback will now fail, which is intentional but needs migration messaging.
- **Experiment:** dependency absence, corrupt checkpoints, malformed entity output, vector-search failure, missing compaction vectors, and default-disabled controls.
- **Success:** requested unavailable features fail with typed errors and mutation compensation; disabled features remain no-ops; ledgers finalize as failed.
- **Failure:** a run labelled with one backend executes another or an enabled stage silently contributes zero evidence.

### H5 — metadata-rich sparse keys improve exact-evidence retrieval on held-out conversations

- **Hypothesis:** Indexing a turn with its speaker and session date, while returning the original source turn and evidence ID, improves LoCoMo temporal/entity retrieval over raw-turn BM25S without using answers, gold IDs, or test-set tuning.
- **Subsystem:** offline experiment first; production sparse document construction only if confirmed.
- **Difficulty:** low.
- **Expected quality impact:** small to moderate recall/MRR gain, concentrated in temporal and entity questions.
- **Latency/cost:** slightly larger sparse corpus and query time; zero provider cost.
- **Risk:** date tokens can dominate, or same-set selection can overfit ten conversations.
- **Experiment:** predeclare a conversation split; select among raw, speaker-prefixed, date-prefixed, and speaker+date on development conversations only; evaluate one locked winner once on held-out conversations. Use paired bootstrap confidence intervals and report per-category deltas and token footprint.
- **Success:** held-out exact recall@10 improves by at least 1.0 absolute point with no more than 10% p95 latency or 15% sparse-text footprint increase, and no category loses more than 2 points.
- **Failure:** development-only gain, held-out regression, or improvement comes from adding multiple evidence IDs to one result.

### H6 — default raw-vector dedup materially reduces storage with bit-identical retrieval

- **Hypothesis:** Storing `raw_embedding=NULL` when it is byte-identical to `embedding`, with readers treating null as “same as embedding,” cuts default SQLite vector payload nearly in half without changing rebuilt indexes or search results.
- **Subsystem:** SQLite schema/write/read path, migration, resource accounting.
- **Difficulty:** medium.
- **Expected quality impact:** none.
- **Latency/cost:** lower I/O and storage; possible branch on read.
- **Risk:** graph-conditioned or legacy readers may distinguish absent raw vectors incorrectly.
- **Experiment:** create/migrate corpora with conditioned and unconditioned embeddings; compare database bytes, checksums of reconstructed vectors, index results, and rollback behavior.
- **Success:** bit-identical reconstructed raw vectors and search ordering, at least 40% vector-blob storage reduction in unconditioned corpora, no reduction for genuinely different raw vectors.
- **Failure:** any ambiguity between “missing/corrupt” and “same as embedding,” or migration loses a distinct vector.

### H7 — query-adaptive sparse retrieval depth improves the latency/context Pareto frontier

- **Hypothesis:** A deterministic score-margin/entropy policy can return smaller `k` for confident sparse queries and expand ambiguous queries without reducing held-out exact evidence recall.
- **Subsystem:** experiment harness, then retrieval policy.
- **Difficulty:** medium.
- **Expected quality impact:** neutral recall at lower average context/latency; possibly positive for ambiguous cases.
- **Latency/cost:** intended reduction; zero provider calls.
- **Risk:** BM25 score scales vary by conversation and query length.
- **Experiment:** learn thresholds only on development conversations, lock them, evaluate paired held-out recall, average/p95 `k`, token proxy, and latency.
- **Success:** recall@adaptive budget is within a predeclared 0.5-point non-inferiority margin of fixed `k=10` while average retrieved-token proxy falls at least 20%.
- **Failure:** threshold depends on evaluation labels at runtime or hides category regressions.

### H8 — graph retrieval helps only with independent explicit anchors and typed edges

- **Hypothesis:** Graph expansion adds exact evidence for relational/multi-hop cases only when anchors are explicit and gold-independent; seeding from already fused results mostly redistributes existing candidates.
- **Subsystem:** graph search and ablation harness.
- **Difficulty:** medium to high because real graph construction requires provider-gated extraction.
- **Expected quality impact:** potentially positive on multi-hop, uncertain overall.
- **Latency/cost:** graph traversal overhead; graph construction cost must be measured.
- **Risk:** synthetic graphs overstate benefit and auto edges leak semantic similarity.
- **Experiment:** first use hand-authored adversarial graphs with explicit query entities; later run priced, preflight-approved ingestion on held-out benchmark data. Compare graph-only, vector+sparse, and vector+sparse+graph with candidate-overlap decomposition.
- **Success:** graph contributes gold evidence absent from non-graph candidate pools and improves held-out multi-hop recall with bounded latency.
- **Failure:** gain disappears with independent anchors, or graph returns only candidates already found by dense/sparse channels.

### H9 — heterogeneous profile/event/record memory can improve fidelity-cost balance

- **Hypothesis:** Mapping existing raw nodes to immutable records, structured facts to profiles/events, and summaries to explicitly derived views enables query-type-specific retrieval without deleting source evidence.
- **Subsystem:** ingestion schema, fact extraction, consolidation, retrieval planning.
- **Difficulty:** high and provider-dependent.
- **Expected quality impact:** potentially large on temporal and profile questions; uncertain on open-domain recall.
- **Latency/cost:** write-time LLM calls and maintenance; retrieval can become cheaper if routing is accurate.
- **Risk:** extraction errors, irreversible filtering, representation drift, benchmark overfitting.
- **Experiment:** requires a fresh offline resource report and priced plan bound to its SHA-256 before any live run. Preserve every source span, compare raw-only, fact-only, and heterogeneous union on held-out questions.
- **Success:** union improves answer and exact-source recall while staying on a defensible cost/latency frontier; fact-only is never allowed to erase raw evidence.
- **Failure:** compression gains cost but reduces exact source recall, or scheduler decisions lack source spans.

### H10 — bounded iterative retrieval helps only after evidence-sufficiency calibration

- **Hypothesis:** A maximum-two-pass loop triggered by explicit evidence insufficiency improves multi-hop answer accuracy more than it increases cost.
- **Subsystem:** query decomposition, reader, budget enforcement.
- **Difficulty:** high and provider-dependent.
- **Expected quality impact:** potentially positive on multi-hop; likely unnecessary on single-hop.
- **Latency/cost:** one or more provider calls and retrievals.
- **Risk:** controller hallucination, runaway cost, evaluation leakage.
- **Experiment:** priced-plan-gated held-out multi-hop set with fixed maximum calls/tokens and failed receipts.
- **Success:** statistically and practically meaningful paired accuracy gain with predeclared cost ceiling and no single-hop regression.
- **Failure:** controller invokes the second pass indiscriminately or cannot identify missing evidence better than chance.

## 8. Implementation sequence justified by evidence

1. Implement H1–H4 first. They make later measurements trustworthy and repair semantic correctness without provider calls.
2. Build the split-aware offline harness and test H5 and H7. Integrate only a held-out winner.
3. Test H6 independently as a resource optimization with bit-identical retrieval.
4. Use adversarial hand-authored graphs for H8 before paying for extracted graphs.
5. Consider H9 and H10 only after a fresh resource report, a priced usage-limited plan bound to its SHA-256, and `scripts/preflight.py --plan <plan> --validate-only` succeed. No live provider evaluation is authorized by this report.

## 9. Success gates and rejected shortcuts

An improvement is accepted only when:

- the experiment has an immutable configuration, dataset hash, source/commit provenance, seed, cost, and completed/failed receipt;
- the production path reports actual execution evidence;
- compared conditions use the same examples and relevance labels;
- exact evidence metrics remain distinct from answer overlap and answer-judge metrics;
- development selection and held-out confirmation are separate;
- latency, context tokens, storage, and provider calls are reported alongside quality;
- a feature requested “on” cannot silently become “off”;
- raw/source evidence survives every lossy representation;
- the final claim matches the measured scope.

The following are explicitly rejected as evidence:

- counting test names or configuration fields as implemented features;
- treating `scripts/ablation_matrix.py` planning output as an executed experiment;
- using historical checkpoints as current held-out confirmation;
- selecting weights on the same questions used for the final score;
- using answer-string overlap as evidence recall;
- calling a deterministic answer matcher an LLM judge;
- comparing HybridMind exact retrieval recall to another system's LLM-judged answer accuracy;
- claiming a prompt/KV-cache reduction from a regex source-token proxy;
- extrapolating 512-vector behavior into a validated 100M-token deployment.

## 10. Current classification and highest-value research direction

**Classification:** neither SOTA nor near-SOTA on the available evidence. HybridMind is competitive as a local, inspectable hybrid-retrieval research service in persistence discipline and benchmark bookkeeping, but it lacks the executed benchmark breadth, temporal query semantics, source-preserving heterogeneous retrieval, and scale evidence required for a stronger label.

**Highest-value next research direction:** establish an execution-attested, source-preserving temporal retrieval core, then evaluate low-cost heterogeneous sparse keys and adaptive budgets on disjoint LoCoMo conversations. This path is testable with zero provider calls, directly repairs correctness, and creates the scientific foundation needed before expensive graph construction or agentic retrieval can produce credible evidence.
