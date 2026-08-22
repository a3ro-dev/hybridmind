# Prior-art mechanism ledger

Status: research ledger, 2026-08-22.  The sources below were checked against
the linked paper, official repository, or official documentation.  A paper
result is not a HybridMind result.  “Measured” means repository-local evidence
already exists; “researched” means the primary source was checked; “planned”
means the smallest experiment has been specified but not run; “rejected”
means the mechanism is currently excluded by a declared compatibility or
correctness gate, not that the paper is invalid.

The comparison ontology is:

    HYBRIDMIND
      RETRIEVAL: dense + sparse + graph
      MEMORY MODEL: episodic + semantic + temporal
      candidate fusion
      reranking / selection
      evidence grounding
      answer generation

Backend and ANN choice, mutation/freshness, filtering and scope, persistence,
and resource cost are orthogonal system axes.  A replacement is admissible if
it wins a source-preserving, exact-evidence comparison; preserving the current
implementation has no value by itself.  SQLite remains the authoritative
source for the experiments, but a challenger may replace any derived index,
fusion stage, memory policy, or reader if it wins the declared gates.

Evidence conventions:

* Primary means the linked paper, official repository, official model card, or
  official product documentation.  Repository-local citations are labeled
  explicitly and do not establish an external benchmark result.
* Reported numbers retain their original dataset, metric, scale, and baseline
  context.  They are not transferred to conversational memory, exact source
  recall, or this machine.
* Every quality experiment must report exact evidence-ID recall and candidate
  pool ceiling before reranking, corpus/session scope, temporal validity,
  complete failures, and an immutable manifest.  Answer overlap is not
  evidence recall.

## Tier S: mandatory investigations

The order below is the requested priority order.

### TurboQuant / Turbovec

* Mechanism: TurboQuant is an online, data-oblivious vector quantization
  method.  Its construction combines rotation/polar-style scalar
  quantization with a residual inner-product correction; the goal is a
  distortion bound without data-dependent codebook training.  Turbovec is an
  independent implementation/adapter, not the paper itself.
* Primary evidence: the paper is
  [TurboQuant, arXiv:2504.19874](https://arxiv.org/abs/2504.19874).  Its
  distortion and vector-search claims are evaluated in the paper’s ANN/vector
  and compression settings, while its most visible memory/speed claims are
  for the authors’ GPU/KV-cache workloads: the abstract reports quality
  neutrality at 3.5 bits/channel, marginal degradation at 2.5 bits/channel,
  and higher nearest-neighbor recall than product quantization in its
  experiments.  Neither setting measures conversational exact-evidence
  recall.  The implementation is
  [Turbovec](https://github.com/ryancodrai/turbovec); its speed and memory
  figures are implementation claims and must be reproduced at a pinned
  commit.
* Constraints: the paper and implementation have different scope; code,
  CUDA/CPU kernels, alignment, and supported distance functions must be
  pinned.  Quantized IDs and codes are derived state and cannot become the
  authority.  The local experiment pins the MIT-licensed `turbovec==1.0.0`
  Windows wheel and hashes both the wheel-derived native binary and package
  source; it remains an opt-in research dependency, not a runtime fallback.
* Causal hypothesis: quantization could reduce 4096-d vector memory and
  improve cache/ANN latency, but can lower candidate recall or alter ties.
  The causal effect is systems frontier movement, not memory-model quality.
* Smallest architecture-neutral experiment: on one immutable SQLite corpus
  and deterministic 4096-d test vectors, compare exact flat search with
  float32, TurboQuant-style codes, and Turbovec codes at equal top-k and
  fixed memory budgets.  Measure exact neighbor recall, exact evidence recall,
  p50/p95/p99 query time, build time, bytes/vector, update/delete behavior,
  and rebuild equivalence across seeds.  Add a real-vector arm only after a
  policy-compatible native 4096-d embedding is available.
* Status: researched and measured for synthetic 4096-d ANN mechanics.  Across
  five seeds, Turbovec 4-bit search retained mean Recall@10 of 0.85344
  uncalibrated and 0.85375 calibrated at about 5.2x compression relative to
  the raw float32 matrix; the 2-bit arms retained about 0.563 Recall@10 at
  about 10.4x compression.  Calibration was effectively null on isotropic
  random data.  This rejects either mode as a default on current evidence:
  there is no real embedding/evidence-quality result, and batch timing was
  not randomized.  Artifact:
  `experiments/results/offline-turbovec-frontier-4096-multiseed-20260822.json`
  (SHA-256 `6cca904276fd13dc8f64513a0aa633d6f490983088c6b7235f7952646d81439f`).

### Vespa

* Mechanism: Vespa combines inverted/lexical retrieval and nearest-neighbor
  retrieval in one query plan, applies metadata filters, and uses rank
  profiles for first-phase and phased second-phase ranking.  Tensor closeness,
  BM25, and custom expressions can feed candidate generation and reranking.
  Exact nearest-neighbor search is available as an accuracy control; HNSW is
  the approximate option.
* Primary evidence: [Vespa nearest-neighbor search
  documentation](https://docs.vespa.ai/en/querying/nearest-neighbor-search)
  describes exact versus HNSW retrieval, filtering, and query-time controls;
  [Vespa ranking documentation](https://docs.vespa.ai/en/basics/ranking.html)
  describes rank profiles and phased ranking.  These are mechanism and
  operational docs, not a conversational-memory quality benchmark.  The
  source is [vespa-engine/vespa](https://github.com/vespa-engine/vespa).
* Constraints: a server/schema deployment and asynchronous indexing create
  synchronization, freshness, and operational costs.  Vespa Cloud is a
  managed commercial service; self-hosting and the repository’s current
  license/dependencies must be pinned before redistribution.  SQLite IDs,
  generation hashes, and temporal filters must be joined without allowing a
  stale Vespa document to look authoritative.
* Causal hypothesis: phased ranking and filter pushdown may improve candidate
  recall/latency at larger corpora; they may also add indexing lag and obscure
  which channel caused a gain.
* Smallest architecture-neutral experiment: export a fixed corpus with
  source IDs, scope, validity intervals, dense vectors, and text into Vespa;
  run exact and HNSW dense, BM25, and hybrid rank profiles against the same
  query manifest.  Compare exact evidence recall, filter/temporal correctness,
  stage traces, p50/p95/p99, index bytes, update visibility, crash/rebuild
  equivalence, and total operator cost against in-process controls.
* Status: researched; planned.  No external Vespa deployment or score is
  evidence for this checkout.

### ColBERTv2 (and PLAID)

* Mechanism: ColBERT uses token-level embeddings and late interaction:
  query-token to document-token similarities are reduced by MaxSim instead
  of forcing a passage into one vector.  ColBERTv2 adds residual compression
  and denoising-oriented training.  PLAID adds centroid interaction and
  pruning to make late interaction cheaper at scale.
* Primary evidence: [ColBERTv2, arXiv:2112.01488](https://arxiv.org/abs/2112.01488)
  reports a 6–10x footprint reduction in its passage-retrieval experiments
  relative to the cited ColBERT baselines, with ranking quality measured on
  the paper’s MS MARCO/BEIR-style evaluations; those figures are not memory
  evidence.  [PLAID, arXiv:2205.09707](https://arxiv.org/abs/2205.09707)
  reports up to 7x GPU and 45x CPU speedups in the stated 140M-passage
  comparison to vanilla ColBERTv2.  The official
  [ColBERT repository](https://github.com/stanford-futuredata/ColBERT) is
  MIT-licensed.  Reported scale and hardware are precise to those papers,
  not portable to this laptop or to LoCoMo.
* Constraints: token vectors multiply storage and indexing work; model
  checkpoints, tokenizer, GPU kernels, and collection preprocessing must be
  pinned.  A late-interaction candidate does not prove source grounding:
  every token-vector result still needs an authoritative evidence ID.  The
  repository and checkpoints may have separate terms.
* Causal hypothesis: token matching may recover entity, date, and paraphrase
  evidence missed by one dense vector.  The null is that a fixed-pool
  cross-encoder or transparent lexical reranker gets the same gain for less
  storage.
* Smallest architecture-neutral experiment: lock candidate IDs from current
  dense+sparse retrieval and compare no reranker, cross-encoder, ColBERTv2
  MaxSim, and a PLAID-like pruned MaxSim on exactly the same candidates.
  Then run ColBERT as an independent candidate channel.  Report pre/post
  exact-ID recall, candidate ceiling, rank changes, token-vector bytes,
  CPU/GPU p50/p95/p99, and no-op/shuffled controls.
* Status: researched; MaxSim mechanism measured, ColBERTv2/PLAID not
  reproduced.  A pinned local BGE-M3 token-vector run improved held-out
  Recall@10 by +0.02922 on fixed BM25S pools, interval
  [+0.01419, +0.04085], and +0.03848 on BGE learned-sparse pools, interval
  [+0.01935, +0.05467].  It required 837,558,272 token-vector bytes for 5,881
  turns and 13.38 minutes of CPU encoding.  No independent ColBERT candidate
  index, residual compression, PLAID pruning, or ColBERTv2 training protocol
  was reproduced.  Artifact:
  `experiments/results/offline-locomo-bgem3-mechanisms-20260822.json`.

### SPLADE++

* Mechanism: SPLADE uses a BERT MLM head plus sparse regularization to expand
  text into weighted vocabulary terms.  The expanded document and query
  vectors are searched with an inverted index; SPLADE++ improves training
  through distillation/hard-negative sampling and related configurations.
  It is learned sparse retrieval, not BM25 with a different tokenizer.
* Primary evidence: the official
  [Naver SPLADE repository](https://github.com/naver/splade) links the
  SPLADE and SPLADE++ papers and exposes indexing, pruning, and retrieval
  code.  The SPLADE++ citation is the SIGIR 2022 paper,
  [From Distillation to Hard Negative Sampling](https://doi.org/10.1145/3477495.3531857),
  whose reported metrics are MS MARCO/TREC-style retrieval metrics under its
  training and index settings.  They do not measure exact IDs in a
  conversational corpus.  A useful methodological check is
  [A Unified Framework for Learned Sparse Retrieval,
  arXiv:2303.13416](https://arxiv.org/abs/2303.13416), which finds
  effectiveness/latency changes depend strongly on weighting and expansion
  choices.
* Constraints: inference creates sparse postings and can be expensive; model,
  tokenizer, vocabulary, checkpoint, pruning threshold, and checkpoint/data
  licenses vary.  The repository cites a CC BY-NC-SA 4.0 paper context for
  SPLADE v2; inspect the selected checkpoint’s terms before production use.
  The sparse output must preserve original source IDs and never substitute
  generated expansion text for evidence.
* Causal hypothesis: learned expansion may add paraphrase/entity candidates
  beyond BM25S/Tantivy/Lucene, but expansion density and inference cost may
  dominate.  The relevant outcome is candidate recall at a fixed budget, not
  a leaderboard number.
* Smallest architecture-neutral experiment: on identical raw turns and
  held-out conversation splits, compare BM25/BM25S, Tantivy/Lucene-style
  BM25, and SPLADE++ with matched top-k and source IDs.  Record exact-ID
  recall by question type, posting count, index bytes, encoding time, query
  p50/p95/p99, failures, and provider calls (zero for offline model
  checkpoints).  Keep source representation and scope fixed.
* Status: researched; planned.  The existing repository has conventional
  sparse experiments, but no SPLADE++ result.

### DiskANN / FreshDiskANN

* Mechanism: DiskANN uses a graph ANN structure designed for SSD plus
  quantization and memory tiers.  FreshDiskANN addresses streaming updates,
  maintaining search freshness during inserts/deletes without requiring a
  complete merge for every change; current DiskANN code also exposes data
  provider/quantizer and filtered-search machinery.
* Primary evidence: [Microsoft DiskANN repository](https://github.com/microsoft/DiskANN)
  and its [project/research overview](https://github.com/microsoft/DiskANN/wiki/DiskANN-Project-and-Research-Overview-%282018%E2%80%90present%29)
  document the implementation lineage and dynamic-update work.
  [FreshDiskANN, arXiv:2105.09613](https://arxiv.org/abs/2105.09613) evaluates
  real-time streaming freshness and search performance in its ANN datasets;
  its machine-count and throughput claims are not a memory-service result.
* Constraints: the C++ build, SIMD/GPU/SSD assumptions, graph files,
  quantizer formats, update log, and filtering semantics are substantial.
  The repository is MIT-licensed at the checked source, but transitive
  dependencies, old branches, and patents/operational requirements still
  need a pinned audit.  The derived index must be rebuildable from SQLite and
  must fail closed on partial update logs.
* Causal hypothesis: disk-backed mutable ANN may be required at the scale
  gate, but for small corpora its build and I/O cost may lose to exact or
  HNSW search.  Freshness, not headline static recall, is the causal question.
* Smallest architecture-neutral experiment: create a deterministic update
  stream with inserts, deletes, and superseding vectors; compare exact Flat,
  FAISS HNSW, DiskANN, and a FreshDiskANN-like mutable path at the same
  vectors.  Measure recall at each stream prefix, filter recall, visibility
  lag, write amplification, rebuild/crash equivalence, bytes, RSS, and
  p50/p95/p99 latency.  Start at host-safe sizes and promote only through
  resource gates.
* Status: researched; planned.  No DiskANN-family adapter has been run here.

### HippoRAG

* Mechanism: HippoRAG builds an entity/phrase graph from passages and uses
  Personalized PageRank-style propagation to combine associative graph
  evidence with passage retrieval.  The key distinction from an arbitrary
  graph expansion is learned/constructed association plus a bounded graph
  propagation score.
* Primary evidence: [HippoRAG, arXiv:2405.14831](https://arxiv.org/abs/2405.14831)
  and the [official repository](https://github.com/OSU-NLP-Group/HippoRAG)
  evaluate multi-hop and associative knowledge-retrieval tasks; reported
  scores and resource comparisons are for those QA datasets, not for
  source-linked conversational memory.  In the paper’s abstract, the method
  is reported as up to 20% better on its multi-hop comparisons and 10–30x
  cheaper/6–13x faster than iterative IRCoT; these are not portable
  conversational-memory measurements.  The repository’s HippoRAG2 material
  is a later implementation and must not be conflated with the original
  paper.
* Constraints: graph construction typically requires an LLM/entity extractor,
  phrase normalization, a graph store, and passage-to-node links.  Errors in
  entity resolution can create false paths; PPR can amplify a leaked or
  gold-dependent anchor.  Preserve raw evidence and require explicit,
  gold-independent anchors.
* Causal hypothesis: associative propagation can recover multi-hop evidence
  outside dense/sparse top-k, but gains may be limited to relationship-rich
  datasets or may duplicate a bounded typed traversal.
* Smallest architecture-neutral experiment: use held-out conversation
  questions and a graph built from the input corpus only.  Compare dense,
  sparse, typed-traversal, PPR, and a graph-sham condition at equal candidate
  budgets.  Report exact-ID candidate recall, path/anchor provenance,
  temporal correctness, graph build cost, and p50/p95/p99; no answer or gold
  evidence may seed the graph.
* Status: researched; mechanism partially tested, not reproduced.  A local
  corpus/query-only term/speaker/session PPR ablation preserved every node
  degree in its sham while retaining only 7.68% of term-turn associations.
  Real PPR beat that sham decisively on held-out LoCoMo, but its clustered
  Recall@10 delta over speaker-prefixed BM25S was only +0.00859 with a 95%
  interval spanning zero; the held-out multi-hop delta was exactly zero.
  This rejects a degree-only explanation while rejecting promotion of the
  lightweight graph as a general default.  It is not HippoRAG: there is no LLM
  entity extraction, trained retriever, or HippoRAG benchmark protocol.
  Artifact: `experiments/results/offline-locomo-associative-graph-20260822-v2.json`.

### Graphiti / Zep

* Mechanism: Graphiti represents episodes, entities, and temporal facts in an
  incrementally updated knowledge graph.  It separates validity/transaction
  time, keeps provenance links to source episodes, and combines graph,
  semantic-vector, and keyword retrieval.  Zep is the broader managed
  product lineage; its Context Graph behavior is not equivalent to the
  open-source Graphiti code.
* Primary evidence: the
  [Graphiti repository](https://github.com/getzep/graphiti) and
  [official documentation](https://help.getzep.com/graphiti/getting-started/welcome)
  specify temporal validity windows, incremental construction, episodes,
  provenance, and hybrid retrieval.  The [Zep paper,
  arXiv:2501.13956](https://arxiv.org/abs/2501.13956) reports graph-memory
  results on its stated benchmark(s) and explicitly notes the limitations of
  a small benchmark that fits in modern context windows; no number is
  transferred here.
* Constraints: Graphiti is Apache-2.0 code but depends on an external graph
  backend (for example Neo4j, FalkorDB, or Neptune) and LLM structured
  extraction.  Zep’s managed service and Context Graph Engine are proprietary
  operational dependencies.  Entity/fact extraction can hallucinate or
  mis-time assertions.  SQLite remains the source of truth and source
  episodes must survive every derived update/invalidation.
* Causal hypothesis: explicit bi-temporal facts and incremental graph
  updates may improve historical/latest correctness and multi-hop recall more
  than an undifferentiated graph score; external graph operations may cost
  more than they save.
* Smallest architecture-neutral experiment: construct a synthetic but
  adversarial stream with assertion time, valid intervals, contradiction,
  supersession, aliases, and deletes.  Compare raw dense+sparse, current
  graph traversal, and Graphiti-inspired temporal graph with identical
  source IDs.  Test current, previous, and as-of queries, future leakage,
  update visibility, source-link precision, rebuild equivalence, and
  p50/p95/p99 without any gold-derived anchor.
* Status: researched; planned for a local source-preserving adapter.

### Matryoshka Representation Learning

* Mechanism: Matryoshka Representation Learning trains one embedding whose
  prefixes remain useful at multiple dimensions.  A single model can trade
  storage/search cost against quality by choosing a trained prefix; this is
  not permission to truncate an arbitrary embedding.
* Primary evidence: [Matryoshka Representation Learning,
  arXiv:2205.13147](https://arxiv.org/abs/2205.13147) reports nested-quality
  and efficiency results on the paper’s vision and text tasks: up to 14x
  smaller ImageNet-1K embeddings at the same accuracy, up to 14x retrieval
  speedups on ImageNet-1K and 4K, and up to 2% long-tail few-shot accuracy
  improvement.  The [official repository](https://github.com/RAIVNLab/MRL)
  contains training/evaluation code.  These reported settings are not
  evidence for a 4096-d memory service.
* Constraints: a compatible provider/model must be trained with the nested
  objective, and each selected dimension is a separate contract.  HybridMind
  production vectors are native, finite, exactly 4096 dimensions; arbitrary
  truncation, projection, padding, or mixed-width fallback is prohibited.
  Check model/data/checkpoint terms before use.
* Causal hypothesis: a provider-native nested 4096/short-prefix model could
  improve storage/latency without a separate model, but an untrained prefix
  could destroy evidence recall.
* Smallest architecture-neutral experiment: screen only a policy-compatible
  native MRL endpoint with a declared 4096-d output and a separately declared
  trained prefix.  Compare exact evidence recall, ANN recall, bytes, and
  latency at each trained width against the full-width model; no arbitrary
  slicing of the current vector.
* Status: researched; rejected for arbitrary truncation; planned only for a
  native compatible model.

### FAISS

* Mechanism: FAISS provides exact flat search and approximate indexes
  including HNSW, inverted/PQ, scalar/product quantization, and GPU paths.
  It separates vector index structures from integer IDs and supports
  reconstruction for several index families.
* Primary evidence: the [official FAISS repository](https://github.com/facebookresearch/faiss)
  is MIT-licensed and documents the index families and GPU support.
  [FAISS: A Library for Efficient Similarity Search,
  arXiv:2401.08281](https://arxiv.org/abs/2401.08281) describes the library
  and evaluates ANN accuracy/performance in its stated datasets and hardware.
  Those throughput/recall numbers are not claims about HybridMind’s corpus.
* Constraints: index IDs, serialized index files, GPU availability, SIMD
  behavior, and version changes must be pinned.  FAISS is a derived index:
  SQLite rows, checksums, scope, and evidence IDs remain authoritative.
  Approximate and quantized search require a Flat oracle and seeded builds.
* Causal hypothesis: exact Flat/HNSW/PQ is the cleanest way to distinguish
  representation loss from ANN loss; FAISS may already be sufficient at
  target sizes, making wholesale backend replacement unnecessary or justified
  only by a scale gate.
* Smallest architecture-neutral experiment: compare Flat, HNSW, scalar/PQ,
  and GPU where available on the same 4096-d vectors, filters, IDs, and query
  set.  Report exact neighbor/evidence recall, p50/p95/p99, build/rebuild,
  memory, disk bytes, mutation behavior, and crash restore.
* Status: measured as the repository’s current vector-index control;
  researched for the external comparison.  A 4096-d deterministic-vector
  sweep found HNSW Recall@10 of 0.6652, 0.8621, and 0.98125 at efSearch 64,
  128, and 256; a separately attested threshold run reached 0.99921875 at
  512 and 1.0 at 1024.  These runs expose search effort as a material control
  and motivated explicit `hnsw_ef_search`/`hnsw_ef_construction` settings,
  while retaining 64 as the default until native semantic evidence confirms
  a promotion.  Existing measurements are local mechanics/resource evidence,
  not a semantic benchmark result.

### ScaNN

* Mechanism: ScaNN prunes the search space with partitioning and asymmetric
  scoring/quantization, with optional exact reranking of a candidate set.  It
  is an ANN library rather than a memory model or grounding layer.
* Primary evidence: the [official ScaNN source](https://github.com/google-research/google-research/tree/master/scann)
  documents the partitioning, quantization, and scoring implementation.  Its
  README labels the code as research-oriented and specifies build/hardware
  assumptions; repository benchmark numbers are library/dataset-specific.
  No conversational exact-evidence metric is reported here.
* Constraints: x86/vectorized builds and optional TensorFlow integration make
  reproducibility and packaging nontrivial; pin compiler, CPU features, and
  index parameters.  Its result IDs must be reconciled with SQLite and
  filters/temporal predicates must be tested independently.
* Causal hypothesis: partitioning may lower latency at large static scale,
  while candidate pruning/quantization may lose evidence or complicate
  mutations.  It should replace FAISS only if it wins recall/resource gates
  on the same vectors.
* Smallest architecture-neutral experiment: equal-vector Flat versus FAISS
  HNSW versus ScaNN at matched recall targets and top-k, then a filter and
  update stress arm.  Record CPU-feature provenance, p50/p95/p99, build
  time, bytes, RSS, exact evidence recall, and failure/rebuild behavior.
* Status: researched; planned.  No ScaNN result is measured locally.

### USearch

* Mechanism: USearch is a compact local HNSW implementation with memory
  mapping, on-disk indexes, configurable scalar quantization, and key/ID
  support.  It targets embedded or lightweight deployments rather than a
  graph/temporal memory policy.
* Primary evidence: the [official USearch repository](https://github.com/unum-cloud/usearch)
  documents HNSW, mmap/on-disk use, quantization, and mutation APIs.  The
  repository’s release/fork lineage and examples are implementation evidence;
  any benchmark is hardware/configuration-specific and is not transferred.
* Constraints: verify the exact upstream/fork and license at the pinned
  revision; language bindings and binary wheels may differ in SIMD support.
  Delete/update semantics, key stability, persistence checksums, and
  concurrent readers require direct tests.  An index file is never the
  authority.
* Causal hypothesis: a compact mmap HNSW may reduce local resource use and
  operational complexity compared with FAISS, but it may not improve quality
  and may have weaker filtering/mutation guarantees.
* Smallest architecture-neutral experiment: export the same immutable
  vectors and stable evidence IDs into USearch and FAISS HNSW/Flat.  Compare
  recall, filter behavior, update/delete visibility, rebuild equivalence,
  bytes, RSS, and p50/p95/p99 with cold and warm reads.
* Status: researched; planned.  No USearch result is measured locally.

### Tantivy / Lucene

* Mechanism: Lucene and Tantivy build inverted indexes with analyzers,
  postings, positions, filters, and BM25 scoring.  Tantivy is a Rust library
  inspired by Lucene; it is not a distributed server.  Both are conventional
  transparent sparse controls against which learned sparse methods should be
  compared.
* Primary evidence: [Apache Lucene](https://lucene.apache.org/core/) is the
  official project and [Tantivy](https://github.com/quickwit-oss/tantivy)
  documents BM25 parity, incremental indexing, mmap, filters, and its
  non-distributed scope.  [Tantivy BM25 source](https://github.com/quickwit-oss/tantivy/blob/main/src/query/bm25.rs)
  and [architecture notes](https://github.com/quickwit-oss/tantivy/blob/main/ARCHITECTURE.md)
  make the scoring mechanism inspectable.  Tantivy’s repository is MIT;
  Lucene is Apache-2.0.  Their benchmark numbers are library/query
  benchmarks, not LoCoMo evidence recall.
* Constraints: analyzers, stemming, stop words, field norms, segment
  commits, deletes, and refresh behavior can change quality and freshness.
  Tantivy edits are delete-plus-reindex and become visible after commit and
  reader reload.  Lucene’s JVM and deployment footprint differ materially.
* Causal hypothesis: mature postings and metadata filtering may beat the
  current BM25S implementation on entity/date queries or scale, but they may
  only improve systems behavior.  Learned sparse is promoted only if it adds
  evidence beyond this transparent baseline.
* Smallest architecture-neutral experiment: use identical raw/source-key
  documents, tokenization policy, scope filters, and query manifest to compare
  current BM25/BM25S, Tantivy, and Lucene.  Measure exact-ID recall by slice,
  filter/temporal correctness, index bytes, build/update/refresh latency,
  p50/p95/p99, and crash/rebuild equivalence.
* Status: researched; planned as the conventional sparse control.  The
  repository has local BM25/BM25S measurements but no Tantivy/Lucene run.

## Tier A: second-wave investigations

Every named Tier-A item is listed separately below.  Closely related systems
share an experiment where that preserves causal attribution.

### RAPTOR

* Mechanism: recursively clusters chunks, summarizes clusters, and builds a
  tree that can retrieve at multiple abstraction levels.  It changes the
  memory model from raw episodic turns toward derived semantic summaries.
* Primary evidence: [RAPTOR, arXiv:2401.18059](https://arxiv.org/abs/2401.18059)
  evaluates tree retrieval on the paper’s long-document QA tasks, comparing
  answer metrics under its reader/prompts; its abstract gives a 20-point
  absolute QuALITY accuracy improvement for one GPT-4-plus-RAPTOR condition.
  That result is not exact evidence recall.  The [official repository](https://github.com/parthsarthi03/raptor)
  is MIT.
* Constraints: summarization is lossy and LLM-dependent; tree rebuilds,
  source links, and summary licenses/cost must be recorded.  Derived nodes
  must point to every source ID and never replace raw evidence.
* Hypothesis/experiment: summaries may help global questions at lower prompt
  cost, while raw turns win exact/temporal questions.  Compare raw,
  RAPTOR-tree, and linked hybrid retrieval with the same reader and candidate
  budget; report source-ID recall, summary provenance, unsupported claims,
  tokens, and latency.
* Status: researched; planned.

### LightRAG

* Mechanism: a lightweight graph-enhanced RAG architecture with dual-level
  retrieval over entity/relation graph structure and local/global context.
  It combines graph organization with vector/text retrieval rather than
  merely swapping ANN indexes.
* Primary evidence: [LightRAG, arXiv:2410.05779](https://arxiv.org/abs/2410.05779)
  reports QA results under its datasets/readers; the
  [official repository](https://github.com/HKUDS/LightRAG) documents
  indexing/retrieval configuration and warns that changing embedding/LLM
  settings requires reindexing.  Published answer scores are not exact-ID
  memory evidence.
* Constraints: graph extraction, model configuration, reindexing, and
  potential LLM calls add cost and mutation hazards.  The repo’s demo/status
  claims are not a production guarantee; pin commit, models, prompts, and
  graph schema.
* Hypothesis/experiment: dual-level graph retrieval may improve global and
  multi-hop evidence, but graph construction may be the only source of the
  gain.  Compare a graph-sham, typed graph, and LightRAG-like dual-level path
  with source-linked summaries and gold-independent anchors.
* Status: researched; planned.

### GraphRAG

* Mechanism: builds an entity graph, community structure, and hierarchical
  community summaries; local search combines graph entities with source text,
  while global search aggregates community-level information.
* Primary evidence: the [Microsoft GraphRAG repository](https://github.com/microsoft/graphrag)
  is MIT and explicitly says indexing is expensive, prompt tuning is needed,
  and the repo is a demonstration/methodology rather than an officially
  supported Microsoft offering.  The [Microsoft research
  page](https://www.microsoft.com/en-us/research/project/graphrag/) describes
  the method and its long-document QA evaluation context.  Reported answer
  gains do not establish exact evidence-ID recall.
* Constraints: global summaries are lossy, indexing is LLM-heavy, and
  community detection introduces rebuild/version drift.  Keep source episode
  links and separate global-answer quality from source-grounding precision.
* Hypothesis/experiment: community summaries can help corpus-level synthesis
  but should not replace raw evidence.  Compare raw, graph-sham, local graph,
  and global community retrieval on questions tagged local/global, with exact
  source recall and citation/unsupported-claim metrics.
* Status: researched; planned.

### Mem0

* Mechanism: an agent memory layer extracts candidate memories, retrieves
  related prior memories, and applies add/update/delete/no-op decisions;
  variants add graph relations.  It is a memory lifecycle policy, not just a
  vector index.
* Primary evidence: [Mem0, arXiv:2504.19413](https://arxiv.org/abs/2504.19413)
  reports LoCoMo answer comparisons under its readers and baselines.  Its
  abstract reports a 26% relative LLM-as-a-Judge improvement over OpenAI,
  about 2% higher overall score for graph memory, 91% lower p95 latency, and
  over 90% token savings versus full-context processing.  These are
  benchmark/model/baseline-specific and are not transferable.  The
  [official repository](https://github.com/mem0ai/mem0)
  is Apache-2.0 and the [contextual-add documentation](https://docs.mem0.ai/features/contextual-add)
  describes the update workflow, but docs’ marketing metrics are not primary
  evidence for this project.
* Constraints: extraction and update decisions are model-dependent; deletes
  and updates must preserve audit history and source IDs.  Vector-store,
  graph-store, LLM, and model licenses are separate from the Apache code.
* Hypothesis/experiment: explicit memory updates may reduce stale/duplicate
  recall, but raw episodic evidence may be lost.  Compare append-only raw,
  Mem0-style add/update/delete, and source-linked hybrid policy on update,
  contradiction, and abstention cases; measure exact source recall, stale
  fact rate, write cost, and unsupported claims.
* Status: researched; planned.

### Letta / MemGPT

* Mechanism: MemGPT introduced OS-like virtual context management with
  model-controlled paging between a bounded working context and archival/
  recall memory.  Letta is the current stateful-agent framework descended
  from MemGPT and adds tool-driven memory blocks and agent state.
* Primary evidence: [MemGPT, arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
  evaluates task behavior under its agent and memory-control setup; it is not
  an isolated candidate-recall benchmark.  The [Letta repository](https://github.com/letta-ai/letta)
  documents the current implementation and its “formerly MemGPT” lineage.
* Constraints: an agent control loop, tool calls, prompt/state policy, and
  external model are required.  It can write or page memory nondeterministically
  and is more than a retrieval service.  Source IDs, cost bounds, and failed
  tool calls need receipts.
* Hypothesis/experiment: model-controlled paging may improve context
  selection on multi-step tasks, but a fixed retrieval baseline may win on
  latency, reproducibility, and evidence grounding.  Replay a locked set of
  memory operations against a mock controller, fixed retrieval policy, and
  Letta-like controller; compare exact recall, action count, tokens, latency,
  and provenance.
* Status: researched; planned as a controller comparison, not assumed
  replacement.

### A-MEM

* Mechanism: A-MEM uses Zettelkasten-style notes with dynamic links and
  memory evolution.  New memories can trigger contextual linking and update
  of existing notes, creating a networked semantic/episodic memory.
* Primary evidence: [A-MEM, arXiv:2502.12110](https://arxiv.org/abs/2502.12110)
  reports six-model experiments and LoCoMo-style categories under its own
  memory and reader setup.  The evaluation repository
  [AgenticMemory](https://github.com/WujiangXu/AgenticMemory) is MIT; the
  paper’s system repository [A-mem](https://github.com/agiresearch/A-mem)
  and checkpoints/dependencies must be pinned separately.  Reported answer
  scores are not exact evidence recall.
* Constraints: LLM note creation/linking is costly and can create provenance
  drift, cycles, or stale links.  Note updates must be append-only or
  transactionally versioned in the experiment.
* Hypothesis/experiment: dynamic links may improve associative retrieval over
  static fact extraction.  Compare raw turns, fixed extracted facts, and
  A-MEM-like notes/links on held-out conversations, with source-link precision,
  exact-ID recall, temporal correctness, write cost, and update determinism.
* Status: researched; planned.

### MTEB

* Mechanism: MTEB is a multi-task embedding benchmark/leaderboard spanning
  retrieval, classification, clustering, reranking, and semantic similarity.
  It screens representation models; it is not a memory-store or grounding
  test.
* Primary evidence: [MTEB, arXiv:2210.07316](https://arxiv.org/abs/2210.07316),
  the [official repository](https://github.com/embeddings-benchmark/mteb),
  and [benchmark API documentation](https://docs.mteb.org/api/benchmark/)
  define task datasets and metrics such as nDCG/accuracy/spearman as
  task-specific.  Leaderboard scores must retain model, language, task, and
  revision context.
* Constraints: model dimensions, license, tokenizer, language coverage, and
  task contamination matter.  A high aggregate score does not prove
  conversational exact-evidence retrieval or native 4096-d compatibility.
* Hypothesis/experiment: MTEB can narrow model candidates before expensive
  memory tests.  Select models using a preregistered subset, then evaluate
  locked candidates on local exact-ID LoCoMo/LongMemEval retrieval with a
  native-width check.
* Status: researched; planned as screening only.

### BEIR

* Mechanism: BEIR is a heterogeneous zero-shot information-retrieval
  benchmark that tests dense, sparse, and reranking systems across domains
  with metrics such as nDCG@10 and recall@k.
* Primary evidence: [BEIR, arXiv:2104.08663](https://arxiv.org/abs/2104.08663)
  and the [official repository](https://github.com/beir-cellar/beir)
  define the dataset suite, zero-shot protocol, and task-level metrics.
  Reported scores are heterogeneous web/IR tasks, not source-linked memory.
* Constraints: dataset licenses, domain mismatch, pooling judgments, and
  task-specific qrels limit transfer.  Do not tune on a test collection and
  call the result a memory gain.
* Hypothesis/experiment: BEIR can expose representation and sparse/dense
  candidate behavior before LoCoMo, but its gains must survive a held-out
  memory corpus.  Run one preregistered model/channel screen, then carry
  finalists unchanged into exact-ID memory evaluation.
* Status: researched; planned as a retrieval control.

### LongMemEval

* Mechanism: LongMemEval decomposes long-term memory into indexing,
  retrieval, and reading and tests information extraction, multi-session
  reasoning, temporal reasoning, knowledge updates, and abstention.
* Primary evidence: [LongMemEval, arXiv:2410.10813](https://arxiv.org/abs/2410.10813)
  and the [official repository](https://github.com/xiaowu0162/LongMemEval)
  define curated questions over timestamped conversation histories.  The
  paper reports answer/evaluation metrics under its readers and subsets; it
  does not automatically expose an exact evidence-ID retrieval denominator.
* Constraints: preserve timestamp, session, updates, and abstention labels;
  answer-judge details and reader models must be pinned.  Do not use the
  question answer or evidence annotation to build keys or graph anchors.
* Hypothesis/experiment: explicit time ranges and fact-augmented keys may
  help temporal retrieval, but weaker extraction can hallucinate or miss
  time cues.  Evaluate exact source recall first, then reader faithfulness,
  on conversation/session-disjoint splits with current/previous/as-of slices.
* Status: researched; the local file was measured only far enough to reject
  benchmark admission.  All 500 examples contain only their gold sessions,
  with 948 haystack sessions total, 948 gold sessions, zero distractors, and
  at most six sessions per example; therefore top-10 returns every session.
  The runner now fails closed and emits a failed receipt instead of metrics:
  `experiments/results/offline-longmemeval-session-retrieval-20260822-invalidated.json`
  (SHA-256 `51a94c39b910b9c604ec17162bfdd632993cd63f91046b33e42b066671892490`).
  A genuine retrieval corpus remains required.

### LoCoMo

* Mechanism: LoCoMo is a long, multi-session conversational-memory benchmark
  with dialogue IDs, evidence annotations, temporal/multi-hop questions,
  summarization, and multimodal dialogue tasks.
* Primary evidence: [LoCoMo, arXiv:2402.17753](https://arxiv.org/abs/2402.17753),
  the [ACL paper PDF](https://aclanthology.org/2024.acl-long.747.pdf), and the
  [official data/code repository](https://github.com/snap-research/locomo)
  define the corpus and task categories.  Published answer scores differ by
  category, reader, judge, retrieval depth, and prompt, so no cross-paper
  headline is compared here.
* Constraints: keep conversation-level splits, canonical dialogue/evidence
  IDs, time semantics, and complete failure denominators.  Answer-string
  overlap is not a substitute for the annotated evidence IDs.
* Hypothesis/experiment: use source-preserving, exact-ID retrieval as the
  first gate, then reader/grounding as a separate gate.  Compare raw,
  metadata-key, dense, sparse, graph, fusion, and rerank conditions with
  held-out conversations and paired confidence intervals.
* Status: measured for a repository-local conventional sparse baseline:
  the checked-out scripts report exact evidence IDs and zero provider calls.
  Five conversation-split seeds gave a mean speaker-prefix Recall@10 lift of
  0.029661 over raw BM25S (all five positive; ten conversations overlap across
  seeds).  A held-out post-hoc router collapsed to unconditional prefixing on
  960/966 pre-exclusion questions and produced identical quality at a 2.043x
  versus 1.043x sparse-index token footprint, while two-field RRF produced a
  smaller lift.  The router and two-field RRF are therefore rejected for this
  dataset; the simple prefixed field remains exploratory pending another
  dataset.  It is not an external LoCoMo leaderboard or answer-quality result.

### RAGBench

* Mechanism: RAGBench is an explainable RAG evaluation framework/dataset
  designed to attribute retrieval context quality, generation quality, and
  faithfulness rather than collapsing them into one answer score.
* Primary evidence: [RAGBench, arXiv:2407.11005](https://arxiv.org/abs/2407.11005)
  evaluates its defined context/answer dimensions on the paper’s RAG
  datasets and models.  The reported metrics are benchmark-specific and
  do not establish exact source identity or temporal correctness.
* Constraints: annotate claim/evidence relationships, distinguish retrieval
  failure from reader failure, and pin judge/annotation policy.  License and
  annotation access must be checked before importing data.
* Hypothesis/experiment: decomposed grounding metrics will identify whether a
  candidate or reranker change helps evidence access or merely changes prose.
  Run fixed-reader closed/retrieved/oracle-context arms after exact-ID
  retrieval, reporting unsupported claims and citation precision/recall.
* Status: researched; planned.

### ARES

* Mechanism: ARES uses synthetic training plus lightweight LM judges to
  evaluate context relevance, answer faithfulness, and answer relevance, with
  prediction-powered inference using a small human-labeled calibration set.
* Primary evidence: [ARES, arXiv:2311.09476](https://arxiv.org/abs/2311.09476)
  and the [official repository](https://github.com/stanford-futuredata/ARES)
  evaluate the three dimensions on eight tasks drawn from KILT,
  SuperGLUE, and attribution/statement settings; the paper’s few-hundred
  annotation calibration and confidence intervals are part of its protocol.
  It is an evaluator result, not evidence that an LLM judge is ground truth.
* Constraints: synthetic judge training, calibration labels, model
  contamination, and confidence intervals must be retained.  A judge cannot
  replace exact evidence-ID retrieval or a deterministic failure ledger.
* Hypothesis/experiment: ARES-like attribution checks can expose grounding
  regressions after candidate changes.  Calibrate a fixed offline judge only
  after exact-ID metrics, and compare reader outputs against oracle evidence,
  retrieved evidence, and closed-book controls.
* Status: researched; planned; no ARES judge result is accepted locally.

### Reciprocal Rank Fusion (RRF)

* Mechanism: RRF adds rank-based contributions from independent lists,
  typically 1 divided by a smoothing constant plus rank, avoiding score-scale
  calibration.  It is a fusion baseline, not a retrieval model.
* Primary evidence: [Cormack, Clarke, and Buettcher,
  SIGIR 2009](https://cormack.uwaterloo.ca/cormacksigir09-rrf) evaluates
  RRF against individual and Condorcet-style combinations on the paper’s
  LETOR/TREC-style runs.  Those list-fusion metrics are not memory recall.
* Constraints: channel identity, duplicate IDs, rank starts, candidate
  budgets, missing channels, and smoothing k must be recorded.  Weight
  routing must not silently override explicit settings.
* Hypothesis/experiment: RRF may be robust to dense/sparse/graph score
  scales, but calibrated linear or union may win after equal budgets.  Run
  fixed k=60, swept k, union, and calibrated-linear arms with per-channel
  oracle recall and paired exact-ID outcomes.
* Status: measured as HybridMind’s current default fusion control (weighted
  RRF k=60).  In the held-out two-sparse-field experiment, RRF improved
  Recall@10 by 0.015825 over raw but was dominated by the single
  speaker-prefix field at 0.026630, while requiring both indexes.  This
  rejects that particular two-field fusion, not RRF across independent dense,
  sparse, and graph channels; the broader fusion hypothesis remains open.

### HyDE

* Mechanism: HyDE asks an instruction-following LLM to generate a
  hypothetical document, embeds that generated text, and retrieves real
  documents by vector similarity.  The hypothetical text is a query
  representation, not evidence.
* Primary evidence: [HyDE paper](https://boston.lti.cs.cmu.edu/luyug/HyDE/HyDE.pdf)
  and the [official repository](https://github.com/texttron/hyde) evaluate
  zero-shot retrieval across the paper’s heterogeneous datasets using their
  dense retriever and nDCG/recall-style metrics.  The LLM, prompt, and
  encoder are part of the reported condition; no conversational source-ID
  result transfers.
* Constraints: generation adds provider calls/cost and can hallucinate
  query drift, leak scope, or be nondeterministic.  Offline testing needs a
  pinned local model or a deterministic test double; production calls need
  the repository’s priced-plan/preflight policy.
* Hypothesis/experiment: hypothetical prose may bridge vocabulary mismatch
  for weak-overlap questions but can introduce distractors.  Compare direct
  query versus HyDE under fixed encoder/candidate budgets, with source scope,
  exact-ID recall, no-answer leakage, token/cost, and p50/p95/p99 recorded.
* Status: researched; planned; no provider call authorized for this ledger.

### RAG-Fusion

* Mechanism: RAG-Fusion generates multiple query rewrites, retrieves each
  independently, and combines the ranked lists, usually with RRF.  Its
  causal unit is query diversification plus fusion.
* Primary evidence: [RAG-Fusion, arXiv:2402.03367](https://arxiv.org/abs/2402.03367)
  reports manual/answer-level evaluations in the paper’s product-information
  setting and uses multi-query retrieval plus rank fusion.  This is not an
  exact-evidence conversational benchmark, and rewrite count/model/prompts
  affect the result.
* Constraints: rewrite LLM calls add cost and can cross corpus/session scope;
  duplicate and contradictory rewrites require deduplication and complete
  traces.  Third-party orchestration repositories are not primary evidence.
* Hypothesis/experiment: diversified queries may add candidates missed by
  direct search, but a deterministic lexical key expansion may achieve the
  same gain cheaper.  Compare direct, fixed rewrites, and RAG-Fusion at
  equal total candidate/query budgets with exact-ID recall, overlap,
  rewrite cost, and temporal/scope leakage tests.
* Status: researched; planned.

### Query2Doc

* Mechanism: Query2Doc uses few-shot LLM prompting to generate a
  pseudo-document for the query, then expands a sparse or dense query with
  that pseudo-document before retrieval.
* Primary evidence: [Query2Doc, arXiv:2303.07678](https://arxiv.org/abs/2303.07678)
  reports BM25 improvements of roughly 3–15 percentage points in its
  MS MARCO and TREC Deep Learning experiments, with metric and baseline
  varying by table.  The numbers are not transferable to LoCoMo; the
  precise context is the paper’s query-expansion experiment, not a general
  RAG claim.
* Constraints: pseudo-document generation can hallucinate entities/dates,
  incurs LLM calls, and must never be indexed as source evidence.  Prompt,
  model, number of demonstrations, and query token budget must be fixed.
* Hypothesis/experiment: expansion may improve BM25 recall on paraphrases
  while hurting entity/date precision.  Compare direct BM25/BM25S with
  Query2Doc expansion on held-out memory questions, report exact-ID recall,
  candidate ceiling, scope errors, cost, and latency.
* Status: researched; planned.

### BGE-M3

* Mechanism: BGE-M3 is a multilingual encoder family exposing dense,
  learned sparse, and multi-vector representations, intended to support
  multiple retrieval modes from one model.
* Primary evidence: [BGE-M3, arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
  evaluates multilingual dense/sparse/multi-vector retrieval on MIRACL,
  MKQA, and related tasks under its model/token limits.  The
  [FlagEmbedding repository](https://github.com/FlagOpen/FlagEmbedding)
  documents inference and reranking interfaces.  These task scores are not
  conversational exact-ID results.
* Constraints: model output dimensions, tokenizer/language coverage,
  checkpoint/license, GPU memory, and sparse posting density must be pinned.
  Most BGE-M3 outputs do not satisfy HybridMind’s production native 4096-d
  contract; no padding, projection, or truncation fallback is allowed.
* Hypothesis/experiment: its unified sparse/dense/multi-vector outputs could
  reduce representation mismatch, but a native-width policy-compatible
  endpoint is a prerequisite.  Screen model mechanics offline only, then
  compare exact-ID recall/cost against separate controls.
* Status: researched and partially measured.  The pinned MIT-licensed local
  checkpoint was used only for learned sparse and MaxSim; its 1024-d dense
  output was explicitly excluded.  On held-out LoCoMo, learned sparse lost to
  speaker-prefixed BM25S by -0.11678 candidate-pool oracle recall and -0.09070
  pre-rerank Recall@10, with both clustered intervals below zero, so that
  learned-sparse configuration is rejected as a default.  MaxSim produced a
  positive fixed-pool selection lift but retains a large token-vector/storage
  cost.  This does not make the BGE dense output production-compatible.
  Artifact: `experiments/results/offline-locomo-bgem3-mechanisms-20260822.json`.

### E5

* Mechanism: E5 trains text embeddings with contrastive objectives and
  query/passage instruction prefixes, with multilingual and large-model
  variants.  It is a dense representation alternative, not a graph or
  memory lifecycle.
* Primary evidence: [multilingual E5 technical report,
  arXiv:2402.05672](https://arxiv.org/abs/2402.05672) reports retrieval
  metrics on multilingual benchmarks and training/evaluation details; the
  [official Microsoft repository](https://github.com/microsoft/unilm/tree/master/e5)
  provides code/checkpoints.  Dimensions and scores vary by model variant.
* Constraints: prefixing, model dimension, checkpoint/data terms, and
  inference hardware must be fixed.  The common E5 dimensions do not meet
  the native 4096-d production contract; projection/padding is prohibited.
* Hypothesis/experiment: E5 may improve dense candidate recall on selected
  languages, but any gain must survive exact-ID memory evaluation at a
  compatible native dimension.
* Status: researched; planned for screening; rejected for production unless
  a policy-compatible native 4096-d variant is verified.

### Nomic Embed

* Mechanism: Nomic Embed trains open embedding models with long context and
  contrastive data; the representation is a single dense vector used for
  ANN retrieval.
* Primary evidence: [Nomic Embed, arXiv:2402.01613](https://arxiv.org/abs/2402.01613)
  reports short-context MTEB and long-context retrieval evaluations under
  its model variants.  The [official Contrastors repository](https://github.com/nomic-ai/contrastors)
  documents training/code openness.  The reported model dimensions (for
  example, 768 in the cited v1 family) are variant-specific and do not meet
  HybridMind’s 4096-d rule.
* Constraints: model/data/checkpoint license, long-context truncation,
  tokenizer, and dimension must be pinned.  A smaller vector cannot be
  silently padded or projected.
* Hypothesis/experiment: long-context training might improve session-level
  representation, but exact source recall and native-width compatibility
  are required.  Compare only a verified native-compatible endpoint against
  the current dense control on held-out sessions.
* Status: researched; rejected for current production width absent a
  compatible variant; planned only as a screened model.

### GTE

* Mechanism: GTE is an encoder family trained for general text/code
  embeddings using contrastive and multi-stage data; variants trade parameter
  size, context, and dimension.
* Primary evidence: [GTE paper, arXiv:2308.03281](https://arxiv.org/abs/2308.03281)
  reports MTEB and text/code retrieval comparisons for its stated variants.
  The [Alibaba-NLP official repository](https://github.com/Alibaba-NLP/gte)
  and [official model card](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)
  state per-variant dimensions and evaluation context.  Model-card MTEB
  numbers are not evidence for memory retrieval.
* Constraints: dimensions vary; the tokenizer, context window, model license,
  and local/remote policy must be pinned.  Only a native exactly-4096-d
  endpoint is admissible in production.
* Hypothesis/experiment: a compatible GTE variant may alter dense candidate
  recall; compare it to the current model with identical corpus, query,
  exact-ID, and ANN conditions, then measure resource cost.
* Status: researched; planned for compatibility screening.

### BGE rerankers

* Mechanism: BGE rerankers are query-passage cross-encoders that score a
  bounded candidate set jointly, replacing independent vector similarity
  ordering at the selection stage.
* Primary evidence: [FlagEmbedding reranking
  repository](https://github.com/FlagOpen/FlagEmbedding) and the
  [official reranking documentation](https://bge-model.com/tutorial/5_Reranking/5.2.html)
  describe pair scoring, model variants, and accuracy/efficiency tradeoffs.
  Their reported benchmark behavior is cross-encoder IR evaluation, not
  evidence-grounded conversational QA.
* Constraints: model/checkpoint license, token limits, batching, CPU/GPU,
  and candidate pool size determine cost.  Reranking cannot recover a gold
  source absent from the candidate pool; execution evidence is mandatory.
* Hypothesis/experiment: BGE cross-encoding may reorder an existing pool
  better than RRF/lexical scoring.  Freeze candidate IDs, compare no-op,
  lexical, BGE, and shuffled controls at several pools, and report pre/post
  exact-ID rank, execution traces, p50/p95/p99, and tokens.
* Status: researched; cross-encoder selection mechanism measured with a
  different model, BGE-specific result still open.  A pinned local MiniLM
  cross-encoder raised held-out Recall@10 by +0.04262 on fixed
  speaker-prefixed BM25S pools, interval [+0.01299, +0.06643], while MRR was
  inconclusive and multi-hop Recall@10 fell by -0.09127, interval
  [-0.13158, -0.03101].  Mean CPU time was 287.06 ms per 25-document pool.
  This establishes that cross-encoding can help selection but rejects an
  unconditional default; it does not establish a BGE reranker result.
  Artifact: `experiments/results/offline-locomo-rerank-minilm-20260822.json`.

### RankLLM

* Mechanism: RankLLM wraps pointwise, pairwise, and listwise LLM ranking
  prompts for reranking candidate passages.  It is an expensive selection
  controller, not a candidate generator.
* Primary evidence: [RankLLM, arXiv:2505.19284](https://arxiv.org/abs/2505.19284)
  and the [official repository](https://github.com/castorini/rank_llm)
  describe reproducible MS MARCO ranking workflows and open/proprietary
  model support.  Reported ranking metrics are paper-specific; they do not
  prove grounded memory answers.
* Constraints: API/local model identity, prompt format, list size, token
  budget, nondeterminism, and rate/cost limits must be recorded.  Provider
  calls require the repository’s priced plan and preflight; offline tests
  must use a deterministic local double or no-op.
* Hypothesis/experiment: listwise LLM judgment may improve ordering on
  ambiguous pools but may lose to cross-encoders on latency and stability.
  Run fixed candidate-list replay with a deterministic mock first, then a
  bounded approved model; compare exact-ID rank, pair/list cost, variance,
  and unsupported-answer rate.
* Status: researched; planned; no provider call made.

### FlashRank

* Mechanism: FlashRank is a lightweight local reranking package that uses
  compact ONNX/cross-encoder-style models for bounded candidate scoring, with
  listwise/pairwise support in its implementation lineage.
* Primary evidence: the [official FlashRank repository](https://github.com/PrithivirajDamodaran/FlashRank)
  documents its local model/ONNX orientation and reranker API.  Repository
  speed claims are implementation benchmarks and must be re-run with a
  pinned model, runtime, CPU, and candidate count; no paper metric is
  treated as memory evidence.
* Constraints: model download/license, ONNX runtime, quantization, token
  limits, and available CPU instructions affect reproducibility.  A small
  reranker still cannot fix candidate recall or provenance.
* Hypothesis/experiment: a local compact reranker may offer a better
  latency/cost frontier than a large cross-encoder.  Replay fixed pools with
  no-op, lexical, FlashRank, and BGE controls; measure exact-ID rank,
  execution proof, p50/p95/p99, RSS, and model bytes without gold access in
  timing inputs.
* Status: researched; planned.  No FlashRank run is measured locally.

### Milvus

* Mechanism: Milvus is a distributed/cloud-native vector database with dense
  ANN, scalar filtering, full-text/BM25 and learned sparse support, and
  multi-vector/hybrid reranking.  Milvus Lite provides a local file option;
  the distributed path has separate compute/storage and segment lifecycle.
* Primary evidence: the [official repository](https://github.com/milvus-io/milvus)
  is Apache-2.0 and documents dense/sparse hybrid, BM25, filtering, and
  local/distributed deployment.  The cited [Milvus system paper](https://dl.acm.org/doi/10.1145/3406526.3458511)
  reports system throughput/scale on its own workloads; those numbers do not
  transfer to HybridMind.
* Constraints: server/segment consistency, collection schema, scalar
  indexes, persistence, network, and operational dependencies must be
  measured.  SQLite commit and Milvus visibility need generation attestation.
* Hypothesis/experiment: Milvus may dominate at scale or hybrid filtering,
  but its operational complexity may lose locally.  Replay the same corpus
  and manifest against Milvus Lite first, then a self-hosted server; compare
  exact-ID recall, filter/temporal correctness, freshness, bytes, RSS,
  p50/p95/p99, and failure recovery.
* Status: researched; planned scale challenger.

### Qdrant

* Mechanism: Qdrant stores named dense/sparse vectors, supports HNSW,
  payload indexes/filters, and a Query API that can prefetch multiple
  channels and combine them with RRF, score fusion, or reranking.
* Primary evidence: the [official repository](https://github.com/qdrant/qdrant)
  is Apache-2.0; [hybrid query documentation](https://qdrant.tech/documentation/concepts/hybrid-queries/)
  specifies multi-prefetch and fusion; [filter documentation](https://qdrant.tech/documentation/concepts/filtering/)
  specifies payload filtering.  These docs establish mechanisms, not
  conversational quality.
* Constraints: service or embedded deployment, payload indexing, segment
  compaction, snapshots, and client/server versions matter.  Qdrant points
  and payloads are derived copies until SQLite commit/generation checks pass.
* Hypothesis/experiment: Qdrant’s named-vector and prefetch pipeline may
  simplify dense+sparse candidate experiments and filters, but not improve
  evidence quality by itself.  Run equal-budget dense/sparse/hybrid arms
  against in-process controls with scope, temporal, update, snapshot, and
  p50/p95/p99 checks.
* Status: researched; planned adapter.

### LanceDB

* Mechanism: LanceDB is an embedded/table-oriented vector store built on
  Lance.  Its hybrid search combines vector and BM25 full-text queries, with
  RRF as the documented default reranker and optional cross-encoder/custom
  rerankers; prefilter/postfilter are explicit controls.
* Primary evidence: [LanceDB hybrid-search documentation](https://docs.lancedb.com/search/hybrid-search)
  documents vector+FTS, RRF reranking, row IDs, distance bounds, and
  prefilter/postfilter.  [Full-text documentation](https://docs.lancedb.com/search/full-text-search)
  specifies BM25.  The [OSS FAQ](https://docs.lancedb.com/faq/faq-oss) states
  Apache-2.0 for LanceDB OSS and distinguishes the commercial enterprise
  product.  Documentation examples are not quality benchmarks.
* Constraints: Lance table/schema, versioned files, compaction, FTS index
  refresh, row-ID joins, and OSS/enterprise split must be pinned.  Metadata
  filter ordering can change candidate recall, so report it explicitly.
* Hypothesis/experiment: an embedded columnar table may improve local
  persistence and hybrid I/O, but it may not beat a simple SQLite plus
  derived-index baseline at current scale.  Export identical source IDs and
  vectors, compare vector/FTS/hybrid and RRF with filter/update/recovery and
  exact-ID/latency/resource metrics.
* Status: researched; planned.

### Elasticsearch / OpenSearch

* Mechanism: both provide mature inverted indexes, BM25, filters, vector
  fields/ANN, and hybrid queries that combine lexical and vector scores.
  Elasticsearch documents RRF as the recommended hybrid fusion; OpenSearch
  uses a hybrid query and search pipelines to combine shard-level scores.
* Primary evidence: [Elasticsearch hybrid search documentation](https://www.elastic.co/docs/solutions/search/hybrid-search)
  and [OpenSearch hybrid query documentation](https://docs.opensearch.org/latest/query-dsl/compound/hybrid/)
  define the mechanisms and shard/filter behavior.  [OpenSearch’s FAQ](https://opensearch.org/faq/)
  states its Apache-2.0 fork lineage.  Elasticsearch’s current license and
  subscription terms must be checked from [official license
  information](https://www.elastic.co/licensing/elastic-license); newer
  versions are not interchangeable with the historical Apache-2.0 release.
* Constraints: JVM/cluster operations, shards, refresh/segment merges,
  network, security, and license choice dominate local cost.  Elasticsearch
  and OpenSearch are separate products; benchmark one pinned version, not a
  blended label.  Source IDs, generation, filters, and update visibility
  require end-to-end receipts.
* Hypothesis/experiment: mature postings and distributed filtering may
  improve scale/operability, not necessarily quality.  Compare each product
  separately with BM25/dense/RRF controls at equal corpus, query, top-k,
  filter, and update workloads; report exact evidence recall, p50/p95/p99,
  refresh lag, index bytes, resource cost, and recovery.
* Status: researched; planned as two separate comparative adapters; no
  license decision made.

## Cross-cutting synthesis

### Candidate recall

The highest-information quality question is whether a gold source enters the
candidate pool.  Conventional BM25/BM25S and Tantivy/Lucene are transparent
controls.  SPLADE++ and BGE-M3 sparse outputs test learned expansion; ColBERT
and multi-vector paths test token-level matching; BGE-M3, E5, Nomic Embed, and
GTE test dense representation; HyDE, Query2Doc, and RAG-Fusion test query
transformation.  FAISS, ScaNN, USearch, DiskANN, TurboQuant, and Vespa
primarily change the systems/ANN axis, so they must be compared against an
exact Flat oracle at identical vectors.  A method that does not add
source-ID candidate recall cannot be credited for reader quality.

### Reranking and selection

RRF, calibrated fusion, lexical selection, BGE rerankers, FlashRank,
ColBERT/PLAID late interaction, and RankLLM act after candidate generation
unless explicitly tested as independent channels.  A reranker can improve
rank only when the candidate pool contains the source.  Every comparison
therefore reports pre-rerank candidate ceiling, pre/post exact-ID ranks,
pool-size curves, a no-op/shuffled control, proof that the stage executed,
and latency/token cost.

### Memory model

RAPTOR and GraphRAG introduce hierarchical/community summaries; LightRAG and
HippoRAG introduce graph-mediated retrieval; Graphiti/Zep emphasizes
source-linked temporal facts and incremental updates; Mem0, Letta/MemGPT, and
A-MEM add policies for memory creation, evolution, paging, or linking.
These mechanisms alter episodic, semantic, and temporal representation, not
just search.  Raw source turns remain the fidelity control.  Derived facts,
notes, summaries, communities, and graph edges must point back to source IDs,
retain validity and assertion time, and be evaluated for stale/future leakage,
contradiction, deletion, and provenance.

### Systems and resource frontier

FAISS Flat/HNSW, ScaNN, USearch, DiskANN/FreshDiskANN, TurboQuant/Turbovec,
Vespa, Milvus, Qdrant, LanceDB, and Elasticsearch/OpenSearch can alter
latency, memory, persistence, filtering, or freshness without improving
semantic quality.  Use exact vectors and synthetic deterministic embeddings
for mechanics, then native production embeddings for semantic claims.
Report p50/p95/p99, cold/warm strata, build and update time, RSS, bytes,
index lag, failures, and crash/rebuild equivalence.  Do not select a backend
from one-shot sub-millisecond timings or incomparable vendor benchmarks.

### Evaluation and grounding

MTEB and BEIR are model/retriever screens.  LoCoMo and LongMemEval test
long-term conversation, time, updates, sessions, and abstention; RAGBench
and ARES separate retrieval/context/answer dimensions.  SRA-Bench and
SkillCorpus are methodological reminders to separate retrieval,
incorporation/use, and downstream task performance.  None removes the need
for an exact-evidence ledger.  The repository-local LoCoMo sparse result is
measured only as a conventional, zero-provider control; no paper number in
this document is a HybridMind result.

## Shortlist ordered by expected information gain

1. **Correctness and oracle gate (measured baseline, then repair):** lock
   corpus/session manifests, source IDs, temporal filters, execution traces,
   exact Flat/SQLite oracle, BM25/BM25S, FAISS HNSW, and RRF k=60.  This
   identifies candidate ceilings and prevents false component claims.
2. **Transparent sparse frontier:** compare raw/speaker/date/fact keys in
   BM25S/Tantivy/Lucene on held-out conversations.  This is cheap, offline,
   interpretable, and can falsify the assumption that a learned sparse model
   is needed.
3. **Learned sparse recall:** run SPLADE++ (and BGE-M3 sparse only if its
   model/checkpoint policy passes) with equal posting/candidate budgets.
   Promote only source-ID recall gains after encoding and query-cost accounting.
4. **Late-interaction selection:** replay fixed pools with ColBERTv2/PLAID
   against lexical and BGE cross-encoder controls; only then test independent
   late-interaction candidate generation.
5. **Temporal/associative graph:** use a graph sham, typed traversal,
   HippoRAG-like PPR, and Graphiti-inspired bi-temporal provenance on
   adversarial updates and held-out multi-hop questions.
6. **Representation screen:** use MTEB/BEIR only to narrow candidates, then
   test E5/Nomic/GTE/BGE-M3 with exact-ID memory metrics and native-width
   policy checks.  Matryoshka is admitted only with a trained native model.
7. **Query transformation:** compare deterministic lexical expansion with
   Query2Doc, HyDE, and RAG-Fusion at equal total query/candidate budgets,
   after offline provider-safe controls exist.
8. **ANN/quantization scale gate:** FAISS Flat/HNSW/PQ, ScaNN, USearch,
   TurboQuant/Turbovec, and DiskANN/FreshDiskANN under staged mutation and
   resource admission.  Replace the backend only when quality, freshness,
   recovery, and cost all generalize.
9. **External serving challengers:** Vespa, Qdrant, Milvus, LanceDB, and
   separately Elasticsearch/OpenSearch, with generation-attested replication
   and recovery.  Operational wins count only if exact evidence and filters
   remain correct.
10. **Lossy/agentic memory policies:** RAPTOR, LightRAG, GraphRAG, Mem0,
    Letta/MemGPT, and A-MEM after source-preserving retrieval gates.  These
    are valuable only if summary/paging/linking gains survive temporal,
    provenance, cost, and abstention tests.

## Uncertainty and conflicts

The strongest published results frequently measure answer quality, nDCG,
latency, or storage on a different corpus, model, reader, hardware target, or
candidate depth.  Several repositories contain newer code than their papers;
official README claims are not peer-reviewed evidence.  Model checkpoints,
datasets, hosted services, and transitive dependencies can carry different
licenses from the source code.  Graphiti’s open-source implementation and
Zep’s managed product must not be treated as one license or one mechanism.
Tantivy and Lucene share BM25 design but differ in language/runtime and
operational scope.  Elasticsearch and OpenSearch must be benchmarked as
separate, version-pinned products because their current licensing and
feature sets diverge.  USearch fork/release lineage and ScaNN build
requirements need a revision-level audit.  TurboQuant’s theory,
Turbovec’s implementation, and any GPU speed claim are separate evidence
objects.

No item above establishes a 70–80% prompt-source substitution result, a
10M–100M context result, or a production semantic-quality win.  Those claims
require fresh, execution-attested, exact-evidence experiments under the
repository research protocol.
