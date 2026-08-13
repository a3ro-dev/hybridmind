# Retrieval-conditioned effective context research protocol

Status: preregistered engineering protocol; no success claim is implied by this document.

## 1. Scope and terminology

HybridMind is an external memory and retrieval system. It can avoid placing every stored
source token into a model prompt, but it does not reproduce the per-layer key/value states
that a transformer creates while prefilling or decoding. Consequently:

- **retrieval-conditioned effective context** means the number of stored source tokens over
  which a system can answer a defined workload while sending only a bounded subset to the
  reader model;
- **model context window** remains the maximum token sequence directly processed by the
  reader;
- **KV-cache replacement** is reserved for a mechanism that preserves or reconstructs the
  attention state used during autoregressive decoding.

The 10M--100M-token goal is evaluated as retrieval-conditioned effective context unless a
future implementation integrates with the model's attention/KV path.

## 2. Primary hypothesis

For a fixed reader model and a corpus containing at least 10 million source tokens,
HybridMind can reduce source tokens placed in the prompt by 70--80% relative to a defined
full-context or oracle-context baseline while retaining at least 95% of the baseline's
downstream task score.

This hypothesis is rejected for a workload if any of the following holds:

1. exact supporting-evidence recall is below 95% at the declared token budget;
2. downstream score is below 95% of the paired oracle-evidence score;
3. the catastrophic evidence-miss rate exceeds 1%;
4. failures, timeouts, malformed outputs, or unavailable providers are omitted from the
   denominator;
5. the claimed ablation cannot be reproduced from an immutable run manifest;
6. the result depends on global-corpus leakage, answer-string relevance, or benchmark labels
   unavailable to the retriever.

The 100M-token moonshot is evaluated only after the 10M-token gate passes. Corpus capacity
alone is not success.

## 2.1 Two distinct research tracks

The project must not merge two technically different objectives into one claim.

### Track A: evidence-memory substitution

HybridMind indexes source records outside the reader, retrieves a bounded evidence set for each
question, and sends that evidence through the reader's ordinary prompt interface. Success means
that most historical source tokens can remain outside the prompt while paired answer quality stays
near an oracle/full-context baseline. LoCoMo, LongMemEval, LongMemEval-V2, RULER-derived
context-gathering tasks, BRIGHT, and exact-support multi-hop tasks evaluate this track.

### Track B: model-integrated sparse attention/KV retrieval

An actual KV-cache replacement must run inside a controllable model-serving stack. It must index
or compress per-layer/per-head key and value states, select entries using the current attention
query, preserve attention sinks/recent tokens where required, and merge sparse attention outputs
with correct softmax normalization. RetrievalAttention additionally shows that ordinary ANNS over
key vectors is insufficient because attention queries and keys are out of distribution; its index is
constructed with query-to-key structure. Quest and PyramidKV likewise make query- or layer-aware
decisions that an HTTP text retriever cannot reproduce.

Track B therefore requires a separate vLLM/Transformers integration, model-internal traces, and
full-attention parity tests. No Track A metric may be reported as Track B progress. The tracks can
eventually compose: Track A selects source blocks before prefill, while Track B sparsifies the KV
state of the selected prompt during decoding.

## 3. Metrics

Every query records both retrieval and reader outcomes.

### Retrieval metrics

- exact evidence Recall@k and Recall@token-budget;
- all-evidence coverage for multi-hop questions;
- MRR and nDCG using stable evidence IDs;
- catastrophic miss rate: fraction with zero required evidence retrieved;
- distractor rate and cross-session leakage rate;
- temporal-version accuracy for latest, previous, before, after, and historical queries;
- contradiction/update accuracy and required abstention accuracy;
- pre-rerank and post-rerank evidence ranks, plus proof that the reranker executed.

### Reader metrics

- exact match/F1 where the dataset supplies deterministic labels;
- benchmark-native score;
- citation/evidence faithfulness;
- abstention precision and recall;
- paired oracle-context gap;
- paired full-context gap when the reader can accept the complete source.

### Systems metrics

- indexed source tokens and bytes;
- retrieved unique source tokens and prompt tokens;
- retrieval-conditioned context multiplier:

  `indexed_source_tokens / retrieved_source_tokens`

- source-token reduction relative to the paired baseline;
- p50/p95/p99 ingestion, retrieval, reranking, and end-to-end latency;
- peak resident memory, index bytes per source token, and build/rebuild time;
- embedding, reranking, and reader requests/tokens/cost;
- explicit counts for success, wrong answer, abstention, timeout, unavailable provider,
  malformed response, and internal error.

Ratios are reported with their numerator and denominator. Failed queries remain in every
applicable denominator.

### Resource, latency, and token-economics acceptance criteria

Quality is not optimized independently of operating cost. Every configuration is placed on a
quality/latency/resource Pareto frontier and compared at the same corpus, query set, reader,
and hardware profile. Report at minimum:

- persistent index bytes and peak resident bytes per indexed source token;
- CPU/GPU model, accelerator memory, build time, and incremental-ingest throughput;
- retrieval and end-to-end p50/p95/p99 latency, including cold-start and timeout counts;
- unique evidence tokens retrieved, duplicate tokens removed, reader input/output tokens, and
  provider requests per query;
- embedding, reranking, decomposition, and reader cost separately, plus total cost per query and
  cost per correct, evidence-grounded answer;
- quality gained per additional millisecond, megabyte, reader token, and dollar relative to the
  next-cheaper configuration.

A more complex method is rejected as dominated when its uncertainty interval shows no material
quality gain while it increases any constrained resource substantially. Before a live run, the
run manifest must fix maximum queries, embedding texts, reranker pairs, LLM input/output tokens,
wall time, and estimated spend. The client must enforce those limits and stop rather than exceed
them. A result without a machine profile and these denominators is not a scalability or economics
result.

### Capacity model and scale gates

The exact 4096-dimensional float32 contract costs 16,384 bytes per stored dense vector before
FAISS/HNSW links, IDs, SQLite rows, BM25 postings, graph edges, allocator overhead, or duplicate
sentence/parent representations. For `T` source tokens and mean `C` tokens per independently
embedded chunk, the raw-vector lower bound is:

`raw_vector_bytes = ceil(T / C) * 4096 * 4`

At a purely illustrative 256 tokens/chunk, this lower bound is about 0.64 GB at 10M tokens,
2.56 GB at 40M, and 6.40 GB at 100M. These are not memory forecasts; the measured total can be
substantially larger. Every scale report must show actual chunk-count distribution, duplicate
representations, raw vector bytes, total index bytes, peak RSS, and graph/BM25/SQLite components.

The progression is 1M -> 10M -> 40M -> 100M source tokens. A stage advances only when its
quality gate passes and p95 latency, peak memory, index size, rebuild time, and cost remain under
the preregistered machine-specific ceilings. The current all-in-memory HNSW design is not assumed
to pass 100M. Disk-resident ANN, quantization, or partitioned incremental indexing are separate
experimental backends and must be evaluated for exact-evidence recall loss against flat/exact or
high-recall ground truth.

## 4. Workloads

No single benchmark can establish the goal. The minimum matrix is:

| Workload | Property tested |
|---|---|
| LoCoMo | long conversational history, temporal and causal questions |
| LongMemEval | extraction, multi-session reasoning, temporal reasoning, updates, abstention |
| LongMemEval-V2 | context gathering over agent trajectories scaling to 115M source tokens |
| MultiHop-RAG or MuSiQue with exact supporting IDs | multi-evidence retrieval |
| RULER-derived external-memory tasks | multiple needles, tracing, aggregation, length scaling |
| BRIGHT | reasoning-intensive retrieval with weak lexical overlap |
| PersistBench-compatible cases | inappropriate recall, cross-domain leakage, safe forgetting |
| deterministic adversarial suite | duplicates, contradictions, stale versions, paraphrases, distractors, malformed timestamps |

All conversational evaluations are scoped to the conversation unless the workload explicitly
tests cross-conversation retrieval. Gold answers and evidence labels are never indexed.

## 5. Baselines and ablations

Each condition runs against the same immutable corpus snapshot and query set:

1. no retrieval / closed book;
2. BM25 only;
3. dense embedding only;
4. dense + BM25 RRF;
5. graph only with explicit anchors;
6. dense + graph;
7. dense + BM25 + graph;
8. hybrid + cross-encoder;
9. hybrid + cross-encoder + query expansion/decomposition;
10. oracle supporting evidence;
11. full context when supported by the reader.

Signal ablations must disable every non-target candidate generator and every non-target
reordering stage. A run is invalid unless the server reports the resolved runtime settings,
executed stages, model identities, and corpus generation in each query trace.

## 6. Statistical protocol

- Predeclare dataset revision, sample selection, query IDs, seeds, configurations, and primary
  metrics before running paid inference.
- Use paired bootstrap confidence intervals for score differences and a paired randomization or
  permutation test for the primary comparison.
- Report per-question results and category slices, not only macro means.
- Correct for multiple comparisons when choosing among more than one experimental variant.
- Never tune on the final test partition; use a development partition for thresholds and weights.
- Re-run deterministic offline retrieval twice and require identical ordered IDs and scores.
- For remote readers, record response hashes and quantify repeated-run variance on a small fixed
  subset instead of claiming bit-identical behavior.

## 7. Cost-controlled execution gates

Paid or serverless inference is permitted only in this order:

1. all offline unit, property, persistence, corruption, and evaluator tests pass;
2. deterministic synthetic retrieval and ablation runs pass;
3. `scripts/preflight.py --plan <validated-live-plan.json>` warms and validates
   only the intended endpoints after binding to a fresh offline resource report;
4. three-query smoke run with saved request counts and a hard timeout;
5. small stratified development run;
6. one bounded confirmatory run using the preregistered configuration.

Any provider failure stops the run. No local, lower-dimensional, padded, projected, or alternate
provider fallback is permitted. The confirmatory run must have an explicit maximum query count,
maximum provider requests, and estimated spend before launch.

## 8. Design implications from prior work

- RETRO demonstrates that retrieval can expose a model to a datastore much larger than its direct
  input, but does so through a model trained with chunked cross-attention; ordinary prompt RAG is
  not equivalent.
- LongMemEval motivates session decomposition, fact-augmented indexing, time-aware query
  expansion, knowledge-update tests, and abstention measurement.
- LoCoMo requires conversation-level provenance and exact temporal/causal grounding.
- RULER shows that one-needle tests are insufficient; multi-key tracing and aggregation are
  mandatory.
- Lost in the Middle requires evidence-order and position controls in reader evaluation.
- ColBERTv2/PLAID motivate a late-interaction recall/reranking track after the correctness gates.
- HippoRAG/HippoRAG 2 motivate entity-to-passage graph propagation such as Personalized
  PageRank, evaluated against strong dense and sparse baselines rather than assumed useful.
- RAPTOR and GraphRAG motivate hierarchical representations for holistic questions, but lossy
  summaries must retain source provenance and cannot replace exact evidence.
- H2O, SnapKV, PyramidKV, Quest, and RetrievalAttention operate on attention/KV state; they are
  relevant to the model-integrated Track B, not evidence that an HTTP retrieval service replaces
  KV cache. RetrievalAttention's query/key distribution mismatch is direct evidence against
  reusing an ordinary semantic HNSW index as a drop-in attention index.
- Memorizing Transformers and Landmark Attention demonstrate model-integrated random-access
  memory designs, but both change the model or its training/inference path. They motivate Track B
  experiments rather than validating prompt-level RAG.
- LongMemEval-V2 directly tests context gathering over histories as large as 115M tokens and
  reports an accuracy/latency frontier; it is the closest current primary benchmark for the
  100M-token Track A moonshot, but remains work in progress and cannot be the only workload.
- HNSW motivates a controllable recall/latency graph index, but build order and randomized levels
  require an exact-search comparison and repeated builds rather than assumed deterministic recall.
- DiskANN shows that SSD-resident ANN can change the RAM/scale frontier; it motivates a future
  disk-backed track instead of pretending an in-memory index scales indefinitely.
- SPFresh shows why rebuild-based dynamic indexes can exhibit latency/accuracy/resource spikes;
  update throughput and rebuild amplification are therefore first-class measurements.

## 9. Primary sources

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020:
  https://arxiv.org/abs/2005.11401
- Borgeaud et al., *Improving language models by retrieving from trillions of tokens*, 2021:
  https://arxiv.org/abs/2112.04426
- Liu et al., *Lost in the Middle*, 2023: https://arxiv.org/abs/2307.03172
- Zhang et al., *H2O: Heavy-Hitter Oracle*, 2023: https://arxiv.org/abs/2306.14048
- Santhanam et al., *ColBERTv2*, 2021: https://arxiv.org/abs/2112.01488
- Santhanam et al., *PLAID*, 2022: https://arxiv.org/abs/2205.09707
- Maharana et al., *Evaluating Very Long-Term Conversational Memory of LLM Agents*, 2024:
  https://arxiv.org/abs/2402.17753
- Hsieh et al., *RULER*, 2024: https://arxiv.org/abs/2404.06654
- Wu et al., *LongMemEval*, 2024: https://arxiv.org/abs/2410.10813
- Tang and Yang, *MultiHop-RAG*, 2024: https://arxiv.org/abs/2401.15391
- Su et al., *BRIGHT*, 2024: https://arxiv.org/abs/2407.12883
- Gutiérrez et al., *HippoRAG*, 2024: https://arxiv.org/abs/2405.14831
- Gutiérrez et al., *HippoRAG 2*, 2025: https://arxiv.org/abs/2502.14802
- Sarthi et al., *RAPTOR*, 2024: https://arxiv.org/abs/2401.18059
- Edge et al., *From Local to Global: A Graph RAG Approach*, 2024:
  https://arxiv.org/abs/2404.16130
- Li et al., *SnapKV*, 2024: https://arxiv.org/abs/2404.14469
- Tang et al., *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*, 2024:
  https://arxiv.org/abs/2406.10774
- Cai et al., *PyramidKV*, 2024: https://arxiv.org/abs/2406.02069
- Liu et al., *RetrievalAttention*, 2024: https://arxiv.org/abs/2409.10516
- Wu et al., *Memorizing Transformers*, 2022: https://arxiv.org/abs/2203.08913
- Mohtashami and Jaggi, *Landmark Attention*, 2023: https://arxiv.org/abs/2305.16300
- Wu et al., *LongMemEval-V2*, 2026: https://arxiv.org/abs/2605.12493
- Pulipaka et al., *PersistBench*, 2026: https://arxiv.org/abs/2602.01146
- Malkov and Yashunin, *Efficient and robust approximate nearest neighbor search using
  Hierarchical Navigable Small World graphs*, 2016: https://arxiv.org/abs/1603.09320
- Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single
  Node*, 2019: https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node
- Xu et al., *SPFresh: Incremental In-Place Update for Billion-Scale Vector Search*, 2023:
  https://dl.acm.org/doi/10.1145/3600006.3613166
