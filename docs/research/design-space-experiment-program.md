# HybridMind architecture-neutral research program

Status: active preregistration and evidence ledger. This document is not a
state-of-the-art claim and does not convert planned experiments into results.

Updated: 2026-08-22

## 1. Objective and fixed constraints

The objective is the best reproducible long-term-memory result, not preservation
of HybridMind's present architecture. A simpler or conventional system replaces
an existing component when it wins on the declared quality, correctness, and
resource objectives. An unsupported repository assumption is a hypothesis, not
a fact.

The following are constraints rather than architectural preferences:

- SQLite/WAL remains the authoritative store.
- Primary mutations remain atomic or compensate by rebuilding every derived
  index.
- Production embeddings remain remote, native, finite, and exactly 4096
  dimensions. No projection, padding, truncation, local fallback, or mixed width
  is permitted.
- Provider identities and credentials remain endpoint-bound.
- Live evaluation remains default-deny behind the offline resource report,
  priced usage-limited plan, SHA-256 binding, and successful preflight.
- Exact evidence IDs, corpus/session scope, immutable manifests, execution
  traces, failed receipts, and complete denominators are mandatory.
- Heavy or experimental components remain opt-in until they win a held-out
  comparison.

## 2. Research tree

```text
                    HYBRIDMIND
                        |
             +----------+----------+
             |                     |
        RETRIEVAL              MEMORY MODEL
             |                     |
     +-------+-------+       +-----+-----+
     |       |       |       |     |     |
   dense   sparse  graph   episodic semantic temporal
     |       |       |       |     |     |
     +-------+-------+-------+-----+-----+
                        |
                  candidate fusion
                        |
                 reranking / selection
                        |
                 evidence grounding
                        |
                  answer generation
```

Four system axes cut across the tree: filtering and scope, mutation and
freshness, persistence and recovery, and resource cost. They are measured
separately so that index engineering is not mistaken for retrieval quality and
answer quality is not mistaken for evidence recall.

## 3. Mechanism design space

### 3.1 Retrieval: dense

| Mechanism | Competing conditions | Primary causal question |
|---|---|---|
| Single-vector representation | current remote 4096-d embedding, exact flat oracle, HNSW | Is semantic candidate recall limited by representation or approximation? |
| Late interaction | ColBERTv2/PLAID-style MaxSim as candidate generator or bounded reranker | Does token-level matching recover evidence missed by one vector per source? |
| Query representation | direct query, HyDE, Query2Doc, multi-query/RAG-Fusion | Does query transformation add relevant candidates without scope leakage or cost domination? |
| Embedding family | BGE-M3, E5, Nomic Embed, GTE when a policy-compatible native 4096-d endpoint exists | Are gains model-specific and reproducible under the production width contract? |
| Nested representations | Matryoshka-trained dimensions only | Can a provider-native nested representation improve the quality/storage frontier? |

Arbitrary truncation of a 4096-d production vector is not a Matryoshka
experiment and is prohibited.

### 3.2 Retrieval: sparse

| Mechanism | Competing conditions | Primary causal question |
|---|---|---|
| Conventional lexical | TF-IDF, pure BM25, BM25S, Lucene, Tantivy | How much exact evidence is recoverable with transparent lexical statistics? |
| Learned sparse | SPLADE++, BGE-M3 sparse | Does learned expansion improve paraphrase/entity recall enough to justify model and index cost? |
| Retrieval key | raw source, speaker prefix, date prefix, speaker/date, fact-augmented key, multi-key | Does metadata in the key improve retrieval without changing source identity? |
| Lexical reranking | query-local lexical score, FlashRank where applicable | Is the gain candidate generation, reordering, or both? |

### 3.3 Retrieval: graph

| Mechanism | Competing conditions | Primary causal question |
|---|---|---|
| Explicit traversal | current typed graph with gold-independent anchors | Does bounded traversal recover multi-hop evidence outside dense/sparse pools? |
| Associative propagation | HippoRAG Personalized PageRank | Does entity-to-passage propagation generalize beyond synthetic relationship cases? |
| Temporal knowledge graph | Graphiti-compatible episodes, validity intervals, provenance, incremental updates | Does temporal graph structure improve update and historical-query correctness? |
| Lightweight graph RAG | LightRAG | Which gain comes from graph construction versus lexical/vector retrieval? |
| Community/global graph | GraphRAG | Do community summaries help holistic questions while preserving exact sources? |
| Hierarchical tree | RAPTOR | Do recursive summaries improve global questions without erasing provenance? |

Graphiti is open-source prior art; Zep's managed product behavior is comparative
evidence, not code to copy. Every graph or summary node remains derived and
links to authoritative source IDs.

### 3.4 Memory model

| Axis | Conditions | Failure modes to expose |
|---|---|---|
| Episodic | raw turn, sentence/semantic chunk, parent-child chunk, full session | boundary misses, duplicates, order loss, cross-session leakage |
| Semantic | extracted fact, profile, entity record, derived summary | extraction error, lossy compression, provenance loss, stale fact |
| Temporal | event time, assertion time, half-open validity interval, supersession chain | future leakage, stale truth, incorrect previous/latest answer |
| Lifecycle | append-only, consolidation, salience/decay, explicit forgetting | inappropriate recall, source deletion, non-deterministic mutation |
| Agent memory policy | Mem0, Letta/MemGPT, A-MEM-inspired memory creation and retrieval | policy cost, uncontrolled writes, ungrounded summaries, unsafe forgetting |

The memory taxonomy can change. Raw episodic memory is the source-preserving
control, not the presumed winner.

### 3.5 Fusion, reranking, grounding, and answering

| Stage | Conditions | Required attribution |
|---|---|---|
| Candidate budget | equal per-channel depth, equal total depth, latency-matched depth | oracle pool recall before fusion |
| Fusion | max/union, linear calibrated score, RRF with fixed and swept `k`, learned fusion | gain after equal candidate budgets |
| Selection | no reranker, BGE cross-encoder, ColBERT MaxSim, RankLLM, FlashRank | pre/post ranks and proof of execution |
| Context assembly | score order, chronological order, dependency order, parent expansion, diversity/dedup | unique source IDs/tokens and evidence coverage |
| Grounding | citation-required, contradiction-aware, unsupported-claim detector, abstention | claim-to-evidence precision and recall |
| Answering | closed book, retrieved context, oracle evidence, full context | paired oracle gap and unsupported claims |

Reranking cannot repair absent candidates. Every reranker experiment therefore
reports candidate-pool oracle recall and final recall separately.

### 3.6 Systems and index backends

| Mechanism | Named systems | Question |
|---|---|---|
| Exact and in-memory ANN | FAISS Flat/HNSW/PQ/SQ, ScaNN, USearch | What recall/latency/memory loss is caused by approximation or quantization? |
| Quantized 4096-d ANN | TurboQuant; Turbovec as an independently reproduced implementation | Does compression move the host frontier without unacceptable evidence-rank loss? |
| Disk and mutable ANN | DiskANN, FreshDiskANN/SPFresh | Can a rebuildable derived index remain fresh and recoverable under sustained mutation? |
| Full retrieval servers | Vespa, Milvus, Qdrant, Elasticsearch/OpenSearch, LanceDB | Does a full external engine beat in-process controls enough to justify synchronization complexity? |
| Serious lexical engines | Lucene, Tantivy, Vespa BM25, Elasticsearch/OpenSearch | Do mature filters/postings improve quality or only scale and operability? |

SQLite remains authoritative even if every current derived retrieval component
is replaced.

## 4. Prior-art coverage ledger

### Tier S: mandatory mechanism investigations

| System or method | Mechanism under test | First admissible experiment |
|---|---|---|
| TurboQuant / Turbovec | data-oblivious rotation and vector quantization | exact-vs-quantized neighbor and evidence recall at fixed memory |
| Vespa | lexical+dense candidates, filters, phased ranking | external comparative backend with corpus-generation attestation |
| ColBERTv2 / PLAID | compressed token-level late interaction | bounded reranker, then independent candidate path |
| SPLADE++ | learned sparse expansion | source-identical BM25S vs SPLADE exact-evidence comparison |
| DiskANN / FreshDiskANN | SSD ANN and dynamic freshness | update stream, recall, p95/p99, amplification, crash/rebuild equivalence |
| HippoRAG | entity graph plus Personalized PageRank | real held-out multi-hop retrieval with non-gold anchors |
| Graphiti / Zep | temporal episodes and evolving knowledge graph | update/historical-query suite with exact source provenance |
| Matryoshka representation learning | provider-trained nested embeddings | native compatible model only; no arbitrary truncation |
| FAISS | exact, HNSW, scalar/PQ baselines | exact oracle plus multiple seeded approximate builds |
| ScaNN | partitioning, scoring, quantization | local hardware comparison at equal recall |
| USearch | compact local HNSW and mmap | stable-ID mutation and recall/resource comparison |
| Tantivy / Lucene | production inverted indexes and filters | BM25 parity, filter correctness, update and index-size comparison |

### Tier A: second-wave investigations

| Group | Items | Role in the design space |
|---|---|---|
| Hierarchical/graph memory | RAPTOR, LightRAG, GraphRAG | graph/tree representation and global retrieval |
| Agent memory products | Mem0, Letta/MemGPT, A-MEM | episodic/semantic memory creation and lifecycle policy |
| Retrieval benchmarks | MTEB, BEIR | embedding and retriever quality controls, not memory proof |
| Long-memory benchmarks | LongMemEval, LoCoMo | session, temporal, update, abstention, exact evidence |
| RAG evaluation | RAGBench, ARES | answer grounding and retrieval/reader attribution |
| Fusion/query transformation | RRF, HyDE, RAG-Fusion, Query2Doc | candidate fusion and query expansion |
| Embeddings | BGE-M3, E5, Nomic Embed, GTE | representation alternatives subject to native-width policy |
| Rerankers | BGE rerankers, RankLLM, FlashRank | post-candidate selection quality/cost |
| Vector/search engines | Milvus, Qdrant, LanceDB, Elasticsearch/OpenSearch | comparative serving, filtering, persistence, and scale |

SRA-Bench and SkillCorpus remain methodological prior art: retrieval quality,
retrieval use, and downstream task performance are measured as separate stages.

## 5. Competing hypothesis registry

Each row includes an explicit null. A mechanism is removed or narrowed when its
null is not rejected under the decision rules in Section 7.

| ID | Experimental hypothesis | Competing null | Eliminating experiment |
|---|---|---|---|
| H-R1 | source representation dominates current sparse recall | raw turns are sufficient | conversation-clustered multi-seed raw/key/chunk/fact factorial |
| H-R2 | learned sparse adds evidence absent from BM25 | BM25S/Tantivy/Lucene reach the same recall frontier | identical corpus, equal pool, paired SPLADE++ vs conventional sparse |
| H-R3 | late interaction fixes a single-vector bottleneck | equal-pool cross-encoder or lexical retrieval is sufficient | ColBERT candidate and rerank arms with oracle pool recall |
| H-R4 | query transformation improves weak-overlap retrieval | transformations add distractors/cost without general gain | direct/HyDE/Query2Doc/RAG-Fusion across paraphrase and adversarial slices |
| H-R5 | graph propagation improves real multi-hop recall | graph helps only constructed relationship cases or leaked anchors | held-out real data with explicit non-gold anchor generator and graph sham control |
| H-M1 | semantic facts/summaries improve evidence access | raw episodic sources outperform lossy derived memory | raw/fact/summary/combined conditions with source-link precision |
| H-M2 | interval validity and supersession are necessary | recency rank alone is sufficient | latest/previous/as-of contradiction suite with pre-filter vs post-filter arms |
| H-F1 | RRF is robust to heterogeneous score scales | union, calibrated linear fusion, or one channel wins | equal-budget factorial with fixed/swept `k=60` baseline |
| H-S1 | a reranker improves final rank when candidates contain gold | candidate generation accounts for all observed gain | fixed candidate lists, shuffled/no-op control, pool-size curve |
| H-A1 | HNSW approximation is harmless at current scale | flat search exposes material evidence loss | exact Flat vs seeded HNSW at identical embeddings and filters |
| H-A2 | quantization improves the feasible frontier | recall loss or rebuild cost dominates memory savings | FAISS SQ/PQ and TurboQuant-style codes against exact oracle |
| H-A3 | disk/mutable ANN is required before the target scale | in-memory rebuilt indexes remain superior within target constraints | staged mutation stream and 1M->10M admission gates |
| H-G1 | grounded context yields accurate, supported answers | retrieval scores fail to predict answer faithfulness | closed/retrieved/oracle/full-context paired reader evaluation |
| H-ARCH | current HybridMind is Pareto-optimal | a simpler or external architecture dominates it | architecture-level Pareto comparison on fixed manifests |

## 6. Experiment program

### Tranche 0: measurement and provenance repair

1. Require source, dataset, dependency, machine, Git commit, dirty-worktree,
   configuration, corpus-generation, and executed-stage hashes.
2. Run deterministic retrieval twice and require identical ordered evidence IDs
   and scores.
3. Do not allow sequential one-shot sub-millisecond latency ratios to select a
   quality winner. Use repeated randomized/interleaved conditions, warm and cold
   strata, robust medians, p95/p99 intervals, and an absolute timer-resolution
   margin.
4. Equalize and report candidate budgets before fusion or reranking.
5. Retain failed conditions and incomplete ledgers as failed receipts.

The schema-v1 LoCoMo representation run violated item 3. Schema v2 separates
the reproducible retrieval-quality decision from exploratory latency and
requires a dedicated latency follow-up.

### Tranche 1Q: offline quality controls

- datasets: LoCoMo and locally present LongMemEval-S;
- conditions: TF-IDF, pure BM25, BM25S, source-key variants, exact dense with a
  deterministic test double for mechanics only, HNSW mechanics, and current
  HybridMind controlled modes;
- split unit: conversation/session, never individual questions from the same
  history across development and test;
- seeds: at least five declared split/build seeds;
- outputs: per-question evidence IDs, ranks, failure category, candidate ceiling,
  latency observations, tokens, index bytes, and zero provider-call attestation.

An offline deterministic embedding double tests index mechanics, not semantic
embedding quality.

### Tranche 1S: early scale and freshness controls

- deterministic native 4096-d vectors;
- exact Flat oracle, HNSW, FAISS scalar/PQ candidates, followed by opt-in
  TurboQuant/Turbovec, USearch, ScaNN, and DiskANN-family adapters where legal
  and compatible;
- bounded host-safe sizes first, then the protocol's 1M gate only if the offline
  resource admission check permits it;
- multiple build orders and seeds;
- static recall@k, filter recall, incremental update recall, deletes, index lag,
  build/rebuild time, p50/p95/p99, peak RSS, persistent bytes, and crash/rebuild
  equivalence.

This track runs in parallel with quality work. It cannot establish semantic
retrieval quality because the vectors are synthetic.

### Tranche 2: candidate mechanisms

1. conventional lexical engines and learned sparse;
2. single-vector dense and late interaction;
3. direct query and query transformations;
4. graph sham, typed traversal, associative propagation, and temporal graph;
5. equal-budget union, linear fusion, and RRF.

Promote only mechanisms that add gold evidence to the candidate pool or improve
rank at a fixed pool with non-dominated resources.

### Tranche 3: memory model factorial

Cross episodic representation (raw, sentence, parent-child, session) with
semantic representation (none, fact, summary) and temporal policy (none,
interval filter, supersession). Use a fractional factorial screen, then a full
factorial over retained factors. Every derived node carries source IDs.

### Tranche 4: selection, grounding, and answering

Use locked candidate generators. Compare no reranker, lexical, BGE cross-
encoder, late interaction, RankLLM, and FlashRank where applicable. Then compare
closed book, retrieved context, oracle evidence, and full context with one fixed
reader. Record unsupported claims, citation correctness, abstention, reader
tokens, and paired oracle gap.

No live reader, embedding, decomposition, or reranker request is permitted
until the resource report and priced plan pass validate-only preflight.

### Tranche 5: external architecture challengers

Vespa, Qdrant, Milvus, LanceDB, Elasticsearch/OpenSearch, Lucene/Tantivy, and
other challengers receive the same immutable corpus/query manifest, evidence
IDs, filters, update stream, and resource accounting. SQLite remains the source
of truth. If a challenger wins reproducibly, replace the losing derived stack
or use the challenger through a rebuildable adapter.

## 7. Decision and statistical rules

- Primary retrieval metric: paired exact-evidence Recall@10, with all-evidence
  coverage and catastrophic-miss rate as co-primary safety metrics.
- Ranking: MRR and nDCG on stable evidence IDs.
- Cluster bootstrap and permutation/randomization operate at the conversation or
  session level, not the individual-question level when histories are shared.
- Correct multiple variant selection with a preregistered family-wise or false-
  discovery procedure. Report all attempted variants.
- Require a minimum effect size in addition to a confidence interval. A tiny
  statistically detectable gain may still be operationally dominated.
- Require no material regression in temporal, multi-hop, adversarial, update,
  and abstention slices.
- Report candidate oracle recall before attributing a result to fusion or
  reranking.
- Put every configuration on a quality/latency/memory/storage/cost Pareto
  frontier. Architectural complexity has no positive weight.
- A heavy component remains off by default until an independent held-out run
  and at least one additional dataset reproduce the effect.
- A result is historical if its source hash or worktree does not match the code
  under evaluation.

## 8. Current evidence ledger

### Demonstrated

- On the current dirty worktree, the final offline verification passed 387
  tests with 3 skipped; the legacy verification suite passed 16 tests; the
  MemoryBench TypeScript suite passed 4 tests; formatting, compilation, and
  `.venv` dependency checks passed. These are correctness checks, not
  retrieval-quality claims.
- The fresh baseline at commit `3422f226` measured LoCoMo BM25S Recall@10 of
  0.544696 across 1,977 exact-evidence questions, with zero provider calls.
- The source-identical schema-v1 reproduction produced exactly the same raw and
  speaker-prefix ranking metrics as the retained run, but ambient timing changed
  the eligible winner. This falsified the old latency-dependent selection rule.
- Schema v2 selects quality using deterministic gates and reproduced a locked
  speaker-prefix held-out Recall@10 gain of 0.031599 with zero provider calls.
  Across five conversation-split seeds the mean gain was 0.029661, standard
  deviation 0.002309, and every seed was positive. The ten conversations are
  reused across seeds, so these are robustness splits, not five independent
  datasets. Artifact:
  `experiments/results/offline-locomo-sparse-multiseed-20260822-v2.json`.
- A dedicated repeated sparse latency experiment found no decision-eligible
  regression: the wall-clock paired mean for speaker prefix versus raw was
  -0.00896 ms with a 95% interval from -0.02231 to 0.00466 ms. The Windows
  process timer quantized most observations to zero and was explicitly
  excluded from decisions. Artifact:
  `experiments/results/offline-sparse-latency-20260822-v1b.json`.
- Failure clustering showed the prefix gain was concentrated in speaker-named
  questions, but a direct routing experiment falsified the proposed router on
  cost. On its held-out split, 960/966 pre-exclusion questions matched a known
  speaker; routed and unconditional prefix Recall@10 were identical at
  0.571365 versus 0.544735 raw. Routing retained both indexes (2.043x raw token
  footprint) while unconditional prefix needed 1.043x. Two-field RRF reached
  only 0.560560. Five malformed/unscorable annotations are preserved as
  question-level failure rows. Artifact:
  `experiments/results/offline-sparse-field-routing-20260822.json`.
- Synthetic native-4096 FAISS mechanics show approximation is not harmless at
  the current default: HNSW Recall@10 rose from 0.6652 at efSearch 64 to 0.8621
  at 128 and 0.98125 at 256. A separate threshold run reached 0.99921875 at 512
  and 1.0 at 1024. HNSW controls are now explicit and execution-attested, but
  the default remains 64 until native semantic evidence justifies promotion.
- Pinned Turbovec 1.0.0 synthetic 4096-d runs across five seeds measured about
  5.2x raw-matrix compression with mean Recall@10 0.8534 for four-bit search,
  and about 10.4x with mean Recall@10 0.563 for two-bit search. Calibration was
  effectively null for isotropic data. This is storage/ANN mechanics evidence,
  not conversational evidence or a latency win. Artifact:
  `experiments/results/offline-turbovec-frontier-4096-multiseed-20260822.json`.
- The adaptive-k experiment failed its retained-context reduction gate.
- A fixed-pool local MiniLM cross-encoder replay separated selection from
  candidate generation. On 991 held-out, exact-evidence questions, reranking
  the speaker-prefixed BM25S pool raised Recall@10 from 0.588057 to 0.630680;
  the conversation-clustered paired lift was 0.042623 with a 95% interval of
  [0.012990, 0.066432]. Candidate-pool oracle recall was 0.673342, so the
  reranker recovered part, not all, of the available evidence. It did not
  reliably improve MRR and it materially regressed the multi-hop slice by
  -0.091270, interval [-0.131579, -0.031008]. Mean reranker latency was
  287.06 ms per 25-document pool on this CPU host. Artifact:
  `experiments/results/offline-locomo-rerank-minilm-20260822.json`.
- A real LoCoMo associative-graph arm now tests PPR against both
  speaker-prefixed BM25S and a degree-preserving sham. The sham preserved every
  node degree while retaining only 7.68% of term-turn edges. On held-out data,
  real PPR exceeded the sham by 0.564474 Recall@10, interval
  [0.557943, 0.570298], which rejects a degree/popularity-only explanation.
  PPR itself improved over BM25S by only 0.008588 under conversation-cluster
  resampling, interval [-0.008094, 0.026386]; RRF(BM25S,PPR) improved by
  0.014330, interval [-0.006940, 0.032565]. Neither is a decision-eligible
  general gain. The held-out multi-hop PPR-minus-BM25S delta was exactly zero;
  only the temporal category had a positive clustered PPR interval. Artifact:
  `experiments/results/offline-locomo-associative-graph-20260822-v2.json`.
- A full offline BGE-M3 mechanism run compared learned sparse retrieval and
  MaxSim without using the model's incompatible 1024-d dense output. On 1,136
  held-out exact-evidence questions, learned sparse lost to speaker-prefixed
  BM25S by -0.116779 candidate-pool oracle recall, interval
  [-0.153183, -0.074689], and by -0.090704 pre-rerank Recall@10, interval
  [-0.130874, -0.042864]. MaxSim improved ranking within both fixed pools:
  +0.029222 Recall@10 for BM25S, interval [0.014188, 0.040847], and +0.038481
  for learned sparse, interval [0.019350, 0.054669]. The run produced 163,804
  sparse postings and 837,558,272 bytes of token vectors for 5,881 turns;
  local encoding took 13.38 minutes on this CPU. Artifact:
  `experiments/results/offline-locomo-bgem3-mechanisms-20260822.json`.

### Rejected or invalidated

- Sequential one-shot sub-millisecond p95 ratios are not a valid architecture
  selection gate.
- The proposed speaker-aware router is dominated by unconditional prefixing on
  the measured held-out split; the two-field RRF arm is also dominated there.
- Unconditional MiniLM reranking is rejected as a global default despite its
  aggregate Recall@10 gain because it causes a significant held-out multi-hop
  regression and adds roughly 287 ms per query on the measured CPU host.
- Associative PPR is not promoted: its semantic term links are real signal
  beyond node degree, but neither PPR nor RRF demonstrates a general held-out
  gain over the stronger conventional sparse baseline. A HippoRAG reproduction
  is not claimed.
- BGE-M3 learned sparse is rejected as a LoCoMo default because it loses both
  candidate ceiling and final pre-rerank recall to conventional BM25S at equal
  budget. This does not reject SPLADE++ or learned sparse on every corpus.
- Turbovec two-bit and four-bit modes are not admissible defaults on synthetic
  recall alone, and calibration does not rescue isotropic inputs.
- The local `longmemeval_s.json` is an oracle-context subset, not a retrieval
  corpus: 948/948 haystack sessions are gold, there are zero distractors, and
  every example has at most six sessions. The prior near-perfect top-10 report
  is invalidated. The runner now refuses metrics and preserves a failed
  dataset-admission receipt at
  `experiments/results/offline-longmemeval-session-retrieval-20260822-invalidated.json`.
- Earlier ANN sweep artifacts that configured efSearch on an ID-map wrapper did
  not execute the requested control and are invalid. Only filenames containing
  `attested` and the pinned Turbovec artifact are admissible mechanics evidence.
- Historical answer-string overlap is not exact evidence recall.
- Existing plan-only graph/hybrid conditions are not completed experiments.
- Configuration fields, code scaffolds, cached models, and passing tests do not
  establish quality gains.

### Open or narrowed hypotheses

H-R1 is provisionally positive only for the speaker-prefix representation on
one ten-conversation corpus; its adaptive router is eliminated and a second
dataset is required. H-R2 is negative for the measured BGE-M3 learned-sparse
representation; SPLADE++ and other corpora remain open. H-R3 is positive only
as a fixed-pool selection claim: MaxSim improves both tested pools, but no
independent ColBERTv2 candidate channel or single-vector dense control was run,
and the token-vector footprint blocks default promotion. H-R5 now separates two claims: real term associations
strongly beat a degree-preserving sham, but graph propagation does not beat the
conventional sparse baseline reliably and has no held-out multi-hop lift. H-S1
is positive for aggregate Recall@10 within a fixed speaker-prefixed pool, but
narrowed to a gated/alternative selection policy because MRR is inconclusive,
multi-hop regresses, and CPU latency is material. H-A1 is positive for
synthetic mechanics because efSearch 64 loses many exact neighbors, but
semantic evidence impact is open. H-A2 is narrowed: Turbovec compression is
real, yet the tested recall loss blocks promotion. H-F1 is rejected for the
measured two-field sparse RRF arm and remains unproved for sparse-plus-graph
fusion. Other learned-sparse models, independent late-interaction retrieval,
memory-model, grounding, answering, external-backend, and production-scale
hypotheses remain open. No current
result supports a SOTA, 10M-100M, end-to-end answer, or transformer
KV-cache-replacement claim.

## 9. Adversarial review outcome

Result: conclusion weakened, not overturned.

The strongest counterargument was that 4096-dimensional vectors and rebuild-
based mutation can make an otherwise accurate architecture infeasible at target
scale. The program therefore runs scale/freshness gates in parallel rather than
after all quality work. The review also found unequal current candidate budgets,
limited local answer-level ground truth, and overfitting risk from LoCoMo's ten
conversations. Sections 6 and 7 incorporate those corrections.

## 10. Immediate next artifacts

1. a compact/pruned MaxSim or cross-encoder gate specifically targeting the
   multi-hop regression, followed by a second independent dataset;
2. a genuine LongMemEval retrieval corpus or another independent memory
   dataset with distractors and exact source labels;
3. semantic exact-Flat versus HNSW evidence recall after the priced native
   4096-d plan passes preflight;
4. a preregistered factorial over episodic, semantic, and temporal memory
   representations, followed by evidence-grounding and answer-generation
   ablations once retrieval survives the independent-corpus gate;
5. persistent mutable-index comparisons (DiskANN/FreshDiskANN and supported
   external stores) only after the quality gates identify a mechanism worth
   scaling.
