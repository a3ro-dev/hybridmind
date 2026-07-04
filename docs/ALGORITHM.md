# Reciprocal Rank Fusion, Cross-Encoder Reranking, and Ingest-Time Neighborhood Averaging

## Abstract
Traditional vector-only retrieval captures semantic similarity but ignores relational structure, while graph-only retrieval lacks semantic nuance. HybridMind addresses this through **Reciprocal Rank Fusion (RRF)** as the default scoring strategy, combined with a **pre-trained cross-encoder reranker** and optional ingest-time neighborhood averaging. RRF provides zero-tuning fusion across diverse benchmarks; the cross-encoder adds text-match precision without deleting graph-discovered candidates.

## 1. Motivation

### 1.1 Limitations of Vector-Only Retrieval
- Captures semantic similarity but ignores relational structure.
- Two documents can be semantically distant but causally related (e.g., "gradient descent" and "Adam optimizer" — connected by derivation, not just similarity).
- Pure vector search fails to leverage the explicit knowledge represented in graph edges.

### 1.2 Limitations of Graph-Only Retrieval
- Structurally connected nodes are not always semantically relevant.
- Deep traversal (d > 2) without semantic filtering quickly introduces noise.
- Requires manual, explicit edge creation; does not generalize well to sparse content.

### 1.3 The Hybrid Approach
Score fusion provides a principled way to combine both signals. RRF is a well-known rank-based fusion method that requires no weight tuning. Signal weights modulate per-benchmark behavior without changing the formula.

## 2. RRF Fusion (Default)

### 2.1 Formal Definition
Reciprocal Rank Fusion over $S$ per-signal rank lists:

$$RRF\_score(n) = \sum_{s \in S} w_s \cdot \frac{1}{k + rank_s(n)}$$

Where:
- $w_s$ is the signal weight (e.g., `vector_weight=0.5`, `graph_weight=0.15`).
- $k=60$ is the RRF smoothing constant (higher = flatter penalty curve).
- $rank_s(n)$ is the 1-based rank of node $n$ in signal $s$'s sorted score list.

RRF treats each signal's rank position, not raw score magnitude, as the primary fusion signal. This eliminates the need for per-corpus weight sweeps — the rank-based formula works across LoCoMo, LongMemEval, and MuSiQue without tuning.

### 2.2 Signal Weights
The `vector_weight` and `graph_weight` parameters multiply each signal's RRF contribution before summing. This preserves the semantic "talk to the ranker" interface while allowing per-request tuning:
- `vector_weight=0.5, graph_weight=0.15` → graph provides a modest re-ranking signal (default).
- `vector_weight=0.1, graph_weight=0.9` → graph dominates; used for multi-hop edge-dependent queries where semantic overlap is absent.

### 2.3 Linear Fallback
`fusion_mode="linear"` (selectable per-request) preserves the original weighted-sum formula:

$$Score(q, n) = w_v \cdot V_{eff}(q, n) + w_g \cdot G_{eff}(A, n)$$

Where $V_{eff}$ = cosine similarity + BM25 keyword overlap boost and $G_{eff}$ = proximity gated by BM25 keyword relevance. Used for A/B comparison and back-compat.

### 2.4 Distance → Graph Score Table
| Distance | Score |
|----------|-------|
| 0 (self/anchor) | 1.0 |
| 1 (direct neighbor) | 0.5 |
| 2 (2-hop) | 0.33 |
| 3 (3-hop) | 0.25 |
| ∞ (no path) | 0.0 |

## 3. Cross-Encoder Reranker

### 3.1 Model
`mixedbread-ai/mxbai-rerank-large-v2` (default, see `config.reranker_model`) — a pre-trained cross-encoder that scores (query, passage) pairs. Loaded at startup with GPU auto-detect (`engine/device.py`). Selectable via `HYBRIDMIND_RERANKER_MODEL`.

### 3.2 Blending Strategy
The cross-encoder re-ranks the top-25 fusion pool. **Both** the pre-rerank combined score and the cross-encoder raw scores are independently normalized to [0,1] before blending:

$$combined\_final = 0.7 \cdot norm(combined\_fusion) + 0.3 \cdot norm(cross\_encoder)$$

This normalization is critical: RRF scores are in the ~0.001-0.03 range while cross-encoder scores span a wider raw range. Without normalization, the cross-encoder would dominate completely — deleting graph-discovered candidates on multi-hop queries where text similarity is low but structural proximity is high.

### 3.3 Reranker Mode
- `RERANK_MODE=cross` (default): Local cross-encoder. Fully offline, no API calls.
- `RERANK_MODE=llm`: HackClub listwise reranker via LLM. A/B fallback.
- `RERANK_MODE=off`: No reranking (passthrough).

## 4. Anchor Node Selection
- **Implicit Anchors**: When `anchor_nodes` are not provided, defaults to the top-3 nodes from the initial vector search results.
- **Explicit Anchors**: When provided by the caller, supports multi-hop queries where the semantic query overlaps with the anchor but not the answer (e.g., "What leads the engineering team?" → anchor on "Company X quantum breakthrough" → traverse to "Dr. Smith leads the team" → traverse to "The new processor operates at near absolute zero").

## 5. Ingest-Time Neighborhood Averaging

### 5.1 Motivation
Standard hybrid retrieval systems treat the embedding space and graph structure as independent, fusing them only at query time. We apply a practical, non-training variant of GraphSAGE-style aggregation to the embedding space.

### 5.2 Formulation
Given node $n$ with text $t$ and semantic neighbor embeddings $\{e_1, \dots, e_k\}$:
1. $e_{raw} = embed(t)$
2. $e_{neighbors} = \text{mean}(\{e_1, \dots, e_k\})$
3. $e_{conditioned} = \text{normalize}(\alpha \cdot e_{raw} + (1-\alpha) \cdot e_{neighbors})$

Where **$\alpha=0.7$** (own embedding weight). Configurable via `HYBRIDMIND_USE_GRAPH_CONDITIONED_EMBEDDINGS` (off by default since Phase 2).

### 5.3 Empirical Observations
Measured on 20 nodes in a 1,000-node database (single-domain, arXiv):
- **Mean Cosine Difference (Raw vs Conditioned)**: 0.00976 (~1%).
- **Variance**: Min delta 0.005, Max delta 0.018.

At multi-domain scale (7,510 nodes across 5 domains), mean cosine difference doubled to **0.01927** (~2× single-domain baseline). See [MULTI_DOMAIN_EVAL.md](MULTI_DOMAIN_EVAL.md) §4.3.

### 5.4 Limitations
- Effect size is modest (~0.01-0.02). Retrieval improvement specifically from this technique has not been isolated in ablation.
- Uses vector neighbors as proxy for graph neighbors (edges may not exist for new nodes).
- Off by default since Phase 2. Use `scripts/train_contrastive.py` for a trained alternative.

## 6. Sparse Retrieval (BM25) & Keyword Exact Match

### 6.1 NLTK Porter Stemmer
Okapi BM25 Index with `nltk` PorterStemmer for suffix stripping (e.g., `researching` → `research`), improving recall for fact-based questions over simple whitespace tokenization.

### 6.2 BM25 Overlap Boost
BM25 overlap fraction (query terms found in node text after stemming) multiplied by `bm25_boost_weight=0.35` and added to cosine similarity. Additionally, for linear fusion mode, graph score is gated by BM25 overlap — candidates with zero keyword relevance receive zero graph score regardless of structural proximity.

## 7. ColBERT MaxSim Late Interaction (Opt-In)

### 7.1 Storage
Per-token ColBERT vectors stored as `.npz` files in `<mind>/colbert/` when `HYBRIDMIND_COLBERT_ENABLED=true`. Requires `FlagEmbedding>=1.2.10` for native bge-m3 colbert output. Storage cost: ~100-200KB/node.

### 7.2 MaxSim Scoring
At query time, encode query as colbert tokens. For each candidate, load stored colbert vectors. MaxSim = mean over query tokens of max cosine similarity with any candidate token. Blended into combined score at α=0.3.

## 8. Auto-Edge Inference (Opt-In)

### 8.1 Cosine-Threshold Edges
At ingest, vector search for top-N neighbors above threshold τ=0.75. Creates `similar_to` edges. `HYBRIDMIND_AUTO_EDGE_COSINE_THRESHOLD`, `HYBRIDMIND_AUTO_EDGE_MAX_PER_NODE`.

### 8.2 Entity Co-Occurrence Edges
Nodes sharing named entities (from fact extraction or spaCy NER) receive `co_occurs` edges. `HYBRIDMIND_AUTO_EDGE_ENTITY_ENABLED`.

### 8.3 Typed Walk Weights
Per-edge-type contribution to graph proximity via `EDGE_TYPE_WALK_WEIGHTS`:
- Causal/logical (led_to, caused_by, depends_on, supports): 0.9-1.0
- Structural/proximity (similar_to, analogous_to): 0.7
- Co-occurrence (co_occurs): 0.6
- Session (next_turn, same_session): 0.5-0.6
- Ancillary (mentions, retrieved_during, belongs_to): 0.3-0.5

## 9. Retrieval Quality

### 9.1 Eval Methodology
The evaluation pipeline uses answer-text overlap as weak supervision ground truth for retrieval relevance. BM25 exact match is the primary relevance signal. All metrics should be treated as directional.

### 9.2 Results

**Test suite**: 34/37 passed, 3 skipped (live SDK tests). All core search, fusion, graph traversal, and edge-CRUD tests pass consistently.

**In-memory retrieval ablation** (5-node ML corpus, 7 queries, bge-m3 1024-dim, RRF):
- All modes (vector, BM25, hybrid, hybrid-heavy-graph): **P@3=0.48, MRR=1.00** (ceiling on tiny set)

**Graph-depth regime benchmark** (9-node multi-hop graph with distractors):
- Semantic paraphrase + exact lexical: **Recall@3=1.0** (all modes)
- Edge-dependent multi-hop (2-hop): **vector=0% / hybrid=100%** (graph surfaces correct answer where vector hits zero)
- Missing anchor: **0%** (correct failure mode)

**LoCoMo**: Peak **48% accuracy** (Qwen3.5 397B), **60% Hit@10**. Single-hop **0%** accuracy — LLM extraction failure, not retrieval. Phase 4 QA retry improves abstention handling.

Full detail: [LOCOMO_BENCHMARK_REPORT.md](LOCOMO_BENCHMARK_REPORT.md), [MULTI_DOMAIN_EVAL.md](MULTI_DOMAIN_EVAL.md).

## 10. Complexity Analysis

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Node ingest | O(d + k·d) | O(d) | embed + k neighbor lookups + BM25 + optional auto-edge |
| Vector search | O(log n) | O(1) | HNSW approximate |
| Graph score | O(V + E) | O(V) | BFS from anchors |
| RRF fusion | O(c·log c) | O(c) | c candidates, sort+rank |
| Cross-encoder | O(p) | O(1) | p pool size, one predict pass |
| ColBERT MaxSim | O(Q·C·dim) | O(Q·C) | Q query tokens × C candidate tokens |
| Compaction | O(n·d) | O(n·d) | Full HNSW rebuild |

*n=nodes, d=dimensions(1024), V=graph vertices, E=edges, c=candidates, p=rerank pool, Q=query tokens, C=candidate tokens*

## 11. Comparison with Related Systems

| System | Vector | Graph | RRF/Reranker | ColBERT | Conditioned Emb | Local-Native |
|--------|:------:|:-----:|:------------:|:-------:|:---------------:|:------------:|
| ChromaDB | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Weaviate | ✓ | ~ | ~ | ✗ | ✗ | ✗ |
| GraphRAG | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Neo4j+pgvec | ✓ | ✓ | manual | ✗ | ✗ | ✓ |
| **HybridMind** | ✓ | ✓ | ✓ | ✓ (opt-in) | ✓ (opt-in) | ✓ |
