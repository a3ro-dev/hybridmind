# HybridMind Architecture

## Overview
HybridMind is a local-native hybrid vector + graph database designed for AI agent memory. It unifies semantic similarity, lexical matching, relational context, and learned re-ranking into a clean, self-contained implementation combining FAISS, NetworkX, SQLite, and optional ColBERT/GNN modules.

## System Architecture

```text
+-------------------------------------------------------------------------+
|                               API Layer                                 |
|            (FastAPI / Pydantic v2 / Request Timing Middleware)          |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                              Engine Layer                               |
|  +------------------+  +------------------+  +------------------------+ |
|  | Embedding Engine |  |  Query Engines   |  |   Hybrid Ranker       | |
|  | (bge-m3 /        |  | (Vector / Graph /|  |  (RRF Fusion +        | |
|  |  all-mpnet)      |  |  BM25 / ColBERT) |  |   Cross-Encoder       | |
|  +------------------+  +------------------+  |   Reranker)           | |
|                                               +------------------------+ |
|  +------------------+  +------------------+  +------------------------+ |
|  | Edge Inference   |  | Fact Extractor   |  |  Device Resolution    | |
|  | (cosine + entity)|  | (structured JSON |  |  (auto: cuda>mps>cpu) | |
|  |                  |  |  + retry)        |  |                       | |
|  +------------------+  +------------------+  +------------------------+ |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                             Storage Layer                               |
|  +-----------------+  +--------------------+  +--------------------+    |
|  |  SQLite Store   |  |  Vector / BM25     |  |    Graph Index     |    |
|  | (WAL Enabled)   |  | (FAISS / NLTK)     |  |   (NetworkX)      |    |
|  +-----------------+  +--------------------+  +--------------------+    |
|  +-----------------+  +--------------------+                           |
|  | ColBERT Store   |  |  GNN Reranker      |   (opt-in, off by default)|
|  |  (.npz files)   |  |  (GraphSAGE/HGT)   |                           |
|  +-----------------+  +--------------------+                           |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                            Persistence                                  |
|  (.mind directory: manifest.json / store.db / vectors / graph.nx /     |
|   bm25.pkl / colbert/*.npz)                                            |
+-------------------------------------------------------------------------+
```

## Component Deep Dives

### Embedding Engine
- **Default model**: `BAAI/bge-m3` (1024-dim). Falls back to `all-mpnet-base-v2` (768-dim) via `HYBRIDMIND_EMBEDDING_MODEL`.
- **FlagEmbedding native**: When `FlagEmbedding>=1.2.10` is installed, bge-m3 provides dense (1024) + sparse (lexical weights) + ColBERT (per-token) vectors natively. Without FlagEmbedding, SentenceTransformer backend provides dense-only.
- **Neighborhood Averaging** (`HYBRIDMIND_USE_GRAPH_CONDITIONED_EMBEDDINGS`): At node ingest, the embedding is conditioned on its semantic neighborhood:
  `final_embedding = normalize(0.7 * own_embedding + 0.3 * mean_neighbor_embeddings)`
  Off by default since Phase 2 to prioritize clean semantic baselines. Use the contrastive fine-tuning script (`scripts/train_contrastive.py`) for a trained alternative.
- **Thread Safety**: The model is serialized under the Python GIL. High-concurrency throughput is limited to single-threaded execution.

### Device Resolution
All model loads (embedding, reranker, ColBERT, GNN) call `engine/device.py:resolve_device()`:
- `auto` → cuda > mps > cpu
- `cuda` → force CUDA, raises RuntimeError if unavailable
- `cpu` → force CPU
- `/health` endpoint includes `gpu` object with status, device name, CUDA version.

### Late Fusion Scoring

**Default: RRF (Reciprocal Rank Fusion, k=60)**. Per-signal rank lists (dense, graph) are fused with rank-based weighting. `vector_weight`/`graph_weight` params multiply the per-signal RRF contribution, making search-request tuning effective in both `rrf` and `linear` modes.

**Linear fallback**: `fusion_mode="linear"` preserves the original weighted-sum formula with BM25 overlap gating — selectable per-request for A/B comparison.

**Cross-encoder reranker**: `BAAI/bge-reranker-v2-m3` re-ranks top-25 fusion pool. Before blending, both the fusion combined score and the cross-encoder score are independently normalized to [0,1]. Final: `0.7 * normalized_fusion + 0.3 * normalized_reranker`. This prevents the pure-text reranker from deleting graph-discovered candidates on multi-hop queries.

**FusionScorer MLP** (opt-in): 2-layer MLP (~200 params) with heuristic init that approximates RRF. When trained via `scripts/train_fusion_mlp.py`, loads from config `HYBRIDMIND_FUSION_MODEL=<checkpoint.npz>`.

**Distance → Graph Score Table**:
- 0 (self/anchor): 1.0
- 1 (direct neighbor): 0.5
- 2 (2-hop): 0.33
- 3 (3-hop): 0.25
- ∞ (no path): 0.0

### Auto-Edge Inference
(`HYBRIDMIND_AUTO_EDGES_ENABLED=true`, `engine/edge_inference.py`)
- **Cosine-threshold** (`HYBRIDMIND_AUTO_EDGE_COSINE_THRESHOLD=0.75`): Top-N vector neighbors above threshold get `similar_to` edges.
- **Entity co-occurrence** (`HYBRIDMIND_AUTO_EDGE_ENTITY_ENABLED=true`): Nodes sharing named entities get `co_occurs` edges. Uses pre-extracted `fact.entities` from the fact extractor, with optional spaCy NER fallback.
- **Typed walk weights** (`models/edge.py:EDGE_TYPE_WALK_WEIGHTS`): Per-edge-type contribution map used by `compute_weighted_proximity_score`. Strong causal edges (led_to, caused_by) weight 1.0; structural edges (similar_to) weight 0.7; session edges weight 0.3-0.6.
- Wired into all three ingest paths: `/nodes`, `/bulk/nodes`, `/ingest/session-facts`.

### Opt-In Research Modules

| Module | Config | Requirement | Storage |
|--------|--------|-------------|---------|
| ColBERT MaxSim | `HYBRIDMIND_COLBERT_ENABLED=true` | `FlagEmbedding>=1.2.10` | `<mind>/colbert/*.npz` (~100-200KB/node) |
| GNN Reranker (GraphSAGE) | `HYBRIDMIND_GNN_ENABLED=true` | `torch-geometric` | Checkpoint `.pt` via `HYBRIDMIND_GNN_MODEL_PATH` |

All modules ship with CPU fallbacks and are off by default.

### Storage Layer

#### SQLite Store
- **Persistence**: SQLite in WAL mode. Schema: `nodes` (text, metadata, raw_embedding BLOB, deleted_at) and `edges` (from/to/type/weight/edge_id).
- **Soft-Delete**: Nodes marked with `deleted_at`. Filtered at search time, cleaned during compaction.
- **Concurrency**: Multiple readers during active writes without blocking.

#### FAISS Vector Index
- **Index Type**: `IndexHNSWFlat` (HNSW with Inner Product, `efSearch=64`, `M=32`). `faiss-gpu` supported on Linux/Docker via `HYBRIDMIND_USE_FAISS_GPU=true`.
- **Dynamic dimension**: Dimension set from `settings.embedding_dimension`. Mismatch between config and stored index raises clear "run reindex" error.
- **Memory**: O(n·d) storage; HNSW search is O(log n).

#### Okapi BM25 Index
- **Engine**: Pure Python with `nltk` PorterStemmer. Pickle serialization.
- **Role**: Exact keyword matching for entities, dates, and facts where semantic similarity is insufficient.

#### NetworkX Graph Index
- **Engine**: In-memory `DiGraph`. BFS traversal for proximity computation. Directed edges with bidirectional traversal.
- **Serialization**: Python Pickle (v5).

#### ColBERT Store (`storage/colbert_store.py`)
- **Format**: Per-node `.npz` files in `<mind>/colbert/`. Each contains `(seq_len, 1024)` float32 for bge-m3.
- **MaxSim rerank** (`engine/colbert_reranker.py`): At query time, encode query as colbert tokens, compute max cosine per query token vs stored candidate tokens, blend into combined score (α=0.3).

### Persistence (.mind format)
The database persists as a directory with the `.mind` extension:
- `manifest.json`: SHA256 checksums, monotonic version counter, embedding model/dimension.
- `store.db`: SQLite database.
- `vectors.faiss` + `vectors.map`: Serialized FAISS index + integer-to-UUID mapping.
- `graph.nx`: Pickled NetworkX graph.
- `bm25.pkl`: Pickled BM25 index with NLTK stemmer state.
- `colbert/`: Per-node `.npz` files (when enabled).

**Atomic Snapshot Protocol**:
1. Create temporary directory.
2. Flush SQLite WAL to disk (checkpoint).
3. Serialize indexes to temp dir.
4. Calculate SHA256 manifest.
5. `fsync` directory and rename to final destination.
6. Rotate backups (keeps 3 most recent snapshots).

### Reindex Script
`scripts/reindex_embeddings.py`: Re-embeds all node texts with the current model and rebuilds FAISS from scratch. Required when switching embedding models (e.g., all-mpnet→bge-m3). Fresh installs need no migration.

### API Layer
Built on FastAPI for performance and Pydantic v2 for strict type safety.
- **Request Timing**: Middleware adds `X-Process-Time-Ms` header to every response.
- **CORS**: Permissive CORS for local development.
- **Validation**: Strict edge type enforcement based on the edge taxonomy (`models/edge.py`).
- **Query Cache**: LRU cache (TTL=300s, maxsize=1000) for vector and hybrid searches.

### SDK
The Python SDK (`HybridMemory`) provides high-level abstractions:
- `store()` / `store_batch()` / `store_with_auto_edges()`: Node creation with optional auto-linking.
- `recall()` / `recall_stream()`: Hybrid or vector retrieval.
- `relate()`: Explicit edge creation.
- `trace()`: Semantic-to-graph traversal.
- `forget()` / `compact()`: Soft-delete and physical rebuild.
- `session.create/recall/archive/list()`: Scoped memory sessions.
- `tools.get_schema()`: OpenAI function-calling compatible tool schemas.

## Data Flow: Hybrid Search Request
1. **Validation**: Pydantic validates parameters, weights, and optional `fusion_mode`.
2. **Embedding**: Query text vectorized by `EmbeddingEngine` (bge-m3 1024-dim).
3. **Candidate Selection**: FAISS k-NN (candidate_k=max(100, top_k*10)) + BM25 top-5000.
4. **BM25 Boost**: Keyword overlap fraction multiplied by `bm25_boost_weight=0.35` added to vector score.
5. **SGMem Chunk Rollup**: Sentence chunks rolled up to parent nodes by max score.
6. **Anchor Identification**: Explicit `anchor_nodes` or top-3 vector hits.
7. **Graph Expansion**: BFS traversal from anchors (depth=max_depth). Pure-graph candidates added to pool with vector_score=0.
8. **Graph Scoring**: Shortest-path proximity `1/(1+d)` from anchors.
9. **RRF Fusion**: Dense and graph rank lists fused with signal weights. `vector_weight`/`graph_weight` multiply per-signal RRF contribution.
10. **Deduplication**: Text-identical candidates removed, preferring highest vector score.
11. **ColBERT MaxSim** (if enabled): Per-token query vectors matched against stored candidate colberts; 30% weight blend into combined score.
12. **Cross-Encoder Reranking**: Top-25 pool re-ranked by `bge-reranker-v2-m3`. Both fusion and CE scores normalized to [0,1], blended 70/30.
13. **Final Sort**: Sorted by combined_score descending, sliced to top_k.

## Design Decisions and Trade-offs
- **RRF over fixed linear weights**: Zero per-corpus tuning. Works across diverse benchmarks without weight sweeps.
- **Normalized reranker blending**: Prevents the pure-text cross-encoder from deleting graph-discovered candidates on multi-hop queries.
- **Local-First Architecture**: SQLite/NetworkX/FAISS over remote DBs to minimize latency within agent reasoning loops.
- **bge-m3 default**: 1024-dim provides richer semantic separation than 768-dim all-mpnet, at the cost of ~30% more RAM and slower CPU encoding.
- **Opt-in heavy modules**: ColBERT (~200KB/node) and GNN (~500MB model) are off by default with CPU fallbacks. The system stays local-native and test-suite-compatible.

## Scalability Ceiling
- **Memory**: FAISS HNSW (n × 1024 × 4 bytes + HNSW graph) + NetworkX overhead. ~40MB for 10k nodes at 1024-dim.
- **Latency**: HNSW search O(log n). Practical ceiling **~8,000-10,000 nodes** for sub-50ms p95.
- **GIL**: Python embedding model serializes concurrent requests. Throughput beyond 10 rps requires external embedding services.
- **ColBERT**: Per-token storage (~100-200KB/node) limits practical corpus to ~50K nodes with typical disk budgets.
