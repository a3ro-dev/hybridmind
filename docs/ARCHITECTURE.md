# HybridMind Architecture

## Overview
HybridMind is a local-native hybrid vector + graph database designed for AI agent memory. It unifies semantic similarity and relational context into a clean, self-contained implementation combining FAISS, NetworkX, and SQLite into a single `.mind` file format.

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
|  | Embedding Engine |  |  Query Engines   |  |   Hybrid Ranker        | |
|  | (transformers)   |  | (Vector / Graph /|  |  (Late Fusion +        | |
|  |                  |  |  BM25)           |  |    Late Fusion)        | |
|  +------------------+  +------------------+  +------------------------+ |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                             Storage Layer                               |
|  +-----------------+  +----------------------+  +--------------------+  |
|  |  SQLite Store   |  |   Vector / BM25      |  |     Graph Index    |  |
|  | (WAL Enabled)   |  |  (FAISS / NLTK)      |  |     (NetworkX)     |  |
|  +-----------------+  +----------------------+  +--------------------+  |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                            Persistence                                  |
|     (.mind directory: manifest.json / store.db / vectors / vectors.map / graph.nx)  |
+-------------------------------------------------------------------------+
```

## Component Deep Dives

### Embedding Engine
- **Model**: `all-mpnet-base-v2` (768 dimensions). Configurable via `HYBRIDMIND_EMBEDDING_MODEL`.
- **Neighborhood Averaging**: At node ingest, the embedding is conditioned on its semantic neighborhood—a practical, non-training variant of GraphSAGE-style aggregation:
  `final_embedding = normalize(0.7 * own_embedding + 0.3 * mean_neighbor_embeddings)`
- **Configuration**: α=0.7 is the default weight, ensuring the node's original content dominates while receiving a 30% contextual pull from its semantic peers.
- **Thread Safety**: The model is serialized under the Python Global Interpreter Lock (GIL). High-concurrency throughput is limited to single-threaded execution (approx. 200ms per embedding).

### Late Fusion Scoring
The system utilizes a weighted linear score fusion to combine semantic, lexical, and structural signals:

```text
Score = vector_weight × V_effective + graph_weight × G_effective
```

Where the default weights are `vector_weight=0.5`, `graph_weight=0.15`, with a `bm25_boost_weight=0.35` applied inside the vector score. Note: these weights do **not** sum to 1.0; the BM25 boost is additive within the vector component.

- **Vector Score (V)**: Cosine similarity between query and node embeddings, plus a BM25 keyword overlap boost (`bm25_overlap × 0.35`). Range: 0.0 to ~1.35.
- **Graph Score (G)**: Proximity based on 1/(1+d), where d is shortest path length from internal or explicit anchor nodes. Gated by BM25 keyword overlap relevance.
- **Default Weights**: `vector_weight=0.5`, `graph_weight=0.15`, `bm25_boost_weight=0.35` (tuned for LoCoMo-style factoid queries).
- **Distance → Score Table**:
  - 0 (self/anchor): 1.0
  - 1 (direct neighbor): 0.5
  - 2 (2-hop): 0.33
  - 3 (3-hop): 0.25
  - ∞ (no path): 0.0

### Storage Layer

#### SQLite Store
- **Persistence**: Relational database (SQLite) in Write-Ahead Logging (WAL) mode.
- **Schema**: `nodes` (full text, metadata, embeddings) and `edges` (from/to/type/weight).
- **Soft-Delete**: Nodes are marked with `deleted_at`. Filtered at search time and cleaned during compaction.
- **Concurrency**: SQLite handles multiple readers during active writes without blocking.

#### FAISS Vector Index
- **Index Type**: `IndexHNSWFlat` (Approximate Nearest Neighbor using HNSW with Inner Product metric, `efSearch=64`, `M=32`).
- **Mapping**: FAISS integer indices are mapped to Node UUIDs via a separate `vectors.map` file persisted alongside the index.
- **Memory**: O(n·d) storage; HNSW search is O(log n) at query time.

#### Okapi BM25 Index
- **Engine**: Pure Python implementation with `nltk` PorterStemmer.
- **Role**: Addresses the single-hop factual recall limitation of vector similarity by prioritizing exact keyword matches, especially for entities and dates.
- **Serialization**: Python Pickle (v5).

#### NetworkX Graph Index
- **Engine**: In-memory `DiGraph`.
- **Traversal**: BFS-based graph proximity computation.
- **Wait Mechanism**: Directed edges are used, but proximity allows for both incoming and outgoing traversal routes.
- **Serialization**: Python Pickle (v5). Direct, fast, and local.

### Persistence (.mind format)
The database persists as a directory with the `.mind` extension:
- `manifest.json`: SHA256 checksums and a monotonic version counter for crash recovery.
- `store.db`: SQLite database.
- `vectors.faiss`: Serialized FAISS index.
- `vectors.map`: FAISS integer-index to Node UUID mapping.
- `graph.nx`: Pickled NetworkX graph.
- `bm25.pkl`: Pickled BM25 index with NLTK stemmer state.

**Atomic Snapshot Protocol**:
1. Create temporary directory.
2. Flush SQLite WAL to disk (checkpoint).
3. Serialize indexes to temp dir.
4. Calculate SHA256 manifest.
5. `fsync` directory and rename to final destination.
6. Rotate backups (keeps 3 most recent snapshots).

### API Layer
Built on FastAPI for performance and Pydantic v2 for strict type safety.
- **Request Timing**: Middleware adds `X-Process-Time-Ms` header to every response.
- **CORS**: Permissive CORS for local development.
- **Soft Filtering**: Queries respect the `deleted_at` field, ensuring "forgotten" nodes are invisible before physical compaction.
- **Validation**: Strict edge type enforcement based on the research edge taxonomy.

### SDK
The Python SDK (`HybridMemory`) provides high-level abstractions:
- `store()` / `store_batch()` / `store_with_auto_edges()`: Node creation with optional auto-linking.
- `recall()` / `recall_stream()`: Hybrid or vector retrieval, with optional streaming.
- `relate()`: Explicit edge creation.
- `trace()`: Semantic-to-graph traversal (finds anchor via vector search, then traverses graph).
- `forget()` / `compact()`: Soft-delete and physical rebuild.
- `session.create/recall/archive/list()`: Scoped memory sessions.
- `tools.get_schema()`: OpenAI function-calling compatible tool schemas.

## Data Flow: Hybrid Search Request
1. **Validation**: Pydantic validates the request parameters and weights.
2. **Embedding**: The query text is vectorized by `EmbeddingEngine`.
3. **Candidate Selection**: FAISS performs a k-NN search to identify 3x the requested `top_k` results.
4. **BM25 Scoring**: BM25 index scores candidates by keyword overlap; high-overlap candidates receive a vector score boost.
5. **Anchor Identification**: If no `anchor_nodes` are provided, the top 3 vector results are used as anchors.
6. **Relational Proximity**: NetworkX calculates shortest path distances from anchors to all candidates. Graph scores are gated by BM25 keyword relevance.
7. **Fusion**: Candidate scores are calculated using the weighted fusion (`vector_weight × V + graph_weight × G`).
8. **Refinement**: Results are re-ranked by their fused score and truncated to `top_k`.

## Design Decisions and Trade-offs
- **HNSW Vector Search**: Chose `IndexHNSWFlat` over exact `IndexFlatIP` for sub-logarithmic query latency at the cost of approximate recall. Acceptable trade-off for the ~10k node scale target.
- **Local-First Architecture**: Chose SQLite/NetworkX (local) over Neo4j (remote) to minimize network latency within agent reasoning loops.
- **Neighborhood Averaging**: Conditioning embeddings at ingest rather than just query-time provides semantic coherence even when graph edges are sparse.

## Scalability Ceiling
- **Memory**: FAISS HNSW (n × 768 × 4 bytes + HNSW graph overhead) + NetworkX overhead. Estimated ~30MB for 10k nodes.
- **Latency**: HNSW search is O(log n). The practical ceiling for sub-50ms p95 is estimated at **8,000-10,000 nodes** on modern hardware.
- **GIL**: Python embedding model serializes concurrent requests. Throughput beyond 10 rps requires external embedding services.
