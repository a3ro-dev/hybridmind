# HybridMind Architecture

## Overview
HybridMind is a hybrid vector + graph database designed for AI agent memory. It unifies semantic similarity and relational context into a single Contextual Relevance Score (CRS), enabling retrieval that is both semantically aware and structurally grounded.

## System Architecture

```text
+-------------------------------------------------------------------------+
|                               API Layer                                 |
|            (FastAPI / Pydantic v2 / Per-IP Rate Limiting)               |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                              Engine Layer                               |
|  +------------------+  +------------------+  +------------------------+ |
|  | Embedding Engine |  |  Query Engines   |  |   Hybrid Ranker        | |
|  | (transformers)   |  | (Vector / Graph) |  |   (CRS Algorithm)      | |
|  +------------------+  +------------------+  +------------------------+ |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                             Storage Layer                               |
|  +-----------------+  +----------------------+  +--------------------+  |
|  |  SQLite Store   |  |     Vector Index     |  |     Graph Index    |  |
|  | (WAL Enabled)   |  |      (FAISS)         |  |     (NetworkX)     |  |
|  +-----------------+  +----------------------+  +--------------------+  |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                            Persistence                                  |
|     (.mind directory: manifest.json / store.db / vectors / graph)       |
+-------------------------------------------------------------------------+
```

## Component Deep Dives

### Embedding Engine
- **Model**: `all-MiniLM-L6-v2` (384 dimensions).
- **Graph Conditioning**: At node ingest, the embedding is conditioned on the semantic neighborhood:
  `final_embedding = normalize(0.7 * own_embedding + 0.3 * mean_neighbor_embeddings)`
- **Configuration**: α=0.7 is the default weight, ensuring the node's original content dominates while receiving a 30% contextual pull from its semantic peers.
- **Thread Safety**: The model is serialized under the Python Global Interpreter Lock (GIL). High-concurrency throughput is limited to single-threaded execution (approx. 200ms per embedding).

### CRS Algorithm
The Contextual Relevance Score (CRS) is the core fusion mechanism:
`CRS = 0.6 * V + 0.4 * G`

- **Vector Score (V)**: Cosine similarity between query and node embeddings. Range: 0.0 to 1.0.
- **Graph Score (G)**: Proximity based on 1/(1+d), where d is shortest path length from internal or explicit anchor nodes.
- **Default Weights**: α=0.6, β=0.4 (Semantic Primacy Principle). Semantic matching provides the base relevance; relationships refine the final ranking.
- **Wait Table (distance → score)**:
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
- **Index Type**: `IndexFlatIP` (Exact Nearest Neighbor using Inner Product).
- **Mapping**: FAISS maintains integer indices mapped back to Node UUIDs via an internal `id_map`.
- **Memory**: O(n·d) brute force search; fits in L2/L3 cache up to ~10,000 nodes for peak performance.

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
- `graph.nx`: Pickled NetworkX graph.

**Atomic Snapshot Protocol**:
1. Create temporary directory.
2. Flush SQLite WAL to disk (checkpoint).
3. Serialize indexes to temp dir.
4. Calculate SHA256 manifest.
5. `fsync` directory and rename to final destination.
6. Rotate backups (keeps 3 most recent snapshots).

### API Layer
Built on FastAPI for performance and Pydantic v2 for strict type safety.
- **Rate Limiting**: Per-IP token bucket limiter.
- **Soft Filtering**: Queries respect the `deleted_at` field, ensuring "forgotten" nodes are invisible before physical compaction.
- **Validation**: Strict edge type enforcement based on the research edge taxonomy.

### SDK
The Python SDK (`HybridMemory`) provides high-level abstractions:
- `recall()`: Hybrid retrieval.
- `trace()`: Semantic-to-graph traversal (finds anchor via vector search, then traverses graph).
- `compact()`: Forces a physical rebuild of FAISS and hard-deletion from SQLite.

## Data Flow: Hybrid Search Request
1. **Validation**: Pydantic validates the request parameters and weights.
2. **Embedding**: The query text is vectorized by `EmbeddingEngine`.
3. **Candidate Selection**: FAISS performs a k-NN search to identify 3x the requested `top_k` results.
4. **Anchor Identification**: If no `anchor_nodes` are provided, the top 3 vector results are used as anchors.
5. **Relational Proximity**: NetworkX calculates shortest path distances from anchors to all candidates.
6. **Fusion**: CRS scores are calculated for each candidate.
7. **Refinement**: Results are re-ranked by CRS and truncated to `top_k`.

## Design Decisions and Trade-offs
- **Exact Vector Search**: Chose `IndexFlatIP` over IVF or HNSW for 100% recall quality. Acceptable up to ~10k nodes based on current L3 cache sizes.
- **Local-First Architecture**: Chose SQLite/NetworkX (local) over Neo4j (remote) to minimize network latency within agent reasoning loops.
- **Graph-Conditioned Embeddings**: Conditioning embeddings at ingest rather than just query-time provides semantic coherence even when graph edges are sparse.

## Scalability Ceiling
- **Memory**: FAISS (n × 384 × 4 bytes) + NetworkX overhead. Estimated ~18MB for 10k nodes.
- **Latency**: O(n·d) grows linearly. The practical ceiling for sub-50ms p95 is estimated at **8,000-10,000 nodes** on modern hardware.
- **GIL**: Python embedding model serializes concurrent requests. Throughput beyond 10 rps requires external embedding services.
