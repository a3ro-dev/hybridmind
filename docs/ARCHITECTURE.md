# HybridMind Architecture

## Overview
HybridMind is a local-native hybrid vector + graph database designed for AI agent memory. It unifies semantic similarity, lexical matching, relational context, learned re-ranking, and multi-hop query decomposition into a self-contained implementation combining FAISS, NetworkX, SQLite, bm25s, and optional ColBERT/GNN modules.

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
|  | (Remote TEI /    |  | (Vector / Graph /|  |  (Time-aware RRF +     | |
|  | OpenAI, 4096-dim)|  |  BM25 / MultiHop)|  |   mxbai Cross-Encoder) | |
|  +------------------+  +------------------+  +------------------------+ |
|  +------------------+  +------------------+  +------------------------+ |
|  | Query Decomp.    |  | Fact Extractor   |  | Serverless Resilience  | |
|  | (RunPod vLLM)    |  | (structured JSON)|  | (6 attempts, 120s max) | |
|  +------------------+  +------------------+  +------------------------+ |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                             Storage Layer                               |
|  +-----------------+  +--------------------+  +--------------------+    |
|  |  SQLite Store   |  |  Vector / BM25     |  |    Graph Index     |    |
|  | (WAL Enabled)   |  | (FAISS HNSW /bm25s)|  |   (NetworkX)       |    |
|  +-----------------+  +--------------------+  +--------------------+    |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                            Persistence                                  |
|  (.mind directory: manifest.json / store.db / vectors / graph.nx /     |
|   bm25.pkl) with SHA256 integrity & 3-backup rotation                   |
+-------------------------------------------------------------------------+
```

## Component Deep Dives

### Embedding Engine

**Backend Selection (4096-dimensional invariant)**:

1. **TEI (Text Embeddings Inference)** — self-hosted HuggingFace TEI endpoint
   - **Config**: `RUNPOD_TEI_EMBEDDING_URL` (base URL), `RUNPOD_API_KEY` (Bearer token)
   - **Model**: Qwen3-Embedding-8B (4096-dim native) or any HF model deployed on TEI
   - **Protocol**: Raw TEI `/embed` endpoint; returns `List[List[float]]` directly
   - **Timeout & Retry**: Configured with a 300-second read timeout for cold starts and 6 retries with exponential backoff up to 120 seconds.
   - **No fallback**: every response is validated against the 4096-dimensional FAISS index.

2. **RemoteEmbeddingEngine** — OpenAI-compatible remote embeddings
   - **Config**: `HC_EMBEDDING_URL` (Hack Club AI proxy) or `RUNPOD_EMBEDDING_URL`
   - The response must already be exactly 4096 dimensions. No truncation, projection, padding, local model, or alternate index is permitted.

### LLM Inference Engine & Query Decomposition

1. **RunPod vLLM** — self-hosted serverless vLLM endpoint (`engine/runpod_llm.py`)
   - **Config**: `RUNPOD_LLM_ENDPOINT_ID`, `RUNPOD_API_KEY`, `RUNPOD_LLM_MODEL` (default: `qwen/qwen3.5-9b`)
   - **Protocol**: Job-queue based `/run` with polling and automatic JSON schema validation.
2. **Multi-Hop Query Decomposition** (`engine/query_decomposition.py`)
   - Decomposes multi-step complex questions into targeted sub-queries.
   - Guarded against single-sub-question loops and hallucinations via novel-entity checks.
3. **Hosted provider policy** (`engine/llm_client.py`)
   - Z.AI `glm-4.6` is canonical for production hosted inference.
   - RunPod vLLM is the supported self-hosted path.
   - `ai.hackclub.com` is permitted only when `HYBRIDMIND_ALLOW_RESEARCH_PROXY=true`; research mode never silently spills over to paid Z.AI.

### Late Fusion Scoring & Reranking

- **RRF Fusion ($k=60$)**: Combines dense vector, BM25 lexical, typed graph proximity, and temporal relevance with query-type-specific weights stored in `config.py`.
- **Cross-Encoder Reranker**: `mixedbread-ai/mxbai-rerank-large-v2` re-ranks the top fusion pool (`config.rerank_pool_size`, default 25).
- **Normalized Score Blending**: $0.70 \cdot Norm(Score_{RRF}) + 0.30 \cdot Norm(Score_{Reranker})$, preventing pure-text rerankers from pruning graph-discovered candidates on multi-hop queries.

### Evaluation & Statistics Tools

- `eval_ledger.py`: Emits determinism and evaluation logs to `benchmarks/results/ledger_<benchmark>_<hash>.jsonl`.
- `eval_stats.py`: Performs bootstrap 95% confidence interval estimation and paired permutation tests for statistical significance.
- `scripts/sweep_edge_threshold.py`: Evaluates cosine thresholds for auto-edges against multi-hop graph reachability.
- `scripts/ablation_matrix.py`: Emits deterministic nine-condition experiment plans and annotates completed ledgers without claiming unrun results.

### Storage Layer

#### SQLite Store
- **Persistence**: SQLite in WAL mode. Nodes store text, metadata, 4096-dimensional embedding blobs, event/valid time, memory kind, confidence, access state, archive provenance, and logical deletion. `node_entities` stores canonical entity mentions; edges retain typed, temporal, confidence, and supersession fields.
- **Graph semantics**: NetworkX `MultiDiGraph` preserves parallel typed relations between the same node pair.
- **Atomic Operations**: Manifest SHA-256 verification and 3-backup directory rotation on save/export.
