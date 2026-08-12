# AGENT.MD & AGENTS.MD
# HybridMind: Vector + Graph Native Database for AI Retrieval

Version: 2.0 (Phase 6 Complete)  
Date: July 2026  
Project: HybridMind  
Team: Solo / Core Engineering  
Stack: Python / FastAPI / FAISS / NetworkX / SQLite / bm25s / RunPod TEI / Z.AI GLM-4.6  

---

## TABLE OF CONTENTS

1. [Core Directives & Guidelines](#1-core-directives--guidelines)
2. [System Overview](#2-system-overview)
3. [Architecture](#3-architecture)
4. [Scoring and Retrieval (RRF + Reranker)](#4-scoring-and-retrieval-rrf--reranker)
5. [Storage Layer (.mind Format)](#5-storage-layer-mind-format)
6. [Engine Layer & Serverless Handling](#6-engine-layer--serverless-handling)
7. [Evaluation Ledger & Statistics](#7-evaluation-ledger--statistics)
8. [API & SDK Specification](#8-api--sdk-specification)
9. [Command-Line Interface (CLI)](#9-command-line-interface-cli)
10. [Known Module States](#10-known-module-states)

---

## 1. CORE DIRECTIVES & GUIDELINES

When working within this repository, all AI agents and developers must strictly adhere to the following directives:

1. **Package Management**:
   - Always use `pnpm` for frontend/benchmarks dependencies (`memorybench`).
   - Use Python virtual environment (`.venv`) for core backend services and scripts.
2. **Single Source of Truth**:
   - `config.py` (`HYBRIDMIND_*` environment variables) is the authoritative source for configuration.
   - Cross-encoder reranker model is strictly loaded via `settings.reranker_model` (`mixedbread-ai/mxbai-rerank-large-v2`).
   - Z.AI is the canonical production hosted LLM provider: `ZAI_API_KEY`, `ZAI_BASE_URL`, and `HYBRIDMIND_QA_MODEL=glm-4.6`. RunPod vLLM is the self-hosted path. The Hack Club proxy is research/testing-only and requires explicit `HYBRIDMIND_ALLOW_RESEARCH_PROXY=true`; it must never be a silent production or paid-provider fallback.
   - RunPod TEI and vLLM must be warmed and verified with `python scripts/preflight.py` before an evaluation. All embeddings and indexes are exactly 4096-dimensional; no lower-dimensional, local, padded, projected, or mixed-index fallback is permitted.
3. **Performance & Security**:
   - Code must prioritize low latency and thread-safety.
   - Input paths and endpoints must enforce validation and checksum verification.
4. **Configuration-Gated Features**:
   - Experimental or heavy modules (ColBERT, GNN, GAE, Graph-Conditioned Embeddings) are opt-in and gated behind flags, defaulting to off with local CPU fallbacks.

---

## 2. SYSTEM OVERVIEW

### 2.1 What It Is

HybridMind is a local-native hybrid database combining vector embeddings, graph-based structural relationships, and Okapi BM25 lexical search for AI retrieval and agent long-term memory.

### 2.2 Core Capabilities

- **Tri-Signal Retrieval**: Dense Vector (FAISS HNSW), Lexical (bm25s), and Graph Proximity (NetworkX).
- **RRF Fusion**: Reciprocal Rank Fusion ($k=60$) combining vector, graph, and sparse signals without per-corpus tuning.
- **Cross-Encoder Reranking**: `mixedbread-ai/mxbai-rerank-large-v2` re-ranks top fusion candidates with normalized score blending.
- **Query Decomposition**: Multi-hop question decomposition via RunPod vLLM (`engine/query_decomposition.py`).
- **Evaluation QA/Judge**: Z.AI OpenAI-compatible API using `glm-4.6`; it is the canonical LoCoMo answer/judge provider.
- **Resilient Serverless Architecture**: TEI embedding integration with no local fallback when dimension is set to 4096, 300s timeout, and exponential backoff retry.
- **Measurement Ledger & Stats**: Bit-identical determinism logging (`eval_ledger.py`) and bootstrap 95% CI / paired permutation tests (`eval_stats.py`).
- **Atomic Persistence**: `.mind` directory package containing SQLite database (WAL mode), FAISS vector index, NetworkX graph pickle, BM25 index, and SHA-256 verified manifests with 3-backup rotation.

### 2.3 Technology Stack

| Component | Technology | Version / Model |
|-----------|------------|-----------------|
| Backend Framework | FastAPI | 0.115+ |
| Vector Index | FAISS (IndexHNSWFlat) | 1.7.4+ |
| Graph Engine | NetworkX | 3.2.1+ |
| Lexical Engine | bm25s (PyStemmer) | 0.2.0+ |
| Embedding Contract | Remote Qwen3-Embedding-8B | Exactly 4096-dim |
| Remote Serverless Embedding | TEI (Qwen3-Embedding-8B) | 4096-dim |
| Evaluation LLM / Judge | Z.AI | glm-4.6 |
| Cross-Encoder Reranker | mixedbread-ai/mxbai-rerank-large-v2 | Top-25 RRF pool |
| Storage | SQLite (WAL mode) | 3.x |
| CLI | Typer + Rich | `cli/main.py` & `cli/agent.py` |

---

## 3. ARCHITECTURE

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
|  | (Remote backend, |  | (Vector / Graph /|  |  (RRF Fusion +         | |
|  |  exact 4096-dim) |  |  BM25 / MultiHop)|  |   mxbai Cross-Encoder) | |
|  +------------------+  +------------------+  +------------------------+ |
|  +------------------+  +------------------+  +------------------------+ |
|  | Query Decomp.    |  | Fact Extractor   |  | Serverless Retry       | |
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

---

## 4. SCORING AND RETRIEVAL (RRF + RERANKER)

### 4.1 Reciprocal Rank Fusion (RRF)

For each candidate item $d$:
$$RRF(d) = \sum_{m \in \{dense, sparse, graph\}} w_m \cdot \frac{1}{k + r_m(d)}$$

Where:
- $k = 60$ (standard RRF constant)
- $r_m(d)$ is the rank of candidate $d$ in retrieval signal $m$
- Signal weights ($w_{dense}, w_{sparse}, w_{graph}$) are query-type routed via `route_query()`

### 4.2 Cross-Encoder Reranking

The top $N$ candidates (configured by `config.rerank_pool_size`, default 25) are passed through `mixedbread-ai/mxbai-rerank-large-v2`. Scores are blended after independent min-max normalization:
$$Score_{final} = 0.70 \cdot Norm(Score_{RRF}) + 0.30 \cdot Norm(Score_{Reranker})$$

---

## 5. STORAGE LAYER (.MIND FORMAT)

HybridMind persists to a unified `.mind` directory containing:
- `manifest.json`: Snapshot version, timestamp, node/edge counts, and SHA256 checksums.
- `store.db`: SQLite database for nodes, edges, sessions, and metadata.
- `vectors.faiss`: Serialized FAISS IndexHNSWFlat index.
- `graph.nx`: NetworkX Graph pickled binary.
- `bm25.pkl`: Persistent bm25s index object.

Atomic snapshots create timestamped bundles with 3-backup automatic rotation.

---

## 6. ENGINE LAYER & SERVERLESS HANDLING

- **TEI Integration**: Self-hosted HuggingFace TEI endpoint via `RUNPOD_TEI_EMBEDDING_URL`. Configured with a 300-second read timeout to handle cold starts cleanly; preflight actively warms it and validates the 4096-vector response.
- **Serverless Retry**: `engine/serverless_util.py` provides `retry_transient()` executing 6 attempts with exponential backoff up to 120 seconds.
- **Query Decomposition**: `engine/query_decomposition.py` decomposes multi-hop queries into sub-questions via LLM with single-sub-question and novel-entity guards.
- **LoCoMo run order**: run `scripts/preflight.py`, start the API, ingest the sessions, run a small `--n 1 --with-answers` sanity check, then run `--n 0 --with-answers --decompose-multihop`. Preserve the generated ledger.

---

## 7. EVALUATION LEDGER & STATISTICS

- **Measurement Ledger** (`eval_ledger.py`): Emits detailed per-question evaluation logs to `benchmarks/results/ledger_<benchmark>_<confighash>.jsonl`.
- **Statistical Significance** (`eval_stats.py`): Computes 95% bootstrap confidence intervals and paired permutation test p-values for rigourous A/B comparisons.
- **Edge Threshold Sweeping** (`scripts/sweep_edge_threshold.py`): Sweeps auto-edge cosine thresholds and evaluates multi-hop graph reachability.

---

## 8. API & SDK SPECIFICATION

- **Python SDK**: `sdk/memory.py` provides `HybridMemory` interface (`store()`, `recall()`, `relate()`, `forget()`).
- **REST Endpoints**:
  - `POST /nodes`, `GET /nodes/{id}`, `DELETE /nodes/{id}`
  - `POST /edges`, `GET /edges/node/{id}`
  - `POST /search/hybrid` (tri-signal RRF + rerank)
  - `POST /ingest/session-facts` (structured LLM fact extraction)
  - `GET /health`, `GET /ready`, `POST /snapshot`

---

## 9. COMMAND-LINE INTERFACE (CLI)

- **Typer Ops CLI** (`cli/main.py`): `python -m cli.main [nodes|search|snapshot|health|stats]`
- **Interactive Shell** (`cli/agent.py`): `python cli/agent.py` with commands `/memory`, `/stats`, `/sessions`, `/archive`, `/forget`, `/clear`, `/help`, `/exit`.
- **Snapshot Inspector** (`cli/mind.py`): Inspects `.mind` manifests and SQLite tables directly.

---

## 10. KNOWN MODULE STATES

| Module | Status | Notes |
|--------|--------|-------|
| Tri-Signal RRF + mxbai Reranker | ✅ **ACTIVE** | Default production retrieval pipeline |
| Multi-hop Decomposition | ✅ **ACTIVE** | Enabled via `--decompose-multihop` in eval scripts |
| TEI / vLLM Serverless | ✅ **ACTIVE** | Priority backends with robust retry policy |
| GNN Reranker (`scripts/train_gnn.py`) | ⚠️ **SCAFFOLDED** | Code complete; requires pre-training checkpoint |
| Fusion MLP (`scripts/train_fusion_mlp.py`) | ⚠️ **SCAFFOLDED** | Code complete; falls back to heuristic init |
| Contrastive Fine-Tuning (`scripts/train_contrastive.py`) | ⚠️ **SCAFFOLDED** | Code complete; opt-in training script |
