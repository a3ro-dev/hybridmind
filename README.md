# HybridMind

**HybridMind** is a local-native hybrid vector–graph store for AI agent memory. It combines FAISS HNSW approximate search, an Okapi BM25 index (`bm25s` backend with PyStemmer stemming), a NetworkX directed graph, and SQLite into a single `.mind` directory format with SHA256-verified atomic snapshots and 3-backup rotation. Repository: [github.com/a3ro-dev/hybridmind](https://github.com/a3ro-dev/hybridmind).

---

## Problem

Pure vector retrieval ignores explicit relational structure; graph-only retrieval lacks semantic filtering and scales poorly when edges are sparse or noisy. Agent memory systems need both: semantic alignment to the query and re-ranking or traversal grounded in declared relationships, without mandatory remote services.

---

## Approach

HybridMind is an engineering system that applies production hybrid retrieval techniques with optional self-hosted cloud acceleration:

1. **RRF Tri-Signal Fusion**. Reciprocal Rank Fusion ($k=60$) combines Dense Vector, BM25 Lexical, and Graph Proximity into a single rank score using query-type specific weights (`vector_weight`, `graph_weight`, `bm25_boost_weight`) routed via `route_query()`.
2. **Pre-trained Cross-Encoder Reranker**. `mixedbread-ai/mxbai-rerank-large-v2` re-ranks top fusion candidates (configured by `config.rerank_pool_size`, default 25) with 70% fusion / 30% cross-encoder normalized blending. Both RRF and cross-encoder scores are independently min-max normalized before blending, preserving graph candidates on multi-hop queries.
3. **Multi-Hop Query Decomposition**. `engine/query_decomposition.py` automatically decomposes multi-step questions into sub-questions via LLM inference, protected by single-sub-question and novel-entity guards.
4. **Flexible Embedding Backends**:
   - **TEI (Text Embeddings Inference)** — self-hosted HuggingFace TEI endpoint via `RUNPOD_TEI_EMBEDDING_URL` (e.g., Qwen3-Embedding-8B, native 4096-dim). Configured with a 300s timeout and 6 exponential-backoff retries. No local fallback by design when dimension is 4096.
   - **OpenAI-compatible remote** — `HC_EMBEDDING_URL` (Hack Club AI proxy) or `RUNPOD_EMBEDDING_URL` with MRL truncation.
   - **Local embeddings** — `BAAI/bge-m3` (1024-dim default) or `all-mpnet-base-v2` (768-dim CPU fallback).
5. **Self-Hosted LLM Integration** (`engine/runpod_llm.py`): Fact extraction and consolidation automatically use RunPod vLLM (`qwen/qwen3.5-9b` or configured model) with structured JSON schema output and retry handling.
6. **Measurement Ledger & Significance Testing**:
   - `eval_ledger.py` records bit-identical per-question retrieval and QA logs to `benchmarks/results/ledger_<benchmark>_<hash>.jsonl`.
   - `eval_stats.py` provides bootstrap 95% confidence intervals and paired permutation tests for rigorous A/B performance evaluation.
7. **Auto-Edge Threshold Sweeper** (`scripts/sweep_edge_threshold.py`): Tools to sweep cosine similarity thresholds for auto-edge creation and measure 2-hop graph reachability.

---

## Architecture

Layered stack:

```text
FastAPI / Pydantic v2 / Request Timing Middleware
  ↓
Embedding Pipeline (TEI Qwen3 4096-dim → OpenAI-compat → local bge-m3)
BM25 Index (bm25s backend with PyStemmer)
Vector / Graph / BM25 / MultiHop Query Engines
Query Decomposition Engine (engine/query_decomposition.py)
Hybrid Ranker with RRF Fusion (k=60) + mxbai Cross-Encoder Reranker
  ↓
Persistent Storage Layer:
  - SQLite (WAL mode) for nodes, edges, metadata, sessions
  - FAISS IndexHNSWFlat for vector search
  - NetworkX DiGraph for graph operations
  - Persistent bm25s index object
  ↓
Atomic .mind Storage Package (manifest with SHA256 checksums, 3-backup rotation)
```

Centralized GPU/CPU device resolution via `engine/device.py` (`cuda` > `mps` > `cpu`). All parameters are configurable via `HYBRIDMIND_*` environment variables in `config.py`.

---

## Quick Start

Use the project virtual environment for all Python commands.

```bash
python3 -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Unix: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Self-Hosted RunPod TEI & vLLM Setup

```bash
export RUNPOD_API_KEY="your-runpod-api-key"
export RUNPOD_TEI_EMBEDDING_URL="https://<tei-endpoint>.api.runpod.ai"
export HYBRIDMIND_EMBEDDING_DIMENSION=4096
export RUNPOD_LLM_ENDPOINT_ID="your-vllm-endpoint-id"
export RUNPOD_LLM_MODEL="qwen/qwen3.5-9b"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Python SDK (`sdk/memory.py`)

```python
from sdk.memory import HybridMemory

memory = HybridMemory(base_url="http://127.0.0.1:8000")
nid = memory.store("Transformer models use self-attention mechanism.")
memory.relate(nid, "target-node-uuid", "derived_from")
results = memory.recall("attention mechanisms", top_k=5, mode="hybrid")
```

### Command-Line Interface (`cli/`)

- **Administrative CLI** ([cli/main.py](cli/main.py)):
  ```bash
  python -m cli.main search "attention mechanism" --mode hybrid --top-k 5
  python -m cli.main snapshot create --label "backup-1"
  python -m cli.main stats
  ```
- **Interactive Terminal Shell** ([cli/agent.py](cli/agent.py)):
  ```bash
  python cli/agent.py
  ```

### Streamlit UI ([ui/app.py](ui/app.py))

```bash
streamlit run ui/app.py
```

### Tests and Benchmarks

```bash
python3 -m pytest tests/ -v                       # Unit and integration test suite
python eval_locomo_retrieval.py --with-answers    # LoCoMo retrieval & QA evaluation
python eval_stats.py compare <ledger_A> <ledger_B> # Statistical significance testing
python scripts/sweep_edge_threshold.py            # Sweep auto-edge cosine thresholds
```

---

## API Summary

| Area | Methods & Endpoints |
|------|---------------------|
| Nodes | `POST /nodes`, `GET /nodes`, `GET /nodes/{id}`, `PUT /nodes/{id}`, `DELETE /nodes/{id}` |
| Edges | `POST /edges`, `GET /edges`, `DELETE /edges/{id}`, `GET /edges/node/{node_id}`, `GET /edges/types` |
| Search | `POST /search/vector`, `GET /search/graph`, `POST /search/hybrid`, `POST /search/compare`, `GET /search/stats` |
| Ingest | `POST /ingest/session-facts` (structured LLM fact extraction) |
| Ops | `GET /health`, `GET /ready`, `POST /snapshot`, `GET /database`, `POST /admin/compact` |

---

## Documentation Index

- [AGENTS.md](AGENTS.md) — System specifications, coding guidelines, and agent rules.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture, storage layers, and thread safety.
- [docs/ALGORITHM.md](docs/ALGORITHM.md) — Mathematical formulas for RRF, reranking, and graph scoring.
- [docs/AGENT_INTEGRATION.md](docs/AGENT_INTEGRATION.md) — SDK, MCP server, and REST API integration guide.
- [docs/PHASE_6_REALISTIC.md](docs/PHASE_6_REALISTIC.md) — Phase 6 implementation roadmap and evaluation setup.
- [docs/MULTI_DOMAIN_EVAL.md](docs/MULTI_DOMAIN_EVAL.md) — Evaluation harness, measurement ledger, and stats guide.
- [docs/LOCAL_SETUP_GUIDE.md](docs/LOCAL_SETUP_GUIDE.md) — Local setup, environment configuration, and server startup.
- [cli/README.md](cli/README.md) — Command Line Interface documentation.
