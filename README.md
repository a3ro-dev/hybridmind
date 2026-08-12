# HybridMind

**HybridMind** is a local-native hybrid vector–graph database built for AI retrieval and agent long-term memory. it combines FAISS HNSW dense search, an Okapi BM25 index (`bm25s` backend with PyStemmer), a NetworkX directed graph, and SQLite into a single atomic `.mind` bundle with SHA256-verified manifests and 3-backup rotation.

repo: [github.com/a3ro-dev/hybridmind](https://github.com/a3ro-dev/hybridmind)

---

## Why

pure vector search drops explicit structural relationships. graph-only search lacks semantic flexibility and degrades when edges are sparse or noisy. agent systems need both: semantic alignment to a query, plus topological traversal and keyword precision—without relying on remote cloud DBs by default.

---

## Technical Architecture

1. **Time-Aware Hybrid Fusion**. Reciprocal Rank Fusion ($k=60$) blends 4096-dimensional dense vectors, BM25 lexical ranks, typed graph proximity, and query-derived time relevance. Request-level `search_mode` controls make vector, sparse, graph, and hybrid ablations real rather than approximate weight changes.
2. **Cross-Encoder Reranking**. `mixedbread-ai/mxbai-rerank-large-v2` reranks top 25 RRF candidate nodes with min-max normalized score blending (70% RRF / 30% cross-encoder).
3. **Multi-Hop Query Decomposition**. `engine/query_decomposition.py` splits multi-step questions into sub-questions via LLM inference with single-sub-question and novel-entity guards.
4. **4096-Dimensional Embedding Invariant**. A remote TEI or OpenAI-compatible embedding endpoint must return exactly 4096 values. Startup, ingestion, and vector insertion fail on any mismatch; there is no local, projected, padded, or lower-dimensional fallback.
5. **Structured Temporal Memory**. Narrative facts carry normalized entities, event time, validity, one of four memory kinds (world, experience, observation, opinion), confidence, supersession state, and optional causal/temporal relations.
6. **Salience and Compression**. Opt-in ACT-R-inspired access/recency/centrality scoring and observer/reflector consolidation can archive source nodes from retrieval while preserving their rows and provenance edges.
7. **Storage Layer (`.mind`)**:
   - SQLite (`store.db` in WAL mode) for nodes, edges, sessions, and metadata
   - FAISS (`vectors.faiss`) for HNSW index
   - NetworkX (`graph.nx`) binary graph pickle
   - Persistent `bm25.pkl` index object
   - `manifest.json` with SHA256 checksums and automated 3-backup rotation

---

## Quick Start

```bash
python3 -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# Unix: source .venv/bin/activate
pip install -r requirements.txt
python scripts/preflight.py
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Python SDK (`sdk/memory.py`)

```python
from sdk.memory import HybridMemory

memory = HybridMemory(base_url="http://127.0.0.1:8000")
nid = memory.store("Transformer models use self-attention mechanisms.")
memory.relate(nid, "target-node-uuid", "derived_from")
results = memory.recall("attention mechanisms", top_k=5, mode="hybrid")
```

### CLI & Evaluation

```bash
# search CLI
python -m cli.main search "attention mechanism" --mode hybrid --top-k 5

# evaluation & statistical significance testing
python eval_locomo_retrieval.py --with-answers
python eval_stats.py compare <ledger_A> <ledger_B>

# review the controlled experiment matrix without making network calls
python scripts/ablation_matrix.py --list
python scripts/ablation_matrix.py --dry-run --benchmark locomo

# execute deterministic signal ablations after preflight and server startup
python eval_locomo_retrieval.py --search-mode vector_only --vector-weight 1 --graph-weight 0 --bm25-boost 0 --rerank-pool 1 --no-route-weights --no-track-access
python eval_locomo_retrieval.py --search-mode graph_only --graph-anchor-strategy vector_top1 --vector-weight 0 --graph-weight 1 --bm25-boost 0 --rerank-pool 1 --no-route-weights --no-track-access
```

---

## API Summary

| Category | Endpoints |
|---|---|
| Nodes | `POST /nodes`, `GET /nodes`, `GET /nodes/{id}`, `PUT /nodes/{id}`, `DELETE /nodes/{id}` |
| Edges | `POST /edges`, `GET /edges`, `DELETE /edges/{id}`, `GET /edges/node/{id}` |
| Search | `POST /search/vector`, `GET /search/graph`, `POST /search/hybrid`, `POST /search/compare` |
| Ingest | `POST /ingest/session-facts` (structured LLM fact extraction) |
| Ops | `GET /health`, `GET /ready`, `POST /snapshot`, `GET /database` |

---

## Documentation Index

- [AGENTS.md](AGENTS.md) — system specs and developer rules
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — thread safety, WAL mode, and storage engines
- [docs/ALGORITHM.md](docs/ALGORITHM.md) — RRF fusion formulas and cross-encoder score normalization
- [docs/KV_CACHE_RESEARCH.md](docs/KV_CACHE_RESEARCH.md) — KV working-set hypotheses and evidence
- [demos/techspec.md](demos/techspec.md) — no-code specification for six user-facing demos
