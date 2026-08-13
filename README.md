# HybridMind

**HybridMind** is a local hybrid retrieval service for AI memory experiments. It combines FAISS HNSW dense search, an Okapi BM25 index (`bm25s` backend with PyStemmer), a NetworkX directed multigraph, and SQLite. Verified `.mind.zip` snapshots use safe JSON/JSONL derived-index components rather than executable pickle payloads.

repo: [github.com/a3ro-dev/hybridmind](https://github.com/a3ro-dev/hybridmind)

---

## Why

pure vector search drops explicit structural relationships. graph-only search lacks semantic flexibility and degrades when edges are sparse or noisy. agent systems need both: semantic alignment to a query, plus topological traversal and keyword precision—without relying on remote cloud DBs by default.

---

## Technical Architecture

1. **Time-Aware Hybrid Fusion**. Reciprocal Rank Fusion ($k=60$) blends 4096-dimensional dense vectors, BM25 lexical ranks, typed graph proximity, and query-derived time relevance. Request-level `search_mode` controls make vector, sparse, graph, and hybrid ablations real rather than approximate weight changes.
2. **Optional Cross-Encoder Reranking**. When enabled and available, `mixedbread-ai/mxbai-rerank-large-v2` reranks a bounded fusion pool with normalized score blending. Search responses expose whether it executed.
3. **Optional Query Decomposition**. `engine/query_decomposition.py` can split a multi-step question into two or three bounded sub-questions through the centralized LLM policy. It rejects novel named entities, duplicate/oversized output, and lost temporal qualifiers; improvement remains an empirical question.
4. **4096-Dimensional Embedding Invariant**. A remote TEI or OpenAI-compatible embedding endpoint must return exactly 4096 values. Startup, ingestion, and vector insertion fail on any mismatch; there is no local, projected, padded, or lower-dimensional fallback.
5. **Structured Fact Fields**. Narrative facts can carry entities, event time, validity, one of four memory kinds (world, experience, observation, opinion), confidence, supersession state, and optional causal/temporal relations. These fields are only credited when the selected retrieval path consumes them.
6. **Optional Salience and Derived Summaries**. Salience is a configurable recency/access/degree score multiplier. Consolidation creates lossy, provenance-linked retrieval summaries; it is not an Observer/Reflector architecture and cannot archive or replace exact source facts.
7. **Storage Layer (`.mind`)**:
   - SQLite (`store.db` in WAL mode) for nodes, edges, sessions, and metadata
   - `vectors.json`, `graph.jsonl`, and `bm25.jsonl` safe derived-index data
   - `manifest.json` with SHA256 checksums and configured backup rotation
   - runtime FAISS, NetworkX, and BM25 indexes rebuilt from validated data

This project does not replace a transformer KV cache. Its 10M–100M-token target is a preregistered research goal for **retrieval-conditioned effective context**: answer over a large external corpus while sending a bounded evidence subset to a reader. See the protocol below; corpus capacity alone is not evidence that the goal works.

---

## Quick Start

```bash
python3 -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# Unix: source .venv/bin/activate
pip install -r requirements.txt
# First create an offline resource report and a matching live-plan file.
python scripts/offline_resource_frontier.py --output benchmarks/results/offline_resource_frontier.json
python scripts/preflight.py --plan path/to/live-plan.json --validate-only
# Omit --validate-only only when the bounded plan is ready to spend/warm.
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Preflight is deliberately default-deny: a bare command makes no provider calls.
See `docs/RESOURCE_SPEED_TOKENOMICS.md` and
`docs/LIVE_EVAL_PLAN.example.json`.

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

# issue a client-request-controlled signal ablation after preflight/server startup;
# this does not by itself attest the external server commit, config, or corpus
python eval_locomo_retrieval.py --search-mode vector_only --vector-weight 1 --graph-weight 0 --bm25-boost 0 --rerank-pool 0 --no-route-weights --no-track-access
# Graph-only additionally requires a gold-independent explicit anchor manifest;
# a vector-derived anchor is not a pure graph-only ablation.
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

- [AGENTS.md](AGENTS.md) — repository invariants and developer rules
- [docs/ADVERSARIAL_AUDIT_REMEDIATION.md](docs/ADVERSARIAL_AUDIT_REMEDIATION.md) — baseline audit, remediation evidence, residual risks, and scores
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — thread safety, WAL mode, and storage engines
- [docs/ALGORITHM.md](docs/ALGORITHM.md) — RRF fusion formulas and cross-encoder score normalization
- [docs/KV_CACHE_RESEARCH.md](docs/KV_CACHE_RESEARCH.md) — KV working-set hypotheses and evidence
- [docs/RETRIEVAL_RESEARCH_PROTOCOL.md](docs/RETRIEVAL_RESEARCH_PROTOCOL.md) — preregistered quality, scale, latency, resource, and cost gates
- [docs/RESOURCE_SPEED_TOKENOMICS.md](docs/RESOURCE_SPEED_TOKENOMICS.md) — bounded local measurements and live spend admission control
- [demos/techspec.md](demos/techspec.md) — no-code specification for six user-facing demos
