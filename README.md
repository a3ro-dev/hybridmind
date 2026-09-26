<p align="center">
  <img src="docs/assets/banner.png" alt="HybridMind — Vector + Graph Native Database for AI Retrieval" width="100%" />
</p>

<p align="center">
  <strong>Local-first dense, sparse, and graph retrieval service for AI memory experiments.</strong>
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Status-Active%20Research-00e5ff?style=for-the-badge&logoColor=black" alt="Status"></a>
  <a href="#technical-architecture"><img src="https://img.shields.io/badge/Architecture-Dense%20%2B%20Sparse%20%2B%20Graph-ff007f?style=for-the-badge" alt="Architecture"></a>
  <a href="tests/"><img src="https://img.shields.io/badge/Tests-390%2B%20Offline%20Passing-00d2d3?style=for-the-badge" alt="Tests"></a>
  <a href="AGENTS.md"><img src="https://img.shields.io/badge/Storage-Authoritative%20SQLite-16e0bd?style=for-the-badge" alt="Storage"></a>
</p>

---

## 🪟 The Premise

most vector databases give you semantic similarity but stay completely blind to explicit graph relationships, and keyword search usually lives in a disconnected silo. HybridMind fixes that retrieval disconnect locally without turning it into a bloated cloud cluster.

HybridMind is a local-first dense, sparse, and graph retrieval service designed for agent memory and long-context experiments. instead of treating retrieval like a black box, it unifies FAISS HNSW vector indexing, BM25/BM25S lexical search, and NetworkX structural traversals on top of an authoritative, bitemporal SQLite/WAL persistence layer.

all candidates are merged through weighted reciprocal rank fusion (RRF) with temporal filtering, giving you explainable, provenance-backed memory instead of hallucinated context. everything packages into portable, cryptographically verified `.mind` snapshots, backed by ~400 offline unit and contract tests with zero network leaks.

the bet here is simple: local, inspectable hybrid retrieval gives agents way more reliable grounded reasoning than throwing raw tokens into massive context windows and hoping for the best.

---

## ⚡ At a glance

| Area | What HybridMind does |
|---|---|
| **Retrieval** | FAISS HNSW dense search, Okapi BM25 (`bm25s` + PyStemmer), and a typed NetworkX directed multigraph |
| **Ranking** | Time-aware weighted reciprocal-rank fusion (`k=60`), with independently controlled retrieval modes |
| **Evidence** | Corpus/session scoping and exact evidence IDs for retrieval metrics |
| **Persistence** | SQLite/WAL as authoritative source of truth; runtime indexes rebuilt from validated data |
| **Portability** | Verified `.mind.zip` snapshots using checksummed JSON/JSONL, never executable pickles |
| **Embeddings** | Remote native embeddings only, validated to exactly 4096 dimensions |

---

## 🕹️ Why Hybrid Retrieval

Pure vector search can miss an explicit relation or exact term. Graph-only retrieval loses semantic flexibility and gets brittle when the graph is sparse or noisy. HybridMind keeps these as separate candidate paths, then fuses them so each path can be measured, ablated, and improved independently.

```
                  ┌───────────────────────┐
                  │      User Query       │
                  └──────────┬────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   FAISS HNSW    │ │    BM25S Lex    │ │ NetworkX Graph  │
│ Dense Vector    │ │  Sparse Keyword │ │ Entity & Citations│
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
              ┌─────────────────────────────┐
              │ Reciprocal Rank Fusion (k=60)│
              │  + Temporal Scoping & Rank  │
              └──────────────┬──────────────┘
                             ▼
              ┌─────────────────────────────┐
              │  Explainable Evidence Set   │
              └─────────────────────────────┘
```

### Design stance

- **Fail closed** on malformed provider output, corrupt persistence, partial batches, and invalid benchmark provenance.
- **Derived indexes are projections**, not authoritative records. SQLite remains the single source of truth.
- **Live provider calls are opt-in and budgeted**; the offline test suite makes zero provider calls.
- **No answer-string shortcuts**: answer-string overlap is not counted as retrieval evidence recall.

---

## ⚙️ Technical Architecture

1. **Time-Aware Hybrid Fusion**. Reciprocal Rank Fusion ($k=60$) blends 4096-dimensional dense vectors, BM25 lexical ranks, typed graph proximity, and query-derived time relevance. Request-level `search_mode` controls make vector, sparse, graph, and hybrid ablations real rather than approximate weight changes.
2. **Tri-Signal Retrieval (`POST /retrieve`)**. Three independent channels rank each scope: exact dense search over native 4096-d vectors, BM25S sparse search, and an entity–memory graph ported from EMG (exact parity with upstream on its LoCoMo artifacts) with query-derived anchors and optional HippoRAG-2 personalized PageRank. Results are fused (RRF k=60, DBSF, z-score), optionally reranked inside a fixed pool, and packed into a token-budgeted evidence set with stable evidence IDs. See `docs/REPRODUCTION_MAP.md`.
3. **Optional Cross-Encoder Reranking**. When enabled, `BAAI/bge-reranker-v2-m3` (local, or via a TEI `/rerank` endpoint) reranks a bounded fusion pool. Responses expose whether it executed.
4. **Optional Query Decomposition**. `engine/query_decomposition.py` can split a multi-step question into two or three bounded sub-questions through the centralized LLM policy. It rejects novel named entities, duplicate/oversized output, and lost temporal qualifiers; improvement remains an empirical question.
5. **4096-Dimensional Embedding Invariant**. A remote TEI or OpenAI-compatible embedding endpoint must return exactly 4096 values. Startup, ingestion, and vector insertion fail on any mismatch; there is no local, projected, padded, or lower-dimensional fallback.
6. **Structured Fact Fields**. Narrative facts can carry entities, event time, validity, one of four memory kinds (world, experience, observation, opinion), confidence, supersession state, and optional causal/temporal relations. These fields are only credited when the selected retrieval path consumes them.
7. **Optional Salience and Derived Summaries**. Salience is a configurable recency/access/degree score multiplier. Consolidation creates lossy, provenance-linked retrieval summaries; it is not an Observer/Reflector architecture and cannot archive or replace exact source facts.
8. **Storage Layer (`.mind`)**:
   - SQLite (`store.db` in WAL mode) for nodes, edges, sessions, and metadata
   - `vectors.json`, `graph.jsonl`, and `bm25.jsonl` safe derived-index data
   - `manifest.json` with SHA256 checksums and configured backup rotation
   - runtime FAISS, NetworkX, and BM25 indexes rebuilt from validated data

This project does not replace a transformer KV cache. Its 10M–100M-token target is a preregistered research goal for **retrieval-conditioned effective context**: answer over a large external corpus while sending a bounded evidence subset to a reader. See the protocol below; corpus capacity alone is not evidence that the goal works.

---

## 🚀 Quick Start

```bash
python3 -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# Unix: source .venv/bin/activate
pip install -r requirements.txt      # or: python install.py (venv + .env + MCP wiring)
cp .env.example .env                 # fill in provider keys; config.py is authoritative
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

## 🔌 API Summary

| Category | Endpoints |
|---|---|
| Nodes | `POST /nodes`, `GET /nodes`, `GET /nodes/{id}`, `PUT /nodes/{id}`, `DELETE /nodes/{id}` |
| Edges | `POST /edges`, `GET /edges`, `DELETE /edges/{id}`, `GET /edges/node/{id}` |
| Search | `POST /search/vector`, `GET /search/graph`, `POST /search/hybrid`, `POST /search/compare` |
| Tri-signal retrieval | `POST /retrieve` (channels, fusion, rerank pool, evidence budget, scope) |
| Ingest | `POST /ingest/session-facts` (structured LLM fact extraction) |
| Ops | `GET /health`, `GET /ready`, `POST /snapshot`, `GET /database` |

---

## 📚 Documentation Index

- [AGENTS.md](AGENTS.md) — agent/developer contract: rules, load-bearing map, doc ownership
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — request/data flow, storage engines, security posture
- [docs/ALGORITHM.md](docs/ALGORITHM.md) — RRF fusion formulas and cross-encoder score normalization
- [docs/EVALUATION.md](docs/EVALUATION.md) — evaluator usage, ledger schema, statistical conventions
- [docs/AGENT_INTEGRATION.md](docs/AGENT_INTEGRATION.md) — SDK / MCP / structured-ingestion contracts
- [cli/README.md](cli/README.md) — CLI command surfaces
- [PHASE_IMPLEMENTATION_STATUS.md](PHASE_IMPLEMENTATION_STATUS.md) — real vs scaffolded inventory
- [docs/ADVERSARIAL_AUDIT_REMEDIATION.md](docs/ADVERSARIAL_AUDIT_REMEDIATION.md) — baseline audit, remediation evidence, residual risks, and scores
- [docs/KV_CACHE_RESEARCH.md](docs/KV_CACHE_RESEARCH.md) — KV working-set hypotheses and evidence
- [docs/RETRIEVAL_RESEARCH_PROTOCOL.md](docs/RETRIEVAL_RESEARCH_PROTOCOL.md) — preregistered quality, scale, latency, resource, and cost gates
- [docs/RESOURCE_SPEED_TOKENOMICS.md](docs/RESOURCE_SPEED_TOKENOMICS.md) — bounded local measurements and live spend admission control
- [demos/techspec.md](demos/techspec.md) — no-code specification for six user-facing demos

The full registry of every tracked document (with ownership and update
triggers) is the Documentation Map in `AGENTS.md`.
