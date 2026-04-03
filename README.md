# HybridMind

**HybridMind** is a local-native hybrid vector–graph store for agent memory. It provides a clean, self-contained implementation that combines FAISS exact inner-product search, an Okapi BM25 index with NLTK stemming, a NetworkX directed graph, and SQLite into a single `.mind` file format. Repository: [github.com/a3ro-dev/hybridmind](https://github.com/a3ro-dev/hybridmind).

## Problem

Pure vector retrieval ignores explicit relational structure; graph-only retrieval lacks semantic filtering and scales poorly when edges are sparse or noisy. Agent memory systems need both: semantic alignment to the query and re-ranking or traversal grounded in declared relationships, without mandatory remote services.

## Approach

HybridMind is an engineering system that correctly applies known hybrid retrieval techniques without external cloud dependencies. 

**Late Fusion Scoring.** Hybrid retrieval ranks candidates by a weighted linear score fusion—a well-known late fusion technique in information retrieval—combining vector similarity and graph proximity:

```text
Score(q,n) = α·V(q,n) + β·G(A,n),     α + β = 1
```

| Symbol | Meaning |
|--------|---------|
| q, n | Query and candidate node |
| V(q,n) | Cosine similarity between query and node embeddings |
| G(A,n) | Graph score: max over anchors a in A of 1/(1 + d(a,n)); d is shortest directed path length (either direction) |
| A | Anchor set; if omitted, defaults to the top-3 vector hits |

Default weights α = 0.6, β = 0.4 (semantic primacy). Full definition, anchors, and weight rationale: [docs/ALGORITHM.md](docs/ALGORITHM.md).

**Ingest-Time Neighborhood Averaging.** Stored vectors are L2-normalized after blending the text embedding with the mean of the top-5 vector neighbors: **0.7·e_raw + 0.3·e_neighbors** ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), Embedding Engine). This is a practical, non-training variant of GraphSAGE-style aggregation used to provide a graph-aware embedding space. Formulation and caveats: [docs/ALGORITHM.md](docs/ALGORITHM.md) §3.

## Architecture

Layered stack: FastAPI / Pydantic → embedding engine, vector and graph query engines, hybrid ranker → SQLite (WAL), FAISS `IndexFlatIP`, NetworkX `DiGraph` → atomic `.mind` persistence (manifest, DB, vectors, graph). ASCII diagram and data-flow for hybrid search: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

Use the project virtual environment for all Python commands.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Unix: source .venv/bin/activate
pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

**Python SDK** ([sdk/memory.py](sdk/memory.py)):

```python
from sdk.memory import HybridMemory

memory = HybridMemory(base_url="http://127.0.0.1:8000")
nid = memory.store("Transformer models use self-attention.")
memory.relate(nid, "other-node-uuid", "derived_from")
results = memory.recall("attention mechanisms", top_k=5, mode="hybrid")
```

**Tests and benchmarks:**

```bash
python -m pytest tests/ -v
python benchmarks/run_benchmarks.py
```

Further integration notes: [docs/AGENT_INTEGRATION.md](docs/AGENT_INTEGRATION.md).

## API overview

| Area | Methods (HTTP) |
|------|----------------|
| Nodes | `POST/GET/PUT/DELETE /nodes`, `GET /nodes/{id}` |
| Edges | `POST/GET/PUT/DELETE /edges`, `GET /edges/node/{node_id}`, `GET /edges/types` |
| Search | `POST /search/vector`, `GET /search/graph`, `POST /search/hybrid`, `POST /search/compare`, `GET /search/path/{source}/{target}`, `GET /search/stats` |
| Bulk | `POST /bulk/nodes`, `POST /bulk/edges`, `POST /bulk/import`, `POST /bulk/unstructured`, `DELETE /bulk/clear` |
| Comparison | `POST /comparison/effectiveness` |
| Ops | `GET /health`, `GET /ready`, `GET /live`, `POST /snapshot`, `POST /cache/clear`, `POST /admin/compact`, `POST /admin/clear` |

**SDK:** `HybridMemory.store`, `relate`, `recall` (`mode`: `hybrid` \| `vector`), `trace` (vector anchor then `GET /search/graph`), `forget`, `compact`, `stats`.

## Evaluation

The system is empirically evaluated against the LoCoMo benchmark with honest reporting of failures: we observed a **36% overall accuracy**, and notably a **0% correlation/accuracy on single-hop queries**. 

Multi-corpus load and retrieval experiments (Wikipedia, Stack Exchange, PubMed QA, AG News, legal text), **7,510 nodes**, embedding `all-MiniLM-L6-v2`, fusion weights 0.6 / 0.4: [docs/MULTI_DOMAIN_EVAL.md](docs/MULTI_DOMAIN_EVAL.md).

| Finding | Reference |
|--------|-----------|
| Cross-domain `analogous_to` edges at cosine 0.45: **6**; at **0.20**: **49** — still **0/10** concept queries with hybrid vs vector top-10 set change | §1–3, §5 |
| Identical top-10 membership vector vs hybrid for all 10 cross-domain concept queries; mean unique domains **2.5** both modes | §3.1 |
| Linked-partner “hidden gem” recall: vector **6/6**, hybrid **6/6**; hybrid improved rank (mean **~3.2** positions) without new members in top-10 | §3.3, §5 |
| Mean raw vs conditioned embedding separation (probes): **0.01927** vs **0.00977** single-corpus baseline in [docs/ALGORITHM.md](docs/ALGORITHM.md) | §4.3 |
| Hybrid latency (100 queries, ~7.5k nodes): wall **16.51 ms** p50, **19.64 ms** p95; vs **13 / 16 ms** p50/p95 at 1k nodes | §3.5 |

ArXiv-scale ablation (NDCG, α sweep) and BM25-limitations caveat: [docs/ALGORITHM.md](docs/ALGORITHM.md) §2.5, §4. Micro-benchmarks (single-machine CPU): [benchmarks/PERFORMANCE.md](benchmarks/PERFORMANCE.md).

## Citation

```bibtex
@software{hybridmind2025,
  title        = {HybridMind: Local-Native Hybrid Vector--Graph Memory},
  author       = {a3ro-dev},
  year         = {2025},
  url          = {https://github.com/a3ro-dev/hybridmind}
}
```

## License

[MIT License](LICENSE).
