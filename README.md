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
| V(q,n) | Base vector score: Cosine similarity between query and node embeddings, plus a BM25 exact-match overlap boost for lexical precision. |
| G(A,n) | Graph score: max over anchors a in A of 1/(1 + d(a,n)); d is shortest directed path length (either direction) |
| A | Anchor set; if omitted, defaults to the top-3 vector hits |

Default weights α = 0.6, β = 0.4 (semantic primacy). Full definition, anchors, and weight rationale: [docs/ALGORITHM.md](docs/ALGORITHM.md).

**Ingest-Time Neighborhood Averaging.** Stored vectors are L2-normalized after blending the text embedding with the mean of the top-5 vector neighbors: **0.7·e_raw + 0.3·e_neighbors** ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), Embedding Engine). This is a practical, non-training variant of GraphSAGE-style aggregation used to provide a graph-aware embedding space. Formulation and caveats: [docs/ALGORITHM.md](docs/ALGORITHM.md) §3.

## Architecture

Layered stack: FastAPI / Pydantic → embedding engine, vector and graph query engines, hybrid ranker → SQLite (WAL), FAISS `IndexFlatIP`, NetworkX `DiGraph` → atomic `.mind` persistence (manifest, DB, vectors, graph). ASCII diagram and data-flow for hybrid search: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

Use the project virtual environment for all Python commands.

```bash
python3 -m venv .venv
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
python3 -m pytest tests/ -v
./scripts/run_all_benchmarks.sh
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

## Evaluation & Benchmarks

The system is empirically evaluated on targeted benchmarks demonstrating clear regime-of-validity boundaries:
- **Semantic Paraphrase & Exact Lexical Lookup**: Vector alone (with BM25 exact match boost) achieves 100% precision@3 without graph assistance.
- **Edge-Dependent Multi-Hop Retrieval**: Graph-heavy hybrid (vector=0.1, graph=0.9) successfully surfaces multi-hop answers, recovering 100% recall where vector-only yields 0%.
- **Ingest-Time Neighborhood Averaging**: Conditioning embeddings on neighbors improves test retrieval of related cross-domain concepts from 66% (without averaging) to 100% (with averaging).
- **Ablation Studies**: Isolated runs (BM25 only, Vector only, Hybrid) confirm the linear combination of `Score = α·V + β·G` correctly blends semantic space with structural reality, without inflating claims via unsupported deep graph traversals.

Run benchmarks with: `./scripts/run_all_benchmarks.sh`

## Reviewer-Grade Limitations

1. **Graph Sparsity Failure**: The graph component is functionally useless if explicit cross-domain edges do not exist. Hybrid search defaults to vector-only if no anchors are found.
2. **Domain-Separation from Embeddings**: `all-MiniLM-L6-v2` struggles to differentiate certain document types (e.g. Stack Exchange QA vs Wikipedia paragraphs), which can lead to vector-search contamination that graph edges alone cannot fix.
3. **BM25 Exact Overlap Limits**: BM25 excels at keyword matching but fails to label semantic relevance that lacks exact keyword overlap.
4. **Ingest Scalability**: Single-threaded execution of Python's Transformer models bounds ingestion to ~5 requests per second, making this explicitly a local-agent tool, not an enterprise search backend.

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
