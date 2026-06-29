# HybridMind

**HybridMind** is a local-native hybrid vector–graph store for agent memory. It combines FAISS HNSW approximate search, an Okapi BM25 index with NLTK stemming, a NetworkX directed graph, and SQLite into a single `.mind` directory format with SHA256-verified atomic snapshots and 3-backup rotation. Repository: [github.com/a3ro-dev/hybridmind](https://github.com/a3ro-dev/hybridmind).

## Problem

Pure vector retrieval ignores explicit relational structure; graph-only retrieval lacks semantic filtering and scales poorly when edges are sparse or noisy. Agent memory systems need both: semantic alignment to the query and re-ranking or traversal grounded in declared relationships, without mandatory remote services.

## Approach

HybridMind is an engineering system that correctly applies known hybrid retrieval techniques without external cloud dependencies. 

**Late Fusion Scoring.** Hybrid retrieval ranks candidates by a weighted linear score fusion—a well-known late fusion technique in information retrieval—combining vector similarity and graph proximity:

```text
Score(q,n) = w_v · V_eff(q,n) + w_g · G_eff(A,n)
```

Where `w_v=0.5`, `w_g=0.15`, and a BM25 keyword overlap boost (`w_bm25=0.35`) is applied within the vector score. Weights do not sum to 1.0.

| Symbol | Meaning |
|--------|---------|
| q, n | Query and candidate node |
| V_eff(q,n) | Effective vector score: Cosine similarity + BM25 keyword overlap boost. |
| G_eff(A,n) | Effective graph score: proximity gated by BM25 keyword relevance. |
| A | Anchor set; if omitted, defaults to the top-3 vector hits |

Default weights: `w_v=0.5`, `w_g=0.15`, `w_bm25=0.35` (tuned for LoCoMo-style factoid queries). Full definition, anchors, and weight rationale: [docs/ALGORITHM.md](docs/ALGORITHM.md).

**Ingest-Time Neighborhood Averaging.** Stored vectors are L2-normalized after blending the text embedding with the mean of the top-5 vector neighbors: **0.7·e_raw + 0.3·e_neighbors** ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), Embedding Engine). The default embedding model is `all-mpnet-base-v2` (768-dim), configurable via `HYBRIDMIND_EMBEDDING_MODEL`. This is a practical, non-training variant of GraphSAGE-style aggregation used to provide a graph-aware embedding space. Formulation and caveats: [docs/ALGORITHM.md](docs/ALGORITHM.md) §3.

## Architecture

Layered stack: FastAPI / Pydantic v2 → embedding engine, BM25 index, vector and graph query engines, hybrid ranker → SQLite (WAL), FAISS `IndexHNSWFlat`, NetworkX `DiGraph` → atomic `.mind` persistence (manifest with SHA256 checksums, DB, vectors, vectors.map, graph, BM25 pickle). All settings are configurable via `HYBRIDMIND_*` environment variables. ASCII diagram and data-flow for hybrid search: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

Use the project virtual environment for all Python commands.

```bash
python3 -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Unix: source .venv/bin/activate
pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

All settings are configurable via `HYBRIDMIND_*` environment variables (e.g. `HYBRIDMIND_EMBEDDING_MODEL`, `HYBRIDMIND_DEFAULT_VECTOR_WEIGHT`). See [config.py](config.py) for the full list.

**Python SDK** ([sdk/memory.py](sdk/memory.py)):

```python
from sdk.memory import HybridMemory

memory = HybridMemory(base_url="http://127.0.0.1:8000")
nid = memory.store("Transformer models use self-attention.")
memory.relate(nid, "other-node-uuid", "derived_from")
results = memory.recall("attention mechanisms", top_k=5, mode="hybrid")
```

**CLI** ([cli/main.py](cli/main.py)) — built with Typer + Rich:

```bash
python -m cli.main --help
```

**Streamlit UI** ([ui/app.py](ui/app.py)):

```bash
streamlit run ui/app.py
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
| Bulk | `POST /bulk/nodes`, `POST /bulk/edges`, `POST /bulk/import` |
| Ingest | `POST /ingest/session-facts` (LLM-based fact extraction) |
| Comparison | `POST /comparison/effectiveness` |
| Ops | `GET /health`, `GET /ready`, `GET /live`, `POST /snapshot`, `GET /database`, `POST /database/export`, `GET /cache/stats`, `POST /cache/clear`, `POST /admin/compact`, `POST /admin/clear` |

**SDK** ([sdk/memory.py](sdk/memory.py)) — `HybridMemory`:

| Method | Description |
|--------|-------------|
| `store(text, metadata, session_id)` | Create a memory node |
| `store_batch(nodes)` | Bulk import |
| `store_with_auto_edges(text, config)` | Store + auto-link via vector similarity |
| `recall(query, top_k, mode, filter_metadata)` | Search (`hybrid`/`vector` mode) |
| `recall_stream(query, top_k, batch_size)` | Generator-based batched recall |
| `relate(source_id, target_id, type, weight)` | Create edge between nodes |
| `trace(concept, depth)` | Vector anchor + graph traversal |
| `forget(node_id)` | Soft-delete node |
| `compact()` | Rebuild indexes, hard-delete |
| `stats()` | Stats with domain distribution, degree analysis |
| `session.create/recall/archive/list()` | Session-scoped memory |
| `tools.get_schema()` | OpenAI function-calling compatible schemas |

## Evaluation & Benchmarks

The system is empirically evaluated on targeted benchmarks demonstrating clear regime-of-validity boundaries:
- **Semantic Paraphrase & Exact Lexical Lookup**: Vector alone (with BM25 exact match boost) achieves 100% precision@3 without graph assistance.
- **Edge-Dependent Multi-Hop Retrieval**: Graph-heavy hybrid (vector=0.1, graph=0.9) successfully surfaces multi-hop answers, recovering 100% recall where vector-only yields 0%.
- **Ingest-Time Neighborhood Averaging**: Conditioning embeddings on neighbors improves test retrieval of related cross-domain concepts from 66% (without averaging) to 100% (with averaging).
- **Ablation Studies**: Isolated runs (BM25 only, Vector only, Hybrid) confirm the weighted fusion correctly blends semantic, lexical, and structural signals, without inflating claims via unsupported deep graph traversals.

**LoCoMo Benchmark** ([docs/LOCOMO_BENCHMARK_REPORT.md](docs/LOCOMO_BENCHMARK_REPORT.md)): Peak 48% accuracy (Qwen3.5 397B), 60% Hit@10. Single-hop factual recall is 0% — conclusively isolated as an LLM extraction failure, not a retrieval failure.

**Multi-Domain Evaluation** ([docs/MULTI_DOMAIN_EVAL.md](docs/MULTI_DOMAIN_EVAL.md)): 7,510 nodes across 5 domains (Wikipedia, Stack Exchange, PubMed, AG News, CUAD Legal). Key finding: cross-domain-only edges at ≤5% density are structurally insufficient for hybrid scoring; intra-domain edges are necessary for non-zero graph signal.

Run benchmarks with: `./scripts/run_all_benchmarks.sh`

## Reviewer-Grade Limitations

1. **Graph Sparsity Failure**: The graph component is functionally useless if explicit cross-domain edges do not exist. Cross-domain-only edges at ≤5% per-node density produce structurally zero graph scores. Hybrid search defaults to vector-only if no anchors are found.
2. **Domain-Separation from Embeddings**: `all-mpnet-base-v2` struggles to differentiate certain document types (e.g. Stack Exchange QA vs Wikipedia paragraphs), which can lead to vector-search contamination that graph edges alone cannot fix.
3. **BM25 Exact Overlap Limits**: BM25 excels at keyword matching but fails to label semantic relevance that lacks exact keyword overlap.
4. **Single-Hop LLM Extraction**: On the LoCoMo benchmark, single-hop factual recall is 0% — caused by downstream LLM parsing failures (returning `Answer: None`), not by retrieval failures. Hit@10 for single-hop retrieval is 60%.
5. **Ingest Scalability**: Single-threaded execution of Python's Transformer models bounds ingestion to ~5 requests per second, making this explicitly a local-agent tool, not an enterprise search backend.
6. **Scalability Ceiling**: FAISS `IndexHNSWFlat` provides O(log n) approximate search, with a practical ceiling of ~8,000–10,000 nodes for sub-50ms p95 latency. Beyond that, tuning HNSW parameters or moving to a dedicated vector DB would be needed.

## Citation

```bibtex
@software{hybridmind2026,
  title        = {HybridMind: Local-Native Hybrid Vector--Graph Memory},
  author       = {a3ro-dev},
  year         = {2026},
  url          = {https://github.com/a3ro-dev/hybridmind}
}
```

## License

[MIT License](LICENSE).
