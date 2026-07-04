# HybridMind

**HybridMind** is a local-native hybrid vector–graph store for agent memory. It combines FAISS HNSW approximate search, an Okapi BM25 index with NLTK stemming, a NetworkX directed graph, and SQLite into a single `.mind` directory format with SHA256-verified atomic snapshots and 3-backup rotation. Repository: [github.com/a3ro-dev/hybridmind](https://github.com/a3ro-dev/hybridmind).

## Problem

Pure vector retrieval ignores explicit relational structure; graph-only retrieval lacks semantic filtering and scales poorly when edges are sparse or noisy. Agent memory systems need both: semantic alignment to the query and re-ranking or traversal grounded in declared relationships, without mandatory remote services.

## Approach

HybridMind is an engineering system that correctly applies known hybrid retrieval techniques without external cloud dependencies.

**Fusion defaults**. RRF (Reciprocal Rank Fusion, k=60) with per-signal weights (`vector_weight`, `graph_weight`) is the default fusion mode, replacing the previous fixed-linear formula. RRF requires zero per-corpus tuning — it works across LoCoMo, LongMemEval, and MuSiQue without weight sweeps. The original linear fusion (`vector_weight × V + graph_weight × G` with BM25 overlap gate) remains selectable via `fusion_mode="linear"` for back-compat and A/B comparison.

**Pre-trained cross-encoder reranker**. `mixedbread-ai/mxbai-rerank-large-v2` re-ranks the top-25 fusion pool with 70% fusion / 30% cross-encoder normalized blending. Both the fusion combined score and the cross-encoder score are independently normalized to [0,1] before blending, preventing the pure-text reranker from deleting graph-discovered candidates on multi-hop queries. This model offers ~84% Hit@1 vs 77% for bge-reranker-v2-m3, with 8x lower latency.

**Flexible embedding backends** (auto-selected in priority order):
1. **TEI (Text Embeddings Inference)** — self-hosted HuggingFace TEI endpoint via `RUNPOD_TEI_EMBEDDING_URL` (e.g., Qwen3-Embedding-8B on RunPod, native 4096-dim, no MRL truncation). Returns dense embeddings with automatic L2 normalization.
2. **OpenAI-compatible remote** — `HC_EMBEDDING_URL` (Hack Club AI proxy) or `RUNPOD_EMBEDDING_URL` for remote vLLM endpoints with Matryoshka Representation Learning (MRL) truncation to `HYBRIDMIND_EMBEDDING_DIMENSION` (default 1024).
3. **Local embeddings** — `BAAI/bge-m3` (1024-dim, default) with native sparse vectors and ColBERT support. Falls back to `all-mpnet-base-v2` (768-dim) on CPU-only deploys.

**LLM backends**:
- **Self-hosted RunPod vLLM** (`engine/runpod_llm.py`): When `RUNPOD_LLM_ENDPOINT_ID` is set, fact extraction and consolidation use Qwen3.5-9b (or configured model) on your own RunPod Serverless pod. Disables thinking mode by default (Qwen3.5's extended reasoning burns output tokens).
- **Hack Club AI proxy** (fallback): OpenAI-compatible `/v1/chat/completions` endpoint for inference.

**Default embedding model**: `BAAI/bge-m3` locally, or Qwen3-Embedding-8B (4096-dim) via TEI. Full FlagEmbedding native sparse + ColBERT vectors available with `pip install FlagEmbedding>=1.2.10`.

**Ingest-Time Neighborhood Averaging** (configurable via `HYBRIDMIND_USE_GRAPH_CONDITIONED_EMBEDDINGS`). Stored vectors are L2-normalized after blending the text embedding with the mean of the top-5 vector neighbors: **0.7·e_raw + 0.3·e_neighbors** ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), Embedding Engine). This is a practical, non-training variant of GraphSAGE-style aggregation. Formulation and caveats: [docs/ALGORITHM.md](docs/ALGORITHM.md) §3.

**Auto-edge inference** (`HYBRIDMIND_AUTO_EDGES_ENABLED=true`): cosine-threshold similarity edges and entity co-occurrence edges created automatically at ingest time across all three ingest paths (nodes, bulk, session-facts).

**Opt-in research modules** (off by default, CPU fallbacks):
- ColBERT MaxSim late interaction (`HYBRIDMIND_COLBERT_ENABLED=true`, requires `FlagEmbedding`)
- GNN reranker with GraphSAGE/HGT (`HYBRIDMIND_GNN_ENABLED=true`, requires `torch-geometric`)
- Post-trainable fusion MLP head (`HYBRIDMIND_FUSION_MODEL=<checkpoint.npz>`)
- Online contrastive fine-tuning of bge-m3 on graph edges (RunPod script: `scripts/train_contrastive.py`)

Full scoring definition, architecture diagram, and data-flow: [docs/ALGORITHM.md](docs/ALGORITHM.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Architecture

Layered stack:
```
FastAPI / Pydantic v2
  ↓
Embedding Pipeline (priority: TEI → OpenAI-compat → local bge-m3 / all-mpnet)
BM25 Index (bm25s backend, 100x faster than pure Python)
Vector/Graph/ColBERT Query Engines
Hybrid Ranker with RRF Fusion + Cross-Encoder Reranker
  ↓
Persistent Storage:
  - SQLite (WAL mode) for nodes, edges, metadata
  - FAISS IndexHNSWFlat for vector search
  - NetworkX DiGraph for graph traversal
  - ColBERT .npz store (opt-in research)
  ↓
Atomic .mind Format (manifest with SHA256 checksums, 3-backup rotation)
```

GPU auto-device selection via centralized `engine/device.py` (cuda > mps > cpu). All settings configurable via `HYBRIDMIND_*` environment variables.

**LLM Integration Layer**:
- Fact extraction and consolidation automatically use RunPod vLLM (when configured) for faster, self-hosted inference.
- Falls back to Hack Club AI proxy if RunPod is not configured.
- Full async pipeline with structured JSON output + retry logic.

ASCII diagram and data-flow: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

Use the project virtual environment for all Python commands.

```bash
python3 -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Unix: source .venv/bin/activate
pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

All settings are configurable via `HYBRIDMIND_*` environment variables. See [config.py](config.py) for the full list.

### Running with self-hosted embeddings (RunPod TEI)

```bash
export RUNPOD_API_KEY="your-runpod-api-key"
export RUNPOD_TEI_EMBEDDING_URL="https://<endpoint-id>.api.runpod.ai"
export HYBRIDMIND_EMBEDDING_DIMENSION=4096
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Running with self-hosted LLM (RunPod vLLM)

```bash
export RUNPOD_API_KEY="your-runpod-api-key"
export RUNPOD_LLM_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_LLM_MODEL="qwen/qwen3.5-9b"  # or any vLLM-registered model
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

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
python3 -m pytest tests/ -v            # 34 passed, 3 skipped (live SDK tests)
python scripts/retrieval_ablation.py    # Fast in-memory ablation (5 docs, 7 queries)
python scripts/targeted_graph_benchmark.py  # Graph-depth regime-of-validity test
./scripts/run_all_benchmarks.sh         # LoCoMo + LongMemEval + MuSiQue retrieval evals
```

Further integration notes: [docs/AGENT_INTEGRATION.md](docs/AGENT_INTEGRATION.md).

## API overview

| Area | Methods (HTTP) |
|------|----------------|
| Nodes | `POST/GET/PUT/DELETE /nodes`, `GET /nodes/{id}` |
| Edges | `POST/GET/PUT/DELETE /edges`, `GET /edges/node/{node_id}`, `GET /edges/types` |
| Search | `POST /search/vector`, `GET /search/graph`, `POST /search/hybrid`, `POST /search/compare`, `GET /search/path/{source}/{target}`, `GET /search/stats` |
| Bulk | `POST /bulk/nodes`, `POST /bulk/edges`, `POST /bulk/import` |
| Ingest | `POST /ingest/session-facts` (LLM fact extraction with structured JSON output + retry) |
| Comparison | `POST /comparison/effectiveness` |
| Ops | `GET /health` (incl. GPU info), `GET /ready`, `GET /live`, `POST /snapshot`, `GET /database`, `POST /database/export`, `GET /cache/stats`, `POST /cache/clear`, `POST /admin/compact`, `POST /admin/clear` |

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

**Test suite**: 34/37 passed, 3 skipped (SDK live-tests require running SDK). All core API, search, fusion, and graph traversal tests pass reliably.

**In-memory retrieval ablation** (5-node ML corpus, 7 queries, bge-m3 1024-dim, RRF fusion):
- Vector-only: **P@3=0.48, MRR=1.00**
- BM25-only: **P@3=0.48, MRR=1.00**
- Hybrid (RRF): **P@3=0.48, MRR=1.00**
- Hybrid (RRF heavy-graph): **P@3=0.48, MRR=1.00**

All modes ceiling at 100% on the tiny test set. BM25 + RRF correctly fuses keyword and semantic signals without regressing either.

**Graph-depth regime benchmark** (9-node multi-hop graph, A→B→C with distractors, bge-m3 1024-dim):
- Semantic paraphrase: **vector Recall@3=1.0, hybrid Recall@3=1.0** (graph had no effect — correct)
- Exact lexical (drug name): **vector Recall@3=1.0, hybrid Recall@3=1.0** (BM25 boost works)
- Edge-dependent multi-hop (2-hop traversal required): **vector Recall@3=0.0, hybrid Recall@3=1.0** (graph surfaces the correct answer node where vector hits nothing — RRF with graph-weight=0.9 successfully elevates the 2-hop candidate above distractors)
- Missing-anchor failure: **vector Recall@3=0.0, hybrid Recall@3=0.0** (without an anchor, graph has no reference — this is the correct failure mode)

The key improvement from Phase 2: signal-weighted RRF preserves `graph_weight`/`vector_weight` influence within the rank-fusion formula, and normalized cross-encoder blending protects graph-discovered candidates from being deleted by the pure-text reranker.

**Auto-edge inference** (config-gated, off by default): cosine-threshold and entity co-occurrence edges wired into all three ingest paths (`/nodes`, `/bulk/nodes`, `/ingest/session-facts`). Typed walk-weight map (`models/edge.py:EDGE_TYPE_WALK_WEIGHTS`) provides per-edge-type proximity contribution.

**LoCoMo Benchmark** ([docs/LOCOMO_BENCHMARK_REPORT.md](docs/LOCOMO_BENCHMARK_REPORT.md)): Peak 48% accuracy (Qwen3.5 397B), 60% Hit@10. Single-hop factual recall was 0% in Phase 3 — isolated as an LLM extraction failure, not a retrieval failure. MemoryBench QA now includes span-extraction retry for abstention.

**Multi-Domain Evaluation** ([docs/MULTI_DOMAIN_EVAL.md](docs/MULTI_DOMAIN_EVAL.md)): 7,510 nodes across 5 domains (Wikipedia, Stack Exchange, PubMed, AG News, CUAD Legal). Key finding: cross-domain-only edges at ≤5% density are structurally insufficient for hybrid scoring; intra-domain edges are necessary for non-zero graph signal.

**LongMemEval + MuSiQue retrieval evals** (`eval_longmemeval_retrieval.py`, `eval_musique_retrieval.py`): Python retrieval-eval scripts for fast local iteration, mirroring the LoCoMo eval pattern. MuSiQue multi-hop relevance keyed on supporting paragraph IDs. MemoryBench benchmark loaders included (`memorybench/src/benchmarks/musique/`, `longmemeval/`).

**RunPod training scripts** (1-step smoke on CPU, full training on GPU):
- `scripts/train_fusion_mlp.py` — pairwise logistic loss, learning-to-rank of FusionScorer MLP
- `scripts/train_gnn.py` — BPR loss, GraphSAGE over typed graph, outputs `.pt` checkpoint
- `scripts/train_contrastive.py` — SimCSE contrastive fine-tuning of bge-m3 on graph-edge pairs

Run benchmarks with: `./scripts/run_all_benchmarks.sh`

## Configuration Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `RUNPOD_TEI_EMBEDDING_URL` | (unset) | Base URL for self-hosted HF TEI embedding endpoint (e.g., Qwen3-Embedding-8B) |
| `RUNPOD_LLM_ENDPOINT_ID` | (unset) | RunPod Serverless endpoint ID for vLLM inference |
| `RUNPOD_LLM_MODEL` | `qwen/qwen3.5-9b` | Model to use on RunPod vLLM endpoint |
| `RUNPOD_API_KEY` | (unset) | RunPod API key (Bearer token) for both TEI and vLLM |
| `HC_EMBEDDING_URL` | (unset) | Hack Club AI proxy base URL for embeddings |
| `HC_API_KEY` | (unset) | Hack Club AI API key |
| `HYBRIDMIND_EMBEDDING_DIMENSION` | 4096 | Output dimension for TEI; 1024 for local bge-m3 or MRL-truncated OpenAI endpoints |
| `HYBRIDMIND_USE_GRAPH_CONDITIONED_EMBEDDINGS` | true | Enable ingest-time neighborhood averaging (0.7·e_raw + 0.3·e_neighbors) |
| `HYBRIDMIND_AUTO_EDGES_ENABLED` | false | Enable automatic edge creation on ingest |
| `HYBRIDMIND_COLBERT_ENABLED` | false | Enable ColBERT MaxSim re-ranking (requires FlagEmbedding) |
| `HYBRIDMIND_GNN_ENABLED` | false | Enable GNN reranker (requires torch-geometric) |
| `HYBRIDMIND_FUSION_MODE` | `rrf` | Fusion strategy: `rrf` (Reciprocal Rank Fusion) or `linear` |
| `HYBRIDMIND_EMBEDDING_MODEL` | `BAAI/bge-m3` | Local fallback embedding model |

## Reviewer-Grade Limitations

1. **Graph Sparsity Failure**: The graph component is functionally useless if explicit cross-domain edges do not exist. Cross-domain-only edges at ≤5% per-node density produce structurally zero graph scores. Hybrid search defaults to vector-only if no anchors are found.
2. **Domain-Separation from Embeddings**: `all-mpnet-base-v2` struggles to differentiate certain document types (e.g. Stack Exchange QA vs Wikipedia paragraphs). bge-m3 (1024-dim) shows improved domain separation but still exhibits contamination on hard negatives. Qwen3-Embedding-8B (when using TEI) offers superior domain separation with MTEB 70.58.
3. **BM25 Exact Overlap Limits**: BM25 excels at keyword matching but fails to label semantic relevance that lacks exact keyword overlap.
4. **Single-Hop LLM Extraction**: On the LoCoMo benchmark, single-hop factual recall was 0% in Phase 3 — caused by downstream LLM parsing failures (returning `Answer: None`), not by retrieval failures. Hit@10 for single-hop retrieval was 60%. Phase 4 added structured-output extraction (`json_schema`) + retry with rephrased prompts for ingest, and span-extraction retry for QA.
5. **Ingest Scalability**: bge-m3 on CPU is ~200ms/embedding (~5 req/s). GPU, SentenceTransformer fallback, or TEI (Qwen3 on RunPod) improves this but remains single-threaded Python. This is a local-agent tool, not an enterprise search backend.
6. **Scalability Ceiling**: FAISS `IndexHNSWFlat` provides O(log n) approximate search, with a practical ceiling of ~8,000–10,000 nodes for sub-50ms p95 latency. Beyond that, tuning HNSW parameters or moving to a dedicated vector DB would be needed.
7. **ColBERT storage cost**: Per-token vectors (~100-200KB/node) are opt-in and stored as `.npz` files in `<mind>/colbert/`. Practical only for small corpora without a dedicated vector DB.
8. **TEI Dimension Mismatch**: Switching between TEI (4096-dim Qwen3) and local bge-m3 (1024-dim) requires re-running `scripts/reindex_embeddings.py` to avoid FAISS index dimension mismatches.

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
