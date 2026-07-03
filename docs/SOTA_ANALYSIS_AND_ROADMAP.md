# HybridMind → SOTA: Complete Gap Analysis & Roadmap

> Deep research completed June 2026. Covers competitive landscape, architectural gaps, model upgrades, and execution plan.

---

## Competitive Landscape

| System | LoCoMo | LongMemEval | Architecture |
|--------|--------|-------------|--------------|
| **EverMind/EverOS** | 93.05% | 83.00% | Engram-inspired lifecycle, MemCells→MemScenes, proprietary |
| **Maximem Synap** | — | 90.2% | 15ms P50 retrieval, verified |
| **Hindsight** | 91.4% | — | Free self-hosted, architecture undisclosed |
| **Supermemory** | 85.4% | — | MCP-native, ~2.6K stars |
| **Letta (MemGPT)** | 83.2% | — | OS-inspired hierarchy, agent self-editing memory |
| **AgentRunbook-C** | — | 74.9% (LME-V2-S) | Coding agent as memory controller, 3 knowledge pools |
| **Zep** | — | 63.8% | Temporal KG: episodic/semantic/community subgraphs, Graphiti+Neo4j |
| **Cognee** | — | — | Multimodal KG extraction, ECL pipeline, open-source |
| **Mem0** | 67.13% | 49.0% | Vector-only, passive extraction, ~48K stars |
| **HybridMind** | **?** | **?** | Vector+Graph+BM25 RRF fusion, zero public benchmarks |

---

## Knowledge Graph & RAG Leaders (Adjacent Space)

| System | Key Innovation | Status |
|--------|---------------|--------|
| **Microsoft GraphRAG** | Leiden hierarchical community detection + summarization | 22K+ stars, production |
| **LightRAG** | Dual-level (low/high) graph retrieval, outperforms GraphRAG on UltraDomain | Open source |
| **HippoRAG** | PageRank-based graph retrieval, strong on GraphRAG-Bench | ICLR 2026 benchmarked |
| **GFM-RAG** | Large-scale pretrained graph foundation model, top on GraphRAG-Bench | Research |
| **ColPali / ColQwen2.5** | Visual document retrieval via late interaction on image patches | ICLR 2025, ViDoRe V2 leader |
| **SPLADE v3** | Learned sparse retrieval, outperforms BM25 on BEIR | Mature, open source |
| **PLAID** | Efficient ColBERT indexing, 0.54ms query on million-scale | Published Nov 2025 |

---

## Embedding & Reranking SOTA (June 2026)

### Embedding Models (MTEB Leaderboard)
| Model | MTEB Score | Dim | License | Notes |
|-------|-----------|-----|---------|-------|
| **Qwen3-Embedding-8B** | 70.58 (#1) | configurable | Apache 2.0 | Multilingual, MRL support |
| **NV-Embed-v2** | 68+ | 4096 | Research | Best English, Mistral-7B backbone |
| **Gemini Embedding 2** | 67.71 | — | API | Best managed option |
| **bge-m3 (current)** | ~63.0 (BEIR) | 1024 | MIT | 2 years old |

### Reranker Models
| Model | Hit@1 | Latency | License | Notes |
|-------|-------|---------|---------|-------|
| **mxbai-rerank-large-v2** | ~84% | ~55ms | Apache 2.0 | 8x faster than bge-reranker-v2-gemma |
| **Jina Reranker v3** | 81.33% | 188ms | CC-BY-NC-4.0 | 100+ languages |
| **Qwen3 Reranker** | competitive | — | Apache 2.0 | Same family as embedding |
| **bge-reranker-v2-m3 (current)** | ~77% | ~120ms | MIT | Decent but surpassed |

---

## Key Competitor Architectural Insights

### Zep — Temporal Knowledge Graph
- **3 subgraphs**: Episode (raw events), Semantic Entity (extracted entities + relationships), Community (Leiden clustering)
- **Temporal edges**: Every relationship has `valid_from`/`valid_until`, fact supersession tracked
- **Time-decay scoring**: Recent facts weighted higher in retrieval
- **Graphiti engine**: Runs on Neo4j, continuous ingestion with incremental updates
- **DMR accuracy**: 94.8% on Deep Memory Retrieval task

### AgentRunbook (LongMemEval-V2 SOTA)
- **3 knowledge pools**: Raw state slices (UI evidence), State transition events (environment dynamics), Procedure/hint notes (workflows + gotchas)
- **LLM controller**: Generates structured multi-stream queries at retrieval time (up to 5 raw-state queries + 1 event + 1 note)
- **Coding agent variant**: Stores trajectories as files, uses Codex with workflow docs + manifest artifacts + helper scripts
- **Key finding**: Coding agents (GPT-5.4-mini + Codex) at 74.9% accuracy beat RAG-only (51.0%) by 24 points
- **Latency tradeoff**: RAG at 26s vs Codex at 108s — different operating points

### EverMind/EverOS — SOTA on LoCoMo (93.05%)
- **Engram-inspired lifecycle**: Conversations → MemCells → MemScenes (thematic consolidation)
- **MemCell**: Atomic memory unit with extraction + embedding
- **MemScene**: Cluster of related MemCells under a thematic representation
- **Memory Sparse Attention (MSA)**: Architecture scaling to 100M tokens with sparse attention + top-k selection
- **Self-organizing**: Memory structure evolves automatically without manual ontology

### Letta (MemGPT) — OS Paradigm
- **Memory hierarchy**: Core context (RAM) → Recall storage (swap) → Archival storage (disk)
- **Agent self-editing**: LLM actively manages its own memory via tool calls (page in/out)
- **Proactive retrieval**: Unlike passive RAG which retrieves before generation, Letta pages context mid-generation
- **Function-calling memory management**: store, search, forget, archive, recall operations

### Cognee — Multimodal Graph Extraction
- **ECL Pipeline**: Extract → Cognify → Load, a 6-stage ingestion process
- **Multimodal**: Handles text, PDFs, images, videos
- **Ontology grounding**: Maps extracted entities to domain ontologies
- **MCP integration**: Plugin for Claude Code, OpenClaw
- **Multi-tenant isolation**: Shared graph with per-agent write scopes

### Supermemory — MCP-First Distribution
- **MCP-native**: One command to expose memory to any MCP client (Claude Code, OpenClaw, etc.)
- **Universal Memory MCP**: Cross-LLM memory, no vendor lock-in
- **Profile injection**: Injects full user profile at conversation start
- **Memory expiration**: First-class operation, not an edge case

---

## The 22 Gaps — Ranked by Impact

### TIER 1 — Engineering Fixes (make the system actually work)

#### 1. GNN node features are ALL ZEROS at inference
In `engine/gnn_reranker.py:137`, `x = torch.zeros(len(node_ids), feat_dim)`. The GraphSAGE model gets NO content features — it can only learn from pure graph topology. The code has a comment acknowledging this: *"In production, use the stored embedding from the SQLite store."* This means the GNN reranker is effectively non-functional.

**Fix**: Load actual bge-m3 embeddings from SQLite/FAISS as node features at inference time.

#### 2. Graph expansion yields ZERO recall on graph-dependent queries
Ablation results show 0.00 Recall@3 on GRAPH_SINGLE_HOP and GRAPH_MULTI_HOP across ALL conditions including FULL_PIPELINE. The core value proposition of graph-augmented retrieval is completely broken for its intended use case.

**Root causes**:
- `graph_weight=0.15` is too low to surface graph-only candidates
- Pure-graph candidates get `vector_score=0.0` and `rolled_up_scores[nid]=0.0` so they can never beat weak semantic matches
- RRF with k=60 heavily penalizes items that only appear in the graph rank list

**Fix**: Debug scoring, increase graph_weight in RRF, ensure graph-only candidates get a reasonable base score.

#### 3. BM25 is O(N) pure Python with no inverted index
`storage/bm25_index.py` iterates ALL documents linearly on every query. The code admits: *"the top_k parameter only affects the final sort-and-slice, not computation cost."* At 50K+ nodes this is unworkable.

**Fix**: Replace with `bm25s` (NumPy-accelerated, SciPy sparse matrix, mmap-backed, 100x faster) or Pyserini (Lucene inverted index, sub-ms for millions of documents).

#### 4. bge-m3 sparse vectors computed but NEVER used
`embed_hybrid()` generates dense + sparse + ColBERT vectors from bge-m3, but only the dense vectors enter the retrieval pipeline. The native sparse lexical weights (which encode term importance directly from the model) are discarded.

**Fix**: Wire bge-m3 sparse vectors as a first-class retrieval signal in parallel to BM25, fused via RRF alongside dense and graph.

#### 5. `query_router.py` exists but isn't wired into the main pipeline
The query router classifies queries as temporal/multihop/entity and is only used in `eval_locomo_retrieval.py`. Integrating it into `HybridRanker.search()` would enable query-type-aware weight tuning.

**Fix**: Wire query_router into HybridRanker, use query type to dynamically adjust vector/graph/BM25 weights and fusion parameters.

---

### TIER 2 — Model Upgrades (immediate quality jumps)

#### 6. bge-m3 is 2 generations behind SOTA embeddings
bge-m3 (~63.0 BEIR) vs Qwen3-Embedding-8B (70.58 MTEB, #1). The +7 point gap translates directly to retrieval quality.

**Fix**: Add Qwen3-Embedding-8B as embedding option with Matryoshka dimension selection (can use 2048-dim for quality, 256-dim for speed).

#### 7. Cross-encoder reranker is outdated
bge-reranker-v2-m3 (~77% Hit@1) vs mxbai-rerank-large-v2 (~84%, 8x faster than gemma variant).

**Fix**: Benchmark mxbai-rerank-large-v2 and Jina Reranker v3, add the best performer as default.

#### 8. No learned sparse retrieval (SPLADE)
BM25 hits vocabulary mismatch problems ("cancel membership" vs "terminate subscription"). SPLADE (learned sparse via BERT MLM head) expands queries implicitly and outperforms BM25 on BEIR.

**Fix**: Add SPLADE-v3 via `splade-index` or FastEmbed as an alternative sparse signal.

#### 9. FAISS-GPU unused, basic HNSW only
`use_faiss_gpu: bool = False` with comment "Linux/Docker only." cuVS-accelerated IVF-PQ gives 4-5x compression and 10x speedup over HNSWFlat at scale.

**Fix**: Enable `faiss-gpu` with IVF-PQ for GPU machines, add auto-detection and fallback logic.

---

### TIER 3 — Architecture Additions (what competitors do that HybridMind doesn't)

#### 10. No temporal knowledge graph
Zep's key advantage: edges have timestamps, relationships can be superseded (`valid_from`/`valid_until`), temporal decay scoring. Zep tracks *when* relationships were established and when they were replaced. HybridMind stores timestamps in metadata but never uses them for retrieval, scoring, or filtering.

**Fix**: Add temporal edge weights (recent edges weighted higher), fact validity windows, contradiction detection, time-decay in graph proximity scores.

#### 11. No memory consolidation/forgetting lifecycle
EverMind uses "engram-inspired lifecycle" — MemCells consolidate into MemScenes. Letta uses agent-driven self-editing with paging. AgentRunbook-R uses 3 separate knowledge pools with different extraction strategies. HybridMind has `fact_extractor.py` but no consolidation, no summarization across sessions, no importance-based retention, no forgetting.

**Fix**: Build a consolidation pipeline: periodic summarization of related facts into higher-level knowledge, importance scoring for retention decisions, explicit forgetting API with audit trail.

#### 12. No 3-pool knowledge architecture
AgentRunbook-R's design is directly applicable:
- **Raw state slice pool** — exact UI/page evidence with radius-1 windows
- **Event pool** — state transitions extracted by LLM controller
- **Note pool** — trajectory-level procedures, hints, gotchas

HybridMind has flat fact extraction but no differentiated knowledge granularities.

**Fix**: Implement multi-granularity memory: raw_facts (exact), events (transitions), notes (procedural), summaries (thematic).

#### 13. No MCP server integration
Supermemory leads with MCP-native architecture. Cognee supports MCP. Claude Code, OpenClaw connect via MCP. HybridMind has SDK + FastAPI but no Model Context Protocol server.

**Fix**: Build an MCP server exposing search, store, recall, relate tools. This gives instant access to Claude Code, OpenClaw, and the entire MCP ecosystem.

#### 14. Text-only — no multimodal memory
Cognee, Zep, and AgentRunbook all handle images in memory. ColPali/ColQwen2 introduced visual document retrieval without OCR. For web agent memory (where UI screenshots are critical evidence), this is essential.

**Fix**: Add image ingestion pipeline (screenshots → CLIP/ColPali embedding), multimodal search, screenshot storage alongside text facts.

#### 15. No community/theme detection in the graph
Zep builds a "community subgraph" via community detection. Microsoft GraphRAG uses Leiden hierarchical clustering to find topic communities, then summarizes each community. This enables answering abstract questions like "what themes have emerged across my conversations."

**Fix**: Add Leiden/Louvain community detection on the knowledge graph, auto-generate community summaries, expose as a search mode.

#### 16. Graph stored as NetworkX pickle — won't scale
NetworkX DiGraph with pickle serialization. No graph queries beyond BFS. Zep uses Neo4j via Graphiti. Cognee uses graph-native storage.

**Fix**: Evaluate KùzuDB (embeddable graph DB, Cypher queries, columnar storage), LanceDB, or Neo4j as a scalable graph backend.

---

### TIER 4 — Benchmarks & Scale

#### 17. ZERO public benchmark results
Every competitor publishes on LoCoMo, LongMemEval, BEAM, or DMR. HybridMind has no published benchmark numbers anywhere.

**Fix**: Run full benchmark suite on: LoCoMo (1,540 Qs), LongMemEval-S (500 Qs), LongMemEval-V2, BEAM, GraphRAG-Bench (ICLR 2026), and the internal canonical dataset.

#### 18. No metadata inverted index
SQLite `WHERE json_extract(metadata, '$.key') = 'value'` is O(N) row scan. Competitors use purpose-built vector DBs with native metadata filtering on inverted indexes.

**Fix**: Add metadata index tables in SQLite or switch to a DB with native hybrid filtering (Qdrant, LanceDB).

#### 19. SGMem chunk rollup exists but isn't tested or documented
The code handles `is_sentence_chunk` → `parent_id` rollup with max() aggregation. This is the right pattern but needs documentation and benchmarks.

**Fix**: Document chunking strategy, write tests for chunk rollup behavior, benchmark parent vs child retrieval.

#### 20. No multi-agent/session isolation model
Cognee supports "tenant and user isolation natively, allowing each agent to read from a shared graph while maintaining its own write scope."

**Fix**: Add namespace/tenant isolation at the storage layer, per-agent read/write scopes, shared memory with controlled access.

---

### TIER 5 — Differentiation Opportunities

#### 21. Trainable fusion MLP pipeline doesn't exist
`FusionScorer` ships with heuristic init that approximates RRF. `scripts/train_fusion_mlp.py` is referenced but training isn't documented or runnable.

**Fix**: Build the training pipeline with real relevance judgments to learn optimal signal weights per query type. Release the training script and documentation.

#### 22. No ColPali/ColQwen2 visual document retrieval
ColPali enables retrieving document *images* by visual similarity — no OCR needed. ColQwen2.5 tops the ViDoRe V2 leaderboard. For agent memory where screenshots, PDFs, and diagrams are evidence, this is transformative.

**Fix**: Add ColPali/ColQwen2.5 support for visual memory ingestion and retrieval. This would make HybridMind the first agent memory system with native visual document retrieval.

---

## Recommended Execution Order

### PHASE A — Make It Work (fix the broken things)

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 1 | Fix GNN zero-vector features — load real embeddings from SQLite | High | Low |
| 2 | Fix graph expansion recall — debug scoring and RRF weights | Critical | Medium |
| 3 | Wire bge-m3 sparse vectors as retrieval signal | High | Low |
| 4 | Wire query_router into HybridRanker.search() | Medium | Low |

**Gate**: Run canonical dataset benchmarks → verify fixes work.
**Gate**: Run LoCoMo + LongMemEval-S → get baseline numbers.

### PHASE B — Make It Fast (production engineering)

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 5 | Replace pure-Python BM25 with bm25s (100x faster) | Critical | Medium |
| 6 | Enable faiss-gpu with IVF-PQ for GPU machines | Medium | Medium |
| 7 | Add metadata inverted indexes in SQLite | Medium | Medium |

**Gate**: Re-run benchmarks → verify latency improvements and no regression.

### PHASE C — Make It Smart (architecture additions)

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 8 | Add Qwen3-Embedding-8B as embedding option | High | Medium |
| 9 | Add mxbai-rerank-large-v2 as reranker option | High | Low |
| 10 | Add temporal knowledge graph (fact validity, time-decay) | High | High |
| 11 | Add 3-pool knowledge architecture (raw + events + notes) | Critical | High |
| 12 | Add memory consolidation pipeline | High | High |

**Gate**: Re-run benchmarks → measure quality improvements.
**Target**: 80%+ on LoCoMo, 65%+ on LongMemEval-S.

### PHASE D — Make It Connect (ecosystem)

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 13 | Build MCP server for Claude Code / OpenClaw integration | Medium | Medium |
| 14 | Add multimodal memory (images, screenshots, PDFs) | High | High |
| 15 | Add Leiden community detection + community summaries | Medium | Medium |

**Gate**: Final benchmark suite run → target SOTA numbers.
**Target**: 88%+ on LoCoMo, 75%+ on LongMemEval-S.

### PHASE E — Differentiate

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 16 | Build trainable fusion MLP pipeline with training script | Medium | High |
| 17 | Add ColPali/ColQwen2.5 visual document retrieval | High | High |
| 18 | Evaluate KùzuDB/LanceDB as scalable graph backend | Medium | High |
| 19 | Multi-agent isolation model with namespaces | Medium | High |

**Target**: Clear differentiation and SOTA on multiple benchmarks.

---

## Summary

The gap between HybridMind and SOTA can be closed. EverMind at 93% LoCoMo and Maximem Synap at 90.2% LongMemEval prove this is solvable. The primary blockers are:

1. **Graph expansion is broken** — the core differentiator doesn't work
2. **GNN is dead code** — zero-vector features, non-functional reranker
3. **BM25 won't scale** — O(N) pure Python without inverted index
4. **Models are outdated** — 2-year-old embeddings and rerankers
5. **No knowledge lifecycle** — no consolidation, no temporal reasoning, no forgetting
6. **No benchmark evidence** — zero published results

Fix these fundamentals in Phases A-B, layer on architectural innovations from Zep/AgentRunbook/EverMind in Phase C, and HybridMind will be competitive with the SOTA systems.
