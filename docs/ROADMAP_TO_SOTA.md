# HybridMind → #1: SOTA Roadmap

> Target: ≥90% LoCoMo, ≥85% LongMemEval-S, paper-worthy contributions.  
> Hardware: RTX 4050 6GB (local), RunPod A100 (training).  
> Baseline: 48% LoCoMo accuracy, 60% Hit@10, **0% single-hop** (harness bug).

---

## Current Bleeding-Edge Targets to Beat

| System | LoCoMo | LongMemEval | Architecture |
|--------|--------|-------------|-------------|
| EverMind/EverOS | **93.05%** | 83.00% | Engram lifecycle, MemCells→MemScenes |
| Maximem Synap | — | **90.2%** | 15ms P50, proprietary |
| Hindsight | 91.4% | — | Self-hosted |
| Letta (MemGPT) | 83.2% | — | OS hierarchy + agent self-editing |
| Zep | — | 63.8% | Temporal KG, Neo4j, Graphiti |
| Mem0 | 67.13% | 49.0% | Vector-only |
| **HybridMind** | **48%** | **?** | Vector+Graph+BM25 RRF |

Gap to #1: **45 points on LoCoMo, ~42 on LongMemEval**. This is achievable via 6 changes, in order.

---

## PHASE 0 — Free Points: Fix the Harness Bug (~+15 pts)

**Problem**: Single-hop retrieval Hit@10 = 60% but accuracy = 0%. Answering LLM returns `Answer: None` even with correct context.  
**Root cause**: MemoryBench prompt parsing drops single-hop answer extraction.  
**This is not a retrieval failure.**

### Fixes (sequential)

**0a. Span-extraction retry** (already partially in Phase 4, complete it):
```python
# eval/locomo_qa.py — after first LLM call returns None
if answer is None:
    rephrased = f"From the provided context, extract the exact answer to: {question}\nAnswer only with the specific fact, no explanation."
    answer = llm_call(rephrased, context)
```

**0b. Structured JSON output for LoCoMo QA**:
```python
schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
response = client.chat.completions.create(model=model, messages=msgs, response_format={"type": "json_schema", "json_schema": {"schema": schema}})
answer = json.loads(response.choices[0].message.content)["answer"]
```

**0c. Use a better answering model**: Switch from GPT-5-mini → `claude-sonnet-4-6` or `Qwen3-235B-A22B` for QA. The 36% gap between Qwen3.5-397B (48%) and GPT-5-mini (36%) proves model choice matters more than retrieval for QA accuracy.

**Expected gain**: +15–20 points on LoCoMo (single-hop goes from 0% to 60%+ accuracy).

---

## PHASE 1 — Model Upgrades (~+8 pts retrieval quality)

### 1a. Embedding: bge-m3 → Qwen3-Embedding-8B

| Model | MTEB Score | Dim | HuggingFace |
|-------|-----------|-----|-------------|
| bge-m3 (current) | ~63.0 | 1024 | `BAAI/bge-m3` |
| **Qwen3-Embedding-8B** | **70.58 (#1)** | 2048/configurable | `Qwen/Qwen3-Embedding` |

Qwen3 supports **Matryoshka Representation Learning (MRL)** — you can query at 2048 for quality or 512 for speed without retraining.

```python
# engine/embedding.py — add Qwen3 backend
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding", trust_remote_code=True)
# Use task-specific prompts — critical for Qwen3
embeddings = model.encode(texts, prompt_name="retrieval.query")  # for queries
embeddings = model.encode(texts, prompt_name="retrieval.passage")  # for docs
```

**RTX 4050 note**: Qwen3-Embedding-8B is ~16GB weights (BF16). Won't fit on 6GB VRAM. Options:
- Use `HYBRIDMIND_EMBEDDING_MODEL=Qwen/Qwen3-Embedding` + offload to CPU/RAM (slow but works)  
- Use `text-embedding-3-large` (OpenAI, 3072-dim, $0.13/1M tokens) for benchmarks
- Use `NV-Embed-v2` (4096-dim, MTEB 68+) — runs on RunPod A100

**Best local option**: `mixedbread-ai/mxbai-embed-large-v1` (1024-dim, MTEB 64.7) or `BAAI/bge-large-en-v1.5` as CPU fallback.

**For benchmarks**: Run on RunPod with Qwen3-Embedding-8B, report both local and full numbers.

### 1b. Reranker: bge-reranker-v2-m3 → mxbai-rerank-large-v2

```bash
# engine/reranker.py
HYBRIDMIND_RERANKER_MODEL=mixedbread-ai/mxbai-rerank-large-v2
```

| Reranker | Hit@1 | Latency | License |
|----------|-------|---------|---------|
| bge-reranker-v2-m3 (current) | ~77% | ~120ms | MIT |
| **mxbai-rerank-large-v2** | **~84%** | ~55ms | Apache 2.0 |
| Jina Reranker v3 | 81.33% | 188ms | CC-BY-NC |

Drop-in replacement via `FlagEmbedding` or `sentence-transformers`.

### 1c. Sparse Retrieval: Replace BM25 with bm25s

Current BM25 is O(N) pure Python — **100x slower than it needs to be**.

```bash
pip install bm25s PyStemmer
```

```python
# storage/bm25_index.py — full replacement
import bm25s, Stemmer

stemmer = Stemmer.Stemmer("english")
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
retriever = bm25s.BM25(corpus=corpus_tokens)
retriever.save("bm25_index")  # mmap-backed, instant load

# At query time
query_tokens = bm25s.tokenize([query], stemmer=stemmer)
results, scores = retriever.retrieve(query_tokens, k=top_k)
```

**Alternative (highest quality)**: SPLADE-v3 via FastEmbed:
```python
from fastembed import SparseTextEmbedding
model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
sparse_vecs = list(model.embed(texts))  # returns {indices, values} pairs
```
SPLADE beats BM25 by 5-8% on BEIR (handles "cancel membership" = "terminate subscription").

---

## PHASE 2 — Fix Graph (the core differentiator, currently broken)

**Root causes (confirmed from MULTI_DOMAIN_EVAL.md)**:
1. `graph_weight=0.15` in RRF → insufficient to override vector at any density
2. Reference nodes have zero edges → graph_score=0 for 73% of queries
3. Pure-graph candidates never enter candidate pool

### 2a. Wire bge-m3 sparse vectors (zero-cost, high impact)

`embed_hybrid()` already generates sparse vectors but discards them. Add as 3rd RRF signal:

```python
# engine/hybrid_ranker.py — in search()
sparse_results = self.sparse_index.search(query_sparse_vec, top_k=candidate_k)
# Add to RRF alongside dense and graph:
rrf_score = (
    vector_weight * 1/(k + rank_dense[n]) +
    bm25_weight   * 1/(k + rank_sparse[n]) +
    graph_weight  * 1/(k + rank_graph[n])
)
```

The 4-signal formula for the paper:

$$\text{RRF}(n) = \sum_{s \in \{\text{dense}, \text{sparse}, \text{graph}, \text{colbert}\}} w_s \cdot \frac{1}{k + \text{rank}_s(n)}$$

### 2b. Fix GNN zero-vector features

`engine/gnn_reranker.py:137` sets `x = torch.zeros(...)`. Load real embeddings:

```python
# engine/gnn_reranker.py — build_pyg_graph()
embeddings = {}
for nid in node_ids:
    row = self.db.get_node(nid)
    if row and row.raw_embedding:
        embeddings[nid] = np.frombuffer(row.raw_embedding, dtype=np.float32)
    else:
        embeddings[nid] = np.zeros(feat_dim)

x = torch.tensor(np.stack([embeddings[nid] for nid in node_ids]), dtype=torch.float32)
```

### 2c. Wire query_router into HybridRanker

`engine/query_router.py` classifies queries but isn't used at search time:

```python
# engine/hybrid_ranker.py — search()
query_type = self.query_router.classify(query)  # → "temporal" | "multihop" | "entity" | "factual"

weight_map = {
    "temporal":  {"vector_weight": 0.3, "graph_weight": 0.5, "bm25_weight": 0.2},
    "multihop":  {"vector_weight": 0.2, "graph_weight": 0.7, "bm25_weight": 0.1},
    "entity":    {"vector_weight": 0.3, "graph_weight": 0.3, "bm25_weight": 0.4},
    "factual":   {"vector_weight": 0.5, "graph_weight": 0.1, "bm25_weight": 0.4},
}
weights = weight_map.get(query_type, default_weights)
```

### 2d. Intra-domain edges (required for non-zero graph signal)

The Phase 3 experiments confirm: cross-domain-only edges at ≤5% density produce **zero graph signal**. Fix:

```python
# At ingest time: auto-edges within same domain
HYBRIDMIND_AUTO_EDGES_ENABLED=true
HYBRIDMIND_AUTO_EDGE_COSINE_THRESHOLD=0.70   # lower threshold than 0.75
HYBRIDMIND_AUTO_EDGE_MAX_PER_NODE=10         # ensure enough neighbors

# Also add: filter to same-domain when building intra-domain edges
filter_metadata={"domain": node.metadata.get("domain")}
```

Target: ≥30% per-node bidirectional edge coverage for non-zero graph scores on all queries.

---

## PHASE 3 — Temporal Knowledge Graph (+10-15 pts on temporal queries)

EverMind/Zep advantage: every edge has time-aware scoring. HybridMind stores timestamps but ignores them.

### 3a. Temporal Edge Schema

Extend `models/edge.py`:
```python
class EdgeModel(BaseModel):
    ...
    created_at: datetime = Field(default_factory=datetime.utcnow)
    valid_from: Optional[datetime] = None   # when this fact became true
    valid_until: Optional[datetime] = None  # when superseded
    superseded_by: Optional[str] = None     # edge_id of replacement
    confidence: float = 1.0
```

### 3b. Temporal Decay Scoring

For LoCoMo temporal queries (20% of benchmark, currently only 14.29% accuracy):

$$w_{\text{temporal}}(e) = \exp\!\left(-\lambda \cdot \frac{t_{\text{now}} - t_e}{\text{half\_life}}\right)$$

$$G_{\text{temporal}}(q, n) = \sum_{e \in \text{path}(\text{anchor}, n)} w_e \cdot w_t(e) \cdot \frac{1}{1 + d}$$

```python
# engine/graph_index.py — compute_temporal_proximity()
import math

def temporal_decay(edge_time: datetime, half_life_days: float = 30.0) -> float:
    delta_days = (datetime.utcnow() - edge_time).days
    return math.exp(-math.log(2) * delta_days / half_life_days)

def score_with_temporal(path_edges: list, decay=True) -> float:
    base = 1.0 / (1 + len(path_edges))
    if not decay:
        return base
    time_factor = min(temporal_decay(e.created_at) for e in path_edges)
    return base * time_factor
```

**Tune `half_life_days`**: 7 for conversation memory, 90 for domain knowledge.

### 3c. Fact Supersession Detection

```python
# engine/fact_extractor.py — detect_contradictions()
# When ingesting new fact, check for contradictions with existing facts
existing = self.db.search_similar_facts(new_fact.text, threshold=0.85)
for old_fact in existing:
    if contradicts(old_fact, new_fact):  # LLM or rule-based
        # Mark old edge as superseded
        self.db.update_edge(old_edge_id, superseded_by=new_edge_id, valid_until=now)
        # Mark old fact as stale
        self.db.update_node(old_fact.id, metadata={"stale": True, "superseded_by": new_fact.id})
```

---

## PHASE 4 — Memory Lifecycle (EverMind-style, +5-8 pts)

EverMind at 93% uses: Conversations → MemCells → MemScenes (thematic consolidation). HybridMind has `fact_extractor.py` but no consolidation.

### 4a. 3-Pool Architecture (AgentRunbook's #1 insight)

```python
# models/memory_pool.py
class MemoryPool(str, Enum):
    RAW = "raw"          # exact extracted facts, high fidelity
    EVENTS = "events"    # state transitions, "X changed to Y"
    NOTES = "notes"      # procedures, patterns, gotchas
    SUMMARY = "summary"  # consolidated, thematic

# At ingest, classify and route
def classify_memory_type(fact: str, context: str) -> MemoryPool:
    # Use LLM or rules
    if any(kw in fact.lower() for kw in ["changed", "became", "now", "was"]):
        return MemoryPool.EVENTS
    if any(kw in fact.lower() for kw in ["always", "never", "usually", "pattern"]):
        return MemoryPool.NOTES
    return MemoryPool.RAW

# Store pool type in metadata
metadata["memory_pool"] = classify_memory_type(fact, context).value
```

### 4b. Consolidation Pipeline

```python
# scripts/consolidate_memory.py — run periodically or via /admin/consolidate
def consolidate_sessions(db, min_facts=5, max_age_hours=24):
    """Roll up related RAW facts into SUMMARY nodes"""
    sessions = db.get_recent_sessions(max_age_hours)
    for session_id in sessions:
        facts = db.get_nodes(filter_metadata={"session_id": session_id, "memory_pool": "raw"})
        if len(facts) < min_facts:
            continue
        
        # LLM summarization
        summary_text = llm_summarize(
            texts=[f.text for f in facts],
            prompt="Synthesize these facts into a coherent thematic summary. Be specific."
        )
        
        # Create summary node
        summary_node = db.create_node(
            text=summary_text,
            metadata={"memory_pool": "summary", "source_session": session_id, "fact_count": len(facts)}
        )
        
        # Link summary → constituent facts
        for fact in facts:
            db.create_edge(summary_node.id, fact.id, "summarizes", weight=1.0)
```

### 4c. Importance-Based Retention

$$I(n) = \alpha \cdot \text{recency}(n) + \beta \cdot \text{frequency}(n) + \gamma \cdot \text{centrality}(n)$$

```python
import networkx as nx

def importance_score(node_id, graph, access_log, alpha=0.4, beta=0.3, gamma=0.3):
    recency = 1.0 / (1 + days_since_last_access(node_id, access_log))
    frequency = min(1.0, access_count(node_id, access_log) / 10)
    centrality = nx.pagerank(graph).get(node_id, 0)
    return alpha * recency + beta * frequency + gamma * centrality

# Soft-forget nodes below threshold
for node in db.get_all_nodes():
    if importance_score(node.id, ...) < 0.05:
        db.soft_delete(node.id, reason="low_importance")
```

---

## PHASE 5 — Community Detection (GraphRAG-style)

Enables answering abstract/thematic queries (currently 0 support for this).

```bash
pip install leidenalg python-igraph
```

```python
# engine/community_detector.py
import igraph as ig
import leidenalg

def detect_communities(nx_graph, resolution=0.5):
    # Convert NetworkX → iGraph
    g = ig.Graph.from_networkx(nx_graph)
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution)
    return {v["_nx_name"]: comm_id for v, comm_id in zip(g.vs, partition.membership)}

def summarize_community(nodes: list[str], db) -> str:
    texts = [db.get_node(n).text for n in nodes[:20]]  # top-20 by centrality
    return llm_summarize(texts, prompt="What is the central theme connecting these memory fragments?")

# Run on /admin/compact or scheduled
communities = detect_communities(graph_index.graph)
for comm_id, node_ids in group_by_community(communities).items():
    summary = summarize_community(node_ids, db)
    db.create_node(summary, metadata={"type": "community_summary", "community_id": comm_id, "member_count": len(node_ids)})
```

---

## PHASE 6 — GPU-Accelerated Benchmarks (RunPod)

### 6a. FAISS-GPU with IVF-PQ

```python
# storage/vector_index.py — on RunPod A100
import faiss

d = 1024  # bge-m3 dim
nlist = 256  # clusters; rule of thumb: sqrt(n_nodes)
m = 64    # subquantizers
nbits = 8 # bits per subquantizer

quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
gpu_index.train(all_embeddings)
gpu_index.add(all_embeddings)
# IVF-PQ: 4-5x compression, 10x faster than HNSWFlat at scale
```

### 6b. Contrastive Fine-tuning (RunPod, SimCSE on graph edges)

Existing script: `scripts/train_contrastive.py`. Extend:

```python
# SimCSE loss on graph edge pairs
# Positive: (node_text, neighbor_text) where edge.weight > 0.7
# Hard negative: (node_text, random_same_domain_text)

loss_fn = losses.MultipleNegativesRankingLoss(model)
# Or for hard negatives:
loss_fn = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=64)

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=edge_pair_dataset,
    loss=loss_fn,
    args=SentenceTransformerTrainingArguments(
        output_dir="checkpoints/hybridmind-embed",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        learning_rate=2e-5,
        fp16=True,  # A100 supports BF16 too
    )
)
```

### 6c. GNN Training (RunPod, BPR loss)

```python
# scripts/train_gnn.py — with real node features
# Node features: bge-m3 embeddings (1024-dim) from SQLite
# Positive edges: high-weight explicit edges (weight > 0.8)
# Negative edges: random non-edge pairs

class GraphSAGEReranker(torch.nn.Module):
    def __init__(self, in_dim=1024, hidden=512, out=256):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# BPR loss
def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
```

### 6d. Fusion MLP Training (pairwise logistic)

```python
# scripts/train_fusion_mlp.py — fully wire this
# Input features: [dense_score, sparse_score, graph_score, colbert_score, query_type_onehot]
# Label: 1 if relevant, 0 if not (from benchmark ground truth)

class FusionMLP(nn.Module):
    def __init__(self, in_dim=9):  # 4 scores + 5 query types
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# Pairwise logistic loss (LTR)
loss = F.binary_cross_entropy(scores_pos, ones) + F.binary_cross_entropy(scores_neg, zeros)
```

---

## PHASE 7 — Visual Memory (ColPali, paper differentiator)

First agent memory system with native visual document retrieval. Zero OCR needed.

```bash
pip install colpali-engine
```

```python
# engine/visual_retrieval.py
from colpali_engine.models import ColQwen2, ColQwen2Processor

model = ColQwen2.from_pretrained("vidore/colqwen2.5-v0.2", torch_dtype=torch.bfloat16)
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2.5-v0.2")

def embed_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path)
    batch = processor.process_images([image])
    return model(**batch)  # patch-level embeddings

def embed_query_for_image(query: str) -> torch.Tensor:
    batch = processor.process_queries([query])
    return model(**batch)

# MaxSim scoring (same as ColBERT, works on image patches)
def maxsim(query_emb, doc_emb):
    scores = torch.einsum("qd,pd->qp", query_emb, doc_emb)
    return scores.max(dim=1).values.sum()
```

Storage: extend `.mind` format with `visual/` subdirectory containing patch embeddings.

---

## PHASE 8 — MCP Server (ecosystem integration)

```python
# mcp_server/main.py
from mcp import FastMCP

mcp = FastMCP("HybridMind")

@mcp.tool()
async def remember(text: str, session_id: str = None) -> str:
    """Store a memory node"""
    return await sdk.store(text, metadata={"session_id": session_id})

@mcp.tool()
async def recall(query: str, top_k: int = 5, mode: str = "hybrid") -> list[dict]:
    """Retrieve relevant memories"""
    return await sdk.recall(query, top_k=top_k, mode=mode)

@mcp.tool()
async def relate(source_id: str, target_id: str, relation_type: str) -> str:
    """Create a relationship between memories"""
    return await sdk.relate(source_id, target_id, relation_type)

if __name__ == "__main__":
    mcp.run()
```

```bash
# In CLAUDE.md / mcp_servers
hybridmind: {command: "python", args: ["-m", "mcp_server.main"]}
```

---

## Benchmark Suite — Run Order

### Quick wins (local RTX 4050, ~2h)
```bash
# 1. Fix single-hop answering + re-run LoCoMo 50-question sample
python scripts/eval_locomo_retrieval.py --sample 50 --answering-model claude-sonnet-4-6

# 2. LongMemEval-S (500 questions)
python eval_longmemeval_retrieval.py --split test-s --top-k 10

# 3. MuSiQue multi-hop (100 questions)
python eval_musique_retrieval.py --split validation --top-k 5
```

### Full benchmark (RunPod A100, ~8h)
```bash
./scripts/run_all_benchmarks.sh --embedding-model Qwen/Qwen3-Embedding --reranker mixedbread-ai/mxbai-rerank-large-v2 --full-locomo
```

### Publish-worthy numbers: run each 3x, report mean ± std

---

## Paper Outline

**Title**: "HybridMind: Temporal-Aware Hybrid Retrieval with Memory Lifecycle for Agent Systems"

**Contributions**:
1. **4-signal RRF fusion** — dense + learned sparse (SPLADE) + graph proximity + ColBERT MaxSim; formal proof that rank-based fusion dominates weighted-sum when score distributions differ
2. **Temporal graph scoring** — time-decayed edge weights with fact validity windows; ablation showing +X% on temporal benchmark questions
3. **Ingest-time neighborhood averaging** — training-free GraphSAGE-style aggregation: `e_c = normalize(0.7·e_raw + 0.3·mean(e_neighbors))`; shown to double embedding separation on multi-domain corpora
4. **3-pool memory architecture** — raw facts / state transitions / procedural notes; quantified uplift vs flat extraction
5. **Query-type-aware dynamic routing** — classifier → per-type weight allocation; ablation on 5 LoCoMo question types
6. **Open-source, local-native** — no cloud dependencies, reproducible on consumer GPU; beats closed systems on SOTA benchmarks

**Benchmarks to report**:
- LoCoMo (all 5 question types, full 1,540 questions)
- LongMemEval-S (500 questions)
- LongMemEval-V2 if available
- MuSiQue multi-hop (answer format + Hit@5)
- Internal graph-depth regime benchmark (the multi-hop graph test you already have)
- Ablation table: each component contribution

**Comparison systems**: Mem0, Zep, Letta, EverMind (use published numbers from their papers)

---

## Priority Implementation Order

| Priority | Task | Estimated LoCoMo Gain | Effort |
|----------|------|----------------------|--------|
| 🔴 P0 | Fix single-hop answering (span extraction + better LLM) | +15–20 pts | 2h |
| 🔴 P1 | Wire bm25s / SPLADE as first-class sparse signal | +3–5 pts | 4h |
| 🔴 P2 | Wire query_router into HybridRanker | +3–5 pts | 2h |
| 🔴 P3 | Upgrade reranker to mxbai-rerank-large-v2 | +3–5 pts | 1h |
| 🟡 P4 | Fix GNN zero-vector features | +2–4 pts | 3h |
| 🟡 P5 | Temporal decay scoring | +5–8 pts (temporal Qs) | 8h |
| 🟡 P6 | Upgrade embedding to Qwen3-Embedding-8B (RunPod) | +5–8 pts | 4h |
| 🟡 P7 | Wire bge-m3 sparse vectors as 3rd RRF signal | +2–3 pts | 2h |
| 🟢 P8 | 3-pool architecture + consolidation | +3–5 pts | 16h |
| 🟢 P9 | Leiden community detection | +2–4 pts (abstract Qs) | 8h |
| 🟢 P10 | SimCSE contrastive fine-tuning (RunPod) | +3–6 pts | 12h |
| 🟢 P11 | Fusion MLP training | +2–4 pts | 8h |
| 🔵 P12 | ColPali visual memory | paper differentiator | 24h |
| 🔵 P13 | MCP server | ecosystem access | 8h |
| 🔵 P14 | KùzuDB as graph backend | scale, not accuracy | 16h |

**Projected total if P0-P10 done**: 48 → **85–92% LoCoMo**

---

## Key Formulas Reference

### 4-Signal RRF (paper formula)
$$\text{RRF}(n) = \sum_{s \in \mathcal{S}} w_s \cdot \frac{1}{k + \text{rank}_s(n)}, \quad k=60$$
$$\mathcal{S} = \{\text{dense}_{\text{Qwen3}},\ \text{sparse}_{\text{SPLADE}},\ \text{graph}_{\text{temporal}},\ \text{colbert}_{\text{MaxSim}}\}$$

### Temporal Graph Score
$$G_t(q,n) = \frac{1}{1+d_{\min}(a,n)} \cdot \prod_{e \in \text{path}(a,n)} \exp\!\left(-\lambda \cdot \frac{t_{\text{now}} - t_e}{\tau}\right)$$

### Normalized Reranker Blend
$$s_{\text{final}} = 0.7 \cdot \hat{s}_{\text{RRF}} + 0.3 \cdot \hat{s}_{\text{CE}}, \quad \hat{x} = \frac{x - \min}{\max - \min}$$

### Neighborhood-Conditioned Embedding
$$e_c = \frac{0.7 \cdot e_{\text{raw}} + 0.3 \cdot \bar{e}_{\text{neighbors}}}{\|0.7 \cdot e_{\text{raw}} + 0.3 \cdot \bar{e}_{\text{neighbors}}\|_2}$$

### Importance Score for Memory Retention
$$I(n) = 0.4 \cdot e^{-\lambda_r \Delta t} + 0.3 \cdot \min\!\left(1, \frac{f(n)}{10}\right) + 0.3 \cdot \text{PageRank}(n)$$

### SimCSE Contrastive Loss
$$\mathcal{L} = -\log \frac{e^{\text{sim}(z, z^+)/\tau}}{\sum_{j=1}^{N} e^{\text{sim}(z, z_j)/\tau}}$$

---

## Exact Models / IDs

| Component | Current | Target | HF ID |
|-----------|---------|--------|-------|
| Dense embedding | `BAAI/bge-m3` | Qwen3-Embedding-8B | `Qwen/Qwen3-Embedding` |
| Sparse retrieval | NLTK BM25 | SPLADE-v3 | `prithivida/Splade_PP_en_v1` |
| Cross-encoder | `BAAI/bge-reranker-v2-m3` | mxbai-rerank-large-v2 | `mixedbread-ai/mxbai-rerank-large-v2` |
| Visual retrieval | None | ColQwen2.5 | `vidore/colqwen2.5-v0.2` |
| BM25 engine | pure Python | bm25s | `pip install bm25s` |
| Graph backend | NetworkX pickle | KùzuDB | `pip install kuzu` |
| Community detect | None | Leiden | `pip install leidenalg` |
| QA answering | GPT-5-mini | claude-sonnet-4-6 | API |

---

## RunPod Checklist

```bash
# GPU: A100 80GB (Qwen3-Embedding-8B = 16GB BF16, fits with headroom)
# Estimated cost at $1.99/hr:
# - Full LoCoMo eval (1,540 Qs): ~3h → $6
# - Contrastive fine-tuning (3 epochs): ~4h → $8
# - GNN training: ~2h → $4
# - Fusion MLP training: ~30min → $1
# Total for full benchmark + training run: ~$25-40

pip install bm25s FlagEmbedding>=1.2.10 torch-geometric sentence-transformers leidenalg colpali-engine

export HYBRIDMIND_EMBEDDING_MODEL=Qwen/Qwen3-Embedding
export HYBRIDMIND_RERANKER_MODEL=mixedbread-ai/mxbai-rerank-large-v2
export HYBRIDMIND_AUTO_EDGES_ENABLED=true
export HYBRIDMIND_COLBERT_ENABLED=true
export HYBRIDMIND_GNN_ENABLED=true
export HYBRIDMIND_USE_FAISS_GPU=true

./scripts/run_all_benchmarks.sh 2>&1 | tee results/sota_run_$(date +%Y%m%d).log
```

---

## The One Thing That Will Win

**Single-hop 0% → 60%+ is the fastest path to winning LoCoMo.** It's a parsing bug, not a retrieval problem. That's 15-20 free points.

After that: temporal scoring + mxbai reranker + Qwen3 embeddings gets you to ~85%.

Community detection + 3-pool architecture + contrastive fine-tuning pushes past 90%.

At 90%+ with a paper claiming #1 on LoCoMo + LongMemEval + MuSiQue simultaneously, you have a submission for EMNLP/ACL 2027.
