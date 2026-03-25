# HybridMind

HybridMind is a local-first hybrid vector + graph database designed for agentic memory. It unifies semantic similarity (FAISS) with relational structure (NetworkX) via the **Contextual Relevance Score (CRS)** algorithm, enabling retrieval that is both semantically aware and structurally grounded.

The system is optimized for sequential reasoning loops, providing sub-15ms retrieval latencies and atomic persistence on standard CPU hardware.

---

## 🚀 Key Features

*   **CRS Hybrid Retrieval:** Score fusion algorithm merging vector cosine similarity ($\alpha$) and graph shortest-path proximity ($\beta$).
*   **Graph-Conditioned Embeddings:** Optional ingest-time conditioning where semantic neighborhood influences node placement in the coordinate space.
*   **Atomic Persistence:** Crash-safe `.mind` snapshot protocol with SHA256 integrity manifests and SQLite WAL logging.
*   **Soft Deletion & Compaction:** Concurrent-safe soft deletes with background-compatible compaction to rebuild indexing.
*   **Python SDK:** High-level memory module for agents: `store()`, `relate()`, `recall()`, `trace()`, and `forget()`.
*   **Local Inference:** Powered by `all-MiniLM-L6-v2` via `sentence-transformers`. CPU-optimized with sub-1ms engine overhead.

---

## ⚙️ Performance at a Glance

Measured on: *Windows 11 · Python 3.13.5 · 12-core Intel · 15.7GB RAM (CPU-only)*

- **Hybrid Search (p50)**: 9.9ms (150 nodes) / 13ms (1000 nodes)
- **Vector Search (p50)**: 6.5ms
- **Graph Traversal (p50)**: 2.3ms
- **Node Ingest (mean)**: 200ms (includes embedding generation)
- **Snapshot (median)**: 4ms
- **Concurrency**: 10 threads @ ~487ms/req (serialization via GIL)

Detailed analysis available in [PERFORMANCE.md](file:///d:/yugaantar/benchmarks/PERFORMANCE.md).

---

## 📦 Stack

- **API Layer**: FastAPI / Pydantic v2
- **Vector Store**: FAISS (IndexFlatIP)
- **Graph Engine**: NetworkX (DiGraph)
- **Relational Store**: SQLite3 (WAL Mode)
- **Embedding Model**: all-MiniLM-L6-v2 (384-dim)

---

## 🧠 Setup & Quickstart

```bash
# 1. Prepare environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Start server
uvicorn main:app --port 8000
```

### Basic SDK Usage

```python
from sdk.memory import HybridMemory

memory = HybridMemory(base_url="http://127.0.0.1:8000")

# Store and establish relationships
node_id = memory.store("Transformer models use self-attention.")
memory.relate(node_id, "other-node-uuid", "derived_from")

# Relational search
results = memory.recall("attention mechanisms", top_k=5)
```

For advanced integration patterns, see [AGENT_INTEGRATION.md](file:///d:/yugaantar/docs/AGENT_INTEGRATION.md).

---

## 🛠 Documentation

- [Architecture Deep-Dive](file:///d:/yugaantar/docs/ARCHITECTURE.md)
- [Algorithm Specification (CRS & GCE)](file:///d:/yugaantar/docs/ALGORITHM.md)
- [Performance Characterization](file:///d:/yugaantar/benchmarks/PERFORMANCE.md)
- [Agent Integration Guide](file:///d:/yugaantar/docs/AGENT_INTEGRATION.md)

---

## 🧪 Development

Run core verification suite:
```bash
python -m pytest tests/ -v
```

Run stress tests & benchmarks:
```bash
python benchmarks/run_benchmarks.py
```

