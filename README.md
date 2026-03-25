# HybridMind

HybridMind is a production-ready hybrid memory layer explicitly designed for autonomous research agents and local AI tooling. It combines vector similarity search with structured graph traversal, yielding dense knowledge retrieval that natively mimics cognitive associative memory.

It is completely localized, crash-safe, and runs sub-`15ms` workloads on CPU-only endpoints. 

---

## 🚀 Key Features

*   **Hybrid Memory:** Merges semantic similarity (FAISS) with relational proximity (NetworkX/Dijkstra) using our proprietary **CRS (Cognitive Retrieval Score)** formula.
*   **Graph-Conditioned Embeddings:** Nodes placed into the graph actively alter the coordinate space of adjacent nodes, enforcing local topology on the global semantic space.
*   **Crash-Safe & Persistent:** Powered by atomic snapshotting capabilities (`.mind` manifest directories), SQLite WAL persistence, and explicit index synchronicity checks on startup.
*   **Soft Delete & Compaction:** Features non-blocking deletion patterns and a background background DAG-safe compaction engine to keep indices tight and optimal.
*   **Python SDK:** A lightweight, idiomatic proxy client (`sdk/memory.py`) mapped to typical agent operations: `store()`, `relate()`, `recall()`, `trace()`, and `forget()`.
*   **Completely Local:** Ships by default with `all-MiniLM-L6-v2` bindings running securely on local hardware via `sentence-transformers`, with built-in GPU auto-detection.

---

## 📦 Stack

- **Framework**: `FastAPI` + `Pydantic`
- **Vector DB**: `FAISS` (IndexFlatIP)
- **Graph DB**: `NetworkX` (DiGraph)
- **Underlying Storage**: `SQLite3` (WAL Mode enabled)
- **Embeddings**: `sentence-transformers` 
- **Tests**: `pytest` + `pytest-asyncio`

---

## ⚙️ Quickstart

### 1. Requirements

*   Python 3.10+
*   `pnpm` (Optional, if interacting with front-end assets)
*   **Windows / Linux / macOS** native compat.

### 2. Setup

```bash
python -m venv .venv
# Activate environment
# On Linux/MacOS: source .venv/bin/activate
# On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Start the Inference Server

```bash
uvicorn main:app --port 8000
```
*The first boot will automatically provision a new `.mind` database directory locally, fetch the default embedding models, and align index buffers.*

---

## 🧠 Using the Agent SDK

Drop `sdk/memory.py` into any LangChain, LlamaIndex, or proprietary agent build.

```python
from sdk.memory import AgentMemory

# Initialize memory connection
memory = AgentMemory(base_url="http://localhost:8000")

# 1. Store knowledge
node_id = memory.store(
    text="The CRS formula aligns dense vectors with discrete graph paths.",
    metadata={"source": "whitepaper", "type": "concept"}
)

# 2. Relate concepts
memory.relate(
    source_id=node_id,
    target_id="existing-node-uuid",
    edge_type="EXPLAINS",
    weight=0.9
)

# 3. Recall associatively (Hybrid Vector + Graph)
results = memory.recall(
    query="How does CRS align representations?",
    top_k=5,
    alpha=0.6,  # Vector weight
    beta=0.4    # Graph weight
)

print(results[0].text)
```

---

## 📊 Performance Benchmarks

In local execution environments (CPU-only, `all-MiniLM-L6-v2`), HybridMind routinely holds `<15ms` SLA responses on atomic reads and writes. It comfortably sustains hundreds of concurrent API interactions completely lock-free on reads via native SQLite WAL configurations. 

See `benchmarks/PERFORMANCE.md` for a comprehensive breakdown of concurrency loads, scale behaviors up to 10k nodes, and graph vs. vector isolated timing sweeps.

---

## 🛠 Active Development

Run the core testing suite (34 test specifications):

```bash
python -m pytest tests/ -v
```

To run the intense production benchmark sweeps:
```bash
python benchmarks/run_benchmarks.py
```
